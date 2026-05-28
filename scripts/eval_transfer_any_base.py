#!/usr/bin/env python3
"""Cross-architecture transfer eval: apply a TAM-BoT adapter (trained on a BC-T base)
to ANY robomimic-loadable policy, including IRIS/HBC/BCQ where FrozenBase doesn't work.

Bypasses FrozenBase entirely. Hardcodes state_dim by task (Lift=19, Can/Sq=23).
Uses policy_from_checkpoint so it works on BC, BCQ, IRIS, HBC, etc.

Usage:
  python scripts/eval_transfer_any_base.py \
      --ckpt <iris_or_hbc_checkpoint> \
      --task can \
      --adapter-ckpt <tam_bot_adapter.pt> \
      --episodes 10
"""
import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import importlib, unittest.mock as _mock
def _mk(name):
    m = _mock.MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    m.__name__ = name
    m.__path__ = []
    return m
for _m in ["tensorflow","tensorflow.python","tensorflow.python.framework",
          "tensorboard","tensorboard.compat","tensorboard.compat.tf",
          "torch.utils.tensorboard","torch.utils.tensorboard.writer",
          "torch.utils.tensorboard._embedding","mujoco_py"]:
    sys.modules.setdefault(_m, _mk(_m))

import robosuite as suite
try:
    import mimicgen, mimicgen.envs  # registers Threading_D0, Stack_D0, etc.
except ImportError:
    pass
from robomimic.utils.file_utils import policy_from_checkpoint
from env.telemetry_model import CurrentModel, VoltageModel
from env.thermal_model import ThermalModel
from thermal_adapters.tam_bot import TAMBoT


def _const(val):
    return lambda n: np.full(n, float(val), dtype=np.float32)


COND_TEMP = {"cool": lambda n: np.random.uniform(20, 42, n).astype(np.float32),
             "warm": lambda n: np.random.uniform(43, 55, n).astype(np.float32),
             "hot":  lambda n: np.random.uniform(56, 75, n).astype(np.float32)}
COND_CUR  = {"normal": lambda n: np.random.uniform(0.10, 0.50, n).astype(np.float32),
             "high":   lambda n: np.random.uniform(0.60, 0.85, n).astype(np.float32),
             "stall":  lambda n: np.random.uniform(0.85, 1.05, n).astype(np.float32)}
COND_VOLT = {"nominal":  lambda n: np.random.uniform(0.92, 1.00, n).astype(np.float32),
             "under":    lambda n: np.random.uniform(0.70, 0.88, n).astype(np.float32),
             "brownout": lambda n: np.random.uniform(0.45, 0.68, n).astype(np.float32)}

CONDITIONS = {
    "cool":     (COND_TEMP["cool"], COND_CUR["normal"], COND_VOLT["nominal"]),
    "hot":      (COND_TEMP["hot"],  COND_CUR["normal"], COND_VOLT["nominal"]),
    "stall":    (COND_TEMP["cool"], COND_CUR["stall"],  COND_VOLT["nominal"]),
    "brownout": (COND_TEMP["cool"], COND_CUR["normal"], COND_VOLT["brownout"]),
    "all_moderate": (COND_TEMP["warm"], COND_CUR["high"],  COND_VOLT["under"]),
    "all_severe":   (COND_TEMP["hot"],  COND_CUR["stall"], COND_VOLT["brownout"]),
    "T_mod":      (_const(58.0), _const(0.30), _const(1.00)),
    "TC_mod":     (_const(55.0), _const(0.75), _const(1.00)),
    "TV_mod":     (_const(55.0), _const(0.30), _const(0.72)),
    "TCV_mod":    (_const(55.0), _const(0.75), _const(0.72)),
}

TASK_TO_ENV = {"lift": "Lift", "can": "PickPlaceCan", "square": "NutAssemblySquare",
               "threading": "Threading_D0", "stack": "Stack_D0",
               "coffee": "Coffee_D0", "mugcleanup": "MugCleanup_D0",
               "stackthree": "StackThree_D0"}
TASK_TO_STATE_DIM = {"lift": 19, "can": 23, "square": 23, "threading": 37, "stack": 32,
                     "coffee": 66, "mugcleanup": 38, "stackthree": 48}
# MimicGen variants (Stack_D0, Threading_D0) require mimicgen import to register the envs.
try:
    import mimicgen  # registers Stack_D0, Threading_D0, etc.
except ImportError:
    pass

STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")


def make_env(task_name, horizon):
    return suite.make(
        TASK_TO_ENV[task_name], robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84, camera_widths=84,
        reward_shaping=False, ignore_done=True,
        horizon=horizon, control_freq=20,
    )


def is_success(env):
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False


def build_robomimic_obs(raw):
    obs = {}
    for k in ("agentview_image", "robot0_eye_in_hand_image"):
        if k in raw:
            img = raw[k]
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img[::-1].copy().transpose(2, 0, 1)
            obs[k] = (img.astype(np.float32) / 255.0)
    for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if k in raw:
            obs[k] = raw[k].astype(np.float32)
    if "object-state" in raw:
        obs["object"] = raw["object-state"].astype(np.float32)
    return obs


def state_from_obs(obs):
    parts = []
    for k in STATE_KEYS:
        if k in obs:
            v = np.asarray(obs[k], dtype=np.float32).reshape(-1)
            parts.append(v)
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


class FrameBuffer:
    def __init__(self, T):
        self.T = T; self.buf = deque(maxlen=T)
    def reset(self): self.buf.clear()
    def push(self, obs):
        self.buf.append({k: v.copy() for k, v in obs.items()})
        while len(self.buf) < self.T:
            self.buf.appendleft({k: v.copy() for k, v in self.buf[0].items()})
    def stacked(self):
        ks = list(self.buf[0].keys())
        return {k: np.stack([f[k] for f in self.buf], axis=0) for k in ks}


@torch.no_grad()
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0); return x*x*(3.0 - 2.0*x)
def _kinky_rho_T(T):
    bp = np.array([20.0, 45.0, 55.0, 65.0, 80.0])
    vv = np.array([1.00, 1.00, 0.70, 0.30, 0.05])
    return np.array([np.interp(np.clip(x, bp[0], bp[-1]), bp, vv) for x in T], dtype=np.float32)

def _kinky_rho_C(C):
    bp = np.array([0.00, 0.55, 0.70, 0.85, 1.10])
    vv = np.array([1.00, 1.00, 0.65, 0.30, 0.10])
    return np.array([np.interp(np.clip(x, bp[0], bp[-1]), bp, vv) for x in C], dtype=np.float32)

def _kinky_rho_V(V):
    bp = np.array([0.40, 0.55, 0.70, 0.85, 1.10])
    vv = np.array([0.10, 0.30, 0.60, 1.00, 1.00])
    return np.array([np.interp(np.clip(x, bp[0], bp[-1]), bp, vv) for x in V], dtype=np.float32)

def _analytic_correction(a_base, T, C, V, fit_error_sigma=0.0, rng=None, ffwd_curve="linear",
                         bounded_gamma_max=None, fitted_rho=None):
    """Per-joint gated analytic feedforward: a' = a*(1-g) + clip(a/rho_hat, -1, 1)*g.
    fit_error_sigma > 0 perturbs rho_hat by multiplicative Gaussian noise (proxy).
    ffwd_curve selects which curve family the inverse assumes ('linear' or 'kinky').
    bounded_gamma_max: if not None, bound |γ-1|≤gamma_max (matches TAM envelope).
    fitted_rho: dict with 'rho_T'/'rho_C'/'rho_V' callables (overrides closed-form)."""
    tg = np.clip((T - 42.0)/13.0, 0, 1); cg = np.clip((C - 0.60)/0.40, 0, 1); vg = np.clip((0.90 - V)/0.40, 0, 1)
    g = np.clip(_smoothstep(tg) + _smoothstep(cg) + _smoothstep(vg), 0.0, 1.0)
    if fitted_rho is not None:
        rho_T = fitted_rho['rho_T'](T); rho_C = fitted_rho['rho_C'](C); rho_V = fitted_rho['rho_V'](V)
    elif ffwd_curve == "kinky":
        rho_T = _kinky_rho_T(T); rho_C = _kinky_rho_C(C); rho_V = _kinky_rho_V(V)
    else:
        rho_T = 1 - 0.95*np.clip((T - 20)/60.0, 0, 1)
        rho_C = 1 - 0.50*np.clip((C - 0.50)/0.70, 0, 1)
        rho_V = 1 - 0.40*np.clip((1.0 - V)/0.55, 0, 1)
    if fit_error_sigma > 0:
        rng = rng if rng is not None else np.random
        rho_T = rho_T * (1.0 + fit_error_sigma * rng.normal(0, 1, rho_T.shape).astype(np.float32))
        rho_C = rho_C * (1.0 + fit_error_sigma * rng.normal(0, 1, rho_C.shape).astype(np.float32))
        rho_V = rho_V * (1.0 + fit_error_sigma * rng.normal(0, 1, rho_V.shape).astype(np.float32))
    rho = np.clip(rho_T*rho_C*rho_V, 0.05, 1.0)
    if bounded_gamma_max is not None:
        gamma = np.clip(1.0/rho, 1.0 - bounded_gamma_max, 1.0 + bounded_gamma_max)
        a_inv = np.clip(a_base[:7] * gamma, -1, 1)
    else:
        a_inv = np.clip(a_base[:7]/rho, -1, 1)
    out = a_base.copy(); out[:7] = a_base[:7]*(1-g) + a_inv*g
    return out

def _mrac_correction(a_base, T, C, V, rho_hat_ema, gamma_max=0.75, gate=None):
    """MRAC / L1-style online-adaptive feedforward.

    Distinct from Oracle FFwd in that it has NO access to ρ. Instead it maintains a per-joint
    EMA of the EFFECTIVE degradation ratio from observed-vs-commanded action mismatch
    (updated by the caller via _update_rho_estimate_from_obs), and applies a bounded
    inverse gain γ̂ = clip(1/ρ̂_ema, 1-γ_max, 1+γ_max). The bound is the L1-adaptive
    projection. The gate g(τ) routes the correction smoothly from 0 (cool) to 1 (hot).

    Initial ρ̂_ema = 1 (no degradation assumed); converges to true ρ within ~5-10 steps
    of consistent stress. Equivalent to a Lyapunov-based gradient law with σ-modification
    projection on the gain estimate.
    """
    tg = np.clip((T - 42.0)/13.0, 0, 1); cg = np.clip((C - 0.60)/0.40, 0, 1); vg = np.clip((0.90 - V)/0.40, 0, 1)
    g = np.clip(_smoothstep(tg) + _smoothstep(cg) + _smoothstep(vg), 0.0, 1.0)
    rho_hat = np.clip(rho_hat_ema, 0.05, 2.0)
    gamma = np.clip(1.0 / rho_hat, 1.0 - gamma_max, 1.0 + gamma_max)
    a_corr = np.clip(a_base[:7] * gamma, -1, 1)
    out = a_base.copy(); out[:7] = a_base[:7] * (1 - g) + a_corr * g
    return out

def _update_rho_estimate_from_obs(rho_hat_ema, action_commanded, action_executed,
                                  alpha_ema=0.2, eps=0.05):
    """Update the per-joint EMA estimate of ρ from observed-vs-commanded action ratio.

    For sim, action_executed is the post-degradation action (a * ρ_true * ...). In the real
    world, this is recovered from joint-velocity sensors: ρ_obs = Δq_actual / Δq_commanded.
    """
    a_c = action_commanded[:7]; a_e = action_executed[:7]
    mask = np.abs(a_c) > eps
    ratio_raw = np.where(mask, a_e / (a_c + np.sign(a_c) * 1e-6), 1.0)
    ratio = np.clip(np.abs(ratio_raw), 0.05, 1.5)
    new_ema = alpha_ema * ratio + (1 - alpha_ema) * rho_hat_ema
    return new_ema.astype(np.float32)


def _scalar_gain_correction(a_base, T, C, V, k=0.6):
    """Naive baseline: scale action toward zero when any channel stressed.
    a' = a * (k + (1-k)*(1-g)) per joint, where g is the same per-joint smoothstep gate.
    At g=0 (cool): a' = a; at g=1 (full stress): a' = k*a. Conservative gain reduction."""
    tg = np.clip((T - 42.0)/13.0, 0, 1); cg = np.clip((C - 0.60)/0.40, 0, 1); vg = np.clip((0.90 - V)/0.40, 0, 1)
    g = np.clip(_smoothstep(tg) + _smoothstep(cg) + _smoothstep(vg), 0.0, 1.0)
    out = a_base.copy(); out[:7] = a_base[:7] * (k + (1.0 - k) * (1.0 - g))
    return np.clip(out, -1, 1)

def _apply_non_factorizable_degradation(action, temps, currents, voltages, thermal, current, voltage, alpha_mag=0.4):
    """Non-factorizable degradation: ρ_C depends jointly on (C, |a|), not just C.
    Physical motivation: at high commanded current → magnetic-flux saturation reduces
    effective Kt; back-EMF at high commanded velocity further reduces available torque.
    Both effects mean the marginal ρ_C(C) over-estimates the actual capacity at high |a|.

    FFwd's 1/ρ̂(τ) structurally cannot capture this — it only sees telemetry, not |a|.
    TAM has a_base as input and can learn the joint correction.

    Returns the degraded action. Should replace the standard apply_thermal/current/voltage cascade.
    """
    a = action.astype(np.float32)
    mag = np.clip(np.abs(a[:7]), 0.0, 1.0)
    # ρ_T(T) — marginal (matches env's compute_degradation)
    rho_T = np.array([thermal.compute_degradation(float(temps[j]), j) for j in range(7)], dtype=np.float32)
    # ρ_C_eff(C, |a|) — non-factorizable: high commanded current degrades faster
    rho_C_marginal = np.array([current.compute_degradation(float(currents[j]), j) for j in range(7)], dtype=np.float32)
    rho_C_eff = np.clip(rho_C_marginal * (1.0 - alpha_mag * mag), 0.05, 1.0)
    # ρ_V(V) — marginal
    rho_V = np.array([voltage.compute_degradation(float(voltages[j]), j) for j in range(7)], dtype=np.float32)
    rho = np.clip(rho_T * rho_C_eff * rho_V, 0.05, 1.0)
    out = a.copy()
    out[:7] = a[:7] * rho
    return out


def _apply_coupled_physics(temps, currents, voltages, alpha_TC=0.3, alpha_CV=0.15):
    """Inject cross-channel coupling that the factorised model does not capture.
    Physics rationale:
      (i)  winding resistance R(T) = R0*(1 + alpha_R*(T-T0)) raises required current at constant torque.
      (ii) bus voltage drop V_eff = V_supply - I*R_line, scales with instantaneous current draw.
    Returns (T, C_eff, V_eff) — the env uses these for degradation; the adapter / analytic
    see the original (T, C, V) so the factorised inverse is miscalibrated by the coupling factor.
    alpha_TC: strength of temperature -> current coupling (per 1.0 of normalized T from 20°C).
    alpha_CV: strength of current -> voltage coupling (per 1.0 of normalized current).
    """
    T_norm = np.clip((temps - 20.0) / 60.0, 0, 1)  # 0 at 20°C, 1 at 80°C
    currents_eff = np.clip(currents * (1.0 + alpha_TC * T_norm), 0.05, 1.5)
    voltages_eff = np.clip(voltages - alpha_CV * currents_eff, 0.05, 1.05)
    return temps, currents_eff.astype(np.float32), voltages_eff.astype(np.float32)

def _oracle_rho_correction(a_base, T, C, V):
    """Oracle baseline: same as analytic FFwd, but assume true rho_hat == sim rho (already perfect).
    In our sim this is identical to analytic_correction; included as a sanity-check name."""
    return _analytic_correction(a_base, T, C, V)

def run_episodes(policy, adapter, task, n, horizon, t_fn, c_fn, v_fn,
                 thermal, current, voltage, device, context_length, is_transformer,
                 noise_T=0.0, noise_C=0.0, noise_V=0.0, mask_channels="",
                 analytic_feedforward=False, scalar_gain=None, coupled_physics=False,
                 coupling_TC=0.3, coupling_CV=0.15, fit_error_sigma=0.0, ffwd_curve="linear",
                 non_factorizable_rho=False, alpha_mag=0.4,
                 bounded_ffwd_gamma_max=None, fitted_rho=None,
                 mrac_baseline=False, mrac_gamma_max=0.75, mrac_ema_alpha=0.2,
                 mrac_low_pass_alpha=None, log_corrections=False):
    env = make_env(task, horizon)
    fb = FrameBuffer(context_length)
    n_joints = 7
    successes = 0
    # Per-joint correction magnitude accumulators: |a_TAM - a_base| per action dim (dim j <-> joint j; dim 6 = gripper)
    corr_abs_sum = np.zeros(7, dtype=np.float64)
    corr_steps = 0
    for ep in range(n):
        # MRAC/L1 baseline: per-joint EMA of observed ρ̂, init to 1 (no degradation assumed).
        # Resets each episode (simulates a fresh deployment with no prior memory).
        rho_hat_ema = np.ones(n_joints, dtype=np.float32) if mrac_baseline else None
        # Optional low-pass on the corrected action (L1-adaptive style filter).
        a_corrected_prev = None
        temps = t_fn(n_joints)
        currents = c_fn(n_joints)
        voltages = v_fn(n_joints)
        # Adapter and analytic correction see the observed (T, C, V); env physics may use coupled versions.
        if coupled_physics:
            temps_env, currents_env, voltages_env = _apply_coupled_physics(
                temps, currents, voltages, alpha_TC=coupling_TC, alpha_CV=coupling_CV)
        else:
            temps_env, currents_env, voltages_env = temps, currents, voltages
        raw = env.reset()
        policy.start_episode()
        fb.reset()
        fb.push(build_robomimic_obs(raw))
        success = False
        for step in range(horizon):
            stacked = fb.stacked()
            # For non-transformer policies, strip the time dim
            policy_obs = stacked if is_transformer else {k: v[-1] for k, v in stacked.items()}
            action = policy(ob=policy_obs)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            action = action.clip(-1.0, 1.0)

            # Analytic feedforward (closed-form inverse): replaces neural TAM
            if analytic_feedforward:
                action = _analytic_correction(action, temps, currents, voltages,
                                              fit_error_sigma=fit_error_sigma,
                                              ffwd_curve=ffwd_curve,
                                              bounded_gamma_max=bounded_ffwd_gamma_max,
                                              fitted_rho=fitted_rho).astype(np.float32)

            # Naive scalar gain reduction (classical baseline): a *= k when stressed
            if scalar_gain is not None:
                action = _scalar_gain_correction(action, temps, currents, voltages, k=scalar_gain).astype(np.float32)

            # MRAC / L1 online-adaptive baseline (classical adaptive control).
            # Uses NO oracle ρ; learns ρ̂ online from action-vs-realized mismatch.
            a_commanded_for_mrac = None
            if mrac_baseline:
                a_commanded_for_mrac = action.copy()  # remember pre-correction action
                action = _mrac_correction(action, temps, currents, voltages,
                                         rho_hat_ema, gamma_max=mrac_gamma_max).astype(np.float32)
                # L1-adaptive style low-pass filter on the corrected action.
                if mrac_low_pass_alpha is not None and a_corrected_prev is not None:
                    a = mrac_low_pass_alpha * action[:7] + (1 - mrac_low_pass_alpha) * a_corrected_prev[:7]
                    action[:7] = np.clip(a, -1, 1)
                a_corrected_prev = action.copy()

            # TAM correction
            if adapter is not None:
                state_vec = state_from_obs({k: stacked[k][-1] for k in stacked})
                # Pad/truncate to expected state_dim
                expected = adapter.state_dim
                if state_vec.shape[0] < expected:
                    state_vec = np.concatenate([state_vec, np.zeros(expected - state_vec.shape[0], dtype=np.float32)])
                elif state_vec.shape[0] > expected:
                    state_vec = state_vec[:expected]
                a_t = torch.as_tensor(action, dtype=torch.float32, device=device)[None]
                # Optional noise on adapter's view of telemetry
                temps_view = temps + np.random.normal(0, noise_T, temps.shape).astype(np.float32) if noise_T > 0 else temps
                currents_view = currents + np.random.normal(0, noise_C, currents.shape).astype(np.float32) if noise_C > 0 else currents
                voltages_view = voltages + np.random.normal(0, noise_V, voltages.shape).astype(np.float32) if noise_V > 0 else voltages
                # Optional channel masking — set the masked channel to nominal mid-value
                masked = {ch.strip().upper() for ch in mask_channels.split(',') if ch.strip()}
                if 'T' in masked: temps_view = np.full_like(temps_view, 30.0, dtype=np.float32)
                if 'C' in masked: currents_view = np.full_like(currents_view, 0.30, dtype=np.float32)
                if 'V' in masked: voltages_view = np.full_like(voltages_view, 0.96, dtype=np.float32)
                T_t = torch.as_tensor(temps_view, dtype=torch.float32, device=device)[None]
                C_t = torch.as_tensor(currents_view, dtype=torch.float32, device=device)[None]
                V_t = torch.as_tensor(voltages_view, dtype=torch.float32, device=device)[None]
                s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device)[None]
                with torch.no_grad():
                    corrected, _, _ = adapter(a_t, T_t, s_t, C_t, V_t)
                a_base_pre = action.copy()
                action = corrected.detach().squeeze(0).cpu().numpy()
                if log_corrections:
                    d = np.abs(action[:7] - a_base_pre[:7])
                    if d.shape[0] == 7:
                        corr_abs_sum += d; corr_steps += 1

            # Degradation
            if non_factorizable_rho:
                # Non-factorizable: ρ_C depends jointly on (C, |action|). FFwd structurally
                # cannot capture this; TAM has a_base in its input and can learn the correction.
                action_executed = _apply_non_factorizable_degradation(action, temps_env, currents_env, voltages_env,
                                                             thermal, current, voltage, alpha_mag=alpha_mag).astype(np.float32)
            else:
                a_post_T = thermal.apply_thermal_physics(action, temps_env)
                a_post_C = current.apply_current_physics(a_post_T, currents_env)
                action_executed = voltage.apply_voltage_physics(a_post_C, voltages_env)

            # MRAC: update online ρ̂ estimate from observed-vs-commanded action ratio.
            if mrac_baseline and a_commanded_for_mrac is not None:
                rho_hat_ema = _update_rho_estimate_from_obs(
                    rho_hat_ema, a_commanded_for_mrac, action_executed, alpha_ema=mrac_ema_alpha)

            action = action_executed
            raw, r, done, info = env.step(action)
            fb.push(build_robomimic_obs(raw))
            if is_success(env):
                success = True; break
        successes += int(success)
    try:
        env.close()
    except Exception as e:
        print(f"  [warn] env.close() raised {type(e).__name__}; ignoring (EGL teardown is best-effort)", flush=True)
    sr = successes / n
    if log_corrections:
        corr_mean = (corr_abs_sum / corr_steps).tolist() if corr_steps > 0 else None
        return sr, corr_mean
    return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_TO_ENV))
    ap.add_argument("--adapter-ckpt", default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gamma-range", type=float, default=0.5)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--telemetry-noise-temp", type=float, default=0.0,
                    help="Gaussian noise std applied to adapter's view of temperature (°C)")
    ap.add_argument("--telemetry-noise-current", type=float, default=0.0,
                    help="Gaussian noise std applied to adapter's view of current (fraction of rated)")
    ap.add_argument("--telemetry-noise-voltage", type=float, default=0.0,
                    help="Gaussian noise std applied to adapter's view of voltage (fraction of rated)")
    ap.add_argument("--mask-channels", type=str, default="",
                    help="Comma-separated channel names to mask (set to nominal): T, C, V. e.g. 'T,V' for current-only adapter input.")
    ap.add_argument("--analytic-feedforward", action="store_true",
                    help="Replace neural TAM with the analytic closed-form inverse a'/rho_hat(tau), gated by the same smoothstep. No learning, no adapter.")
    ap.add_argument("--scalar-gain", type=float, default=None,
                    help="Naive scalar gain baseline: a' = a * (k + (1-k)*(1-g)). Recommended k in [0.5, 0.8]. Mutually exclusive with --analytic-feedforward.")
    ap.add_argument("--coupled-physics", action="store_true",
                    help="Inject cross-channel physics coupling (R(T) raises required current; bus voltage drop scales with current). Adapter / analytic correction continue to see (T,C,V); env uses (T, C_eff(T,C), V_eff(C,V)). Models 'physics misspecification' that the factorised inverse cannot capture.")
    ap.add_argument("--coupling-TC", type=float, default=0.3,
                    help="Temperature->current coupling strength alpha_TC (default 0.3).")
    ap.add_argument("--coupling-CV", type=float, default=0.15,
                    help="Current->voltage coupling strength alpha_CV (default 0.15).")
    ap.add_argument("--fit-error-sigma", type=float, default=0.0,
                    help="Realistic-deployment FFwd baseline: perturb rho_hat by N(0, sigma) multiplicative noise. sigma=0 is oracle (current default). sigma=0.10 models a 10%% fit-error on rho.")
    ap.add_argument("--env-curve", type=str, default="linear",
                    choices=["linear", "exponential", "sigmoid", "polynomial", "kinky"],
                    help="Family of degradation curves applied env-side.")
    ap.add_argument("--ffwd-curve", type=str, default="linear",
                    choices=["linear", "kinky"],
                    help="Curve family the analytic FFwd uses for its inverse. When env-curve matches, this is true Oracle FFwd; otherwise it is mis-specified.")
    ap.add_argument("--non-factorizable-rho", action="store_true",
                    help="Env applies non-factorizable degradation ρ_C(C, |a|). FFwd's 1/ρ̂(τ) structurally cannot see |a|; TAM with a_base as input can.")
    ap.add_argument("--alpha-mag", type=float, default=0.4,
                    help="Action-magnitude coupling strength (default 0.4). At |a|=1, ρ_C is reduced by this fraction.")
    ap.add_argument("--bounded-ffwd-gamma-max", type=float, default=None,
                    help="Bound FFwd's multiplicative correction |γ-1| ≤ this. Matches TAM's envelope (use 0.75). "
                         "Isolates 'bounded inversion' from 'neural residual' — answers whether the +10pp Can win is "
                         "structural (neural) or about action-saturation handling.")
    ap.add_argument("--fitted-rho-samples", type=int, default=0,
                    help="If >0, fit a polynomial ρ̂ from this many random (T,C,V) samples drawn from the env "
                         "and use the FITTED ρ̂ in the FFwd inverse (instead of the exact closed-form). "
                         "Replaces the i.i.d.-Gaussian fit-error proxy with a realistic-deployment fit.")
    ap.add_argument("--fitted-rho-degree", type=int, default=3,
                    help="Polynomial degree for fitted ρ̂ (default 3). Higher = more flexibility, may overfit.")
    ap.add_argument("--fitted-rho-safe-band-only", action="store_true",
                    help="Constrain bench samples to the safe operating range (T<=50C, C<=0.6 rated, V>=0.85). "
                         "This models the realistic-deployment constraint that you cannot bench-test motors at "
                         "stall/overheat for safety reasons. The polynomial fit must then extrapolate into "
                         "the stress regime at deployment time.")
    ap.add_argument("--mrac-baseline", action="store_true",
                    help="MRAC / L1-style online-adaptive feedforward baseline. NO ρ access; learns ρ̂ online "
                         "from observed-vs-commanded action ratio, applies bounded inverse-gain γ̂ = clip(1/ρ̂, 1-γ_max, 1+γ_max).")
    ap.add_argument("--mrac-gamma-max", type=float, default=0.75,
                    help="MRAC bounded-gain projection envelope (default 0.75, matches TAM's released envelope).")
    ap.add_argument("--mrac-ema-alpha", type=float, default=0.2,
                    help="MRAC EMA update rate (default 0.2). Higher = faster adaptation, more variance.")
    ap.add_argument("--mrac-low-pass-alpha", type=float, default=None,
                    help="L1-adaptive style low-pass filter on corrected action (default off). Set e.g. 0.5 to enable.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-corrections", action="store_true",
                    help="Log per-joint TAM correction magnitude |a_TAM - a_base| (mean over rollout steps, per action dim) into the per-condition results.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # If fitted-rho-samples > 0, fit a polynomial ρ̂ from samples drawn from the env's true ρ.
    # This is the realistic-deployment FFwd: you measure (telemetry, ρ) on the bench and use the fit.
    fitted_rho_obj = None
    if args.fitted_rho_samples > 0:
        env_curve = args.env_curve if args.env_curve != "linear" else "linear"
        true_T = ThermalModel.from_predefined(env_curve, n_joints=7)
        true_C = CurrentModel.from_predefined(env_curve, n_joints=7)
        true_V = VoltageModel.from_predefined(env_curve, n_joints=7)
        rng_fit = np.random.RandomState(args.seed)
        N = args.fitted_rho_samples
        # Safe-band sampling: bench-realistic constraint. Hardware bench operators avoid
        # stalling motors / extreme thermal loading for safety + equipment-lifetime reasons,
        # so bench samples concentrate in the safe operating range. At deployment, the
        # polynomial fit must extrapolate into the stress regime.
        if getattr(args, "fitted_rho_safe_band_only", False):
            T_samples = rng_fit.uniform(20, 50, N)   # safe (no hot/severe)
            C_samples = rng_fit.uniform(0.1, 0.6, N) # safe (no stall)
            V_samples = rng_fit.uniform(0.85, 1.0, N) # safe (no brownout)
        else:
            T_samples = rng_fit.uniform(20, 80, N)
            C_samples = rng_fit.uniform(0.1, 1.05, N)
            V_samples = rng_fit.uniform(0.4, 1.0, N)
        rho_T_samples = np.array([true_T.compute_degradation(float(t), 0) for t in T_samples])
        rho_C_samples = np.array([true_C.compute_degradation(float(c), 0) for c in C_samples])
        rho_V_samples = np.array([true_V.compute_degradation(float(v), 0) for v in V_samples])
        coefs_T = np.polyfit(T_samples, rho_T_samples, deg=args.fitted_rho_degree)
        coefs_C = np.polyfit(C_samples, rho_C_samples, deg=args.fitted_rho_degree)
        coefs_V = np.polyfit(V_samples, rho_V_samples, deg=args.fitted_rho_degree)
        def fit_T(x): return np.clip(np.polyval(coefs_T, x), 0.05, 1.0).astype(np.float32)
        def fit_C(x): return np.clip(np.polyval(coefs_C, x), 0.05, 1.0).astype(np.float32)
        def fit_V(x): return np.clip(np.polyval(coefs_V, x), 0.05, 1.0).astype(np.float32)
        fitted_rho_obj = {'rho_T': fit_T, 'rho_C': fit_C, 'rho_V': fit_V}
        # Report fit RMSE for diagnostic
        rmse_T = np.sqrt(np.mean((fit_T(T_samples) - rho_T_samples)**2))
        rmse_C = np.sqrt(np.mean((fit_C(C_samples) - rho_C_samples)**2))
        rmse_V = np.sqrt(np.mean((fit_V(V_samples) - rho_V_samples)**2))
        print(f"[fitted ρ̂] N={N}, deg={args.fitted_rho_degree}  RMSE: T={rmse_T:.4f} C={rmse_C:.4f} V={rmse_V:.4f}", flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[load policy] {args.ckpt}", flush=True)
    policy, _ = policy_from_checkpoint(ckpt_path=args.ckpt, device=device, verbose=False)
    # Infer if transformer (BC-T) by checking the model architecture
    is_transformer = "Transformer" in type(policy.policy).__name__
    # Determine context length
    context_length = 10 if is_transformer else 1
    print(f"  policy={type(policy.policy).__name__} ctx={context_length}", flush=True)

    adapter = None
    if args.adapter_ckpt:
        state_dim = TASK_TO_STATE_DIM[args.task]
        # Inspect ckpt to verify state_dim matches
        sd = torch.load(args.adapter_ckpt, map_location='cpu', weights_only=False)
        if 'state_proj.weight' in sd:
            ckpt_sd = sd['state_proj.weight'].shape[1]
            if ckpt_sd != state_dim:
                print(f"  WARN: adapter state_dim={ckpt_sd}, task expects {state_dim} — will pad/truncate")
                state_dim = ckpt_sd
        # Auto-detect architecture: joint_in (5 or 4), hidden, n_layers
        use_action_magnitude = (sd['joint_proj.weight'].shape[1] == 5)
        use_state_token = ('state_proj.weight' in sd)
        hidden = sd['joint_proj.weight'].shape[0]
        # Count transformer layers
        n_layers = sum(1 for k in sd if k.startswith('transformer.layers.') and k.endswith('.norm1.weight'))
        if n_layers == 0:
            n_layers = args.n_layers
        print(f"[load adapter] {args.adapter_ckpt} (state_dim={state_dim}, hidden={hidden}, n_layers={n_layers}, use_action_magnitude={use_action_magnitude}, use_state_token={use_state_token})", flush=True)
        adapter = TAMBoT(
            state_dim=state_dim, act_dim=7,
            hidden=hidden, n_layers=n_layers, n_heads=args.n_heads,
            alpha=args.alpha, gamma_range=args.gamma_range, gate_shape="smoothstep",
            use_action_magnitude=use_action_magnitude,
            use_state_token=use_state_token,
        ).to(device)
        adapter.load_state_dict(sd, strict=False)
        adapter.eval()

    thermal = ThermalModel.from_predefined(args.env_curve, n_joints=7)
    current = CurrentModel.from_predefined(args.env_curve, n_joints=7)
    voltage = VoltageModel.from_predefined(args.env_curve, n_joints=7)
    if args.env_curve != "linear":
        print(f"[curve mismatch eval] env uses {args.env_curve} ρ; FFwd inverse uses linear (mis-specified); TAM trained on linear targets (also mis-specified)", flush=True)

    all_results = {}
    for cond in args.conditions:
        if cond not in CONDITIONS:
            print(f"  WARN: unknown condition {cond}"); continue
        t_fn, c_fn, v_fn = CONDITIONS[cond]
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        sr = run_episodes(policy, adapter, args.task, args.episodes, args.horizon,
                          t_fn, c_fn, v_fn, thermal, current, voltage, device,
                          context_length, is_transformer,
                          noise_T=args.telemetry_noise_temp,
                          noise_C=args.telemetry_noise_current,
                          noise_V=args.telemetry_noise_voltage,
                          mask_channels=args.mask_channels,
                          analytic_feedforward=args.analytic_feedforward,
                          scalar_gain=args.scalar_gain,
                          coupled_physics=args.coupled_physics,
                          coupling_TC=args.coupling_TC,
                          coupling_CV=args.coupling_CV,
                          fit_error_sigma=args.fit_error_sigma,
                          ffwd_curve=args.ffwd_curve,
                          non_factorizable_rho=args.non_factorizable_rho,
                          alpha_mag=args.alpha_mag,
                          bounded_ffwd_gamma_max=args.bounded_ffwd_gamma_max,
                          fitted_rho=fitted_rho_obj if args.fitted_rho_samples > 0 else None,
                          mrac_baseline=args.mrac_baseline,
                          mrac_gamma_max=args.mrac_gamma_max,
                          mrac_ema_alpha=args.mrac_ema_alpha,
                          mrac_low_pass_alpha=args.mrac_low_pass_alpha,
                          log_corrections=args.log_corrections)
        if args.log_corrections:
            sr, corr = sr
            all_results[cond] = {"sr": sr, "n": args.episodes, "corr": corr}
        else:
            all_results[cond] = {"sr": sr, "n": args.episodes}
        print(f"  {cond}: sr={sr:.0%}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "task": args.task, "ckpt": args.ckpt,
        "adapter_ckpt": args.adapter_ckpt,
        "episodes": args.episodes, "horizon": args.horizon, "seed": args.seed,
        "results": all_results,
    }, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
