#!/usr/bin/env python3
"""SFT -> RL with KL anchor for BC-Transformer bases (multi-task port of the
OpenVLA recipe in openvla/training/train_sft_rl_kl.py).

Stage 1 (SFT): Train the adapter on (state, action) pairs from the robomimic
demo HDF5. For each pair we synthesise four telemetry severities and the
deterministic inverse-degradation factor; the adapter is supervised to output
clip(action / factor, -1, 1) plus a 3x identity loss on clean conditions.

Stage 2 (RL): PPO on the live thermal env with a KL penalty to a frozen SFT
snapshot. This is the recipe that worked for OpenVLA Lift; the only changes
are: BC-Transformer base instead of OpenVLA, configurable robosuite task,
configurable state_dim (Lift=19, Can=23, Square=23).

Usage:
  python thermal_adapters/train_sft_rl_kl_bct.py \\
    --task-name PickPlaceCan \\
    --base-ckpt multi_task_runs/can/bct_can_image/.../models/model_epoch_300.pth \\
    --demo-hdf5 datasets/can/ph/image_v141.hdf5 \\
    --hidden 512 --n-blocks 2 --alpha 0.3 \\
    --sft-steps 20000 --rl-steps 60000 --kl-coef 0.1 --entropy-coef 0.05 \\
    --output-dir multi_task_runs/can/tam_sftrlkl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Mock TF/TB so robomimic imports cleanly.
import importlib, unittest.mock as _mock
def _mk(name):
    m = _mock.MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    m.__name__ = name
    m.__path__ = []
    return m
for _m in ["tensorflow", "tensorflow.python", "tensorflow.python.framework",
           "tensorboard", "tensorboard.compat", "tensorboard.compat.tf",
           "torch.utils.tensorboard", "torch.utils.tensorboard.writer",
           "torch.utils.tensorboard._embedding", "mujoco_py"]:
    sys.modules.setdefault(_m, _mk(_m))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.thermal_model import ThermalModel
from env.telemetry_model import CurrentModel, VoltageModel
from frozen_base import FrozenBase, STATE_KEYS

import robosuite as suite
try:
    import mimicgen, mimicgen.envs
except ImportError:
    pass


# ── Adapter (mirrors openvla/SFTRLAdapter, but state_dim is configurable) ──

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
    def forward(self, x):
        return x + F.relu(self.fc(self.norm(x)))


class BCTSFTRLAdapter(nn.Module):
    """Telemetry-gated residual adapter with PPO-compatible interface."""

    def __init__(self, state_dim=19, act_dim=7, hidden=512, n_blocks=2,
                 alpha=0.3, log_std_init=-2.0, gate_shape="linear"):
        super().__init__()
        self.alpha = alpha
        self.act_dim = act_dim
        self.gate_shape = gate_shape  # "linear" (legacy) or "smoothstep"
        # Input: a_base (act_dim) + temps (7) + currents (7) + voltages (7) + state (state_dim)
        in_dim = act_dim + 7 + 7 + 7 + state_dim
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.mean_head = nn.Linear(hidden, act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _ramp(self, x):
        """Apply per-channel ramp shape. x is already clamped to [0,1]."""
        if self.gate_shape == "smoothstep":
            # Hermite smoothstep: 3x^2 - 2x^3. Zero derivative at boundaries.
            # Preserves x=0 -> 0 and x=1 -> 1 exactly.
            return x * x * (3.0 - 2.0 * x)
        return x

    def gate(self, temps, currents, voltages):
        """Hard-thresholded telemetry gate. Returns *exactly* 0 under healthy
        telemetry (max_temp <= 42, max_current <= 0.60, min_voltage >= 0.90),
        ramping to 1 at the per-channel `full` corner. With gate_shape='linear'
        (default) the ramp is linear; with 'smoothstep' it follows 3x^2-2x^3
        (zero-derivative at endpoints, less aggressive at the warm shoulder)."""
        tg_x = ((temps.max(-1, keepdim=True).values - 42.0) / 13.0).clamp(0.0, 1.0)
        cg_x = ((currents.max(-1, keepdim=True).values - 0.60) / 0.40).clamp(0.0, 1.0)
        vg_x = ((0.90 - voltages.min(-1, keepdim=True).values) / 0.40).clamp(0.0, 1.0)
        tg = self._ramp(tg_x); cg = self._ramp(cg_x); vg = self._ramp(vg_x)
        return (tg + cg + vg).clamp(0.0, 1.0)

    def forward(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        h = F.relu(self.proj(x))
        for blk in self.blocks:
            h = blk(h)
        delta = torch.tanh(self.mean_head(h)) * self.alpha
        g = self.gate(temps, currents, voltages)
        mean = (a_base + g * delta).clamp(-1, 1)
        value = self.value_head(x).squeeze(-1)
        return mean, self.log_std, value

    # ── Action-space sampling (legacy: noise on env action) ──
    def get_action_and_value(self, a_base, temps, state, currents, voltages, action=None):
        mean, log_std, value = self.forward(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        lp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        return action, lp, ent, value

    # ── Delta-space sampling (gated-noise: noise on pre-gate logits) ──
    def forward_delta(self, a_base, temps, state, currents, voltages):
        """Return the *pre-gate, pre-tanh, pre-alpha* delta logits and the
        gate. The actual env action is g * alpha * tanh(delta_logits) + a_base.
        This parameterization is the basis for gated-noise PPO: noise added to
        delta_logits is automatically zeroed by the gate at cool conditions."""
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        h = F.relu(self.proj(x))
        for blk in self.blocks:
            h = blk(h)
        delta_logits = self.mean_head(h)
        g = self.gate(temps, currents, voltages)
        value = self.value_head(x).squeeze(-1)
        return delta_logits, self.log_std, g, value

    def delta_to_action(self, a_base, delta_logits, g):
        """Compute env action from delta logits and gate (deterministic)."""
        return (a_base + g * self.alpha * torch.tanh(delta_logits)).clamp(-1, 1)

    def gated_sample(self, a_base, temps, state, currents, voltages):
        """Sample action via noise on the pre-gate delta. At cool (g=0) the
        action equals a_base regardless of delta noise; at hot (g=1) noise is
        bounded by alpha * tanh-Jacobian."""
        delta_mean, log_std, g, value = self.forward_delta(
            a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(delta_mean)
        dist = Normal(delta_mean, std)
        delta_sample = dist.sample()
        action = self.delta_to_action(a_base, delta_sample, g)
        log_prob = dist.log_prob(delta_sample).sum(-1)
        return action, delta_sample, log_prob, value

    def gated_log_prob(self, a_base, delta_taken, temps, state, currents, voltages):
        """Recompute (log_prob, entropy, value) of a stored delta_pre under the
        current policy. Used in PPO updates."""
        delta_mean, log_std, g, value = self.forward_delta(
            a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(delta_mean)
        dist = Normal(delta_mean, std)
        log_prob = dist.log_prob(delta_taken).sum(-1)
        ent = dist.entropy().sum(-1)
        return log_prob, ent, value

    def get_value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ── Demo loading from robomimic HDF5 ────────────────────────────────────────

def load_demos(demo_hdf5, state_keys=("robot0_eef_pos", "robot0_eef_quat",
                                       "robot0_gripper_qpos", "object")):
    """Concatenate (state, action) pairs across all demos."""
    states = []
    actions = []
    with h5py.File(demo_hdf5, "r") as f:
        for k in f["data"].keys():
            d = f["data"][k]
            parts = [np.asarray(d["obs"][sk], dtype=np.float32) for sk in state_keys]
            s = np.concatenate(parts, axis=-1)
            a = np.asarray(d["actions"], dtype=np.float32)
            states.append(s)
            actions.append(a)
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


# ── Stage 1: SFT ────────────────────────────────────────────────────────────

def run_sft(adapter, all_states, all_actions, args, device):
    print("\n=== STAGE 1: SFT (BC-T) ===", flush=True)
    n = len(all_states)
    print(f"  demos: {n} (state_dim={all_states.shape[1]})", flush=True)

    curve_family = getattr(args, "curve_family", "linear")
    thermal = ThermalModel.from_predefined(curve_family, n_joints=7)
    current_m = CurrentModel.from_predefined(curve_family, n_joints=7)
    voltage_m = VoltageModel.from_predefined(curve_family, n_joints=7)
    if curve_family != "linear":
        print(f"  [curve family] {curve_family}", flush=True)
    alpha_mag = getattr(args, "alpha_mag", 0.0)
    if alpha_mag > 0:
        print(f"  [non-factorizable ρ] alpha_mag={alpha_mag} -- ρ_C(C,|a|) = ρ_C(C) * (1 - alpha_mag*|a|)", flush=True)

    # Pre-build augmented dataset: 4 severities of telemetry per demo.
    if getattr(args, "tam_single_severity", False):
        # Ablation: use ONLY 'mod' severity (single tier)
        sev_tiers = [("mod", ((50, 65), (0.65, 0.90), (0.65, 0.85)))]
        print("  [ablation] single-severity SFT (mod tier only)", flush=True)
    else:
        all_sev_tiers = [
            ("clean", ((20, 42), (0.10, 0.50), (0.92, 1.00))),
            ("mild",  ((43, 55), (0.50, 0.75), (0.80, 0.92))),
            ("mod",   ((50, 65), (0.65, 0.90), (0.65, 0.85))),
            ("sev",   ((60, 75), (0.80, 1.05), (0.45, 0.70))),
        ]
        n_tiers = int(getattr(args, "num_sev_tiers", 4))
        sev_tiers = all_sev_tiers[:n_tiers]
        if n_tiers < 4:
            print(f"  [severity-tier ablation] using only {n_tiers} of 4 tiers: {[t[0] for t in sev_tiers]}", flush=True)
    big_s, big_a, big_t, big_c, big_v, big_tgt = [], [], [], [], [], []
    coupling_TC = getattr(args, "coupling_TC", 0.0)
    coupling_CV = getattr(args, "coupling_CV", 0.0)
    if coupling_TC > 0 or coupling_CV > 0:
        print(f"  [coupled physics] alpha_TC={coupling_TC}, alpha_CV={coupling_CV}", flush=True)
    for sev_name, ranges in sev_tiers:
        tr, cr, vr = ranges
        T = np.random.uniform(*tr, (n, 7)).astype(np.float32)
        C = np.random.uniform(*cr, (n, 7)).astype(np.float32)
        V = np.random.uniform(*vr, (n, 7)).astype(np.float32)
        # If coupling is enabled, derive effective C and V used for target generation.
        # The adapter still sees nominal (T, C, V); the coupling is what the adapter must implicitly learn.
        if coupling_TC > 0 or coupling_CV > 0:
            T_norm = np.clip((T - 20.0) / 60.0, 0, 1)
            C_eff = np.clip(C * (1.0 + coupling_TC * T_norm), 0.05, 1.5)
            V_eff = np.clip(V - coupling_CV * C_eff, 0.05, 1.05)
        else:
            C_eff, V_eff = C, V
        # Per-(sample,joint) deterministic factor.
        factors = np.ones((n, 7), np.float32)
        if alpha_mag > 0:
            # Non-factorizable: ρ_C depends on |action|. Use demo action magnitude.
            mag = np.clip(np.abs(all_actions[:, :7]), 0.0, 1.0)
        # Bench-calibration noise on the SFT-target ρ (Limitation ii experiment).
        # ρ̂_target = ρ_true * (1 + σ * N(0,1)) — simulates noisy bench measurements.
        bench_sigma = getattr(args, "bench_calib_sigma", 0.0)
        for i in range(n):
            for j in range(7):
                td = thermal.compute_degradation(float(T[i, j]), j)
                cd_marginal = current_m.compute_degradation(float(C_eff[i, j]), j)
                if alpha_mag > 0:
                    cd = max(cd_marginal * (1.0 - alpha_mag * float(mag[i, j])), 0.05)
                else:
                    cd = cd_marginal
                vd = voltage_m.compute_degradation(float(V_eff[i, j]), j)
                f = td * cd * vd
                if bench_sigma > 0:
                    f = max(f * (1.0 + bench_sigma * float(np.random.normal())), 0.05)
                factors[i, j] = f
        # action axes: 6 arm + 1 gripper; gripper not degraded — keep as-is.
        f7 = np.concatenate([factors[:, :6], np.ones((n, 1), np.float32)], axis=-1)
        targets = np.clip(all_actions / np.maximum(f7, 0.05), -1, 1).astype(np.float32)
        big_s.append(all_states); big_a.append(all_actions); big_t.append(T)
        big_c.append(C); big_v.append(V); big_tgt.append(targets)

    big_s = np.concatenate(big_s); big_a = np.concatenate(big_a)
    big_t = np.concatenate(big_t); big_c = np.concatenate(big_c)
    big_v = np.concatenate(big_v); big_tgt = np.concatenate(big_tgt)
    N = len(big_s)
    print(f"  augmented: {N} samples (4x severity)", flush=True)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.sft_lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sft_steps)

    adapter.train()
    t0 = time.time()
    sft_input_noise = getattr(args, "sft_input_noise", 0.0)
    if sft_input_noise > 0.0:
        print(f"  [augmentation] a_base input noise σ={sft_input_noise}", flush=True)
    for step in range(args.sft_steps):
        idx = np.random.choice(N, args.batch_size, replace=False)
        s = torch.as_tensor(big_s[idx], device=device)
        a = torch.as_tensor(big_a[idx], device=device)
        t = torch.as_tensor(big_t[idx], device=device)
        c = torch.as_tensor(big_c[idx], device=device)
        v = torch.as_tensor(big_v[idx], device=device)
        tgt = torch.as_tensor(big_tgt[idx], device=device)

        # Optionally inject noise into the a_base INPUT only (target unchanged).
        # Simulates base-policy outputs that differ from the demo action.
        if sft_input_noise > 0.0:
            a_input = (a + sft_input_noise * torch.randn_like(a)).clamp(-1, 1)
        else:
            a_input = a

        mean, _, _ = adapter(a_input, t, s, c, v)
        loss_tgt = F.l1_loss(mean, tgt)
        clean_mask = ((t.max(-1).values < 42) & (c.max(-1).values < 0.6)
                      & (v.min(-1).values > 0.9))
        # Identity loss compares to the ORIGINAL a (no noise) at clean conditions —
        # we want the adapter to recover the clean action even if input was noisy.
        loss_id = F.l1_loss(mean[clean_mask], a[clean_mask]) if clean_mask.any() \
                  else torch.tensor(0.0, device=device)
        loss = loss_tgt + 3.0 * loss_id

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0 or step == 0:
            print(f"  SFT step={step+1:>6} loss={loss.item():.4f} "
                  f"(tgt={loss_tgt.item():.4f} id={loss_id.item():.4f})", flush=True)

    torch.save(adapter.state_dict(), os.path.join(args.output_dir, "sft_adapter.pt"))
    print(f"  SFT done ({time.time()-t0:.0f}s).", flush=True)


# ── BC-T obs builder (matches scripts/eval_multitask_telemetry.py) ─────────

def build_robomimic_obs(raw):
    obs = {}
    if "agentview_image" in raw:
        img = raw["agentview_image"]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img[::-1].copy()
            img = img.transpose(2, 0, 1)
        obs["agentview_image"] = (img.astype(np.float32) / 255.0)
    for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if k in raw:
            obs[k] = raw[k].astype(np.float32)
    if "object-state" in raw:
        obs["object"] = raw["object-state"].astype(np.float32)
    return obs


class FrameBuffer:
    def __init__(self, T):
        self.T = T
        self.buf = deque(maxlen=T)
    def reset(self):
        self.buf.clear()
    def push(self, obs):
        self.buf.append({k: v.copy() for k, v in obs.items()})
        while len(self.buf) < self.T:
            self.buf.appendleft({k: v.copy() for k, v in self.buf[0].items()})
    def stacked(self):
        ks = list(self.buf[0].keys())
        return {k: np.stack([f[k] for f in self.buf], axis=0) for k in ks}


# ── Stage 2: RL with KL anchor ─────────────────────────────────────────────

TASK_TO_ENV = {"Lift": "Lift",
               "PickPlaceCan": "PickPlaceCan",
               "NutAssemblySquare": "NutAssemblySquare",
               "Threading_D0": "Threading_D0",
               "Stack_D0": "Stack_D0",
               "MugCleanup_D0": "MugCleanup_D0",
               "Coffee_D0": "Coffee_D0",
               "StackThree_D0": "StackThree_D0"}
# Register MimicGen envs if available
try:
    import mimicgen  # registers Stack_D0, MugCleanup_D0, etc.
except ImportError:
    pass


def make_env(task_name, horizon=200, reward_shaping=False):
    try:
        ctrl = suite.load_controller_config(default_controller="OSC_POSE")
    except Exception:
        ctrl = None
    kw = dict(robots="Panda", has_renderer=False, has_offscreen_renderer=True,
              use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
              camera_heights=84, camera_widths=84,
              reward_shaping=reward_shaping, ignore_done=True,
              horizon=horizon, control_freq=20)
    if ctrl is not None:
        kw["controller_configs"] = ctrl
    return suite.make(TASK_TO_ENV[task_name], **kw)


def is_success(env):
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    if hasattr(env, "is_success"):
        s = env.is_success()
        return bool(s.get("task", False)) if isinstance(s, dict) else bool(s)
    return False


def state_from_stacked(stacked):
    parts = []
    for k in STATE_KEYS:
        v = stacked[k]
        if v.ndim >= 2:
            v = v[-1]
        parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def run_rl_with_kl(adapter, sft_snapshot, base, args, device,
                   policy_act=None):
    print("\n=== STAGE 2: RL with KL penalty (BC-T) ===", flush=True)
    print(f"  KL coef: {args.kl_coef}", flush=True)
    print(f"  Entropy coef: {args.entropy_coef}", flush=True)
    # Use the supplied policy_act callable for getting a_base if provided;
    # otherwise fall back to base.act. The robomimic RolloutPolicy preprocesses
    # observations correctly for the BC-Transformer network, while FrozenBase.act
    # double-normalises images (producing wrong actions).
    if policy_act is None:
        policy_act = base.act

    thermal = ThermalModel.from_predefined("linear", n_joints=7)
    current_m = CurrentModel.from_predefined("linear", n_joints=7)
    voltage_m = VoltageModel.from_predefined("linear", n_joints=7)

    if getattr(args, "freeze_log_std", False):
        adapter.log_std.requires_grad = False
        print(f"  log_std FROZEN at {adapter.log_std.exp().mean().item():.4f}", flush=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, adapter.parameters()),
        lr=args.rl_lr, eps=1e-5,
    )
    env = make_env(args.task_name, horizon=args.horizon,
                   reward_shaping=getattr(args, "reward_shaping", False))
    fb = FrameBuffer(base.context_length)

    n_steps = args.rl_n_steps
    best_score = -float("inf")
    gstep = 0
    recent_rews = deque(maxlen=30)
    recent_cool = deque(maxlen=20)
    recent_hot = deque(maxlen=20)
    t0 = time.time()

    def sample_telemetry(is_hot):
        n_j = 7
        if is_hot:
            return (np.random.uniform(56, 75, n_j).astype(np.float32),
                    np.random.uniform(0.7, 1.0, n_j).astype(np.float32),
                    np.random.uniform(0.5, 0.75, n_j).astype(np.float32))
        return (np.random.uniform(20, 42, n_j).astype(np.float32),
                np.random.uniform(0.1, 0.5, n_j).astype(np.float32),
                np.random.uniform(0.92, 1.0, n_j).astype(np.float32))

    use_gated_noise = getattr(args, "gated_noise", False)
    if use_gated_noise:
        print("  [gated-noise PPO] noise sampled on pre-gate delta logits", flush=True)

    def current_p_hot():
        # Curriculum: anneal p_hot from p_hot_start (early) to args.p_hot (late).
        if args.p_hot_start is None:
            return args.p_hot
        progress = min(1.0, gstep / max(1, args.rl_steps))
        return args.p_hot_start + (args.p_hot - args.p_hot_start) * progress

    while gstep < args.rl_steps:
        states, a_bases, deltas_taken, log_probs, rewards, dones, values = [], [], [], [], [], [], []
        temps_buf, currs_buf, volts_buf = [], [], []

        raw = env.reset()
        fb.reset()
        fb.push(build_robomimic_obs(raw))
        is_hot = np.random.random() < current_p_hot()
        ep_temps, ep_currs, ep_volts = sample_telemetry(is_hot)
        ep_rew = 0.0

        for _ in range(n_steps):
            stacked = fb.stacked()
            state = state_from_stacked(stacked)
            a_base = policy_act(stacked)

            s_t = torch.as_tensor(state, device=device)[None]
            ab_t = torch.as_tensor(a_base, device=device)[None]
            t_t = torch.as_tensor(ep_temps, device=device)[None]
            c_t = torch.as_tensor(ep_currs, device=device)[None]
            v_t = torch.as_tensor(ep_volts, device=device)[None]

            with torch.no_grad():
                if use_gated_noise:
                    action, delta_taken, lp, val = adapter.gated_sample(ab_t, t_t, s_t, c_t, v_t)
                    action_np = action[0].cpu().numpy().clip(-1, 1)
                    delta_np = delta_taken[0].cpu().numpy()
                else:
                    action, lp, _, val = adapter.get_action_and_value(ab_t, t_t, s_t, c_t, v_t)
                    action_np = action[0].cpu().numpy().clip(-1, 1)
                    delta_np = action_np  # legacy: action IS the policy output

            a_exec = thermal.apply_thermal_physics(action_np.copy(), ep_temps)
            current_m.reset()
            a_exec = current_m.apply_current_physics(a_exec, ep_currs)
            a_exec = voltage_m.apply_voltage_physics(a_exec, ep_volts)

            raw, r, done, info = env.step(a_exec)
            success = is_success(env)
            if success:
                r += 300.0

            states.append(state); a_bases.append(a_base); deltas_taken.append(delta_np)
            log_probs.append(lp.item()); values.append(val.item())
            rewards.append(float(r)); dones.append(float(done or success))
            temps_buf.append(ep_temps); currs_buf.append(ep_currs); volts_buf.append(ep_volts)
            ep_rew += r; gstep += 1

            if done or success:
                recent_rews.append(ep_rew)
                (recent_hot if is_hot else recent_cool).append(int(success))
                ep_rew = 0.0
                raw = env.reset()
                fb.reset()
                fb.push(build_robomimic_obs(raw))
                # Reset BC-T policy episode state (no-op for transformer).
                if hasattr(policy_act, "__self__") and hasattr(policy_act.__self__, "start_episode"):
                    policy_act.__self__.start_episode()
                is_hot = np.random.random() < current_p_hot()
                ep_temps, ep_currs, ep_volts = sample_telemetry(is_hot)
            else:
                fb.push(build_robomimic_obs(raw))

            if gstep >= args.rl_steps:
                break

        # PPO update with KL
        S = torch.as_tensor(np.array(states), device=device)
        AB = torch.as_tensor(np.array(a_bases), device=device)
        A = torch.as_tensor(np.array(deltas_taken), device=device)
        LP = torch.as_tensor(np.array(log_probs), device=device)
        R = torch.as_tensor(np.array(rewards), device=device)
        D = torch.as_tensor(np.array(dones), device=device)
        V = torch.as_tensor(np.array(values), device=device)
        T = torch.as_tensor(np.array(temps_buf), device=device)
        C = torch.as_tensor(np.array(currs_buf), device=device)
        VV = torch.as_tensor(np.array(volts_buf), device=device)

        with torch.no_grad():
            last_val = adapter.get_value(AB[-1:], T[-1:], S[-1:], C[-1:], VV[-1:])
        adv = torch.zeros_like(R)
        lastgae = torch.tensor(0.0, device=device)
        for k in reversed(range(len(R))):
            nv = last_val if k == len(R) - 1 else V[k + 1]
            nd = 1.0 - (D[k] if k == len(R) - 1 else D[k + 1])
            delta = R[k] + 0.99 * nv * nd - V[k]
            adv[k] = lastgae = delta + 0.99 * 0.95 * nd * lastgae
        returns = adv + V

        adapter.train()
        kl_loss_val = 0.0
        for _ in range(args.rl_epochs):
            idx = torch.randperm(len(S))
            for start in range(0, len(S), args.batch_size):
                ix = idx[start:start + args.batch_size]
                if use_gated_noise:
                    new_lp, ent, new_val = adapter.gated_log_prob(
                        AB[ix], A[ix], T[ix], S[ix], C[ix], VV[ix])
                    # KL between current and SFT policy distributions over delta_logits.
                    if args.kl_coef > 0 and sft_snapshot is not None:
                        new_dm, new_log_std, _, _ = adapter.forward_delta(
                            AB[ix], T[ix], S[ix], C[ix], VV[ix])
                        with torch.no_grad():
                            sft_dm, sft_log_std, _, _ = sft_snapshot.forward_delta(
                                AB[ix], T[ix], S[ix], C[ix], VV[ix])
                            dist_sft = Normal(sft_dm, sft_log_std.exp().expand_as(sft_dm))
                        dist_new = Normal(new_dm, new_log_std.exp().expand_as(new_dm))
                        kl_loss = torch.distributions.kl_divergence(dist_new, dist_sft).sum(-1).mean()
                    else:
                        kl_loss = torch.tensor(0.0, device=device)
                else:
                    new_mean, new_log_std, new_val = adapter(AB[ix], T[ix], S[ix], C[ix], VV[ix])
                    new_std = new_log_std.exp().expand_as(new_mean)
                    dist_new = Normal(new_mean, new_std)
                    new_lp = dist_new.log_prob(A[ix]).sum(-1)
                    ent = dist_new.entropy().sum(-1)
                    if args.kl_coef > 0 and sft_snapshot is not None:
                        with torch.no_grad():
                            sft_mean, sft_log_std, _ = sft_snapshot(AB[ix], T[ix], S[ix], C[ix], VV[ix])
                            sft_std = sft_log_std.exp().expand_as(sft_mean)
                            dist_sft = Normal(sft_mean, sft_std)
                        kl_loss = torch.distributions.kl_divergence(dist_new, dist_sft).sum(-1).mean()
                    else:
                        kl_loss = torch.tensor(0.0, device=device)
                a_norm = (adv[ix] - adv[ix].mean()) / (adv[ix].std() + 1e-8)
                ratio = (new_lp - LP[ix]).exp()
                pg_loss = torch.max(-a_norm * ratio,
                                    -a_norm * ratio.clamp(0.8, 1.2)).mean()
                vf_loss = ((returns[ix] - new_val) ** 2).mean()
                loss = (pg_loss + 0.5 * vf_loss
                        - args.entropy_coef * ent.mean()
                        + args.kl_coef * kl_loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)
                optimizer.step()
                kl_loss_val = kl_loss.item()

        fps = gstep / max(1.0, time.time() - t0)
        cs = float(np.mean(recent_cool)) if recent_cool else 0.0
        hs = float(np.mean(recent_hot))  if recent_hot  else 0.0
        mr = float(np.mean(recent_rews)) if recent_rews else 0.0
        print(f"  RL step={gstep:>6} rew={mr:.0f} cool={cs:.0%} hot={hs:.0%} "
              f"kl={kl_loss_val:.4f} fps={fps:.1f}", flush=True)

        score = 2 * hs + cs
        if score > best_score:
            best_score = score
            torch.save(adapter.state_dict(),
                       os.path.join(args.output_dir, "best_adapter.pt"))

        if args.checkpoint_every and gstep % args.checkpoint_every < n_steps:
            torch.save(adapter.state_dict(),
                       os.path.join(args.output_dir, f"adapter_step{gstep}.pt"))

    env.close()
    torch.save(adapter.state_dict(),
               os.path.join(args.output_dir, "final_adapter.pt"))
    print(f"  RL done. Best score={best_score:.2f}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", required=True, choices=list(TASK_TO_ENV.keys()))
    ap.add_argument("--base-ckpt", required=True, help="BC-Transformer .pth")
    ap.add_argument("--demo-hdf5", required=True, help="robomimic demo HDF5")
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.3)

    ap.add_argument("--sft-steps", type=int, default=20000)
    ap.add_argument("--sft-lr", type=float, default=5e-4)

    ap.add_argument("--rl-steps", type=int, default=60000)
    ap.add_argument("--rl-lr", type=float, default=1e-5)
    ap.add_argument("--rl-n-steps", type=int, default=512)
    ap.add_argument("--rl-epochs", type=int, default=5)
    ap.add_argument("--p-hot", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=200)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--entropy-coef", type=float, default=0.05)
    ap.add_argument("--checkpoint-every", type=int, default=20000)
    ap.add_argument("--log-std-init", type=float, default=-2.0,
                    help="Initial log std for the action distribution. -2.0 = std 0.135 (OpenVLA default), -3.0 = 0.05, -3.5 = 0.03.")
    ap.add_argument("--freeze-log-std", action="store_true",
                    help="Keep log_std fixed at its init value (RL only learns mean+value).")
    ap.add_argument("--gated-noise", action="store_true",
                    help="Sample noise on the pre-gate delta logits (not on the env action). "
                         "At cool conditions gate=0, so action is bit-identical to base regardless "
                         "of noise. At stress conditions noise is bounded by alpha*tanh-Jacobian. "
                         "Strongly recommended for BC-T bases.")
    ap.add_argument("--reward-shaping", action="store_true",
                    help="Use robosuite's built-in dense reward shaping (reach/grasp/lift "
                         "partials). Necessary for PPO to get gradient at hot conditions where "
                         "successful trajectories are rare.")
    ap.add_argument("--p-hot-start", type=float, default=None,
                    help="Curriculum: linearly anneal p_hot from this value (early) to "
                         "--p-hot (late). If None, p_hot is constant at --p-hot.")
    ap.add_argument("--load-sft", type=str, default=None,
                    help="Skip SFT and load weights from this path (for fast RL-only experiments).")
    ap.add_argument("--skip-rl", action="store_true",
                    help="Run only SFT stage and exit (for sanity tests).")
    ap.add_argument("--gate-shape", type=str, default="linear",
                    choices=["linear", "smoothstep"],
                    help="Per-channel ramp: 'linear' (legacy) or 'smoothstep' "
                         "(3x^2-2x^3, gentler at the warm shoulder).")
    ap.add_argument("--adapter-class", type=str, default="bct",
                    choices=["bct", "tambot", "tambotv2",
                             "tambot_xl", "tambot_film", "tambot_moe"],
                    help="bct = BCTSFTRLAdapter (legacy); "
                         "tambot = TAM-BoT body-token transformer; "
                         "tambotv2 = TAM-BoT-v2 with per-channel γ/δ heads; "
                         "tambot_xl = scaled-up TAM-BoT (h=256, 6 layers); "
                         "tambot_film = TAM-BoT with per-layer FiLM conditioning; "
                         "tambot_moe = severity-expert mixture (3 experts).")
    ap.add_argument("--tam-n-layers", type=int, default=2,
                    help="Transformer encoder layers (TAM-BoT only).")
    ap.add_argument("--tam-n-heads", type=int, default=4,
                    help="Attention heads (TAM-BoT only). Must divide --hidden.")
    ap.add_argument("--tam-gamma-range", type=float, default=0.5,
                    help="γ ∈ [1-r, 1+r] when gate=1 (TAM-BoT only).")
    ap.add_argument("--tam-mask-gripper", action="store_true",
                    help="Force γ_6=1, δ_6=0 (no correction on the gripper action).")
    ap.add_argument("--tam-gamma-log-space", action="store_true",
                    help="γ = exp(g·r·tanh(logit)) instead of 1+g·r·tanh(logit).")
    ap.add_argument("--tam-action-magnitude", action="store_true",
                    help="Include |a_base| in the per-joint token features.")
    ap.add_argument("--tam-no-state-token", action="store_true",
                    help="Skip the state token (joints only).")
    ap.add_argument("--tam-single-token", action="store_true",
                    help="Pool joints into a single token (no per-joint tokens).")
    ap.add_argument("--tam-gate-scope", type=str, default="per_joint",
                    choices=["per_joint", "scalar"])
    ap.add_argument("--tam-backbone", type=str, default="transformer",
                    choices=["transformer", "mlp"])
    ap.add_argument("--tam-action-mag-gate", action="store_true",
                    help="Modulate gate by |a_base| — only correct when base "
                         "is committed. Useful for cross-base transfer.")
    ap.add_argument("--tam-action-mag-threshold", type=float, default=0.3)
    ap.add_argument("--coupling-TC", type=float, default=0.0,
                    help="Temperature->current coupling strength in SFT target generation. "
                         "If >0, the adapter is trained against targets derived from coupled physics "
                         "(C_eff = C * (1 + alpha_TC * T_norm)); the adapter still observes nominal (T,C,V).")
    ap.add_argument("--coupling-CV", type=float, default=0.0,
                    help="Current->voltage coupling strength in SFT target generation. "
                         "If >0, V_eff = V - alpha_CV * C_eff during target generation.")
    ap.add_argument("--curve-family", type=str, default="linear",
                    choices=["linear", "exponential", "sigmoid", "polynomial", "kinky"],
                    help="Curve family for the degradation factor used in the SFT target. "
                         "'kinky' = piecewise-linear with sharp kinks; trains the adapter against "
                         "a curve that polynomial regression cannot fit.")
    ap.add_argument("--alpha-mag", type=float, default=0.0,
                    help="Non-factorizable ρ: ρ_C(C, |a|) = ρ_C(C) * (1 - alpha_mag * |a|). "
                         "Trains the adapter against targets that capture action-magnitude dependence "
                         "of the current capacity (motor saturation). FFwd's 1/ρ̂(τ) cannot see |a|.")
    ap.add_argument("--tam-action-mag-steepness", type=float, default=5.0)
    ap.add_argument("--tam-single-severity", action="store_true",
                    help="Use only one severity tier during SFT (instead of clean+mild+mod+sev).")
    ap.add_argument("--num-sev-tiers", type=int, default=4, choices=[1, 2, 3, 4],
                    help="Severity-tier curriculum ablation. 4=full (clean+mild+mod+sev); "
                         "3=clean+mild+mod; 2=clean+mild; 1=clean only. Lower values test whether "
                         "the full curriculum is necessary or if subsets suffice.")
    ap.add_argument("--bench-calib-sigma", type=float, default=0.0,
                    help="Bench-calibration noise σ on the SFT-target ρ. ρ̂_target = ρ_true * (1 + σ*N(0,1)). "
                         "Simulates training TAM against noisy bench measurements rather than oracle ρ. "
                         "Addresses Limitation (ii) self-validation: tests how well TAM generalises when "
                         "its training-time ρ̂ is imperfect relative to the env's true ρ.")
    ap.add_argument("--sft-input-noise", type=float, default=0.0,
                    help="Std-dev of Gaussian noise added to the a_base input "
                         "during SFT (simulates base-policy variation; helps "
                         "cross-base transfer). 0.0 = off.")
    ap.add_argument("--extra-demo-hdf5", type=str, default=None,
                    help="Optional second demo HDF5 to mix in (for multi-base / "
                         "multi-task distillation training).")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)

    print("=" * 70)
    print(f"  BC-T SFT->RL+KL  task={args.task_name}  base={Path(args.base_ckpt).name}")
    print(f"  hidden={args.hidden} alpha={args.alpha} kl={args.kl_coef} ent={args.entropy_coef}")
    print(f"  sft_steps={args.sft_steps:,}  rl_steps={args.rl_steps:,}")
    print("=" * 70, flush=True)

    base = FrozenBase(args.base_ckpt, device=str(device))
    state_dim = base.state_dim
    print(f"  base loaded. state_dim={state_dim} act_dim={base.act_dim} "
          f"context={base.context_length}", flush=True)

    if args.adapter_class == "tambot":
        from thermal_adapters.tam_bot import TAMBoT
        adapter = TAMBoT(
            state_dim=state_dim, act_dim=base.act_dim,
            hidden=args.hidden, n_layers=args.tam_n_layers,
            n_heads=args.tam_n_heads, alpha=args.alpha,
            gamma_range=args.tam_gamma_range,
            log_std_init=args.log_std_init, gate_shape=args.gate_shape,
            mask_gripper=args.tam_mask_gripper,
            gamma_log_space=args.tam_gamma_log_space,
            use_action_magnitude=args.tam_action_magnitude,
            use_state_token=not args.tam_no_state_token,
            use_per_joint_tokens=not args.tam_single_token,
            gate_scope=args.tam_gate_scope,
            backbone=args.tam_backbone,
            action_mag_gate=args.tam_action_mag_gate,
            action_mag_threshold=args.tam_action_mag_threshold,
            action_mag_steepness=args.tam_action_mag_steepness,
        ).to(device)
        # TAMBoT now supports gated-noise PPO via forward_delta/gated_sample/
        # gated_log_prob. RL is allowed.
    elif args.adapter_class == "tambotv2":
        from thermal_adapters.tam_bot_v2 import TAMBoTv2
        adapter = TAMBoTv2(
            state_dim=state_dim, act_dim=base.act_dim,
            hidden=args.hidden, n_layers=args.tam_n_layers,
            n_heads=args.tam_n_heads, alpha=args.alpha,
            gamma_range=args.tam_gamma_range,
            log_std_init=args.log_std_init, gate_shape=args.gate_shape,
        ).to(device)
        if not args.skip_rl:
            print("  WARNING: TAM-BoT-v2 does not yet support gated-noise PPO. "
                  "Forcing --skip-rl.", flush=True)
            args.skip_rl = True
    elif args.adapter_class in ("tambot_xl", "tambot_film", "tambot_moe"):
        from thermal_adapters.tam_bot_novel import TAMBoTXL, TAMBoTFiLM, TAMBoTMoE
        if args.adapter_class == "tambot_xl":
            adapter = TAMBoTXL(
                state_dim=state_dim, act_dim=base.act_dim,
                hidden=max(args.hidden, 256), n_layers=max(args.tam_n_layers, 6),
                n_heads=args.tam_n_heads, alpha=args.alpha,
                gamma_range=args.tam_gamma_range, log_std_init=args.log_std_init,
            ).to(device)
        elif args.adapter_class == "tambot_film":
            adapter = TAMBoTFiLM(
                state_dim=state_dim, act_dim=base.act_dim,
                hidden=args.hidden, n_layers=max(args.tam_n_layers, 4),
                n_heads=args.tam_n_heads, alpha=args.alpha,
                gamma_range=args.tam_gamma_range, log_std_init=args.log_std_init,
            ).to(device)
        else:  # tambot_moe
            adapter = TAMBoTMoE(
                state_dim=state_dim, act_dim=base.act_dim,
                hidden=args.hidden, n_layers=args.tam_n_layers,
                n_heads=args.tam_n_heads, alpha=args.alpha,
                gamma_range=args.tam_gamma_range, log_std_init=args.log_std_init,
            ).to(device)
        if not args.skip_rl:
            print(f"  WARNING: {args.adapter_class} does not yet support gated-noise PPO. "
                  "Forcing --skip-rl.", flush=True)
            args.skip_rl = True
    else:
        adapter = BCTSFTRLAdapter(
            state_dim=state_dim, act_dim=base.act_dim,
            hidden=args.hidden, n_blocks=args.n_blocks, alpha=args.alpha,
            log_std_init=args.log_std_init, gate_shape=args.gate_shape,
        ).to(device)
    print(f"  adapter params: {sum(p.numel() for p in adapter.parameters()):,}",
          flush=True)

    # Stage 1: SFT (or skip and load)
    if args.load_sft is not None:
        print(f"  Loading SFT weights from {args.load_sft} (skipping SFT stage)", flush=True)
        adapter.load_state_dict(torch.load(args.load_sft, map_location=device))
        # Ensure log_std is at requested init value (override the loaded value).
        with torch.no_grad():
            adapter.log_std.fill_(float(args.log_std_init))
        torch.save(adapter.state_dict(), os.path.join(args.output_dir, "sft_adapter.pt"))
    else:
        states, actions = load_demos(args.demo_hdf5)
        print(f"  loaded demos: {len(states)} (state_dim={states.shape[1]})", flush=True)
        if getattr(args, "extra_demo_hdf5", None):
            extra_states, extra_actions = load_demos(args.extra_demo_hdf5)
            if extra_states.shape[1] == states.shape[1]:
                states = np.concatenate([states, extra_states], axis=0)
                actions = np.concatenate([actions, extra_actions], axis=0)
                print(f"  + extra demos: {len(extra_states)} (combined: {len(states)})", flush=True)
            else:
                print(f"  WARN: extra-demo state_dim {extra_states.shape[1]} != "
                      f"primary {states.shape[1]} — skipping mix", flush=True)
        # Demo state_dim may exceed base.state_dim (e.g. Threading: demo=37, BC base=19).
        # In that case, slice demo states to base's state_dim — the SFT objective uses
        # demo actions as targets and only needs the state slice the adapter consumes.
        if states.shape[1] != state_dim:
            if states.shape[1] > state_dim:
                print(f"  Demo state_dim {states.shape[1]} > base.state_dim {state_dim}; "
                      f"slicing demo states to first {state_dim} dims", flush=True)
                states = states[:, :state_dim]
            else:
                states = np.concatenate(
                    [states, np.zeros((states.shape[0], state_dim - states.shape[1]),
                                      dtype=states.dtype)], axis=1)
                print(f"  Padded demo state_dim {states.shape[1] - (state_dim - states.shape[1])} -> {state_dim}",
                      flush=True)
        run_sft(adapter, states, actions, args, device)

    if args.skip_rl:
        print("  --skip-rl set; exiting after SFT.", flush=True)
        return

    # Snapshot for KL — same class as adapter
    if args.adapter_class == "tambot":
        from thermal_adapters.tam_bot import TAMBoT
        sft_snapshot = TAMBoT(
            state_dim=state_dim, act_dim=base.act_dim,
            hidden=args.hidden, n_layers=args.tam_n_layers,
            n_heads=args.tam_n_heads, alpha=args.alpha,
            gamma_range=args.tam_gamma_range,
            log_std_init=args.log_std_init, gate_shape=args.gate_shape,
            mask_gripper=args.tam_mask_gripper,
            gamma_log_space=args.tam_gamma_log_space,
            use_action_magnitude=args.tam_action_magnitude,
            use_state_token=not args.tam_no_state_token,
            use_per_joint_tokens=not args.tam_single_token,
            gate_scope=args.tam_gate_scope,
            backbone=args.tam_backbone,
            action_mag_gate=args.tam_action_mag_gate,
            action_mag_threshold=args.tam_action_mag_threshold,
            action_mag_steepness=args.tam_action_mag_steepness,
        ).to(device)
    else:
        sft_snapshot = BCTSFTRLAdapter(
            state_dim=state_dim, act_dim=base.act_dim,
            hidden=args.hidden, n_blocks=args.n_blocks, alpha=args.alpha,
            log_std_init=args.log_std_init, gate_shape=args.gate_shape,
        ).to(device)
    sft_snapshot.load_state_dict(adapter.state_dict())
    sft_snapshot.eval()
    for p in sft_snapshot.parameters():
        p.requires_grad = False
    print("  SFT snapshot frozen.", flush=True)

    # Build RolloutPolicy (correct image preprocessing) for getting a_base
    # during the RL rollout. FrozenBase.act double-normalises images, leading
    # to wrong base actions; this avoids that.
    from robomimic.utils.file_utils import policy_from_checkpoint
    rollout_policy, _ = policy_from_checkpoint(
        ckpt_path=args.base_ckpt, device=device, verbose=False,
    )
    rollout_policy.start_episode()

    def policy_act(stacked):
        # Returns a numpy action (act_dim,) from the BC-T base, with the
        # same preprocessing the policy was trained on.
        return rollout_policy(ob=stacked)

    # Stage 2: RL with KL
    run_rl_with_kl(adapter, sft_snapshot, base, args, device,
                   policy_act=policy_act)
    print(f"\n  Done. Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
