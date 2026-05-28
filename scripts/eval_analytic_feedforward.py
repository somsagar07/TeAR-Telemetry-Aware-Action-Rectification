#!/usr/bin/env python3
"""Analytic feedforward baseline: a' = a_base / rho_hat(tau).

This is exactly the SFT target TAM is trained against, but applied
*directly* as a controller — no neural adapter, no training, no parameters.
The structural smoothstep gate is still applied (a'_j = a_base_j only when
gate_j(tau) == 0; otherwise a'_j = a_base_j / rho_hat_j gated).

If this baseline matches TAM's headline gain, then TAM is just learning
the closed-form inverse the simulator implements. If TAM beats it,
the learned features add value beyond the analytic formula.

Usage:
  python scripts/eval_analytic_feedforward.py --ckpt <BC-T> --task square --out <json>
"""
import argparse, os, sys, json
from collections import deque
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import importlib, unittest.mock as _mock
def _mk(name):
    m = _mock.MagicMock(); m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    m.__name__ = name; m.__path__ = []
    return m
for _m in ["tensorflow","tensorflow.python","tensorflow.python.framework",
          "tensorboard","tensorboard.compat","tensorboard.compat.tf",
          "torch.utils.tensorboard","torch.utils.tensorboard.writer",
          "torch.utils.tensorboard._embedding","mujoco_py"]:
    sys.modules.setdefault(_m, _mk(_m))

import robosuite as suite
try:
    import mimicgen, mimicgen.envs
except Exception:
    pass

from robomimic.utils.file_utils import policy_from_checkpoint
from env.thermal_model import ThermalModel
from env.telemetry_model import CurrentModel, VoltageModel


# Smoothstep ramp identical to TAMBoT.per_joint_gate
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)


def per_joint_gate(T, C, V):
    """Per-joint gate in [0,1]. Same as TAMBoT.per_joint_gate."""
    tg = np.clip((T - 42.0)/13.0, 0, 1)
    cg = np.clip((C - 0.60)/0.40, 0, 1)
    vg = np.clip((0.90 - V)/0.40, 0, 1)
    g = np.clip(smoothstep(tg) + smoothstep(cg) + smoothstep(vg), 0.0, 1.0)
    return g  # (7,)


def rho_hat(T, C, V):
    """Analytic per-joint capacity factor (product of three channel factors).
    Uses the SAME closed forms as env/thermal_model.ThermalModel.linear and
    env/telemetry_model.{CurrentModel,VoltageModel}.linear.
    """
    # ThermalModel.linear: ρ_T = 1 - 0.95 * clip((T - 20)/(80 - 20), 0, 1)
    rho_T = 1 - 0.95 * np.clip((T - 20)/60.0, 0, 1)
    # CurrentModel.linear: ρ_C = 1 - 0.50 * clip((C - 0.50)/(1.20 - 0.50), 0, 1)
    rho_C = 1 - 0.50 * np.clip((C - 0.50)/0.70, 0, 1)
    # VoltageModel.linear: ρ_V = 1 - 0.40 * clip((1.00 - V)/(1.00 - 0.45), 0, 1)
    rho_V = 1 - 0.40 * np.clip((1.0 - V)/0.55, 0, 1)
    rho = np.clip(rho_T * rho_C * rho_V, 0.05, 1.0)
    return rho  # (7,)


def analytic_correction(a_base, T, C, V):
    """Apply analytic feedforward inverse, gated.
    a'[j] = a_base[j] * (1 - g[j]) + clip(a_base[j] / rho_hat[j], -1, 1) * g[j]
    Acts on the 7-DoF action."""
    g = per_joint_gate(T, C, V)
    rho = rho_hat(T, C, V)
    a_corrected = np.clip(a_base[:7] / rho, -1, 1)
    a_out = a_base.copy()
    a_out[:7] = a_base[:7] * (1 - g) + a_corrected * g
    return a_out


CONDITIONS = {
    "cool":     dict(T=(20,42), C=(0.10,0.50), V=(0.92,1.00)),
    "hot":      dict(T=(56,75), C=(0.10,0.50), V=(0.92,1.00)),
    "stall":    dict(T=(20,42), C=(0.85,1.05), V=(0.92,1.00)),
    "brownout": dict(T=(20,42), C=(0.10,0.50), V=(0.45,0.68)),
    "T_mod":    dict(T=(58,58), C=(0.30,0.30), V=(1.00,1.00)),
    "TC_mod":   dict(T=(55,55), C=(0.75,0.75), V=(1.00,1.00)),
    "TV_mod":   dict(T=(55,55), C=(0.30,0.30), V=(0.72,0.72)),
    "TCV_mod":  dict(T=(55,55), C=(0.75,0.75), V=(0.72,0.72)),
}


def sample_telemetry(cond, n_joints=7):
    spec = CONDITIONS[cond]
    T = np.random.uniform(*spec["T"], n_joints).astype(np.float32)
    C = np.random.uniform(*spec["C"], n_joints).astype(np.float32)
    V = np.random.uniform(*spec["V"], n_joints).astype(np.float32)
    return T, C, V


TASK_TO_ENV = {"lift": "Lift", "can": "PickPlaceCan", "square": "NutAssemblySquare",
               "threading": "Threading_D0"}


def make_env(task, horizon):
    return suite.make(
        TASK_TO_ENV[task], robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview","robot0_eye_in_hand"],
        camera_heights=84, camera_widths=84,
        reward_shaping=False, ignore_done=True,
        horizon=horizon, control_freq=20,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_TO_ENV))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"[load] {args.ckpt}", flush=True)
    policy, _ = policy_from_checkpoint(ckpt_path=args.ckpt)

    thermal = ThermalModel.from_predefined("linear", n_joints=7)
    current = CurrentModel.from_predefined("linear", n_joints=7)
    voltage = VoltageModel.from_predefined("linear", n_joints=7)

    results = {}
    env = make_env(args.task, args.horizon)
    import time
    for cond in args.conditions:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        t0 = time.time(); successes = 0
        for ep in range(args.episodes):
            T, C, V = sample_telemetry(cond)
            raw = env.reset(); policy.start_episode()
            success = False
            for step in range(args.horizon):
                obs = {}
                for k in ("agentview_image","robot0_eye_in_hand_image"):
                    if k in raw:
                        img = raw[k][::-1].copy().transpose(2, 0, 1).astype(np.float32)/255.0
                        obs[k] = img
                for k in ("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos"):
                    if k in raw: obs[k] = raw[k].astype(np.float32)
                if "object-state" in raw: obs["object"] = raw["object-state"].astype(np.float32)
                action = policy(ob=obs)
                action = np.asarray(action, dtype=np.float32).reshape(-1).clip(-1.0, 1.0)
                # APPLY ANALYTIC FEEDFORWARD
                action = analytic_correction(action, T, C, V)
                # Apply degradation physics
                action = thermal.apply_thermal_physics(action, T)
                action = current.apply_current_physics(action, C)
                action = voltage.apply_voltage_physics(action, V)
                raw, _, _, _ = env.step(action)
                if env._check_success(): success = True; break
            successes += int(success)
        sr = successes / args.episodes
        print(f"  {cond}: sr={sr:.0%}  ({time.time()-t0:.0f}s)", flush=True)
        results[cond] = {"sr": sr, "n": args.episodes}
    env.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": "analytic_feedforward",
        "task": args.task,
        "ckpt": args.ckpt,
        "method": "a' = base * (1-g) + clip(base/rho_hat, -1, 1) * g, g=smoothstep gate, rho_hat=linear curves",
        "episodes": args.episodes, "horizon": args.horizon, "seed": args.seed,
        "results": results,
    }, open(args.out, 'w'), indent=2)
    print(f"Saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
