#!/usr/bin/env python3
"""Eval an architecture variant from sft_arch_variants.py at N=20 across all
6 telemetry conditions. Supports default and tight gate via --gate-corners.
"""
import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime
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
for _m in ["tensorflow", "tensorflow.python", "tensorflow.python.framework",
           "tensorboard", "tensorboard.compat", "tensorboard.compat.tf",
           "torch.utils.tensorboard", "torch.utils.tensorboard.writer",
           "torch.utils.tensorboard._embedding", "mujoco_py"]:
    sys.modules.setdefault(_m, _mk(_m))

import robosuite as suite
from robomimic.utils.file_utils import policy_from_checkpoint
from env.telemetry_model import CurrentModel, VoltageModel
from env.thermal_model import ThermalModel
from frozen_base import FrozenBase, STATE_KEYS
from thermal_adapters.sft_arch_variants import build, hard_gate, VARIANT_REGISTRY


CONDITIONS = {
    "cool":         (lambda n: np.random.uniform(20, 42, n).astype(np.float32),
                     lambda n: np.random.uniform(0.10, 0.50, n).astype(np.float32),
                     lambda n: np.random.uniform(0.92, 1.00, n).astype(np.float32)),
    "hot":          (lambda n: np.random.uniform(56, 75, n).astype(np.float32),
                     lambda n: np.random.uniform(0.10, 0.50, n).astype(np.float32),
                     lambda n: np.random.uniform(0.92, 1.00, n).astype(np.float32)),
    "stall":        (lambda n: np.random.uniform(20, 42, n).astype(np.float32),
                     lambda n: np.random.uniform(0.85, 1.05, n).astype(np.float32),
                     lambda n: np.random.uniform(0.92, 1.00, n).astype(np.float32)),
    "brownout":     (lambda n: np.random.uniform(20, 42, n).astype(np.float32),
                     lambda n: np.random.uniform(0.10, 0.50, n).astype(np.float32),
                     lambda n: np.random.uniform(0.45, 0.68, n).astype(np.float32)),
    "all_moderate": (lambda n: np.random.uniform(43, 55, n).astype(np.float32),
                     lambda n: np.random.uniform(0.60, 0.85, n).astype(np.float32),
                     lambda n: np.random.uniform(0.70, 0.88, n).astype(np.float32)),
    "all_severe":   (lambda n: np.random.uniform(56, 75, n).astype(np.float32),
                     lambda n: np.random.uniform(0.85, 1.05, n).astype(np.float32),
                     lambda n: np.random.uniform(0.45, 0.68, n).astype(np.float32)),
    "T_mod":      (lambda n: np.full(n, 58.0, dtype=np.float32),
                   lambda n: np.full(n, 0.30, dtype=np.float32),
                   lambda n: np.full(n, 1.00, dtype=np.float32)),
    "TC_mod":     (lambda n: np.full(n, 55.0, dtype=np.float32),
                   lambda n: np.full(n, 0.75, dtype=np.float32),
                   lambda n: np.full(n, 1.00, dtype=np.float32)),
    "TV_mod":     (lambda n: np.full(n, 55.0, dtype=np.float32),
                   lambda n: np.full(n, 0.30, dtype=np.float32),
                   lambda n: np.full(n, 0.72, dtype=np.float32)),
    "TCV_mod":    (lambda n: np.full(n, 55.0, dtype=np.float32),
                   lambda n: np.full(n, 0.75, dtype=np.float32),
                   lambda n: np.full(n, 0.72, dtype=np.float32)),
}
TASK_TO_ENV = {"lift": "Lift", "can": "PickPlaceCan", "square": "NutAssemblySquare",
               "threading": "Threading_D0", "stack": "Stack_D0"}
# MimicGen variants (Stack_D0, Threading_D0) require importing mimicgen to register the envs.
try:
    import mimicgen  # noqa: F401  registers Stack_D0, Threading_D0, etc.
except ImportError:
    pass


def make_env(task_name, horizon):
    try:
        ctrl = suite.load_controller_config(default_controller="OSC_POSE")
    except Exception:
        ctrl = None
    kw = dict(robots="Panda", has_renderer=False, has_offscreen_renderer=True,
              use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
              camera_heights=84, camera_widths=84,
              reward_shaping=False, ignore_done=True,
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


def build_robomimic_obs(raw):
    obs = {}
    for k_in in ("agentview_image", "robot0_eye_in_hand_image"):
        if k_in in raw:
            img = raw[k_in]
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img[::-1].copy()
                img = img.transpose(2, 0, 1)
            obs[k_in] = (img.astype(np.float32) / 255.0)
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


def state_from_stacked(stacked):
    parts = []
    for k in STATE_KEYS:
        v = stacked[k]
        if v.ndim >= 2:
            v = v[-1]
        parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def run_episodes(policy, base, adapter, task_name, n_episodes, horizon,
                 t_fn, c_fn, v_fn, thermal_model, current_model, voltage_model,
                 device="cuda:0", gate_corners=None, verbose=False):
    env = make_env(task_name, horizon)
    n_joints = 7
    successes = 0
    rewards, lengths = [], []
    fb = FrameBuffer(base.context_length)
    # Non-transformer (BC) policies expect a single frame, not the time-stacked
    # sequence; feeding the stacked tensor makes images 5-D and breaks conv2d.
    is_transformer = base.context_length > 1

    for ep in range(n_episodes):
        T = t_fn(n_joints)
        C = c_fn(n_joints)
        V = v_fn(n_joints)
        raw = env.reset()
        policy.start_episode()
        fb.reset()
        fb.push(build_robomimic_obs(raw))
        ep_ret = 0.0
        success = False
        for t in range(horizon):
            stacked = fb.stacked()
            policy_obs = stacked if is_transformer else {k: v[-1] for k, v in stacked.items()}
            a_base = policy(ob=policy_obs)
            state = state_from_stacked(stacked)
            s_t = torch.as_tensor(state, dtype=torch.float32, device=device)[None]
            ab_t = torch.as_tensor(a_base, dtype=torch.float32, device=device)[None]
            tm_t = torch.as_tensor(T, dtype=torch.float32, device=device)[None]
            cur_t = torch.as_tensor(C, dtype=torch.float32, device=device)[None]
            vol_t = torch.as_tensor(V, dtype=torch.float32, device=device)[None]
            mean, _, _ = adapter(ab_t, tm_t, s_t, cur_t, vol_t)
            if gate_corners is not None:
                T_lo, T_hi, C_lo, C_hi, V_lo, V_hi = gate_corners
                g = hard_gate(tm_t, cur_t, vol_t, T_lo, T_hi, C_lo, C_hi, V_lo, V_hi)
                mean = ab_t + g * (mean - ab_t)
            a = mean[0].cpu().numpy().clip(-1, 1)
            ad = thermal_model.apply_thermal_physics(a.copy(), T)
            current_model.reset()
            ad = current_model.apply_current_physics(ad, C)
            ad = voltage_model.apply_voltage_physics(ad, V)
            raw, r, done, info = env.step(ad)
            ep_ret += r
            fb.push(build_robomimic_obs(raw))
            if is_success(env):
                success = True
                break
        successes += int(success)
        rewards.append(ep_ret)
        lengths.append(t + 1)
        if verbose:
            print(f"    ep {ep+1}/{n_episodes}: success={success} len={t+1}", flush=True)
    env.close()
    return {
        "sr": successes / n_episodes,
        "mean_rew": float(np.mean(rewards)),
        "std_rew": float(np.std(rewards)),
        "mean_len": float(np.mean(lengths)),
        "n": n_episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_TO_ENV.keys()))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=list(VARIANT_REGISTRY))
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gate-corners", type=float, nargs=6, default=None,
                    metavar=("T_lo", "T_hi", "C_lo", "C_hi", "V_lo", "V_hi"))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[load policy] {args.ckpt}", flush=True)
    policy, _ = policy_from_checkpoint(ckpt_path=args.ckpt, device=device, verbose=False)
    base = FrozenBase(args.ckpt, device=device)

    print(f"[build adapter] {args.variant}", flush=True)
    adapter = build(args.variant, state_dim=base.state_dim, act_dim=base.act_dim,
                    alpha=args.alpha).to(device)
    sd = torch.load(args.adapter_ckpt, map_location=device)
    adapter.load_state_dict(sd)
    adapter.eval()
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  params: {n_params:,}", flush=True)

    thermal = ThermalModel.from_predefined("linear", n_joints=7)
    current = CurrentModel.from_predefined("linear", n_joints=7)
    voltage = VoltageModel.from_predefined("linear", n_joints=7)

    results = {}
    for cond in args.conditions:
        if cond not in CONDITIONS:
            continue
        t_fn, c_fn, v_fn = CONDITIONS[cond]
        print(f"\n[{args.task}/{cond}] N={args.episodes} ...", flush=True)
        r = run_episodes(policy, base, adapter, args.task, args.episodes, args.horizon,
                         t_fn, c_fn, v_fn, thermal, current, voltage, device=device,
                         gate_corners=args.gate_corners)
        results[cond] = r
        print(f"  -> SR={r['sr']:.0%}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for cond, r in results.items():
        print(f"  {args.task:<8} {cond:<14} SR={r['sr']:.0%}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "task": args.task,
            "ckpt": args.ckpt,
            "adapter_ckpt": args.adapter_ckpt,
            "variant": args.variant,
            "n_params": n_params,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "seed": args.seed,
            "gate_corners": args.gate_corners,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
