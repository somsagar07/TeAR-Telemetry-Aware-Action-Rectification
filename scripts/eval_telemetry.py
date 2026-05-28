#!/usr/bin/env python3
"""
Telemetry-Aware Evaluation: test a frozen BC-Transformer policy under
temperature, current, voltage, and combined degradation conditions.

For each telemetry channel, evaluates across three severity bands:
  Temperature : cool (20-42C) / warm (43-55C) / hot (56-75C)
  Current     : normal (0.1-0.5) / high (0.6-0.85) / near-stall (0.85-1.05)
  Voltage     : nominal (0.92-1.0) / undervolt (0.7-0.88) / brownout (0.45-0.68)

Also evaluates combined conditions (all channels stressed simultaneously).

Usage:
    python scripts/eval_telemetry.py \
        --ckpt manipulation_policies/BC_Transformer/20241028115532/models/model_epoch_100_Lift_success_1.0.pth \
        --episodes 30 --horizon 400
"""
import argparse
import json
import sys
import os
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Prevent tensorboard/tensorflow import crash in train_wrapper
import importlib, unittest.mock as _mock
def _make_mock_module(name):
    mod = _mock.MagicMock()
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__name__ = name
    mod.__path__ = []
    return mod
for _mod in ["tensorflow", "tensorflow.python", "tensorflow.python.framework",
             "tensorboard", "tensorboard.compat", "tensorboard.compat.tf",
             "torch.utils.tensorboard", "torch.utils.tensorboard.writer",
             "torch.utils.tensorboard._embedding"]:
    sys.modules.setdefault(_mod, _make_mock_module(_mod))

from frozen_base import FrozenBase
from train_wrapper import ThermalWrapperEnv
from env.telemetry_model import CurrentModel, VoltageModel

# Curve templates to test for each channel
THERMAL_CURVES = ["linear", "exponential", "sigmoid", "polynomial"]
CURRENT_CURVES = ["linear", "exponential", "sigmoid", "polynomial"]
VOLTAGE_CURVES = ["linear", "exponential", "sigmoid", "polynomial"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Condition definitions
# ═══════════════════════════════════════════════════════════════════════════════

TEMP_CONDITIONS = {
    "cool":  ("Cool  (20-42C)",   lambda n: np.random.uniform(20, 42, n).astype(np.float32)),
    "warm":  ("Warm  (43-55C)",   lambda n: np.random.uniform(43, 55, n).astype(np.float32)),
    "hot":   ("Hot   (56-75C)",   lambda n: np.random.uniform(56, 75, n).astype(np.float32)),
}

CURRENT_CONDITIONS = {
    "normal":     ("Normal  (0.1-0.5)",    lambda n: np.random.uniform(0.10, 0.50, n).astype(np.float32)),
    "high":       ("High    (0.6-0.85)",   lambda n: np.random.uniform(0.60, 0.85, n).astype(np.float32)),
    "near_stall": ("Stall   (0.85-1.05)",  lambda n: np.random.uniform(0.85, 1.05, n).astype(np.float32)),
}

VOLTAGE_CONDITIONS = {
    "nominal":    ("Nominal  (0.92-1.0)",  lambda n: np.random.uniform(0.92, 1.00, n).astype(np.float32)),
    "undervolt":  ("Undervolt(0.7-0.88)",  lambda n: np.random.uniform(0.70, 0.88, n).astype(np.float32)),
    "brownout":   ("Brownout (0.45-0.68)", lambda n: np.random.uniform(0.45, 0.68, n).astype(np.float32)),
}

COMBINED_CONDITIONS = {
    "all_normal": ("All Normal",
                   lambda n: np.random.uniform(20, 42, n).astype(np.float32),
                   lambda n: np.random.uniform(0.10, 0.50, n).astype(np.float32),
                   lambda n: np.random.uniform(0.92, 1.00, n).astype(np.float32)),
    "all_moderate": ("All Moderate",
                     lambda n: np.random.uniform(43, 55, n).astype(np.float32),
                     lambda n: np.random.uniform(0.60, 0.85, n).astype(np.float32),
                     lambda n: np.random.uniform(0.70, 0.88, n).astype(np.float32)),
    "all_severe": ("All Severe",
                   lambda n: np.random.uniform(56, 75, n).astype(np.float32),
                   lambda n: np.random.uniform(0.85, 1.05, n).astype(np.float32),
                   lambda n: np.random.uniform(0.45, 0.68, n).astype(np.float32)),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame buffer (same as used in existing evaluators)
# ═══════════════════════════════════════════════════════════════════════════════

class FrameBuffer:
    def __init__(self, T=10):
        self.T = T
        self.buf = deque(maxlen=T)

    def reset(self):
        self.buf.clear()

    def push(self, obs):
        self.buf.append({k: v.copy() for k, v in obs.items()})
        while len(self.buf) < self.T:
            self.buf.appendleft({k: v.copy() for k, v in self.buf[0].items()})

    def get(self):
        frames = list(self.buf)
        return {k: np.stack([f[k] for f in frames]) for k in frames[0]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation functions
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_episodes(base, env, n_episodes, horizon, temps_fn=None,
                 currents_fn=None, voltages_fn=None):
    """
    Run N episodes with fixed telemetry conditions.
    Whichever *_fn is None, that channel uses the env's default sampling.
    """
    fb = FrameBuffer(base.context_length)
    successes, returns, lengths = 0, [], []

    for ep in range(n_episodes):
        obs, info = env.reset()

        # Override telemetry if functions provided
        if temps_fn is not None:
            fixed = temps_fn(env.n_joints)
            env.joint_temps = fixed.copy()
            obs["joint_temps"] = fixed.copy()
        if currents_fn is not None and env.enable_current:
            fixed = currents_fn(env.n_joints)
            env.joint_currents = fixed.copy()
            if "joint_currents" in obs:
                obs["joint_currents"] = fixed.copy()
        if voltages_fn is not None and env.enable_voltage:
            fixed = voltages_fn(env.n_joints)
            env.joint_voltages = fixed.copy()
            if "joint_voltages" in obs:
                obs["joint_voltages"] = fixed.copy()

        fb.reset()
        fb.push(obs)
        ep_ret, success = 0.0, False

        for t in range(horizon):
            stacked = fb.get()
            a = base.act(stacked)
            obs, r, term, trunc, info = env.step(a)
            ep_ret += r
            fb.push(obs)
            if term or trunc:
                success = bool(info.get("task_success", False)) or term
                break

        successes += int(success)
        returns.append(ep_ret)
        lengths.append(t + 1)

    env.close()
    return {
        "sr": successes / n_episodes,
        "mean_rew": float(np.mean(returns)),
        "std_rew": float(np.std(returns)),
        "mean_len": float(np.mean(lengths)),
        "n": n_episodes,
    }


def make_env(enable_current=False, enable_voltage=False,
             current_curve="linear", voltage_curve="linear",
             horizon=400):
    """Create a ThermalWrapperEnv with specified telemetry channels."""
    cur_model = CurrentModel.from_predefined(current_curve) if enable_current else None
    vol_model = VoltageModel.from_predefined(voltage_curve) if enable_voltage else None
    return ThermalWrapperEnv(
        hot_joint_prob=0.0,  # we override temps manually
        horizon=horizon,
        terminate_on_success=True,
        enable_current=enable_current,
        enable_voltage=enable_voltage,
        current_model=cur_model,
        voltage_model=vol_model,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_temperature(base, n_episodes, horizon, curve_type="linear"):
    """Evaluate across temperature conditions with a given thermal curve."""
    print(f"\n{'='*70}")
    print(f"  TEMPERATURE degradation (curve: {curve_type})")
    print(f"{'='*70}")
    results = {}
    for cond_name, (label, temps_fn) in TEMP_CONDITIONS.items():
        env = make_env(horizon=horizon)
        r = run_episodes(base, env, n_episodes, horizon, temps_fn=temps_fn)
        results[cond_name] = r
        print(f"  {label}: SR={r['sr']:.0%}  rew={r['mean_rew']:7.1f}+/-{r['std_rew']:5.1f}  len={r['mean_len']:5.1f}")
    return results


def evaluate_current(base, n_episodes, horizon, curve_type="linear"):
    """Evaluate across current conditions (temps at cool baseline)."""
    print(f"\n{'='*70}")
    print(f"  CURRENT degradation (curve: {curve_type})")
    print(f"{'='*70}")
    results = {}
    cool_fn = lambda n: np.random.uniform(20, 42, n).astype(np.float32)
    cur_model = CurrentModel.from_predefined(curve_type)
    for cond_name, (label, currents_fn) in CURRENT_CONDITIONS.items():
        env = ThermalWrapperEnv(
            hot_joint_prob=0.0, horizon=horizon, terminate_on_success=True,
            enable_current=True, current_model=cur_model,
        )
        r = run_episodes(base, env, n_episodes, horizon,
                         temps_fn=cool_fn, currents_fn=currents_fn)
        results[cond_name] = r
        print(f"  {label}: SR={r['sr']:.0%}  rew={r['mean_rew']:7.1f}+/-{r['std_rew']:5.1f}  len={r['mean_len']:5.1f}")
    return results


def evaluate_voltage(base, n_episodes, horizon, curve_type="linear"):
    """Evaluate across voltage conditions (temps at cool baseline)."""
    print(f"\n{'='*70}")
    print(f"  VOLTAGE degradation (curve: {curve_type})")
    print(f"{'='*70}")
    results = {}
    cool_fn = lambda n: np.random.uniform(20, 42, n).astype(np.float32)
    vol_model = VoltageModel.from_predefined(curve_type)
    for cond_name, (label, voltages_fn) in VOLTAGE_CONDITIONS.items():
        env = ThermalWrapperEnv(
            hot_joint_prob=0.0, horizon=horizon, terminate_on_success=True,
            enable_voltage=True, voltage_model=vol_model,
        )
        r = run_episodes(base, env, n_episodes, horizon,
                         temps_fn=cool_fn, voltages_fn=voltages_fn)
        results[cond_name] = r
        print(f"  {label}: SR={r['sr']:.0%}  rew={r['mean_rew']:7.1f}+/-{r['std_rew']:5.1f}  len={r['mean_len']:5.1f}")
    return results


def evaluate_combined(base, n_episodes, horizon,
                      current_curve="linear", voltage_curve="linear"):
    """Evaluate with all three channels active simultaneously."""
    print(f"\n{'='*70}")
    print(f"  COMBINED degradation (current: {current_curve}, voltage: {voltage_curve})")
    print(f"{'='*70}")
    results = {}
    cur_model = CurrentModel.from_predefined(current_curve)
    vol_model = VoltageModel.from_predefined(voltage_curve)
    for cond_name, (label, temps_fn, currents_fn, voltages_fn) in COMBINED_CONDITIONS.items():
        env = ThermalWrapperEnv(
            hot_joint_prob=0.0, horizon=horizon, terminate_on_success=True,
            enable_current=True, enable_voltage=True,
            current_model=cur_model, voltage_model=vol_model,
        )
        r = run_episodes(base, env, n_episodes, horizon,
                         temps_fn=temps_fn, currents_fn=currents_fn,
                         voltages_fn=voltages_fn)
        results[cond_name] = r
        print(f"  {label}: SR={r['sr']:.0%}  rew={r['mean_rew']:7.1f}+/-{r['std_rew']:5.1f}  len={r['mean_len']:5.1f}")
    return results


def main():
    p = argparse.ArgumentParser(description="Evaluate BC-T under telemetry degradation")
    p.add_argument("--ckpt", required=True, help="Path to BC-Transformer .pth checkpoint")
    p.add_argument("--episodes", type=int, default=30, help="Episodes per condition")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--all-curves", action="store_true",
                   help="Test all curve templates (linear, exponential, sigmoid, polynomial)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.ckpt}")
    base = FrozenBase(args.ckpt, device=str(device))
    print(f"  is_transformer={base.is_transformer}  T={base.context_length}  "
          f"state_dim={base.state_dim}  act_dim={base.act_dim}")
    print(f"  Episodes per condition: {args.episodes}")
    print(f"  Horizon: {args.horizon}")

    all_results = {}
    curves_to_test = THERMAL_CURVES if args.all_curves else ["linear"]

    # ── Temperature ──
    for curve in curves_to_test:
        key = f"temperature_{curve}"
        all_results[key] = evaluate_temperature(base, args.episodes, args.horizon, curve)

    # ── Current ──
    for curve in (CURRENT_CURVES if args.all_curves else ["linear"]):
        key = f"current_{curve}"
        all_results[key] = evaluate_current(base, args.episodes, args.horizon, curve)

    # ── Voltage ──
    for curve in (VOLTAGE_CURVES if args.all_curves else ["linear"]):
        key = f"voltage_{curve}"
        all_results[key] = evaluate_voltage(base, args.episodes, args.horizon, curve)

    # ── Combined ──
    all_results["combined_linear"] = evaluate_combined(
        base, args.episodes, args.horizon, "linear", "linear")

    # ── Summary table ──
    print(f"\n{'='*90}")
    print("  TELEMETRY EVALUATION SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Channel':<15} {'Curve':<14} {'Condition':<18} {'SR':>6} {'Reward':>12}")
    print(f"  {'-'*70}")

    for key, conditions in all_results.items():
        parts = key.split("_", 1)
        channel = parts[0]
        curve = parts[1] if len(parts) > 1 else "linear"
        for cond_name, r in conditions.items():
            print(f"  {channel:<15} {curve:<14} {cond_name:<18} {r['sr']:>5.0%} "
                  f"{r['mean_rew']:>7.1f}+/-{r['std_rew']:4.1f}")

    print(f"{'='*90}")

    # ── Save results ──
    out_dir = Path(PROJECT_ROOT) / "results" / "telemetry_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"telemetry_results_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "seed": args.seed,
            "all_curves": args.all_curves,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved -> {out_file}")


if __name__ == "__main__":
    main()
