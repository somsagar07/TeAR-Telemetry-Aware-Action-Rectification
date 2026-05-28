#!/usr/bin/env python3
"""
Evaluate OpenVLA-OFT under telemetry degradation (temperature + current + voltage).

Wraps the standard OpenVLA-OFT eval with the same degradation physics used by
ThermalWrapperEnv, so results are directly comparable to the BC-T adapter evaluations.

Usage:
  python openvla/eval_openvla_oft_telemetry.py \
    --pretrained_checkpoint runs/openvla-oft-v1/<run_dir> \
    --episodes 10 --condition all_normal

Conditions: cool, warm, hot, current_normal, current_high, current_stall,
            voltage_nominal, voltage_undervolt, voltage_brownout,
            all_normal, all_moderate, all_severe
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import models directly to avoid gymnasium dependency in openvla-oft env
import importlib.util as _ilu

def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    import sys; sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_tm = _load_module("env.thermal_model",
                    os.path.join(PROJECT_ROOT, "env", "thermal_model.py"))
_tel = _load_module("env.telemetry_model",
                    os.path.join(PROJECT_ROOT, "env", "telemetry_model.py"))
ThermalModel = _tm.ThermalModel
CurrentModel = _tel.CurrentModel
VoltageModel = _tel.VoltageModel

# --- Condition definitions (same as scripts/eval_telemetry.py) ---

CONDITIONS = {
    # Temperature only (currents/voltages at nominal)
    "cool":     {"temps": (20, 42),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "warm":     {"temps": (43, 55),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "hot":      {"temps": (56, 75),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    # Current only (temps/voltages at nominal)
    "current_normal":  {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "current_high":    {"temps": (20, 42), "currents": (0.60, 0.85), "voltages": (0.92, 1.00)},
    "current_stall":   {"temps": (20, 42), "currents": (0.85, 1.05), "voltages": (0.92, 1.00)},
    # Voltage only (temps/currents at nominal)
    "voltage_nominal":   {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "voltage_undervolt": {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.70, 0.88)},
    "voltage_brownout":  {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.45, 0.68)},
    # Combined
    "all_normal":   {"temps": (20, 42),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "all_moderate": {"temps": (43, 55),   "currents": (0.60, 0.85), "voltages": (0.70, 0.88)},
    "all_severe":   {"temps": (56, 75),   "currents": (0.85, 1.05), "voltages": (0.45, 0.68)},
}


def main():
    # Import OpenVLA dependencies (only available in openvla-oft conda env)
    import robosuite as suite
    import torch

    OFT_REPO = "<DATA_PATH>"
    sys.path.insert(0, OFT_REPO)
    from experiments.robot.openvla_utils import (
        get_action_head, get_proprio_projector, get_processor, get_vla_action,
    )
    from experiments.robot.robot_utils import set_seed_everywhere
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    # Reuse EvalCfg and helpers from the standard eval script
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "openvla"))
    from eval_openvla_oft import EvalCfg, build_obs, load_model_and_components

    parser = argparse.ArgumentParser(description="OpenVLA-OFT telemetry evaluation")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--unnorm_key", default="robosuite_lift_ph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+",
                        default=["cool", "hot", "current_stall", "voltage_brownout",
                                 "all_normal", "all_moderate", "all_severe"],
                        help="Conditions to evaluate")
    parser.add_argument("--output_dir", default="outputs/openvla_oft_telemetry_eval")
    args = parser.parse_args()

    set_seed_everywhere(args.seed)
    cfg = EvalCfg(pretrained_checkpoint=args.pretrained_checkpoint, unnorm_key=args.unnorm_key)
    vla, processor, action_head, proprio_projector = load_model_and_components(cfg)

    # Degradation models
    thermal = ThermalModel.from_predefined("linear")
    current = CurrentModel.from_predefined("linear")
    voltage = VoltageModel.from_predefined("linear")
    n_joints = 7

    all_results = {}

    for cond_name in args.conditions:
        if cond_name not in CONDITIONS:
            print(f"  WARNING: unknown condition '{cond_name}', skipping")
            continue

        cond = CONDITIONS[cond_name]
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name}")
        print(f"  temps={cond['temps']}  currents={cond['currents']}  voltages={cond['voltages']}")
        print(f"{'='*60}")

        env = suite.make(
            "Lift", robots="Panda",
            has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, camera_names=["agentview"],
            camera_heights=256, camera_widths=256,
            reward_shaping=True, horizon=args.horizon, control_freq=20,
        )

        successes, rewards, lengths = [], [], []

        for ep in range(args.episodes):
            obs = env.reset()

            # Sample fixed telemetry for this episode
            temps = np.random.uniform(*cond["temps"], n_joints).astype(np.float32)
            currents = np.random.uniform(*cond["currents"], n_joints).astype(np.float32)
            voltages = np.random.uniform(*cond["voltages"], n_joints).astype(np.float32)

            current.reset()
            total_rew, success, ep_len = 0.0, False, 0
            action_queue = deque()

            for step in range(args.horizon):
                if not action_queue:
                    vla_obs = build_obs(obs)
                    actions = get_vla_action(
                        cfg=cfg, vla=vla, processor=processor,
                        obs=vla_obs, task_label="pick up the red cube",
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                    )
                    action_queue.extend(actions)

                action = np.array(action_queue.popleft(), dtype=np.float32)
                # Convert gripper convention
                action[6] = 1.0 - 2.0 * np.clip(action[6], 0, 1)
                action = np.clip(action, -1.0, 1.0)

                # Apply telemetry degradation (same physics as ThermalWrapperEnv)
                action = thermal.apply_thermal_physics(action, temps)
                action = current.apply_current_physics(action, currents)
                action = voltage.apply_voltage_physics(action, voltages)

                obs, reward, done, info = env.step(action)
                total_rew += reward
                ep_len += 1

                if hasattr(env, "_check_success") and env._check_success():
                    success = True
                if done:
                    break

            successes.append(success)
            rewards.append(total_rew)
            lengths.append(ep_len)
            tag = "SUCCESS" if success else "fail   "
            print(f"  ep {ep+1:2d}/{args.episodes}  {tag}  reward={total_rew:7.2f}  len={ep_len:3d}")

        env.close()

        sr = float(np.mean(successes))
        mr = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        ml = float(np.mean(lengths))

        all_results[cond_name] = {
            "sr": sr, "mean_rew": mr, "std_rew": std_r,
            "mean_len": ml, "n": args.episodes,
        }

        print(f"\n  {cond_name}: SR={sr:.0%}  rew={mr:.1f}+/-{std_r:.1f}  len={ml:.1f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  OPENVLA-OFT TELEMETRY EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':<22} {'SR':>6} {'Reward':>12} {'Length':>8}")
    print(f"  {'-'*52}")
    for cond_name, r in all_results.items():
        print(f"  {cond_name:<22} {r['sr']:>5.0%} {r['mean_rew']:>7.1f}+/-{r['std_rew']:4.1f} {r['mean_len']:>7.1f}")
    print(f"{'='*70}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"telemetry_results_{int(time.time())}.json"
    json.dump({
        "checkpoint": str(args.pretrained_checkpoint),
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed": args.seed,
        "results": all_results,
    }, open(out_path, "w"), indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
