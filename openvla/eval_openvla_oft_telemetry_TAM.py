#!/usr/bin/env python3
"""OpenVLA-OFT + TAM-BoT-Large evaluation under telemetry degradation.

Wraps OpenVLA-OFT's chunked action prediction with the TAM-BoT adapter.
At each env step:
  1. OpenVLA predicts action (or pops from chunk queue)
  2. **TAM adapter corrects the action given (T, C, V, state)**
  3. Degradation physics applied
  4. env.step

Usage:
  conda run -n openvla-oft python openvla/eval_openvla_oft_telemetry_TAM.py \\
      --pretrained_checkpoint runs/openvla-oft-v1/<run_dir> \\
      --tam_adapter multi_task_runs/cross_base/lift_baseline/sft_adapter.pt \\
      --episodes 10
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import models directly to avoid robomimic load fight
import importlib.util as _ilu
def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_tm = _load_module("env.thermal_model", os.path.join(PROJECT_ROOT, "env", "thermal_model.py"))
_tel = _load_module("env.telemetry_model", os.path.join(PROJECT_ROOT, "env", "telemetry_model.py"))
ThermalModel = _tm.ThermalModel
CurrentModel = _tel.CurrentModel
VoltageModel = _tel.VoltageModel


CONDITIONS = {
    "cool":     {"temps": (20, 42),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "warm":     {"temps": (43, 55),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "hot":      {"temps": (56, 75),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "current_normal":  {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "current_high":    {"temps": (20, 42), "currents": (0.60, 0.85), "voltages": (0.92, 1.00)},
    "current_stall":   {"temps": (20, 42), "currents": (0.85, 1.05), "voltages": (0.92, 1.00)},
    "voltage_nominal":   {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "voltage_undervolt": {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.70, 0.88)},
    "voltage_brownout":  {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.45, 0.68)},
    "all_normal":   {"temps": (20, 42),   "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "all_moderate": {"temps": (43, 55),   "currents": (0.60, 0.85), "voltages": (0.70, 0.88)},
    "all_severe":   {"temps": (56, 75),   "currents": (0.85, 1.05), "voltages": (0.45, 0.68)},
    # --- Table 1 short-name aliases + multi-channel moderate combinations ---
    "stall":   {"temps": (20, 42), "currents": (0.85, 1.05), "voltages": (0.92, 1.00)},
    "brown":   {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.45, 0.68)},
    "T_mod":   {"temps": (43, 55), "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "TC_mod":  {"temps": (43, 55), "currents": (0.60, 0.85), "voltages": (0.92, 1.00)},
    "TV_mod":  {"temps": (43, 55), "currents": (0.10, 0.50), "voltages": (0.70, 0.88)},
    "TCV_mod": {"temps": (43, 55), "currents": (0.60, 0.85), "voltages": (0.70, 0.88)},
}


def build_state_vec(obs, state_keys=("robot0_eef_pos", "robot0_eef_quat",
                                      "robot0_gripper_qpos", "object")):
    """Reconstruct the 19-dim state vector TAM-BoT expects.
    Lift: 3 + 4 + 2 + 10 = 19."""
    parts = []
    for k in state_keys:
        if k in obs:
            v = np.asarray(obs[k], dtype=np.float32).reshape(-1)
            parts.append(v)
        elif k == "object" and "object-state" in obs:
            parts.append(np.asarray(obs["object-state"], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def main():
    import robosuite as suite
    import torch

    OFT_REPO = "<DATA_PATH>"
    sys.path.insert(0, OFT_REPO)
    from experiments.robot.openvla_utils import (
        get_action_head, get_proprio_projector, get_processor, get_vla_action,
    )
    from experiments.robot.robot_utils import set_seed_everywhere
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "openvla"))
    from eval_openvla_oft import EvalCfg, build_obs, load_model_and_components

    # Load TAM-BoT adapter (with project's env, not openvla-oft)
    sys.path.insert(0, PROJECT_ROOT)
    from thermal_adapters.tam_bot import TAMBoT

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--tam_adapter", default=None,
                        help="Path to TAM-BoT adapter .pt (omit to skip TAM and run baseline)")
    parser.add_argument("--tam_hidden", type=int, default=128)
    parser.add_argument("--tam_n_layers", type=int, default=3)
    parser.add_argument("--tam_n_heads", type=int, default=4)
    parser.add_argument("--tam_alpha", type=float, default=0.3)
    parser.add_argument("--tam_gamma_range", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--unnorm_key", default="robosuite_lift_ph")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+",
                        default=["cool", "hot", "all_moderate", "all_severe"])
    parser.add_argument("--output_dir", default="outputs/openvla_oft_telemetry_TAM_eval")
    args = parser.parse_args()

    set_seed_everywhere(args.seed)
    cfg = EvalCfg(pretrained_checkpoint=args.pretrained_checkpoint, unnorm_key=args.unnorm_key)
    vla, processor, action_head, proprio_projector = load_model_and_components(cfg)

    # Load TAM
    tam_adapter = None
    state_dim = 19  # Lift task default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.tam_adapter:
        print(f"[TAM] Loading from {args.tam_adapter}")
        # Inspect ckpt state_dim
        sd = torch.load(args.tam_adapter, map_location='cpu', weights_only=False)
        if 'state_proj.weight' in sd:
            state_dim = sd['state_proj.weight'].shape[1]
        print(f"[TAM] state_dim={state_dim} hidden={args.tam_hidden}")
        tam_adapter = TAMBoT(
            state_dim=state_dim, act_dim=7,
            hidden=args.tam_hidden, n_layers=args.tam_n_layers,
            n_heads=args.tam_n_heads, alpha=args.tam_alpha,
            gamma_range=args.tam_gamma_range, gate_shape="smoothstep",
        ).to(device)
        tam_adapter.load_state_dict(sd)
        tam_adapter.eval()
        print(f"[TAM] {sum(p.numel() for p in tam_adapter.parameters()):,} params loaded.")

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
        print(f"\n{'='*60}\n  Condition: {cond_name}\n{'='*60}")
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
                action[6] = 1.0 - 2.0 * np.clip(action[6], 0, 1)
                action = np.clip(action, -1.0, 1.0)

                # === TAM correction (the new bit) ===
                if tam_adapter is not None:
                    state_vec = build_state_vec(obs)
                    if state_vec.shape[0] != state_dim:
                        # Pad/truncate to match
                        if state_vec.shape[0] < state_dim:
                            state_vec = np.concatenate([state_vec, np.zeros(state_dim - state_vec.shape[0], dtype=np.float32)])
                        else:
                            state_vec = state_vec[:state_dim]
                    with torch.no_grad():
                        a_t = torch.as_tensor(action, dtype=torch.float32, device=device)[None]
                        T_t = torch.as_tensor(temps, dtype=torch.float32, device=device)[None]
                        C_t = torch.as_tensor(currents, dtype=torch.float32, device=device)[None]
                        V_t = torch.as_tensor(voltages, dtype=torch.float32, device=device)[None]
                        s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device)[None]
                        corrected, _, _ = tam_adapter(a_t, T_t, s_t, C_t, V_t)
                        action = corrected.squeeze(0).cpu().numpy()

                # Degradation physics
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
            print(f"  ep {ep+1:2d}/{args.episodes}  {tag}  rew={total_rew:6.2f}  len={ep_len:3d}")

        env.close()
        all_results[cond_name] = {
            "sr": float(np.mean(successes)),
            "mean_rew": float(np.mean(rewards)),
            "std_rew": float(np.std(rewards)),
            "mean_len": float(np.mean(lengths)),
            "n": args.episodes,
        }
        print(f"\n  {cond_name}: SR={all_results[cond_name]['sr']:.0%}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"openvla_oft_tam_telemetry_{int(time.time())}.json")
    json.dump({
        "checkpoint": args.pretrained_checkpoint,
        "tam_adapter": args.tam_adapter,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed": args.seed,
        "results": all_results,
    }, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
