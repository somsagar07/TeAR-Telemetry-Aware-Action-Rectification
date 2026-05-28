#!/usr/bin/env python3
"""Eval ACT (Action Chunking Transformer) policy on a robosuite task under
multi-channel telemetry degradation, with optional TAM adapter.

Usage:
  python openvla/eval_act_telemetry.py \\
      --act-ckpt external_policies/act_finetuned/lift/checkpoints/last/pretrained_model \\
      --task Lift --horizon 250 \\
      --adapter-ckpt multi_task_runs/square/.../sft_adapter.pt \\
      --episodes 20 \\
      --out results/paper_v2/act_lift_n20.json
"""
import argparse, json, os, sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

PROJECT_ROOT = "."
sys.path.insert(0, PROJECT_ROOT)

import robosuite as suite
try:
    import mimicgen, mimicgen.envs
except ImportError:
    pass

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import DataProcessorPipeline

from env.thermal_model import ThermalModel
from env.telemetry_model import CurrentModel, VoltageModel
from thermal_adapters.tam_bot import TAMBoT


def _const(val):
    return lambda n: np.full(n, float(val), dtype=np.float32)

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
    "T_mod":      (_const(58.0), _const(0.30), _const(1.00)),
    "TC_mod":     (_const(55.0), _const(0.75), _const(1.00)),
    "TV_mod":     (_const(55.0), _const(0.30), _const(0.72)),
    "TCV_mod":    (_const(55.0), _const(0.75), _const(0.72)),
}


TASK_CONFIG = {
    "Lift":      {"env_name": "Lift",            "horizon": 250},
    "Can":       {"env_name": "PickPlaceCan",    "horizon": 400},
    "Square":    {"env_name": "NutAssemblySquare","horizon": 400},
    "Threading": {"env_name": "Threading_D0",    "horizon": 500},
}


def make_env(task, horizon):
    env_name = TASK_CONFIG[task]["env_name"]
    try:
        from robosuite.controllers import load_controller_config
        controller_cfg = load_controller_config(default_controller="OSC_POSE")
    except Exception:
        controller_cfg = None
    kwargs = dict(
        robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84, camera_widths=84,
        reward_shaping=False, ignore_done=True,
        horizon=horizon, control_freq=20,
    )
    if controller_cfg is not None:
        kwargs["controller_configs"] = controller_cfg
    return suite.make(env_name, **kwargs)


def build_act_obs(raw, device):
    """Convert robosuite obs dict to ACT's expected format.
    State: eef_pos(3) + eef_quat(4) + tool(2) = 9-dim
    Images: 84x84 RGB CHW float32 [0,1]
    """
    eef_pos  = np.asarray(raw["robot0_eef_pos"],   dtype=np.float32)
    eef_quat = np.asarray(raw["robot0_eef_quat"],  dtype=np.float32)
    tool     = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate([eef_pos, eef_quat, tool], axis=-1)
    assert state.shape == (9,)

    def to_chw(img):
        img = img[::-1].copy()  # robosuite is upside-down
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    base_img  = to_chw(raw["agentview_image"])
    wrist_img = to_chw(raw["robot0_eye_in_hand_image"])

    return {
        "observation.state":                    torch.as_tensor(state,    dtype=torch.float32).unsqueeze(0).to(device),
        "observation.images.base_0_rgb":        torch.as_tensor(base_img, dtype=torch.float32).unsqueeze(0).to(device),
        "observation.images.right_wrist_0_rgb": torch.as_tensor(wrist_img,dtype=torch.float32).unsqueeze(0).to(device),
        "task": ["do the task"],
    }


def run_episode(policy, preprocessor, postprocessor, env, n_joints, t_fn, c_fn, v_fn,
                thermal, current_m, voltage_m, adapter, tam_state_dim, device, horizon):
    raw = env.reset()
    T = t_fn(n_joints); C = c_fn(n_joints); V = v_fn(n_joints)
    if hasattr(policy, 'reset'):
        policy.reset()
    success = False
    for step in range(horizon):
        obs = build_act_obs(raw, device)
        if preprocessor is not None:
            obs = preprocessor(obs)
        with torch.no_grad():
            action_t = policy.select_action(obs)
        if postprocessor is not None:
            out = postprocessor({"action": action_t})
            action_t = out["action"] if isinstance(out, dict) else out
        a = action_t.cpu().numpy().reshape(-1)[:7].astype(np.float32).clip(-1, 1)

        if adapter is not None:
            base_state = np.concatenate([
                np.asarray(raw['robot0_eef_pos']).flatten(),
                np.asarray(raw['robot0_eef_quat']).flatten(),
                np.asarray(raw['robot0_gripper_qpos']).flatten(),
                np.asarray(raw.get('object-state', np.zeros(10))).flatten(),
            ]).astype(np.float32)
            state_vec = np.zeros(tam_state_dim, dtype=np.float32)
            n = min(state_vec.shape[0], base_state.shape[0])
            state_vec[:n] = base_state[:n]
            with torch.no_grad():
                a_t = torch.as_tensor(a, device=device)[None]
                T_t = torch.as_tensor(T, device=device)[None]
                C_t = torch.as_tensor(C, device=device)[None]
                V_t = torch.as_tensor(V, device=device)[None]
                s_t = torch.as_tensor(state_vec, device=device)[None]
                corrected, _, _ = adapter(a_t, T_t, s_t, C_t, V_t)
            a = corrected.squeeze(0).cpu().numpy()

        a = thermal.apply_thermal_physics(a, T)
        a = current_m.apply_current_physics(a, C)
        a = voltage_m.apply_voltage_physics(a, V)
        raw, _, _, _ = env.step(a)
        if env._check_success():
            success = True
            break
    return int(success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-ckpt", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_CONFIG))
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--adapter-ckpt", default=None)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gamma-range", type=float, default=0.75)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizon = args.horizon or TASK_CONFIG[args.task]["horizon"]

    print(f"[load act] {args.act_ckpt}", flush=True)
    policy = ACTPolicy.from_pretrained(args.act_ckpt).to(device).eval()
    print(f"  Loaded. params: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M", flush=True)

    try:
        preprocessor = DataProcessorPipeline.from_pretrained(
            args.act_ckpt, config_filename="policy_preprocessor.json")
    except Exception as e:
        print(f"  no preprocessor: {e}")
        preprocessor = None
    try:
        postprocessor = DataProcessorPipeline.from_pretrained(
            args.act_ckpt, config_filename="policy_postprocessor.json")
    except Exception as e:
        print(f"  no postprocessor: {e}")
        postprocessor = None

    adapter = None
    tam_state_dim = 19
    if args.adapter_ckpt:
        sd = torch.load(args.adapter_ckpt, map_location='cpu', weights_only=False)
        if 'state_proj.weight' in sd:
            tam_state_dim = sd['state_proj.weight'].shape[1]
        # Detect hidden size and n_layers from ckpt
        hidden = sd['joint_proj.weight'].shape[0] if 'joint_proj.weight' in sd else 128
        n_layers = sum(1 for k in sd if k.startswith('transformer.layers.') and k.endswith('.norm1.weight'))
        if n_layers == 0:
            n_layers = 3
        print(f"  detected hidden={hidden}, n_layers={n_layers}, state_dim={tam_state_dim}", flush=True)
        adapter = TAMBoT(
            state_dim=tam_state_dim, act_dim=7,
            hidden=hidden, n_layers=n_layers, n_heads=4,
            alpha=args.alpha, gamma_range=args.gamma_range, gate_shape="smoothstep",
        ).to(device)
        adapter.load_state_dict(sd); adapter.eval()
        print(f"[load adapter] {args.adapter_ckpt}", flush=True)

    thermal   = ThermalModel.from_predefined("linear", n_joints=7)
    current_m = CurrentModel.from_predefined("linear", n_joints=7)
    voltage_m = VoltageModel.from_predefined("linear", n_joints=7)

    env = make_env(args.task, horizon)
    results = {}
    import time
    for cond in args.conditions:
        if cond not in CONDITIONS:
            print(f"  WARN: unknown condition {cond}"); continue
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        t_fn, c_fn, v_fn = CONDITIONS[cond]
        successes = 0
        t0 = time.time()
        for ep in range(args.episodes):
            s = run_episode(policy, preprocessor, postprocessor, env, 7,
                            t_fn, c_fn, v_fn, thermal, current_m, voltage_m,
                            adapter, tam_state_dim, device, horizon)
            successes += s
        sr = successes / args.episodes
        print(f"  {cond}: sr={sr:.0%}  ({time.time()-t0:.0f}s)", flush=True)
        results[cond] = {"sr": sr, "n": args.episodes}
    env.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": "act",
        "task": args.task,
        "ckpt": args.act_ckpt,
        "adapter_ckpt": args.adapter_ckpt,
        "alpha": args.alpha, "gamma_range": args.gamma_range,
        "episodes": args.episodes, "horizon": horizon, "seed": args.seed,
        "results": results,
    }, open(args.out, 'w'), indent=2)
    print(f"Saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
