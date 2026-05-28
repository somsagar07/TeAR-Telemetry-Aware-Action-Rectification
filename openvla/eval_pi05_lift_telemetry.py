#!/usr/bin/env python3
"""Eval pi05 (robotgeneralist/pi05_lerobot__robosuite_lift_ph_4) on robosuite Lift
under multi-channel telemetry degradation. Optionally apply TAM adapter.

Usage:
  python openvla/eval_pi05_lift_telemetry.py \\
      --pi05-ckpt external_policies/pi05_robosuite_lift \\
      --adapter-ckpt multi_task_runs/square/ablation_hp_sq_a030_gr075/sft_adapter.pt \\
      --episodes 20 --horizon 250 \\
      --out results/paper_v2/pi05_lift_n20.json
"""
import argparse, json, os, sys
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

PROJECT_ROOT = "."
sys.path.insert(0, PROJECT_ROOT)

# Mock TF/tensorboard for robosuite
import importlib, unittest.mock as _mock
def _mk(name):
    m = _mock.MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, None); m.__name__ = name; m.__path__ = []
    return m
for _m in ["tensorflow","tensorflow.python","tensorflow.python.framework",
          "tensorboard","tensorboard.compat","tensorboard.compat.tf",
          "torch.utils.tensorboard","torch.utils.tensorboard.writer",
          "torch.utils.tensorboard._embedding","mujoco_py"]:
    sys.modules.setdefault(_m, _mk(_m))

import robosuite as suite
try:
    import mimicgen, mimicgen.envs
except ImportError:
    pass

# Lerobot imports (lerobot_no_groot env)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
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


def make_env(horizon=250):
    return suite.make(
        "Lift", robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84, camera_widths=84,
        reward_shaping=False, ignore_done=True,
        horizon=horizon, control_freq=20,
    )


def build_pi05_obs(raw, device):
    """Convert robosuite obs dict to pi05's expected format."""
    # State: eef_pos(3) + eef_quat(4) + tool(2) — last is gripper_qpos
    eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float32)
    tool = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate([eef_pos, eef_quat, tool], axis=-1)  # (9,)
    assert state.shape == (9,), f"state shape: {state.shape}"

    # Images: 84x84 RGB. robosuite returns HWC uint8; pi05 wants CHW float32 in [0,1].
    def to_chw(img):
        # robosuite image is upside-down — flip
        img = img[::-1].copy()  # flip vertical
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img
    base_img = to_chw(raw["agentview_image"])
    wrist_img = to_chw(raw["robot0_eye_in_hand_image"])

    # Pi05 expects observation.images.empty_camera_0 too (zero-fill).
    empty = np.zeros((3, 84, 84), dtype=np.float32)

    return {
        "observation.state": torch.as_tensor(state, dtype=torch.float32).to(device),
        "observation.images.base_0_rgb": torch.as_tensor(base_img, dtype=torch.float32).to(device),
        "observation.images.right_wrist_0_rgb": torch.as_tensor(wrist_img, dtype=torch.float32).to(device),
        "observation.images.empty_camera_0": torch.as_tensor(empty, dtype=torch.float32).to(device),
        "task": "lift the red cube",
    }


def run_episode(policy, preprocessor, postprocessor, env, n_joints, t_fn, c_fn, v_fn,
                thermal, current_m, voltage_m, adapter, tam_adapter_state_dim, device, horizon):
    raw = env.reset()
    T = t_fn(n_joints); C = c_fn(n_joints); V = v_fn(n_joints)
    if hasattr(policy, '_action_queue'):
        policy._action_queue.clear()
    success = False
    for step in range(horizon):
        obs = build_pi05_obs(raw, device)
        processed = preprocessor(obs)
        with torch.no_grad():
            action_t = policy.select_action(processed)
        if postprocessor is not None:
            out = postprocessor({"action": action_t.cpu() if action_t.is_cuda else action_t})
            action_t = out["action"] if isinstance(out, dict) else out
        else:
            pass
        a = action_t.cpu().numpy().reshape(-1)[:7]
        a = a.astype(np.float32).clip(-1, 1)
        # TAM correction
        if adapter is not None:
            state_vec = np.zeros(tam_adapter_state_dim, dtype=np.float32)
            # eef_pos(3) + eef_quat(4) + gripper_qpos(2) + object(?) — pad to expected
            base_state = np.concatenate([
                np.asarray(raw['robot0_eef_pos']).flatten(),
                np.asarray(raw['robot0_eef_quat']).flatten(),
                np.asarray(raw['robot0_gripper_qpos']).flatten(),
                np.asarray(raw.get('object-state', np.zeros(10))).flatten(),
            ]).astype(np.float32)
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
        # Apply degradation
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
    ap.add_argument("--pi05-ckpt", required=True)
    ap.add_argument("--adapter-ckpt", default=None)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gamma-range", type=float, default=0.75)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load pi05] {args.pi05_ckpt}", flush=True)
    policy = PI05Policy.from_pretrained(args.pi05_ckpt).to(device).eval()
    print(f"  Loaded. params: {sum(p.numel() for p in policy.parameters())/1e9:.2f}B", flush=True)

    # Load preprocessor / postprocessor
    preprocessor = DataProcessorPipeline.from_pretrained(
        args.pi05_ckpt,
        config_filename="policy_preprocessor.json",
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.pi05_ckpt,
        config_filename="policy_postprocessor.json",
    )
    print(f"  Preprocessor steps: {len(preprocessor.steps)}, Postprocessor steps: {len(postprocessor.steps)}", flush=True)

    adapter = None
    tam_state_dim = 19
    if args.adapter_ckpt:
        sd = torch.load(args.adapter_ckpt, map_location='cpu', weights_only=False)
        if 'state_proj.weight' in sd:
            tam_state_dim = sd['state_proj.weight'].shape[1]
        adapter = TAMBoT(
            state_dim=tam_state_dim, act_dim=7,
            hidden=128, n_layers=3, n_heads=4,
            alpha=args.alpha, gamma_range=args.gamma_range, gate_shape="smoothstep",
        ).to(device)
        adapter.load_state_dict(sd)
        adapter.eval()
        print(f"[load adapter] {args.adapter_ckpt} (state_dim={tam_state_dim})", flush=True)

    thermal = ThermalModel.from_predefined("linear", n_joints=7)
    current_m = CurrentModel.from_predefined("linear", n_joints=7)
    voltage_m = VoltageModel.from_predefined("linear", n_joints=7)

    env = make_env(args.horizon)
    results = {}
    for cond in args.conditions:
        if cond not in CONDITIONS:
            print(f"  WARN: unknown condition {cond}"); continue
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        t_fn, c_fn, v_fn = CONDITIONS[cond]
        successes = 0
        import time
        t0 = time.time()
        for ep in range(args.episodes):
            s = run_episode(policy, preprocessor, postprocessor, env, 7,
                            t_fn, c_fn, v_fn, thermal, current_m, voltage_m,
                            adapter, tam_state_dim, device, args.horizon)
            successes += s
        sr = successes / args.episodes
        print(f"  {cond}: sr={sr:.0%}  ({time.time()-t0:.0f}s)", flush=True)
        results[cond] = {"sr": sr, "n": args.episodes}
    env.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": "pi05",
        "task": "robosuite Lift",
        "ckpt": args.pi05_ckpt,
        "adapter_ckpt": args.adapter_ckpt,
        "alpha": args.alpha, "gamma_range": args.gamma_range,
        "episodes": args.episodes, "horizon": args.horizon, "seed": args.seed,
        "results": results,
    }, open(args.out, 'w'), indent=2)
    print(f"Saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
