#!/usr/bin/env python3
"""Eval GR00T-N1.5 (Libero-style data config) on robosuite Lift under telemetry degradation.

Uses Gr00tPolicy from Isaac-GR00T. The Libero data config expects:
- state.x, y, z (eef position, 3)
- state.roll, pitch, yaw (Euler from eef_quat, 3)
- state.gripper (2)  -- 8D total
- action.x..yaw, gripper (7D)
- video.image, video.wrist_image (84x84 RGB)

We convert robosuite obs accordingly.
"""
import argparse, json, os, sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PROJECT_ROOT = "."
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, "<DATA_PATH>")
sys.path.insert(0, "<DATA_PATH>")

import importlib, unittest.mock as _mock
def _mk(name):
    m = _mock.MagicMock()
    m.__spec__ = importlib.machinery.ModuleSpec(name, None); m.__name__ = name; m.__path__ = []
    return m
for _m in ["tensorflow.python.framework",
          "tensorboard","tensorboard.compat","tensorboard.compat.tf",
          "torch.utils.tensorboard","torch.utils.tensorboard.writer",
          "torch.utils.tensorboard._embedding","mujoco_py"]:
    sys.modules.setdefault(_m, _mk(_m))

import robosuite as suite

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import EmbodimentTag
from Libero.custom_data_config import LiberoDataConfig

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


def build_gr00t_obs(raw):
    """Convert robosuite obs to GR00T Libero-style obs format."""
    eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float32)  # [w,x,y,z]
    tool = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
    # Convert quat (wxyz) → euler (xyz)
    r = Rotation.from_quat([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
    rpy = r.as_euler('xyz').astype(np.float32)

    # Images: HWC uint8 -> HWC uint8 (gr00t expects), flip vertically
    img = raw["agentview_image"][::-1].copy()  # vertical flip
    wrist = raw["robot0_eye_in_hand_image"][::-1].copy()

    obs = {
        "state.x": np.array([[eef_pos[0]]], dtype=np.float32),       # (1, 1)
        "state.y": np.array([[eef_pos[1]]], dtype=np.float32),
        "state.z": np.array([[eef_pos[2]]], dtype=np.float32),
        "state.roll": np.array([[rpy[0]]], dtype=np.float32),
        "state.pitch": np.array([[rpy[1]]], dtype=np.float32),
        "state.yaw": np.array([[rpy[2]]], dtype=np.float32),
        "state.gripper": np.array([[tool[0], tool[1]]], dtype=np.float32),  # (1, 2)
        "video.image": img[None],          # (1, H, W, C)
        "video.wrist_image": wrist[None],
        "annotation.human.action.task_description": ["lift the red cube"],
    }
    return obs


def make_env(horizon=250):
    return suite.make(
        "Lift", robots="Panda",

        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84, camera_widths=84,
        reward_shaping=False, ignore_done=True,
        horizon=horizon, control_freq=20,
    )


def run_episode(policy, env, n_joints, t_fn, c_fn, v_fn,
                thermal, current_m, voltage_m, adapter, tam_state_dim, device, horizon):
    raw = env.reset()
    T = t_fn(n_joints); C = c_fn(n_joints); V = v_fn(n_joints)
    success = False
    action_chunk = None
    chunk_idx = 0
    for step in range(horizon):
        if action_chunk is None or chunk_idx >= len(action_chunk):
            obs = build_gr00t_obs(raw)
            actions = policy.get_action(obs)
            # Unpack: action.x,y,z,roll,pitch,yaw,gripper each shape (T, 1)
            T_chunk = actions["action.x"].shape[0]
            action_chunk = np.zeros((T_chunk, 7), dtype=np.float32)
            for i, k in enumerate(["action.x", "action.y", "action.z",
                                    "action.roll", "action.pitch", "action.yaw", "action.gripper"]):
                action_chunk[:, i] = actions[k].squeeze(-1) if actions[k].ndim > 1 else actions[k]
            chunk_idx = 0
        a = action_chunk[chunk_idx]
        chunk_idx += 1
        a = a.clip(-1, 1)

        if adapter is not None:
            # state for TAM: pad to expected
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
    ap.add_argument("--gr00t-ckpt", required=True)
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

    print(f"[load gr00t] {args.gr00t_ckpt}", flush=True)
    cfg = LiberoDataConfig()
    policy = Gr00tPolicy(
        model_path=args.gr00t_ckpt,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        denoising_steps=4,
        device=device,
    )
    print(f"  Loaded", flush=True)

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
        print(f"[load adapter] {args.adapter_ckpt}", flush=True)

    thermal = ThermalModel.from_predefined("linear", n_joints=7)
    current_m = CurrentModel.from_predefined("linear", n_joints=7)
    voltage_m = VoltageModel.from_predefined("linear", n_joints=7)

    env = make_env(args.horizon)
    results = {}
    for cond in args.conditions:
        if cond not in CONDITIONS: continue
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        t_fn, c_fn, v_fn = CONDITIONS[cond]
        successes = 0
        import time
        t0 = time.time()
        for ep in range(args.episodes):
            s = run_episode(policy, env, 7, t_fn, c_fn, v_fn, thermal, current_m, voltage_m,
                           adapter, tam_state_dim, device, args.horizon)
            successes += s
        sr = successes / args.episodes
        print(f"  {cond}: sr={sr:.0%}  ({time.time()-t0:.0f}s)", flush=True)
        results[cond] = {"sr": sr, "n": args.episodes}
    env.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": "gr00t_n1.5",
        "task": "robosuite Lift",
        "ckpt": args.gr00t_ckpt,
        "adapter_ckpt": args.adapter_ckpt,
        "episodes": args.episodes, "horizon": args.horizon, "seed": args.seed,
        "results": results,
    }, open(args.out, 'w'), indent=2)
    print(f"Saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
