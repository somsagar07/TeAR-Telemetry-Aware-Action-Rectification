#!/usr/bin/env python3
"""Train TAM-BoT with OpenVLA-OFT as the BASE policy (PPO fine-tune).

Differs from train_sft_rl_kl_bct.py:
  - Base policy is OpenVLA-OFT (7B VLA), not robomimic BC-T
  - Rollouts use OpenVLA's chunked action prediction (8-step chunks)
  - TAM corrects each OpenVLA-predicted action before degradation+env.step
  - PPO updates only the TAM adapter (OpenVLA frozen)
  - Cool-condition identity preserved via gated-noise PPO (γ=1, δ=0 at gate=0)

Usage (in openvla-oft conda env):
  conda run -n openvla-oft python openvla/train_tam_on_openvla.py \\
      --pretrained_checkpoint runs/openvla-oft-v1/<run_dir> \\
      --tam_init multi_task_runs/cross_base/lift_baseline/sft_adapter.pt \\
      --rl_steps 50000 --rl_n_steps 1024 --horizon 200 \\
      --p_hot 0.7 --kl_coef 0.05 \\
      --output_dir multi_task_runs/lift/tam_on_openvla
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
    "cool":     {"temps": (20, 42), "currents": (0.10, 0.50), "voltages": (0.92, 1.00)},
    "warm":     {"temps": (43, 55), "currents": (0.50, 0.75), "voltages": (0.80, 0.92)},
    "mod":      {"temps": (50, 65), "currents": (0.65, 0.90), "voltages": (0.65, 0.85)},
    "hot":      {"temps": (60, 75), "currents": (0.80, 1.05), "voltages": (0.45, 0.70)},
}
SEV_NAMES = list(CONDITIONS.keys())
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")


def build_state(obs):
    parts = []
    for k in STATE_KEYS:
        if k in obs:
            parts.append(np.asarray(obs[k], dtype=np.float32).reshape(-1))
        elif k == "object" and "object-state" in obs:
            parts.append(np.asarray(obs["object-state"], dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def sample_telemetry(severity, n_joints=7):
    """Sample (T, C, V) for one episode based on severity bucket."""
    cond = CONDITIONS[severity]
    T = np.random.uniform(*cond["temps"], n_joints).astype(np.float32)
    C = np.random.uniform(*cond["currents"], n_joints).astype(np.float32)
    V = np.random.uniform(*cond["voltages"], n_joints).astype(np.float32)
    return T, C, V


def main():
    import robosuite as suite
    import torch

    OFT_REPO = "<DATA_PATH>"
    sys.path.insert(0, OFT_REPO)
    from experiments.robot.openvla_utils import (
        get_action_head, get_proprio_projector, get_processor, get_vla_action,
    )
    from experiments.robot.robot_utils import set_seed_everywhere

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "openvla"))
    from eval_openvla_oft import EvalCfg, build_obs, load_model_and_components

    sys.path.insert(0, PROJECT_ROOT)
    from thermal_adapters.tam_bot import TAMBoT

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", required=True)
    ap.add_argument("--tam_init", required=True,
                    help="TAM-BoT adapter to initialize from (lift_baseline cross-base ckpt)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--rl_steps", type=int, default=50000)
    ap.add_argument("--rl_n_steps", type=int, default=512)
    ap.add_argument("--rl_epochs", type=int, default=4)
    ap.add_argument("--rl_lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--log_std_init", type=float, default=-1.5)
    ap.add_argument("--p_hot", type=float, default=0.7)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unnorm_key", default="robosuite_lift_ph")
    ap.add_argument("--gamma_range", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--save_every", type=int, default=10000)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[load] OpenVLA-OFT: {args.pretrained_checkpoint}", flush=True)
    cfg = EvalCfg(pretrained_checkpoint=args.pretrained_checkpoint, unnorm_key=args.unnorm_key)
    vla, processor, action_head, proprio_projector = load_model_and_components(cfg)

    print(f"[load] TAM-BoT init: {args.tam_init}", flush=True)
    sd = torch.load(args.tam_init, map_location='cpu', weights_only=False)
    state_dim = sd['state_proj.weight'].shape[1] if 'state_proj.weight' in sd else 19
    adapter = TAMBoT(
        state_dim=state_dim, act_dim=7,
        hidden=128, n_layers=3, n_heads=4,
        alpha=args.alpha, gamma_range=args.gamma_range,
        log_std_init=args.log_std_init, gate_shape="smoothstep",
    ).to(device)
    adapter.load_state_dict(sd)
    print(f"  adapter params: {sum(p.numel() for p in adapter.parameters()):,}", flush=True)

    # SFT snapshot for KL
    sft_snapshot = TAMBoT(
        state_dim=state_dim, act_dim=7,
        hidden=128, n_layers=3, n_heads=4,
        alpha=args.alpha, gamma_range=args.gamma_range,
        log_std_init=args.log_std_init, gate_shape="smoothstep",
    ).to(device)
    sft_snapshot.load_state_dict(sd)
    sft_snapshot.eval()
    for p in sft_snapshot.parameters():
        p.requires_grad = False

    thermal = ThermalModel.from_predefined("linear")
    current = CurrentModel.from_predefined("linear")
    voltage = VoltageModel.from_predefined("linear")
    n_joints = 7

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.rl_lr, weight_decay=1e-4)

    # Sample severity for each episode according to p_hot
    def sample_severity():
        if np.random.rand() < args.p_hot:
            return np.random.choice(["warm", "mod", "hot"])
        return "cool"

    # ─── Single-env PPO loop ───
    print(f"\n=== PPO RL on top of OpenVLA-OFT ===", flush=True)
    print(f"  rl_steps={args.rl_steps:,}  n_steps_per_rollout={args.rl_n_steps}", flush=True)
    print(f"  KL={args.kl_coef}  lr={args.rl_lr}  p_hot={args.p_hot}", flush=True)

    env = suite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview"],
        camera_heights=256, camera_widths=256,
        reward_shaping=True, horizon=args.horizon, control_freq=20,
    )

    total_steps = 0
    rollout_idx = 0
    best_score = -1e9
    t0 = time.time()

    while total_steps < args.rl_steps:
        rollout_idx += 1
        # Buffers
        S, AB, T_arr, C_arr, V_arr = [], [], [], [], []  # state, openvla action, telemetry
        LOGIT_TAKEN, LOG_PROBS, VALUES, REWARDS, DONES = [], [], [], [], []

        rollout_size = 0
        ep_rewards = []
        ep_success = []
        while rollout_size < args.rl_n_steps:
            severity = sample_severity()
            temps, currents, voltages = sample_telemetry(severity, n_joints)
            obs = env.reset()
            current.reset()
            action_queue = deque()
            total_ep_rew = 0.0
            success = False
            for step in range(args.horizon):
                if not action_queue:
                    vla_obs = build_obs(obs)
                    actions = get_vla_action(
                        cfg=cfg, vla=vla, processor=processor,
                        obs=vla_obs, task_label="pick up the red cube",
                        action_head=action_head, proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                    )
                    action_queue.extend(actions)
                openvla_action = np.array(action_queue.popleft(), dtype=np.float32)
                openvla_action[6] = 1.0 - 2.0 * np.clip(openvla_action[6], 0, 1)
                openvla_action = np.clip(openvla_action, -1.0, 1.0)

                state_vec = build_state(obs)
                if state_vec.shape[0] < state_dim:
                    state_vec = np.concatenate([state_vec, np.zeros(state_dim - state_vec.shape[0], dtype=np.float32)])
                elif state_vec.shape[0] > state_dim:
                    state_vec = state_vec[:state_dim]

                # Sample action via gated-noise
                a_t = torch.as_tensor(openvla_action, dtype=torch.float32, device=device)[None]
                T_t = torch.as_tensor(temps, dtype=torch.float32, device=device)[None]
                C_t = torch.as_tensor(currents, dtype=torch.float32, device=device)[None]
                V_t = torch.as_tensor(voltages, dtype=torch.float32, device=device)[None]
                s_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device)[None]
                with torch.no_grad():
                    action_t, logit_sample, log_prob, value = adapter.gated_sample(
                        a_t, T_t, s_t, C_t, V_t)
                action = action_t.squeeze(0).cpu().numpy()
                logit_taken = logit_sample.squeeze(0).cpu().numpy()

                # Apply degradation
                action = thermal.apply_thermal_physics(action, temps)
                action = current.apply_current_physics(action, currents)
                action = voltage.apply_voltage_physics(action, voltages)

                obs, reward, done, info = env.step(action)
                total_ep_rew += reward
                if hasattr(env, "_check_success") and env._check_success():
                    success = True

                # Store
                S.append(state_vec); AB.append(openvla_action)
                T_arr.append(temps); C_arr.append(currents); V_arr.append(voltages)
                LOGIT_TAKEN.append(logit_taken)
                LOG_PROBS.append(log_prob.item())
                VALUES.append(value.item())
                REWARDS.append(float(reward))
                DONES.append(float(done or success))

                rollout_size += 1
                if rollout_size >= args.rl_n_steps:
                    break
                if done or success:
                    break
            ep_rewards.append(total_ep_rew)
            ep_success.append(1.0 if success else 0.0)

        # Convert to tensors
        S = torch.as_tensor(np.stack(S), dtype=torch.float32, device=device)
        AB = torch.as_tensor(np.stack(AB), dtype=torch.float32, device=device)
        T_arr = torch.as_tensor(np.stack(T_arr), dtype=torch.float32, device=device)
        C_arr = torch.as_tensor(np.stack(C_arr), dtype=torch.float32, device=device)
        V_arr = torch.as_tensor(np.stack(V_arr), dtype=torch.float32, device=device)
        LOGIT_TAKEN = torch.as_tensor(np.stack(LOGIT_TAKEN), dtype=torch.float32, device=device)
        LOG_PROBS = torch.as_tensor(np.array(LOG_PROBS), dtype=torch.float32, device=device)
        VALUES = torch.as_tensor(np.array(VALUES), dtype=torch.float32, device=device)
        REWARDS = torch.as_tensor(np.array(REWARDS), dtype=torch.float32, device=device)
        DONES = torch.as_tensor(np.array(DONES), dtype=torch.float32, device=device)

        # GAE
        gamma_g = 0.99; lam = 0.95
        N = len(REWARDS)
        adv = torch.zeros(N, device=device)
        last_gae = 0.0
        for t in reversed(range(N)):
            next_val = 0.0 if t == N - 1 or DONES[t] > 0.5 else VALUES[t + 1]
            delta = REWARDS[t] + gamma_g * next_val - VALUES[t]
            last_gae = delta + gamma_g * lam * (0.0 if DONES[t] > 0.5 else last_gae)
            adv[t] = last_gae
        returns = adv + VALUES
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        idx = np.arange(N)
        for epoch in range(args.rl_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, args.batch_size):
                ix = torch.as_tensor(idx[start:start + args.batch_size], device=device)
                new_lp, ent, new_val = adapter.gated_log_prob(
                    AB[ix], LOGIT_TAKEN[ix], T_arr[ix], S[ix], C_arr[ix], V_arr[ix])
                ratio = torch.exp(new_lp - LOG_PROBS[ix])
                surr1 = ratio * adv[ix]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv[ix]
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = 0.5 * ((new_val - returns[ix]) ** 2).mean()
                # KL to SFT snapshot
                with torch.no_grad():
                    sft_logit, sft_ls, _, _ = sft_snapshot.forward_delta(
                        AB[ix], T_arr[ix], S[ix], C_arr[ix], V_arr[ix])
                cur_logit, cur_ls, _, _ = adapter.forward_delta(
                    AB[ix], T_arr[ix], S[ix], C_arr[ix], V_arr[ix])
                # MSE on logits as KL surrogate
                kl_loss = (cur_logit - sft_logit).pow(2).mean()
                loss = pg_loss + 0.5 * vf_loss + args.kl_coef * kl_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)
                optimizer.step()

        total_steps += N
        avg_ep_rew = sum(ep_rewards) / max(1, len(ep_rewards))
        sr = sum(ep_success) / max(1, len(ep_success))
        elapsed = time.time() - t0
        print(f"  RL rollout={rollout_idx:>3d} step={total_steps:>6,d} n_eps={len(ep_rewards):>2d} "
              f"avg_rew={avg_ep_rew:6.2f} sr={sr:.0%} kl={kl_loss.item():.3f} "
              f"vf={vf_loss.item():.3f} pg={pg_loss.item():+.3f} elapsed={elapsed:.0f}s", flush=True)

        # Best-checkpoint logic
        score = sr * 1000 + avg_ep_rew
        if score > best_score:
            best_score = score
            torch.save(adapter.state_dict(), os.path.join(args.output_dir, "best_adapter.pt"))
        if total_steps % args.save_every < N:
            torch.save(adapter.state_dict(),
                       os.path.join(args.output_dir, f"adapter_step{total_steps}.pt"))

    env.close()
    torch.save(adapter.state_dict(), os.path.join(args.output_dir, "final_adapter.pt"))
    print(f"\nDone in {time.time()-t0:.0f}s. Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
