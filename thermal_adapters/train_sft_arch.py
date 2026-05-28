#!/usr/bin/env python3
"""SFT trainer for arbitrary architecture variants from sft_arch_variants.py.

Usage:
    python thermal_adapters/train_sft_arch.py \\
        --task-name PickPlaceCan \\
        --base-ckpt <BC-T> \\
        --demo-hdf5 <robomimic.hdf5> \\
        --variant mlp_h256_b2 \\
        --sft-steps 20000 \\
        --output-dir multi_task_runs/can/arch_<variant>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

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
from frozen_base import FrozenBase
from thermal_adapters.sft_arch_variants import build, VARIANT_REGISTRY


def load_demos(demo_hdf5, state_keys=("robot0_eef_pos", "robot0_eef_quat",
                                      "robot0_gripper_qpos", "object")):
    states, actions = [], []
    with h5py.File(demo_hdf5, "r") as f:
        for k in f["data"].keys():
            d = f["data"][k]
            parts = [np.asarray(d["obs"][sk], dtype=np.float32) for sk in state_keys]
            s = np.concatenate(parts, axis=-1)
            a = np.asarray(d["actions"], dtype=np.float32)
            states.append(s)
            actions.append(a)
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


def run_sft(adapter, all_states, all_actions, args, device):
    print("\n=== SFT (arch variant) ===", flush=True)
    n = len(all_states)
    print(f"  demos: {n} (state_dim={all_states.shape[1]})", flush=True)

    thermal = ThermalModel.from_predefined("linear", n_joints=7)
    current_m = CurrentModel.from_predefined("linear", n_joints=7)
    voltage_m = VoltageModel.from_predefined("linear", n_joints=7)

    # 4 severity tiers (matches the OpenVLA SFT setup).
    big_s, big_a, big_t, big_c, big_v, big_tgt = [], [], [], [], [], []
    for sev_name, ranges in [
        ("clean", ((20, 42), (0.10, 0.50), (0.92, 1.00))),
        ("mild",  ((43, 55), (0.50, 0.75), (0.80, 0.92))),
        ("mod",   ((50, 65), (0.65, 0.90), (0.65, 0.85))),
        ("sev",   ((60, 75), (0.80, 1.05), (0.45, 0.70))),
    ]:
        tr, cr, vr = ranges
        T = np.random.uniform(*tr, (n, 7)).astype(np.float32)
        C = np.random.uniform(*cr, (n, 7)).astype(np.float32)
        V = np.random.uniform(*vr, (n, 7)).astype(np.float32)
        factors = np.ones((n, 7), np.float32)
        for i in range(n):
            for j in range(7):
                td = thermal.compute_degradation(float(T[i, j]), j)
                cd = current_m.compute_degradation(float(C[i, j]), j)
                vd = voltage_m.compute_degradation(float(V[i, j]), j)
                factors[i, j] = td * cd * vd
        # gripper not degraded
        f7 = np.concatenate([factors[:, :6], np.ones((n, 1), np.float32)], axis=-1)
        targets = np.clip(all_actions / np.maximum(f7, 0.05), -1, 1).astype(np.float32)
        big_s.append(all_states); big_a.append(all_actions); big_t.append(T)
        big_c.append(C); big_v.append(V); big_tgt.append(targets)

    big_s = np.concatenate(big_s); big_a = np.concatenate(big_a)
    big_t = np.concatenate(big_t); big_c = np.concatenate(big_c)
    big_v = np.concatenate(big_v); big_tgt = np.concatenate(big_tgt)
    N = len(big_s)
    print(f"  augmented: {N} samples (4× severity)", flush=True)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.sft_lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sft_steps)

    adapter.train()
    t0 = time.time()
    last_loss = None
    loss_history = []  # (step, loss, loss_tgt, loss_id) sampled every 100 steps
    for step in range(args.sft_steps):
        idx = np.random.choice(N, args.batch_size, replace=False)
        s = torch.as_tensor(big_s[idx], device=device)
        a = torch.as_tensor(big_a[idx], device=device)
        t = torch.as_tensor(big_t[idx], device=device)
        c = torch.as_tensor(big_c[idx], device=device)
        v = torch.as_tensor(big_v[idx], device=device)
        tgt = torch.as_tensor(big_tgt[idx], device=device)

        mean, _, _ = adapter(a, t, s, c, v)
        loss_tgt = F.l1_loss(mean, tgt)
        clean_mask = ((t.max(-1).values < 42) & (c.max(-1).values < 0.6)
                      & (v.min(-1).values > 0.9))
        loss_id = F.l1_loss(mean[clean_mask], a[clean_mask]) if clean_mask.any() \
                  else torch.tensor(0.0, device=device)
        loss = loss_tgt + 3.0 * loss_id
        last_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step == 0 or (step + 1) % 100 == 0:
            loss_history.append((step + 1, loss.item(), loss_tgt.item(), loss_id.item()))
        if (step + 1) % 2000 == 0 or step == 0:
            print(f"  SFT step={step+1:>6} loss={loss.item():.4f} "
                  f"(tgt={loss_tgt.item():.4f} id={loss_id.item():.4f})", flush=True)

    torch.save(adapter.state_dict(), os.path.join(args.output_dir, "sft_adapter.pt"))
    json.dump(loss_history, open(os.path.join(args.output_dir, "loss_history.json"), "w"))
    print(f"  SFT done ({time.time()-t0:.0f}s). final_loss={last_loss:.4f}", flush=True)
    return last_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--demo-hdf5", required=True)
    ap.add_argument("--variant", required=True, choices=list(VARIANT_REGISTRY))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--sft-steps", type=int, default=20000)
    ap.add_argument("--sft-lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "config.json"), "w"), indent=2)

    print("=" * 70, flush=True)
    print(f"  variant={args.variant}  task={args.task_name}", flush=True)
    print("=" * 70, flush=True)

    base = FrozenBase(args.base_ckpt, device=str(device))
    adapter = build(args.variant, state_dim=base.state_dim, act_dim=base.act_dim,
                    alpha=args.alpha).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  adapter params: {n_params:,}", flush=True)

    states, actions = load_demos(args.demo_hdf5)
    assert states.shape[1] == base.state_dim, \
        f"demo state_dim {states.shape[1]} != base.state_dim {base.state_dim}"

    final_loss = run_sft(adapter, states, actions, args, device)
    summary = {"variant": args.variant, "n_params": n_params, "final_loss": final_loss}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Done. n_params={n_params:,}  loss={final_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
