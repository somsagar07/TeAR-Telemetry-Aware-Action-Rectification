#!/usr/bin/env python3
"""
Fine-tune OpenVLA-OFT with telemetry degradation augmentation.

Extends the standard OFT fine-tuning by applying simulated temperature,
current, and voltage degradation to demonstration actions during training.
This teaches the model to predict actions that compensate for degradation.

Strategy: For each training sample, with probability p_degrade, sample
random telemetry conditions and apply degradation physics to the expert
action. The model learns to map (image, proprio) → degraded_action,
effectively learning an inverse degradation model.

Usage:
  # From openvla-oft conda env:
  python openvla/finetune_openvla_telemetry.py \
    --data_root_dir datasets/rlds \
    --dataset_name robosuite_lift_ph \
    --p_degrade 0.5 \
    --max_steps 50000 \
    --run_id_note telemetry_v1

  # To resume from existing OFT checkpoint and fine-tune with degradation:
  python openvla/finetune_openvla_telemetry.py \
    --vla_path runs/openvla-oft-v1/<checkpoint> \
    --p_degrade 0.5 \
    --max_steps 20000
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch

# Make project root importable for telemetry models
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.thermal_model import ThermalModel
from env.telemetry_model import CurrentModel, VoltageModel


class TelemetryAugmentor:
    """Applies random telemetry degradation to expert actions during training.

    Each call to `augment(action)` either returns the action unchanged
    (with probability 1-p_degrade) or applies random temperature, current,
    and voltage degradation physics (with probability p_degrade).
    """

    def __init__(self, p_degrade=0.5, n_joints=7):
        self.p_degrade = p_degrade
        self.n_joints = n_joints
        self.thermal = ThermalModel.from_predefined("linear", n_joints=n_joints)
        self.current = CurrentModel.from_predefined("linear", n_joints=n_joints)
        self.voltage = VoltageModel.from_predefined("linear", n_joints=n_joints)

    def augment(self, action):
        """Apply random telemetry degradation to an action.

        Args:
            action: (7,) numpy array in [-1, 1]
        Returns:
            degraded_action: (7,) numpy array in [-1, 1]
            telemetry: dict with temps, currents, voltages (for logging)
        """
        if np.random.random() > self.p_degrade:
            return action.copy(), None

        # Sample severity: uniform mix of mild to severe
        severity = np.random.uniform(0, 1)

        if severity < 0.33:
            # Mild: warm temps, moderate current, slight undervoltage
            temps = np.random.uniform(43, 55, self.n_joints).astype(np.float32)
            currents = np.random.uniform(0.5, 0.75, self.n_joints).astype(np.float32)
            voltages = np.random.uniform(0.80, 0.92, self.n_joints).astype(np.float32)
        elif severity < 0.66:
            # Moderate: hot temps, high current, undervoltage
            temps = np.random.uniform(50, 65, self.n_joints).astype(np.float32)
            currents = np.random.uniform(0.65, 0.90, self.n_joints).astype(np.float32)
            voltages = np.random.uniform(0.65, 0.85, self.n_joints).astype(np.float32)
        else:
            # Severe: very hot, near-stall, brownout
            temps = np.random.uniform(60, 75, self.n_joints).astype(np.float32)
            currents = np.random.uniform(0.80, 1.05, self.n_joints).astype(np.float32)
            voltages = np.random.uniform(0.45, 0.70, self.n_joints).astype(np.float32)

        degraded = action.copy()
        degraded = self.thermal.apply_thermal_physics(degraded, temps)
        self.current.reset()
        degraded = self.current.apply_current_physics(degraded, currents)
        degraded = self.voltage.apply_voltage_physics(degraded, voltages)

        telemetry = {"temps": temps, "currents": currents, "voltages": voltages}
        return degraded, telemetry


def patch_rlds_dataset(dataset_cls, augmentor):
    """Monkey-patch an RLDS dataset class to apply telemetry augmentation.

    The OFT pipeline uses RLDSDataset which yields dicts with 'action' key.
    We wrap the __getitem__ to apply degradation to the action before
    the model sees it.
    """
    original_getitem = dataset_cls.__getitem__

    def augmented_getitem(self, idx):
        sample = original_getitem(self, idx)
        if "action" in sample and augmentor is not None:
            action = sample["action"]
            if isinstance(action, torch.Tensor):
                action_np = action.numpy()
            else:
                action_np = np.array(action, dtype=np.float32)
            degraded, _ = augmentor.augment(action_np)
            sample["action"] = torch.as_tensor(degraded, dtype=torch.float32)
        return sample

    dataset_cls.__getitem__ = augmented_getitem
    return dataset_cls


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune OpenVLA-OFT with telemetry degradation augmentation")
    parser.add_argument("--p_degrade", type=float, default=0.5,
                        help="Probability of applying degradation to each sample")
    parser.add_argument("--finetune_args", nargs=argparse.REMAINDER, default=[],
                        help="All remaining args are passed to the OFT finetune script")
    args, remaining = parser.parse_known_args()

    print("=" * 70)
    print("  OpenVLA-OFT Telemetry Fine-tuning")
    print(f"  p_degrade={args.p_degrade}")
    print("=" * 70)

    # Create augmentor
    augmentor = TelemetryAugmentor(p_degrade=args.p_degrade)
    print(f"  TelemetryAugmentor: p={args.p_degrade}, models=linear")
    print(f"  Degradation sample:")
    test_action = np.array([0.5, -0.3, 0.8, 0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    deg_action, tel = augmentor.augment(test_action)
    if tel is not None:
        print(f"    Original: {test_action[:3]}")
        print(f"    Degraded: {deg_action[:3]}")
        print(f"    Temps: {tel['temps'][:3].round(1)}")

    # Import OFT training components
    OFT_REPO = "<DATA_PATH>"
    sys.path.insert(0, OFT_REPO)

    from prismatic.vla.datasets import RLDSDataset

    # Monkey-patch the dataset
    patch_rlds_dataset(RLDSDataset, augmentor)
    print("  Patched RLDSDataset with telemetry augmentation")

    # Now run the standard OFT finetune script with remaining args
    # We import and call the main finetune function
    print(f"  Passing to OFT finetune: {remaining}")
    print("=" * 70)

    # Set sys.argv for draccus config parsing
    sys.argv = ["finetune.py"] + remaining

    from importlib import import_module
    finetune_module = import_module("vla-scripts.finetune")
    # draccus.wrap handles the main() call


if __name__ == "__main__":
    main()
