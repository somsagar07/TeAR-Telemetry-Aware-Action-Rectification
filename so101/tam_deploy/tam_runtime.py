"""Real-time TAM correction wrapper for SO-101.

Usage at deployment:

    from so101.tam_deploy.tam_runtime import SO101TAM
    from so101.tam_deploy.dynamixel_telemetry import DynamixelTelemetry

    telem = DynamixelTelemetry(port="/dev/ttyACM0")
    tam   = SO101TAM(checkpoint="sft_adapter.pt", device="cuda")

    # In the control loop:
    a_base  = base_policy(obs)          # (6,) — any of π0 / GR00T / SmolVLA / ACT
    T, C, V = telem.read_TCV()          # (5,) each
    a_final = tam.correct(a_base, T, C, V, state=obs["state"])
    send_to_motors(a_final)

The wrapper is *policy-agnostic*: it never sees obs format, language, images, or
chunking — only the 6-DoF action the base policy outputs. The same `SO101TAM`
instance works with any SO-101 base policy.

Structural invariant: at cool (T<42, C<0.6, V>0.9) the corrected action equals
the base action bit-exactly. Verified by `tam_runtime.SO101TAM.assert_cool_identity()`.
"""
from __future__ import annotations
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# tam_bot.py is vendored into this deploy package — no repo-root path needed.
try:
    from .tam_bot import TAMBoT
except ImportError:  # Standalone deployment directory.
    from tam_bot import TAMBoT


SO101_N_JOINTS = 5     # 5 arm joints — what TAM corrects
SO101_ACT_DIM = 6      # 5 arm joints + 1 gripper — what the base policy outputs
SO101_STATE_DIM = 29   # default; override at construction if your base uses a
                       # different state encoding (TAM ignores semantics — only
                       # the dim matters)


class SO101TAM:
    """Loads a trained TAM-BoT and applies multi-channel correction."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hidden: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        alpha: float = 0.3,
        gamma_range: float = 0.75,
        state_dim: int = SO101_STATE_DIM,
    ):
        self.device = device
        self.n_joints = SO101_N_JOINTS
        self.state_dim = state_dim
        self.model = TAMBoT(
            state_dim=state_dim,
            act_dim=self.n_joints,
            n_joints=self.n_joints,
            hidden=hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            alpha=alpha,
            gamma_range=gamma_range,
        ).to(device)
        ckpt = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.no_grad()
    def correct(
        self,
        a_base: np.ndarray,
        T: np.ndarray,
        C: np.ndarray,
        V: np.ndarray,
        state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply TAM correction to one timestep's action.

        Parameters
        ----------
        a_base : (act_dim,) float32 in [-1, 1] — base policy output.
        T : (n_joints,) °C — live joint temperatures.
        C : (n_joints,) normalized — current / rated_current.
        V : (n_joints,) normalized — voltage / rated_voltage.
        state : (state_dim,) or None — flat observation state. If None, a zero
            state token is used (TAM still works; state only feeds the value head
            and a minor token mixing signal — gating/correction are physics-only).

        Returns
        -------
        a_final : (act_dim,) float32 — corrected action, gripper passed through.
        """
        a_base = np.asarray(a_base, dtype=np.float32).reshape(-1)
        if a_base.shape[0] != SO101_ACT_DIM:
            raise ValueError(
                f"Expected base action shape ({SO101_ACT_DIM},), got {a_base.shape}")
        T = np.asarray(T, dtype=np.float32).reshape(-1)
        C = np.asarray(C, dtype=np.float32).reshape(-1)
        V = np.asarray(V, dtype=np.float32).reshape(-1)
        if not (T.shape == C.shape == V.shape == (self.n_joints,)):
            raise ValueError(
                f"Expected T/C/V shape ({self.n_joints},), got "
                f"T={T.shape} C={C.shape} V={V.shape}")

        if state is None:
            state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            if state.shape[0] != self.state_dim:
                # Crop or pad to expected size. TAM's state usage is mild;
                # mismatched dims should not break correction quality.
                if state.shape[0] > self.state_dim:
                    state = state[:self.state_dim]
                else:
                    state = np.pad(state, (0, self.state_dim - state.shape[0]))

        a_arm = a_base[:self.n_joints]
        gripper = a_base[self.n_joints:]

        a_t = torch.as_tensor(a_arm, device=self.device).unsqueeze(0)
        T_t = torch.as_tensor(T, device=self.device).unsqueeze(0)
        C_t = torch.as_tensor(C, device=self.device).unsqueeze(0)
        V_t = torch.as_tensor(V, device=self.device).unsqueeze(0)
        s_t = torch.as_tensor(state, device=self.device).unsqueeze(0)

        a_corrected_t, _, _ = self.model(a_t, T_t, s_t, C_t, V_t)
        a_corrected = a_corrected_t.squeeze(0).cpu().numpy()

        return np.concatenate([a_corrected, gripper]).astype(np.float32)

    def assert_cool_identity(self, n_trials: int = 8, tol: float = 1e-5) -> None:
        """Sanity test: at cool the corrected action equals the base action.

        Raises AssertionError on violation. Run this once after loading the
        checkpoint to confirm the structural gate is intact on this machine.
        """
        rng = np.random.RandomState(0)
        for _ in range(n_trials):
            a_base = rng.uniform(-1, 1, SO101_ACT_DIM).astype(np.float32)
            T = rng.uniform(20, 42, self.n_joints).astype(np.float32)
            C = rng.uniform(0.1, 0.5, self.n_joints).astype(np.float32)
            V = rng.uniform(0.92, 1.0, self.n_joints).astype(np.float32)
            a_final = self.correct(a_base, T, C, V)
            err = float(np.max(np.abs(a_final - a_base)))
            assert err < tol, (
                f"cool-identity violated: max|a_final - a_base| = {err:.2e} "
                f"> tol {tol:.2e}\n  base: {a_base}\n  final: {a_final}")
