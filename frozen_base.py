"""Frozen-base-policy abstraction for the thermal adapter.

Loads any robomimic .pth checkpoint (BC, BC-Transformer, later Diffusion
Policy) and exposes a uniform interface:

    base = FrozenBase(ckpt_path, device="cuda")
    a_base      = base.act(obs_dict)                       # (B, act_dim)
    a_base, feats = base.act_and_features(obs_dict)        # for FiLM adapter
    state_vec   = base.state_from_obs(obs_dict)            # (B, state_dim)
    info        = dict(feat_dim, state_dim, act_dim, context_length, need_stack)

`obs_dict` values are numpy arrays of shape (T, ...) — i.e. frame-stacked
by the caller (needed for BC-Transformer). For non-sequence BC we just
take the last frame internally.

All parameters are frozen; the module is set to eval() and stays that way.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from robomimic.utils import file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils


# Keys we consider "low-dim state" when flattening for the adapter.
# Must match both the BC and BC-Transformer training obs layout.
STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
STATE_DIMS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4,
              "robot0_gripper_qpos": 2, "object": 10}


class FrozenBase(nn.Module):
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        rollout_policy, ckpt = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path, device=self.device, verbose=False,
        )
        # The actual nn.Module we'll call forward on (BC/BC-T/BCQ/IRIS/HBC).
        self.rollout_policy = rollout_policy
        self.algo = rollout_policy.policy
        try:
            self.algo.set_eval()
            if hasattr(self.algo, "nets"):
                for p in self.algo.nets.parameters():
                    p.requires_grad = False
        except Exception:
            pass  # some algos lock config or differ in net naming

        # Detect architecture type from algo config
        try:
            cfg = self.algo.global_config
            self.is_transformer = bool(
                getattr(cfg.algo, "transformer", {}).get("enabled", False))
            if self.is_transformer:
                self.context_length = int(cfg.algo.transformer.context_length)
            else:
                self.context_length = 1
        except Exception:
            self.is_transformer = False
            self.context_length = 1

        # Detect action dim and feature dim (lenient — fall back to defaults)
        try:
            self.act_dim = self._detect_act_dim()
        except Exception:
            self.act_dim = 7  # Panda OSC_POSE default
        try:
            self.feat_dim = self._detect_feat_dim()
        except Exception:
            self.feat_dim = 1024
        # Detect per-key state dims from the trained policy's obs spec — handles
        # tasks where `object` differs (e.g., Can: 14, Lift: 10, Square: 14).
        try:
            self.state_dims = {
                k: int(self.algo.obs_shapes[k][0])
                for k in STATE_KEYS if k in self.algo.obs_shapes
            }
        except Exception:
            self.state_dims = dict(STATE_DIMS)
        self.state_dim = sum(self.state_dims.get(k, STATE_DIMS[k])
                             for k in STATE_KEYS)

    # ── detection helpers ──────────────────────────────────────────────────
    def _detect_act_dim(self) -> int:
        try:
            dec = self.algo.nets["policy"].nets["decoder"]
            n_modes = dec.nets["logits"].out_features
            return dec.nets["mean"].out_features // n_modes
        except Exception:
            return 7  # fall back to Panda default

    def _detect_feat_dim(self) -> int:
        """For FiLM: return the dim of the hidden features we'll hook.
        - BC-Transformer → transformer output dim (512)
        - BC (MLP)       → actor_layer_dims[-1] (1024)
        """
        if self.is_transformer:
            try:
                return int(
                    self.algo.global_config.algo.transformer.embed_dim)
            except Exception:
                return 512
        try:
            dims = list(self.algo.global_config.algo.actor_layer_dims)
            return int(dims[-1])
        except Exception:
            return 1024

    # ── obs processing ─────────────────────────────────────────────────────
    @staticmethod
    def state_from_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten the state_keys of obs_dict (last timestep) → (state_dim,).

        Accepts dict values of shape (...) or (T, ...) and takes [-1] in the
        latter case.
        """
        parts = []
        for k in STATE_KEYS:
            v = obs_dict[k]
            # If we got a stacked sequence (T, D), keep only the last frame.
            if v.ndim >= 2 and v.shape[0] >= 1 and v.shape[-1] != 1:
                v = v[-1]
            parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0)

    def _obs_to_bc_input(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert a single stacked obs dict → (1, T, ...) tensors on device,
        with image processing applied."""
        out = {}
        for k, v in obs_dict.items():
            if k == "joint_temps":
                continue
            arr = np.asarray(v)
            if arr.ndim == 1:  # low-dim (D,) → (T=1, D)
                arr = arr[None, :]
            # Ensure leading T dim matches context length by repeating
            if arr.shape[0] != self.context_length and not self.is_transformer:
                arr = arr[-1:][None] if arr.ndim > 1 else arr[None]
            # Robomimic stores RGB as HWC; our env returns CHW. Transpose.
            if ObsUtils.key_is_obs_modality(k, "rgb") and arr.ndim == 4 and arr.shape[1] == 3:
                arr = np.transpose(arr, (0, 2, 3, 1))  # (T, C, H, W) → (T, H, W, C)
            t = torch.from_numpy(arr.astype(np.float32)).to(self.device)
            if ObsUtils.key_is_obs_modality(k, "rgb"):
                t = ObsUtils.process_obs(t, obs_modality="rgb")
            out[k] = t.unsqueeze(0)  # add batch dim → (1, T, ...)
        return out

    @torch.no_grad()
    def act(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute base action. Returns shape (act_dim,) numpy array."""
        bc_in = self._obs_to_bc_input(obs_dict)
        ac = self.algo.get_action(obs_dict=bc_in, goal_dict=None)
        return ac[0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act_batched(self, batched_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Same as act() but for a pre-batched tensor dict (for PPO update).
        batched_obs values have shape (B, T, ...). Returns (B, act_dim)."""
        ac = self.algo.get_action(obs_dict=batched_obs, goal_dict=None)
        return ac.detach()
