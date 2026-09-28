"""Strict state-dict loading with the training configuration kept explicit."""
import json
from pathlib import Path

import torch

from .tam_bot import TAMBoT


def load_adapter(checkpoint, *, config=None, device="cpu", alpha=None, gamma_range=None):
    """Load a TeAR/TAM state dict and its adjacent config.json.

    ``config`` may be a mapping or a JSON path. Non-tensor settings such as the
    number of attention heads and gain formula cannot be inferred from weights.
    ``alpha`` and ``gamma_range`` explicitly override deployment bounds.
    """
    path = Path(checkpoint)
    if config is None:
        config = path.with_name("config.json")
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    c = config
    for field in ("tam_n_heads", "tam_gamma_range"):
        if field not in c:
            raise ValueError(f"Adapter config requires {field}; it cannot be inferred from weights")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    hidden, joint_in = sd["joint_proj.weight"].shape
    has_state = "state_proj.weight" in sd
    layers = sum(k.startswith("transformer.layers.") and k.endswith(".norm1.weight") for k in sd)
    model = TAMBoT(
        state_dim=sd["state_proj.weight"].shape[1] if has_state else c.get("state_dim", 19),
        act_dim=sd["log_std"].numel(), hidden=hidden,
        n_layers=layers or c.get("tam_n_layers", 3), n_heads=c["tam_n_heads"],
        alpha=c.get("alpha", .3) if alpha is None else alpha,
        gamma_range=c["tam_gamma_range"] if gamma_range is None else gamma_range,
        gate_shape=c.get("gate_shape", "smoothstep"),
        mask_gripper=c.get("tam_mask_gripper", False),
        gamma_log_space=c.get("tam_gamma_log_space", False),
        use_action_magnitude=(joint_in == 5), use_state_token=has_state,
        use_per_joint_tokens=not c.get("tam_single_token", False),
        gate_scope=c.get("tam_gate_scope", "per_joint"),
        backbone=c.get("tam_backbone", "transformer"),
        mlp_hidden_mult=c.get("tam_mlp_hidden_mult", 2),
        action_mag_gate=c.get("tam_action_mag_gate", False),
        action_mag_threshold=c.get("tam_action_mag_threshold", .3),
        action_mag_steepness=c.get("tam_action_mag_steepness", 5.),
        n_joints=c.get("n_joints", 7),
    )
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()
