"""TAM-BoT — Body-Token Telemetry Adapter.

A small (~70K-param) transformer-based adapter that wraps a frozen base policy
under multi-channel joint degradation. Combines:

  1. Per-joint tokens (Body-Transformer style; Sferrazza 2024 arXiv:2408.06316).
     Each joint gets its own token; self-attention discovers cross-joint coupling.

  2. Dual output per joint: multiplicative scale γ (FiLM-style; Perez 2018
     arXiv:1709.07871) AND additive residual δ. The multiplicative head matches
     the physics — degradation is multiplicative (T_factor × C_factor × V_factor),
     so the natural inverse is also multiplicative. The residual head handles
     direction-correcting nudges that pure scaling can't express.

  3. Per-joint smoothstep gate. Hard-zero at nominal (max_T<=42, max_C<=0.60,
     min_V>=0.90), 3x²-2x³ ramp through the moderate range, fully on at full
     stress. Gentler at the warm shoulder than linear ramp (fixes the Can T=55
     regression observed in track 1).

Action rule (per joint j):

    γ_j = 1 + g_j · γ_range · tanh(γ_logit_j)     ∈ [1-γ_range, 1+γ_range]
    δ_j = g_j · α · tanh(δ_logit_j)               ∈ [-α, α]
    a_final[j] = clip(γ_j · a_base[j] + δ_j, -1, 1)

At cool (g=0): γ=1, δ=0  ⇒  a_final = clip(a_base) = a_base  bit-exactly.
This is structural — no reliance on SFT to "discover" identity.

Inputs (matching BCTSFTRLAdapter signature):
  forward(a_base, temps, state, currents, voltages) -> (mean, log_std, value)
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class TAMBoT(nn.Module):

    def __init__(
        self,
        state_dim: int = 19,
        act_dim: int = 7,
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        alpha: float = 0.3,
        gamma_range: float = 0.5,
        log_std_init: float = -2.0,
        gate_shape: str = "smoothstep",
        mask_gripper: bool = False,
        gamma_log_space: bool = False,
        use_action_magnitude: bool = False,
        use_state_token: bool = True,
        use_per_joint_tokens: bool = True,
        gate_scope: str = "per_joint",  # "per_joint" | "scalar"
        backbone: str = "transformer",  # "transformer" | "mlp"
        mlp_hidden_mult: int = 2,
        action_mag_gate: bool = False,
        action_mag_threshold: float = 0.3,
        action_mag_steepness: float = 5.0,
        n_joints: int = 7,
    ):
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        assert gate_scope in ("per_joint", "scalar")
        assert backbone in ("transformer", "mlp")
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.alpha = alpha
        self.gamma_range = gamma_range
        self.gate_shape = gate_shape
        self.mask_gripper = mask_gripper
        self.gamma_log_space = gamma_log_space
        self.use_action_magnitude = use_action_magnitude
        self.action_mag_gate = action_mag_gate
        self.action_mag_threshold = action_mag_threshold
        self.action_mag_steepness = action_mag_steepness
        self.use_state_token = use_state_token
        self.use_per_joint_tokens = use_per_joint_tokens
        self.gate_scope = gate_scope
        self.backbone = backbone
        self.n_joints = int(n_joints)  # 7 for Panda, 5 for SO-101

        # Per-joint token: [T_j, C_j, V_j, a_base_j (, |a_base_j|)] -> hidden
        joint_in = 5 if use_action_magnitude else 4
        self.joint_proj = nn.Linear(joint_in, hidden)
        # State token: full state -> hidden
        self.state_proj = nn.Linear(state_dim, hidden) if use_state_token else None
        # Positional embeddings: (1 + 7) tokens or just 7 if no state, or 1 if scalar
        n_tokens = (1 if use_state_token else 0) + (self.n_joints if use_per_joint_tokens else 1)
        self.n_tokens = n_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, hidden))
        nn.init.normal_(self.pos_emb, std=0.02)

        if backbone == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads,
                dim_feedforward=hidden * 2,
                batch_first=True, activation="gelu",
                norm_first=True, dropout=0.0,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.final_norm = nn.LayerNorm(hidden)
        else:
            # MLP backbone: flatten tokens, run through MLP, reshape back
            self.transformer = None
            ff_hidden = hidden * mlp_hidden_mult
            mlp_layers = []
            in_dim = n_tokens * hidden
            for _ in range(n_layers):
                mlp_layers += [nn.Linear(in_dim, ff_hidden), nn.GELU()]
                in_dim = ff_hidden
            mlp_layers += [nn.Linear(ff_hidden, n_tokens * hidden)]
            self.mlp_backbone = nn.Sequential(*mlp_layers)
            self.final_norm = nn.LayerNorm(hidden)

        # Fallback projection for non-per-joint (single-token) modes
        if not use_per_joint_tokens:
            # When we don't have per-joint tokens, we expand the single representation
            # back to 7 outputs via this head
            self.expand_to_joints = nn.Linear(hidden, self.n_joints * hidden)
        else:
            self.expand_to_joints = None

        # Per-joint output heads (zero-init -> identity behavior at start)
        self.gamma_head = nn.Linear(hidden, 1)
        self.delta_head = nn.Linear(hidden, 1)
        for h in (self.gamma_head, self.delta_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        # Value head reads from state token
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Per-action log_std (compatible with legacy action-space PPO sampling).
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def _ramp(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_shape == "smoothstep":
            return x * x * (3.0 - 2.0 * x)
        return x  # linear

    def per_joint_gate(self, T, C, V):
        """g_j ∈ [0,1] per joint. T/C/V are (B, 7)."""
        tg = ((T - 42.0) / 13.0).clamp(0.0, 1.0)
        cg = ((C - 0.60) / 0.40).clamp(0.0, 1.0)
        vg = ((0.90 - V) / 0.40).clamp(0.0, 1.0)
        g = (self._ramp(tg) + self._ramp(cg) + self._ramp(vg)).clamp(0.0, 1.0)
        if self.gate_scope == "scalar":
            # Single scalar gate = max across joints (still per-batch)
            g = g.max(dim=-1, keepdim=True).values.expand(-1, self.n_joints)
        return g

    def _encode(self, a_base, T, C, V, state):
        feats = [T, C, V, a_base]
        if self.use_action_magnitude:
            feats.append(a_base.abs())
        joint_features = torch.stack(feats, dim=-1)  # (B, 7, joint_in)

        if self.use_per_joint_tokens:
            joint_tokens = self.joint_proj(joint_features)  # (B, 7, H)
            if self.use_state_token and self.state_proj is not None:
                state_token = self.state_proj(state).unsqueeze(1)
                tokens = torch.cat([state_token, joint_tokens], dim=1)
            else:
                tokens = joint_tokens
        else:
            # Single token mode: pool joint features
            pooled = joint_features.mean(dim=1)  # (B, joint_in)
            single_token = self.joint_proj(pooled).unsqueeze(1)
            if self.use_state_token and self.state_proj is not None:
                state_token = self.state_proj(state).unsqueeze(1)
                tokens = torch.cat([state_token, single_token], dim=1)
            else:
                tokens = single_token

        tokens = tokens + self.pos_emb
        if self.backbone == "transformer":
            out = self.final_norm(self.transformer(tokens))
        else:
            # MLP backbone: flatten -> mlp -> reshape
            B = tokens.shape[0]
            flat = tokens.reshape(B, -1)
            out_flat = self.mlp_backbone(flat)
            out = self.final_norm(out_flat.reshape(B, self.n_tokens, self.hidden))

        # Slice joint vs state outputs
        if self.use_per_joint_tokens:
            joint_start = 1 if self.use_state_token else 0
            joint_out = out[:, joint_start:joint_start + self.n_joints]
            state_out = out[:, 0] if self.use_state_token else joint_out.mean(dim=1)
        else:
            # Single token mode: expand back to 7 joint outputs
            single_idx = 1 if self.use_state_token else 0
            single = out[:, single_idx]
            expanded = self.expand_to_joints(single).reshape(
                tokens.shape[0], self.n_joints, self.hidden)
            joint_out = expanded
            state_out = out[:, 0] if self.use_state_token else single
        return joint_out, state_out

    def forward_logits(self, a_base, temps, state, currents, voltages):
        """Pre-gate γ_logit, δ_logit + per-joint gate + value (used by gated-noise PPO)."""
        joint_out, state_out = self._encode(a_base, temps, currents, voltages, state)
        gamma_logit = self.gamma_head(joint_out).squeeze(-1)  # (B, 7)
        delta_logit = self.delta_head(joint_out).squeeze(-1)
        g = self.per_joint_gate(temps, currents, voltages)    # (B, 7)
        value = self.value_head(state_out).squeeze(-1)
        return gamma_logit, delta_logit, g, value

    def logit_to_action(self, a_base, gamma_logit, delta_logit, g):
        # Optional: modulate gate by action magnitude (only correct when base
        # is "committed", i.e. |a_base| above a threshold). Preserves cool
        # identity since g already zero at cool.
        if self.action_mag_gate:
            mag = a_base.abs()  # (B, 7)
            mag_factor = torch.sigmoid(
                self.action_mag_steepness * (mag - self.action_mag_threshold))
            g = g * mag_factor  # (B, 7) — per-joint
        if self.gamma_log_space:
            # Multiplicative γ via log-space: γ = exp(g · r · tanh(logit))
            # At gate=0: γ=1. At gate=1, tanh=+1: γ=exp(r). At tanh=-1: γ=exp(-r).
            # Log-symmetric → linearly asymmetric (biased above 1).
            gamma = torch.exp(g * self.gamma_range * torch.tanh(gamma_logit))
        else:
            gamma = 1.0 + g * self.gamma_range * torch.tanh(gamma_logit)
        delta = g * self.alpha * torch.tanh(delta_logit)
        # Mask the gripper action (action[6]) from correction — it's not motor-degraded
        if self.mask_gripper:
            mask = torch.ones_like(gamma)
            mask_zero = torch.zeros_like(delta)
            gamma = torch.cat([gamma[..., :6], mask[..., 6:7]], dim=-1)
            delta = torch.cat([delta[..., :6], mask_zero[..., 6:7]], dim=-1)
        return (gamma * a_base + delta).clamp(-1.0, 1.0)

    def forward(self, a_base, temps, state, currents, voltages):
        gamma_logit, delta_logit, g, value = self.forward_logits(
            a_base, temps, state, currents, voltages)
        action = self.logit_to_action(a_base, gamma_logit, delta_logit, g)
        return action, self.log_std, value

    # ---- PPO-compatible sampling (legacy action-noise) ----
    def get_action_and_value(self, a_base, temps, state, currents, voltages, action=None):
        mean, log_std, value = self.forward(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        lp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        return action, lp, ent, value

    # ---- Gated-noise PPO (cool-identity preserved during training rollout) ----
    # Treats (γ_logit, δ_logit) ∈ R^14 as the sampling space. Noise on logits
    # doesn't perturb the action at cool because gate=0 → γ=1, δ=0 regardless.
    def forward_delta(self, a_base, temps, state, currents, voltages):
        gamma_logit, delta_logit, g, value = self.forward_logits(
            a_base, temps, state, currents, voltages)
        flat = torch.cat([gamma_logit, delta_logit], dim=-1)            # (B, 14)
        log_std_flat = torch.cat([self.log_std, self.log_std], dim=-1)  # (14,)
        return flat, log_std_flat, g, value

    def delta_to_action(self, a_base, flat_logits, g):
        gamma_logit = flat_logits[..., :7]
        delta_logit = flat_logits[..., 7:]
        return self.logit_to_action(a_base, gamma_logit, delta_logit, g)

    def gated_sample(self, a_base, temps, state, currents, voltages):
        flat_mean, log_std, g, value = self.forward_delta(
            a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(flat_mean)
        dist = Normal(flat_mean, std)
        flat_sample = dist.sample()
        action = self.delta_to_action(a_base, flat_sample, g)
        log_prob = dist.log_prob(flat_sample).sum(-1)
        return action, flat_sample, log_prob, value

    def gated_log_prob(self, a_base, flat_taken, temps, state, currents, voltages):
        flat_mean, log_std, g, value = self.forward_delta(
            a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(flat_mean)
        dist = Normal(flat_mean, std)
        log_prob = dist.log_prob(flat_taken).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def get_value(self, a_base, temps, state, currents, voltages):
        """Value estimate alone (used by PPO bootstrap at episode boundaries)."""
        _, state_out = self._encode(a_base, temps, currents, voltages, state)
        return self.value_head(state_out).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    m = TAMBoT(state_dim=19, hidden=64, n_layers=2, n_heads=4)
    print(f"TAM-BoT params: {count_params(m):,}")
    B = 4
    a = torch.randn(B, 7).clamp(-1, 1)
    T = torch.full((B, 7), 25.0)
    C = torch.full((B, 7), 0.30)
    V = torch.ones(B, 7)
    s = torch.randn(B, 19)
    out, log_std, val = m(a, T, s, C, V)
    print(f"Cool action diff from a_base (should be 0): {(out - a).abs().max().item():.2e}")
    # Hot conditions
    T = torch.full((B, 7), 70.0)
    C = torch.full((B, 7), 0.95)
    V = torch.full((B, 7), 0.50)
    out_hot, _, _ = m(a, T, s, C, V)
    print(f"Hot action diff from a_base: {(out_hot - a).abs().max().item():.2e}")
