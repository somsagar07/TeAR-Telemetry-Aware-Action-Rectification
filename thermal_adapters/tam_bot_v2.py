"""TAM-BoT-v2 — Per-channel decomposed multiplicative-residual adapter.

Builds on TAM-BoT (per-joint Body Transformer) with one critical change:
each telemetry channel (T, C, V) gets its OWN γ and δ heads, gated by its
OWN per-joint gate. Final correction composes multiplicatively across
channels (matching the underlying physics where degradations stack
multiplicatively: T_factor × C_factor × V_factor).

Why per-channel decomposition matters
-------------------------------------
TAM-BoT's single γ head was driven by a combined OR-style gate
(g = clamp(g_T + g_C + g_V, 0, 1)). When current is stressed, the SAME
γ head activates that's also responsible for thermal correction. The
adapter has to learn from a confused training signal — every gate
activation could be due to T, C, or V, and the head must marginalise.

In the data this manifested as:
  - Current axis: every method gives Δ=0 (heads can't isolate current
    correction from thermal noise during training).
  - Voltage moderate (0.65-0.78): regressions for most methods (mixed
    signal causes over-correction).

Per-channel heads isolate the signal. γ_T is trained ONLY when g_T > 0;
γ_C only when g_C > 0; γ_V only when g_V > 0. Each head learns the
inverse of its own physics channel.

Action rule (per joint j)
-------------------------
  g_T_j = smoothstep((T_j - 42) / 13).clamp(0,1)
  g_C_j = smoothstep((C_j - 0.60) / 0.40).clamp(0,1)
  g_V_j = smoothstep((0.90 - V_j) / 0.40).clamp(0,1)

  γ_T_j = 1 + g_T_j · r · tanh(γ_T_logit_j)         ∈ [1-r, 1+r]
  γ_C_j = 1 + g_C_j · r · tanh(γ_C_logit_j)
  γ_V_j = 1 + g_V_j · r · tanh(γ_V_logit_j)
  γ_j   = γ_T_j · γ_C_j · γ_V_j                     (multiplicative composition)

  δ_j   = (g_T_j · tanh(δ_T_logit_j)
         + g_C_j · tanh(δ_C_logit_j)
         + g_V_j · tanh(δ_V_logit_j)) · α           (additive, gated)

  a_final[j] = clip(γ_j · a_base[j] + δ_j, -1, 1)

Cool guarantee (zero corruption)
--------------------------------
At nominal: g_T = g_C = g_V = 0 → γ_T = γ_C = γ_V = 1 → γ = 1; δ = 0
        ⇒ a_final = clip(a_base) = a_base bit-exactly.

This is structural, not learned.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class TAMBoTv2(nn.Module):

    def __init__(
        self,
        state_dim: int = 19,
        act_dim: int = 7,
        hidden: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        alpha: float = 0.3,
        gamma_range: float = 0.4,
        log_std_init: float = -2.0,
        gate_shape: str = "smoothstep",
        use_action_magnitude: bool = True,
    ):
        super().__init__()
        assert hidden % n_heads == 0
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.alpha = alpha
        self.gamma_range = gamma_range
        self.gate_shape = gate_shape
        self.use_action_magnitude = use_action_magnitude
        self.n_joints = 7

        # Per-joint token: [T_j, C_j, V_j, a_base_j, |a_base_j|] (or 4-d if no |a|)
        joint_in = 5 if use_action_magnitude else 4
        self.joint_proj = nn.Linear(joint_in, hidden)
        self.state_proj = nn.Linear(state_dim, hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.n_joints, hidden))
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu",
            norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(hidden)

        # Per-channel γ heads (each outputs per-joint logit)
        self.gamma_T_head = nn.Linear(hidden, 1)
        self.gamma_C_head = nn.Linear(hidden, 1)
        self.gamma_V_head = nn.Linear(hidden, 1)
        # Per-channel δ heads
        self.delta_T_head = nn.Linear(hidden, 1)
        self.delta_C_head = nn.Linear(hidden, 1)
        self.delta_V_head = nn.Linear(hidden, 1)
        for h in (self.gamma_T_head, self.gamma_C_head, self.gamma_V_head,
                  self.delta_T_head, self.delta_C_head, self.delta_V_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def _ramp(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_shape == "smoothstep":
            return x * x * (3.0 - 2.0 * x)
        return x

    def per_joint_gates(self, T, C, V):
        """Returns (g_T, g_C, g_V), each (B, 7) ∈ [0,1]."""
        gT = self._ramp(((T - 42.0) / 13.0).clamp(0.0, 1.0))
        gC = self._ramp(((C - 0.60) / 0.40).clamp(0.0, 1.0))
        gV = self._ramp(((0.90 - V) / 0.40).clamp(0.0, 1.0))
        return gT, gC, gV

    def _encode(self, a_base, T, C, V, state):
        feats = [T, C, V, a_base]
        if self.use_action_magnitude:
            feats.append(a_base.abs())
        joint_features = torch.stack(feats, dim=-1)  # (B, 7, joint_in)
        joint_tokens = self.joint_proj(joint_features)
        state_token = self.state_proj(state).unsqueeze(1)
        tokens = torch.cat([state_token, joint_tokens], dim=1) + self.pos_emb
        out = self.final_norm(self.transformer(tokens))
        return out[:, 1:], out[:, 0]

    def forward(self, a_base, temps, state, currents, voltages):
        joint_out, state_out = self._encode(
            a_base, temps, currents, voltages, state)

        # Per-channel gates (per-joint)
        gT, gC, gV = self.per_joint_gates(temps, currents, voltages)

        # Per-channel γ heads → per-joint logits → multiplicative scale
        gT_logit = self.gamma_T_head(joint_out).squeeze(-1)
        gC_logit = self.gamma_C_head(joint_out).squeeze(-1)
        gV_logit = self.gamma_V_head(joint_out).squeeze(-1)
        gamma_T = 1.0 + gT * self.gamma_range * torch.tanh(gT_logit)
        gamma_C = 1.0 + gC * self.gamma_range * torch.tanh(gC_logit)
        gamma_V = 1.0 + gV * self.gamma_range * torch.tanh(gV_logit)
        gamma = gamma_T * gamma_C * gamma_V    # (B, 7)

        # Per-channel δ heads → additive nudges, summed
        dT_logit = self.delta_T_head(joint_out).squeeze(-1)
        dC_logit = self.delta_C_head(joint_out).squeeze(-1)
        dV_logit = self.delta_V_head(joint_out).squeeze(-1)
        delta = (gT * torch.tanh(dT_logit)
                 + gC * torch.tanh(dC_logit)
                 + gV * torch.tanh(dV_logit)) * self.alpha   # (B, 7)

        a_final = (gamma * a_base + delta).clamp(-1.0, 1.0)
        value = self.value_head(state_out).squeeze(-1)
        return a_final, self.log_std, value

    def get_action_and_value(self, a_base, temps, state, currents, voltages, action=None):
        mean, log_std, value = self.forward(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        lp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        return action, lp, ent, value


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = TAMBoTv2(state_dim=19, hidden=128, n_layers=3, n_heads=4,
                 alpha=0.3, gamma_range=0.4, gate_shape="smoothstep")
    print(f"TAM-BoT-v2 params: {count_params(m):,}")

    B = 4
    a = torch.randn(B, 7).clamp(-1, 1)
    s = torch.randn(B, 19)

    # Cool: T=25, C=0.30, V=1.0  →  identity
    T = torch.full((B, 7), 25.0); C = torch.full((B, 7), 0.30); V = torch.ones(B, 7)
    out, _, _ = m(a, T, s, C, V)
    print(f"Cool diff (should be 0): {(out - a).abs().max().item():.2e}")

    # T-only stress: T=68, C=0.30, V=1.0  →  only γ_T should activate
    T = torch.full((B, 7), 68.0); C = torch.full((B, 7), 0.30); V = torch.ones(B, 7)
    out, _, _ = m(a, T, s, C, V)
    print(f"T-only diff (zero-init heads still give 0): {(out - a).abs().max().item():.2e}")

    # Train a few steps with a target action that includes per-channel signal
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    for step in range(30):
        a = torch.randn(B*4, 7).clamp(-1, 1)
        T = torch.rand(B*4, 7) * 50 + 25  # 25-75
        C = torch.rand(B*4, 7) * 0.85 + 0.10  # 0.10-0.95
        V = torch.rand(B*4, 7) * 0.55 + 0.45  # 0.45-1.00
        s = torch.randn(B*4, 19)
        # Mock target: invert simple multiplicative degradation
        Tf = (1 - ((T - 30).clamp(0, 50) / 100)).clamp(0.1, 1)
        Cf = (1 - ((C - 0.30).clamp(0, 0.7) / 1.5)).clamp(0.1, 1)
        Vf = V.clamp(0.3, 1)
        f7 = Tf * Cf * Vf  # (B, 7)
        target = (a / f7.clamp(0.1)).clamp(-1, 1)
        out, _, _ = m(a, T, s, C, V)
        loss = (out - target).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"After 30 steps: loss={loss.item():.4f}")

    # Verify cool still bit-exact identity after training
    a = torch.randn(B, 7).clamp(-1, 1)
    T = torch.full((B, 7), 25.0); C = torch.full((B, 7), 0.30); V = torch.ones(B, 7)
    s = torch.randn(B, 19)
    out, _, _ = m(a, T, s, C, V)
    print(f"Cool diff AFTER training (should be 0): {(out - a).abs().max().item():.2e}")
