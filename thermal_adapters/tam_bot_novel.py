"""Novel TAM-BoT variants for the architecture push:

  TAM-BoT-XL    — scaled-up TAM-BoT-Large (h=256, 6 layers, ~1.5M params)
  TAM-BoT-FiLM  — per-layer FiLM conditioning on the gate value
                  (severity modulates the transformer's internal computation,
                   not just the final γ/δ output).
  TAM-BoT-MoE   — three severity experts (cool/moderate/severe) softmax-routed
                  by gate magnitude; final γ/δ is a weighted blend.

All preserve the structural cool-condition identity (γ=1, δ=0 when gate=0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3.0 - 2.0 * x)


# ─────────────────────────────────────────────────────────────────────────────
# Shared base: per-joint tokens, per-joint smoothstep gate.
# ─────────────────────────────────────────────────────────────────────────────
class _SharedTAMBase(nn.Module):
    """Per-joint token encoder + smoothstep gate. Subclasses add the backbone
    and output heads."""

    def __init__(self, state_dim=19, act_dim=7, hidden=128):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.n_joints = 7
        self.joint_proj = nn.Linear(4, hidden)
        self.state_proj = nn.Linear(state_dim, hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.n_joints, hidden))
        nn.init.normal_(self.pos_emb, std=0.02)

    def per_joint_gate(self, T, C, V):
        tg = ((T - 42.0) / 13.0).clamp(0.0, 1.0)
        cg = ((C - 0.60) / 0.40).clamp(0.0, 1.0)
        vg = ((0.90 - V) / 0.40).clamp(0.0, 1.0)
        return (smoothstep(tg) + smoothstep(cg) + smoothstep(vg)).clamp(0.0, 1.0)

    def build_tokens(self, a_base, T, C, V, state):
        joint_features = torch.stack([T, C, V, a_base], dim=-1)
        joint_tokens = self.joint_proj(joint_features)
        state_token = self.state_proj(state).unsqueeze(1)
        return torch.cat([state_token, joint_tokens], dim=1) + self.pos_emb


# ─────────────────────────────────────────────────────────────────────────────
# 1. TAM-BoT-XL — scaled-up TAM-BoT-Large
# ─────────────────────────────────────────────────────────────────────────────
class TAMBoTXL(_SharedTAMBase):
    """h=256, 6 layers, 4 heads. Same architecture as TAM-BoT-Large, scaled."""

    def __init__(self, state_dim=19, act_dim=7, hidden=256, n_layers=6,
                 n_heads=4, alpha=0.3, gamma_range=0.5, log_std_init=-2.0):
        super().__init__(state_dim, act_dim, hidden)
        self.alpha = alpha
        self.gamma_range = gamma_range

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu", norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(hidden)

        self.gamma_head = nn.Linear(hidden, 1)
        self.delta_head = nn.Linear(hidden, 1)
        for h in (self.gamma_head, self.delta_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def forward(self, a_base, temps, state, currents, voltages):
        tokens = self.build_tokens(a_base, temps, currents, voltages, state)
        out = self.final_norm(self.transformer(tokens))
        joint_out, state_out = out[:, 1:], out[:, 0]
        gamma_logit = self.gamma_head(joint_out).squeeze(-1)
        delta_logit = self.delta_head(joint_out).squeeze(-1)
        g = self.per_joint_gate(temps, currents, voltages)
        gamma = 1.0 + g * self.gamma_range * torch.tanh(gamma_logit)
        delta = g * self.alpha * torch.tanh(delta_logit)
        a_final = (gamma * a_base + delta).clamp(-1.0, 1.0)
        value = self.value_head(state_out).squeeze(-1)
        return a_final, self.log_std, value

    def get_action_and_value(self, a_base, temps, state, currents, voltages, action=None):
        mean, log_std, value = self.forward(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value


# ─────────────────────────────────────────────────────────────────────────────
# 2. TAM-BoT-FiLM — per-layer FiLM modulation on the gate value
# ─────────────────────────────────────────────────────────────────────────────
class _FiLMTransformerLayer(nn.Module):
    """Standard transformer encoder layer + post-FF FiLM modulation.

    After the standard FF step we apply x ← (1 + g·γ_l) · x + g·β_l where
    γ_l, β_l are functions of the gate value. At gate=0 this reduces to
    identity (γ=0 → 1 multiplier, β=0 → 0 shift), so the FiLM modulation
    *also* preserves cool-condition identity if all heads zero-init."""

    def __init__(self, hidden, n_heads, ctx_dim, film_scale=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads,
                                          batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        # FiLM context → (γ_per_dim, β_per_dim)
        self.film = nn.Linear(ctx_dim, 2 * hidden)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.film_scale = film_scale

    def forward(self, x, ctx, gate_scalar):
        # ctx: (B, ctx_dim);  gate_scalar: (B, 1) ∈ [0, 1] — broadcast across tokens
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.norm2(x)
        x = x + self.ff(h)
        # FiLM modulation gated by severity
        gb = self.film(ctx)                                         # (B, 2H)
        gamma_l, beta_l = gb.chunk(2, dim=-1)                       # each (B, H)
        gamma_l = self.film_scale * torch.tanh(gamma_l)
        beta_l = self.film_scale * torch.tanh(beta_l)
        gate_broadcast = gate_scalar.view(-1, 1, 1)                 # (B,1,1)
        x = x * (1.0 + gate_broadcast * gamma_l.unsqueeze(1)) \
              + gate_broadcast * beta_l.unsqueeze(1)
        return x


class TAMBoTFiLM(_SharedTAMBase):
    """Per-joint transformer with layer-wise FiLM conditioning on gate value.

    The FiLM context is derived from gate features (max/mean across joints
    and channels), so each transformer layer is severity-modulated *internally*
    — attention attends differently in stress regimes vs cool."""

    def __init__(self, state_dim=19, act_dim=7, hidden=128, n_layers=4,
                 n_heads=4, alpha=0.3, gamma_range=0.5, log_std_init=-2.0,
                 ctx_dim=16, film_scale=0.4):
        super().__init__(state_dim, act_dim, hidden)
        self.alpha = alpha
        self.gamma_range = gamma_range

        # Context encoder: gate features → ctx_dim
        # We feed (max_T, mean_T, max_C, mean_C, min_V, mean_V) per channel.
        self.ctx_encoder = nn.Sequential(
            nn.Linear(6, ctx_dim * 2), nn.GELU(),
            nn.Linear(ctx_dim * 2, ctx_dim),
        )

        self.layers = nn.ModuleList([
            _FiLMTransformerLayer(hidden, n_heads, ctx_dim, film_scale)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)

        self.gamma_head = nn.Linear(hidden, 1)
        self.delta_head = nn.Linear(hidden, 1)
        for h in (self.gamma_head, self.delta_head):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def _ctx_features(self, T, C, V):
        # (B, 6): six summary stats
        f = torch.stack([
            T.max(-1).values, T.mean(-1),
            C.max(-1).values, C.mean(-1),
            V.min(-1).values, V.mean(-1),
        ], dim=-1)
        return self.ctx_encoder(f)  # (B, ctx_dim)

    def _scalar_gate(self, T, C, V):
        g = self.per_joint_gate(T, C, V)             # (B, 7)
        return g.max(dim=-1, keepdim=True).values    # (B, 1) — fires when any joint stressed

    def forward(self, a_base, temps, state, currents, voltages):
        tokens = self.build_tokens(a_base, temps, currents, voltages, state)
        ctx = self._ctx_features(temps, currents, voltages)
        gate_scalar = self._scalar_gate(temps, currents, voltages)
        x = tokens
        for layer in self.layers:
            x = layer(x, ctx, gate_scalar)
        out = self.final_norm(x)
        joint_out, state_out = out[:, 1:], out[:, 0]
        gamma_logit = self.gamma_head(joint_out).squeeze(-1)
        delta_logit = self.delta_head(joint_out).squeeze(-1)
        g = self.per_joint_gate(temps, currents, voltages)
        gamma = 1.0 + g * self.gamma_range * torch.tanh(gamma_logit)
        delta = g * self.alpha * torch.tanh(delta_logit)
        a_final = (gamma * a_base + delta).clamp(-1.0, 1.0)
        value = self.value_head(state_out).squeeze(-1)
        return a_final, self.log_std, value

    def get_action_and_value(self, a_base, temps, state, currents, voltages, action=None):
        mean, log_std, value = self.forward(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value


# ─────────────────────────────────────────────────────────────────────────────
# 3. TAM-BoT-MoE — severity-expert mixture
# ─────────────────────────────────────────────────────────────────────────────
class TAMBoTMoE(_SharedTAMBase):
    """Three severity experts (cool/moderate/severe), each with its own γ/δ
    heads, softmax-routed by gate magnitude.

    Routing weight depends only on the gate scalar (max joint gate). Each
    expert is a small head off the shared transformer output:
        w = softmax_T(α_e · g + b_e)
        γ = Σ_e w_e · γ_e_logit
        δ = Σ_e w_e · δ_e_logit
    """

    def __init__(self, state_dim=19, act_dim=7, hidden=128, n_layers=3,
                 n_heads=4, alpha=0.3, gamma_range=0.5, log_std_init=-2.0,
                 n_experts=3, route_temperature=1.0):
        super().__init__(state_dim, act_dim, hidden)
        self.alpha = alpha
        self.gamma_range = gamma_range
        self.n_experts = n_experts
        self.route_temperature = route_temperature

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu", norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(hidden)

        # Per-expert γ/δ heads
        self.gamma_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(n_experts)])
        self.delta_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(n_experts)])
        for h in list(self.gamma_heads) + list(self.delta_heads):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

        # Router: maps scalar gate → expert weights
        # Initialise so the cool expert dominates at g=0 (smooth ramp at higher g)
        # Three anchors: cool at g=0, moderate at g≈0.5, severe at g≈1
        anchors = torch.linspace(0.0, 1.0, n_experts).view(-1, 1)
        # Routing logits: -(g - anchor)^2 / temperature
        self.register_buffer("anchors", anchors.view(1, n_experts))

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def route(self, gate_scalar):
        # gate_scalar: (B, 1). Compute expert weights via -dist² softmax.
        # at g=0 expert 0 wins; at g=1 expert n-1 wins.
        d2 = (gate_scalar.view(-1, 1) - self.anchors).pow(2)         # (B, n_experts)
        logits = -d2 / max(1e-6, self.route_temperature)
        return F.softmax(logits, dim=-1)                              # (B, n_experts)

    def forward(self, a_base, temps, state, currents, voltages):
        tokens = self.build_tokens(a_base, temps, currents, voltages, state)
        out = self.final_norm(self.transformer(tokens))
        joint_out, state_out = out[:, 1:], out[:, 0]

        # Per-expert γ/δ logits, then mix via routing weights
        g_full = self.per_joint_gate(temps, currents, voltages)       # (B, 7)
        gate_scalar = g_full.max(dim=-1, keepdim=True).values         # (B, 1)
        w = self.route(gate_scalar)                                    # (B, n_experts)

        gamma_logit = torch.zeros_like(g_full)
        delta_logit = torch.zeros_like(g_full)
        for e in range(self.n_experts):
            ge = self.gamma_heads[e](joint_out).squeeze(-1)            # (B, 7)
            de = self.delta_heads[e](joint_out).squeeze(-1)
            gamma_logit = gamma_logit + w[:, e:e+1] * ge
            delta_logit = delta_logit + w[:, e:e+1] * de

        gamma = 1.0 + g_full * self.gamma_range * torch.tanh(gamma_logit)
        delta = g_full * self.alpha * torch.tanh(delta_logit)
        a_final = (gamma * a_base + delta).clamp(-1.0, 1.0)
        value = self.value_head(state_out).squeeze(-1)
        return a_final, self.log_std, value

    def get_action_and_value(self, a_base, temps, state, currents, voltages, action=None):
        mean, log_std, value = self.forward(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    a = torch.randn(B, 7).clamp(-1, 1)
    s = torch.randn(B, 19)
    T_cool = torch.full((B, 7), 25.0); C_cool = torch.full((B, 7), 0.30); V_cool = torch.ones(B, 7)
    T_hot = torch.full((B, 7), 70.0); C_hot = torch.full((B, 7), 0.95); V_hot = torch.full((B, 7), 0.50)

    for name, cls in [("XL", TAMBoTXL), ("FiLM", TAMBoTFiLM), ("MoE", TAMBoTMoE)]:
        m = cls(state_dim=19)
        params = count_params(m)
        out, _, _ = m(a, T_cool, s, C_cool, V_cool)
        cool_diff = (out - a).abs().max().item()
        out, _, _ = m(a, T_hot, s, C_hot, V_hot)
        hot_diff = (out - a).abs().max().item()
        print(f"{name:6s}  params={params:>9,}  cool_diff={cool_diff:.2e}  hot_diff={hot_diff:.2e}")

    # Train a few steps to verify cool-identity holds after gradient updates
    for name, cls in [("XL", TAMBoTXL), ("FiLM", TAMBoTFiLM), ("MoE", TAMBoTMoE)]:
        m = cls(state_dim=19)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        for _ in range(30):
            T = torch.rand(B, 7) * 50 + 25; C = torch.rand(B, 7) * 0.85 + 0.10
            V = torch.rand(B, 7) * 0.55 + 0.45
            tgt = (a * 1.3).clamp(-1, 1)
            out, _, _ = m(a, T, s, C, V)
            loss = (out - tgt).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        out, _, _ = m(a, T_cool, s, C_cool, V_cool)
        post_cool_diff = (out - a).abs().max().item()
        print(f"{name:6s}  cool_diff AFTER 30 steps (should be 0): {post_cool_diff:.2e}")
