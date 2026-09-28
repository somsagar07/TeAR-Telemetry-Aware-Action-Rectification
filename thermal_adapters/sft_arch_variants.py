"""SFT-compatible adapter architecture variants.

All variants share the same interface:
    forward(a_base, temps, state, currents, voltages) -> (mean, log_std, value)

and the same hard-threshold telemetry gate so cool conditions are bit-identical
to the base policy.

The adapters here are designed for the SFT (deterministic-inverse) training
target plus optional gated-noise PPO refinement on top. Architectural axes:
  - capacity: width / depth
  - locality: shared MLP vs per-joint
  - coupling: concat vs cross-attention vs FiLM
  - inductive bias: bottleneck (low-rank), MoE (expert routing)
  - input vs output side: hybrid (state + action residual)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ────────────────────────────────────────────────────────────────────────────
#  Shared building blocks
# ────────────────────────────────────────────────────────────────────────────


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return x + F.relu(self.fc(self.norm(x)))


def hard_gate(temps, currents, voltages,
              T_lo=42.0, T_hi=55.0,
              C_lo=0.60, C_hi=1.00,
              V_lo=0.90, V_hi=0.50):
    """Hard ReLU-threshold gate: exactly 0 at healthy telemetry, ramps to 1 at
    severe. OR-combination of three per-channel ramps."""
    tg = ((temps.max(-1, keepdim=True).values - T_lo) / max(1e-6, T_hi - T_lo)).clamp(0.0, 1.0)
    cg = ((currents.max(-1, keepdim=True).values - C_lo) / max(1e-6, C_hi - C_lo)).clamp(0.0, 1.0)
    vg = ((V_lo - voltages.min(-1, keepdim=True).values) / max(1e-6, V_lo - V_hi)).clamp(0.0, 1.0)
    return (tg + cg + vg).clamp(0.0, 1.0)


class BaseSFTAdapter(nn.Module):
    """Common scaffold: gate, log_std, value head, a_base + g*α*tanh(δ) form."""

    def __init__(self, state_dim=23, act_dim=7, alpha=0.3, log_std_init=-2.0):
        super().__init__()
        self.alpha = alpha
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.in_dim = act_dim + 7 + 7 + 7 + state_dim  # a_base + T + C + V + state
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def gate(self, temps, currents, voltages):
        return hard_gate(temps, currents, voltages)

    # Subclasses must implement two methods:
    #   _delta(self, a_base, temps, state, currents, voltages) -> (B, act_dim)
    #   _value(self, a_base, temps, state, currents, voltages) -> (B,)
    # The generic forward composes them with the gate.
    def forward(self, a_base, temps, state, currents, voltages):
        delta = torch.tanh(self._delta(a_base, temps, state, currents, voltages)) * self.alpha
        g = self.gate(temps, currents, voltages)
        mean = (a_base + g * delta).clamp(-1.0, 1.0)
        value = self._value(a_base, temps, state, currents, voltages)
        return mean, self.log_std, value

    # Same gated-noise interface used by train_sft_rl_kl_bct.py.
    def forward_delta(self, a_base, temps, state, currents, voltages):
        delta_logits = self._delta(a_base, temps, state, currents, voltages)
        g = self.gate(temps, currents, voltages)
        value = self._value(a_base, temps, state, currents, voltages)
        return delta_logits, self.log_std, g, value

    def delta_to_action(self, a_base, delta_logits, g):
        return (a_base + g * self.alpha * torch.tanh(delta_logits)).clamp(-1, 1)

    def gated_sample(self, a_base, temps, state, currents, voltages):
        delta_mean, log_std, g, value = self.forward_delta(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(delta_mean)
        dist = Normal(delta_mean, std)
        delta_sample = dist.sample()
        action = self.delta_to_action(a_base, delta_sample, g)
        log_prob = dist.log_prob(delta_sample).sum(-1)
        return action, delta_sample, log_prob, value

    def gated_log_prob(self, a_base, delta_taken, temps, state, currents, voltages):
        delta_mean, log_std, g, value = self.forward_delta(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(delta_mean)
        dist = Normal(delta_mean, std)
        log_prob = dist.log_prob(delta_taken).sum(-1)
        ent = dist.entropy().sum(-1)
        return log_prob, ent, value

    def get_action_and_value(self, a_base, temps, state, currents, voltages, action=None):
        mean, log_std, value = self.forward(a_base, temps, state, currents, voltages)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        lp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        return action, lp, ent, value

    def get_value(self, a_base, temps, state, currents, voltages):
        return self._value(a_base, temps, state, currents, voltages)


# ────────────────────────────────────────────────────────────────────────────
#  Variant 1 — MLP (current best, scaled width / depth)
# ────────────────────────────────────────────────────────────────────────────


class MLPAdapter(BaseSFTAdapter):
    def __init__(self, hidden=128, n_blocks=2, **kw):
        super().__init__(**kw)
        self.proj = nn.Linear(self.in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.mean_head = nn.Linear(hidden, self.act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _delta(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        h = F.relu(self.proj(x))
        for blk in self.blocks:
            h = blk(h)
        return self.mean_head(h)

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Variant 2 — JointWise (per-joint MLPs)
# ────────────────────────────────────────────────────────────────────────────


class JointWiseAdapter(BaseSFTAdapter):
    """One small MLP per output dimension (7 separate networks for 7 actions)."""

    def __init__(self, hidden=64, n_blocks=2, **kw):
        super().__init__(**kw)
        self.per_joint = nn.ModuleList()
        for _ in range(self.act_dim):
            net = nn.Sequential(
                nn.Linear(self.in_dim, hidden),
                nn.ReLU(),
                *[ResBlock(hidden) for _ in range(n_blocks)],
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.per_joint.append(net)
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _delta(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        outs = [net(x) for net in self.per_joint]
        return torch.cat(outs, dim=-1)

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Variant 3 — Bottleneck (low-rank delta)
# ────────────────────────────────────────────────────────────────────────────


class BottleneckAdapter(BaseSFTAdapter):
    """Hidden→bottleneck→delta. Reduces capacity for the delta head, useful as
    a regularizer."""

    def __init__(self, hidden=128, bottleneck=32, n_blocks=2, **kw):
        super().__init__(**kw)
        self.proj = nn.Linear(self.in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.bn = nn.Linear(hidden, bottleneck)
        self.mean_head = nn.Linear(bottleneck, self.act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _delta(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        h = F.relu(self.proj(x))
        for blk in self.blocks:
            h = blk(h)
        h = F.relu(self.bn(h))
        return self.mean_head(h)

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Variant 4 — FiLM (telemetry modulates state features)
# ────────────────────────────────────────────────────────────────────────────


class FiLMAdapter(BaseSFTAdapter):
    """State+a_base passes through an MLP whose hidden activations are
    multiplicatively+additively modulated by telemetry (FiLM)."""

    def __init__(self, hidden=128, n_blocks=2, **kw):
        super().__init__(**kw)
        # Pathway 1: state + a_base
        self.proj = nn.Linear(self.act_dim + self.state_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        # Pathway 2: telemetry → (gamma, beta)
        self.film = nn.Sequential(
            nn.Linear(7 + 7 + 7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.mean_head = nn.Linear(hidden, self.act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _delta(self, a_base, temps, state, currents, voltages):
        sa = torch.cat([a_base, state], dim=-1)
        h = F.relu(self.proj(sa))
        tcv = torch.cat([temps, currents, voltages], dim=-1)
        gamma_beta = self.film(tcv)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        for blk in self.blocks:
            h = blk(h)
        return self.mean_head(h)

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Variant 5 — Cross-Attention (telemetry tokens attend over state tokens)
# ────────────────────────────────────────────────────────────────────────────


class CrossAttnAdapter(BaseSFTAdapter):
    """Tokenize per-joint telemetry (3 channels × 7 joints = 21 tokens) and
    cross-attend with state+a_base tokens. Captures per-joint coupling."""

    def __init__(self, hidden=128, n_heads=4, **kw):
        super().__init__(**kw)
        self.tok_dim = hidden
        # State+a_base → 1 token
        self.state_proj = nn.Linear(self.act_dim + self.state_dim, hidden)
        # Per-joint telemetry tokens: each (T, C, V) → 1 token of size hidden
        self.telem_proj = nn.Linear(3, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.mean_head = nn.Linear(hidden, self.act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _delta(self, a_base, temps, state, currents, voltages):
        # State token (B, 1, H)
        s_tok = self.state_proj(torch.cat([a_base, state], dim=-1)).unsqueeze(1)
        # Telemetry tokens (B, 7, H) — per-joint (T, C, V) triple
        tcv = torch.stack([temps, currents, voltages], dim=-1)  # (B, 7, 3)
        t_tok = self.telem_proj(tcv)
        # Concatenate state token + telem tokens, then self-attend.
        toks = torch.cat([s_tok, t_tok], dim=1)  # (B, 8, H)
        attn_out, _ = self.attn(toks, toks, toks)
        toks = self.norm1(toks + attn_out)
        toks = self.norm2(toks + self.mlp(toks))
        # Use the state token's output for the action delta.
        return self.mean_head(toks[:, 0, :])

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Variant 6 — Hybrid (state-modification AND action-residual)
# ────────────────────────────────────────────────────────────────────────────


class HybridAdapter(BaseSFTAdapter):
    """Two heads: state delta (input-side) AND action delta (output-side).
    State delta is applied to the BC base's input feature dim only as a
    soft signal here (no re-query); action delta is the usual residual.
    Implemented as: predict state-conditioned offset that's added into the
    delta head's input. Effectively a deeper compute path."""

    def __init__(self, hidden=128, n_blocks=2, **kw):
        super().__init__(**kw)
        # State-delta head
        self.state_proj = nn.Linear(self.in_dim, hidden)
        self.state_blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.state_delta_head = nn.Linear(hidden, self.state_dim)
        nn.init.zeros_(self.state_delta_head.weight)
        nn.init.zeros_(self.state_delta_head.bias)
        # Action-delta head (uses original state + state-delta as input)
        self.act_proj = nn.Linear(self.in_dim + self.state_dim, hidden)
        self.act_blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.mean_head = nn.Linear(hidden, self.act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _delta(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        # State delta path
        h = F.relu(self.state_proj(x))
        for blk in self.state_blocks:
            h = blk(h)
        ds = torch.tanh(self.state_delta_head(h)) * 0.1  # bounded state perturbation
        # Action delta path with augmented input
        x2 = torch.cat([x, ds], dim=-1)
        h2 = F.relu(self.act_proj(x2))
        for blk in self.act_blocks:
            h2 = blk(h2)
        return self.mean_head(h2)

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Variant 7 — Mixture of Experts (severity-routed)
# ────────────────────────────────────────────────────────────────────────────


class MoEAdapter(BaseSFTAdapter):
    """Multiple expert MLPs; routing weights from telemetry severity.
    The intuition: different experts specialize in different stress regimes
    (warm vs hot vs near-stall vs brownout)."""

    def __init__(self, hidden=64, n_experts=4, n_blocks=2, **kw):
        super().__init__(**kw)
        self.n_experts = n_experts
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            net = nn.Sequential(
                nn.Linear(self.in_dim, hidden),
                nn.ReLU(),
                *[ResBlock(hidden) for _ in range(n_blocks)],
                nn.Linear(hidden, self.act_dim),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.experts.append(net)
        # Router: telemetry → expert weights (softmax over experts)
        self.router = nn.Sequential(
            nn.Linear(7 + 7 + 7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _delta(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        tcv = torch.cat([temps, currents, voltages], dim=-1)
        weights = F.softmax(self.router(tcv), dim=-1)  # (B, n_experts)
        outs = torch.stack([e(x) for e in self.experts], dim=-2)  # (B, n_experts, act_dim)
        return (weights.unsqueeze(-1) * outs).sum(dim=-2)

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, temps, currents, voltages, state], dim=-1)
        return self.value_head(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────────
#  Registry
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
#  TRANSIC-spirit no-telemetry baseline (state + a_base only, no gate)
#  Same training recipe, same parameter budget, but the network never sees
#  T / C / V and applies its correction unconditionally (gate ≡ 1).
# ────────────────────────────────────────────────────────────────────────────


class NoTelemetryAdapter(BaseSFTAdapter):
    """Learns the correction from (state, a_base) alone. No telemetry, no
    structural gate. The closest in-paper analogue of a TRANSIC-style
    learn-from-correction baseline: it sees the same demonstrations and the
    same severity-augmented targets as TAM, but conditions only on
    proprioceptive state."""

    def __init__(self, hidden=128, n_blocks=2, **kw):
        super().__init__(**kw)
        # input dim: a_base + state (no T/C/V)
        self.in_dim_nt = self.act_dim + self.state_dim
        self.proj = nn.Linear(self.in_dim_nt, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.mean_head = nn.Linear(hidden, self.act_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.value_head = nn.Sequential(
            nn.Linear(self.in_dim_nt, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def gate(self, temps, currents, voltages):
        # Always-on: no telemetry-based gating. The correction is applied at every step.
        return torch.ones_like(temps[..., :1])

    def _delta(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, state], dim=-1)
        h = F.relu(self.proj(x))
        for blk in self.blocks:
            h = blk(h)
        return self.mean_head(h)

    def _value(self, a_base, temps, state, currents, voltages):
        x = torch.cat([a_base, state], dim=-1)
        return self.value_head(x).squeeze(-1)


VARIANT_REGISTRY = {
    # name -> (class, ctor_kwargs)
    "mlp_h128_b2":    (MLPAdapter,        {"hidden": 128, "n_blocks": 2}),
    "mlp_h256_b2":    (MLPAdapter,        {"hidden": 256, "n_blocks": 2}),
    "mlp_h512_b2":    (MLPAdapter,        {"hidden": 512, "n_blocks": 2}),
    "mlp_h128_b4":    (MLPAdapter,        {"hidden": 128, "n_blocks": 4}),
    "mlp_h256_b4":    (MLPAdapter,        {"hidden": 256, "n_blocks": 4}),
    "jointwise_h64":  (JointWiseAdapter,  {"hidden": 64,  "n_blocks": 2}),
    "bottleneck_lr32": (BottleneckAdapter, {"hidden": 128, "bottleneck": 32, "n_blocks": 2}),
    "film_h128":      (FiLMAdapter,       {"hidden": 128, "n_blocks": 2}),
    "crossattn_h128": (CrossAttnAdapter,  {"hidden": 128, "n_heads": 4}),
    "hybrid_h128":    (HybridAdapter,     {"hidden": 128, "n_blocks": 2}),
    "moe4_h64":       (MoEAdapter,        {"hidden": 64,  "n_experts": 4, "n_blocks": 2}),
    "no_telemetry":   (NoTelemetryAdapter, {"hidden": 128, "n_blocks": 2}),  # TRANSIC-spirit
}

# Channel-tokenized variant — destroys per-joint identity (paired ablation
# vs TAM-BoT's per-joint tokenization). Imported lazily to avoid a circular
# import with thermal_adapters.tam_bot_channel.
def _channel_token_adapter(**kwargs):
    from thermal_adapters.tam_bot_channel import ChannelTokenizedAdapter
    return ChannelTokenizedAdapter(**kwargs)

VARIANT_REGISTRY["channel_tok_h128"] = (_channel_token_adapter,
    {"hidden": 128, "n_layers": 2, "n_heads": 4})
VARIANT_REGISTRY["channel_tok_h128_l3"] = (_channel_token_adapter,
    {"hidden": 128, "n_layers": 3, "n_heads": 4})


def build(variant_name, state_dim, act_dim=7, alpha=0.3, log_std_init=-2.0):
    if variant_name not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant '{variant_name}'. Options: "
                         f"{list(VARIANT_REGISTRY)}")
    cls, kw = VARIANT_REGISTRY[variant_name]
    return cls(state_dim=state_dim, act_dim=act_dim, alpha=alpha,
               log_std_init=log_std_init, **kw)
