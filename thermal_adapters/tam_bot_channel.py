"""Channel-tokenized architecture ablation with no joint-identity embeddings.

Shares the telemetry gate and supervised objective with the other variants.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from thermal_adapters.sft_arch_variants import BaseSFTAdapter


class ChannelTokenizedAdapter(BaseSFTAdapter):

    # Token order in the sequence (used by attention-extraction plotter):
    #   indices 0..6   -> T_0 .. T_6
    #   indices 7..13  -> C_0 .. C_6
    #   indices 14..20 -> V_0 .. V_6
    #   index 21       -> state
    #   index 22       -> action
    @staticmethod
    def token_labels(n_joints: int = 7) -> list[str]:
        labels  = [f"T{j}" for j in range(n_joints)]
        labels += [f"C{j}" for j in range(n_joints)]
        labels += [f"V{j}" for j in range(n_joints)]
        labels += ["state", "action"]
        return labels

    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_joints: int = 7,
        **kw,
    ):
        super().__init__(**kw)
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.hidden   = hidden
        self.n_joints = int(n_joints)

        # Every TCV scalar uses the SAME projection.  Distinguished only by
        # channel-type embedding — no joint-id leaks in.
        self.scalar_proj   = nn.Linear(1, hidden)
        self.chan_type_emb = nn.Embedding(3, hidden)   # 0=T, 1=C, 2=V

        # State and action tokens with their own type embeddings.
        self.state_proj      = nn.Linear(self.state_dim, hidden)
        self.action_proj     = nn.Linear(self.act_dim, hidden)
        self.state_type_emb  = nn.Parameter(torch.zeros(1, 1, hidden))
        self.action_type_emb = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.state_type_emb,  std=0.02)
        nn.init.normal_(self.action_type_emb, std=0.02)

        self.n_telem_tokens = 3 * self.n_joints
        self.n_tokens       = self.n_telem_tokens + 2
        self.SLOT_STATE     = self.n_telem_tokens
        self.SLOT_ACTION    = self.n_telem_tokens + 1

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu",
            norm_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.final_norm  = nn.LayerNorm(hidden)

        # Delta head reads from the ACTION token, projects to act_dim.
        self.delta_head = nn.Linear(hidden, self.act_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _encode(self, a_base, temps, state, currents, voltages):
        B = a_base.shape[0]
        device = a_base.device

        # 21 TCV tokens: each scalar gets its own slot.
        scalars = torch.cat([temps, currents, voltages], dim=-1).unsqueeze(-1)
        tcv_tok = self.scalar_proj(scalars)                                 # (B, 21, H)

        # Channel-type embedding: T=0 for first 7 slots, C=1 for next 7, V=2.
        type_ids = torch.cat([
            torch.zeros(self.n_joints,                 dtype=torch.long, device=device),
            torch.ones (self.n_joints,                 dtype=torch.long, device=device),
            torch.full((self.n_joints,), 2,            dtype=torch.long, device=device),
        ])
        tcv_tok = tcv_tok + self.chan_type_emb(type_ids).unsqueeze(0)

        s_tok = self.state_proj(state).unsqueeze(1)  + self.state_type_emb
        a_tok = self.action_proj(a_base).unsqueeze(1) + self.action_type_emb

        tokens = torch.cat([tcv_tok, s_tok, a_tok], dim=1)                  # (B, 23, H)
        return self.final_norm(self.transformer(tokens))                    # (B, 23, H)

    def _delta(self, a_base, temps, state, currents, voltages):
        out = self._encode(a_base, temps, state, currents, voltages)
        return self.delta_head(out[:, self.SLOT_ACTION])

    def _value(self, a_base, temps, state, currents, voltages):
        out = self._encode(a_base, temps, state, currents, voltages)
        return self.value_head(out[:, self.SLOT_STATE]).squeeze(-1)


if __name__ == "__main__":
    m = ChannelTokenizedAdapter(state_dim=19, hidden=128, n_layers=2, n_heads=4)
    n = sum(p.numel() for p in m.parameters())
    print(f"ChannelTokenizedAdapter params: {n:,}  tokens: {m.n_tokens}")
    B = 4
    a = torch.randn(B, 7).clamp(-1, 1)
    s = torch.randn(B, 19)
    T = torch.full((B, 7), 25.0); C = torch.full((B, 7), 0.30); V = torch.ones(B, 7)
    mean, ls, val = m(a, T, s, C, V)
    print(f"Cool diff (should be 0): {(mean - a).abs().max().item():.2e}")
    T = torch.full((B, 7), 70.0); C = torch.full((B, 7), 0.95); V = torch.full((B, 7), 0.50)
    mean_hot, _, _ = m(a, T, s, C, V)
    print(f"Hot diff from a_base: {(mean_hot - a).abs().max().item():.2e}")
