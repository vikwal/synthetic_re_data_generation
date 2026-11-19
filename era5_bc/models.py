"""TR-LSTM and TR-Transformer — faithful port of the reference implementation
(https://github.com/kindo/TRWindBC, TR_model/model/{LSTM,Tranformer}.py).

Architecture (paper Fig. 3 + reference code):
- static branch: Linear(n_static, 32) + GELU + Dropout(0.05) -> chi
- dynamic branch: [static replicated over seq ‖ dynamics] -> Linear(-> hidden)
  + summed month/day/hour embeddings (+ sinusoidal positional encoding for the
  Transformer) -> temporal core -> last future_len steps
- head: concat[core output, chi] -> Linear -> Softplus (per target hour)

Transformer core = pre-norm GPT2 blocks with causal attention + final
LayerNorm (reference uses minGPT-style blocks, NOT nn.TransformerEncoder).
"""

import math

import torch
import torch.nn as nn


class TempEmbed(nn.Module):
    """Summed month/day/hour embeddings (Autoformer-style, as in reference)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.m_emb = nn.Embedding(13, hidden_size)
        self.d_emb = nn.Embedding(32, hidden_size)
        self.h_emb = nn.Embedding(24, hidden_size)

    def forward(self, x):  # x: [B, T, 3] = (month, day, hour)
        x = x.long()
        return (self.m_emb(x[..., 0]) + self.d_emb(x[..., 1])
                + self.h_emb(x[..., 2]))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])  # odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, x):
        b, t, h = x.shape
        qkv = (self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2).contiguous().view(b, t, h)


class GPT2Block(nn.Module):
    def __init__(self, hidden_size, nheads, dropout, actf="GELU"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size), getattr(nn, actf)(),
            nn.Linear(4 * hidden_size, hidden_size), nn.Dropout(dropout))
        self.mattn = CausalSelfAttention(hidden_size, nheads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.mattn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TRModel(nn.Module):
    """Two-branch scaling-factor model with swappable temporal core."""

    def __init__(self, core: str, n_dyn: int, n_static: int, hidden: int,
                 static_hidden: int = 32, dropout: float = 0.2,
                 static_dropout: float = 0.05, num_layers: int = 1,
                 nblock: int = 4, nheads: int = 4, future_len: int = 24,
                 actf: str = "GELU"):
        super().__init__()
        assert core in ("lstm", "transformer")
        self.core_type = core
        self.future_len = future_len

        self.mlp_static = nn.Sequential(
            nn.Linear(n_static, static_hidden), getattr(nn, actf)(),
            nn.Dropout(static_dropout))
        self.xd_emb = nn.Linear(n_dyn + n_static, hidden)
        self.temp_emb = TempEmbed(hidden)
        self.fc_out = nn.Linear(hidden + static_hidden, 1)

        if core == "lstm":
            self.lstm = nn.LSTM(hidden, hidden, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
        else:
            self.pos_emb = PositionalEmbedding(hidden)
            self.encoder = nn.Sequential(
                *[GPT2Block(hidden, nheads, dropout, actf) for _ in range(nblock)])
            self.ln_out = nn.LayerNorm(hidden)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, xd, xs, xdate):
        # xd [B, T, n_dyn], xs [B, n_static], xdate [B, T, 3]
        xs_rep = xs.unsqueeze(1).expand(-1, xd.size(1), -1)
        chi = self.mlp_static(xs)

        h = self.xd_emb(torch.cat([xs_rep, xd], dim=-1)) + self.temp_emb(xdate)
        if self.core_type == "lstm":
            h, _ = self.lstm(h)
            out = self.dropout(h[:, -self.future_len:, :])
        else:
            h = h + self.pos_emb(h)
            h = self.ln_out(self.encoder(h))
            out = h[:, -self.future_len:, :]

        chi_rep = chi.unsqueeze(1).expand(-1, self.future_len, -1)
        out = self.fc_out(torch.cat([out, chi_rep], dim=-1)).squeeze(-1)
        return nn.functional.softplus(out)


def build_model(cfg: dict, core: str, overrides: dict | None = None) -> TRModel:
    """Instantiate from config (+ optional HPO overrides)."""
    mc = dict(cfg["model_common"])
    sec = dict(cfg["lstm"] if core == "lstm" else cfg["transformer"])
    if overrides:
        sec.update({k: v for k, v in overrides.items() if k in
                    ("hidden", "dropout", "num_layers", "nblock", "nheads")})
        mc.update({k: v for k, v in overrides.items() if k in
                   ("static_hidden", "static_dropout")})
    return TRModel(
        core=core,
        n_dyn=len(cfg["dynamic_covariates"]),
        n_static=len(cfg["static_covariates"]),
        hidden=int(sec["hidden"]),
        static_hidden=int(mc["static_hidden"]),
        dropout=float(sec["dropout"]),
        static_dropout=float(mc["static_dropout"]),
        num_layers=int(sec.get("num_layers", 1)),
        nblock=int(sec.get("nblock", 4)),
        nheads=int(sec.get("nheads", 4)),
        future_len=int(cfg["windows"]["future_len"]),
        actf=mc["activation"],
    )
