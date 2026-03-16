import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L, :].unsqueeze(0)


class Seq2PointTransformerAllocation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        n_outputs: int,
        center_idx: int,
        dropout: float = 0.1,
        max_len: int = 4096,
        norm_first: bool = True,
        use_time_embedding: bool = True
    ):
        super().__init__()

        self.n_outputs = int(n_outputs)
        self.center_idx = int(center_idx)
        self.use_time_embedding = use_time_embedding

        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        if self.use_time_embedding:
            self.hour_emb = nn.Embedding(24, d_model)
            self.dow_emb = nn.Embedding(7, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=norm_first
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.enc_norm = nn.LayerNorm(d_model)

        self.logit_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_outputs)
        )

    def forward(self, x, total_center, hour_ids=None, dow_ids=None, return_logits: bool = False):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        h = self.in_proj(x)
        h = self.pos_enc(h)

        if self.use_time_embedding:
            if hour_ids is None or dow_ids is None:
                raise ValueError("hour_ids and dow_ids are required when use_time_embedding=True")
            h = h + self.hour_emb(hour_ids) + self.dow_emb(dow_ids)

        h = self.encoder(h)
        h = self.enc_norm(h)

        c = h[:, self.center_idx, :]
        logits = self.logit_head(c)

        alloc = torch.softmax(logits, dim=1)
        y_hat = alloc * total_center.unsqueeze(1).clamp(min=0.0)

        if return_logits:
            return y_hat, logits
        return y_hat