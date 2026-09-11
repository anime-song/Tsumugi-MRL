"""Conformer encoder used as the audio backbone.

A Conformer block wraps self-attention in a convolution and a pair of
half-strength feed-forwards. Attention alone has no notion of adjacency, so
the depthwise convolution is what gives every layer a local view of the
sequence; on music that is where onsets and transients live.

The block follows the arrangement MuQ pretrains with, which is the
``Wav2Vec2Conformer`` encoder: macaron feed-forwards, rotary attention, a
GLU-gated depthwise convolution, and a layer norm closing each block.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .transformer import RoPE


class ConformerFeedForward(nn.Module):
    """Position-wise feed-forward, applied twice per block at half strength."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))


class ConformerSelfAttention(nn.Module):
    """Multi-head self-attention with rotary positions."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.rope = RoPE(dim=self.head_dim)
        self.attention_dropout = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch, length, _ = x.shape
        hidden = self.norm(x)

        def heads(projection: nn.Linear) -> Tensor:
            return projection(hidden).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = heads(self.to_q), heads(self.to_k), heads(self.to_v)
        query, key = self.rope(query, key)

        # ``attention_mask`` is True for a key a query may read, the same
        # convention the Transformer in this package uses.
        mask = None if attention_mask is None else attention_mask.bool()[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.dropout(self.to_out(attended))


class ConformerConvolution(nn.Module):
    """Depthwise separable convolution over the sequence.

    The pointwise convolution doubles the width so the GLU can gate it, the
    depthwise convolution mixes neighbouring frames within each channel, and a
    second pointwise convolution projects back.
    """

    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so the convolution preserves the sequence length.")
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, 2 * dim, kernel_size=1, bias=False)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        # [B, T, D] -> [B, D, T]: convolutions run over the sequence axis.
        hidden = self.norm(x).transpose(1, 2)
        if padding_mask is not None:
            # Padded frames are zeroed every block. Left alone they would leak
            # into their neighbours through the kernel and into the batch
            # statistics, which attention avoids by masking its keys.
            hidden = hidden.masked_fill(padding_mask.unsqueeze(1), 0.0)
        hidden = F.glu(self.pointwise_in(hidden), dim=1)
        hidden = self.activation(self.batch_norm(self.depthwise(hidden)))
        hidden = self.dropout(self.pointwise_out(hidden))
        return hidden.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Half feed-forward, attention, convolution, half feed-forward."""

    def __init__(self, dim: int, num_heads: int, ffn_hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(dim, ffn_hidden_dim, dropout)
        self.attention = ConformerSelfAttention(dim, num_heads, dropout)
        self.convolution = ConformerConvolution(dim, kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(dim, ffn_hidden_dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Each feed-forward contributes half a residual, one before attention
        # and one after the convolution: the "macaron" arrangement.
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attention(x, attention_mask)
        x = x + self.convolution(x, padding_mask)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class Conformer(nn.Module):
    """Stack of Conformer blocks: ``[B, T, D] -> [B, T, D]``."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_size_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            ConformerBlock(
                input_dim,
                num_heads,
                input_dim * ffn_hidden_size_factor,
                conv_kernel_size,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask, padding_mask)
        return self.norm(x)


__all__ = [
    "Conformer",
    "ConformerBlock",
    "ConformerConvolution",
    "ConformerFeedForward",
    "ConformerSelfAttention",
]
