"""Residual vector quantization for discrete acoustic targets."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class RVQOutput:
    """Outputs returned by :class:`ResidualVectorQuantizer`."""

    quantized: Tensor
    codes: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor
    loss: Tensor
    reconstruction_loss: Tensor | None = None


class ResidualVectorQuantizer(nn.Module):
    """Greedy residual vector quantizer.

    The input can have any leading dimensions, with the vector dimension last.
    For example, ``[batch, time, dim]`` becomes codes with shape
    ``[batch, time, num_codebooks]``.

    Each codebook quantizes the residual left by the previous codebook. The
    straight-through quantized output is suitable for training an encoder,
    while ``codes`` can be used as discrete prediction targets.
    """

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be positive.")
        if codebook_size <= 1:
            raise ValueError("codebook_size must be greater than one.")
        if commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative.")

        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight

        self.codebooks = nn.Parameter(torch.empty(num_codebooks, codebook_size, input_dim))
        bound = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.codebooks, -bound, bound)

    def _nearest_code(self, residual: Tensor, codebook: Tensor) -> Tensor:
        # MuQ uses an L2-normalized codebook lookup. This makes the
        # assignment depend on direction rather than the raw feature scale.
        flat = F.normalize(residual.reshape(-1, self.input_dim), dim=-1)
        normalized_codebook = F.normalize(codebook, dim=-1)
        similarity = flat @ normalized_codebook.transpose(0, 1)
        return similarity.argmax(dim=-1).reshape(residual.shape[:-1])

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> RVQOutput:
        if x.ndim < 2:
            raise ValueError("x must have at least one leading dimension and a vector dimension.")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected x.size(-1)={self.input_dim}, got {x.size(-1)}.")
        if valid_mask is not None and valid_mask.shape != x.shape[:-1]:
            raise ValueError(f"valid_mask must have shape {tuple(x.shape[:-1])}, got {tuple(valid_mask.shape)}.")

        def mse(left: Tensor, right: Tensor) -> Tensor:
            squared_error = (left - right).square()
            if valid_mask is None:
                return squared_error.mean()
            weight = valid_mask.bool().unsqueeze(-1).to(dtype=squared_error.dtype)
            return (squared_error * weight).sum() / weight.expand_as(squared_error).sum().clamp_min(1.0)

        residual = x
        quantized = torch.zeros_like(x)
        codes = []
        codebook_losses = []
        commitment_losses = []

        for codebook in self.codebooks:
            code = self._nearest_code(residual, codebook)
            q = F.embedding(code, codebook)

            # Match the released MuQ implementation: normalization is used
            # for nearest-neighbor assignment, while the losses retain the
            # original feature magnitude.
            codebook_losses.append(mse(q, residual.detach()))
            commitment_losses.append(mse(q.detach(), residual))

            codes.append(code)
            quantized = quantized + q
            # Stop earlier codebooks from being updated through later stages.
            residual = residual - q.detach()

        code_tensor = torch.stack(codes, dim=-1)
        codebook_loss = torch.stack(codebook_losses).mean()
        commitment_loss = torch.stack(commitment_losses).mean()
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Forward values are quantized, while the gradient to an upstream
        # encoder follows the identity path.
        quantized_st = x + (quantized - x).detach()
        return RVQOutput(
            quantized=quantized_st,
            codes=code_tensor,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            loss=loss,
        )

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Return discrete codes with shape ``x.shape[:-1] + (N,)``."""

        return self(x).codes

    def decode(self, codes: Tensor) -> Tensor:
        """Sum codebook entries for codes shaped ``[..., num_codebooks]``."""

        if codes.ndim < 1:
            raise ValueError("codes must have at least one dimension.")
        if codes.size(-1) != self.num_codebooks:
            raise ValueError(f"Expected the last code dimension to be {self.num_codebooks}, got {codes.size(-1)}.")

        codes = codes.long()
        quantized = torch.zeros(
            *codes.shape[:-1],
            self.input_dim,
            device=self.codebooks.device,
            dtype=self.codebooks.dtype,
        )
        for index, codebook in enumerate(self.codebooks):
            quantized = quantized + F.embedding(codes[..., index], codebook)
        return quantized


__all__ = ["RVQOutput", "ResidualVectorQuantizer"]
