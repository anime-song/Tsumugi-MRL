"""Residual vector quantization for discrete acoustic targets."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm


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
    """Greedy residual vector quantizer with factorized codes.

    The input can have any leading dimensions, with the vector dimension last.
    For example, ``[batch, time, dim]`` becomes codes with shape
    ``[batch, time, num_codebooks]``.

    Each codebook quantizes the residual left by the previous codebook. The
    straight-through quantized output is suitable for training an encoder,
    while ``codes`` can be used as discrete prediction targets.

    Following MuQ, every stage owns a pair of weight-normalized projections:
    the residual is projected down to ``codebook_dim`` for the lookup and the
    quantized vector is projected back to ``input_dim``. Matching a
    low-dimensional code improves codebook usage, and the projections let each
    stage choose the subspace it quantizes.
    """

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        codebook_dim: int | None = None,
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
        codebook_dim = input_dim if codebook_dim is None else codebook_dim
        if codebook_dim <= 0:
            raise ValueError("codebook_dim must be positive.")

        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebook_dim = codebook_dim

        # MuQ weight-normalizes both projections, which separates each
        # projection's scale from its direction and keeps a stage from
        # shrinking its own input to make the quantization losses look small.
        self.input_projections = nn.ModuleList(
            weight_norm(nn.Linear(input_dim, codebook_dim)) for _ in range(num_codebooks)
        )
        self.output_projections = nn.ModuleList(
            weight_norm(nn.Linear(codebook_dim, input_dim)) for _ in range(num_codebooks)
        )
        self.codebooks = nn.Parameter(torch.empty(num_codebooks, codebook_size, codebook_dim))
        bound = 1.0 / math.sqrt(codebook_dim)
        nn.init.uniform_(self.codebooks, -bound, bound)

    def _nearest_code(self, residual: Tensor, codebook: Tensor) -> Tensor:
        # MuQ uses an L2-normalized codebook lookup. This makes the
        # assignment depend on direction rather than the raw feature scale.
        flat = F.normalize(residual.reshape(-1, self.codebook_dim), dim=-1)
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

        for index in range(self.num_codebooks):
            codebook = self.codebooks[index]
            # Project the residual into this stage's codebook space, quantize
            # there, and project the result back to the residual space.
            projected = self.input_projections[index](residual)
            code = self._nearest_code(projected, codebook)
            q = F.embedding(code, codebook)

            # Match the released MuQ implementation: normalization is used
            # for nearest-neighbor assignment, while the losses retain the
            # original feature magnitude. Both terms live in the projected
            # space, so the projections train alongside the codebook.
            codebook_losses.append(mse(q, projected.detach()))
            commitment_losses.append(mse(q.detach(), projected))

            codes.append(code)
            # The straight-through estimator keeps the projections in the
            # gradient path of the stages that follow.
            q_st = projected + (q - projected).detach()
            expanded = self.output_projections[index](q_st)
            quantized = quantized + expanded
            # The residual keeps its gradient path, so each stage also learns
            # to leave a residual the later stages can quantize.
            residual = residual - expanded

        code_tensor = torch.stack(codes, dim=-1)
        # Sum over stages rather than averaging: with eight codebooks an
        # average shrinks each stage's gradient as the stack grows deeper.
        codebook_loss = torch.stack(codebook_losses).sum()
        commitment_loss = torch.stack(commitment_losses).sum()
        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Each stage already carries its own straight-through estimator, so the
        # summed output stays differentiable with respect to the projections
        # and any upstream encoder. Wrapping the sum in a second estimator
        # against ``x`` would detach the whole quantizer from a reconstruction
        # loss computed on this output.
        return RVQOutput(
            quantized=quantized,
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
        """Sum projected codebook entries for codes ``[..., num_codebooks]``."""

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
            entry = F.embedding(codes[..., index], codebook)
            quantized = quantized + self.output_projections[index](entry)
        return quantized


__all__ = ["RVQOutput", "ResidualVectorQuantizer"]
