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
    perplexity: Tensor | None = None  # [num_codebooks], effective codes in use


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
        stale_tolerance: int = 1_000,
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
        if stale_tolerance <= 0:
            raise ValueError("stale_tolerance must be positive.")

        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.codebook_dim = codebook_dim
        self.stale_tolerance = stale_tolerance

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

        # Consecutive steps each entry has gone unused. An entry that stays
        # unused for stale_tolerance steps is revived from the current batch,
        # which keeps a stage from training against codes it never selects.
        self.register_buffer("stale_counter", torch.zeros(num_codebooks, codebook_size))

    @torch.no_grad()
    def _revive_stale_codes(self, codes: list[Tensor], projections: list[Tensor]) -> None:
        """Reset entries that have gone unused for ``stale_tolerance`` steps.

        Following MuQ, a revived entry is copied from a randomly chosen vector
        of the batch being quantized, so it lands where the data actually is
        instead of in an arbitrary corner of the space. Revived entries were
        by definition not selected this step, so rewriting them here changes
        nothing about the codes already returned.
        """

        for index, (code, projected) in enumerate(zip(codes, projections, strict=True)):
            used = torch.bincount(code.reshape(-1), minlength=self.codebook_size) > 0
            stale = (~used).to(self.stale_counter.dtype)
            self.stale_counter[index] = self.stale_counter[index] * stale + stale

            replace = (self.stale_counter[index] >= self.stale_tolerance).to(self.codebooks.dtype)
            if not bool(replace.any()):
                continue

            encodings = projected.reshape(-1, self.codebook_dim)
            if encodings.size(0) < self.codebook_size:
                repeats = self.codebook_size // encodings.size(0) + 1
                encodings = encodings.repeat(repeats, 1)
            sampled = encodings[torch.randperm(encodings.size(0), device=encodings.device)][: self.codebook_size]
            replace = replace.unsqueeze(-1)
            # Write through .data: the codebook rows are live views inside the
            # graph built above, and an in-place update of the parameter would
            # invalidate them.
            self.codebooks.data[index] = self.codebooks.data[index] * (1 - replace) + sampled * replace
            self.stale_counter[index] = self.stale_counter[index] * (1 - replace.squeeze(-1))

    @torch.no_grad()
    def _nearest_code(self, residual: Tensor, codebook: Tensor) -> Tensor:
        # MuQ uses an L2-normalized codebook lookup. This makes the
        # assignment depend on direction rather than the raw feature scale.
        # The assignment itself is not differentiable, so it stays out of the
        # graph and the codebook can be rewritten afterwards.
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

        # The nearest-code search compares cosine similarities that often
        # differ by less than one percent, and bfloat16 keeps only seven
        # fraction bits. Under autocast those comparisons tie and argmax keeps
        # returning the same entry: on the symbolic teacher one stage fell to a
        # perplexity of 1.5 and half the first-stage codes changed. The
        # quantizer therefore runs in float32 whatever precision the caller
        # uses, which also keeps the pretraining targets identical to the codes
        # the teacher was evaluated with.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._quantize(x.float(), valid_mask)

    def _quantize(self, x: Tensor, valid_mask: Optional[Tensor]) -> RVQOutput:
        def mse(left: Tensor, right: Tensor) -> Tensor:
            squared_error = (left - right).square()
            if valid_mask is None:
                return squared_error.mean()
            weight = valid_mask.bool().unsqueeze(-1).to(dtype=squared_error.dtype)
            return (squared_error * weight).sum() / weight.expand_as(squared_error).sum().clamp_min(1.0)

        residual = x
        quantized = torch.zeros_like(x)
        codes = []
        projections = []
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
            projections.append(projected.detach())
            # The straight-through estimator keeps the projections in the
            # gradient path of the stages that follow.
            q_st = projected + (q - projected).detach()
            expanded = self.output_projections[index](q_st)
            quantized = quantized + expanded
            # The residual keeps its gradient path, so each stage also learns
            # to leave a residual the later stages can quantize.
            residual = residual - expanded

        if self.training:
            self._revive_stale_codes(codes, projections)

        code_tensor = torch.stack(codes, dim=-1)
        perplexity = self.code_perplexity(code_tensor, valid_mask)
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
            perplexity=perplexity,
        )

    @torch.no_grad()
    def code_perplexity(self, codes: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        """Return the effective number of codes each stage used.

        The perplexity is ``exp(entropy)`` of the code distribution in this
        batch, so a stage that spreads its assignments evenly over the whole
        codebook approaches ``codebook_size`` while a collapsed stage
        approaches one. It measures usage, not reconstruction quality, and a
        batch smaller than the codebook caps the value at the token count.
        """

        if codes.size(-1) != self.num_codebooks:
            raise ValueError(f"Expected the last code dimension to be {self.num_codebooks}, got {codes.size(-1)}.")

        perplexities = []
        for index in range(self.num_codebooks):
            stage = codes[..., index]
            stage = stage[valid_mask.bool()] if valid_mask is not None else stage
            counts = torch.bincount(stage.reshape(-1), minlength=self.codebook_size).float()
            probabilities = counts / counts.sum().clamp_min(1.0)
            positive = probabilities[probabilities > 0]
            perplexities.append(torch.exp(-(positive * positive.log()).sum()))
        return torch.stack(perplexities)

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
