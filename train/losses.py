"""Losses for the three Tsumugi-MRL pretraining paths."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .symbolic import SymbolicFrameTargets


def masked_multicodebook_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Cross entropy for logits [B, T, N, K] and targets [B, T, N]."""

    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected logits [B,T,N,K] and targets [B,T,N].")
    if logits.shape[:3] != targets.shape:
        raise ValueError(f"Logit/target shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}.")

    batch, time, codebooks, classes = logits.shape
    loss = F.cross_entropy(
        logits.reshape(batch * time * codebooks, classes),
        targets.reshape(batch * time * codebooks),
        reduction="none",
    ).reshape(batch, time, codebooks)

    active = torch.ones_like(loss, dtype=torch.bool)
    if mask is not None:
        active = active & mask.unsqueeze(-1).bool()
    if valid_mask is not None:
        if valid_mask.ndim == 2:
            active = active & valid_mask.unsqueeze(-1).bool()
        else:
            active = active & valid_mask.bool()

    active = active.to(dtype=loss.dtype)
    return (loss * active).sum() / active.sum().clamp_min(1.0)


def symmetric_info_nce(
    audio_embedding: Tensor,
    symbolic_embedding: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric in-batch InfoNCE for aligned audio/MIDI segments."""

    if audio_embedding.ndim != 2 or symbolic_embedding.ndim != 2:
        raise ValueError("Embeddings must have shape [batch, dimension].")
    if audio_embedding.shape != symbolic_embedding.shape:
        raise ValueError(
            "Audio and symbolic embeddings must have identical shape, "
            f"got {tuple(audio_embedding.shape)} and {tuple(symbolic_embedding.shape)}."
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    audio_embedding = F.normalize(audio_embedding, dim=-1)
    symbolic_embedding = F.normalize(symbolic_embedding, dim=-1)
    logits = audio_embedding @ symbolic_embedding.transpose(0, 1) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def symbolic_frame_reconstruction_loss(
    predictions: dict[str, Tensor],
    targets: SymbolicFrameTargets,
    frame_padding_mask: Optional[Tensor] = None,
    binary_pos_weights: Optional[dict[str, Tensor]] = None,
    categorical_class_counts: Optional[dict[str, Tensor]] = None,
    balanced_softmax_tau: float = 0.0,
) -> dict[str, Tensor]:
    """Reconstruct MIDI information from the ``FRAME_t`` rows.

    The loss is intentionally frame-wise.  It is the mechanism that makes a
    frame anchor collect the events belonging to its own time position before
    the hidden row is compressed by the symbolic RVQ.
    """

    if frame_padding_mask is None:
        valid_frame = torch.ones_like(targets.beat, dtype=torch.bool)
    else:
        valid_frame = ~frame_padding_mask.bool()

    def binary(name: str, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        pos_weight = None if binary_pos_weights is None else binary_pos_weights.get(name)
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=predictions[name].device, dtype=predictions[name].dtype)
        loss = F.binary_cross_entropy_with_logits(
            predictions[name],
            target,
            reduction="none",
            pos_weight=pos_weight,
        )
        active = valid_frame
        while active.ndim < loss.ndim:
            active = active.unsqueeze(-1)
        if mask is not None:
            active = active & mask.bool()
        # Average over valid elements, including pitch/instrument classes.
        active = active.expand_as(loss).to(dtype=loss.dtype)
        return (loss * active).sum() / active.sum().clamp_min(1.0)

    def categorical(name: str, target: Tensor) -> Tensor:
        logits = predictions[name]
        if categorical_class_counts is not None and name in categorical_class_counts:
            counts = categorical_class_counts[name].to(device=logits.device, dtype=logits.dtype)
            logits = logits + balanced_softmax_tau * counts.clamp_min(1.0).log()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(target.shape)
        active = (target != -100) & valid_frame
        active = active.to(dtype=loss.dtype)
        return (loss * active).sum() / active.sum().clamp_min(1.0)

    result = {
        "loss_note_activity": binary("note_activity", targets.note_activity),
        "loss_note_onset": binary("note_onset", targets.note_onset),
        "loss_note_offset": binary("note_offset", targets.note_offset),
        "loss_instrument_activity": binary("instrument_activity", targets.instrument_activity),
        "loss_beat": binary("beat", targets.beat),
        "loss_downbeat": binary("downbeat", targets.downbeat),
        "loss_chord": categorical("chord", targets.chord),
        "loss_bass": categorical("bass", targets.bass),
        "loss_key": categorical("key", targets.key),
        "loss_meter": categorical("meter", targets.meter),
    }
    result["loss_total"] = sum(result.values())
    return result


class SymbolicTeacherLoss(nn.Module):
    """Loss used in the short, separate symbolic-teacher pretraining stage."""

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        rvq_weight: float = 1.0,
        binary_pos_weights: Optional[dict[str, Tensor]] = None,
        categorical_class_counts: Optional[dict[str, Tensor]] = None,
        balanced_softmax_tau: float = 0.0,
    ) -> None:
        super().__init__()
        if reconstruction_weight < 0 or rvq_weight < 0 or balanced_softmax_tau < 0:
            raise ValueError("symbolic teacher loss weights must be non-negative.")
        self.reconstruction_weight = reconstruction_weight
        self.rvq_weight = rvq_weight
        self.binary_pos_weights = binary_pos_weights
        self.categorical_class_counts = categorical_class_counts
        self.balanced_softmax_tau = balanced_softmax_tau

    def forward(
        self,
        output,
        targets: SymbolicFrameTargets,
        frame_padding_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if output.reconstruction is None:
            raise ValueError("Symbolic teacher output has no reconstruction predictions.")
        result = symbolic_frame_reconstruction_loss(
            output.reconstruction,
            targets,
            frame_padding_mask=frame_padding_mask,
            binary_pos_weights=self.binary_pos_weights,
            categorical_class_counts=self.categorical_class_counts,
            balanced_softmax_tau=self.balanced_softmax_tau,
        )
        result["loss_reconstruction"] = result.pop("loss_total")
        result["loss_rvq"] = output.rvq_loss
        result["loss_total"] = (
            self.reconstruction_weight * result["loss_reconstruction"] + self.rvq_weight * result["loss_rvq"]
        )
        return result


class PretrainingLoss(nn.Module):
    """Weighted sum of acoustic MLM, musical MLM, and Audio--MIDI contrastive loss."""

    def __init__(
        self,
        acoustic_weight: float = 1.0,
        musical_weight: float = 0.5,
        contrastive_weight: float = 0.2,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.acoustic_weight = acoustic_weight
        self.musical_weight = musical_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(
        self,
        output,
        acoustic_targets: Tensor,
        musical_targets: Tensor,
        acoustic_mask: Optional[Tensor] = None,
        musical_mask: Optional[Tensor] = None,
        acoustic_valid_mask: Optional[Tensor] = None,
        musical_valid_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        acoustic = masked_multicodebook_cross_entropy(
            output.acoustic_logits,
            acoustic_targets,
            mask=acoustic_mask,
            valid_mask=acoustic_valid_mask,
        )
        musical = masked_multicodebook_cross_entropy(
            output.musical_logits,
            musical_targets,
            mask=musical_mask,
            valid_mask=musical_valid_mask,
        )

        total = self.acoustic_weight * acoustic + self.musical_weight * musical
        result = {
            "loss_acoustic": acoustic,
            "loss_musical": musical,
        }

        if output.symbolic_embedding is not None:
            contrastive = symmetric_info_nce(
                output.audio_embedding,
                output.symbolic_embedding,
                temperature=self.temperature,
            )
            total = total + self.contrastive_weight * contrastive
            result["loss_contrastive"] = contrastive

        result["loss_total"] = total
        return result
