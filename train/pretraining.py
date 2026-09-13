"""Three-objective audio pretraining model and inference export."""

from dataclasses import dataclass, fields
from typing import Optional

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from train.config import TrainingConfig
from train.symbolic_teacher.model import SymbolicEncoder, SymbolicTeacher, SymbolicTeacherOutput
from tsumugi_mrl import ModelConfig as AudioModelConfig
from tsumugi_mrl import TsumugiMRLModel
from tsumugi_mrl.model import MaskedAudioEncoder, ProjectionHead


class MultiCodebookHead(nn.Module):
    """Independent classification head for each RVQ codebook.

    Input ``[B, T_a, D_a]`` becomes logits ``[B, T_a, N, K]`` where ``N``
    is the number of codebooks and ``K`` is the vocabulary size of one
    codebook.
    """

    def __init__(self, d_model: int, codebooks: int, vocab_size: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Linear(d_model, vocab_size) for _ in range(codebooks))

    def forward(self, hidden: Tensor) -> Tensor:
        # Each projection classifies the same hidden state independently;
        # stacking on dim=2 adds the codebook axis.
        return torch.stack([projection(hidden) for projection in self.projections], dim=2)


@dataclass
class TsumugiMRLOutput:
    hidden: Tensor  # [B, T_a, D_a]
    acoustic_logits: Tensor  # [B, T_a, N_acoustic, K_acoustic]
    musical_logits: Tensor  # [B, T_a, N_musical, K_musical]
    audio_embedding: Tensor  # [B, D_p]
    symbolic_embedding: Optional[Tensor]  # [B, D_p] when symbolic input is given
    symbolic_codes: Optional[Tensor] = None  # [B, T_s, N_musical]
    symbolic_frame_hidden: Optional[Tensor] = None  # [B, T_s, D_s]
    symbolic_reconstruction: Optional[dict[str, Tensor]] = None  # frame-wise heads
    symbolic_rvq_loss: Optional[Tensor] = None  # scalar


class TsumugiMRLPretrainingModel(nn.Module, PyTorchModelHubMixin):
    """Shared masked audio encoder with three pretraining paths.

    Audio and symbolic data have different sequence axes:
    audio is ``[B, C, S] -> [B, T_a, D_a]`` at about 25 Hz, while MIDI is
    ``[B, L] -> [B, L, D_s]`` at its original event-token length.  The
    symbolic branch gathers ``T_s`` ``FRAME_t`` rows from that event sequence,
    and only those rows become frame-level symbolic targets.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        super().__init__()
        self.config = config or TrainingConfig()
        self.audio_encoder = MaskedAudioEncoder(self.config)
        self.acoustic_head = MultiCodebookHead(
            self.config.d_model,
            self.config.acoustic_codebooks,
            self.config.acoustic_vocab_size,
        )
        self.musical_head = MultiCodebookHead(
            self.config.d_model,
            self.config.musical_codebooks,
            self.config.musical_vocab_size,
        )
        self.audio_projection = ProjectionHead(self.config.d_model, self.config.projection_dim)
        self.symbolic_teacher = SymbolicTeacher(self.config)
        # Keep the full dataclass so save_pretrained() writes every
        # architectural setting needed by from_pretrained().
        self._hub_mixin_config = self.config

    @property
    def symbolic_encoder(self) -> SymbolicEncoder:
        """Compatibility access to the encoder inside the symbolic teacher."""

        return self.symbolic_teacher.encoder

    def set_mel_stats(self, mean: float, std: float) -> None:
        """Set the Mel statistics used by the shared audio encoder."""

        self.audio_encoder.set_mel_stats(mean, std)

    def encode_audio(
        self,
        audio: Optional[Tensor],
        mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        mel_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Return audio hidden states with shape ``[B, T_a, D_a]``."""

        return self.audio_encoder(
            audio,
            mask=mask,
            padding_mask=padding_mask,
            mel_features=mel_features,
        )

    @staticmethod
    def mean_pool(hidden: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pool ``[B, T, D]`` to ``[B, D]`` while ignoring padded rows."""

        if padding_mask is None:
            return hidden.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def encode_symbolic_frames(
        self,
        token_ids: Tensor,
        anchor_positions: Tensor,
        token_instrument_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        frame_padding_mask: Optional[Tensor] = None,
    ) -> SymbolicTeacherOutput:
        """Encode ``[B, L]`` events and gather ``FRAME_t`` rows as ``[B, T_s, D_s]``."""

        return self.symbolic_teacher(
            token_ids,
            anchor_positions=anchor_positions,
            token_instrument_ids=token_instrument_ids,
            token_type_ids=token_type_ids,
            padding_mask=padding_mask,
            position_ids=position_ids,
            frame_padding_mask=frame_padding_mask,
        )

    @torch.no_grad()
    def encode_symbolic_codes(self, *args, **kwargs) -> Tensor:
        """Create fixed frame-level symbolic targets for audio pretraining."""

        return self.symbolic_teacher.encode(*args, **kwargs)

    def forward(
        self,
        audio: Optional[Tensor],
        audio_mask: Optional[Tensor] = None,
        audio_padding_mask: Optional[Tensor] = None,
        symbolic_token_ids: Optional[Tensor] = None,
        symbolic_token_instrument_ids: Optional[Tensor] = None,
        symbolic_token_type_ids: Optional[Tensor] = None,
        symbolic_padding_mask: Optional[Tensor] = None,
        symbolic_position_ids: Optional[Tensor] = None,
        symbolic_anchor_positions: Optional[Tensor] = None,
        symbolic_frame_padding_mask: Optional[Tensor] = None,
        audio_mel: Optional[Tensor] = None,
    ) -> TsumugiMRLOutput:
        # 1) Shared masked audio encoder.
        # [B, C, S] -> frontend [B, T_a, D_mel] -> hidden [B, T_a, D_a].
        hidden = self.encode_audio(
            audio,
            mask=audio_mask,
            padding_mask=audio_padding_mask,
            mel_features=audio_mel,
        )

        # 2) Two frame-wise SSL classifiers read the same audio hidden states.
        # Acoustic targets and musical targets both live on the audio 25 Hz
        # axis, so both outputs keep T_a.
        #   acoustic_logits: [B, T_a, N_acoustic, K_acoustic]
        #   musical_logits:  [B, T_a, N_musical, K_musical]
        acoustic_logits = self.acoustic_head(hidden)
        musical_logits = self.musical_head(hidden)

        # 3) Mean-pool audio frames and project to a unit-norm clip embedding
        # as a clip-level summary: [B, T_a, D_a] -> [B, D_p].
        audio_pooled = self.mean_pool(hidden, audio_padding_mask)
        audio_embedding = self.audio_projection(audio_pooled)

        symbolic_embedding: Optional[Tensor] = None
        symbolic_codes: Optional[Tensor] = None
        symbolic_frame_hidden: Optional[Tensor] = None
        symbolic_reconstruction: Optional[dict[str, Tensor]] = None
        symbolic_rvq_loss: Optional[Tensor] = None
        if symbolic_token_ids is not None:
            if symbolic_token_instrument_ids is None:
                raise ValueError("Symbolic input requires per-event instrument IDs.")
            # 4) Optional symbolic teacher branch.  symbolic_token_ids is
            # [B, L], and L is independent of T_a.  With anchors, the teacher
            # additionally returns [B, T_s, D_s], [B, T_s, N_musical], and
            # frame reconstruction heads.  Without anchors it only provides a
            # whole-sequence clip embedding.
            if symbolic_anchor_positions is None:
                _, symbolic_embedding = self.symbolic_encoder(
                    symbolic_token_ids,
                    token_instrument_ids=symbolic_token_instrument_ids,
                    token_type_ids=symbolic_token_type_ids,
                    padding_mask=symbolic_padding_mask,
                    position_ids=symbolic_position_ids,
                )
            else:
                symbolic_output = self.encode_symbolic_frames(
                    symbolic_token_ids,
                    anchor_positions=symbolic_anchor_positions,
                    token_instrument_ids=symbolic_token_instrument_ids,
                    token_type_ids=symbolic_token_type_ids,
                    padding_mask=symbolic_padding_mask,
                    position_ids=symbolic_position_ids,
                    frame_padding_mask=symbolic_frame_padding_mask,
                )
                symbolic_embedding = symbolic_output.embedding
                symbolic_codes = symbolic_output.codes
                symbolic_frame_hidden = symbolic_output.frame_hidden
                symbolic_reconstruction = symbolic_output.reconstruction
                symbolic_rvq_loss = symbolic_output.rvq_loss

        # Final output combines the audio-frame predictions, clip embedding,
        # and (when supplied) the symbolic-frame teacher outputs.
        return TsumugiMRLOutput(
            hidden=hidden,
            acoustic_logits=acoustic_logits,
            musical_logits=musical_logits,
            audio_embedding=audio_embedding,
            symbolic_embedding=symbolic_embedding,
            symbolic_codes=symbolic_codes,
            symbolic_frame_hidden=symbolic_frame_hidden,
            symbolic_reconstruction=symbolic_reconstruction,
            symbolic_rvq_loss=symbolic_rvq_loss,
        )

    def export_audio_model(self) -> TsumugiMRLModel:
        """Copy audio weights into an independent, teacher-free inference model."""
        config = AudioModelConfig(
            **{field.name: getattr(self.config, field.name) for field in fields(AudioModelConfig)}
        )
        model = TsumugiMRLModel(config)
        model.audio_encoder.load_state_dict(self.audio_encoder.state_dict())
        model.audio_projection.load_state_dict(self.audio_projection.state_dict())
        return model.eval()
