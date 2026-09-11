"""MIDI teacher and its frame reconstruction heads."""

from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from train.config import TrainingConfig
from train.rvq import ResidualVectorQuantizer
from train.symbolic import NO_INSTRUMENT_ID
from tsumugi_mrl.model import ProjectionHead, _checkpoint
from tsumugi_mrl.transformer import Transformer


class SymbolicEncoder(nn.Module):
    """Encoder for Tsumugi-derived MIDI event tokens.

    Token semantics are learned.  token_type_ids merely identify the input
    interface (note, beat, chord, bass, key, instrument, ...).

    The full MIDI event stream has its own length ``L`` and is not forced to
    be 25 Hz.  ``[B, L]`` token IDs become ``[B, L, D_s]`` hidden states;
    only the rows at ``FRAME_t`` positions are later used as frame states.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.use_gradient_checkpoint = bool(config.gradient_checkpointing)
        self.token_embedding = nn.Embedding(config.symbolic_vocab_size, config.symbolic_d_model)
        self.type_embedding = nn.Embedding(config.symbolic_type_vocab_size, config.symbolic_d_model)
        self.instrument_embedding = nn.Embedding(
            config.symbolic_instrument_classes + 1,
            config.symbolic_d_model,
            padding_idx=NO_INSTRUMENT_ID,
        )
        self.encoder = Transformer(
            input_dim=config.symbolic_d_model,
            head_dim=config.symbolic_d_model // config.symbolic_heads,
            num_heads=config.symbolic_heads,
            num_layers=config.symbolic_layers,
            ffn_hidden_size_factor=config.symbolic_dim_feedforward // config.symbolic_d_model,
            dropout=config.dropout,
            output_norm=True,
        )
        self.projection = ProjectionHead(config.symbolic_d_model, config.projection_dim)

    def forward(
        self,
        token_ids: Tensor,
        token_instrument_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence].")

        # [B, L] -> [B, L, D_s].
        if token_instrument_ids.shape != token_ids.shape:
            raise ValueError("token_instrument_ids must match token_ids [batch, sequence].")
        # Pitch/type and instrument are bound in the SAME event vector [B,L,D].
        x = self.token_embedding(token_ids) + self.instrument_embedding(token_instrument_ids)
        if token_type_ids is not None:
            # Type information is another learned [B, L, D_s] embedding.
            x = x + self.type_embedding(token_type_ids)

        attention_mask = None if padding_mask is None else ~padding_mask.bool()
        use_checkpoint = (
            self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()
        )

        def encode(hidden: Tensor) -> Tensor:
            return self.encoder(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # Event-level self-attention with RoPE.  Sequence length remains L;
        # no resampling or pooling is performed inside this Transformer.
        x = _checkpoint(encode, x, enabled=use_checkpoint)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        # Whole-event embedding: [B, L, D_s] -> [B, D_s] -> [B, D_p].
        return x, self.projection(pooled)

    def encode_frame_hidden(
        self,
        token_ids: Tensor,
        anchor_positions: Tensor,
        token_instrument_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return only the hidden states at ``FRAME_t`` positions.

        Event tokens remain in the transformer sequence and therefore affect
        every anchor through self-attention. Only the anchor rows are exposed
        to the frame-level symbolic bottleneck and decoder.
        """

        # Validate indices before running the Transformer.  The checks only
        # depend on the inputs and do not need the encoded hidden states.
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence].")
        if anchor_positions.ndim != 2:
            raise ValueError("anchor_positions must have shape [batch, frames].")
        if anchor_positions.size(0) != token_ids.size(0):
            raise ValueError("anchor_positions batch size must match token_ids.")
        if anchor_positions.numel() and (anchor_positions.min() < 0 or anchor_positions.max() >= token_ids.size(1)):
            raise ValueError("anchor_positions contains an index outside the token sequence.")

        # Encode the complete sequence first so event tokens can communicate
        # with the FRAME_t query rows through self-attention.
        hidden, _ = self(
            token_ids,
            token_instrument_ids=token_instrument_ids,
            token_type_ids=token_type_ids,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )
        # [B, T_s] -> [B, T_s, D_s]; event-token rows are discarded here.
        gather_index = anchor_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        frame_hidden = torch.gather(hidden, dim=1, index=gather_index)
        # Pooling the anchor rows gives one symbolic clip embedding [B, D_p].
        return frame_hidden, self.projection(frame_hidden.mean(dim=1))


class SymbolicFrameDecoder(nn.Module):
    """Simple frame-wise heads used to train the symbolic teacher.

    Every head consumes ``[B, T_s, D_s]``.  The heads retain the frame axis,
    so their targets are aligned to the ``FRAME_t`` anchors rather than to
    the original event-token positions.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        dim = config.symbolic_d_model
        self.note_activity = nn.Linear(dim, config.symbolic_pitch_classes)
        self.note_onset = nn.Linear(dim, config.symbolic_pitch_classes)
        self.note_offset = nn.Linear(dim, config.symbolic_pitch_classes)
        self.instrument_activity = nn.Linear(dim, config.symbolic_instrument_classes)
        self.beat = nn.Linear(dim, 1)
        self.downbeat = nn.Linear(dim, 1)
        self.chord = nn.Linear(dim, config.symbolic_chord_classes)
        self.bass = nn.Linear(dim, config.symbolic_bass_classes)
        self.key = nn.Linear(dim, config.symbolic_key_classes)
        self.meter = nn.Linear(dim, config.symbolic_meter_classes)

    def forward(self, frame_hidden: Tensor) -> dict[str, Tensor]:
        # Outputs are [B, T_s, class_count], except beat/downbeat which are
        # squeezed to [B, T_s].  The default instrument/chord widths are 36
        # and 745 respectively; bass uses 13 pitch/no-bass classes.
        return {
            "note_activity": self.note_activity(frame_hidden),
            "note_onset": self.note_onset(frame_hidden),
            "note_offset": self.note_offset(frame_hidden),
            "instrument_activity": self.instrument_activity(frame_hidden),
            "beat": self.beat(frame_hidden).squeeze(-1),
            "downbeat": self.downbeat(frame_hidden).squeeze(-1),
            "chord": self.chord(frame_hidden),
            "bass": self.bass(frame_hidden),
            "key": self.key(frame_hidden),
            "meter": self.meter(frame_hidden),
        }


@dataclass
class SymbolicTeacherOutput:
    token_hidden: Tensor  # [B, L, D_s]
    frame_hidden: Tensor  # [B, T_s, D_s], gathered FRAME_t rows
    quantized: Tensor  # [B, T_s, D_s], symbolic RVQ output
    codes: Tensor  # [B, T_s, N_musical], integer RVQ codes
    rvq_loss: Tensor  # scalar
    reconstruction: dict[str, Tensor]  # frame-wise outputs, keyed by task
    embedding: Tensor  # [B, D_p]


class SymbolicTeacher(nn.Module, PyTorchModelHubMixin):
    """FRAME-query symbolic encoder, RVQ bottleneck, and frame decoder.

    ``[B, L]`` event tokens are encoded at their original sequence length.
    The ``FRAME_t`` rows are gathered to form ``[B, T_s, D_s]``; only these
    rows pass through the symbolic RVQ and frame decoder.  Thus the decoder
    loss trains each FRAME query to summarize the events around it.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = SymbolicEncoder(config)
        self.rvq = ResidualVectorQuantizer(
            input_dim=config.symbolic_d_model,
            num_codebooks=config.musical_codebooks,
            codebook_size=config.musical_vocab_size,
            commitment_weight=config.rvq_commitment_weight,
            codebook_dim=config.musical_codebook_dim,
            stale_tolerance=config.rvq_stale_tolerance,
        )
        self.decoder = SymbolicFrameDecoder(config)
        self._frozen = False
        self._hub_mixin_config = self.config

    def forward(
        self,
        token_ids: Tensor,
        anchor_positions: Tensor,
        token_instrument_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        frame_padding_mask: Optional[Tensor] = None,
    ) -> SymbolicTeacherOutput:
        # Validate indices before running the symbolic Transformer.
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence].")
        if anchor_positions.ndim != 2:
            raise ValueError("anchor_positions must have shape [batch, frames].")
        if anchor_positions.size(0) != token_ids.size(0):
            raise ValueError("anchor_positions batch size must match token_ids.")
        if anchor_positions.numel() and (anchor_positions.min() < 0 or anchor_positions.max() >= token_ids.size(1)):
            raise ValueError("anchor_positions contains an index outside the token sequence.")

        # 1) Event encoder: [B, L] -> [B, L, D_s].  L is the number of MIDI
        # event tokens, not a fixed 25 Hz sequence length.
        token_hidden, _ = self.encoder(
            token_ids,
            token_instrument_ids=token_instrument_ids,
            token_type_ids=token_type_ids,
            padding_mask=padding_mask,
            position_ids=position_ids,
        )
        # 2) FRAME query extraction: [B, T_s] anchor indices -> [B, T_s, D_s].
        # Event rows affect these states inside the Transformer but are not
        # sent to the symbolic bottleneck or decoder.
        gather_index = anchor_positions.unsqueeze(-1).expand(-1, -1, token_hidden.size(-1))
        frame_hidden = torch.gather(token_hidden, dim=1, index=gather_index)
        # 3) Symbolic RVQ: [B, T_s, D_s] -> quantized [B, T_s, D_s] and
        # integer codes [B, T_s, N_musical].  valid_frame is [B, T_s].
        valid_frame = None if frame_padding_mask is None else ~frame_padding_mask.bool()
        rvq = self.rvq(frame_hidden, valid_mask=valid_frame)
        quantized = rvq.quantized
        if frame_padding_mask is not None:
            # Remove padded FRAME rows before creating the clip embedding.
            quantized = quantized.masked_fill(frame_padding_mask.unsqueeze(-1), 0.0)
            valid = (~frame_padding_mask).unsqueeze(-1).to(dtype=quantized.dtype)
            pooled = (quantized * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            pooled = quantized.mean(dim=1)
        # 4) Clip-level symbolic embedding: [B, T_s, D_s] -> [B, D_s] -> [B, D_p].
        embedding = self.encoder.projection(pooled)
        # 5) Frame reconstruction uses the quantized FRAME states only.  Each
        # returned head keeps the [B, T_s, ...] alignment to symbolic frames.
        return SymbolicTeacherOutput(
            token_hidden=token_hidden,
            frame_hidden=frame_hidden,
            quantized=quantized,
            codes=rvq.codes,
            rvq_loss=rvq.loss,
            reconstruction=self.decoder(quantized),
            embedding=embedding,
        )

    @torch.no_grad()
    def encode(self, *args, **kwargs) -> Tensor:
        """Return fixed symbolic RVQ codes shaped ``[B, T_s, N_musical]``."""

        return self(*args, **kwargs).codes

    def freeze(self, train_projection: bool = True) -> None:
        """Fix RVQ targets while allowing the contrastive projection to learn.

        Pass ``train_projection=False`` for completely frozen inference.
        Call this after loading the separately trained teacher checkpoint.
        """

        self._frozen = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.encoder.projection.parameters():
            parameter.requires_grad_(train_projection)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            # Parent model.train() must not reactivate teacher dropout.
            self.encoder.eval()
            self.rvq.eval()
            self.decoder.eval()
            self.encoder.projection.train(mode and any(p.requires_grad for p in self.encoder.projection.parameters()))
        return self
