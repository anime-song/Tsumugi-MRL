"""Teacher and pretraining settings extending the audio inference config."""

from dataclasses import dataclass

from train.mel_rvq.config import MelRVQConfig

from .symbolic import BASS_CLASSES as SYMBOLIC_BASS_CLASSES
from .symbolic import CHORD_CLASSES as SYMBOLIC_CHORD_CLASSES
from .symbolic import INSTRUMENT_CLASSES as SYMBOLIC_INSTRUMENT_CLASSES
from .symbolic import METER_CLASSES as SYMBOLIC_METER_CLASSES
from .symbolic import PITCH_CLASSES as SYMBOLIC_PITCH_CLASSES
from .symbolic import SYMBOLIC_REQUIRED_VOCAB_SIZE


@dataclass
class TrainingConfig(MelRVQConfig):
    """Model and target dimensions.

    Audio uses stereo input at 22,050 Hz. Encoder dimensions and RVQ
    codebook sizes are configured independently for each teacher.
    """

    musical_codebooks: int = 8
    musical_vocab_size: int = 512
    musical_codebook_dim: int = 16

    # The event vocabulary is deliberately fixed and small.  It contains
    # pitch on/off, beat, downbeat, chord, bass, key, meter, instrument,
    # and FRAME tokens.  The hidden symbolic representation is still learned.
    # ``SYMBOLIC_REQUIRED_VOCAB_SIZE`` is exactly the size of that fixed
    # layout, so the embedding table has no unreachable rows.
    symbolic_vocab_size: int = SYMBOLIC_REQUIRED_VOCAB_SIZE
    symbolic_type_vocab_size: int = 11
    symbolic_d_model: int = 256
    symbolic_heads: int = 8
    symbolic_layers: int = 4
    symbolic_dim_feedforward: int = 1_024
    symbolic_frame_rate: float = 25.0
    symbolic_pitch_classes: int = SYMBOLIC_PITCH_CLASSES
    symbolic_instrument_classes: int = SYMBOLIC_INSTRUMENT_CLASSES
    symbolic_chord_classes: int = SYMBOLIC_CHORD_CLASSES
    symbolic_bass_classes: int = SYMBOLIC_BASS_CLASSES
    symbolic_key_classes: int = 25  # 12 major + 12 minor + unknown
    symbolic_meter_classes: int = SYMBOLIC_METER_CLASSES
    symbolic_reconstruction_weight: float = 1.0

    contrastive_temperature: float = 0.07

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.symbolic_vocab_size < SYMBOLIC_REQUIRED_VOCAB_SIZE:
            raise ValueError(
                "symbolic_vocab_size must cover the fixed event-token layout "
                f"and therefore be at least {SYMBOLIC_REQUIRED_VOCAB_SIZE}."
            )
        if self.symbolic_type_vocab_size < 11:
            raise ValueError("symbolic_type_vocab_size must be at least 11.")
        if self.symbolic_frame_rate <= 0:
            raise ValueError("symbolic_frame_rate must be positive.")
        for name in (
            "symbolic_pitch_classes",
            "symbolic_instrument_classes",
            "symbolic_chord_classes",
            "symbolic_bass_classes",
            "symbolic_key_classes",
            "symbolic_meter_classes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.symbolic_reconstruction_weight < 0:
            raise ValueError("symbolic_reconstruction_weight must be non-negative.")
        if self.symbolic_d_model % self.symbolic_heads != 0:
            raise ValueError("symbolic_d_model must be divisible by symbolic_heads.")
        if (self.symbolic_d_model // self.symbolic_heads) % 2 != 0:
            raise ValueError("RoPE requires an even symbolic attention head dimension.")
        if self.symbolic_dim_feedforward % self.symbolic_d_model != 0:
            raise ValueError("symbolic_dim_feedforward must be a multiple of symbolic_d_model.")
