"""Acoustic tokenizer settings, independent of the symbolic teacher."""

from dataclasses import dataclass

from tsumugi_mrl.config import ModelConfig


@dataclass
class MelRVQConfig(ModelConfig):
    acoustic_codebooks: int = 8
    acoustic_vocab_size: int = 1_024
    acoustic_codebook_dim: int = 16
    rvq_commitment_weight: float = 0.25
    rvq_reconstruction_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.acoustic_codebook_dim <= 0:
            raise ValueError("acoustic_codebook_dim must be positive.")
        if self.rvq_commitment_weight < 0 or self.rvq_reconstruction_weight < 0:
            raise ValueError("RVQ loss weights must be non-negative.")
