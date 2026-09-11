"""Configuration needed to load and use the audio representation model."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    sample_rate: int = 22_050
    audio_channels: int = 2
    n_mels: int = 128
    n_fft: int = 2_048
    hop_length: int = 441
    temporal_fold: int = 2
    max_audio_seconds: float = 30.0
    conv_channels: int = 512
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2_048
    dropout: float = 0.1
    gradient_checkpointing: bool = False
    projection_dim: int = 256

    def __post_init__(self) -> None:
        if self.sample_rate != 22_050 or self.audio_channels != 2:
            raise ValueError("Tsumugi-MRL expects stereo audio at 22050 Hz.")
        if self.hop_length <= 0 or self.sample_rate % self.hop_length:
            raise ValueError("hop_length must divide sample_rate exactly.")
        if self.temporal_fold not in (1, 2, 4):
            raise ValueError("temporal_fold must be 1, 2 or 4: two convolution blocks divide the time axis.")
        if self.n_mels % 4:
            raise ValueError("n_mels must be a multiple of four: each convolution block halves the frequency axis.")
        if self.conv_channels <= 0:
            raise ValueError("conv_channels must be positive.")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension.")
        if self.dim_feedforward % self.d_model:
            raise ValueError("dim_feedforward must be a multiple of d_model.")

    @property
    def input_frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def encoder_frame_rate(self) -> float:
        return self.input_frame_rate / self.temporal_fold
