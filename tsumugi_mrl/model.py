"""Audio representations for inference and downstream task training."""

from collections.abc import Callable
from typing import Optional

import torch
import torchaudio
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .conformer import Conformer
from .config import ModelConfig


def _checkpoint(function: Callable[[Tensor], Tensor], x: Tensor, *, enabled: bool) -> Tensor:
    if not enabled:
        return function(x)
    return checkpoint(function, x, use_reentrant=False)


class StereoMelFrontend(nn.Module):
    """Convert stereo audio into normalized, folded 25 Hz Mel tokens.

    ``mel_mean`` and ``mel_std`` are persistent buffers so the exact
    dataset normalization used by a tokenizer is saved with its checkpoint.
    They default to identity normalization until dataset statistics are set.

    With the default configuration, ``D_mel = 2 * 2 * 128 = 512`` and
    ``T_a`` is approximately 25 tokens per second.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.audio_channels = config.audio_channels
        self.n_mels = config.n_mels
        self.temporal_fold = config.temporal_fold
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            center=False,
            power=2.0,
        )
        # torchaudio's Hann window can have a few bytes of extra storage,
        # which safetensors rejects as a shared/partial tensor.  It is a
        # deterministic buffer, so a compact clone is sufficient and is
        # recreated from the saved ModelConfig when loading from the Hub.
        self.mel.spectrogram.window = self.mel.spectrogram.window.clone()
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.output_rate = config.encoder_frame_rate
        self.register_buffer("mel_mean", torch.zeros(1), persistent=True)
        self.register_buffer("mel_std", torch.ones(1), persistent=True)

    @property
    def mel_stats(self) -> tuple[float, float]:
        """Return the saved Mel mean and standard deviation."""

        return self.mel_mean.item(), self.mel_std.item()

    def set_mel_stats(self, mean: float, std: float) -> None:
        """Set the dataset-level Mel normalization statistics."""

        mean_tensor = torch.as_tensor(mean, dtype=self.mel_mean.dtype, device=self.mel_mean.device)
        std_tensor = torch.as_tensor(std, dtype=self.mel_std.dtype, device=self.mel_std.device)
        if mean_tensor.numel() != 1 or not torch.isfinite(mean_tensor).all():
            raise ValueError("mel mean must be one finite scalar.")
        if std_tensor.numel() != 1 or not torch.isfinite(std_tensor).all() or std_tensor.item() <= 0:
            raise ValueError("mel std must be one finite positive scalar.")
        self.mel_mean.copy_(mean_tensor.reshape_as(self.mel_mean))
        self.mel_std.copy_(std_tensor.reshape_as(self.mel_std))

    def forward(self, audio: Tensor) -> Tensor:
        if audio.ndim != 3:
            raise ValueError("audio must have shape [batch, channels, samples].")
        if audio.size(1) != self.audio_channels:
            raise ValueError(f"Expected {self.audio_channels} audio channels, got {audio.size(1)}.")

        # [B, C, S] -> [B, C, n_mels, F_mel].  At 22,050 Hz with hop_length
        # 441, the un-folded Mel sequence is approximately 50 frames/second.
        mel = self.to_db(self.mel(audio))
        batch, channels, n_mels, frames = mel.shape
        # Fold adjacent Mel frames into the feature dimension instead of
        # averaging them. This keeps the local temporal detail available to
        # the learned input projection, while reducing the token rate.
        # [B, C, n_mels, F_mel] -> [B, F_mel, C * n_mels]
        mel = mel.permute(0, 3, 1, 2).reshape(batch, frames, channels * n_mels)
        usable_frames = frames - frames % self.temporal_fold
        if usable_frames == 0:
            raise ValueError("The audio is too short to form one folded Mel token.")
        mel = mel[:, :usable_frames]
        # [B, F_mel, C * n_mels] ->
        # [B, T_a, temporal_fold, C * n_mels], where
        # T_a = floor(F_mel / temporal_fold).
        mel = mel.reshape(
            batch,
            usable_frames // self.temporal_fold,
            self.temporal_fold,
            channels * n_mels,
        )
        mel = mel.reshape(
            batch,
            usable_frames // self.temporal_fold,
            self.temporal_fold * channels * n_mels,
        )  # [B, T_a, D_mel], D_mel = temporal_fold * C * n_mels
        return (mel - self.mel_mean) / self.mel_std.clamp_min(1e-6)

    def unfold(self, features: Tensor) -> Tensor:
        """Undo the temporal fold: ``[B, T_a, D_mel] -> [B, C, n_mels, F_mel]``.

        The folded view is what the acoustic teacher quantizes and what the
        feature cache stores, so the frontend keeps producing it. The
        convolutional subsampling wants the spectrogram back; folding is a
        pure reshape, so nothing is lost in either direction.
        """

        expected = self.temporal_fold * self.audio_channels * self.n_mels
        if features.ndim != 3 or features.size(-1) != expected:
            raise ValueError(f"features must have shape [batch, tokens, {expected}], got {tuple(features.shape)}.")

        batch, tokens, _ = features.shape
        # forward() packs the feature axis as (fold, channel, mel), so reading
        # it back in that order and moving time last recovers the spectrogram.
        features = features.reshape(batch, tokens, self.temporal_fold, self.audio_channels, self.n_mels)
        features = features.permute(0, 3, 4, 1, 2)
        return features.reshape(batch, self.audio_channels, self.n_mels, tokens * self.temporal_fold)


class ResidualConvBlock(nn.Module):
    """Two 3x3 convolutions over a Mel spectrogram, with a strided skip path.

    The block halves the frequency axis and divides the time axis by
    ``time_stride``. Both paths are strided, so the skip connection is a
    projection rather than an identity.
    """

    def __init__(self, in_channels: int, out_channels: int, time_stride: int) -> None:
        super().__init__()
        stride = (2, time_stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.skip_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.activation(self.norm1(self.conv1(x)))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(hidden + self.skip_norm(self.skip(x)))


class Conv2dSubsampling(nn.Module):
    """Turn a Mel spectrogram into encoder tokens.

    ``[B, C, n_mels, F_mel] -> [B, F_mel // temporal_fold, D_a]``.

    Two residual blocks quarter the frequency axis and divide the time axis
    by ``temporal_fold``; the surviving frequency bands are then flattened
    into one vector per token. Compared with a linear projection of folded
    Mel frames, this gives each token a local receptive field over both time
    and frequency before any attention runs, which is how MuQ reads its
    spectrogram.
    """

    def __init__(
        self,
        audio_channels: int,
        n_mels: int,
        hidden_channels: int,
        d_model: int,
        temporal_fold: int,
    ) -> None:
        super().__init__()
        if n_mels % 4:
            raise ValueError("n_mels must be a multiple of four; each block halves the frequency axis.")
        if temporal_fold not in (1, 2, 4):
            raise ValueError("temporal_fold must be 1, 2 or 4 so two blocks can divide the time axis.")

        # Spread the time reduction over the two blocks: 4 -> (2, 2), 2 -> (2, 1).
        first_stride = min(temporal_fold, 2)
        self.blocks = nn.Sequential(
            ResidualConvBlock(audio_channels, hidden_channels, first_stride),
            ResidualConvBlock(hidden_channels, hidden_channels, temporal_fold // first_stride),
        )
        self.projection = nn.Linear(hidden_channels * (n_mels // 4), d_model)

    def forward(self, mel: Tensor) -> Tensor:
        if mel.ndim != 4:
            raise ValueError("mel must have shape [batch, channels, mels, frames].")

        # [B, C, n_mels, F_mel] -> [B, hidden, n_mels // 4, T_a]
        hidden = self.blocks(mel)
        # Every band of every channel contributes to the token: [B, T_a, hidden * n_mels // 4]
        hidden = hidden.permute(0, 3, 1, 2).flatten(2)
        return self.projection(hidden)


class MaskedAudioEncoder(nn.Module):
    """MuQ-style masked audio encoder for stereo 22.05 kHz audio.

    The frontend creates a 50 Hz Mel spectrogram, the convolutional
    subsampling turns it into 25 Hz tokens, and the Conformer maps those to
    shared hidden states ``[B, T_a, D_a]``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_gradient_checkpoint = bool(config.gradient_checkpointing)
        self.temporal_fold = config.temporal_fold
        self.frontend = StereoMelFrontend(config)
        self.subsampling = Conv2dSubsampling(
            audio_channels=config.audio_channels,
            n_mels=config.n_mels,
            hidden_channels=config.conv_channels,
            d_model=config.d_model,
            temporal_fold=config.temporal_fold,
        )
        # One learned Mel column replaces a masked frame. It has to be applied
        # before the convolution: afterwards a token has already mixed in its
        # neighbours, so what the model is asked to predict would be part of
        # its own input.
        self.mask_token = nn.Parameter(torch.zeros(config.audio_channels, config.n_mels))
        self.encoder = Conformer(
            input_dim=config.d_model,
            num_heads=config.n_heads,
            num_layers=config.num_layers,
            ffn_hidden_size_factor=config.dim_feedforward // config.d_model,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
        )

    def set_mel_stats(self, mean: float, std: float) -> None:
        """Use the same Mel normalization as the pretrained tokenizer."""

        self.frontend.set_mel_stats(mean, std)

    def forward(
        self,
        audio: Optional[Tensor],
        mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        mel_features: Optional[Tensor] = None,
    ) -> Tensor:
        # [B, C, S] -> [B, T_a, D_mel] -> [B, T_a, D_a].
        # Pretraining can pass the Mel features already computed for the
        # acoustic teacher so the expensive STFT is not repeated. Those are
        # stored folded, which ``unfold`` reverses exactly.
        if mel_features is None:
            if audio is None:
                raise ValueError("audio is required when mel_features is not provided.")
            x = self.frontend(audio)
        else:
            x = mel_features
        # [B, T_a, D_mel] -> [B, C, n_mels, F_mel] at the pre-subsampling rate.
        mel = self.frontend.unfold(x)

        if mask is not None:
            # ``mask`` is [B, T_a], one flag per output token; each token covers
            # ``temporal_fold`` Mel frames. Targets stay unchanged.
            tokens = mel.size(-1) // self.temporal_fold
            if mask.shape != (mel.size(0), tokens):
                raise ValueError(f"mask must have shape {(mel.size(0), tokens)}, got {tuple(mask.shape)}.")
            frame_mask = mask.bool().repeat_interleave(self.temporal_fold, dim=1)
            mel = torch.where(
                frame_mask[:, None, None, :],
                self.mask_token[None, :, :, None],
                mel,
            )

        # [B, C, n_mels, F_mel] -> [B, T_a, D_a].
        x = self.subsampling(mel)

        # Transformer.py expects True for an allowed key position, while this
        # module exposes the usual padding convention: True means padding.
        attention_mask = None if padding_mask is None else ~padding_mask.bool()
        use_checkpoint = self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()

        def encode(hidden: Tensor) -> Tensor:
            return self.encoder(hidden, attention_mask=attention_mask)

        # Conformer: [B, T_a, D_a] -> [B, T_a, D_a].
        x = _checkpoint(encode, x, enabled=use_checkpoint)
        if padding_mask is not None:
            # Keep padded rows from entering pooling or contrastive learning.
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class ProjectionHead(nn.Module):
    """Map pooled hidden states to unit-normalized contrastive embeddings."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # The MLP preserves all leading dimensions: [..., D_in] -> [..., D_p].
        return nn.functional.normalize(self.net(x), dim=-1)


class TsumugiMRLModel(nn.Module, PyTorchModelHubMixin):
    """Audio-only model. Forward returns frame states [B, T, D]."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.audio_encoder = MaskedAudioEncoder(self.config)
        self.audio_projection = ProjectionHead(self.config.d_model, self.config.projection_dim)
        self._hub_mixin_config = self.config

    def set_mel_stats(self, mean: float, std: float) -> None:
        self.audio_encoder.set_mel_stats(mean, std)

    def encode_audio(
        self, audio: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        return self.audio_encoder(audio, mask=mask, padding_mask=padding_mask)

    def forward(self, audio: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.encode_audio(audio, padding_mask=padding_mask)

    @staticmethod
    def mean_pool(hidden: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if padding_mask is None:
            return hidden.mean(dim=1)
        valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
        return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def encode_embedding(self, audio: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """Return normalized clip embeddings [B, projection_dim]."""
        return self.audio_projection(self.mean_pool(self(audio, padding_mask), padding_mask))
