"""Acoustic target tokenizer used during pretraining."""

from dataclasses import fields

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from train.mel_rvq.config import MelRVQConfig
from train.rvq import ResidualVectorQuantizer, RVQOutput
from tsumugi_mrl.model import StereoMelFrontend


class MelRVQTokenizer(nn.Module, PyTorchModelHubMixin):
    """Mel-RVQ tokenizer used to create offline acoustic targets.

    Train this tokenizer (or load a pretrained one) before masked-audio
    pretraining, then call :meth:`encode` on unmasked audio and keep the
    resulting codes fixed as acoustic targets.

    The tokenizer operates entirely in folded Mel space:
    ``[B, C, S] -> [B, T_a, D_mel]`` -> RVQ -> ``[B, T_a, D_mel]``. The Mel
    features are quantized directly; the bottleneck lives inside the
    quantizer, where each stage projects to ``acoustic_codebook_dim``.

    Reconstruction is measured on the summed codes themselves, as in MuQ.
    A decoder placed after the quantizer would let the codes drift away from
    the Mel features and be linearly corrected afterwards, which removes the
    pressure that keeps the codes faithful.
    """

    def __init__(self, config: MelRVQConfig) -> None:
        super().__init__()
        self.config = config or MelRVQConfig()
        config = self.config
        self.frontend = StereoMelFrontend(config)
        frontend_dim = config.temporal_fold * config.audio_channels * config.n_mels
        self.rvq = ResidualVectorQuantizer(
            input_dim=frontend_dim,
            num_codebooks=config.acoustic_codebooks,
            codebook_size=config.acoustic_vocab_size,
            commitment_weight=config.rvq_commitment_weight,
            codebook_dim=config.acoustic_codebook_dim,
        )
        self.reconstruction_weight = config.rvq_reconstruction_weight
        # Keep the full dataclass so save_pretrained() writes every
        # architectural setting needed by from_pretrained().
        self._hub_mixin_config = self.config

    @property
    def mel_stats(self) -> tuple[float, float]:
        """Return the dataset statistics used by this tokenizer."""

        return self.frontend.mel_stats

    @classmethod
    def from_checkpoint(cls, path) -> "MelRVQTokenizer":
        """Load a training .pt checkpoint, including legacy joint configs.

        Historical checkpoints also saved unrelated symbolic settings. Only
        acoustic/inference fields affect these weights; do not validate the
        obsolete symbolic vocabulary while loading an acoustic tokenizer.
        """
        checkpoint = torch.load(path, map_location="cpu")
        config_fields = {field.name for field in fields(MelRVQConfig)}
        config = MelRVQConfig(**{key: value for key, value in checkpoint["config"].items() if key in config_fields})
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.eval()

    def set_mel_stats(self, mean: float, std: float) -> None:
        """Set and persist the dataset-level Mel normalization statistics."""

        self.frontend.set_mel_stats(mean, std)

    def forward(self, audio: Tensor) -> RVQOutput:
        # Frontend: stereo waveform [B, C, S] -> folded Mel [B, T_a, D_mel].
        mel = self.frontend(audio)
        # RVQ quantizes the Mel features directly, keeping the continuous
        # representation at [B, T_a, D_mel] and emitting one integer code per
        # codebook: [B, T_a, N_acoustic].
        result = self.rvq(mel)
        # L1 between the summed codes and the Mel features. It keeps loud Mel
        # bins from dominating the gradient the way a squared error does, so
        # quiet detail still shapes the codebooks.
        reconstruction_loss = nn.functional.l1_loss(result.quantized, mel)
        return RVQOutput(
            quantized=result.quantized,
            codes=result.codes,
            codebook_loss=result.codebook_loss,
            commitment_loss=result.commitment_loss,
            loss=result.loss + self.reconstruction_weight * reconstruction_loss,
            reconstruction_loss=reconstruction_loss,
        )

    @torch.no_grad()
    def encode(self, audio: Tensor) -> Tensor:
        """Create fixed acoustic targets with shape ``[B, T, N]``."""

        return self.encode_features(self.frontend(audio))

    @torch.no_grad()
    def encode_features(self, mel_features: Tensor) -> Tensor:
        """Create fixed acoustic targets from normalized folded Mel features."""

        return self.rvq.encode(mel_features)

    def decode(self, codes: Tensor) -> Tensor:
        """Decode ``[B, T_a, N_acoustic]`` codes to ``[B, T_a, D_mel]``."""

        return self.rvq.decode(codes)
