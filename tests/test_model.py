import torch

from train.config import TrainingConfig as ModelConfig
from train.losses import PretrainingLoss
from train.mel_rvq.model import MelRVQTokenizer
from train.pretraining import TsumugiMRLPretrainingModel as TsumugiMRLModel
from train.rvq import ResidualVectorQuantizer
from tsumugi_mrl.model import MaskedAudioEncoder, StereoMelFrontend


def test_rvq_codes_quantized_output_and_decode():
    torch.manual_seed(0)
    rvq = ResidualVectorQuantizer(
        input_dim=6,
        num_codebooks=3,
        codebook_size=8,
    )
    x = torch.randn(2, 5, 6, requires_grad=True)
    result = rvq(x)

    assert result.codes.shape == (2, 5, 3)
    assert result.quantized.shape == x.shape
    assert rvq.decode(result.codes).shape == x.shape

    result.loss.backward()
    assert x.grad is not None
    assert torch.isfinite(result.loss)


def _set_identity_projections(rvq: ResidualVectorQuantizer) -> None:
    """Make every stage project through unchanged for exact loss checks."""

    from torch.nn.utils import parametrize

    for projection in (*rvq.input_projections, *rvq.output_projections):
        # Drop the weight-norm parametrization so the weight can be set directly.
        if parametrize.is_parametrized(projection, "weight"):
            parametrize.remove_parametrizations(projection, "weight", leave_parametrized=True)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(rvq.input_dim))
            projection.bias.zero_()


def test_rvq_uses_normalized_lookup_and_raw_losses():
    rvq = ResidualVectorQuantizer(
        input_dim=2,
        num_codebooks=1,
        codebook_size=2,
    )
    _set_identity_projections(rvq)
    with torch.no_grad():
        rvq.codebooks.copy_(torch.tensor([[[1.0, 0.0], [10.0, 1.0]]]))

    x = torch.tensor([[[10.0, 0.0]]])
    result = rvq(x)

    # Raw Euclidean distance would choose code 1; cosine/L2-normalized
    # lookup chooses the collinear code 0.
    assert result.codes.item() == 0
    expected = torch.nn.functional.mse_loss(torch.tensor([[[1.0, 0.0]]]), x)
    assert torch.allclose(result.codebook_loss, expected)
    assert torch.allclose(result.commitment_loss, expected)


def test_mel_rvq_reconstruction_is_measured_on_the_codes():
    from train.mel_rvq.config import MelRVQConfig

    config = MelRVQConfig(n_mels=8, n_fft=512, acoustic_codebooks=2, acoustic_vocab_size=8)
    tokenizer = MelRVQTokenizer(config)
    audio = torch.randn(1, 2, 22_050)
    result = tokenizer(audio)

    mel = tokenizer.frontend(audio)
    expected = (result.quantized - mel).abs().mean()
    assert torch.allclose(result.reconstruction_loss, expected)


def test_rvq_projects_each_stage_into_a_codebook_bottleneck():
    rvq = ResidualVectorQuantizer(
        input_dim=6,
        num_codebooks=3,
        codebook_size=8,
        codebook_dim=2,
    )

    assert rvq.codebooks.shape == (3, 8, 2)
    assert len(rvq.input_projections) == len(rvq.output_projections) == 3
    assert rvq.input_projections[0].in_features == 6
    assert rvq.input_projections[0].out_features == 2
    assert rvq.output_projections[0].in_features == 2
    assert rvq.output_projections[0].out_features == 6

    x = torch.randn(2, 4, 6)
    result = rvq(x)
    assert result.codes.shape == (2, 4, 3)
    assert result.quantized.shape == x.shape
    assert rvq.decode(result.codes).shape == x.shape


def test_rvq_reconstruction_reaches_the_projections():
    # The per-stage straight-through is the only one: a reconstruction loss
    # taken on the quantizer output must train the projections even when the
    # input is plain data with no upstream encoder.
    rvq = ResidualVectorQuantizer(input_dim=4, num_codebooks=2, codebook_size=8, codebook_dim=2)
    x = torch.randn(2, 5, 4)
    result = rvq(x)

    reconstruction = (result.quantized - x).abs().mean()
    assert reconstruction.requires_grad
    reconstruction.backward()
    for stage in range(2):
        assert rvq.input_projections[stage].parametrizations.weight.original1.grad is not None
        assert rvq.output_projections[stage].parametrizations.weight.original1.grad is not None


def test_rvq_sums_stage_losses():
    rvq = ResidualVectorQuantizer(
        input_dim=2,
        num_codebooks=2,
        codebook_size=2,
    )
    _set_identity_projections(rvq)
    with torch.no_grad():
        rvq.codebooks.copy_(torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]))

    x = torch.tensor([[[4.0, 0.0]]])
    result = rvq(x)

    # Stage one leaves the residual [3, 0] and stage two leaves [2, 0], so an
    # average would report 6.5 where the sum reports 13.
    assert torch.allclose(result.codebook_loss, torch.tensor(13.0) / 2)
    assert torch.allclose(result.commitment_loss, torch.tensor(13.0) / 2)


def test_mel_stats_are_applied_and_saved_in_state_dict():
    config = ModelConfig(n_mels=8, n_fft=512, hop_length=441, temporal_fold=2)
    frontend = StereoMelFrontend(config)
    audio = torch.randn(1, 2, 22_050)
    raw_mel = frontend(audio)
    frontend.set_mel_stats(raw_mel.mean().item(), raw_mel.std(unbiased=False).item())

    normalized_mel = frontend(audio)
    assert abs(normalized_mel.mean().item()) < 1e-5
    assert "mel_mean" in frontend.state_dict()
    assert "mel_std" in frontend.state_dict()

    restored = StereoMelFrontend(config)
    restored.load_state_dict(frontend.state_dict())
    assert restored.mel_stats == frontend.mel_stats
    assert torch.allclose(restored(audio), normalized_mel)


def test_stereo_forward_and_three_losses():
    config = ModelConfig(
        n_mels=32,
        n_fft=512,
        hop_length=441,
        max_audio_seconds=2.0,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        acoustic_codebooks=2,
        acoustic_vocab_size=16,
        musical_codebooks=2,
        musical_vocab_size=16,
        symbolic_d_model=32,
        symbolic_heads=4,
        symbolic_layers=1,
        symbolic_dim_feedforward=64,
        projection_dim=16,
    )
    model = TsumugiMRLModel(config)
    assert model.audio_encoder.input_projection.in_features == 2 * config.n_mels * config.temporal_fold

    audio = torch.randn(2, 2, 22_050 * 2)
    rvq_teacher = MelRVQTokenizer(config)
    tokenizer_result = rvq_teacher(audio)
    frontend_dim = 2 * config.n_mels * config.temporal_fold
    # Mel features are quantized directly; the bottleneck is inside the RVQ.
    assert rvq_teacher.rvq.input_dim == frontend_dim
    assert rvq_teacher.rvq.codebook_dim == config.acoustic_codebook_dim
    # There is no decoder: the summed codes are the reconstruction.
    assert not hasattr(rvq_teacher, "decoder")
    assert tokenizer_result.quantized.shape[-1] == frontend_dim
    assert tokenizer_result.reconstruction_loss is not None
    assert torch.isfinite(tokenizer_result.reconstruction_loss)
    acoustic_targets_from_rvq = rvq_teacher.encode(audio)
    assert acoustic_targets_from_rvq.shape == (2, model.encode_audio(audio).size(1), config.acoustic_codebooks)
    # The exact token count is determined by the non-centered Mel frontend.
    token_count = model.encode_audio(audio).size(1)
    audio_mask = torch.zeros(2, token_count, dtype=torch.bool)
    audio_mask[:, token_count // 3 : token_count // 2] = True

    symbolic_tokens = torch.randint(0, config.symbolic_vocab_size, (2, 8))
    symbolic_types = torch.zeros_like(symbolic_tokens)
    symbolic_padding = torch.zeros_like(symbolic_tokens, dtype=torch.bool)

    output = model(
        audio,
        audio_mask=audio_mask,
        symbolic_token_ids=symbolic_tokens,
        symbolic_token_instrument_ids=torch.full_like(symbolic_tokens, config.symbolic_instrument_classes),
        symbolic_token_type_ids=symbolic_types,
        symbolic_padding_mask=symbolic_padding,
    )

    acoustic_targets = torch.randint(0, config.acoustic_vocab_size, (2, token_count, config.acoustic_codebooks))
    musical_targets = torch.randint(0, config.musical_vocab_size, (2, token_count, config.musical_codebooks))
    losses = PretrainingLoss()(  # default weights are intentionally simple
        output,
        acoustic_targets,
        musical_targets,
        acoustic_mask=audio_mask,
        musical_mask=audio_mask,
    )

    assert output.hidden.shape[:2] == (2, token_count)
    assert output.acoustic_logits.shape == (
        2,
        token_count,
        config.acoustic_codebooks,
        config.acoustic_vocab_size,
    )
    assert output.musical_logits.shape == (
        2,
        token_count,
        config.musical_codebooks,
        config.musical_vocab_size,
    )
    assert output.audio_embedding.shape == (2, config.projection_dim)
    assert output.symbolic_embedding.shape == (2, config.projection_dim)
    assert torch.isfinite(losses["loss_total"])


def test_audio_encoder_gradient_checkpointing_backpropagates():
    config = ModelConfig(
        n_mels=8,
        n_fft=512,
        d_model=16,
        n_heads=4,
        num_layers=2,
        dim_feedforward=32,
        gradient_checkpointing=True,
    )
    encoder = MaskedAudioEncoder(config).train()
    audio = torch.randn(1, 2, 8_820)

    encoder(audio).square().mean().backward()

    assert encoder.use_gradient_checkpoint
    assert encoder.input_projection.weight.grad is not None


def test_audio_export_reload_and_downstream_gradients(tmp_path):
    from tsumugi_mrl import TsumugiMRLModel as AudioModel

    config = ModelConfig(
        n_mels=8,
        n_fft=512,
        d_model=32,
        n_heads=4,
        num_layers=1,
        dim_feedforward=64,
        symbolic_d_model=32,
        symbolic_heads=4,
        symbolic_layers=1,
        symbolic_dim_feedforward=64,
        acoustic_codebooks=2,
        acoustic_vocab_size=8,
        musical_codebooks=2,
        musical_vocab_size=8,
        projection_dim=8,
    )
    training = TsumugiMRLModel(config).eval()
    training.set_mel_stats(-20.0, 10.0)
    audio = torch.randn(2, 2, 8820)
    expected = training(audio)
    exported = training.export_audio_model()
    assert torch.allclose(exported(audio), expected.hidden)
    assert torch.allclose(exported.encode_embedding(audio), expected.audio_embedding)
    assert not any(key.startswith(("symbolic", "acoustic_head", "musical_head")) for key in exported.state_dict())
    directory = tmp_path / "audio_model"
    exported.save_pretrained(directory)
    restored = AudioModel.from_pretrained(directory).eval()
    assert torch.allclose(restored(audio), expected.hidden)
    assert torch.allclose(restored.encode_embedding(audio), expected.audio_embedding)
    restored.train()
    restored(audio).square().mean().backward()
    assert restored.audio_encoder.input_projection.weight.grad is not None


def test_inference_import_does_not_load_training_or_midi_modules():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tsumugi_mrl; "
            "assert not any(k == 'train' or k.startswith('train.') for k in sys.modules); "
            "assert 'mido' not in sys.modules; assert 'dlchordx' not in sys.modules",
        ],
        check=True,
    )


def test_legacy_mel_checkpoint_ignores_unrelated_symbolic_settings(tmp_path):
    from dataclasses import asdict
    from train.mel_rvq.config import MelRVQConfig

    config = MelRVQConfig(n_mels=8, n_fft=512, acoustic_codebooks=2, acoustic_vocab_size=8)
    teacher = MelRVQTokenizer(config).eval()
    legacy_config = asdict(config)
    legacy_config.update(symbolic_type_vocab_size=8, symbolic_vocab_size=32)
    path = tmp_path / "legacy.pt"
    torch.save({"config": legacy_config, "state_dict": teacher.state_dict()}, path)
    restored = MelRVQTokenizer.from_checkpoint(path)
    audio = torch.randn(1, 2, 8820)
    assert torch.equal(teacher.encode(audio), restored.encode(audio))
