import json
from dataclasses import asdict

import mido
import numpy as np
import pytest
import soundfile as sf
import torch

from train.config import TrainingConfig
from train.data import PairedAudioDataset, collate_pretraining_windows
from train.losses import PretrainingLoss
from train.mel_rvq.model import MelRVQTokenizer
from train.symbolic import MIDIEventTokenizer, load_pretraining_cache, save_pretraining_cache
from train.symbolic_teacher.model import SymbolicTeacher
from train.symbolic_teacher.prepare_dataset import _save_token_cache
from train.train import (
    build_parser,
    load_teachers,
    make_span_mask,
    polynomial_decay_lr,
    pretraining_losses,
    train,
)
from tsumugi_mrl import TsumugiMRLModel


@pytest.fixture
def training_files(tmp_path):
    config = TrainingConfig(
        n_mels=8,
        n_fft=512,
        conv_channels=4,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        symbolic_d_model=16,
        symbolic_heads=2,
        symbolic_layers=1,
        symbolic_dim_feedforward=32,
        projection_dim=8,
        acoustic_codebooks=2,
        acoustic_vocab_size=8,
        musical_codebooks=2,
        musical_vocab_size=8,
    )
    mel = MelRVQTokenizer(config)
    mel.set_mel_stats(-10.0, 5.0)
    mel_path = tmp_path / "mel.pt"
    symbolic_path = tmp_path / "symbolic.pt"
    torch.save({"config": asdict(config), "state_dict": mel.state_dict()}, mel_path)
    torch.save({"config": asdict(config), "state_dict": SymbolicTeacher(config).state_dict()}, symbolic_path)
    entries = []
    for index, duration in enumerate((0.4, 0.8)):
        audio = tmp_path / f"{index}.wav"
        midi = tmp_path / f"{index}.mid"
        cache = tmp_path / f"{index}.pt"
        sf.write(audio, torch.randn(round(22050 * duration)).numpy() * 0.01, 22050)
        song = mido.MidiFile()
        track = mido.MidiTrack()
        song.tracks.append(track)
        track.append(mido.Message("note_on", note=60 + index, velocity=80))
        track.append(mido.Message("note_off", note=60 + index, time=round(duration * 960)))
        song.save(midi)
        _save_token_cache(MIDIEventTokenizer(), midi, cache)
        entries.append({"audio_path": audio.name, "token_path": cache.name})
    entries.append({"audio_path": "failed.wav", "status": "error"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    return config, manifest, mel_path, symbolic_path


def test_pretraining_parser_accepts_wandb_options():
    args = build_parser().parse_args(
        [
            "--wandb",
            "--wandb-project",
            "project",
            "--wandb-name",
            "run",
            "--wandb-entity",
            "entity",
        ]
    )

    assert args.wandb is True
    assert args.wandb_project == "project"
    assert args.wandb_name == "run"
    assert args.wandb_entity == "entity"


def test_padded_batch_updates_student_and_projection_only(training_files):
    config, manifest, mel_path, symbolic_path = training_files
    dataset = PairedAudioDataset(manifest, config, crop_frames=20)
    assert len(dataset) == 2
    batch = collate_pretraining_windows([dataset[0], dataset[1]])
    assert batch["audio_padding_mask"][0].any()
    assert not batch["audio_padding_mask"][1].any()
    model, mel = load_teachers(mel_path, symbolic_path, {})
    model.symbolic_teacher.freeze()
    mel.requires_grad_(False)
    model.train()
    original = {k: v.clone() for k, v in model.symbolic_teacher.state_dict().items()}
    losses = pretraining_losses(model, mel, batch, PretrainingLoss(), 0.5, 3)
    assert set(losses) == {"loss_total", "loss_acoustic", "loss_musical", "loss_contrastive"}
    losses["loss_total"].backward()
    assert model.audio_encoder.subsampling.projection.weight.grad.abs().sum() > 0
    assert model.symbolic_teacher.encoder.projection.net[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in mel.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
    optimizer.step()
    for name, value in model.symbolic_teacher.state_dict().items():
        if not name.startswith("encoder.projection."):
            assert torch.equal(value, original[name])
    assert not model.symbolic_teacher.encoder.training


def test_compact_pretraining_cache_preserves_crop_inputs(training_files, tmp_path):
    config, manifest, _, _ = training_files
    entry = json.loads(manifest.read_text(encoding="utf-8"))[0]
    source = manifest.parent / entry["token_path"]
    compact = tmp_path / "pretraining.pt"
    save_pretraining_cache(source, compact)

    sequence = load_pretraining_cache(compact)
    original = torch.load(source, map_location="cpu")
    assert sequence.token_ids.dtype == torch.long
    assert sequence.frame_targets.note_activity.shape[1] == 0
    assert torch.equal(sequence.frame_targets.instrument_activity, original["frame_targets"]["instrument_activity"])

    compact_manifest = tmp_path / "compact_manifest.json"
    compact_manifest.write_text(
        json.dumps([{"audio_path": entry["audio_path"], "pretraining_token_path": str(compact)}]),
        encoding="utf-8",
    )
    dataset = PairedAudioDataset(compact_manifest, config, crop_frames=20)
    assert dataset[0].symbolic.token_ids.numel() > 0


def test_cached_mel_windows_replace_waveform_batches(training_files, tmp_path):
    config, manifest, mel_path, _ = training_files
    mel_teacher = MelRVQTokenizer.from_checkpoint(mel_path)
    entries = json.loads(manifest.read_text(encoding="utf-8"))[:2]
    cached_entries = []
    for index, entry in enumerate(entries):
        audio, rate = sf.read(manifest.parent / entry["audio_path"], dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T.copy()).unsqueeze(0)
        waveform = waveform.repeat(1, 2, 1) if waveform.size(1) == 1 else waveform[:, :2]
        features = mel_teacher.frontend(waveform).squeeze(0).numpy()
        mel_path_cache = tmp_path / f"{index}.npy"
        np.save(mel_path_cache, features.astype(np.float16))
        cached_entries.append(
            {
                "audio_path": entry["audio_path"],
                "token_path": entry["token_path"],
                "mel_path": str(mel_path_cache),
                "mel_frames": features.shape[0],
            }
        )
    cached_manifest = tmp_path / "mel_manifest.json"
    cached_manifest.write_text(json.dumps(cached_entries), encoding="utf-8")

    dataset = PairedAudioDataset(cached_manifest, config, crop_frames=20)
    batch = collate_pretraining_windows([dataset[0], dataset[1]])
    assert "mel_features" in batch
    assert "audio" not in batch


def test_ablation_modes_select_only_requested_losses(training_files):
    config, manifest, mel_path, symbolic_path = training_files
    dataset = PairedAudioDataset(manifest, config, crop_frames=20)
    batch = collate_pretraining_windows([dataset[0], dataset[1]])
    expected_losses = {
        "mel_rvq": {"loss_total", "loss_acoustic"},
        "symbolic_teacher": {"loss_total", "loss_acoustic", "loss_musical"},
        "contrastive": {"loss_total", "loss_acoustic", "loss_musical", "loss_contrastive"},
    }

    for ablation, expected in expected_losses.items():
        use_symbolic = ablation != "mel_rvq"
        model, mel = load_teachers(
            mel_path,
            symbolic_path if use_symbolic else None,
            {},
            use_symbolic=use_symbolic,
        )
        model.symbolic_teacher.freeze(train_projection=ablation == "contrastive")
        mel.requires_grad_(False)
        model.train()
        losses = pretraining_losses(
            model,
            mel,
            batch,
            PretrainingLoss(),
            0.5,
            3,
            ablation=ablation,
        )
        assert set(losses) == expected
        assert torch.isfinite(losses["loss_total"])


def test_mel_rvq_ablation_trains_without_symbolic_checkpoint(training_files, tmp_path):
    _, manifest, mel_path, _ = training_files
    output_dir = tmp_path / "mel_only"
    args = build_parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--mel-checkpoint",
            str(mel_path),
            "--ablation",
            "mel_rvq",
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--crop-frames",
            "20",
            "--device",
            "cpu",
        ]
    )
    train(args)

    checkpoint = torch.load(output_dir / "last.pt", map_location="cpu")
    assert checkpoint["training_settings"]["ablation"] == "mel_rvq"
    assert set(checkpoint["losses"]) == {"loss_acoustic", "loss_total"}


def test_span_mask_excludes_padding():
    padding = torch.tensor([[False] * 13 + [True] * 7, [False] * 20])
    mask = make_span_mask(padding, 0.5, 4)
    assert not (mask & padding).any()
    assert mask.sum(dim=1).tolist() == [6, 10]
    with pytest.raises(ValueError):
        make_span_mask(padding, 0, 4)


def test_training_resume_matches_uninterrupted_run_and_exports(training_files, tmp_path):
    _, manifest, mel_path, symbolic_path = training_files
    parser = build_parser()

    def run(output, epochs, resume=None):
        arguments = [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--epochs",
            str(epochs),
            "--batch-size",
            "2",
            "--crop-frames",
            "20",
            "--device",
            "cpu",
            # Both legs must share the schedule horizon: the rate at a given
            # update depends on how long the run was declared to be.
            "--decay-steps",
            "2",
        ]
        arguments += (
            ["--resume", str(resume)]
            if resume
            else ["--mel-checkpoint", str(mel_path), "--symbolic-checkpoint", str(symbolic_path)]
        )
        train(parser.parse_args(arguments))

    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    run(continuous, 2)
    run(resumed, 1)
    run(resumed, 2, resumed / "last.pt")
    expected = torch.load(continuous / "last.pt", map_location="cpu")
    actual = torch.load(resumed / "last.pt", map_location="cpu")
    assert actual["epoch"] == 2 and actual["step"] == 2
    for key, value in expected["state_dict"].items():
        assert torch.equal(value, actual["state_dict"][key]), key
    exported = TsumugiMRLModel.from_pretrained(resumed / "audio_model").eval()
    assert exported(torch.randn(1, 2, 8820)).shape[-1] == 16
    assert exported.audio_encoder.frontend.mel_stats == (-10.0, 5.0)


def test_save_interval_thins_numbered_checkpoints(training_files, tmp_path):
    _, manifest, mel_path, symbolic_path = training_files
    parser = build_parser()
    output = tmp_path / "interval"
    train(
        parser.parse_args(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--epochs",
                "3",
                "--save-interval",
                "2",
                "--batch-size",
                "2",
                "--crop-frames",
                "20",
                "--device",
                "cpu",
                "--mel-checkpoint",
                str(mel_path),
                "--symbolic-checkpoint",
                str(symbolic_path),
            ]
        )
    )
    # Epoch 2 lands on the interval and epoch 3 is the final epoch; the
    # resume checkpoint is written after every epoch either way.
    assert sorted(path.name for path in output.glob("epoch_*.pt")) == ["epoch_0002.pt", "epoch_0003.pt"]
    assert torch.load(output / "last.pt", map_location="cpu")["epoch"] == 3


def test_pretraining_defaults_follow_the_muq_recipe():
    args = build_parser().parse_args(["--manifest", "m.json"])

    assert args.lr == 5e-4
    assert tuple(args.adam_betas) == (0.9, 0.98)
    assert args.adam_eps == 1e-6
    assert args.weight_decay == 0.01
    assert args.grad_clip == 10.0
    # MuQ warms up for 32,000 of its 400,000 updates.
    assert args.warmup_ratio == 32_000 / 400_000
    assert (args.warmup_steps, args.decay_steps, args.lr_end, args.lr_power) == (0, 0, 0.0, 1.0)


def test_polynomial_decay_warms_up_then_decays_linearly():
    schedule = lambda update, **kwargs: polynomial_decay_lr(update, 5e-4, 32_000, 400_000, **kwargs)

    # The warmup is a straight line from one step's worth of rate up to the peak.
    assert schedule(1) == pytest.approx(5e-4 / 32_000)
    assert schedule(16_000) == pytest.approx(2.5e-4)
    assert schedule(32_000) == pytest.approx(5e-4)

    # The decay spends half the remaining updates reaching half the rate, and
    # holds at the floor once the declared run is over.
    assert schedule(216_000) == pytest.approx(2.5e-4)
    assert schedule(400_000) == 0.0
    assert schedule(500_000) == 0.0

    assert schedule(216_000, end_lr=1e-4) == pytest.approx(3e-4)
    # A higher power leaves the peak faster; both still meet at the ends.
    assert schedule(216_000, power=2.0) < schedule(216_000)
    assert schedule(32_000, power=2.0) == pytest.approx(5e-4)


def test_polynomial_decay_without_warmup_starts_at_the_peak():
    assert polynomial_decay_lr(1, 1e-3, 0, 10) == pytest.approx(9e-4)
    assert polynomial_decay_lr(5, 1e-3, 0, 10) == pytest.approx(5e-4)
