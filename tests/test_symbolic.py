import mido
import pytest
import torch

from train.config import TrainingConfig as ModelConfig
from train.losses import SymbolicTeacherLoss
from train.symbolic import (
    BASS_CLASSES,
    BASS_NO_BASS_ID,
    BASS_TOKEN_BASE,
    CHORD_CLASSES,
    CHORD_NO_CHORD_ID,
    CHORD_TOKEN_BASE,
    FRAME_TOKEN_ID,
    INSTRUMENT_CLASSES,
    KEY_CLASSES,
    METER_CLASSES,
    METER_TOKEN_BASE,
    MIN_MIDI_PITCH,
    NO_INSTRUMENT_ID,
    NOTE_ON_TOKEN_BASE,
    PITCH_CLASSES,
    SYMBOLIC_REQUIRED_VOCAB_SIZE,
    MIDIEventTokenizer,
    SymbolicFrameTargets,
    SymbolicMIDIDataset,
    SymbolicTokenType,
    collate_symbolic_sequences,
    crop_symbolic_sequence,
    encode_chord_and_bass,
    encode_chord_label,
    encode_key_name,
    encode_meter,
)
from train.symbolic_teacher.model import SymbolicTeacher


def _small_config() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_heads=4,
        num_layers=1,
        dim_feedforward=64,
        symbolic_d_model=32,
        symbolic_heads=4,
        symbolic_layers=1,
        symbolic_dim_feedforward=64,
        musical_codebooks=2,
        musical_vocab_size=16,
        projection_dim=8,
    )


def test_frame_query_is_gathered_from_augmented_sequence():
    torch.manual_seed(0)
    config = _small_config()
    teacher = SymbolicTeacher(config)
    token_ids = torch.tensor([[FRAME_TOKEN_ID, 2, 3, FRAME_TOKEN_ID, 4, FRAME_TOKEN_ID]])
    token_types = torch.tensor(
        [
            [
                int(SymbolicTokenType.CONTROL),
                int(SymbolicTokenType.NOTE),
                int(SymbolicTokenType.NOTE_OFF),
                int(SymbolicTokenType.CONTROL),
                int(SymbolicTokenType.CHORD),
                int(SymbolicTokenType.CONTROL),
            ]
        ]
    )
    position_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    output = teacher(
        token_ids,
        anchor_positions=torch.tensor([[0, 3, 5]]),
        token_instrument_ids=torch.full_like(token_ids, NO_INSTRUMENT_ID),
        token_type_ids=token_types,
        position_ids=position_ids,
    )

    assert output.token_hidden.shape == (1, 6, config.symbolic_d_model)
    assert output.frame_hidden.shape == (1, 3, config.symbolic_d_model)
    assert output.codes.shape == (1, 3, config.musical_codebooks)
    assert output.reconstruction["chord"].shape == (1, 3, CHORD_CLASSES)
    assert output.reconstruction["bass"].shape == (1, 3, BASS_CLASSES)


def test_symbolic_teacher_loss_uses_frame_outputs():
    config = _small_config()
    teacher = SymbolicTeacher(config)
    output = teacher(
        torch.tensor([[FRAME_TOKEN_ID, 2, FRAME_TOKEN_ID]]),
        anchor_positions=torch.tensor([[0, 2]]),
        token_instrument_ids=torch.tensor([[NO_INSTRUMENT_ID, 21, NO_INSTRUMENT_ID]]),
        token_type_ids=torch.tensor(
            [[int(SymbolicTokenType.CONTROL), int(SymbolicTokenType.NOTE), int(SymbolicTokenType.CONTROL)]]
        ),
        position_ids=torch.tensor([[0, 0, 1]]),
    )
    targets = SymbolicFrameTargets(
        note_activity=torch.zeros(1, 2, config.symbolic_pitch_classes),
        note_onset=torch.zeros(1, 2, config.symbolic_pitch_classes),
        note_offset=torch.zeros(1, 2, config.symbolic_pitch_classes),
        instrument_activity=torch.zeros(1, 2, config.symbolic_instrument_classes),
        beat=torch.ones(1, 2),
        downbeat=torch.zeros(1, 2),
        chord=torch.zeros(1, 2, dtype=torch.long),
        bass=torch.zeros(1, 2, dtype=torch.long),
        key=torch.zeros(1, 2, dtype=torch.long),
        meter=torch.zeros(1, 2, dtype=torch.long),
    )
    losses = SymbolicTeacherLoss()(output, targets)
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()


def test_symbolic_label_layout_is_deterministic():
    assert encode_chord_label("C:M7") == 10
    assert encode_chord_label("N") == CHORD_NO_CHORD_ID
    assert CHORD_CLASSES == 745
    assert encode_chord_and_bass("C:m7/Bb") == (encode_chord_label("C:m7"), 10)
    assert encode_chord_and_bass("C:m7")[1] == 0  # implicit root bass
    assert encode_chord_and_bass("N") == (CHORD_NO_CHORD_ID, BASS_NO_BASS_ID)
    assert BASS_CLASSES == 13
    assert INSTRUMENT_CLASSES == 36
    assert PITCH_CLASSES == 88
    assert encode_key_name("A minor") == 19
    assert KEY_CLASSES == 25
    assert METER_CLASSES == 61
    assert encode_meter(21, 32) != METER_CLASSES - 1
    assert encode_meter(99, 99) == METER_CLASSES - 1


def test_midi_tokenizer_keeps_midi_events_and_adds_frame_anchors(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    conductor.append(mido.MetaMessage("key_signature", key="C", time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=1920))
    midi.tracks.append(conductor)

    chords = mido.MidiTrack()
    chords.append(mido.MetaMessage("track_name", name="Predicted Chords", time=0))
    chords.append(mido.MetaMessage("marker", text="C:M7/E", time=0))
    chords.append(mido.MetaMessage("marker", text="G:7", time=960))
    chords.append(mido.MetaMessage("end_of_track", time=960))
    midi.tracks.append(chords)

    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="Piano", time=0))
    notes.append(mido.Message("program_change", program=0, channel=0, time=0))
    notes.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    notes.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(notes)
    midi_path = tmp_path / "example.mid"
    midi.save(midi_path)

    sequence = MIDIEventTokenizer().encode(midi_path)
    assert sequence.token_ids[sequence.anchor_positions].eq(FRAME_TOKEN_ID).all()
    assert sequence.anchor_positions.numel() == 50
    assert sequence.frame_targets.note_activity[:, 60 - MIN_MIDI_PITCH].sum() > 0
    assert sequence.frame_targets.beat.sum() > 0
    assert sequence.frame_targets.downbeat.sum() > 0
    assert (sequence.frame_targets.chord >= 0).any()
    assert (sequence.frame_targets.bass >= 0).any()
    assert sequence.frame_targets.bass[0].item() == 4
    assert (sequence.frame_targets.key >= 0).any()
    assert sequence.frame_targets.instrument_activity.shape[-1] == INSTRUMENT_CLASSES


def test_default_vocab_covers_the_fixed_layout_exactly():
    config = ModelConfig()
    assert config.symbolic_vocab_size == SYMBOLIC_REQUIRED_VOCAB_SIZE == 1060

    highest_token_id = METER_TOKEN_BASE + METER_CLASSES - 1
    assert highest_token_id == config.symbolic_vocab_size - 1

    with pytest.raises(ValueError):
        ModelConfig(symbolic_vocab_size=64)


def _sustained_midi(path):
    """One note held across the whole file, with a chord change in the middle."""

    midi = mido.MidiFile(ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    conductor.append(mido.MetaMessage("key_signature", key="C", time=0))
    conductor.append(mido.MetaMessage("end_of_track", time=3840))
    midi.tracks.append(conductor)

    chords = mido.MidiTrack()
    chords.append(mido.MetaMessage("track_name", name="Predicted Chords", time=0))
    chords.append(mido.MetaMessage("marker", text="C:M7", time=0))
    chords.append(mido.MetaMessage("end_of_track", time=3840))
    midi.tracks.append(chords)

    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="Piano", time=0))
    notes.append(mido.Message("program_change", program=0, channel=0, time=0))
    notes.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
    notes.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=3840))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(notes)
    midi.save(path)
    return path


def test_crop_carries_held_notes_and_state_into_the_window(tmp_path):
    sequence = MIDIEventTokenizer().encode(_sustained_midi(tmp_path / "sustained.mid"))
    start_frame, num_frames = 40, 20
    window = crop_symbolic_sequence(sequence, start_frame, num_frames)

    # The frame layout survives: one FRAME anchor per frame, rebased to 0.
    assert window.num_frames == num_frames
    assert window.token_ids[window.anchor_positions].eq(FRAME_TOKEN_ID).all()
    assert window.position_ids.min() == 0
    assert window.position_ids.max() == num_frames - 1
    assert window.anchor_positions[0] == 0
    assert window.anchor_positions.max() < window.token_ids.numel()

    # The note started before the window, so a plain slice would drop it.
    pitch_token = NOTE_ON_TOKEN_BASE + 60 - MIN_MIDI_PITCH
    assert (
        not sequence.token_ids[
            int(sequence.anchor_positions[start_frame]) : int(sequence.anchor_positions[start_frame + num_frames])
        ]
        .eq(pitch_token)
        .any()
    )
    assert window.token_ids.eq(pitch_token).any()
    held = window.token_ids.eq(pitch_token)
    assert window.token_type_ids[held].eq(int(SymbolicTokenType.CONTROL)).all()
    assert window.frame_targets.note_onset[:, 60 - MIN_MIDI_PITCH].sum() == 0
    assert window.frame_targets.note_offset[:, 60 - MIN_MIDI_PITCH].sum() == 0

    # The chord/key state written at frame 0 is restored at the window start.
    carried_chord = int(sequence.frame_targets.chord[start_frame])
    assert carried_chord >= 0
    assert window.token_ids.eq(CHORD_TOKEN_BASE + carried_chord).any()
    assert window.token_type_ids.eq(int(SymbolicTokenType.CHORD)).any()
    carried_bass = int(sequence.frame_targets.bass[start_frame])
    assert carried_bass >= 0
    assert window.token_ids.eq(BASS_TOKEN_BASE + carried_bass).any()
    assert window.token_type_ids.eq(int(SymbolicTokenType.BASS)).any()

    # Frame targets are simply the same frames of the full sequence.
    for name, value in window.frame_targets.as_dict().items():
        assert torch.equal(value, sequence.frame_targets.as_dict()[name][start_frame : start_frame + num_frames])


def test_crop_at_frame_zero_is_a_plain_prefix(tmp_path):
    sequence = MIDIEventTokenizer().encode(_sustained_midi(tmp_path / "sustained.mid"))
    window = crop_symbolic_sequence(sequence, 0, 20)

    end_token = int(sequence.anchor_positions[20])
    assert torch.equal(window.token_ids, sequence.token_ids[:end_token])
    assert torch.equal(window.anchor_positions, sequence.anchor_positions[:20])


def test_cropped_windows_collate_and_train(tmp_path):
    path = _sustained_midi(tmp_path / "sustained.mid")
    dataset = SymbolicMIDIDataset(tmp_path, crop_frames=16)
    batch = collate_symbolic_sequences([dataset[0], dataset[0]])
    assert batch["anchor_positions"].shape == (2, 16)

    teacher = SymbolicTeacher(_small_config())
    output = teacher(
        batch["token_ids"],
        anchor_positions=batch["anchor_positions"],
        token_instrument_ids=batch["token_instrument_ids"],
        token_type_ids=batch["token_type_ids"],
        padding_mask=batch["padding_mask"],
        position_ids=batch["position_ids"],
        frame_padding_mask=batch["frame_padding_mask"],
    )
    losses = SymbolicTeacherLoss()(output, batch["targets"], frame_padding_mask=batch["frame_padding_mask"])
    assert torch.isfinite(losses["loss_total"])
    assert path.is_file()


def test_binary_reconstruction_averages_classes_and_ignores_padding(tmp_path):
    from train.losses import symbolic_frame_reconstruction_loss

    sequence = MIDIEventTokenizer().encode(_sustained_midi(tmp_path / "sustained.mid"))
    batch = collate_symbolic_sequences(
        [
            crop_symbolic_sequence(sequence, 0, 3),
            crop_symbolic_sequence(sequence, 0, 1),
        ]
    )
    teacher = SymbolicTeacher(_small_config())
    predictions = teacher.decoder(torch.zeros(2, 3, teacher.config.symbolic_d_model))
    # Zero logits give log(2) for every binary element, independent of label.
    for name, value in predictions.items():
        predictions[name] = torch.zeros_like(value).masked_fill(
            batch["frame_padding_mask"].unsqueeze(-1) if value.ndim == 3 else batch["frame_padding_mask"], 100.0
        )
    losses = symbolic_frame_reconstruction_loss(predictions, batch["targets"], batch["frame_padding_mask"])
    for name in ("note_activity", "note_onset", "note_offset", "instrument_activity", "beat", "downbeat"):
        assert torch.allclose(losses[f"loss_{name}"], torch.tensor(2.0).log())
    empty = symbolic_frame_reconstruction_loss(
        predictions, batch["targets"], torch.ones_like(batch["frame_padding_mask"])
    )
    assert empty["loss_total"].item() == 0


def test_frozen_teacher_projection_learns_without_changing_codes():
    from train.pretraining import TsumugiMRLPretrainingModel as TsumugiMRLModel
    from train.losses import symmetric_info_nce

    torch.manual_seed(12)
    model = TsumugiMRLModel(_small_config())
    teacher = model.symbolic_teacher
    teacher.freeze()
    model.train()
    assert not teacher.encoder.training
    assert teacher.encoder.projection.training
    inputs = torch.tensor([[FRAME_TOKEN_ID, 2, FRAME_TOKEN_ID], [FRAME_TOKEN_ID, 5, FRAME_TOKEN_ID]])
    anchors = torch.tensor([[0, 2], [0, 2]])
    before = teacher(inputs, anchors, torch.full_like(inputs, NO_INSTRUMENT_ID))
    repeated = teacher(inputs, anchors, torch.full_like(inputs, NO_INSTRUMENT_ID))
    assert torch.equal(before.quantized, repeated.quantized)
    audio_embedding = model.audio_projection(torch.randn(2, model.config.d_model))
    loss = symmetric_info_nce(audio_embedding, before.embedding)
    loss.backward()
    for name, parameter in teacher.named_parameters():
        if name.startswith("encoder.projection."):
            assert parameter.grad is not None
        else:
            assert parameter.grad is None
            assert not parameter.requires_grad
    assert sum(p.grad.abs().sum() for p in teacher.encoder.projection.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.audio_projection.parameters()) > 0
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    optimizer.step()
    after = teacher(inputs, anchors, torch.full_like(inputs, NO_INSTRUMENT_ID))
    assert torch.equal(before.codes, after.codes)
    assert torch.equal(before.quantized, after.quantized)
    assert not torch.allclose(before.embedding, after.embedding)
    teacher.freeze(train_projection=False)
    model.train()
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.encoder.projection.training


@pytest.mark.parametrize("n_fft", [512, 2048])
def test_paired_crop_matches_audio_context_and_frontend_centers(tmp_path, n_fft):
    from train.data import crop_pretraining_window
    from tsumugi_mrl.model import StereoMelFrontend

    sequence = MIDIEventTokenizer().encode(_sustained_midi(tmp_path / "sustained.mid"))
    config = ModelConfig(n_fft=n_fft, n_mels=8)
    audio = torch.arange(4 * config.sample_rate, dtype=torch.float32).repeat(2, 1)
    window = crop_pretraining_window(audio, sequence, 40, 20, config)
    assert torch.equal(window.audio, audio[:, 40 * 882 : 60 * 882])
    assert window.symbolic.num_frames == 20
    frontend_count = StereoMelFrontend(config)(window.audio.unsqueeze(0)).size(1)
    indices = window.symbolic_frame_indices
    assert indices.numel() == frontend_count
    expected = [int((i * 882 + n_fft / 2 + 441 / 2) // 882) for i in range(frontend_count)]
    assert indices.tolist() == expected
    assert indices.max() < window.symbolic.num_frames
    # Both modalities end at the common available boundary.
    tail = crop_pretraining_window(audio[:, : 95 * 882], sequence, 90, 20, config)
    assert tail.audio.shape[-1] == 5 * 882
    assert tail.symbolic.num_frames == 5


def _two_instrument_midi(path, swap=False, same_pitch=False):
    midi = mido.MidiFile(ticks_per_beat=480)
    for name, program, pitch, start in (
        ("Piano", 0, 64 if swap else 60, 0),
        ("Acoustic Guitar", 24, 60 if swap or same_pitch else 64, 480 if same_pitch else 0),
    ):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        track.append(mido.Message("program_change", program=program, time=0))
        track.append(mido.Message("note_on", note=pitch, velocity=100, time=start))
        track.append(mido.Message("note_off", note=pitch, time=1440 - start))
        track.append(mido.MetaMessage("end_of_track", time=480))
        midi.tracks.append(track)
    midi.save(path)
    return path


def test_note_instrument_binding_changes_teacher_frame_states(tmp_path):
    torch.manual_seed(7)
    a = MIDIEventTokenizer().encode(_two_instrument_midi(tmp_path / "a.mid"))
    b = MIDIEventTokenizer().encode(_two_instrument_midi(tmp_path / "b.mid", swap=True))
    # Same event multiset and global instrument activity, different pairings.
    assert torch.equal(a.token_ids, b.token_ids)
    assert not torch.equal(a.token_instrument_ids, b.token_instrument_ids)
    from train.symbolic import NOTE_OFF_TOKEN_BASE

    for sequence, piano_pitch in ((a, 60), (b, 64)):
        for base in (NOTE_ON_TOKEN_BASE, NOTE_OFF_TOKEN_BASE):
            note = sequence.token_ids == base + piano_pitch - MIN_MIDI_PITCH
            assert sequence.token_instrument_ids[note].eq(21).all()
    batch = collate_symbolic_sequences([a, b])
    teacher = SymbolicTeacher(_small_config()).eval()
    output = teacher(
        batch["token_ids"],
        batch["anchor_positions"],
        batch["token_instrument_ids"],
        token_type_ids=batch["token_type_ids"],
        position_ids=batch["position_ids"],
        padding_mask=batch["padding_mask"],
        frame_padding_mask=batch["frame_padding_mask"],
    )
    assert not torch.allclose(output.frame_hidden[0], output.frame_hidden[1])
    SymbolicTeacherLoss()(output, batch["targets"])["loss_total"].backward()
    assert teacher.encoder.instrument_embedding.weight.grad[:36].abs().sum() > 0


def test_crop_keeps_held_instrument_when_other_instrument_starts_same_pitch(tmp_path):
    sequence = MIDIEventTokenizer().encode(_two_instrument_midi(tmp_path / "same.mid", same_pitch=True))
    # Guitar starts at 0.5 s, in frame 12, while the piano C4 is held.
    window = crop_symbolic_sequence(sequence, 12, 10)
    pitch_id = NOTE_ON_TOKEN_BASE + 60 - MIN_MIDI_PITCH
    first_frame = (window.position_ids == 0) & (window.token_ids == pitch_id)
    events = set(zip(window.token_type_ids[first_frame].tolist(), window.token_instrument_ids[first_frame].tolist()))
    assert events == {(int(SymbolicTokenType.CONTROL), 21), (int(SymbolicTokenType.NOTE), 2)}
    nested = crop_symbolic_sequence(window, 2, 5)
    held = (nested.position_ids == 0) & (nested.token_ids == pitch_id)
    assert set(nested.token_instrument_ids[held].tolist()) == {2, 21}
    assert nested.token_type_ids[held].eq(int(SymbolicTokenType.CONTROL)).all()


def test_cache_pairing_roundtrip_and_midi_only_migration(tmp_path, monkeypatch):
    import sys
    from train.symbolic import load_symbolic_cache, SYMBOLIC_CACHE_VERSION
    from train.symbolic_teacher.prepare_dataset import main

    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    midi_path = _two_instrument_midi(midi_dir / "pair.mid")
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    token_path = token_dir / "pair.pt"
    torch.save({"token_ids": torch.tensor([1])}, token_path)
    with pytest.raises(ValueError, match="--retokenize"):
        load_symbolic_cache(token_path)
    # No audio directory or AMT checkout is needed for MIDI-only migration.
    monkeypatch.setattr(sys, "argv", ["prepare_dataset", "--retokenize", "--output-dir", str(tmp_path)])
    main()
    restored = load_symbolic_cache(token_path)
    expected = MIDIEventTokenizer().encode(midi_path)
    assert torch.equal(restored.token_instrument_ids, expected.token_instrument_ids)
    assert torch.equal(restored.note_spans, expected.note_spans)
    assert torch.load(token_path)["cache_version"] == SYMBOLIC_CACHE_VERSION


def test_instrument_and_meter_tokens_are_only_state_changes(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    conductor.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=960))
    conductor.append(mido.MetaMessage("end_of_track", time=960))
    midi.tracks.append(conductor)
    notes = mido.MidiTrack()
    notes.append(mido.Message("program_change", program=0, time=0))
    for index in range(4):
        if index == 2:
            notes.append(mido.Message("program_change", program=24, time=0))
        notes.append(mido.Message("note_on", note=60, velocity=100, time=0))
        notes.append(mido.Message("note_off", note=60, time=480))
    midi.tracks.append(notes)
    path = tmp_path / "changes.mid"
    midi.save(path)
    sequence = MIDIEventTokenizer().encode(path)
    for kind in (SymbolicTokenType.INSTRUMENT, SymbolicTokenType.METER):
        selected = sequence.token_type_ids == int(kind)
        assert sequence.position_ids[selected].tolist() == [0, 25]
    onsets = sequence.token_type_ids == int(SymbolicTokenType.NOTE)
    assert sequence.token_instrument_ids[onsets].tolist() == [21, 21, 2, 2]
    targets = sequence.frame_targets
    assert targets.note_onset.sum() == 4
    assert targets.note_offset.sum() == 4
    assert targets.beat.sum() == 4
    assert targets.downbeat.sum() == 2
    assert targets.meter[:25].eq(encode_meter(4, 4)).all()
    assert targets.meter[25:].eq(encode_meter(3, 4)).all()
    # A crop inside the new meter still carries the effective state.
    cropped = crop_symbolic_sequence(sequence, 30, 10)
    meter_events = cropped.token_type_ids == int(SymbolicTokenType.METER)
    assert cropped.position_ids[meter_events].tolist() == [0]
    assert cropped.token_ids[meter_events].tolist() == [METER_TOKEN_BASE + encode_meter(3, 4)]


def test_symbolic_rvq_reconstruction_ties_codes_to_frame_states():
    torch.manual_seed(0)
    config = _small_config()
    teacher = SymbolicTeacher(config)
    output = teacher(
        torch.tensor([[FRAME_TOKEN_ID, 2, FRAME_TOKEN_ID]]),
        anchor_positions=torch.tensor([[0, 2]]),
        token_instrument_ids=torch.tensor([[NO_INSTRUMENT_ID, 21, NO_INSTRUMENT_ID]]),
        token_type_ids=torch.tensor(
            [[int(SymbolicTokenType.CONTROL), int(SymbolicTokenType.NOTE), int(SymbolicTokenType.CONTROL)]]
        ),
        position_ids=torch.tensor([[0, 0, 1]]),
    )
    assert torch.allclose(output.rvq_reconstruction_loss, (output.quantized - output.frame_hidden).abs().mean())

    targets = SymbolicFrameTargets(
        note_activity=torch.zeros(1, 2, config.symbolic_pitch_classes),
        note_onset=torch.zeros(1, 2, config.symbolic_pitch_classes),
        note_offset=torch.zeros(1, 2, config.symbolic_pitch_classes),
        instrument_activity=torch.zeros(1, 2, config.symbolic_instrument_classes),
        beat=torch.ones(1, 2),
        downbeat=torch.zeros(1, 2),
        chord=torch.zeros(1, 2, dtype=torch.long),
        bass=torch.zeros(1, 2, dtype=torch.long),
        key=torch.zeros(1, 2, dtype=torch.long),
        meter=torch.zeros(1, 2, dtype=torch.long),
    )
    unweighted = SymbolicTeacherLoss()(output, targets)
    weighted = SymbolicTeacherLoss(rvq_reconstruction_weight=2.0)(output, targets)
    # The distance is reported either way but only counts when weighted.
    assert torch.equal(unweighted["loss_rvq_reconstruction"], output.rvq_reconstruction_loss)
    assert torch.allclose(weighted["loss_total"] - unweighted["loss_total"], 2.0 * output.rvq_reconstruction_loss)

    output.rvq_reconstruction_loss.backward()
    # Without this gradient the output projections are free to drift away
    # from the states they are meant to rebuild.
    for projection in teacher.rvq.output_projections:
        assert projection.parametrizations.weight.original1.grad.abs().sum() > 0

