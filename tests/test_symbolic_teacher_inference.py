from train.symbolic_teacher.run_inference import (
    DEFAULT_MIDI_RESOLUTION,
    _resolve_midi_resolution,
)


def test_long_audio_uses_a_parseable_midi_resolution():
    resolution = _resolve_midi_resolution(74.6 * 60.0, None)

    assert resolution < DEFAULT_MIDI_RESOLUTION
    assert resolution >= 96


def test_short_audio_keeps_default_and_explicit_resolution_wins():
    assert _resolve_midi_resolution(180.0, None) == DEFAULT_MIDI_RESOLUTION
    assert _resolve_midi_resolution(180.0, 480) == 480
