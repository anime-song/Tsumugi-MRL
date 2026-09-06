"""MIDI event tokens and 25 Hz frame targets.

The token IDs below are only an input format.  The useful symbolic
representation is learned by :class:`SymbolicEncoder` and its RVQ bottleneck.
One ``FRAME`` token is inserted at the beginning of every 25 Hz frame.  The
hidden state at that position is the frame query; event tokens in the same
sequence are allowed to influence it through self-attention.

The MIDI reader intentionally uses the information written by the normal
Tsumugi inference pipeline: notes/instruments, tempo and time signatures,
key-signature messages, and chord marker messages. Beat events are restored
from the MIDI tempo map and time signatures. Chord marker syntax is parsed by
``dlchordx`` and mapped to the reference ``quality.json`` vocabulary; slash
bass is emitted as a separate pitch-class state token.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from math import ceil, floor
from pathlib import Path
from typing import Iterable, Optional, Sequence

import mido
import torch
from dlchordx import Chord
from torch import Tensor
from torch.utils.data import Dataset


class SymbolicTokenType(IntEnum):
    """Small type vocabulary for the symbolic input interface."""

    PAD = 0
    NOTE = 1  # note-on and pitch
    BEAT = 2
    DOWNBEAT = 3
    CHORD = 4
    KEY = 5
    INSTRUMENT = 6
    CONTROL = 7  # FRAME and carried note state (NOTE_ON ID, not an onset)
    NOTE_OFF = 8
    METER = 9
    BASS = 10


# Fixed token layout. Keeping this layout deterministic makes token caches
# portable across processes and avoids a dataset-specific symbolic vocabulary.
PAD_TOKEN_ID = 0
FRAME_TOKEN_ID = 1
MIN_MIDI_PITCH = 21  # A0, the lowest key on an 88-key piano
MAX_MIDI_PITCH = 108  # C8, the highest key on an 88-key piano
PITCH_CLASSES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1
NOTE_ON_TOKEN_BASE = 2  # 88 MIDI piano pitches
NOTE_OFF_TOKEN_BASE = NOTE_ON_TOKEN_BASE + PITCH_CLASSES
BEAT_TOKEN_ID = NOTE_OFF_TOKEN_BASE + PITCH_CLASSES
DOWNBEAT_TOKEN_ID = BEAT_TOKEN_ID + 1
CHORD_TOKEN_BASE = DOWNBEAT_TOKEN_ID + 1

# This is the exact quality vocabulary used by
# ``instrument_agnostic_amt/beat_chord_dataset/chord_dataset/quality.json``.
# The final class is the reference model's no-chord class.  In particular,
# ``""`` is a major triad and ``"M7"`` is not written as ``"maj7"``.
CHORD_QUALITIES: tuple[str, ...] = (
    "5",
    "",
    "-5",
    "m",
    "dim",
    "aug",
    "sus2",
    "sus4",
    "6",
    "7",
    "M7",
    "m6",
    "m7",
    "mM7",
    "7-5",
    "m7-5",
    "aug7",
    "augM7",
    "7sus4",
    "dim7",
    "add9",
    "madd9",
    "69",
    "7(9)",
    "7(13)",
    "7(b9)",
    "7(#9)",
    "7(#11)",
    "7(b13)",
    "7-5(b13)",
    "M7(9)",
    "M7(b9)",
    "M7(13)",
    "M7(#11)",
    "m69",
    "m7(9)",
    "m7(11)",
    "m7(13)",
    "m7(b9)",
    "mM7(9)",
    "mM7(13)",
    "aug7(9)",
    "augM7(#9)",
    "add9(#11)",
    "m7-5(b9)",
    "7(9,13)",
    "7(9,b13)",
    "7(9,#11)",
    "7(b9,#9)",
    "7(b9,13)",
    "7(b9,b13)",
    "7(b9,#11)",
    "7(#9,13)",
    "7(#9,b13)",
    "7(#11,13)",
    "m7(9,11)",
    "m7(9,13)",
    "M7(9,13)",
    "M7(9,#11)",
    "7(9,#11,13)",
    "7(9,#11,b13)",
    "M7(9,#11,13)",
    "N",
)
CHORD_QUALITY_TO_ID = {name: index for index, name in enumerate(CHORD_QUALITIES)}
CHORD_NO_CHORD_QUALITY_ID = CHORD_QUALITY_TO_ID["N"]
CHORD_QUALITY_COUNT = len(CHORD_QUALITIES)
CHORD_CLASSES = 12 * (CHORD_QUALITY_COUNT - 1) + 1
CHORD_NO_CHORD_ID = CHORD_CLASSES - 1
BASS_TOKEN_BASE = CHORD_TOKEN_BASE + CHORD_CLASSES
BASS_CLASSES = 13  # 12 pitch classes plus an explicit no-bass/no-chord state
BASS_NO_BASS_ID = BASS_CLASSES - 1
KEY_TOKEN_BASE = BASS_TOKEN_BASE + BASS_CLASSES
KEY_CLASSES = 25  # 12 major + 12 minor + unknown
INSTRUMENT_TOKEN_BASE = KEY_TOKEN_BASE + KEY_CLASSES

# ``instrument_classes.py`` in the reference repository contains the 33
# merged labels from ``instrument_merge.json`` plus the three inference-only
# classes appended at runtime (melody, vocal_harmony, wind_chimes).
INSTRUMENT_CLASS_NAMES: tuple[str, ...] = (
    "accordion_family",
    "acoustic_bass",
    "acoustic_guitar",
    "brass",
    "choir",
    "chromatic_percussion",
    "distorted_guitar",
    "drums",
    "electric_bass",
    "electric_guitar_clean",
    "electric_guitar_muted",
    "electric_piano",
    "ethnic",
    "flute_pipe",
    "guitar_harmonics",
    "harmonica",
    "orchestra_hit",
    "orchestral_harp",
    "orchestral_woodwind",
    "organ",
    "percussive_fx",
    "piano",
    "pizzicato_strings",
    "plucked_keyboard",
    "sax",
    "slap_bass",
    "sound_fx",
    "strings",
    "synth_bass",
    "synth_fx",
    "synth_lead",
    "synth_pad",
    "timpani",
    "melody",
    "vocal_harmony",
    "wind_chimes",
)
INSTRUMENT_CLASSES = len(INSTRUMENT_CLASS_NAMES)
NO_INSTRUMENT_ID = INSTRUMENT_CLASSES
SYMBOLIC_CACHE_VERSION = 4
INSTRUMENT_NAME_TO_ID = {name: index for index, name in enumerate(INSTRUMENT_CLASS_NAMES)}

# The fallback from a MIDI GM program to the reference merged taxonomy.  A
# Tsumugi MIDI normally carries the class name in its track name; this table
# handles ordinary MIDI tracks that only carry a program_change message.
_PROGRAM_TO_INSTRUMENT_CLASS: tuple[int, ...] = (
    21,
    21,
    11,
    21,
    11,
    11,
    23,
    23,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    19,
    19,
    19,
    19,
    19,
    0,
    15,
    0,
    2,
    2,
    9,
    9,
    10,
    6,
    6,
    14,
    1,
    8,
    8,
    8,
    25,
    25,
    28,
    28,
    27,
    27,
    27,
    27,
    27,
    22,
    17,
    32,
    27,
    27,
    27,
    27,
    4,
    4,
    4,
    4,
    16,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    24,
    24,
    24,
    24,
    18,
    18,
    18,
    18,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    13,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    30,
    31,
    31,
    31,
    31,
    31,
    31,
    31,
    31,
    29,
    29,
    29,
    29,
    29,
    29,
    29,
    29,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
    26,
)
DRUMS_INSTRUMENT_ID = INSTRUMENT_NAME_TO_ID["drums"]

# The reference beat/chord checkpoint contains these 60 meter classes.  Keep
# the same sorted order so meter IDs are stable across tokenization runs.
_METER_VALUES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (2, 2),
    (2, 4),
    (2, 8),
    (3, 2),
    (3, 4),
    (3, 8),
    (3, 16),
    (4, 2),
    (4, 4),
    (4, 8),
    (4, 16),
    (5, 2),
    (5, 4),
    (5, 8),
    (5, 16),
    (6, 4),
    (6, 8),
    (6, 16),
    (7, 4),
    (7, 8),
    (7, 16),
    (8, 4),
    (8, 8),
    (8, 16),
    (9, 4),
    (9, 8),
    (9, 16),
    (10, 4),
    (10, 8),
    (10, 16),
    (11, 4),
    (11, 8),
    (11, 16),
    (12, 4),
    (12, 8),
    (12, 16),
    (13, 4),
    (13, 8),
    (13, 16),
    (14, 4),
    (14, 8),
    (14, 16),
    (15, 8),
    (15, 16),
    (16, 8),
    (17, 8),
    (17, 16),
    (18, 8),
    (18, 16),
    (19, 16),
    (21, 8),
    (21, 16),
    (21, 32),
    (22, 16),
    (25, 16),
    (27, 16),
)
METER_CLASSES = len(_METER_VALUES) + 1  # plus one unknown-meter class
METER_TOKEN_BASE = INSTRUMENT_TOKEN_BASE + INSTRUMENT_CLASSES
SYMBOLIC_REQUIRED_VOCAB_SIZE = METER_TOKEN_BASE + METER_CLASSES

UNKNOWN_KEY_ID = 24
UNKNOWN_METER_ID = METER_CLASSES - 1

_PITCH_CLASS = {
    name: index for index, name in enumerate(("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"))
}
_PITCH_CLASS.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})
_KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b]?)\s*(m|min|major|minor)?\s*$", re.IGNORECASE)

_METER_TO_ID = {value: index for index, value in enumerate(_METER_VALUES)}


def encode_chord_and_bass(label: str) -> tuple[int, int]:
    """Parse a chord label into separate chord and bass pitch-class IDs.

    ``dlchordx`` treats a chord without an explicit slash bass as root bass,
    which is the useful default for the symbolic state representation.  The
    no-chord label clears the carried bass with ``BASS_NO_BASS_ID``.
    """

    text = label.strip()
    if not text or text.upper() in {"N", "NO_CHORD", "NONE"}:
        return CHORD_NO_CHORD_ID, BASS_NO_BASS_ID

    chord = Chord(text)
    root = int(chord.root.get_interval()) % 12
    quality = str(chord.quality.name).replace(" ", "").removeprefix(":")
    quality_id = CHORD_QUALITY_TO_ID[quality]
    bass = int(chord.bass.get_interval()) % 12
    return root * (CHORD_QUALITY_COUNT - 1) + quality_id, bass


def encode_chord_label(label: str) -> int:
    """Parse a Tsumugi chord with dlchordx and return its class ID."""

    return encode_chord_and_bass(label)[0]


def encode_key_name(name: str) -> int:
    """Map a MIDI key-signature name to ``root * 2 + mode``."""

    match = _KEY_RE.match(name)
    if match is None:
        return UNKNOWN_KEY_ID
    root = _PITCH_CLASS.get(match.group(1).upper() + match.group(2), None)
    if root is None:
        return UNKNOWN_KEY_ID
    mode = 1 if (match.group(3) or "").lower() in {"m", "min", "minor"} else 0
    return 2 * root + mode


def encode_meter(numerator: int, denominator: int) -> int:
    return _METER_TO_ID.get((numerator, denominator), UNKNOWN_METER_ID)


def _instrument_id_for_track(track_name: str, program: int, *, is_drum: bool) -> int:
    """Resolve a MIDI track to the reference merged instrument taxonomy."""

    if is_drum:
        return DRUMS_INSTRUMENT_ID

    normalized_name = "_".join(track_name.strip().casefold().replace("-", " ").split())
    return INSTRUMENT_NAME_TO_ID.get(normalized_name, _PROGRAM_TO_INSTRUMENT_CLASS[program])


@dataclass(frozen=True)
class SymbolicFrameTargets:
    """Frame-level labels used only to train the symbolic teacher."""

    note_activity: Tensor  # [T, 88]
    note_onset: Tensor  # [T, 88]
    note_offset: Tensor  # [T, 88]
    instrument_activity: Tensor  # [T, 36], reference merged taxonomy
    beat: Tensor  # [T]
    downbeat: Tensor  # [T]
    chord: Tensor  # [T], -100 means unavailable
    bass: Tensor  # [T], -100 means unavailable; 12 means no bass/no chord
    key: Tensor  # [T], -100 means unavailable
    meter: Tensor  # [T], -100 means unavailable

    @property
    def num_frames(self) -> int:
        return int(self.beat.shape[0])

    def to(self, device: torch.device | str) -> "SymbolicFrameTargets":
        return SymbolicFrameTargets(*(value.to(device) for value in self.__dict__.values()))

    def as_dict(self) -> dict[str, Tensor]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SymbolicSequence:
    """One variable-length event sequence plus aligned frame targets."""

    token_ids: Tensor  # [L]
    token_type_ids: Tensor  # [L]
    token_instrument_ids: Tensor  # [L], instrument bound to each ON/OFF/held event
    position_ids: Tensor  # [L], frame index for RoPE
    anchor_positions: Tensor  # [T], indices into the event sequence
    frame_targets: SymbolicFrameTargets
    duration_seconds: float
    note_spans: Tensor  # [N, 4]: start frame, end frame (inclusive), pitch index, instrument

    @property
    def num_frames(self) -> int:
        return int(self.anchor_positions.numel())

    def to(self, device: torch.device | str) -> "SymbolicSequence":
        return SymbolicSequence(
            token_ids=self.token_ids.to(device),
            token_type_ids=self.token_type_ids.to(device),
            token_instrument_ids=self.token_instrument_ids.to(device),
            position_ids=self.position_ids.to(device),
            anchor_positions=self.anchor_positions.to(device),
            frame_targets=self.frame_targets.to(device),
            duration_seconds=self.duration_seconds,
            note_spans=self.note_spans.to(device),
        )


@dataclass(frozen=True)
class _MidiNote:
    start_tick: int
    end_tick: int
    pitch: int
    instrument: int


class MIDIEventTokenizer:
    """Turn a Tsumugi output MIDI into ``FRAME_t``-anchored tokens."""

    def __init__(self, frame_rate: float = 25.0) -> None:
        self.frame_rate = float(frame_rate)

    def encode(self, midi_path: str | Path, num_frames: Optional[int] = None) -> SymbolicSequence:
        midi = mido.MidiFile(str(midi_path), clip=True)

        tracks: list[tuple[str, list[tuple[int, object]]]] = []
        max_tick = 0
        tempo_by_tick: dict[int, int] = {0: 500_000}
        time_signatures: list[tuple[int, int, int]] = []
        key_events: list[tuple[int, int]] = []
        chord_events: list[tuple[int, int, int]] = []
        program_events: list[tuple[int, int]] = []
        notes: list[_MidiNote] = []

        for track in midi.tracks:
            absolute_tick = 0
            entries: list[tuple[int, object]] = []
            track_name = ""
            for message in track:
                absolute_tick += message.time
                entries.append((absolute_tick, message))
                max_tick = max(max_tick, absolute_tick)
                message_type = message.type
                if message_type == "track_name":
                    track_name = message.name
                if message_type == "set_tempo":
                    tempo_by_tick[absolute_tick] = message.tempo
                elif message_type == "time_signature":
                    time_signatures.append((absolute_tick, message.numerator, message.denominator))
                elif message_type == "key_signature":
                    key_events.append((absolute_tick, encode_key_name(message.key)))
                elif message_type == "marker":
                    marker_text = message.text
                    if "chord" in track_name.casefold():
                        chord_id, bass_id = encode_chord_and_bass(marker_text)
                        chord_events.append((absolute_tick, chord_id, bass_id))
            tracks.append((track_name, entries))

        tempo_ticks = sorted(tempo_by_tick)
        tempo_seconds = [0.0]
        for previous_tick, current_tick in zip(tempo_ticks, tempo_ticks[1:]):
            previous_tempo = tempo_by_tick[previous_tick]
            tempo_seconds.append(
                tempo_seconds[-1]
                + (current_tick - previous_tick) * previous_tempo / (midi.ticks_per_beat * 1_000_000.0)
            )

        def tick_to_seconds(tick: int) -> float:
            index = max(0, bisect_right(tempo_ticks, tick) - 1)
            return tempo_seconds[index] + (tick - tempo_ticks[index]) * tempo_by_tick[tempo_ticks[index]] / (
                midi.ticks_per_beat * 1_000_000.0
            )

        for track_name, entries in tracks:
            program = 0
            active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
            track_is_drum = "drum" in track_name.casefold()
            for tick, message in entries:
                message_type = message.type
                if message_type == "program_change":
                    program = message.program
                    instrument = _instrument_id_for_track(
                        track_name,
                        program,
                        is_drum=track_is_drum or message.channel == 9,
                    )
                    program_events.append((tick, instrument))
                    continue
                if message_type not in {"note_on", "note_off"}:
                    continue
                pitch = message.note
                channel = message.channel
                instrument = _instrument_id_for_track(
                    track_name,
                    program,
                    is_drum=track_is_drum or channel == 9,
                )
                key = (channel, pitch)
                if message_type == "note_on" and message.velocity > 0:
                    active[key].append((tick, instrument))
                    # The note event already carries its instrument ID.
                else:
                    pending = active.get(key)
                    if not pending:
                        continue
                    start_tick, start_instrument = pending.pop(0)
                    notes.append(
                        _MidiNote(
                            start_tick=start_tick,
                            end_tick=max(start_tick + 1, tick),
                            pitch=pitch,
                            instrument=start_instrument,
                        )
                    )
            for (_channel, pitch), pending_notes in active.items():
                for start_tick, start_instrument in pending_notes:
                    notes.append(
                        _MidiNote(
                            start_tick=start_tick,
                            end_tick=max(start_tick + 1, max_tick),
                            pitch=pitch,
                            instrument=start_instrument,
                        )
                    )

        if max_tick <= 0:
            max_tick = max((note.end_tick for note in notes), default=1)
        duration_seconds = max(tick_to_seconds(max_tick), 1.0 / self.frame_rate)
        frame_count = max(1, int(num_frames) if num_frames is not None else ceil(duration_seconds * self.frame_rate))
        return self._build_sequence(
            notes=notes,
            time_signatures=time_signatures,
            key_events=key_events,
            chord_events=chord_events,
            program_events=program_events,
            max_tick=max_tick,
            tick_to_seconds=tick_to_seconds,
            ticks_per_beat=midi.ticks_per_beat,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
        )

    def _build_sequence(
        self,
        *,
        notes: Sequence[_MidiNote],
        time_signatures: Sequence[tuple[int, int, int]],
        key_events: Sequence[tuple[int, int]],
        chord_events: Sequence[tuple[int, int, int]],
        program_events: Sequence[tuple[int, int]],
        max_tick: int,
        tick_to_seconds,
        ticks_per_beat: int,
        frame_count: int,
        duration_seconds: float,
    ) -> SymbolicSequence:
        events_by_frame: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        note_spans: list[tuple[int, int, int, int]] = []
        note_activity = torch.zeros(frame_count, PITCH_CLASSES)
        note_onset = torch.zeros_like(note_activity)
        note_offset = torch.zeros_like(note_activity)
        instrument_activity = torch.zeros(frame_count, INSTRUMENT_CLASSES)
        beat = torch.zeros(frame_count)
        downbeat = torch.zeros(frame_count)
        chord = torch.full((frame_count,), -100, dtype=torch.long)
        bass = torch.full((frame_count,), -100, dtype=torch.long)
        key = torch.full((frame_count,), -100, dtype=torch.long)
        meter = torch.full((frame_count,), -100, dtype=torch.long)

        def frame_for_tick(tick: int) -> Optional[int]:
            frame = int(floor(tick_to_seconds(tick) * self.frame_rate))
            if frame < 0 or frame >= frame_count:
                return None
            return frame

        def add_event(
            frame: Optional[int],
            priority: int,
            token_type: SymbolicTokenType,
            token_id: int,
            instrument: int = NO_INSTRUMENT_ID,
        ) -> None:
            if frame is not None:
                events_by_frame[frame].append((priority, int(token_type), int(token_id), instrument))

        for note in notes:
            if not MIN_MIDI_PITCH <= note.pitch <= MAX_MIDI_PITCH:
                continue
            pitch_index = note.pitch - MIN_MIDI_PITCH
            start_frame = frame_for_tick(note.start_tick)
            end_frame = frame_for_tick(note.end_tick)
            if start_frame is None and end_frame is None:
                continue
            start_frame = 0 if start_frame is None else start_frame
            end_frame = frame_count - 1 if end_frame is None else max(start_frame, end_frame)
            active_end = min(frame_count, max(start_frame + 1, end_frame + 1))
            note_activity[start_frame:active_end, pitch_index] = 1.0
            instrument_activity[start_frame:active_end, note.instrument] = 1.0
            note_onset[start_frame, pitch_index] = 1.0
            note_offset[end_frame, pitch_index] = 1.0
            note_spans.append((start_frame, end_frame, pitch_index, note.instrument))
            add_event(start_frame, 20, SymbolicTokenType.NOTE, NOTE_ON_TOKEN_BASE + pitch_index, note.instrument)
            add_event(end_frame, 10, SymbolicTokenType.NOTE_OFF, NOTE_OFF_TOKEN_BASE + pitch_index, note.instrument)

        for tick, instrument in program_events:
            add_event(frame_for_tick(tick), 40, SymbolicTokenType.INSTRUMENT, INSTRUMENT_TOKEN_BASE + instrument)

        # The beat/chord inference writes a tempo map and time signatures to
        # the MIDI. A quarter/eighth-note grid from that map is the compact
        # beat event representation used by the symbolic teacher.
        sorted_meters = sorted(time_signatures) or [(0, 4, 4)]
        meter_ticks = [item[0] for item in sorted_meters]
        beat_tick = 0
        beat_number = 0
        while beat_tick <= max_tick:
            meter_index = max(0, bisect_right(meter_ticks, beat_tick) - 1)
            numerator, denominator = sorted_meters[meter_index][1:]
            current_meter_id = encode_meter(numerator, denominator)
            beat_frame = frame_for_tick(beat_tick)
            add_event(beat_frame, 50, SymbolicTokenType.BEAT, BEAT_TOKEN_ID)
            if beat_frame is not None:
                beat[beat_frame] = 1.0
            if beat_number % max(1, numerator) == 0:
                add_event(beat_frame, 51, SymbolicTokenType.DOWNBEAT, DOWNBEAT_TOKEN_ID)
                if beat_frame is not None:
                    downbeat[beat_frame] = 1.0
            if beat_frame is not None:
                meter[beat_frame] = current_meter_id
            # Meter tokens are emitted at initialization/changes below, not every beat.
            step = int(round(ticks_per_beat * 4.0 / denominator))
            next_meter_tick = next((tick for tick in meter_ticks if tick > beat_tick), None)
            if next_meter_tick is not None and beat_tick + step >= next_meter_tick:
                beat_tick = next_meter_tick
                beat_number = 0
            else:
                beat_tick += step
                beat_number += 1

        def add_state_events(
            raw_events: Iterable[tuple[int, int]], token_type: SymbolicTokenType, token_base: int, target: Tensor
        ) -> None:
            for tick, value in sorted(raw_events):
                frame = frame_for_tick(tick)
                add_event(frame, 60, token_type, token_base + int(value))
                if frame is not None:
                    target[frame:] = int(value)

        for tick, chord_value, bass_value in sorted(chord_events):
            frame = frame_for_tick(tick)
            add_event(frame, 60, SymbolicTokenType.CHORD, CHORD_TOKEN_BASE + int(chord_value))
            add_event(frame, 61, SymbolicTokenType.BASS, BASS_TOKEN_BASE + int(bass_value))
            if frame is not None:
                chord[frame:] = int(chord_value)
                bass[frame:] = int(bass_value)
        add_state_events(key_events, SymbolicTokenType.KEY, KEY_TOKEN_BASE, key)
        add_state_events(
            [(tick, encode_meter(numerator, denominator)) for tick, numerator, denominator in sorted_meters],
            SymbolicTokenType.METER,
            METER_TOKEN_BASE,
            meter,
        )

        token_ids: list[int] = []
        token_types: list[int] = []
        token_instruments: list[int] = []
        position_ids: list[int] = []
        anchor_positions: list[int] = []
        for frame_index in range(frame_count):
            anchor_positions.append(len(token_ids))
            token_ids.append(FRAME_TOKEN_ID)
            token_types.append(int(SymbolicTokenType.CONTROL))
            token_instruments.append(NO_INSTRUMENT_ID)
            position_ids.append(frame_index)
            for _priority, token_type, token_id, instrument in sorted(events_by_frame.get(frame_index, [])):
                token_ids.append(token_id)
                token_types.append(token_type)
                token_instruments.append(instrument)
                position_ids.append(frame_index)

        frame_targets = SymbolicFrameTargets(
            note_activity=note_activity,
            note_onset=note_onset,
            note_offset=note_offset,
            instrument_activity=instrument_activity,
            beat=beat,
            downbeat=downbeat,
            chord=chord,
            bass=bass,
            key=key,
            meter=meter,
        )
        return SymbolicSequence(
            token_ids=torch.tensor(token_ids, dtype=torch.long),
            token_type_ids=torch.tensor(token_types, dtype=torch.long),
            token_instrument_ids=torch.tensor(token_instruments, dtype=torch.long),
            position_ids=torch.tensor(position_ids, dtype=torch.long),
            anchor_positions=torch.tensor(anchor_positions, dtype=torch.long),
            frame_targets=frame_targets,
            duration_seconds=float(duration_seconds),
            note_spans=torch.tensor(note_spans, dtype=torch.long).reshape(-1, 4),
        )


def crop_symbolic_sequence(
    sequence: SymbolicSequence,
    start_frame: int,
    num_frames: int,
    frame_rate: float = 25.0,
) -> SymbolicSequence:
    """Cut a frame window out of a sequence without losing carried-over state.

    Frame targets are per-frame, so they only need slicing.  The event stream
    does not survive a plain slice: a note that started before ``start_frame``
    has its ``NOTE_ON`` outside the window, and chord, bass, key, meter, and
    instrument events are only written where they change.  The window
    therefore opens with carry-in tokens describing what is already sounding
    and which chord/key/meter is in effect at ``start_frame``, so a cropped
    window carries the same information as the same frames of the full song.

    Notes that continue past the window simply lose their ``NOTE_OFF``; the
    frame targets inside the window are unaffected by that.
    """

    total_frames = sequence.num_frames
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame must be in [0, {total_frames}), got {start_frame}.")
    end_frame = min(total_frames, start_frame + num_frames)

    anchors = sequence.anchor_positions
    start_token = int(anchors[start_frame])
    end_token = int(anchors[end_frame]) if end_frame < total_frames else int(sequence.token_ids.numel())

    targets = sequence.frame_targets
    carry_ids: list[int] = []
    carry_types: list[int] = []
    carry_instruments: list[int] = []
    if start_frame > 0:
        # Per-note spans preserve a held instrument even when another
        # instrument starts the same pitch at the crop boundary.
        spans = sequence.note_spans
        held = spans[(spans[:, 0] < start_frame) & (spans[:, 1] >= start_frame)]
        for _, _, pitch_index, instrument in held.tolist():
            # A carried pitch is active, but is not a new onset at the boundary.
            # Reuse the pitch ID with CONTROL type to keep cache IDs and
            # existing checkpoint vocabulary sizes compatible.
            carry_ids.append(NOTE_ON_TOKEN_BASE + int(pitch_index))
            carry_types.append(int(SymbolicTokenType.CONTROL))
            carry_instruments.append(instrument)
        for instrument in (targets.instrument_activity[start_frame] > 0).nonzero(as_tuple=False).flatten().tolist():
            carry_ids.append(INSTRUMENT_TOKEN_BASE + int(instrument))
            carry_types.append(int(SymbolicTokenType.INSTRUMENT))
            carry_instruments.append(NO_INSTRUMENT_ID)
        for value, token_base, token_type in (
            (int(targets.chord[start_frame]), CHORD_TOKEN_BASE, SymbolicTokenType.CHORD),
            (int(targets.bass[start_frame]), BASS_TOKEN_BASE, SymbolicTokenType.BASS),
            (int(targets.key[start_frame]), KEY_TOKEN_BASE, SymbolicTokenType.KEY),
            (int(targets.meter[start_frame]), METER_TOKEN_BASE, SymbolicTokenType.METER),
        ):
            if value >= 0:
                carry_ids.append(token_base + value)
                carry_types.append(int(token_type))
                carry_instruments.append(NO_INSTRUMENT_ID)

    window_ids = sequence.token_ids[start_token:end_token]
    window_types = sequence.token_type_ids[start_token:end_token]
    window_instruments = sequence.token_instrument_ids[start_token:end_token]
    window_positions = sequence.position_ids[start_token:end_token] - start_frame

    carry_count = len(carry_ids)
    if carry_count:
        # Keep the ``FRAME_t, event, event, ...`` layout: the carry-in tokens
        # belong to the first frame and therefore follow its anchor.
        token_ids = torch.cat((window_ids[:1], torch.tensor(carry_ids, dtype=window_ids.dtype), window_ids[1:]))
        token_type_ids = torch.cat(
            (window_types[:1], torch.tensor(carry_types, dtype=window_types.dtype), window_types[1:])
        )
        token_instrument_ids = torch.cat(
            (window_instruments[:1], window_instruments.new_tensor(carry_instruments), window_instruments[1:])
        )
        position_ids = torch.cat(
            (
                window_positions[:1],
                torch.zeros(carry_count, dtype=window_positions.dtype),
                window_positions[1:],
            )
        )
    else:
        token_ids = window_ids.clone()
        token_type_ids = window_types.clone()
        token_instrument_ids = window_instruments.clone()
        position_ids = window_positions

    # The first anchor is always 0 and stays there; every later anchor moves
    # by the number of inserted carry-in tokens.
    relative_anchors = anchors[start_frame:end_frame] - start_token
    anchor_positions = torch.where(relative_anchors > 0, relative_anchors + carry_count, relative_anchors)

    frame_window = slice(start_frame, end_frame)
    frame_targets = SymbolicFrameTargets(
        **{name: value[frame_window].clone() for name, value in targets.as_dict().items()}
    )
    spans = sequence.note_spans
    spans = spans[(spans[:, 0] < end_frame) & (spans[:, 1] >= start_frame)].clone()
    spans[:, :2] -= start_frame  # Negative start preserves carry state for a subsequent crop.
    return SymbolicSequence(
        token_ids=token_ids,
        token_type_ids=token_type_ids,
        token_instrument_ids=token_instrument_ids,
        position_ids=position_ids,
        anchor_positions=anchor_positions,
        frame_targets=frame_targets,
        duration_seconds=(end_frame - start_frame) / float(frame_rate),
        note_spans=spans,
    )


def random_crop_symbolic_sequence(
    sequence: SymbolicSequence,
    num_frames: int,
    frame_rate: float = 25.0,
    generator: Optional[torch.Generator] = None,
) -> SymbolicSequence:
    """Crop a uniformly random window, or return the sequence when too short."""

    total_frames = sequence.num_frames
    if num_frames <= 0 or total_frames <= num_frames:
        return sequence
    start_frame = int(torch.randint(0, total_frames - num_frames + 1, (1,), generator=generator).item())
    return crop_symbolic_sequence(sequence, start_frame, num_frames, frame_rate=frame_rate)


class SymbolicMIDIDataset(Dataset[SymbolicSequence]):
    """Small on-the-fly dataset for symbolic-teacher training.

    ``crop_frames`` returns a random window instead of the whole song, which
    keeps the quadratic event self-attention affordable.
    """

    def __init__(
        self,
        midi_dir: str | Path,
        frame_rate: float = 25.0,
        crop_frames: Optional[int] = None,
    ) -> None:
        self.paths = sorted(
            path for path in Path(midi_dir).rglob("*") if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No MIDI files found under {midi_dir}.")
        if crop_frames is not None and crop_frames <= 0:
            raise ValueError("crop_frames must be positive when given.")
        self.frame_rate = float(frame_rate)
        self.crop_frames = crop_frames
        self.tokenizer = MIDIEventTokenizer(frame_rate=frame_rate)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> SymbolicSequence:
        sequence = self.tokenizer.encode(self.paths[index])
        if self.crop_frames is None:
            return sequence
        return random_crop_symbolic_sequence(sequence, self.crop_frames, frame_rate=self.frame_rate)


def load_symbolic_cache(path: str | Path) -> SymbolicSequence:
    """Load a cache written by ``prepare_symbolic_dataset.py``."""

    payload = torch.load(path, map_location="cpu")
    if payload.get("cache_version") != SYMBOLIC_CACHE_VERSION:
        raise ValueError(
            f"{path}: outdated symbolic cache; current format is version {SYMBOLIC_CACHE_VERSION}. "
            "Rebuild from existing MIDI with python -m train.symbolic_teacher.prepare_dataset "
            "--retokenize (no AMT rerun is needed for completed MIDI)."
        )
    return SymbolicSequence(
        token_ids=payload["token_ids"],
        token_type_ids=payload["token_type_ids"],
        token_instrument_ids=payload["token_instrument_ids"],
        position_ids=payload["position_ids"],
        anchor_positions=payload["anchor_positions"],
        frame_targets=SymbolicFrameTargets(**payload["frame_targets"]),
        duration_seconds=float(payload["duration_seconds"]),
        note_spans=payload["note_spans"],
    )


class CachedSymbolicDataset(Dataset[SymbolicSequence]):
    """Dataset for the precomputed ``.pt`` symbolic caches.

    This avoids re-parsing every MIDI file once per epoch.  ``crop_frames``
    returns a random window instead of the whole song, which keeps the
    quadratic event self-attention affordable.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        frame_rate: float = 25.0,
        crop_frames: Optional[int] = None,
    ) -> None:
        self.paths = sorted(Path(cache_dir).rglob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No symbolic caches found under {cache_dir}.")
        if crop_frames is not None and crop_frames <= 0:
            raise ValueError("crop_frames must be positive when given.")
        self.frame_rate = float(frame_rate)
        self.crop_frames = crop_frames

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> SymbolicSequence:
        sequence = load_symbolic_cache(self.paths[index])
        if self.crop_frames is None:
            return sequence
        return random_crop_symbolic_sequence(sequence, self.crop_frames, frame_rate=self.frame_rate)


def collate_symbolic_sequences(samples: Sequence[SymbolicSequence]) -> dict[str, object]:
    """Pad event and frame dimensions for a DataLoader."""

    if not samples:
        raise ValueError("samples must not be empty.")
    batch_size = len(samples)
    max_length = max(int(sample.token_ids.numel()) for sample in samples)
    max_frames = max(sample.num_frames for sample in samples)
    token_ids = torch.full((batch_size, max_length), PAD_TOKEN_ID, dtype=torch.long)
    token_type_ids = torch.full((batch_size, max_length), int(SymbolicTokenType.PAD), dtype=torch.long)
    token_instrument_ids = torch.full((batch_size, max_length), NO_INSTRUMENT_ID, dtype=torch.long)
    position_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
    anchor_positions = torch.zeros((batch_size, max_frames), dtype=torch.long)
    frame_padding_mask = torch.ones((batch_size, max_frames), dtype=torch.bool)

    float_names = (
        "note_activity",
        "note_onset",
        "note_offset",
        "instrument_activity",
        "beat",
        "downbeat",
    )
    long_names = ("chord", "bass", "key", "meter")
    target_values: dict[str, Tensor] = {}
    first_targets = samples[0].frame_targets.as_dict()
    for name in float_names:
        source = first_targets[name]
        target_values[name] = torch.zeros((batch_size, max_frames, *source.shape[1:]), dtype=source.dtype)
    for name in long_names:
        source = first_targets[name]
        target_values[name] = torch.full((batch_size, max_frames, *source.shape[1:]), -100, dtype=source.dtype)

    for batch_index, sample in enumerate(samples):
        length = sample.token_ids.numel()
        frames = sample.num_frames
        token_ids[batch_index, :length] = sample.token_ids
        token_type_ids[batch_index, :length] = sample.token_type_ids
        token_instrument_ids[batch_index, :length] = sample.token_instrument_ids
        position_ids[batch_index, :length] = sample.position_ids
        padding_mask[batch_index, :length] = False
        anchor_positions[batch_index, :frames] = sample.anchor_positions
        frame_padding_mask[batch_index, :frames] = False
        sample_targets = sample.frame_targets.as_dict()
        for name in (*float_names, *long_names):
            target_values[name][batch_index, :frames] = sample_targets[name]

    targets = SymbolicFrameTargets(**target_values)
    return {
        "token_ids": token_ids,
        "token_type_ids": token_type_ids,
        "token_instrument_ids": token_instrument_ids,
        "position_ids": position_ids,
        "padding_mask": padding_mask,
        "anchor_positions": anchor_positions,
        "frame_padding_mask": frame_padding_mask,
        "targets": targets,
    }


__all__ = [
    "BASS_CLASSES",
    "BASS_NO_BASS_ID",
    "BASS_TOKEN_BASE",
    "BEAT_TOKEN_ID",
    "CHORD_CLASSES",
    "CHORD_NO_CHORD_ID",
    "CHORD_QUALITIES",
    "CHORD_TOKEN_BASE",
    "DOWNBEAT_TOKEN_ID",
    "DRUMS_INSTRUMENT_ID",
    "FRAME_TOKEN_ID",
    "INSTRUMENT_CLASSES",
    "INSTRUMENT_CLASS_NAMES",
    "INSTRUMENT_TOKEN_BASE",
    "KEY_CLASSES",
    "KEY_TOKEN_BASE",
    "MAX_MIDI_PITCH",
    "METER_CLASSES",
    "METER_TOKEN_BASE",
    "MIN_MIDI_PITCH",
    "NOTE_OFF_TOKEN_BASE",
    "NOTE_ON_TOKEN_BASE",
    "PAD_TOKEN_ID",
    "PITCH_CLASSES",
    "SYMBOLIC_REQUIRED_VOCAB_SIZE",
    "CachedSymbolicDataset",
    "MIDIEventTokenizer",
    "SymbolicFrameTargets",
    "SymbolicMIDIDataset",
    "SymbolicSequence",
    "SymbolicTokenType",
    "collate_symbolic_sequences",
    "crop_symbolic_sequence",
    "encode_chord_and_bass",
    "encode_chord_label",
    "encode_key_name",
    "encode_meter",
    "load_symbolic_cache",
    "random_crop_symbolic_sequence",
]
