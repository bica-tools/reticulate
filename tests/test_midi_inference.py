"""Tests for midi_inference module (Step 56c)."""

from __future__ import annotations

import struct

import pytest

from reticulate.midi_inference import (
    MidiEvent,
    MidiFile,
    MidiInferenceResult,
    MidiTrack,
    Pattern,
    Voice,
    build_minimal_midi,
    build_multi_track_midi,
    detect_patterns,
    extract_voices,
    infer_session_type,
    midi_to_type_str,
    parse_midi,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------

class TestMidiEvent:
    def test_creation(self) -> None:
        evt = MidiEvent(0, "note_on", 0, {"note": 60, "velocity": 100})
        assert evt.delta_time == 0
        assert evt.event_type == "note_on"
        assert evt.channel == 0
        assert evt.data["note"] == 60

    def test_frozen(self) -> None:
        evt = MidiEvent(0, "note_on", 0, {"note": 60})
        with pytest.raises(AttributeError):
            evt.event_type = "other"  # type: ignore[misc]


class TestVoice:
    def test_creation(self) -> None:
        v = Voice(0, ("C4", "D4", "E4"), (480, 480, 480), 0)
        assert v.channel == 0
        assert len(v.notes) == 3


# ---------------------------------------------------------------------------
# MIDI building tests
# ---------------------------------------------------------------------------

class TestBuildMinimalMidi:
    def test_header(self) -> None:
        data = build_minimal_midi(((60, 480),))
        assert data[:4] == b'MThd'

    def test_track_chunk(self) -> None:
        data = build_minimal_midi(((60, 480),))
        assert b'MTrk' in data

    def test_parseable(self) -> None:
        data = build_minimal_midi(((60, 480), (64, 480)))
        midi = parse_midi(data)
        assert midi.format_type == 0
        assert midi.num_tracks == 1


class TestBuildMultiTrackMidi:
    def test_format1(self) -> None:
        data = build_multi_track_midi((
            ((60, 480),),
            ((64, 480),),
        ))
        midi = parse_midi(data)
        assert midi.format_type == 1
        assert midi.num_tracks == 2


# ---------------------------------------------------------------------------
# MIDI parsing tests
# ---------------------------------------------------------------------------

class TestParseMidi:
    def test_basic_parse(self) -> None:
        data = build_minimal_midi(((60, 480), (64, 480), (67, 480)))
        midi = parse_midi(data)
        assert isinstance(midi, MidiFile)
        assert len(midi.tracks) == 1

    def test_note_events(self) -> None:
        data = build_minimal_midi(((60, 480),))
        midi = parse_midi(data)
        note_ons = [e for e in midi.tracks[0].events if e.event_type == "note_on"]
        assert len(note_ons) == 1
        assert note_ons[0].data["note"] == 60

    def test_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            parse_midi(b'\x00' * 10)

    def test_bad_header(self) -> None:
        with pytest.raises(ValueError, match="Not a MIDI"):
            parse_midi(b'XXXX' + b'\x00' * 10)

    def test_multi_track(self) -> None:
        data = build_multi_track_midi((
            ((60, 480), (64, 480)),
            ((67, 480), (72, 480)),
        ))
        midi = parse_midi(data)
        assert len(midi.tracks) == 2


# ---------------------------------------------------------------------------
# Voice extraction tests
# ---------------------------------------------------------------------------

class TestExtractVoices:
    def test_single_channel(self) -> None:
        data = build_minimal_midi(((60, 480), (64, 480)))
        midi = parse_midi(data)
        voices = extract_voices(midi)
        assert len(voices) == 1
        assert voices[0].channel == 0

    def test_note_names(self) -> None:
        data = build_minimal_midi(((60, 480), (64, 480)))
        midi = parse_midi(data)
        voices = extract_voices(midi)
        assert "C4" in voices[0].notes
        assert "E4" in voices[0].notes

    def test_multi_channel(self) -> None:
        data = build_multi_track_midi((
            ((60, 480),),
            ((64, 480),),
        ))
        midi = parse_midi(data)
        voices = extract_voices(midi)
        assert len(voices) == 2


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------

class TestDetectPatterns:
    def test_simple_repeat(self) -> None:
        notes = ("C4", "D4", "C4", "D4")
        patterns = detect_patterns(notes)
        assert len(patterns) > 0
        assert any(p.notes == ("C4", "D4") for p in patterns)

    def test_no_repeat(self) -> None:
        notes = ("C4", "D4", "E4", "F4")
        patterns = detect_patterns(notes)
        assert len(patterns) == 0

    def test_min_length(self) -> None:
        notes = ("C4", "C4", "D4", "D4")
        patterns = detect_patterns(notes, min_length=3)
        # Only length-3+ patterns, which don't repeat in length-4 input
        assert all(len(p.notes) >= 3 for p in patterns)

    def test_occurrences(self) -> None:
        notes = ("C4", "D4", "E4", "C4", "D4", "E4")
        patterns = detect_patterns(notes)
        cdep = [p for p in patterns if p.notes == ("C4", "D4", "E4")]
        assert len(cdep) == 1
        assert cdep[0].occurrences == 2

    def test_sorted_by_significance(self) -> None:
        notes = ("C4", "D4", "C4", "D4", "C4", "D4")
        patterns = detect_patterns(notes)
        # Should be sorted by occurrences * length
        if len(patterns) >= 2:
            scores = [p.occurrences * len(p.notes) for p in patterns]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Session type inference tests
# ---------------------------------------------------------------------------

class TestInferSessionType:
    def test_basic(self) -> None:
        data = build_minimal_midi(((60, 480), (64, 480), (67, 480)))
        midi = parse_midi(data)
        result = infer_session_type(midi)
        assert isinstance(result, MidiInferenceResult)
        assert result.num_voices == 1

    def test_type_str_parseable(self) -> None:
        data = build_minimal_midi(((60, 480), (64, 480)))
        midi = parse_midi(data)
        result = infer_session_type(midi)
        st = parse(result.session_type_str)
        ss = build_statespace(st)
        assert len(ss.states) > 1

    def test_is_lattice(self) -> None:
        data = build_minimal_midi(((60, 480), (64, 480)))
        midi = parse_midi(data)
        result = infer_session_type(midi)
        assert result.is_lattice

    def test_multi_voice(self) -> None:
        data = build_multi_track_midi((
            ((60, 480), (64, 480)),
            ((67, 480), (72, 480)),
        ))
        midi = parse_midi(data)
        result = infer_session_type(midi)
        assert result.num_voices == 2
        assert "||" in result.session_type_str

    def test_max_depth(self) -> None:
        notes = tuple((60 + i % 12, 480) for i in range(20))
        data = build_minimal_midi(notes)
        midi = parse_midi(data)
        result = infer_session_type(midi, max_depth=4)
        # Type should be truncated to 4 notes
        assert result.state_count <= 10  # 4 notes + end


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestMidiToTypeStr:
    def test_basic(self) -> None:
        data = build_minimal_midi(((60, 480),))
        result = midi_to_type_str(data)
        assert "C4" in result


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_track(self) -> None:
        # Build a MIDI with a track containing only end-of-track
        header = bytearray()
        header.extend(b'MThd')
        header.extend(struct.pack(">I", 6))
        header.extend(struct.pack(">HHH", 0, 1, 480))
        track = bytearray()
        track.extend(b'MTrk')
        track_data = b'\x00\xFF\x2F\x00'
        track.extend(struct.pack(">I", len(track_data)))
        track.extend(track_data)
        data = bytes(header + track)
        midi = parse_midi(data)
        voices = extract_voices(midi)
        assert len(voices) == 0
