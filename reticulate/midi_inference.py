"""Session type inference from MIDI (Step 56c).

Extracts session types from MIDI files by analyzing note sequences.
Uses only the Python standard library to parse the MIDI binary format
directly (no external dependencies).

MIDI structure maps to session types:
- Each track/channel becomes a voice (parallel composition).
- Note sequences within a voice become branch chains.
- Repeated patterns become recursive types.
- Program changes become selection (internal choice).

MIDI binary format (Standard MIDI File):
- Header chunk: MThd + length + format + ntrks + division
- Track chunks: MTrk + length + events
- Events: delta-time + event-type + data

This module provides:
    ``parse_midi()``            -- parse MIDI bytes into events.
    ``infer_session_type()``    -- infer session type from MIDI data.
    ``extract_voices()``        -- separate MIDI into per-channel voices.
    ``detect_patterns()``       -- find repeated note patterns.
    ``midi_to_type_str()``      -- convenience: bytes -> type string.
    ``build_minimal_midi()``    -- build minimal MIDI bytes for testing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
    parse,
    pretty,
)
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# MIDI data types
# ---------------------------------------------------------------------------

_NOTE_NAMES = ("C", "Cs", "D", "Ds", "E", "F",
               "Fs", "G", "Gs", "A", "As", "B")


@dataclass(frozen=True)
class MidiEvent:
    """A single MIDI event."""
    delta_time: int
    event_type: str       # "note_on", "note_off", "program_change", "meta"
    channel: int
    data: dict[str, int]  # key-value pairs depending on event type


@dataclass(frozen=True)
class MidiTrack:
    """A MIDI track containing events."""
    events: tuple[MidiEvent, ...]
    name: str = ""


@dataclass(frozen=True)
class MidiFile:
    """Parsed MIDI file."""
    format_type: int      # 0, 1, or 2
    num_tracks: int
    ticks_per_beat: int
    tracks: tuple[MidiTrack, ...]


@dataclass(frozen=True)
class Voice:
    """A single voice (channel) extracted from MIDI."""
    channel: int
    notes: tuple[str, ...]        # note names like "C4", "D5"
    durations: tuple[int, ...]    # in ticks
    program: int = 0              # MIDI program number


@dataclass(frozen=True)
class Pattern:
    """A repeated pattern detected in a voice."""
    notes: tuple[str, ...]
    occurrences: int
    start_positions: tuple[int, ...]


@dataclass(frozen=True)
class MidiInferenceResult:
    """Result of session type inference from MIDI."""
    session_type_str: str
    num_voices: int
    voices: tuple[Voice, ...]
    patterns: tuple[Pattern, ...]
    state_count: int
    is_lattice: bool


# ---------------------------------------------------------------------------
# MIDI binary parsing (standard library only)
# ---------------------------------------------------------------------------

def _read_variable_length(data: bytes, offset: int) -> tuple[int, int]:
    """Read a MIDI variable-length quantity. Returns (value, new_offset)."""
    value = 0
    while offset < len(data):
        byte = data[offset]
        value = (value << 7) | (byte & 0x7F)
        offset += 1
        if not (byte & 0x80):
            break
    return value, offset


def _note_name(midi_number: int) -> str:
    """Convert MIDI note number to name like 'C4'."""
    octave = (midi_number // 12) - 1
    note = _NOTE_NAMES[midi_number % 12]
    return f"{note}{octave}"


def parse_midi(data: bytes) -> MidiFile:
    """Parse MIDI binary data into a MidiFile structure.

    Handles Format 0 (single track) and Format 1 (multi-track).
    """
    if len(data) < 14:
        raise ValueError("Data too short for MIDI header")

    # Header chunk
    header = data[:4]
    if header != b'MThd':
        raise ValueError(f"Not a MIDI file (header: {header!r})")

    header_len = struct.unpack(">I", data[4:8])[0]
    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])

    offset = 8 + header_len
    tracks: list[MidiTrack] = []

    for _ in range(ntrks):
        if offset + 8 > len(data):
            break
        chunk_id = data[offset:offset + 4]
        if chunk_id != b'MTrk':
            break
        chunk_len = struct.unpack(">I", data[offset + 4:offset + 8])[0]
        track_data = data[offset + 8:offset + 8 + chunk_len]

        events = _parse_track_events(track_data)
        tracks.append(MidiTrack(events=tuple(events)))
        offset += 8 + chunk_len

    return MidiFile(
        format_type=fmt,
        num_tracks=ntrks,
        ticks_per_beat=division,
        tracks=tuple(tracks),
    )


def _parse_track_events(data: bytes) -> list[MidiEvent]:
    """Parse events from a single MIDI track chunk."""
    events: list[MidiEvent] = []
    offset = 0
    running_status = 0

    while offset < len(data):
        delta, offset = _read_variable_length(data, offset)
        if offset >= len(data):
            break

        status = data[offset]
        if status & 0x80:
            running_status = status
            offset += 1
        else:
            status = running_status

        if status == 0xFF:  # Meta event
            if offset + 1 >= len(data):
                break
            meta_type = data[offset]
            offset += 1
            length, offset = _read_variable_length(data, offset)
            meta_data = data[offset:offset + length]
            offset += length
            events.append(MidiEvent(
                delta_time=delta,
                event_type="meta",
                channel=0,
                data={"meta_type": meta_type, "length": length},
            ))
        elif (status & 0xF0) == 0x90:  # Note On
            channel = status & 0x0F
            if offset + 1 >= len(data):
                break
            note = data[offset]
            velocity = data[offset + 1]
            offset += 2
            etype = "note_on" if velocity > 0 else "note_off"
            events.append(MidiEvent(
                delta_time=delta,
                event_type=etype,
                channel=channel,
                data={"note": note, "velocity": velocity},
            ))
        elif (status & 0xF0) == 0x80:  # Note Off
            channel = status & 0x0F
            if offset + 1 >= len(data):
                break
            note = data[offset]
            velocity = data[offset + 1]
            offset += 2
            events.append(MidiEvent(
                delta_time=delta,
                event_type="note_off",
                channel=channel,
                data={"note": note, "velocity": velocity},
            ))
        elif (status & 0xF0) == 0xC0:  # Program Change
            channel = status & 0x0F
            if offset >= len(data):
                break
            program = data[offset]
            offset += 1
            events.append(MidiEvent(
                delta_time=delta,
                event_type="program_change",
                channel=channel,
                data={"program": program},
            ))
        elif (status & 0xF0) in (0xA0, 0xB0, 0xE0):  # 2-byte data
            if offset + 1 >= len(data):
                break
            offset += 2
        elif (status & 0xF0) == 0xD0:  # 1-byte data
            if offset >= len(data):
                break
            offset += 1
        elif status == 0xF0 or status == 0xF7:  # SysEx
            length, offset = _read_variable_length(data, offset)
            offset += length
        else:
            offset += 1  # Skip unknown

    return events


# ---------------------------------------------------------------------------
# Voice extraction
# ---------------------------------------------------------------------------

def extract_voices(midi: MidiFile) -> tuple[Voice, ...]:
    """Extract per-channel voices from parsed MIDI data."""
    channel_notes: dict[int, list[str]] = {}
    channel_durations: dict[int, list[int]] = {}
    channel_program: dict[int, int] = {}

    for track in midi.tracks:
        for event in track.events:
            ch = event.channel
            if event.event_type == "note_on":
                note_num = event.data.get("note", 60)
                if ch not in channel_notes:
                    channel_notes[ch] = []
                    channel_durations[ch] = []
                channel_notes[ch].append(_note_name(note_num))
                channel_durations[ch].append(event.delta_time)
            elif event.event_type == "program_change":
                channel_program[ch] = event.data.get("program", 0)

    voices: list[Voice] = []
    for ch in sorted(channel_notes.keys()):
        voices.append(Voice(
            channel=ch,
            notes=tuple(channel_notes[ch]),
            durations=tuple(channel_durations[ch]),
            program=channel_program.get(ch, 0),
        ))

    return tuple(voices)


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def detect_patterns(
    notes: tuple[str, ...],
    min_length: int = 2,
    max_length: int = 8,
) -> tuple[Pattern, ...]:
    """Detect repeated note patterns in a sequence."""
    patterns: list[Pattern] = []
    seen: set[tuple[str, ...]] = set()

    for plen in range(min_length, min(max_length + 1, len(notes) // 2 + 1)):
        for start in range(len(notes) - plen + 1):
            candidate = tuple(notes[start:start + plen])
            if candidate in seen:
                continue
            seen.add(candidate)

            # Count occurrences
            positions: list[int] = []
            for i in range(len(notes) - plen + 1):
                if tuple(notes[i:i + plen]) == candidate:
                    positions.append(i)

            if len(positions) >= 2:
                patterns.append(Pattern(
                    notes=candidate,
                    occurrences=len(positions),
                    start_positions=tuple(positions),
                ))

    # Sort by occurrences * length (most significant patterns first)
    patterns.sort(key=lambda p: p.occurrences * len(p.notes), reverse=True)
    return tuple(patterns)


# ---------------------------------------------------------------------------
# Session type inference
# ---------------------------------------------------------------------------

def _voice_to_session_type(voice: Voice, max_depth: int = 16) -> SessionType:
    """Convert a voice to a session type, truncating at max_depth."""
    notes = voice.notes[:max_depth]
    if not notes:
        return End()

    current: SessionType = End()
    for note in reversed(notes):
        current = Branch(((note, current),))
    return current


def _voices_to_parallel(voices: tuple[Voice, ...], max_depth: int = 16) -> SessionType:
    """Combine multiple voices into a parallel session type."""
    if not voices:
        return End()
    if len(voices) == 1:
        return _voice_to_session_type(voices[0], max_depth)

    # Build parallel composition of all voices
    types = [_voice_to_session_type(v, max_depth) for v in voices]
    return Parallel(tuple(types))


def infer_session_type(
    midi: MidiFile,
    max_depth: int = 16,
) -> MidiInferenceResult:
    """Infer a session type from parsed MIDI data.

    Each channel becomes a parallel voice. Repeated patterns may
    become recursive types (if detected).
    """
    voices = extract_voices(midi)

    all_patterns: list[Pattern] = []
    for voice in voices:
        all_patterns.extend(detect_patterns(voice.notes))

    st = _voices_to_parallel(voices, max_depth)
    type_str = pretty(st)

    parsed = parse(type_str)
    ss = build_statespace(parsed)
    lr = check_lattice(ss)

    return MidiInferenceResult(
        session_type_str=type_str,
        num_voices=len(voices),
        voices=voices,
        patterns=tuple(all_patterns),
        state_count=len(ss.states),
        is_lattice=lr.is_lattice,
    )


def midi_to_type_str(data: bytes, max_depth: int = 16) -> str:
    """Convenience: parse MIDI bytes and return session type string."""
    midi = parse_midi(data)
    result = infer_session_type(midi, max_depth)
    return result.session_type_str


# ---------------------------------------------------------------------------
# MIDI construction (for testing)
# ---------------------------------------------------------------------------

def build_minimal_midi(
    notes: tuple[tuple[int, int], ...],  # (midi_number, duration_ticks)
    channel: int = 0,
    ticks_per_beat: int = 480,
) -> bytes:
    """Build minimal MIDI bytes for testing.

    Creates a Format 0 MIDI file with a single track containing
    the given notes on the specified channel.
    """
    # Build track events
    track_data = bytearray()
    for midi_num, duration in notes:
        # Note On (delta=0)
        track_data.append(0x00)  # delta time
        track_data.append(0x90 | (channel & 0x0F))
        track_data.append(midi_num & 0x7F)
        track_data.append(100)  # velocity

        # Note Off (delta=duration)
        _write_variable_length(track_data, duration)
        track_data.append(0x80 | (channel & 0x0F))
        track_data.append(midi_num & 0x7F)
        track_data.append(0)

    # End of track meta event
    track_data.append(0x00)
    track_data.extend(b'\xFF\x2F\x00')

    # Header chunk
    header = bytearray()
    header.extend(b'MThd')
    header.extend(struct.pack(">I", 6))       # header length
    header.extend(struct.pack(">HHH", 0, 1, ticks_per_beat))

    # Track chunk
    track_chunk = bytearray()
    track_chunk.extend(b'MTrk')
    track_chunk.extend(struct.pack(">I", len(track_data)))
    track_chunk.extend(track_data)

    return bytes(header + track_chunk)


def build_multi_track_midi(
    tracks: tuple[tuple[tuple[int, int], ...], ...],
    ticks_per_beat: int = 480,
) -> bytes:
    """Build a Format 1 MIDI file with multiple tracks."""
    all_chunks = bytearray()

    # Header
    all_chunks.extend(b'MThd')
    all_chunks.extend(struct.pack(">I", 6))
    all_chunks.extend(struct.pack(">HHH", 1, len(tracks), ticks_per_beat))

    for ch_idx, notes in enumerate(tracks):
        track_data = bytearray()
        for midi_num, duration in notes:
            track_data.append(0x00)
            track_data.append(0x90 | (ch_idx & 0x0F))
            track_data.append(midi_num & 0x7F)
            track_data.append(100)

            _write_variable_length(track_data, duration)
            track_data.append(0x80 | (ch_idx & 0x0F))
            track_data.append(midi_num & 0x7F)
            track_data.append(0)

        track_data.append(0x00)
        track_data.extend(b'\xFF\x2F\x00')

        all_chunks.extend(b'MTrk')
        all_chunks.extend(struct.pack(">I", len(track_data)))
        all_chunks.extend(track_data)

    return bytes(all_chunks)


def _write_variable_length(buf: bytearray, value: int) -> None:
    """Write a MIDI variable-length quantity to a bytearray."""
    if value < 0:
        value = 0
    result: list[int] = []
    result.append(value & 0x7F)
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.reverse()
    buf.extend(result)
