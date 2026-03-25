"""Session types to music scores via LilyPond notation.

Translates session type state spaces into musical scores:
- Transition labels encode notes: C4q = C4 quarter note
- Parallel composition = polyphony (multiple staves)
- Paths through state space = valid performances
- Recursion = repeated sections

Label format: <note>[s|f]<octave><duration>[d]
  note: C D E F G A B (case-insensitive)
  s = sharp, f = flat (after note letter)
  octave: 0-8
  duration: w=whole h=half q=quarter e=eighth x=sixteenth
  d = dotted (append after duration)

Rest: R<duration>[d]  e.g. Rq = quarter rest

Usage:
    python3 -m reticulate.music "&{C4q: &{E4q: &{G4q: end}}}"
    python3 -m reticulate.music --output score.ly "(<voice1> || <voice2>)"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from reticulate.parser import parse
from reticulate.statespace import build_statespace, StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MusicalNote:
    """A single musical note or rest."""
    pitch: str          # "c".."b" or "r" for rest
    accidental: str     # "" | "is" (sharp) | "es" (flat)
    octave: int         # 0-8 (-1 for rests)
    duration: str       # "1"=whole "2"=half "4"=quarter "8"=eighth "16"=sixteenth
    dotted: bool

    @property
    def is_rest(self) -> bool:
        return self.pitch == "r"


@dataclass(frozen=True)
class MusicalPath:
    """A sequence of notes forming one voice."""
    notes: tuple[MusicalNote, ...]
    voice_name: str = "Voice"


@dataclass(frozen=True)
class MusicResult:
    """Result of session-type-to-music translation."""
    voices: tuple[MusicalPath, ...]
    lilypond_source: str
    is_polyphonic: bool
    warnings: tuple[str, ...]


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

_NOTE_NAMES = {"c", "d", "e", "f", "g", "a", "b"}

_DURATION_MAP = {
    "w": "1",   # whole
    "h": "2",   # half
    "q": "4",   # quarter
    "e": "8",   # eighth
    "x": "16",  # sixteenth
}

# Pattern: (note letter)(optional s/f)(octave digit)(duration char)(optional d)
_LABEL_RE = re.compile(
    r"^([A-Ga-g])(s|f)?(\d)([whqex])(d)?$"
)
_REST_RE = re.compile(
    r"^[Rr]([whqex])(d)?$"
)


def parse_musical_label(label: str) -> MusicalNote | None:
    """Parse a transition label as a musical note.

    Returns None if the label is not a valid musical label.

    Examples:
        >>> parse_musical_label("C4q")
        MusicalNote(pitch='c', accidental='', octave=4, duration='4', dotted=False)
        >>> parse_musical_label("Gs5ed")
        MusicalNote(pitch='g', accidental='is', octave=5, duration='8', dotted=True)
        >>> parse_musical_label("Bf3h")
        MusicalNote(pitch='b', accidental='es', octave=3, duration='2', dotted=False)
        >>> parse_musical_label("Rq")
        MusicalNote(pitch='r', accidental='', octave=-1, duration='4', dotted=False)
    """
    # Try rest
    m = _REST_RE.match(label)
    if m:
        dur = _DURATION_MAP[m.group(1)]
        dotted = m.group(2) is not None
        return MusicalNote("r", "", -1, dur, dotted)

    # Try note
    m = _LABEL_RE.match(label)
    if not m:
        return None

    pitch = m.group(1).lower()
    acc_raw = m.group(2) or ""
    octave = int(m.group(3))
    dur = _DURATION_MAP[m.group(4)]
    dotted = m.group(5) is not None

    # Convert accidental to LilyPond
    accidental = ""
    if acc_raw == "s":
        accidental = "is"
    elif acc_raw == "f":
        accidental = "es"

    return MusicalNote(pitch, accidental, octave, dur, dotted)


def is_musical_label(label: str) -> bool:
    """Check if a label is a valid musical notation."""
    return parse_musical_label(label) is not None


# ---------------------------------------------------------------------------
# LilyPond conversion
# ---------------------------------------------------------------------------

def note_to_lilypond(note: MusicalNote) -> str:
    """Convert a MusicalNote to a LilyPond token.

    Uses absolute octave mode:
      C3 = c    C4 = c'   C5 = c''   C2 = c,   C1 = c,,

    Examples:
        >>> note_to_lilypond(MusicalNote('c', '', 4, '4', False))
        "c'4"
        >>> note_to_lilypond(MusicalNote('g', 'is', 5, '8', True))
        "gis''8."
        >>> note_to_lilypond(MusicalNote('r', '', -1, '4', False))
        'r4'
    """
    if note.is_rest:
        s = f"r{note.duration}"
        if note.dotted:
            s += "."
        return s

    # Build: pitch + accidental + octave marks + duration
    s = note.pitch + note.accidental

    # Octave marks: C3 = c (no mark), C4 = c', C5 = c'', C2 = c,
    # LilyPond absolute: c = C3, c' = C4, c'' = C5, c, = C2
    if note.octave >= 4:
        s += "'" * (note.octave - 3)
    elif note.octave <= 2:
        s += "," * (3 - note.octave)
    # octave 3 = no mark

    s += note.duration
    if note.dotted:
        s += "."
    return s


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

def _extract_paths_from_statespace(
    ss: StateSpace,
    max_paths: int = 1,
    max_revisits: int = 2,
) -> list[list[str]]:
    """Extract label sequences (paths) from a state space.

    Each path is a list of transition labels from top to bottom.
    """
    from reticulate.testgen import enumerate_valid_paths

    valid, _truncated = enumerate_valid_paths(ss, max_revisits=max_revisits, max_paths=max_paths)
    paths: list[list[str]] = []
    for vp in valid:
        labels = [step.label for step in vp.steps]
        paths.append(labels)
    return paths


def extract_musical_paths(
    ss: StateSpace,
    voice_name: str = "Voice",
    max_paths: int = 1,
) -> tuple[list[MusicalPath], list[str]]:
    """Extract musical paths from a state space.

    Returns (paths, warnings).
    """
    raw_paths = _extract_paths_from_statespace(ss, max_paths=max_paths)
    results: list[MusicalPath] = []
    warnings: list[str] = []

    for i, labels in enumerate(raw_paths):
        notes: list[MusicalNote] = []
        for label in labels:
            note = parse_musical_label(label)
            if note is not None:
                notes.append(note)
            else:
                warnings.append(f"Non-musical label '{label}' in path {i}, skipped")
        name = voice_name if len(raw_paths) == 1 else f"{voice_name} {i+1}"
        results.append(MusicalPath(tuple(notes), name))

    return results, warnings


def extract_polyphonic_score(
    ss: StateSpace,
    voice_names: list[str] | None = None,
) -> tuple[list[MusicalPath], list[str]]:
    """Extract polyphonic voices from a parallel session type.

    If the state space has product_factors (from parallel composition),
    extract each factor as an independent voice.
    """
    warnings: list[str] = []

    if ss.product_factors and len(ss.product_factors) >= 2:
        voices: list[MusicalPath] = []
        default_names = ["Right Hand", "Left Hand", "Voice 3", "Voice 4",
                        "Voice 5", "Voice 6", "Voice 7", "Voice 8"]
        names = voice_names or default_names

        for i, factor in enumerate(ss.product_factors):
            name = names[i] if i < len(names) else f"Voice {i+1}"
            paths, w = extract_musical_paths(factor, voice_name=name, max_paths=1)
            warnings.extend(w)
            if paths:
                voices.append(paths[0])

        return voices, warnings

    # Not polyphonic — single voice
    paths, w = extract_musical_paths(ss, voice_name="Melody", max_paths=1)
    return paths, w


# ---------------------------------------------------------------------------
# Repeat detection
# ---------------------------------------------------------------------------

def _detect_repeats(notes: tuple[MusicalNote, ...]) -> list[dict[str, Any]]:
    """Detect repeated note patterns for volta repeats.

    Returns a list of segments: {"type": "normal"|"repeat", "notes": [...], "count": N}
    """
    if len(notes) < 4:
        return [{"type": "normal", "notes": list(notes), "count": 1}]

    # Try to find the longest repeating prefix
    n = len(notes)
    for seg_len in range(n // 2, 1, -1):
        seg = notes[:seg_len]
        count = 1
        pos = seg_len
        while pos + seg_len <= n and notes[pos:pos + seg_len] == seg:
            count += 1
            pos += seg_len
        if count >= 2:
            segments: list[dict[str, Any]] = []
            segments.append({"type": "repeat", "notes": list(seg), "count": count})
            remainder = notes[pos:]
            if remainder:
                segments.append({"type": "normal", "notes": list(remainder), "count": 1})
            return segments

    return [{"type": "normal", "notes": list(notes), "count": 1}]


# ---------------------------------------------------------------------------
# LilyPond generation
# ---------------------------------------------------------------------------

def _notes_to_lilypond(notes: list[MusicalNote] | tuple[MusicalNote, ...]) -> str:
    """Convert a sequence of notes to LilyPond tokens."""
    return " ".join(note_to_lilypond(n) for n in notes)


def _voice_to_lilypond(path: MusicalPath, use_repeats: bool = True) -> str:
    """Convert a voice to LilyPond notation with optional repeat detection."""
    if not path.notes:
        return "s1"  # spacer rest

    if use_repeats:
        segments = _detect_repeats(path.notes)
        parts: list[str] = []
        for seg in segments:
            if seg["type"] == "repeat" and seg["count"] >= 2:
                inner = _notes_to_lilypond(seg["notes"])
                parts.append(f"\\repeat volta {seg['count']} {{ {inner} }}")
            else:
                parts.append(_notes_to_lilypond(seg["notes"]))
        return " ".join(parts)

    return _notes_to_lilypond(path.notes)


def _guess_clef(path: MusicalPath) -> str:
    """Guess the appropriate clef based on average pitch."""
    pitched = [n for n in path.notes if not n.is_rest]
    if not pitched:
        return "treble"
    avg_octave = sum(n.octave for n in pitched) / len(pitched)
    return "bass" if avg_octave < 4 else "treble"


def generate_lilypond(
    voices: list[MusicalPath],
    title: str = "",
    composer: str = "",
    key: str = "c \\major",
    time_sig: str = "4/4",
) -> str:
    """Generate a complete LilyPond source file.

    Args:
        voices: One or more voice paths.
        title: Piece title.
        composer: Composer name.
        key: LilyPond key signature (e.g., "c \\major", "g \\minor").
        time_sig: Time signature (e.g., "4/4", "3/4").

    Returns:
        Complete LilyPond source string.
    """
    lines: list[str] = []
    lines.append('\\version "2.24.0"')
    lines.append("")

    if title or composer:
        lines.append("\\header {")
        if title:
            lines.append(f'  title = "{title}"')
        if composer:
            lines.append(f'  composer = "{composer}"')
        lines.append("}")
        lines.append("")

    if len(voices) == 0:
        lines.append("{ s1 }")
        return "\n".join(lines)

    if len(voices) == 1:
        # Single staff
        clef = _guess_clef(voices[0])
        note_str = _voice_to_lilypond(voices[0])
        lines.append("{")
        lines.append(f"  \\clef {clef}")
        lines.append(f"  \\key {key}")
        lines.append(f"  \\time {time_sig}")
        lines.append(f"  {note_str}")
        lines.append("}")
    else:
        # Multiple staves
        lines.append("\\new StaffGroup <<")
        for v in voices:
            clef = _guess_clef(v)
            note_str = _voice_to_lilypond(v)
            lines.append(f'  \\new Staff \\with {{ instrumentName = "{v.voice_name}" }} {{')
            lines.append(f"    \\clef {clef}")
            lines.append(f"    \\key {key}")
            lines.append(f"    \\time {time_sig}")
            lines.append(f"    {note_str}")
            lines.append("  }")
        lines.append(">>")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def session_to_music(
    type_str: str,
    title: str = "",
    composer: str = "",
    voice_names: list[str] | None = None,
    key: str = "c \\major",
    time_sig: str = "4/4",
) -> MusicResult:
    """Convert a session type string to a LilyPond music score.

    Args:
        type_str: Session type string with musical labels.
        title: Piece title.
        composer: Composer name.
        voice_names: Names for voices in parallel types.
        key: Key signature.
        time_sig: Time signature.

    Returns:
        MusicResult with voices and LilyPond source.
    """
    ast = parse(type_str)
    ss = build_statespace(ast)

    voices, warnings = extract_polyphonic_score(ss, voice_names=voice_names)
    is_poly = len(voices) > 1

    ly_source = generate_lilypond(
        voices, title=title, composer=composer, key=key, time_sig=time_sig,
    )

    return MusicResult(
        voices=tuple(voices),
        lilypond_source=ly_source,
        is_polyphonic=is_poly,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Pre-composed pieces as session types
# ---------------------------------------------------------------------------

# Bach Invention No. 1 in C Major (simplified opening, 2 voices)
BACH_INVENTION_1 = (
    "("
    "&{C5e: &{D5e: &{E5e: &{F5e: &{D5e: &{E5e: &{C5e: &{G5e: wait}}}}}}}}"
    " || "
    "&{C4e: &{D4e: &{E4e: &{F4e: &{D4e: &{E4e: &{C4e: &{G4e: wait}}}}}}}}"
    ")"
)

# Simple C major scale (single voice)
C_MAJOR_SCALE = "&{C4q: &{D4q: &{E4q: &{F4q: &{G4q: &{A4q: &{B4q: &{C5q: end}}}}}}}}"

# Chorale-style cadence (2 voices)
CADENCE_IV_V_I = (
    "("
    "&{F4h: &{G4h: &{C5w: wait}}}"
    " || "
    "&{A3h: &{B3h: &{C4w: wait}}}"
    ")"
)

# Canon — same melody at different octaves
CANON_SIMPLE = (
    "("
    "&{C5q: &{E5q: &{G5q: &{C6h: wait}}}}"
    " || "
    "&{C4q: &{E4q: &{G4q: &{C5h: wait}}}}"
    ")"
)

# Minuet pattern (3/4 time)
MINUET_PATTERN = (
    "("
    "&{G4q: &{A4q: &{B4q: &{C5q: &{D5q: &{E5q: wait}}}}}}"
    " || "
    "&{G3h: &{E3q: &{C3h: &{G3q: wait}}}}"
    ")"
)

# Bach-style invention with subject and answer (transposed to G)
BACH_INVENTION_FULL = (
    "("
    "&{C5e: &{D5e: &{E5e: &{F5e: &{D5e: &{E5e: &{C5e: &{G5e: "
    "&{E5e: &{F5e: &{G5e: &{A5e: &{F5e: &{G5e: &{E5e: &{C6q: wait}}}}}}}}}}}}}}}}"
    " || "
    "&{Rh: &{G4e: &{A4e: &{B4e: &{C5e: &{A4e: &{B4e: &{G4e: &{D5e: "
    "&{B4e: &{C5e: &{D5e: &{E5e: &{C5e: &{D5e: &{B4q: wait}}}}}}}}}}}}}}}}"
    ")"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for session-type music generation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="reticulate.music",
        description="Convert session types to LilyPond music scores",
    )
    parser.add_argument(
        "type_string", nargs="?",
        help="Session type string with musical labels",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--title", "-t", default="", help="Piece title")
    parser.add_argument("--composer", "-c", default="", help="Composer name")
    parser.add_argument("--voice-names", help="Comma-separated voice names")
    parser.add_argument("--key", default="c \\major", help="Key signature")
    parser.add_argument("--time", default="4/4", help="Time signature")
    parser.add_argument(
        "--example", choices=["scale", "cadence", "canon", "minuet",
                              "invention", "invention-full"],
        help="Use a built-in example instead of a type string",
    )

    args = parser.parse_args(argv)

    # Resolve type string
    examples = {
        "scale": C_MAJOR_SCALE,
        "cadence": CADENCE_IV_V_I,
        "canon": CANON_SIMPLE,
        "minuet": MINUET_PATTERN,
        "invention": BACH_INVENTION_1,
        "invention-full": BACH_INVENTION_FULL,
    }

    if args.example:
        type_str = examples[args.example]
    elif args.type_string:
        type_str = args.type_string
    else:
        parser.error("Provide a type string or --example")
        return

    voice_names = args.voice_names.split(",") if args.voice_names else None

    try:
        result = session_to_music(
            type_str,
            title=args.title or (args.example or "").replace("-", " ").title(),
            composer=args.composer,
            voice_names=voice_names,
            key=args.key,
            time_sig=args.time,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(result.lilypond_source)
        print(f"Wrote {args.output}", file=sys.stderr)
        if result.warnings:
            for w in result.warnings:
                print(f"Warning: {w}", file=sys.stderr)
        # Summary
        n_voices = len(result.voices)
        n_notes = sum(len(v.notes) for v in result.voices)
        print(f"{n_voices} voice(s), {n_notes} notes", file=sys.stderr)
    else:
        print(result.lilypond_source)

    if result.warnings and not args.output:
        import sys as _sys
        for w in result.warnings:
            print(f"% Warning: {w}", file=_sys.stderr)


if __name__ == "__main__":
    main()
