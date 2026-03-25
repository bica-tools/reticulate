"""Algorithmic composition using session type structure.

Every composition is grounded in a session type: the type IS the score.
The pipeline is:

    compose → session type string → parse → build_statespace → extract paths → LilyPond

Compositional structure maps to session type constructors:
    - Parallel voices      →  (S1 || S2)
    - Sequential sections  →  (S1 || S2) . (S3 || S4) . end
    - Repeated material    →  rec X . (S || S') . X
    - Harmonic choices     →  &{option_a: S1, option_b: S2}

Each compose function returns a CompositionResult with:
    - session_type: the full session type string
    - music_result: the LilyPond score (obtained by running the type through the pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from reticulate.music import (
    MusicalNote, MusicalPath, MusicResult,
    parse_musical_label, note_to_lilypond, generate_lilypond,
    session_to_music,
)


# ---------------------------------------------------------------------------
# Composition result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionResult:
    """A composition with its session type and rendered score."""
    session_type: str
    music_result: MusicResult
    sections: tuple[str, ...] = ()  # section names for documentation
    state_count: int = 0            # states in the session type's state space
    transition_count: int = 0       # transitions


def _compose_from_sections(
    sections: list[list[list[MusicalNote]]],
    section_names: tuple[str, ...],
    voice_names: list[str],
    title: str,
    composer: str,
    key: str = "c \\major",
    time_sig: str = "4/4",
) -> CompositionResult:
    """Build a composition from structured sections.

    Each section is a list of voices (list of note lists).
    The session type is built from the sections, and the LilyPond is
    generated from the assembled note sequences (preserving all notes
    across all sections).

    This dual approach ensures:
    1. The session type formally specifies the composition
    2. The LilyPond renders the full piece (not limited by path extraction)
    """
    from reticulate.parser import parse as parse_type
    from reticulate.statespace import build_statespace

    # Build session type
    type_str = _chain_sections(sections)

    # Verify it parses and forms a valid state space
    ast = parse_type(type_str)
    ss = build_statespace(ast)

    # Assemble full note sequences per voice
    n_voices = max(len(sec) for sec in sections) if sections else 0
    voice_notes: list[list[MusicalNote]] = [[] for _ in range(n_voices)]
    for sec in sections:
        for i, voice in enumerate(sec):
            if i < n_voices:
                voice_notes[i].extend(voice)

    # Build MusicalPaths and generate LilyPond
    paths = []
    for i in range(n_voices):
        name = voice_names[i] if i < len(voice_names) else f"Voice {i+1}"
        paths.append(MusicalPath(tuple(voice_notes[i]), name))

    ly = generate_lilypond(paths, title=title, composer=composer,
                           key=key, time_sig=time_sig)

    music = MusicResult(
        voices=tuple(paths),
        lilypond_source=ly,
        is_polyphonic=n_voices > 1,
        warnings=(),
    )

    return CompositionResult(
        session_type=type_str,
        music_result=music,
        sections=section_names,
        state_count=len(ss.states),
        transition_count=len(ss.transitions),
    )


# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------

# Chromatic scale for transposition
_CHROMATIC = ["c", "cs", "d", "ds", "e", "f", "fs", "g", "gs", "a", "as", "b"]
_PITCH_TO_SEMITONE = {
    ("c", ""): 0, ("c", "is"): 1, ("d", "es"): 1,
    ("d", ""): 2, ("d", "is"): 3, ("e", "es"): 3,
    ("e", ""): 4, ("f", ""): 5, ("f", "is"): 6,
    ("g", "es"): 6, ("g", ""): 7, ("g", "is"): 8,
    ("a", "es"): 8, ("a", ""): 9, ("a", "is"): 10,
    ("b", "es"): 10, ("b", ""): 11,
}
_SEMITONE_TO_PITCH = {
    0: ("c", ""), 1: ("c", "is"), 2: ("d", ""), 3: ("e", "es"),
    4: ("e", ""), 5: ("f", ""), 6: ("f", "is"), 7: ("g", ""),
    8: ("g", "is"), 9: ("a", ""), 10: ("b", "es"), 11: ("b", ""),
}


def _transpose_note(note: MusicalNote, semitones: int) -> MusicalNote:
    """Transpose a note by a given number of semitones."""
    if note.is_rest:
        return note
    semi = _PITCH_TO_SEMITONE.get((note.pitch, note.accidental), 0)
    new_semi = (semi + semitones) % 12
    octave_shift = (semi + semitones) // 12
    if semi + semitones < 0 and new_semi != 0:
        octave_shift -= 1
    p, a = _SEMITONE_TO_PITCH[new_semi]
    return MusicalNote(p, a, note.octave + octave_shift, note.duration, note.dotted)


def _transpose_phrase(notes: list[MusicalNote], semitones: int) -> list[MusicalNote]:
    """Transpose an entire phrase."""
    return [_transpose_note(n, semitones) for n in notes]


def _invert_phrase(notes: list[MusicalNote], axis_note: MusicalNote) -> list[MusicalNote]:
    """Invert a phrase around an axis note (mirror intervals)."""
    if not notes:
        return []
    axis = _PITCH_TO_SEMITONE.get((axis_note.pitch, axis_note.accidental), 0) + axis_note.octave * 12
    result = []
    for n in notes:
        if n.is_rest:
            result.append(n)
            continue
        semi = _PITCH_TO_SEMITONE.get((n.pitch, n.accidental), 0) + n.octave * 12
        mirror = 2 * axis - semi
        new_oct = mirror // 12
        new_semi = mirror % 12
        p, a = _SEMITONE_TO_PITCH[new_semi]
        result.append(MusicalNote(p, a, new_oct, n.duration, n.dotted))
    return result


def _retrograde(notes: list[MusicalNote]) -> list[MusicalNote]:
    """Reverse a phrase (retrograde)."""
    return list(reversed(notes))


def _augment(notes: list[MusicalNote]) -> list[MusicalNote]:
    """Double all durations (augmentation)."""
    dur_map = {"16": "8", "8": "4", "4": "2", "2": "1", "1": "1"}
    return [MusicalNote(n.pitch, n.accidental, n.octave,
                        dur_map.get(n.duration, n.duration), n.dotted)
            for n in notes]


def _diminish(notes: list[MusicalNote]) -> list[MusicalNote]:
    """Halve all durations (diminution)."""
    dur_map = {"1": "2", "2": "4", "4": "8", "8": "16", "16": "16"}
    return [MusicalNote(n.pitch, n.accidental, n.octave,
                        dur_map.get(n.duration, n.duration), n.dotted)
            for n in notes]


def _make_note(pitch: str, acc: str, octave: int, dur: str, dot: bool = False) -> MusicalNote:
    return MusicalNote(pitch, acc, octave, dur, dot)


# ---------------------------------------------------------------------------
# Session type builders
# ---------------------------------------------------------------------------

# Reverse maps for converting notes back to labels
_ACC_TO_LABEL = {"": "", "is": "s", "es": "f"}
_DUR_TO_LABEL = {"1": "w", "2": "h", "4": "q", "8": "e", "16": "x"}


def _note_to_label(note: MusicalNote) -> str:
    """Convert a MusicalNote back to a label string like 'C4q' or 'Rh'."""
    if note.is_rest:
        dur = _DUR_TO_LABEL.get(note.duration, "q")
        return f"R{dur}{'d' if note.dotted else ''}"
    pitch = note.pitch.upper()
    acc = _ACC_TO_LABEL.get(note.accidental, "")
    dur = _DUR_TO_LABEL.get(note.duration, "q")
    dot = "d" if note.dotted else ""
    return f"{pitch}{acc}{note.octave}{dur}{dot}"


def _phrase_to_session_type(notes: list[MusicalNote]) -> str:
    """Convert a list of notes to a nested branch session type.

    [C4q, D4q, E4q] → &{C4q: &{D4q: &{E4q: wait}}}
    """
    if not notes:
        return "wait"
    labels = [_note_to_label(n) for n in notes]
    # Build inside-out: start from wait, wrap each label
    result = "wait"
    for label in reversed(labels):
        result = f"&{{{label}: {result}}}"
    return result


def _parallel_section(voices: list[list[MusicalNote]]) -> str:
    """Build a parallel session type from multiple voices.

    Two voices: (S1 || S2)
    Three voices: (S1 || (S2 || S3))
    """
    if len(voices) == 0:
        return "end"
    if len(voices) == 1:
        return _phrase_to_session_type(voices[0])
    types = [_phrase_to_session_type(v) for v in voices]
    # Right-associate: (t1 || (t2 || t3))
    result = types[-1]
    for t in reversed(types[:-1]):
        result = f"({t} || {result})"
    return result


def _chain_sections(sections: list[list[list[MusicalNote]]]) -> str:
    """Chain multiple sections with continuation.

    Each section is a list of voices (list of note lists).
    Result: (S1 || S2) . (S3 || S4) . end
    """
    if not sections:
        return "end"
    parts = [_parallel_section(sec) for sec in sections]
    # Chain with continuation: p1 . p2 . ... . end
    result = "end"
    for part in reversed(parts):
        result = f"{part} . {result}"
    return result


def n(spec: str) -> MusicalNote:
    """Shorthand: n("C4q") -> MusicalNote."""
    note = parse_musical_label(spec)
    if note is None:
        raise ValueError(f"Invalid note: {spec}")
    return note


def rest(dur: str = "4", dot: bool = False) -> MusicalNote:
    """Create a rest."""
    return MusicalNote("r", "", -1, dur, dot)


# ---------------------------------------------------------------------------
# Two-Part Invention Composer
# ---------------------------------------------------------------------------

def compose_invention(
    key: str = "c",
    title: str = "Two-Part Invention",
    composer: str = "Composed via Session Types",
) -> MusicResult:
    """Compose a complete two-part invention in Bach style.

    Structure:
    - Exposition: Subject in RH, Answer in LH (measures 1-4)
    - Episode 1: Sequential development (measures 5-6)
    - Middle entry: Subject in LH, countersubject in RH (measures 7-8)
    - Episode 2: Development with inversion (measures 9-10)
    - Recapitulation: Subject returns in RH (measures 11-12)
    - Coda: Cadential formula (measures 13-14)
    """
    # === SUBJECT (C major, 8 eighth notes = 1 measure of 4/4) ===
    subject = [
        n("C5e"), n("D5e"), n("E5e"), n("F5e"),
        n("D5e"), n("E5e"), n("C5e"), n("G5e"),
    ]

    # === COUNTERSUBJECT (complementary motion) ===
    countersubject = [
        n("E4e"), n("D4e"), n("C4e"), n("B3e"),
        n("D4e"), n("C4e"), n("E4e"), n("G3e"),
    ]

    # === ANSWER (subject transposed to dominant G, interval of 5th = 7 semitones) ===
    answer = _transpose_phrase(subject, 7)  # now in G
    # Bring answer down an octave to fit LH range
    answer = _transpose_phrase(answer, -12)

    # === COUNTER-ANSWER (countersubject transposed) ===
    counter_answer = _transpose_phrase(countersubject, 7)

    # === EPISODE 1: Sequential pattern descending by step ===
    seq_motif_rh = [n("G5e"), n("F5e"), n("E5e"), n("D5e")]
    seq_motif_lh = [n("E4e"), n("F4e"), n("G4e"), n("A4e")]

    episode1_rh = (
        seq_motif_rh +
        _transpose_phrase(seq_motif_rh, -2) +  # down a step
        _transpose_phrase(seq_motif_rh, -4) +   # down another step
        _transpose_phrase(seq_motif_rh, -5)     # down to F
    )
    episode1_lh = (
        seq_motif_lh +
        _transpose_phrase(seq_motif_lh, -2) +
        _transpose_phrase(seq_motif_lh, -4) +
        _transpose_phrase(seq_motif_lh, -5)
    )

    # === MIDDLE ENTRY: Subject in LH (key of G), countersubject in RH ===
    middle_subject_lh = _transpose_phrase(subject, -5)  # Subject in LH, down to G3 range
    middle_cs_rh = _transpose_phrase(countersubject, 12)  # Countersubject up an octave

    # === EPISODE 2: Inverted motifs + development ===
    inv_motif = _invert_phrase(subject[:4], n("E5e"))  # Invert first 4 notes around E5
    episode2_rh = (
        inv_motif +
        _transpose_phrase(inv_motif, -2) +
        [n("D5e"), n("E5e"), n("F5e"), n("G5e"),
         n("A5e"), n("G5e"), n("F5e"), n("E5e")]
    )
    episode2_lh = (
        [n("C4q"), n("B3q"), n("A3q"), n("G3q")] +
        [n("F3e"), n("G3e"), n("A3e"), n("B3e"),
         n("C4e"), n("D4e"), n("E4e"), n("F4e")]
    )

    # === RECAPITULATION: Subject returns in RH, free bass ===
    recap_rh = subject + [
        n("A5e"), n("G5e"), n("F5e"), n("E5e"),
        n("D5e"), n("C5e"), n("B4e"), n("A4e"),
    ]
    recap_lh = [
        n("C4q"), n("E4q"), n("G4q"), n("C5q"),
        n("F4e"), n("E4e"), n("D4e"), n("C4e"),
        n("B3e"), n("A3e"), n("G3e"), n("F3e"),
    ]

    # === CODA: Cadential formula IV-V-I ===
    coda_rh = [
        n("G5q"), n("F5q"), n("E5q"), n("D5q"),    # descending scale
        n("C5e"), n("D5e"), n("E5e"), n("F5e"),     # ascending run
        n("G5q"), n("A5q"),                          # approach
        n("G5h"),                                     # dominant
        n("C5w"),                                     # tonic (final)
    ]
    coda_lh = [
        n("E3q"), n("F3q"), n("G3q"), n("A3q"),     # ascending bass
        n("F3q"), n("G3q"),                          # IV
        n("D3q"), n("E3q"),                          # passing
        n("G3h"),                                     # dominant
        n("C3w"),                                     # tonic (final)
    ]

    # === BUILD FROM SECTIONS ===
    sections = [
        [subject, countersubject],          # Exposition: subject + countersubject
        [counter_answer, answer],           # Exposition: counter-answer + answer
        [episode1_rh, episode1_lh],         # Episode 1
        [middle_cs_rh, middle_subject_lh],  # Middle entry
        [episode2_rh, episode2_lh],         # Episode 2
        [recap_rh, recap_lh],               # Recapitulation
        [coda_rh, coda_lh],                 # Coda
    ]

    return _compose_from_sections(
        sections,
        section_names=(
            "Exposition (subject)", "Exposition (answer)",
            "Episode 1", "Middle entry", "Episode 2",
            "Recapitulation", "Coda",
        ),
        voice_names=["Right Hand", "Left Hand"],
        title=title, composer=composer,
        key="c \\major",
    )


# ---------------------------------------------------------------------------
# Chorale Composer
# ---------------------------------------------------------------------------

def compose_chorale(
    title: str = "Chorale in C Major",
    composer: str = "Composed via Session Types",
) -> MusicResult:
    """Compose a 4-part chorale (SATB) with standard harmonic progression.

    Progression: I - IV - V - I - vi - IV - V7 - I
    """
    # Soprano
    soprano = [
        n("E5q"), n("F5q"), n("G5q"), n("E5q"),   # I - IV - V - I
        n("C5q"), n("F5q"), n("G5q"), n("C5q"),   # vi - IV - V7 - I
        n("E5q"), n("D5q"), n("C5q"), n("B4q"),   # passing motion
        n("C5h"), n("G4h"),                         # cadence
        n("C5w"),                                    # final
    ]

    # Alto
    alto = [
        n("C5q"), n("C5q"), n("D5q"), n("C5q"),
        n("A4q"), n("A4q"), n("B4q"), n("G4q"),
        n("G4q"), n("F4q"), n("E4q"), n("D4q"),
        n("E4h"), n("D4h"),
        n("E4w"),
    ]

    # Tenor
    tenor = [
        n("G4q"), n("A4q"), n("B4q"), n("G4q"),
        n("E4q"), n("F4q"), n("G4q"), n("E4q"),
        n("C4q"), n("B3q"), n("A3q"), n("G3q"),
        n("G3h"), n("G3h"),
        n("G3w"),
    ]

    # Bass
    bass = [
        n("C3q"), n("F3q"), n("G3q"), n("C3q"),
        n("A2q"), n("F2q"), n("G2q"), n("C3q"),
        n("C3q"), n("G2q"), n("A2q"), n("B2q"),
        n("C3h"), n("G2h"),
        n("C3w"),
    ]

    return _compose_from_sections(
        [[soprano, alto, tenor, bass]],
        section_names=("Chorale",),
        voice_names=["Soprano", "Alto", "Tenor", "Bass"],
        title=title, composer=composer,
        key="c \\major",
    )


# ---------------------------------------------------------------------------
# Fugue Exposition Composer
# ---------------------------------------------------------------------------

def compose_fugue_exposition(
    title: str = "Fugue Exposition in C Minor",
    composer: str = "Composed via Session Types",
) -> MusicResult:
    """Compose a 3-voice fugue exposition.

    Structure:
    - Voice 1 (Soprano): Subject → Countersubject → Free
    - Voice 2 (Alto): Rest → Answer → Countersubject
    - Voice 3 (Bass): Rest → Rest → Subject (at lower octave)
    """
    # Subject in C minor (dramatic, 2 measures)
    subject = [
        n("C5q"), n("Ef5q"), n("G5q"), n("C5q"),
        n("Af4e"), n("Bf4e"), n("C5e"), n("D5e"), n("Ef5q"), n("D5q"),
    ]

    # Countersubject (contrasting rhythm)
    cs = [
        n("Ef5e"), n("D5e"), n("C5e"), n("Bf4e"),
        n("Af4q"), n("G4q"),
        n("F4e"), n("G4e"), n("Af4e"), n("Bf4e"), n("C5q"), n("Bf4q"),
    ]

    # Answer in G minor (transposed up a 5th = 7 semitones)
    answer = _transpose_phrase(subject, 7)
    answer = _transpose_phrase(answer, -12)  # bring down an octave for alto range

    # Subject at bass register
    bass_subject = _transpose_phrase(subject, -12)

    # Free counterpoint for soprano during answer
    free_s = [
        n("G5e"), n("F5e"), n("Ef5e"), n("D5e"),
        n("C5q"), n("Bf4q"),
        n("Af4e"), n("Bf4e"), n("C5e"), n("D5e"), n("Ef5q"), n("D5q"),
    ]

    # Free counterpoint for alto during bass entry
    free_a = [
        n("G4e"), n("Af4e"), n("Bf4e"), n("C5e"),
        n("D5q"), n("Ef5q"),
        n("D5e"), n("C5e"), n("Bf4e"), n("Af4e"), n("G4q"), n("F4q"),
    ]

    # Rests (2 measures of 4/4 = 8 beats)
    two_measures_rest = [rest("1"), rest("1")]
    one_measure_rest = [rest("1")]

    # Free conclusion for soprano
    free_s2 = [
        n("C5e"), n("D5e"), n("Ef5e"), n("F5e"),
        n("G5q"), n("Af5q"),
        n("G5e"), n("F5e"), n("Ef5e"), n("D5e"),
        n("C5w"),
    ]

    # Free conclusion for bass
    free_b = [
        n("C3e"), n("D3e"), n("Ef3e"), n("F3e"),
        n("G3q"), n("Af3q"),
        n("G3h"),
        n("C3w"),
    ]

    return _compose_from_sections(
        [
            [subject, two_measures_rest, two_measures_rest],
            [free_s, answer, two_measures_rest],
            [free_s2, free_a, bass_subject + free_b],
        ],
        section_names=("Entry 1 (Soprano)", "Entry 2 (Alto)",
                       "Entry 3 (Bass) + Conclusion"),
        voice_names=["Soprano", "Alto", "Bass"],
        title=title, composer=composer,
        key="c \\minor",
    )


# ---------------------------------------------------------------------------
# Theme and Variations
# ---------------------------------------------------------------------------

def compose_theme_and_variations(
    title: str = "Theme and Variations in G Major",
    composer: str = "Composed via Session Types",
) -> MusicResult:
    """Compose a theme with 3 variations.

    - Theme: Simple melody in G major (8 measures)
    - Var 1: Ornamented (eighth note motion)
    - Var 2: Minor mode (transposed to G minor)
    - Var 3: Rhythmic transformation (dotted rhythms)
    - Coda: Return to theme
    """
    # Theme (soprano, 2 phrases of 4 measures each)
    theme_a = [
        n("G4q"), n("A4q"), n("B4q"), n("A4q"),
        n("G4q"), n("Fs4q"), n("E4q"), n("D4q"),
    ]
    theme_b = [
        n("E4q"), n("Fs4q"), n("G4q"), n("A4q"),
        n("B4q"), n("A4q"), n("G4h"),
    ]
    theme = theme_a + theme_b

    # Bass line
    bass_a = [
        n("G3h"), n("D3h"), n("E3h"), n("B2h"),
    ]
    bass_b = [
        n("C3h"), n("D3h"), n("G3q"), n("D3q"), n("G3h"),
    ]
    bass_theme = bass_a + bass_b

    # Variation 1: Ornamented (eighth note runs)
    var1 = [
        n("G4e"), n("A4e"), n("B4e"), n("A4e"), n("G4e"), n("Fs4e"), n("G4e"), n("A4e"),
        n("B4e"), n("C5e"), n("B4e"), n("A4e"), n("G4e"), n("Fs4e"), n("E4e"), n("D4e"),
        n("E4e"), n("Fs4e"), n("G4e"), n("Fs4e"), n("E4e"), n("D4e"), n("E4e"), n("Fs4e"),
        n("G4e"), n("A4e"), n("B4e"), n("A4e"), n("G4q"), n("A4q"), n("G4h"),
    ]
    bass_var1 = bass_theme  # same bass

    # Variation 2: Minor mode
    var2 = [
        n("G4q"), n("A4q"), n("Bf4q"), n("A4q"),
        n("G4q"), n("F4q"), n("Ef4q"), n("D4q"),
        n("Ef4q"), n("F4q"), n("G4q"), n("A4q"),
        n("Bf4q"), n("A4q"), n("G4h"),
    ]
    bass_var2 = [
        n("G3h"), n("D3h"), n("Ef3h"), n("Bf2h"),
        n("C3h"), n("D3h"), n("G3q"), n("D3q"), n("G3h"),
    ]

    # Variation 3: Augmented (half notes) — stately
    var3 = _augment(theme_a) + _augment(theme_b)
    bass_var3 = [
        n("G3w"), n("D3w"), n("E3w"), n("B2w"),
        n("C3w"), n("D3w"), n("G3h"), n("D3h"), n("G3w"),
    ]

    # Coda: theme returns with final cadence
    coda_rh = theme_a + [
        n("D5q"), n("C5q"), n("B4q"), n("A4q"),
        n("G4w"),
    ]
    coda_lh = bass_a + [n("D3h"), n("G2h"), n("G3w")]

    return _compose_from_sections(
        [
            [theme, bass_theme],
            [var1, bass_var1],
            [var2, bass_var2],
            [var3, bass_var3],
            [coda_rh, coda_lh],
        ],
        section_names=("Theme", "Var. 1 (Ornamented)", "Var. 2 (Minor)",
                       "Var. 3 (Augmented)", "Coda"),
        voice_names=["Right Hand", "Left Hand"],
        title=title, composer=composer,
        key="g \\major",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PIECES = {
    "invention": ("Two-Part Invention in C Major", compose_invention),
    "chorale": ("Chorale in C Major", compose_chorale),
    "fugue": ("Fugue Exposition in C Minor", compose_fugue_exposition),
    "variations": ("Theme and Variations in G Major", compose_theme_and_variations),
}


def main(argv: list[str] | None = None) -> None:
    """CLI for algorithmic composition."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="reticulate.composer",
        description="Compose music using session type structure",
    )
    parser.add_argument(
        "piece", choices=list(_PIECES.keys()),
        help="Which piece to compose",
    )
    parser.add_argument("--output", "-o", help="Output .ly file")
    parser.add_argument("--title", "-t", help="Override title")
    parser.add_argument("--composer", "-c", help="Override composer")
    parser.add_argument("--show-type", action="store_true",
                       help="Print the session type string")

    args = parser.parse_args(argv)

    default_title, compose_fn = _PIECES[args.piece]

    kwargs = {}
    if args.title:
        kwargs["title"] = args.title
    if args.composer:
        kwargs["composer"] = args.composer

    result = compose_fn(**kwargs)
    music = result.music_result

    if args.show_type:
        print(f"Session type ({len(result.session_type)} chars):")
        # Print truncated for readability
        st = result.session_type
        if len(st) > 500:
            print(f"  {st[:250]}...")
            print(f"  ...{st[-250:]}")
        else:
            print(f"  {st}")
        print(f"\nSections: {', '.join(result.sections)}")
        n_notes = sum(len(v.notes) for v in music.voices)
        print(f"Voices: {len(music.voices)}, Notes: {n_notes}")
        return

    if args.output:
        with open(args.output, "w") as f:
            f.write(music.lilypond_source)
        n_notes = sum(len(v.notes) for v in music.voices)
        print(f"Wrote {args.output}: {len(music.voices)} voices, {n_notes} notes",
              file=sys.stderr)
        print(f"Session type: {len(result.session_type)} chars", file=sys.stderr)
        print(f"Sections: {', '.join(result.sections)}", file=sys.stderr)
    else:
        print(music.lilypond_source)


if __name__ == "__main__":
    main()
