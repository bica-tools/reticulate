"""Algorithmic composition using session type structure.

Composes multi-section pieces by building session types programmatically,
then translating to LilyPond via the music module.

Compositional forms:
- Two-part invention (Bach style)
- Chorale (SATB hymn setting)
- Theme and variations
- Minuet and trio

Each form is encoded as a session type template with musical labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from reticulate.music import (
    MusicalNote, MusicalPath, MusicResult,
    parse_musical_label, note_to_lilypond, generate_lilypond,
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

    # === ASSEMBLE ===
    rh_notes = (
        subject +           # m1-2: subject
        counter_answer +    # m3-4: counter-answer while LH has answer
        episode1_rh +       # m5-6: episode 1
        middle_cs_rh +      # m7-8: countersubject
        episode2_rh +       # m9-10: episode 2
        recap_rh +          # m11-12: recapitulation
        coda_rh             # m13-14: coda
    )
    lh_notes = (
        countersubject +    # m1-2: countersubject
        answer +            # m3-4: answer
        episode1_lh +       # m5-6: episode 1
        middle_subject_lh + # m7-8: subject in LH
        episode2_lh +       # m9-10: episode 2
        recap_lh +          # m11-12: free bass
        coda_lh             # m13-14: coda
    )

    rh_path = MusicalPath(tuple(rh_notes), "Right Hand")
    lh_path = MusicalPath(tuple(lh_notes), "Left Hand")

    ly = generate_lilypond(
        [rh_path, lh_path],
        title=title,
        composer=composer,
        key="c \\major",
        time_sig="4/4",
    )

    return MusicResult(
        voices=(rh_path, lh_path),
        lilypond_source=ly,
        is_polyphonic=True,
        warnings=(),
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

    voices = [
        MusicalPath(tuple(soprano), "Soprano"),
        MusicalPath(tuple(alto), "Alto"),
        MusicalPath(tuple(tenor), "Tenor"),
        MusicalPath(tuple(bass), "Bass"),
    ]

    ly = generate_lilypond(voices, title=title, composer=composer,
                           key="c \\major", time_sig="4/4")
    return MusicResult(
        voices=tuple(voices), lilypond_source=ly,
        is_polyphonic=True, warnings=(),
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

    # Assemble voices
    soprano_notes = subject + free_s + free_s2
    alto_notes = two_measures_rest + answer + free_a
    bass_notes = two_measures_rest + two_measures_rest + bass_subject + free_b

    voices = [
        MusicalPath(tuple(soprano_notes), "Soprano"),
        MusicalPath(tuple(alto_notes), "Alto"),
        MusicalPath(tuple(bass_notes), "Bass"),
    ]

    ly = generate_lilypond(voices, title=title, composer=composer,
                           key="c \\minor", time_sig="4/4")
    return MusicResult(
        voices=tuple(voices), lilypond_source=ly,
        is_polyphonic=True, warnings=(),
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

    # Separator rests (double barline effect)
    sep = [rest("2")]

    # Assemble
    rh = theme + sep + var1 + sep + var2 + sep + var3 + sep + coda_rh
    lh = bass_theme + sep + bass_var1 + sep + bass_var2 + sep + bass_var3 + sep + coda_lh

    voices = [
        MusicalPath(tuple(rh), "Right Hand"),
        MusicalPath(tuple(lh), "Left Hand"),
    ]

    ly = generate_lilypond(voices, title=title, composer=composer,
                           key="g \\major", time_sig="4/4")
    return MusicResult(
        voices=tuple(voices), lilypond_source=ly,
        is_polyphonic=True, warnings=(),
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

    args = parser.parse_args(argv)

    default_title, compose_fn = _PIECES[args.piece]

    kwargs = {}
    if args.title:
        kwargs["title"] = args.title
    if args.composer:
        kwargs["composer"] = args.composer

    result = compose_fn(**kwargs)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result.lilypond_source)
        n_notes = sum(len(v.notes) for v in result.voices)
        print(f"Wrote {args.output}: {len(result.voices)} voices, {n_notes} notes",
              file=sys.stderr)
    else:
        print(result.lilypond_source)


if __name__ == "__main__":
    main()
