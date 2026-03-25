"""Tests for the music module — session types to LilyPond scores."""

import pytest
from reticulate.music import (
    MusicalNote,
    MusicalPath,
    parse_musical_label,
    is_musical_label,
    note_to_lilypond,
    extract_musical_paths,
    extract_polyphonic_score,
    generate_lilypond,
    session_to_music,
    C_MAJOR_SCALE,
    CADENCE_IV_V_I,
    CANON_SIMPLE,
    MINUET_PATTERN,
    BACH_INVENTION_1,
    BACH_INVENTION_FULL,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

class TestParseMusicalLabel:

    def test_simple_note(self):
        n = parse_musical_label("C4q")
        assert n == MusicalNote("c", "", 4, "4", False)

    def test_sharp(self):
        n = parse_musical_label("Cs4q")
        assert n == MusicalNote("c", "is", 4, "4", False)

    def test_flat(self):
        n = parse_musical_label("Bf3h")
        assert n == MusicalNote("b", "es", 3, "2", False)

    def test_dotted(self):
        n = parse_musical_label("G5ed")
        assert n == MusicalNote("g", "", 5, "8", True)

    def test_whole_note(self):
        n = parse_musical_label("C4w")
        assert n == MusicalNote("c", "", 4, "1", False)

    def test_half_note(self):
        n = parse_musical_label("D5h")
        assert n == MusicalNote("d", "", 5, "2", False)

    def test_eighth_note(self):
        n = parse_musical_label("E4e")
        assert n == MusicalNote("e", "", 4, "8", False)

    def test_sixteenth_note(self):
        n = parse_musical_label("F4x")
        assert n == MusicalNote("f", "", 4, "16", False)

    def test_rest_quarter(self):
        n = parse_musical_label("Rq")
        assert n is not None
        assert n.is_rest
        assert n.duration == "4"

    def test_rest_half_dotted(self):
        n = parse_musical_label("Rhd")
        assert n is not None
        assert n.is_rest
        assert n.duration == "2"
        assert n.dotted

    def test_lowercase(self):
        n = parse_musical_label("c4q")
        assert n == MusicalNote("c", "", 4, "4", False)

    def test_invalid_returns_none(self):
        assert parse_musical_label("hello") is None
        assert parse_musical_label("open") is None
        assert parse_musical_label("") is None
        assert parse_musical_label("X") is None

    def test_octave_0(self):
        n = parse_musical_label("C0w")
        assert n is not None
        assert n.octave == 0

    def test_octave_8(self):
        n = parse_musical_label("C8q")
        assert n is not None
        assert n.octave == 8

    def test_sharp_dotted(self):
        n = parse_musical_label("Fs4qd")
        assert n == MusicalNote("f", "is", 4, "4", True)


class TestIsMusicalLabel:

    def test_musical(self):
        assert is_musical_label("C4q")
        assert is_musical_label("Gs5e")
        assert is_musical_label("Rh")

    def test_non_musical(self):
        assert not is_musical_label("open")
        assert not is_musical_label("close")
        assert not is_musical_label("")


# ---------------------------------------------------------------------------
# LilyPond conversion
# ---------------------------------------------------------------------------

class TestNoteToLilypond:

    def test_middle_c_quarter(self):
        n = MusicalNote("c", "", 4, "4", False)
        assert note_to_lilypond(n) == "c'4"

    def test_c3_no_mark(self):
        n = MusicalNote("c", "", 3, "4", False)
        assert note_to_lilypond(n) == "c4"

    def test_c5_two_marks(self):
        n = MusicalNote("c", "", 5, "4", False)
        assert note_to_lilypond(n) == "c''4"

    def test_c2_comma(self):
        n = MusicalNote("c", "", 2, "4", False)
        assert note_to_lilypond(n) == "c,4"

    def test_c1_two_commas(self):
        n = MusicalNote("c", "", 1, "4", False)
        assert note_to_lilypond(n) == "c,,4"

    def test_sharp(self):
        n = MusicalNote("f", "is", 4, "4", False)
        assert note_to_lilypond(n) == "fis'4"

    def test_flat(self):
        n = MusicalNote("b", "es", 3, "2", False)
        assert note_to_lilypond(n) == "bes2"

    def test_dotted(self):
        n = MusicalNote("g", "", 4, "8", True)
        assert note_to_lilypond(n) == "g'8."

    def test_rest(self):
        n = MusicalNote("r", "", -1, "4", False)
        assert note_to_lilypond(n) == "r4"

    def test_rest_dotted(self):
        n = MusicalNote("r", "", -1, "2", True)
        assert note_to_lilypond(n) == "r2."

    def test_whole_note(self):
        n = MusicalNote("c", "", 4, "1", False)
        assert note_to_lilypond(n) == "c'1"


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

class TestExtractMusicalPaths:

    def test_simple_melody(self):
        ast = parse("&{C4q: &{D4q: &{E4q: end}}}")
        ss = build_statespace(ast)
        paths, warnings = extract_musical_paths(ss)
        assert len(paths) == 1
        assert len(paths[0].notes) == 3
        assert paths[0].notes[0].pitch == "c"
        assert paths[0].notes[1].pitch == "d"
        assert paths[0].notes[2].pitch == "e"
        assert warnings == []

    def test_non_musical_labels_warn(self):
        ast = parse("&{open: &{C4q: &{close: end}}}")
        ss = build_statespace(ast)
        paths, warnings = extract_musical_paths(ss)
        assert len(paths) == 1
        assert len(paths[0].notes) == 1  # only C4q
        assert len(warnings) == 2  # open and close

    def test_scale(self):
        ast = parse(C_MAJOR_SCALE)
        ss = build_statespace(ast)
        paths, warnings = extract_musical_paths(ss)
        assert len(paths) == 1
        assert len(paths[0].notes) == 8


class TestExtractPolyphonicScore:

    def test_parallel_two_voices(self):
        ast = parse(CADENCE_IV_V_I)
        ss = build_statespace(ast)
        voices, warnings = extract_polyphonic_score(ss)
        assert len(voices) == 2
        assert voices[0].voice_name == "Right Hand"
        assert voices[1].voice_name == "Left Hand"

    def test_single_voice_fallback(self):
        ast = parse(C_MAJOR_SCALE)
        ss = build_statespace(ast)
        voices, warnings = extract_polyphonic_score(ss)
        assert len(voices) == 1

    def test_custom_voice_names(self):
        ast = parse(CADENCE_IV_V_I)
        ss = build_statespace(ast)
        voices, _ = extract_polyphonic_score(ss, voice_names=["Soprano", "Bass"])
        assert voices[0].voice_name == "Soprano"
        assert voices[1].voice_name == "Bass"


# ---------------------------------------------------------------------------
# LilyPond generation
# ---------------------------------------------------------------------------

class TestGenerateLilypond:

    def test_version_header(self):
        ly = generate_lilypond([])
        assert '\\version "2.24.0"' in ly

    def test_title_in_header(self):
        path = MusicalPath((MusicalNote("c", "", 4, "4", False),), "Voice")
        ly = generate_lilypond([path], title="Test Piece")
        assert 'title = "Test Piece"' in ly

    def test_composer_in_header(self):
        path = MusicalPath((MusicalNote("c", "", 4, "4", False),), "Voice")
        ly = generate_lilypond([path], composer="Bach")
        assert 'composer = "Bach"' in ly

    def test_single_voice_no_staffgroup(self):
        path = MusicalPath((MusicalNote("c", "", 4, "4", False),), "Voice")
        ly = generate_lilypond([path])
        assert "StaffGroup" not in ly
        assert "c'4" in ly

    def test_two_voices_staffgroup(self):
        v1 = MusicalPath((MusicalNote("c", "", 5, "4", False),), "Right Hand")
        v2 = MusicalPath((MusicalNote("c", "", 3, "4", False),), "Left Hand")
        ly = generate_lilypond([v1, v2])
        assert "StaffGroup" in ly
        assert "Right Hand" in ly
        assert "Left Hand" in ly

    def test_bass_clef_for_low_notes(self):
        v = MusicalPath((MusicalNote("c", "", 2, "4", False),), "Bass")
        ly = generate_lilypond([v])
        assert "\\clef bass" in ly

    def test_treble_clef_for_high_notes(self):
        v = MusicalPath((MusicalNote("c", "", 5, "4", False),), "Treble")
        ly = generate_lilypond([v])
        assert "\\clef treble" in ly

    def test_key_signature(self):
        path = MusicalPath((MusicalNote("c", "", 4, "4", False),), "Voice")
        ly = generate_lilypond([path], key="g \\major")
        assert "\\key g \\major" in ly

    def test_time_signature(self):
        path = MusicalPath((MusicalNote("c", "", 4, "4", False),), "Voice")
        ly = generate_lilypond([path], time_sig="3/4")
        assert "\\time 3/4" in ly


# ---------------------------------------------------------------------------
# Integration: session_to_music
# ---------------------------------------------------------------------------

class TestSessionToMusic:

    def test_c_major_scale(self):
        r = session_to_music(C_MAJOR_SCALE, title="Scale")
        assert not r.is_polyphonic
        assert len(r.voices) == 1
        assert len(r.voices[0].notes) == 8
        assert "c'4" in r.lilypond_source

    def test_cadence_polyphonic(self):
        r = session_to_music(CADENCE_IV_V_I, title="Cadence")
        assert r.is_polyphonic
        assert len(r.voices) == 2
        assert "StaffGroup" in r.lilypond_source

    def test_canon(self):
        r = session_to_music(CANON_SIMPLE, title="Canon")
        assert r.is_polyphonic
        assert len(r.voices) == 2

    def test_minuet(self):
        r = session_to_music(MINUET_PATTERN, title="Minuet")
        assert r.is_polyphonic

    def test_bach_invention(self):
        r = session_to_music(BACH_INVENTION_1, title="Invention",
                            composer="J.S. Bach")
        assert r.is_polyphonic
        assert len(r.voices) == 2
        assert all(len(v.notes) == 8 for v in r.voices)
        assert "J.S. Bach" in r.lilypond_source

    def test_bach_invention_full(self):
        r = session_to_music(BACH_INVENTION_FULL, title="Invention No. 1",
                            composer="J.S. Bach")
        assert r.is_polyphonic
        assert len(r.voices) == 2
        # Right hand: 16 notes, Left hand: 15 notes + 1 rest
        assert len(r.voices[0].notes) == 16
        assert "StaffGroup" in r.lilypond_source

    def test_warnings_for_mixed_labels(self):
        r = session_to_music("&{open: &{C4q: &{close: end}}}")
        assert len(r.warnings) == 2

    def test_custom_voice_names(self):
        r = session_to_music(CADENCE_IV_V_I,
                            voice_names=["Soprano", "Bass"])
        assert "Soprano" in r.lilypond_source
        assert "Bass" in r.lilypond_source


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:

    def test_example_scale(self, capsys):
        from reticulate.music import main
        main(["--example", "scale"])
        out = capsys.readouterr().out
        assert "\\version" in out
        assert "c'" in out

    def test_example_invention(self, capsys):
        from reticulate.music import main
        main(["--example", "invention", "--title", "Test"])
        out = capsys.readouterr().out
        assert "StaffGroup" in out
        assert "Test" in out

    def test_type_string_arg(self, capsys):
        from reticulate.music import main
        main(["&{C4q: &{D4q: end}}"])
        out = capsys.readouterr().out
        assert "c'4" in out
        assert "d'4" in out

    def test_output_file(self, tmp_path):
        from reticulate.music import main
        outfile = tmp_path / "test.ly"
        main(["--example", "scale", "--output", str(outfile)])
        assert outfile.exists()
        content = outfile.read_text()
        assert "\\version" in content
