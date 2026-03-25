"""Tests for the venue matching agent.

Covers:
- Venue dataclass construction and immutability
- VenueMatchResult dataclass
- VENUE_DATABASE completeness (31 venues)
- Keyword extraction from LaTeX files
- Jaccard similarity computation
- Venue ranking logic
- Recommendation labels
- Edge cases (empty keywords, missing files, etc.)
"""

import pytest
import tempfile
from pathlib import Path

from reticulate.venue_matcher import (
    Venue,
    VenueMatchResult,
    VENUE_DATABASE,
    extract_paper_keywords,
    match_score,
    rank_venues,
    _clean_latex,
    _tokenize,
    _recommend,
)


# ---------------------------------------------------------------------------
# Venue dataclass
# ---------------------------------------------------------------------------

class TestVenueDataclass:
    def test_venue_frozen(self) -> None:
        v = Venue("Test", "T", ("a",), 0.3, "A")
        with pytest.raises(AttributeError):
            v.name = "X"  # type: ignore[misc]

    def test_venue_fields(self) -> None:
        v = VENUE_DATABASE["CONCUR"]
        assert v.name == "International Conference on Concurrency Theory"
        assert v.acronym == "CONCUR"
        assert isinstance(v.scope_keywords, tuple)
        assert 0.0 < v.acceptance_rate < 1.0
        assert v.tier in ("A*", "A", "B", "C", "workshop", "journal")

    def test_venue_match_result_frozen(self) -> None:
        v = VENUE_DATABASE["ICE"]
        r = VenueMatchResult(v, 0.5, ("session types",), "strong")
        with pytest.raises(AttributeError):
            r.score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Venue database completeness
# ---------------------------------------------------------------------------

class TestVenueDatabase:
    EXPECTED_ACRONYMS = {
        "CONCUR", "ICE", "PLACES", "FORTE", "COORDINATION", "TACAS", "CAV",
        "LICS", "FoSSaCS", "ESOP", "POPL", "OOPSLA", "ECOOP", "ICFP",
        "APLAS", "ICTAC", "CALCO", "AAMAS", "BPM", "ISMIR", "CMSB",
        "CCS", "VLDB", "ITiCSE", "FC", "IoTDI", "JLAMP", "LMCS",
        "TOPLAS", "MSCS", "SCP",
    }

    def test_all_venues_present(self) -> None:
        assert set(VENUE_DATABASE.keys()) == self.EXPECTED_ACRONYMS

    def test_venue_count(self) -> None:
        assert len(VENUE_DATABASE) >= 25

    def test_all_have_keywords(self) -> None:
        for acronym, venue in VENUE_DATABASE.items():
            assert len(venue.scope_keywords) >= 5, f"{acronym} has too few keywords"

    def test_acceptance_rates_valid(self) -> None:
        for acronym, venue in VENUE_DATABASE.items():
            assert 0.0 < venue.acceptance_rate <= 1.0, f"{acronym} rate invalid"

    def test_tiers_valid(self) -> None:
        valid_tiers = {"A*", "A", "B", "C", "workshop", "journal"}
        for acronym, venue in VENUE_DATABASE.items():
            assert venue.tier in valid_tiers, f"{acronym} tier invalid: {venue.tier}"


# ---------------------------------------------------------------------------
# LaTeX cleaning and tokenization
# ---------------------------------------------------------------------------

class TestLatexCleaning:
    def test_clean_command(self) -> None:
        assert "session types" in _clean_latex(r"\textbf{session types}")

    def test_clean_math(self) -> None:
        result = _clean_latex(r"we prove $S_1 \spar S_2$ forms a lattice")
        assert "prove" in result
        assert "$" not in result

    def test_clean_braces(self) -> None:
        result = _clean_latex("{hello} world")
        assert "{" not in result

    def test_tokenize_stopwords(self) -> None:
        tokens = _tokenize("the session types of a protocol")
        assert "the" not in tokens
        assert "session" in tokens
        assert "protocol" in tokens

    def test_tokenize_lowercase(self) -> None:
        tokens = _tokenize("Session Types AND Lattices")
        assert "session" in tokens
        assert "lattices" in tokens

    def test_tokenize_short_words(self) -> None:
        tokens = _tokenize("a b cd ef")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cd" in tokens
        assert "ef" in tokens


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    def _write_tex(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False)
        f.write(content)
        f.close()
        return Path(f.name)

    def test_extract_title(self) -> None:
        p = self._write_tex(r"\title{Session Types and Lattice Theory}" "\n"
                            r"\begin{document}\end{document}")
        kws = extract_paper_keywords(p)
        assert "session" in kws
        assert "lattice" in kws

    def test_extract_abstract(self) -> None:
        p = self._write_tex(
            r"\title{Test}" "\n"
            r"\begin{document}" "\n"
            r"\begin{abstract}" "\n"
            "We study concurrency and bisimulation in protocol verification.\n"
            r"\end{abstract}" "\n"
            r"\end{document}")
        kws = extract_paper_keywords(p)
        assert "concurrency" in kws
        assert "bisimulation" in kws

    def test_extract_sections(self) -> None:
        p = self._write_tex(
            r"\title{Test}" "\n"
            r"\begin{document}" "\n"
            r"\section{Product Construction}" "\n"
            r"\subsection{Parallel Composition}" "\n"
            r"\end{document}")
        kws = extract_paper_keywords(p)
        assert "product" in kws
        assert "parallel" in kws
        assert "composition" in kws

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_paper_keywords("/nonexistent/path.tex")

    def test_real_paper(self) -> None:
        """Extract keywords from a real paper in the repo."""
        p = Path(__file__).parent.parent.parent / "papers" / "steps" / "step1-statespace" / "main.tex"
        if not p.exists():
            pytest.skip("step1 paper not found")
        kws = extract_paper_keywords(p)
        # The step1 paper is about state-space construction for session types
        assert len(kws) > 10
        # Should find at least some core terms
        assert "session" in kws or "state" in kws


# ---------------------------------------------------------------------------
# Match score (Jaccard)
# ---------------------------------------------------------------------------

class TestMatchScore:
    def test_identical_sets(self) -> None:
        v = Venue("T", "T", ("a", "b", "c"), 0.3, "A")
        assert match_score({"a", "b", "c"}, v) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        v = Venue("T", "T", ("a", "b"), 0.3, "A")
        assert match_score({"x", "y"}, v) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        v = Venue("T", "T", ("a", "b", "c"), 0.3, "A")
        # intersection = {a}, union = {a,b,c,x} = 4
        score = match_score({"a", "x"}, v)
        assert score == pytest.approx(1.0 / 4.0)

    def test_empty_paper_keywords(self) -> None:
        v = Venue("T", "T", ("a",), 0.3, "A")
        assert match_score(set(), v) == pytest.approx(0.0)

    def test_symmetric_like(self) -> None:
        """Jaccard is symmetric in the set sizes."""
        v = Venue("T", "T", ("a", "b"), 0.3, "A")
        s1 = match_score({"a", "c"}, v)
        # intersection={a}, union={a,b,c} => 1/3
        assert s1 == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Recommendation labels
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_strong(self) -> None:
        assert _recommend(0.30) == "strong"

    def test_good(self) -> None:
        assert _recommend(0.18) == "good"

    def test_possible(self) -> None:
        assert _recommend(0.12) == "possible"

    def test_weak(self) -> None:
        assert _recommend(0.06) == "weak"

    def test_poor(self) -> None:
        assert _recommend(0.02) == "poor"

    def test_boundary_strong(self) -> None:
        assert _recommend(0.25) == "strong"

    def test_boundary_good(self) -> None:
        assert _recommend(0.15) == "good"


# ---------------------------------------------------------------------------
# Venue ranking
# ---------------------------------------------------------------------------

class TestRankVenues:
    def _write_session_types_paper(self) -> Path:
        content = (
            r"\title{Session Types as Algebraic Reticulates}" "\n"
            r"\begin{document}" "\n"
            r"\begin{abstract}" "\n"
            "We prove that session type state spaces form lattices. "
            "Our approach uses bisimulation, process algebra, and "
            "concurrency theory to establish lattice properties for "
            "protocol verification with type systems.\n"
            r"\end{abstract}" "\n"
            r"\section{Session Type Constructors}" "\n"
            r"\section{Lattice Structure}" "\n"
            r"\section{Verification}" "\n"
            r"\end{document}"
        )
        f = tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False)
        f.write(content)
        f.close()
        return Path(f.name)

    def test_top_n(self) -> None:
        p = self._write_session_types_paper()
        results = rank_venues(p, top_n=5)
        assert len(results) == 5

    def test_results_sorted_descending(self) -> None:
        p = self._write_session_types_paper()
        results = rank_venues(p, top_n=10)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_session_types_paper_prefers_concurrency_venues(self) -> None:
        """A session types paper should rank CONCUR/ICE/PLACES highly."""
        p = self._write_session_types_paper()
        results = rank_venues(p, top_n=10)
        top_acronyms = {r.venue.acronym for r in results[:5]}
        # At least one of these should be in top 5
        session_venues = {"CONCUR", "ICE", "PLACES", "APLAS", "FoSSaCS", "LICS", "CALCO"}
        assert top_acronyms & session_venues, f"No session type venue in top 5: {top_acronyms}"

    def test_shared_keywords_populated(self) -> None:
        p = self._write_session_types_paper()
        results = rank_venues(p, top_n=3)
        # Top result should have at least one shared keyword
        assert len(results[0].shared_keywords) > 0

    def test_recommendation_populated(self) -> None:
        p = self._write_session_types_paper()
        results = rank_venues(p, top_n=5)
        valid = {"strong", "good", "possible", "weak", "poor"}
        for r in results:
            assert r.recommendation in valid

    def test_custom_venue_database(self) -> None:
        p = self._write_session_types_paper()
        custom = {
            "TEST": Venue("Test Venue", "TEST",
                          ("session", "lattice", "concurrency"), 0.5, "B"),
        }
        results = rank_venues(p, top_n=5, venues=custom)
        assert len(results) == 1
        assert results[0].venue.acronym == "TEST"
        assert results[0].score > 0.0

    def test_irrelevant_paper_low_scores(self) -> None:
        """A paper on music should not match concurrency venues well."""
        content = (
            r"\title{Melody Recognition in Jazz Improvisation}" "\n"
            r"\begin{document}" "\n"
            r"\begin{abstract}" "\n"
            "We study melody recognition and rhythm classification "
            "using machine learning on audio signals.\n"
            r"\end{abstract}" "\n"
            r"\end{document}"
        )
        f = tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False)
        f.write(content)
        f.close()
        results = rank_venues(Path(f.name), top_n=3)
        # ISMIR should rank highly for a music paper
        top_acronyms = {r.venue.acronym for r in results[:3]}
        assert "ISMIR" in top_acronyms
