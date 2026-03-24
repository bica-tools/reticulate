"""Tests for the literature_scout module.

Tests cover:
- PaperResult and RelatedWorkReport dataclasses
- Title normalization and similarity
- arXiv XML response parsing
- DBLP JSON response parsing
- Semantic Scholar JSON response parsing
- Multi-source search with deduplication
- .bib parsing and new-work detection
"""

import os
import tempfile
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from reticulate.literature_scout import (
    PaperResult,
    RelatedWorkReport,
    _normalize_title,
    _titles_similar,
    _parse_arxiv_response,
    _parse_dblp_response,
    _parse_s2_response,
    _parse_bib_titles,
    _SOURCE_FUNCTIONS,
    find_related_work,
    check_for_new_work,
    search_arxiv,
    search_dblp,
    search_semantic_scholar,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestPaperResult:
    def test_paper_result_creation(self):
        p = PaperResult(
            title="Session Types",
            authors=["Honda", "Vasconcelos"],
            year=1998,
            source="arxiv",
        )
        assert p.title == "Session Types"
        assert len(p.authors) == 2
        assert p.year == 1998
        assert p.source == "arxiv"

    def test_paper_result_defaults(self):
        p = PaperResult(title="Test", authors=[])
        assert p.year is None
        assert p.venue == ""
        assert p.abstract == ""
        assert p.url == ""
        assert p.doi == ""
        assert p.arxiv_id == ""
        assert p.source == ""

    def test_paper_result_frozen(self):
        p = PaperResult(title="Test", authors=[])
        with pytest.raises(AttributeError):
            p.title = "Other"  # type: ignore

    def test_related_work_report(self):
        r = RelatedWorkReport(
            new_papers=[PaperResult(title="New", authors=[])],
            already_cited=[PaperResult(title="Old", authors=[])],
            keywords_used=["session types"],
        )
        assert len(r.new_papers) == 1
        assert len(r.already_cited) == 1
        assert r.keywords_used == ["session types"]


# ---------------------------------------------------------------------------
# Title similarity tests
# ---------------------------------------------------------------------------

class TestTitleSimilarity:
    def test_normalize_basic(self):
        assert _normalize_title("Session Types!") == "session types"

    def test_normalize_whitespace(self):
        assert _normalize_title("  Session   Types  ") == "session types"

    def test_normalize_punctuation(self):
        assert _normalize_title("Session-Types: A Survey.") == "sessiontypes a survey"

    def test_identical_titles(self):
        assert _titles_similar("Session Types", "Session Types")

    def test_case_insensitive(self):
        assert _titles_similar("Session Types", "session types")

    def test_different_titles(self):
        assert not _titles_similar("Session Types", "Lambda Calculus")

    def test_similar_with_minor_differences(self):
        assert _titles_similar(
            "Subtyping for Session Types in the Pi Calculus",
            "Subtyping for session types in the pi calculus",
        )

    def test_empty_titles(self):
        assert not _titles_similar("", "")
        assert not _titles_similar("Session", "")


# ---------------------------------------------------------------------------
# arXiv response parsing
# ---------------------------------------------------------------------------

ARXIV_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>Session Types and Lattice Theory</title>
    <summary>We study session types using lattice theory.</summary>
    <published>2023-01-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link href="http://arxiv.org/abs/2301.12345v1" rel="alternate"/>
    <link href="http://arxiv.org/pdf/2301.12345v1" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2302.54321v2</id>
    <title>Multiparty Session Types</title>
    <summary>A study of MPST.</summary>
    <published>2023-02-20T00:00:00Z</published>
    <author><name>Charlie Brown</name></author>
  </entry>
</feed>"""


class TestArxivParsing:
    def test_parse_arxiv_response(self):
        results = _parse_arxiv_response(ARXIV_SAMPLE)
        assert len(results) == 2

    def test_arxiv_first_paper(self):
        results = _parse_arxiv_response(ARXIV_SAMPLE)
        p = results[0]
        assert p.title == "Session Types and Lattice Theory"
        assert p.authors == ["Alice Smith", "Bob Jones"]
        assert p.year == 2023
        assert p.arxiv_id == "2301.12345"
        assert p.source == "arxiv"

    def test_arxiv_abstract(self):
        results = _parse_arxiv_response(ARXIV_SAMPLE)
        assert "lattice theory" in results[0].abstract

    def test_arxiv_malformed_xml(self):
        results = _parse_arxiv_response(b"not xml")
        assert results == []

    def test_arxiv_empty_feed(self):
        xml = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        results = _parse_arxiv_response(xml)
        assert results == []


# ---------------------------------------------------------------------------
# DBLP response parsing
# ---------------------------------------------------------------------------

class TestDblpParsing:
    def test_parse_dblp_response(self):
        data = {
            "result": {
                "hits": {
                    "hit": [
                        {
                            "info": {
                                "title": "Session Types for Objects.",
                                "authors": {"author": [
                                    {"text": "Simon Gay"},
                                    {"text": "Vasco Vasconcelos"},
                                ]},
                                "year": "2010",
                                "venue": "POPL",
                                "doi": "10.1145/1706299.1706335",
                                "url": "https://dblp.org/rec/conf/popl/GayV10",
                            }
                        }
                    ]
                }
            }
        }
        results = _parse_dblp_response(data)
        assert len(results) == 1
        p = results[0]
        assert p.title == "Session Types for Objects"  # trailing dot stripped
        assert "Simon Gay" in p.authors
        assert p.year == 2010
        assert p.source == "dblp"

    def test_parse_dblp_single_author(self):
        data = {
            "result": {"hits": {"hit": [
                {"info": {
                    "title": "Test Paper",
                    "authors": {"author": {"text": "Solo Author"}},
                    "year": "2020",
                }}
            ]}}
        }
        results = _parse_dblp_response(data)
        assert results[0].authors == ["Solo Author"]

    def test_parse_dblp_empty(self):
        assert _parse_dblp_response({}) == []
        assert _parse_dblp_response({"result": {}}) == []

    def test_parse_dblp_no_hits(self):
        data = {"result": {"hits": {"hit": []}}}
        assert _parse_dblp_response(data) == []


# ---------------------------------------------------------------------------
# Semantic Scholar response parsing
# ---------------------------------------------------------------------------

class TestS2Parsing:
    def test_parse_s2_response(self):
        data = {
            "data": [
                {
                    "title": "Lattice Theory in Session Types",
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "year": 2024,
                    "venue": "CONCUR",
                    "abstract": "We prove lattice properties.",
                    "url": "https://api.semanticscholar.org/...",
                    "externalIds": {
                        "DOI": "10.1000/test",
                        "ArXiv": "2401.99999",
                    },
                }
            ]
        }
        results = _parse_s2_response(data)
        assert len(results) == 1
        p = results[0]
        assert p.title == "Lattice Theory in Session Types"
        assert p.year == 2024
        assert p.doi == "10.1000/test"
        assert p.arxiv_id == "2401.99999"
        assert p.source == "semantic_scholar"

    def test_parse_s2_empty(self):
        assert _parse_s2_response({}) == []
        assert _parse_s2_response({"data": []}) == []

    def test_parse_s2_missing_fields(self):
        data = {"data": [{"title": "Minimal", "authors": []}]}
        results = _parse_s2_response(data)
        assert len(results) == 1
        assert results[0].title == "Minimal"
        assert results[0].year is None


# ---------------------------------------------------------------------------
# .bib parsing
# ---------------------------------------------------------------------------

class TestBibParsing:
    def test_parse_bib_titles(self):
        bib = """\
@inproceedings{honda1998,
  author = {Honda, Kohei},
  title  = {Language Primitives and Type Discipline},
  year   = {1998},
}

@article{gay2005subtyping,
  author = {Gay, Simon J.},
  title  = {Subtyping for Session Types in the Pi Calculus},
  year   = {2005},
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            f.write(bib)
            f.flush()
            try:
                titles = _parse_bib_titles(f.name)
                assert "honda1998" in titles
                assert "Language Primitives" in titles["honda1998"]
                assert "gay2005subtyping" in titles
            finally:
                os.unlink(f.name)

    def test_parse_bib_nonexistent(self):
        titles = _parse_bib_titles("/nonexistent/path/file.bib")
        assert titles == {}


# ---------------------------------------------------------------------------
# find_related_work (with mocked network)
# ---------------------------------------------------------------------------

class TestFindRelatedWork:
    def test_deduplication(self):
        def fake_search(query, max_results=20):
            return [PaperResult(title="Session Types Survey", authors=["A"], source="fake")]

        original = dict(_SOURCE_FUNCTIONS)
        try:
            _SOURCE_FUNCTIONS["src1"] = fake_search
            _SOURCE_FUNCTIONS["src2"] = fake_search
            report = find_related_work(["session types"], sources=("src1", "src2"))
            # Should deduplicate identical titles from two sources
            assert len(report.new_papers) == 1
        finally:
            _SOURCE_FUNCTIONS.clear()
            _SOURCE_FUNCTIONS.update(original)

    def test_multiple_keywords(self):
        call_count = 0
        def fake_search(query, max_results=20):
            nonlocal call_count
            call_count += 1
            return [PaperResult(title=f"Paper {call_count}", authors=[], source="fake")]

        original = dict(_SOURCE_FUNCTIONS)
        try:
            _SOURCE_FUNCTIONS["fake"] = fake_search
            report = find_related_work(["kw1", "kw2"], sources=("fake",))
            assert call_count == 2
            assert report.keywords_used == ["kw1", "kw2"]
        finally:
            _SOURCE_FUNCTIONS.clear()
            _SOURCE_FUNCTIONS.update(original)

    def test_unknown_source_ignored(self):
        report = find_related_work(["test"], sources=("unknown_source",))
        assert isinstance(report, RelatedWorkReport)
        assert report.new_papers == []


# ---------------------------------------------------------------------------
# check_for_new_work (with mocked network)
# ---------------------------------------------------------------------------

class TestCheckForNewWork:
    @patch("reticulate.literature_scout.find_related_work")
    def test_separates_new_and_cited(self, mock_frw):
        mock_frw.return_value = RelatedWorkReport(
            new_papers=[
                PaperResult(title="Language Primitives and Type Discipline", authors=[], source="arxiv"),
                PaperResult(title="Totally New Paper on Quantum Types", authors=[], source="arxiv"),
            ],
            already_cited=[],
            keywords_used=["test"],
        )
        bib = """\
@inproceedings{honda1998,
  title = {Language Primitives and Type Discipline},
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            f.write(bib)
            f.flush()
            try:
                report = check_for_new_work(f.name, ["test"])
                assert len(report.already_cited) == 1
                assert len(report.new_papers) == 1
                assert "Quantum" in report.new_papers[0].title
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# Network-level search functions (mocked)
# ---------------------------------------------------------------------------

class TestNetworkSearch:
    @patch("urllib.request.urlopen")
    def test_search_arxiv_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        results = search_arxiv("test")
        assert results == []

    @patch("urllib.request.urlopen")
    def test_search_dblp_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        results = search_dblp("test")
        assert results == []

    @patch("urllib.request.urlopen")
    def test_search_semantic_scholar_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        results = search_semantic_scholar("test")
        assert results == []
