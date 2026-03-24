"""Tests for the librarian module.

Tests cover:
- DownloadResult dataclass
- .bib parsing
- PDF detection
- download_from_arxiv (mocked)
- download_from_doi (mocked)
- download_references (mocked)
- organize_references
- check_missing
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from reticulate.librarian import (
    DownloadResult,
    _parse_bib_entries,
    _is_pdf,
    download_from_doi,
    download_from_arxiv,
    download_references,
    organize_references,
    check_missing,
    _download_pdf,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDownloadResult:
    def test_creation(self):
        r = DownloadResult(
            key="honda1998",
            title="Language Primitives",
            path="/tmp/honda1998.pdf",
            success=True,
            source="arxiv",
            size_bytes=12345,
        )
        assert r.key == "honda1998"
        assert r.success
        assert r.size_bytes == 12345

    def test_defaults(self):
        r = DownloadResult(key="test", title="", path="", success=False, source="")
        assert r.size_bytes == 0
        assert r.error == ""

    def test_frozen(self):
        r = DownloadResult(key="test", title="", path="", success=False, source="")
        with pytest.raises(AttributeError):
            r.key = "other"  # type: ignore


# ---------------------------------------------------------------------------
# .bib parsing
# ---------------------------------------------------------------------------

SAMPLE_BIB = """\
@inproceedings{honda1998,
  author    = {Honda, Kohei and Vasconcelos, Vasco T. and Kubo, Makoto},
  title     = {Language Primitives and Type Discipline},
  booktitle = {ESOP},
  year      = {1998},
  doi       = {10.1007/BFb0053567},
}

@article{gay2005subtyping,
  author  = {Gay, Simon J. and Hole, Malcolm},
  title   = {Subtyping for Session Types},
  journal = {Acta Informatica},
  year    = {2005},
  doi     = {10.1007/s00236-005-0177-z},
  eprint  = {cs/0501015},
}
"""


class TestBibParsing:
    def test_parse_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            f.write(SAMPLE_BIB)
            f.flush()
            try:
                entries = _parse_bib_entries(f.name)
                assert len(entries) == 2
                assert entries[0]["key"] == "honda1998"
                assert entries[1]["key"] == "gay2005subtyping"
            finally:
                os.unlink(f.name)

    def test_parse_doi(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            f.write(SAMPLE_BIB)
            f.flush()
            try:
                entries = _parse_bib_entries(f.name)
                assert entries[0]["doi"] == "10.1007/BFb0053567"
            finally:
                os.unlink(f.name)

    def test_parse_eprint(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            f.write(SAMPLE_BIB)
            f.flush()
            try:
                entries = _parse_bib_entries(f.name)
                assert entries[1].get("eprint") == "cs/0501015"
            finally:
                os.unlink(f.name)

    def test_parse_nonexistent(self):
        entries = _parse_bib_entries("/nonexistent/path.bib")
        assert entries == []

    def test_parse_title(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bib", delete=False) as f:
            f.write(SAMPLE_BIB)
            f.flush()
            try:
                entries = _parse_bib_entries(f.name)
                assert "Language Primitives" in entries[0]["title"]
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# PDF detection
# ---------------------------------------------------------------------------

class TestPdfDetection:
    def test_valid_pdf(self):
        assert _is_pdf(b"%PDF-1.4 rest of content")

    def test_not_pdf(self):
        assert not _is_pdf(b"<html>not a pdf</html>")

    def test_empty(self):
        assert not _is_pdf(b"")

    def test_short(self):
        assert not _is_pdf(b"%PD")


# ---------------------------------------------------------------------------
# download_from_arxiv (mocked)
# ---------------------------------------------------------------------------

class TestDownloadFromArxiv:
    def test_no_arxiv_id(self):
        result = download_from_arxiv("", "/tmp")
        assert not result.success
        assert "No arXiv ID" in result.error

    @patch("reticulate.librarian._download_pdf")
    def test_success(self, mock_dl):
        mock_dl.return_value = DownloadResult(
            key="2301.12345", title="", path="/tmp/2301.12345.pdf",
            success=True, source="arxiv", size_bytes=50000,
        )
        with tempfile.TemporaryDirectory() as d:
            result = download_from_arxiv("2301.12345", d)
            assert result.success
            mock_dl.assert_called_once()

    @patch("reticulate.librarian._download_pdf")
    def test_custom_filename(self, mock_dl):
        mock_dl.return_value = DownloadResult(
            key="mykey", title="", path="/tmp/mykey.pdf",
            success=True, source="arxiv", size_bytes=50000,
        )
        with tempfile.TemporaryDirectory() as d:
            result = download_from_arxiv("2301.12345", d, filename="mykey")
            assert result.key == "mykey"


# ---------------------------------------------------------------------------
# download_from_doi (mocked)
# ---------------------------------------------------------------------------

class TestDownloadFromDoi:
    def test_no_doi(self):
        result = download_from_doi("", "/tmp")
        assert not result.success
        assert "No DOI" in result.error

    @patch("reticulate.librarian._try_doi_direct")
    @patch("reticulate.librarian._try_semantic_scholar_pdf")
    @patch("reticulate.librarian._try_unpaywall")
    def test_unpaywall_success(self, mock_unp, mock_s2, mock_doi):
        mock_unp.return_value = DownloadResult(
            key="test", title="Test Paper", path="/tmp/test.pdf",
            success=True, source="unpaywall", size_bytes=30000,
        )
        with tempfile.TemporaryDirectory() as d:
            result = download_from_doi("10.1000/test", d, filename="test")
            assert result.success
            assert result.source == "unpaywall"
            mock_s2.assert_not_called()
            mock_doi.assert_not_called()

    @patch("reticulate.librarian._try_doi_direct")
    @patch("reticulate.librarian._try_semantic_scholar_pdf")
    @patch("reticulate.librarian._try_unpaywall")
    def test_all_fail(self, mock_unp, mock_s2, mock_doi):
        fail = DownloadResult(key="test", title="", path="", success=False, source="", error="fail")
        mock_unp.return_value = fail
        mock_s2.return_value = fail
        mock_doi.return_value = fail
        with tempfile.TemporaryDirectory() as d:
            result = download_from_doi("10.1000/test", d)
            assert not result.success
            assert "All download strategies failed" in result.error


# ---------------------------------------------------------------------------
# organize_references
# ---------------------------------------------------------------------------

class TestOrganizeReferences:
    def test_organize_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            results = organize_references(d)
            assert results == []

    def test_organize_with_pdfs(self):
        with tempfile.TemporaryDirectory() as d:
            # Create fake PDF files
            for name in ["honda1998.pdf", "gay2005.pdf", "notes.txt"]:
                with open(os.path.join(d, name), "wb") as f:
                    f.write(b"%PDF-1.4 fake content" if name.endswith(".pdf") else b"text")
            results = organize_references(d)
            assert len(results) == 2  # only .pdf files
            keys = {r.key for r in results}
            assert "honda1998" in keys
            assert "gay2005" in keys

    def test_organize_nonexistent_dir(self):
        results = organize_references("/nonexistent/dir/path")
        assert results == []


# ---------------------------------------------------------------------------
# check_missing
# ---------------------------------------------------------------------------

class TestCheckMissing:
    def test_all_missing(self):
        with tempfile.TemporaryDirectory() as d:
            bib_path = os.path.join(d, "refs.bib")
            with open(bib_path, "w") as f:
                f.write(SAMPLE_BIB)
            pdf_dir = os.path.join(d, "pdfs")
            os.makedirs(pdf_dir)

            missing = check_missing(bib_path, pdf_dir)
            assert len(missing) == 2
            assert all(not m.success for m in missing)

    def test_some_present(self):
        with tempfile.TemporaryDirectory() as d:
            bib_path = os.path.join(d, "refs.bib")
            with open(bib_path, "w") as f:
                f.write(SAMPLE_BIB)
            pdf_dir = os.path.join(d, "pdfs")
            os.makedirs(pdf_dir)

            # Create one PDF (large enough)
            with open(os.path.join(pdf_dir, "honda1998.pdf"), "wb") as f:
                f.write(b"%PDF-1.4 " + b"x" * 2000)

            missing = check_missing(bib_path, pdf_dir)
            assert len(missing) == 1
            assert missing[0].key == "gay2005subtyping"

    def test_none_missing(self):
        with tempfile.TemporaryDirectory() as d:
            bib_path = os.path.join(d, "refs.bib")
            with open(bib_path, "w") as f:
                f.write(SAMPLE_BIB)
            pdf_dir = os.path.join(d, "pdfs")
            os.makedirs(pdf_dir)

            for key in ["honda1998", "gay2005subtyping"]:
                with open(os.path.join(pdf_dir, f"{key}.pdf"), "wb") as f:
                    f.write(b"%PDF-1.4 " + b"x" * 2000)

            missing = check_missing(bib_path, pdf_dir)
            assert len(missing) == 0


# ---------------------------------------------------------------------------
# download_references (mocked)
# ---------------------------------------------------------------------------

class TestDownloadReferences:
    @patch("reticulate.librarian.download_from_doi")
    @patch("reticulate.librarian.download_from_arxiv")
    def test_skips_cached(self, mock_arxiv, mock_doi):
        with tempfile.TemporaryDirectory() as d:
            bib_path = os.path.join(d, "refs.bib")
            with open(bib_path, "w") as f:
                f.write(SAMPLE_BIB)

            # Pre-create a cached PDF
            with open(os.path.join(d, "honda1998.pdf"), "wb") as f:
                f.write(b"%PDF-1.4 " + b"x" * 2000)

            results = download_references(bib_path, d)
            # honda1998 should be cached, gay2005subtyping should be attempted
            cached = [r for r in results if r.source == "cached"]
            assert len(cached) == 1
            assert cached[0].key == "honda1998"
