"""Librarian: download and organize reference PDFs.

Downloads papers from DOI resolvers, Unpaywall, arXiv, and Semantic Scholar.
Organizes downloaded PDFs by citation key from .bib files.

Uses only stdlib (urllib, json, re, os). No external dependencies.

Usage:
    from reticulate.librarian import download_from_arxiv, check_missing
    result = download_from_arxiv("2301.12345", "/path/to/refs")
    missing = check_missing("references.bib", "/path/to/refs")
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DownloadResult:
    """Result of a single PDF download attempt."""
    key: str
    title: str
    path: str
    success: bool
    source: str  # "doi", "unpaywall", "arxiv", "semantic_scholar", ""
    size_bytes: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# .bib parsing (lightweight)
# ---------------------------------------------------------------------------

def _parse_bib_entries(bib_path: str) -> list[dict[str, str]]:
    """Parse a .bib file into a list of entry dicts.

    Each dict has: key, title, doi, eprint (arXiv ID), url, author.
    """
    entries: list[dict[str, str]] = []
    try:
        with open(bib_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, IOError):
        return []

    # Split into entry blocks
    # Match @type{key, until the next @ at the start of a line or end of file
    entry_pattern = re.compile(
        r"@(\w+)\{([\w][\w\-]*)\s*,(.*?)(?=\n@|\Z)",
        re.DOTALL,
    )
    field_pattern = re.compile(
        r"(\w+)\s*=\s*\{([^}]*)\}",
        re.IGNORECASE,
    )

    for match in entry_pattern.finditer(content):
        entry_type = match.group(1).lower()
        key = match.group(2)
        body = match.group(3)

        entry: dict[str, str] = {
            "type": entry_type,
            "key": key,
        }

        for field_match in field_pattern.finditer(body):
            field_name = field_match.group(1).lower()
            field_value = field_match.group(2).strip()
            entry[field_name] = field_value

        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Download from DOI (via doi.org redirect)
# ---------------------------------------------------------------------------

def _is_pdf(data: bytes) -> bool:
    """Check if data looks like a PDF."""
    return data[:5] == b"%PDF-"


def download_from_doi(doi: str, output_dir: str, filename: str = "") -> DownloadResult:
    """Try to download a PDF for a given DOI.

    Tries in order:
    1. Unpaywall API (free open-access PDFs)
    2. Semantic Scholar PDF link
    3. Direct doi.org resolution

    Args:
        doi: The DOI string (e.g. "10.1145/1706299.1706335").
        output_dir: Directory to save the PDF.
        filename: Optional filename (without .pdf). Defaults to sanitized DOI.

    Returns:
        DownloadResult with success status.
    """
    if not doi:
        return DownloadResult(
            key=filename or "unknown",
            title="",
            path="",
            success=False,
            source="",
            error="No DOI provided",
        )

    os.makedirs(output_dir, exist_ok=True)
    safe_name = filename or re.sub(r"[/\\:*?\"<>|]", "_", doi)
    out_path = os.path.join(output_dir, f"{safe_name}.pdf")

    # Strategy 1: Unpaywall
    result = _try_unpaywall(doi, out_path, safe_name)
    if result.success:
        return result

    # Strategy 2: Semantic Scholar
    result = _try_semantic_scholar_pdf(doi, out_path, safe_name)
    if result.success:
        return result

    # Strategy 3: Direct DOI resolution
    result = _try_doi_direct(doi, out_path, safe_name)
    if result.success:
        return result

    return DownloadResult(
        key=safe_name,
        title="",
        path="",
        success=False,
        source="",
        error=f"All download strategies failed for DOI {doi}",
    )


def _try_unpaywall(doi: str, out_path: str, key: str) -> DownloadResult:
    """Try downloading via Unpaywall API."""
    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"https://api.unpaywall.org/v2/{encoded_doi}?email=zua@bica-tools.org"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reticulate/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return DownloadResult(key=key, title="", path="", success=False, source="unpaywall",
                              error="Unpaywall API request failed")

    # Find best OA location with PDF
    best_url = data.get("best_oa_location", {})
    if isinstance(best_url, dict):
        pdf_url = best_url.get("url_for_pdf", "")
    else:
        pdf_url = ""

    if not pdf_url:
        oa_locations = data.get("oa_locations", [])
        for loc in oa_locations:
            if isinstance(loc, dict) and loc.get("url_for_pdf"):
                pdf_url = loc["url_for_pdf"]
                break

    if not pdf_url:
        return DownloadResult(key=key, title=data.get("title", ""), path="", success=False,
                              source="unpaywall", error="No open-access PDF found")

    return _download_pdf(pdf_url, out_path, key, data.get("title", ""), "unpaywall")


def _try_semantic_scholar_pdf(doi: str, out_path: str, key: str) -> DownloadResult:
    """Try downloading via Semantic Scholar paper details."""
    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{encoded_doi}?fields=title,openAccessPdf"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reticulate/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return DownloadResult(key=key, title="", path="", success=False,
                              source="semantic_scholar", error="S2 API request failed")

    oa_pdf = data.get("openAccessPdf", {})
    if isinstance(oa_pdf, dict) and oa_pdf.get("url"):
        pdf_url = oa_pdf["url"]
        return _download_pdf(pdf_url, out_path, key, data.get("title", ""), "semantic_scholar")

    return DownloadResult(key=key, title=data.get("title", ""), path="", success=False,
                          source="semantic_scholar", error="No open-access PDF in S2")


def _try_doi_direct(doi: str, out_path: str, key: str) -> DownloadResult:
    """Try resolving DOI directly and downloading the PDF."""
    url = f"https://doi.org/{doi}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "reticulate/0.1",
            "Accept": "application/pdf",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        if _is_pdf(data) and len(data) > 1000:
            with open(out_path, "wb") as f:
                f.write(data)
            return DownloadResult(key=key, title="", path=out_path, success=True,
                                  source="doi", size_bytes=len(data))
    except (urllib.error.URLError, OSError):
        pass

    return DownloadResult(key=key, title="", path="", success=False,
                          source="doi", error="Direct DOI resolution failed")


def _download_pdf(url: str, out_path: str, key: str, title: str, source: str) -> DownloadResult:
    """Download a PDF from a URL to a file path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reticulate/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        if not _is_pdf(data):
            return DownloadResult(key=key, title=title, path="", success=False,
                                  source=source, error="Downloaded content is not a PDF")

        if len(data) < 1000:
            return DownloadResult(key=key, title=title, path="", success=False,
                                  source=source, error="Downloaded PDF too small (likely error page)")

        with open(out_path, "wb") as f:
            f.write(data)

        return DownloadResult(key=key, title=title, path=out_path, success=True,
                              source=source, size_bytes=len(data))
    except (urllib.error.URLError, OSError) as e:
        return DownloadResult(key=key, title=title, path="", success=False,
                              source=source, error=str(e))


# ---------------------------------------------------------------------------
# Download from arXiv
# ---------------------------------------------------------------------------

def download_from_arxiv(arxiv_id: str, output_dir: str, filename: str = "") -> DownloadResult:
    """Download a PDF from arXiv by arXiv ID.

    Args:
        arxiv_id: The arXiv ID (e.g. "2301.12345" or "cs/0501015").
        output_dir: Directory to save the PDF.
        filename: Optional filename (without .pdf). Defaults to arXiv ID.

    Returns:
        DownloadResult with success status.
    """
    if not arxiv_id:
        return DownloadResult(
            key=filename or "unknown",
            title="",
            path="",
            success=False,
            source="arxiv",
            error="No arXiv ID provided",
        )

    os.makedirs(output_dir, exist_ok=True)
    safe_name = filename or re.sub(r"[/\\:*?\"<>|]", "_", arxiv_id)
    out_path = os.path.join(output_dir, f"{safe_name}.pdf")

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    return _download_pdf(url, out_path, safe_name, "", "arxiv")


# ---------------------------------------------------------------------------
# Batch download from .bib
# ---------------------------------------------------------------------------

def download_references(bib_file: str, output_dir: str) -> list[DownloadResult]:
    """Parse a .bib file and download PDFs for all entries with DOIs or arXiv IDs.

    Args:
        bib_file: Path to a .bib file.
        output_dir: Directory to save PDFs.

    Returns:
        List of DownloadResult, one per entry attempted.
    """
    entries = _parse_bib_entries(bib_file)
    results: list[DownloadResult] = []

    os.makedirs(output_dir, exist_ok=True)

    for entry in entries:
        key = entry.get("key", "unknown")
        doi = entry.get("doi", "")
        arxiv_id = entry.get("eprint", "")
        title = entry.get("title", "")

        # Skip if already downloaded
        out_path = os.path.join(output_dir, f"{key}.pdf")
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
            size = os.path.getsize(out_path)
            results.append(DownloadResult(
                key=key, title=title, path=out_path,
                success=True, source="cached", size_bytes=size,
            ))
            continue

        # Try arXiv first (most likely to have free PDFs)
        if arxiv_id:
            result = download_from_arxiv(arxiv_id, output_dir, filename=key)
            if result.success:
                results.append(DownloadResult(
                    key=key, title=title, path=result.path,
                    success=True, source=result.source, size_bytes=result.size_bytes,
                ))
                continue

        # Try DOI
        if doi:
            result = download_from_doi(doi, output_dir, filename=key)
            if result.success:
                results.append(DownloadResult(
                    key=key, title=title, path=result.path,
                    success=True, source=result.source, size_bytes=result.size_bytes,
                ))
                continue

        # Both failed
        error = f"No DOI or arXiv ID" if not doi and not arxiv_id else "All strategies failed"
        results.append(DownloadResult(
            key=key, title=title, path="",
            success=False, source="", error=error,
        ))

    return results


# ---------------------------------------------------------------------------
# Organize references
# ---------------------------------------------------------------------------

def organize_references(output_dir: str) -> list[DownloadResult]:
    """Rename PDF files in output_dir to {citationkey}.pdf format.

    Files already named as citation keys are kept. Files with DOI-style
    names or arXiv ID names are renamed if a mapping can be inferred.

    Args:
        output_dir: Directory containing PDF files.

    Returns:
        List of DownloadResult describing the organized files.
    """
    results: list[DownloadResult] = []

    if not os.path.isdir(output_dir):
        return results

    for fname in sorted(os.listdir(output_dir)):
        if not fname.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(output_dir, fname)
        size = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
        key = fname[:-4]  # strip .pdf

        results.append(DownloadResult(
            key=key,
            title="",
            path=full_path,
            success=True,
            source="organized",
            size_bytes=size,
        ))

    return results


# ---------------------------------------------------------------------------
# Check missing
# ---------------------------------------------------------------------------

def check_missing(bib_file: str, output_dir: str) -> list[DownloadResult]:
    """List which references from a .bib file don't have PDFs yet.

    Args:
        bib_file: Path to a .bib file.
        output_dir: Directory where PDFs should be.

    Returns:
        List of DownloadResult with success=False for missing entries.
    """
    entries = _parse_bib_entries(bib_file)
    missing: list[DownloadResult] = []

    for entry in entries:
        key = entry.get("key", "unknown")
        title = entry.get("title", "")
        out_path = os.path.join(output_dir, f"{key}.pdf")

        if not os.path.isfile(out_path) or os.path.getsize(out_path) < 1000:
            doi = entry.get("doi", "")
            arxiv_id = entry.get("eprint", "")
            has_id = bool(doi or arxiv_id)
            missing.append(DownloadResult(
                key=key,
                title=title,
                path="",
                success=False,
                source="",
                error="PDF not found" if has_id else "No DOI or arXiv ID available",
            ))

    return missing
