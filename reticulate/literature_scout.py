"""Literature scout: search academic APIs for related session type papers.

Searches arXiv, DBLP, and Semantic Scholar for papers matching keywords,
deduplicates results, and compares against existing .bib entries to find
new related work.

Uses only stdlib (urllib, xml.etree, json). No external dependencies.

Usage:
    from reticulate.literature_scout import search_arxiv, find_related_work
    papers = search_arxiv("session types lattice")
    report = find_related_work(["session types", "lattice theory"])
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperResult:
    """A paper found by searching an academic API."""
    title: str
    authors: list[str]
    year: int | None = None
    venue: str = ""
    abstract: str = ""
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    source: str = ""  # "arxiv" | "dblp" | "semantic_scholar"


@dataclass(frozen=True)
class RelatedWorkReport:
    """Report from a related-work search."""
    new_papers: list[PaperResult]
    already_cited: list[PaperResult]
    keywords_used: list[str]


# ---------------------------------------------------------------------------
# Title similarity for deduplication
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, strip punctuation, collapse whitespace."""
    t = title.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _titles_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check if two titles are similar enough to be considered the same paper.

    Uses Jaccard similarity on word sets.
    """
    wa = set(_normalize_title(a).split())
    wb = set(_normalize_title(b).split())
    if not wa or not wb:
        return False
    intersection = len(wa & wb)
    union = len(wa | wb)
    return (intersection / union) >= threshold


# ---------------------------------------------------------------------------
# arXiv search
# ---------------------------------------------------------------------------

_ARXIV_API = "https://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, max_results: int = 20) -> list[PaperResult]:
    """Search arXiv API for papers matching a query string.

    Args:
        query: Search keywords (e.g. "session types lattice").
        max_results: Maximum number of results to return (default 20).

    Returns:
        List of PaperResult with source="arxiv".
    """
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{_ARXIV_API}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reticulate/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError):
        return []

    return _parse_arxiv_response(data)


def _parse_arxiv_response(xml_data: bytes) -> list[PaperResult]:
    """Parse arXiv Atom XML response into PaperResult list."""
    results: list[PaperResult] = []
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return []

    for entry in root.findall("atom:entry", _ARXIV_NS):
        title_el = entry.find("atom:title", _ARXIV_NS)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        if not title:
            continue

        authors = []
        for author_el in entry.findall("atom:author", _ARXIV_NS):
            name_el = author_el.find("atom:name", _ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        abstract_el = entry.find("atom:summary", _ARXIV_NS)
        abstract = abstract_el.text.strip().replace("\n", " ") if abstract_el is not None and abstract_el.text else ""

        published_el = entry.find("atom:published", _ARXIV_NS)
        year = None
        if published_el is not None and published_el.text:
            m = re.match(r"(\d{4})", published_el.text)
            if m:
                year = int(m.group(1))

        # Extract arXiv ID and URL
        arxiv_id = ""
        paper_url = ""
        for link_el in entry.findall("atom:link", _ARXIV_NS):
            href = link_el.get("href", "")
            rel = link_el.get("rel", "")
            link_type = link_el.get("type", "")
            if rel == "alternate" or (not rel and "abs" in href):
                paper_url = href
            if link_type == "application/pdf":
                paper_url = paper_url or href

        id_el = entry.find("atom:id", _ARXIV_NS)
        if id_el is not None and id_el.text:
            # e.g. http://arxiv.org/abs/2301.12345v1
            id_text = id_el.text.strip()
            paper_url = paper_url or id_text
            m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", id_text)
            if m:
                arxiv_id = m.group(1)

        # Extract DOI if present
        doi = ""
        doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
        if doi_el is not None and doi_el.text:
            doi = doi_el.text.strip()

        results.append(PaperResult(
            title=title,
            authors=authors,
            year=year,
            venue="arXiv",
            abstract=abstract,
            url=paper_url,
            doi=doi,
            arxiv_id=arxiv_id,
            source="arxiv",
        ))

    return results


# ---------------------------------------------------------------------------
# DBLP search
# ---------------------------------------------------------------------------

_DBLP_API = "https://dblp.org/search/publ/api"


def search_dblp(query: str, max_results: int = 20) -> list[PaperResult]:
    """Search DBLP API for papers matching a query string.

    Args:
        query: Search keywords (e.g. "session types lattice").
        max_results: Maximum number of results to return (default 20).

    Returns:
        List of PaperResult with source="dblp".
    """
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "h": max_results,
    })
    url = f"{_DBLP_API}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reticulate/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []

    return _parse_dblp_response(data)


def _parse_dblp_response(data: dict) -> list[PaperResult]:
    """Parse DBLP JSON response into PaperResult list."""
    results: list[PaperResult] = []

    try:
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
    except (AttributeError, TypeError):
        return []

    if not isinstance(hits, list):
        return []

    for hit in hits:
        info = hit.get("info", {})
        if not isinstance(info, dict):
            continue

        title = info.get("title", "").strip().rstrip(".")
        if not title:
            continue

        # Authors can be a dict (single) or list
        authors_raw = info.get("authors", {}).get("author", [])
        if isinstance(authors_raw, dict):
            authors_raw = [authors_raw]
        authors = []
        for a in authors_raw:
            if isinstance(a, dict):
                authors.append(a.get("text", a.get("@text", "")))
            elif isinstance(a, str):
                authors.append(a)

        year_str = info.get("year", "")
        year = int(year_str) if year_str and str(year_str).isdigit() else None

        venue = info.get("venue", "")
        if isinstance(venue, list):
            venue = venue[0] if venue else ""

        paper_url = info.get("url", "")
        doi = info.get("doi", "")

        results.append(PaperResult(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract="",  # DBLP doesn't provide abstracts
            url=paper_url,
            doi=doi,
            arxiv_id="",
            source="dblp",
        ))

    return results


# ---------------------------------------------------------------------------
# Semantic Scholar search
# ---------------------------------------------------------------------------

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_semantic_scholar(query: str, max_results: int = 20) -> list[PaperResult]:
    """Search Semantic Scholar API for papers matching a query string.

    Args:
        query: Search keywords (e.g. "session types lattice").
        max_results: Maximum number of results to return (default 20).

    Returns:
        List of PaperResult with source="semantic_scholar".
    """
    params = urllib.parse.urlencode({
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,authors,year,venue,abstract,url,externalIds",
    })
    url = f"{_S2_API}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reticulate/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []

    return _parse_s2_response(data)


def _parse_s2_response(data: dict) -> list[PaperResult]:
    """Parse Semantic Scholar JSON response into PaperResult list."""
    results: list[PaperResult] = []

    papers = data.get("data", [])
    if not isinstance(papers, list):
        return []

    for paper in papers:
        if not isinstance(paper, dict):
            continue

        title = paper.get("title", "").strip()
        if not title:
            continue

        authors_raw = paper.get("authors", [])
        authors = []
        for a in authors_raw:
            if isinstance(a, dict):
                name = a.get("name", "")
                if name:
                    authors.append(name)

        year = paper.get("year")
        if year is not None and not isinstance(year, int):
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        venue = paper.get("venue", "") or ""
        abstract = paper.get("abstract", "") or ""
        paper_url = paper.get("url", "") or ""

        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "") or ""
        arxiv_id = ext_ids.get("ArXiv", "") or ""

        results.append(PaperResult(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract=abstract,
            url=paper_url,
            doi=doi,
            arxiv_id=arxiv_id,
            source="semantic_scholar",
        ))

    return results


# ---------------------------------------------------------------------------
# Multi-source search with deduplication
# ---------------------------------------------------------------------------

_SOURCE_FUNCTIONS = {
    "arxiv": search_arxiv,
    "dblp": search_dblp,
    "semantic_scholar": search_semantic_scholar,
}


def find_related_work(
    keywords: Sequence[str],
    sources: Sequence[str] = ("arxiv", "dblp"),
    max_per_source: int = 20,
) -> RelatedWorkReport:
    """Search multiple academic sources and deduplicate results.

    Args:
        keywords: List of search keyword strings.
        sources: Which APIs to query (default: arxiv, dblp).
        max_per_source: Max results per source per keyword.

    Returns:
        RelatedWorkReport with deduplicated new_papers.
    """
    all_papers: list[PaperResult] = []

    for kw in keywords:
        for source_name in sources:
            fn = _SOURCE_FUNCTIONS.get(source_name)
            if fn is None:
                continue
            papers = fn(kw, max_results=max_per_source)
            all_papers.extend(papers)

    # Deduplicate by title similarity
    unique: list[PaperResult] = []
    for paper in all_papers:
        is_dup = False
        for existing in unique:
            if _titles_similar(paper.title, existing.title):
                is_dup = True
                break
        if not is_dup:
            unique.append(paper)

    return RelatedWorkReport(
        new_papers=unique,
        already_cited=[],
        keywords_used=list(keywords),
    )


# ---------------------------------------------------------------------------
# .bib parsing (lightweight)
# ---------------------------------------------------------------------------

def _parse_bib_titles(bib_path: str) -> dict[str, str]:
    """Extract citation keys and titles from a .bib file.

    Returns: {citation_key: title}
    """
    entries: dict[str, str] = {}
    try:
        with open(bib_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, IOError):
        return {}

    # Match @type{key, ... title = {Title}, ...}
    # Simple regex-based parser for .bib files
    key_pattern = re.compile(r"@\w+\{(\w[\w\-]*)\s*,", re.IGNORECASE)
    title_pattern = re.compile(r"title\s*=\s*\{([^}]+)\}", re.IGNORECASE)

    current_key = None
    for line in content.split("\n"):
        key_match = key_pattern.search(line)
        if key_match:
            current_key = key_match.group(1)

        title_match = title_pattern.search(line)
        if title_match and current_key:
            entries[current_key] = title_match.group(1).strip()

    return entries


def check_for_new_work(
    bib_file: str,
    keywords: Sequence[str],
    sources: Sequence[str] = ("arxiv", "dblp"),
    max_per_source: int = 20,
) -> RelatedWorkReport:
    """Search for new work and compare against existing .bib entries.

    Args:
        bib_file: Path to a .bib file with existing citations.
        keywords: Search keyword strings.
        sources: Which APIs to query.
        max_per_source: Max results per source per keyword.

    Returns:
        RelatedWorkReport with new_papers (not in bib) and already_cited.
    """
    existing_titles = _parse_bib_titles(bib_file)

    report = find_related_work(keywords, sources=sources, max_per_source=max_per_source)

    new_papers: list[PaperResult] = []
    already_cited: list[PaperResult] = []

    for paper in report.new_papers:
        is_cited = False
        for _key, bib_title in existing_titles.items():
            if _titles_similar(paper.title, bib_title, threshold=0.80):
                is_cited = True
                break
        if is_cited:
            already_cited.append(paper)
        else:
            new_papers.append(paper)

    return RelatedWorkReport(
        new_papers=new_papers,
        already_cited=already_cited,
        keywords_used=list(keywords),
    )
