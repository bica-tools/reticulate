"""Venue matching agent: evaluates paper-venue fit using keyword analysis.

Extracts keywords from LaTeX papers (title, abstract, section headings)
and ranks academic venues by Jaccard similarity on scope keywords.

Usage:
    from reticulate.venue_matcher import rank_venues, VENUE_DATABASE
    results = rank_venues("papers/steps/step1-statespace/main.tex", top_n=5)
    for r in results:
        print(f"{r.venue.acronym}: {r.score:.2f} — {r.recommendation}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Venue:
    """An academic venue with scope metadata."""
    name: str
    acronym: str
    scope_keywords: tuple[str, ...]
    acceptance_rate: float   # 0.0–1.0
    tier: str                # "A*" | "A" | "B" | "C" | "workshop" | "journal"


@dataclass(frozen=True)
class VenueMatchResult:
    """Result of matching a paper against a venue."""
    venue: Venue
    score: float             # Jaccard similarity 0.0–1.0
    shared_keywords: tuple[str, ...]
    recommendation: str      # "strong", "good", "possible", "weak", "poor"


# ---------------------------------------------------------------------------
# Venue database (31 venues)
# ---------------------------------------------------------------------------

VENUE_DATABASE: dict[str, Venue] = {
    "CONCUR": Venue(
        name="International Conference on Concurrency Theory",
        acronym="CONCUR",
        scope_keywords=(
            "concurrency", "session types", "process algebra", "bisimulation",
            "pi-calculus", "lattice", "type theory", "verification",
            "communicating automata", "labelled transition system",
        ),
        acceptance_rate=0.30,
        tier="A",
    ),
    "ICE": Venue(
        name="Interaction and Concurrency Experience",
        acronym="ICE",
        scope_keywords=(
            "session types", "concurrency", "interaction", "protocol",
            "choreography", "multiparty", "behavioural types", "process calculi",
            "communication", "verification",
        ),
        acceptance_rate=0.50,
        tier="workshop",
    ),
    "PLACES": Venue(
        name="Workshop on Programming Language Approaches to Concurrency and Communication-cEntric Software",
        acronym="PLACES",
        scope_keywords=(
            "session types", "concurrency", "programming languages", "protocol",
            "communication", "behavioural types", "multiparty", "choreography",
            "type systems", "deadlock freedom",
        ),
        acceptance_rate=0.55,
        tier="workshop",
    ),
    "FORTE": Venue(
        name="International Conference on Formal Techniques for Distributed Objects, Components, and Systems",
        acronym="FORTE",
        scope_keywords=(
            "formal methods", "distributed systems", "verification", "protocol",
            "concurrency", "testing", "model checking", "specification",
            "conformance", "runtime verification",
        ),
        acceptance_rate=0.30,
        tier="A",
    ),
    "COORDINATION": Venue(
        name="International Conference on Coordination Models and Languages",
        acronym="COORDINATION",
        scope_keywords=(
            "coordination", "distributed systems", "concurrency", "protocol",
            "choreography", "service-oriented", "multiparty", "composition",
            "middleware", "interaction",
        ),
        acceptance_rate=0.33,
        tier="B",
    ),
    "TACAS": Venue(
        name="International Conference on Tools and Algorithms for the Construction and Analysis of Systems",
        acronym="TACAS",
        scope_keywords=(
            "tools", "verification", "model checking", "static analysis",
            "testing", "formal methods", "state space", "reachability",
            "lattice", "abstract interpretation",
        ),
        acceptance_rate=0.28,
        tier="A",
    ),
    "CAV": Venue(
        name="International Conference on Computer-Aided Verification",
        acronym="CAV",
        scope_keywords=(
            "verification", "model checking", "formal methods", "automated reasoning",
            "theorem proving", "satisfiability", "abstraction", "state space",
            "temporal logic", "safety",
        ),
        acceptance_rate=0.25,
        tier="A*",
    ),
    "LICS": Venue(
        name="IEEE Symposium on Logic in Computer Science",
        acronym="LICS",
        scope_keywords=(
            "logic", "type theory", "category theory", "semantics",
            "proof theory", "lambda calculus", "bisimulation", "fixpoint",
            "coinduction", "decidability",
        ),
        acceptance_rate=0.28,
        tier="A*",
    ),
    "FoSSaCS": Venue(
        name="International Conference on Foundations of Software Science and Computation Structures",
        acronym="FoSSaCS",
        scope_keywords=(
            "semantics", "type theory", "category theory", "process algebra",
            "automata", "coalgebra", "bisimulation", "rewriting",
            "denotational semantics", "operational semantics",
        ),
        acceptance_rate=0.30,
        tier="A",
    ),
    "ESOP": Venue(
        name="European Symposium on Programming",
        acronym="ESOP",
        scope_keywords=(
            "programming languages", "type systems", "semantics", "type theory",
            "static analysis", "subtyping", "polymorphism", "effect systems",
            "gradual typing", "verification",
        ),
        acceptance_rate=0.28,
        tier="A",
    ),
    "POPL": Venue(
        name="ACM SIGPLAN Symposium on Principles of Programming Languages",
        acronym="POPL",
        scope_keywords=(
            "programming languages", "type theory", "semantics", "verification",
            "type systems", "lambda calculus", "subtyping", "polymorphism",
            "abstract interpretation", "proof assistants",
        ),
        acceptance_rate=0.22,
        tier="A*",
    ),
    "OOPSLA": Venue(
        name="ACM SIGPLAN Conference on Object-Oriented Programming, Systems, Languages, and Applications",
        acronym="OOPSLA",
        scope_keywords=(
            "object-oriented", "programming languages", "type systems",
            "software engineering", "design patterns", "refactoring",
            "concurrency", "testing", "runtime", "annotations",
        ),
        acceptance_rate=0.28,
        tier="A*",
    ),
    "ECOOP": Venue(
        name="European Conference on Object-Oriented Programming",
        acronym="ECOOP",
        scope_keywords=(
            "object-oriented", "type systems", "programming languages",
            "concurrency", "ownership", "typestate", "gradual typing",
            "annotations", "design patterns", "behavioural types",
        ),
        acceptance_rate=0.25,
        tier="A",
    ),
    "ICFP": Venue(
        name="ACM SIGPLAN International Conference on Functional Programming",
        acronym="ICFP",
        scope_keywords=(
            "functional programming", "type theory", "lambda calculus",
            "type systems", "algebraic effects", "monads", "dependent types",
            "polymorphism", "category theory", "proof assistants",
        ),
        acceptance_rate=0.28,
        tier="A*",
    ),
    "APLAS": Venue(
        name="Asian Symposium on Programming Languages and Systems",
        acronym="APLAS",
        scope_keywords=(
            "programming languages", "type systems", "semantics",
            "static analysis", "verification", "session types",
            "subtyping", "effect systems", "concurrency", "formal methods",
        ),
        acceptance_rate=0.35,
        tier="B",
    ),
    "ICTAC": Venue(
        name="International Colloquium on Theoretical Aspects of Computing",
        acronym="ICTAC",
        scope_keywords=(
            "formal methods", "verification", "specification", "refinement",
            "process algebra", "concurrency", "semantics", "logic",
            "type theory", "model checking",
        ),
        acceptance_rate=0.35,
        tier="B",
    ),
    "CALCO": Venue(
        name="Conference on Algebra and Coalgebra in Computer Science",
        acronym="CALCO",
        scope_keywords=(
            "algebra", "coalgebra", "category theory", "bisimulation",
            "automata", "lattice", "universal algebra", "equational logic",
            "fixpoint", "semantics",
        ),
        acceptance_rate=0.40,
        tier="B",
    ),
    "AAMAS": Venue(
        name="International Conference on Autonomous Agents and Multi-Agent Systems",
        acronym="AAMAS",
        scope_keywords=(
            "agents", "multi-agent", "protocol", "interaction",
            "game theory", "negotiation", "coordination", "communication",
            "autonomous", "mechanism design",
        ),
        acceptance_rate=0.25,
        tier="A*",
    ),
    "BPM": Venue(
        name="International Conference on Business Process Management",
        acronym="BPM",
        scope_keywords=(
            "business process", "workflow", "choreography", "orchestration",
            "process mining", "protocol", "conformance", "compliance",
            "petri nets", "service composition",
        ),
        acceptance_rate=0.18,
        tier="A",
    ),
    "ISMIR": Venue(
        name="International Society for Music Information Retrieval Conference",
        acronym="ISMIR",
        scope_keywords=(
            "music", "audio", "signal processing", "retrieval",
            "classification", "machine learning", "melody", "rhythm",
            "transcription", "representation",
        ),
        acceptance_rate=0.30,
        tier="A",
    ),
    "CMSB": Venue(
        name="International Conference on Computational Methods in Systems Biology",
        acronym="CMSB",
        scope_keywords=(
            "systems biology", "biological networks", "stochastic",
            "petri nets", "simulation", "model checking", "pathway",
            "reaction networks", "differential equations", "verification",
        ),
        acceptance_rate=0.40,
        tier="B",
    ),
    "CCS": Venue(
        name="ACM Conference on Computer and Communications Security",
        acronym="CCS",
        scope_keywords=(
            "security", "cryptography", "protocol", "authentication",
            "privacy", "access control", "vulnerability", "malware",
            "network security", "formal verification",
        ),
        acceptance_rate=0.20,
        tier="A*",
    ),
    "VLDB": Venue(
        name="International Conference on Very Large Data Bases",
        acronym="VLDB",
        scope_keywords=(
            "database", "query", "transaction", "indexing",
            "distributed data", "data management", "SQL", "optimization",
            "storage", "scalability",
        ),
        acceptance_rate=0.22,
        tier="A*",
    ),
    "ITiCSE": Venue(
        name="ACM Conference on Innovation and Technology in Computer Science Education",
        acronym="ITiCSE",
        scope_keywords=(
            "education", "teaching", "programming", "curriculum",
            "assessment", "learning", "pedagogy", "tools",
            "visualization", "student",
        ),
        acceptance_rate=0.30,
        tier="B",
    ),
    "FC": Venue(
        name="Financial Cryptography and Data Security",
        acronym="FC",
        scope_keywords=(
            "cryptography", "blockchain", "smart contracts", "privacy",
            "financial", "protocol", "security", "consensus",
            "cryptocurrency", "payment",
        ),
        acceptance_rate=0.25,
        tier="B",
    ),
    "IoTDI": Venue(
        name="IEEE/ACM International Conference on Internet of Things Design and Implementation",
        acronym="IoTDI",
        scope_keywords=(
            "IoT", "embedded", "protocol", "sensor", "actuator",
            "edge computing", "middleware", "communication",
            "device management", "real-time",
        ),
        acceptance_rate=0.25,
        tier="B",
    ),
    "JLAMP": Venue(
        name="Journal of Logical and Algebraic Methods in Programming",
        acronym="JLAMP",
        scope_keywords=(
            "formal methods", "algebraic methods", "specification", "verification",
            "process algebra", "rewriting", "logic programming", "semantics",
            "type theory", "lattice",
        ),
        acceptance_rate=0.40,
        tier="journal",
    ),
    "LMCS": Venue(
        name="Logical Methods in Computer Science",
        acronym="LMCS",
        scope_keywords=(
            "logic", "type theory", "category theory", "semantics",
            "automata", "verification", "proof theory", "lambda calculus",
            "computability", "decidability",
        ),
        acceptance_rate=0.45,
        tier="journal",
    ),
    "TOPLAS": Venue(
        name="ACM Transactions on Programming Languages and Systems",
        acronym="TOPLAS",
        scope_keywords=(
            "programming languages", "type systems", "semantics", "compilers",
            "static analysis", "verification", "concurrency", "runtime systems",
            "memory management", "program analysis",
        ),
        acceptance_rate=0.20,
        tier="journal",
    ),
    "MSCS": Venue(
        name="Mathematical Structures in Computer Science",
        acronym="MSCS",
        scope_keywords=(
            "category theory", "algebra", "lattice", "logic",
            "semantics", "type theory", "coalgebra", "domain theory",
            "order theory", "topology",
        ),
        acceptance_rate=0.40,
        tier="journal",
    ),
    "SCP": Venue(
        name="Science of Computer Programming",
        acronym="SCP",
        scope_keywords=(
            "programming", "formal methods", "specification", "software engineering",
            "verification", "testing", "tools", "refactoring",
            "design patterns", "methodology",
        ),
        acceptance_rate=0.35,
        tier="journal",
    ),
}


# ---------------------------------------------------------------------------
# Keyword extraction from LaTeX
# ---------------------------------------------------------------------------

# Common LaTeX commands and stopwords to strip
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?")
_BRACE_RE = re.compile(r"[{}]")
_MATH_RE = re.compile(r"\$[^$]*\$")
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "is",
    "are", "was", "were", "be", "been", "being", "with", "by", "from",
    "at", "as", "this", "that", "these", "those", "it", "its", "we",
    "our", "their", "which", "how", "what", "when", "where", "not", "no",
    "all", "each", "every", "both", "such", "than", "into", "over",
    "also", "can", "may", "will", "shall", "has", "have", "had",
    "do", "does", "did", "but", "if", "then", "so", "yet", "nor",
    "about", "up", "down", "out", "off", "between", "through",
    "during", "before", "after", "above", "below", "under",
})


def _clean_latex(text: str) -> str:
    """Strip LaTeX commands, math, and braces from text."""
    text = _MATH_RE.sub(" ", text)
    text = _LATEX_CMD_RE.sub(" ", text)
    text = _BRACE_RE.sub(" ", text)
    text = re.sub(r"[~%\\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> set[str]:
    """Tokenize cleaned text into lowercase keyword tokens."""
    words = re.findall(r"[a-z][a-z-]*[a-z]|[a-z]", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def extract_paper_keywords(tex_path: str | Path) -> set[str]:
    """Extract keywords from a LaTeX paper's title, abstract, and section headings.

    Reads the .tex file and extracts text from:
    - \\title{...}
    - \\begin{abstract}...\\end{abstract}
    - \\section{...}, \\subsection{...}, \\subsubsection{...}

    Returns a set of lowercase keyword tokens.
    """
    path = Path(tex_path)
    if not path.exists():
        raise FileNotFoundError(f"TeX file not found: {tex_path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    keywords: set[str] = set()

    # Extract title
    title_match = re.search(r"\\title\{([^}]+)\}", content)
    if title_match:
        keywords |= _tokenize(_clean_latex(title_match.group(1)))

    # Extract abstract
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        content, re.DOTALL,
    )
    if abstract_match:
        keywords |= _tokenize(_clean_latex(abstract_match.group(1)))

    # Extract section headings
    for heading_match in re.finditer(
        r"\\(?:sub)*section\*?\{([^}]+)\}", content,
    ):
        keywords |= _tokenize(_clean_latex(heading_match.group(1)))

    # Extract explicit \keywords{...} if present
    kw_match = re.search(r"\\keywords\{([^}]+)\}", content)
    if kw_match:
        keywords |= _tokenize(_clean_latex(kw_match.group(1)))

    return keywords


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_score(paper_keywords: set[str], venue: Venue) -> float:
    """Compute Jaccard similarity between paper keywords and venue scope.

    Returns a float in [0.0, 1.0].
    """
    venue_kws = set(venue.scope_keywords)
    if not paper_keywords and not venue_kws:
        return 0.0
    intersection = paper_keywords & venue_kws
    union = paper_keywords | venue_kws
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _recommend(score: float) -> str:
    """Map a score to a recommendation label."""
    if score >= 0.25:
        return "strong"
    elif score >= 0.15:
        return "good"
    elif score >= 0.10:
        return "possible"
    elif score >= 0.05:
        return "weak"
    else:
        return "poor"


def rank_venues(
    tex_path: str | Path,
    top_n: int = 10,
    venues: dict[str, Venue] | None = None,
) -> list[VenueMatchResult]:
    """Rank venues by fit for a given LaTeX paper.

    Args:
        tex_path: Path to the .tex file.
        top_n: Maximum number of results to return.
        venues: Venue database to search (defaults to VENUE_DATABASE).

    Returns:
        List of VenueMatchResult sorted by score descending.
    """
    if venues is None:
        venues = VENUE_DATABASE

    paper_kws = extract_paper_keywords(tex_path)
    results: list[VenueMatchResult] = []

    for venue in venues.values():
        venue_kws = set(venue.scope_keywords)
        score = match_score(paper_kws, venue)
        shared = tuple(sorted(paper_kws & venue_kws))
        results.append(VenueMatchResult(
            venue=venue,
            score=score,
            shared_keywords=shared,
            recommendation=_recommend(score),
        ))

    results.sort(key=lambda r: (-r.score, r.venue.acronym))
    return results[:top_n]
