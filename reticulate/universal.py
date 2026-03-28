"""The Universal Session Type Repository (Step 100).

A universal repository of session types organized as an ontology, with
automatic composition for system design.  Every human process, natural
phenomenon, and abstract pattern can be expressed as a session type.
This module collects canonical examples across domains and provides:

- **Lookup and search** by name, domain, or tag.
- **Composition** of multiple session types via negotiation.
- **Analogy finding** — discover structural similarities across domains.
- **Repository statistics** and lattice verification.

The key insight is that session types are a universal modelling language:
the same algebraic structure that describes network protocols also
captures biological processes, legal procedures, musical forms, and
philosophical methods.  The repository makes these connections explicit.

This module provides:
    ``lookup(name)``               -- look up an entry by name.
    ``search_by_tag(tag)``         -- find all entries with a given tag.
    ``search_by_domain(domain)``   -- find all entries in a domain.
    ``compose(names)``             -- compose named types via negotiation.
    ``find_analogies(name)``       -- find structurally similar entries.
    ``domains()``                  -- sorted list of all domains.
    ``tags()``                     -- sorted list of all tags.
    ``repository_stats()``         -- summary statistics.
    ``all_form_lattices()``        -- verify every entry forms a lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.negotiation import compatibility_score, negotiate


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OntologyEntry:
    """A single entry in the universal session type repository.

    Attributes:
        name: Unique identifier for this entry.
        domain: Knowledge domain (e.g. "biology", "physics", "law").
        session_type_str: Session type in textual syntax.
        tags: Classification tags for cross-domain search.
        description: Short human-readable description.
    """

    name: str
    domain: str
    session_type_str: str
    tags: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class CompositionResult:
    """Result of composing multiple session types via negotiation.

    Attributes:
        components: Names of the composed entries.
        composed_type_str: Pretty-printed negotiated session type.
        state_count: Number of states in the composed state space.
        is_lattice: Whether the composed state space forms a lattice.
        compatibility_matrix: Pairwise compatibility scores.
    """

    components: tuple[str, ...]
    composed_type_str: str
    state_count: int
    is_lattice: bool
    compatibility_matrix: dict[tuple[str, str], float]


@dataclass(frozen=True)
class AnalogResult:
    """Result of an analogy search between two entries.

    Attributes:
        source: Name of the source entry.
        target: Name of the compared entry.
        similarity: Compatibility score (0.0 to 1.0).
        shared_structure: Description of the structural similarity.
    """

    source: str
    target: str
    similarity: float
    shared_structure: str


# ---------------------------------------------------------------------------
# The Universal Repository
# ---------------------------------------------------------------------------

UNIVERSAL_REPOSITORY: dict[str, OntologyEntry] = {
    # -- Nature --
    "photosynthesis": OntologyEntry(
        "photosynthesis", "biology",
        "&{absorb_light: +{split_water: &{fix_carbon: +{release_oxygen: end}}}}",
        ("biology", "energy", "transformation"),
        "Light energy to chemical energy",
    ),
    "cell_division": OntologyEntry(
        "cell_division", "biology",
        "&{interphase: +{prophase: &{metaphase: +{anaphase: &{telophase: end}}}}}",
        ("biology", "growth", "replication"),
        "Mitotic cell division",
    ),
    "water_cycle": OntologyEntry(
        "water_cycle", "earth_science",
        "rec X . &{evaporate: +{condense: &{precipitate: +{collect: X, drought: end}}}}",
        ("cycle", "nature", "water"),
        "Hydrological cycle",
    ),
    "predator_prey": OntologyEntry(
        "predator_prey", "ecology",
        "rec X . +{hunt: &{catch: +{consume: X, rest: end}, escape: X}}",
        ("ecology", "interaction", "survival"),
        "Predator-prey interaction",
    ),

    # -- Physics --
    "gravity": OntologyEntry(
        "gravity", "physics",
        "rec X . &{attract: +{orbit: X, escape: end, collapse: end}}",
        ("physics", "force", "fundamental"),
        "Gravitational interaction",
    ),
    "thermodynamics": OntologyEntry(
        "thermodynamics", "physics",
        "&{heat_transfer: +{work: &{entropy_increase: end}}}",
        ("physics", "energy", "irreversible"),
        "Heat engine cycle",
    ),
    "quantum_measurement": OntologyEntry(
        "quantum_measurement", "physics",
        "&{prepare: +{measure: &{collapse: +{eigenstate_1: end, eigenstate_2: end}}}}",
        ("physics", "quantum", "measurement"),
        "Quantum measurement process",
    ),

    # -- Human processes --
    "scientific_method": OntologyEntry(
        "scientific_method", "epistemology",
        "rec X . &{observe: +{hypothesize: &{experiment: +{confirm: end, refute: X}}}}",
        ("science", "method", "recursive"),
        "The scientific method",
    ),
    "democratic_vote": OntologyEntry(
        "democratic_vote", "governance",
        "&{nominate: +{campaign: &{vote: +{elect: end, runoff: end}}}}",
        ("governance", "democracy", "choice"),
        "Democratic election",
    ),
    "trial": OntologyEntry(
        "trial", "law",
        "&{charge: +{plead: &{evidence: +{deliberate: &{verdict: +{sentence: end, acquit: end}}}}}}",
        ("law", "justice", "adversarial"),
        "Criminal trial",
    ),
    "negotiation": OntologyEntry(
        "negotiation", "commerce",
        "rec X . +{offer: &{accept: end, reject: +{counter: X}, walk_away: end}}",
        ("commerce", "bargaining", "recursive"),
        "Price negotiation",
    ),

    # -- Emotions / Psychology --
    "grief": OntologyEntry(
        "grief", "psychology",
        "&{denial: +{anger: &{bargaining: +{depression: &{acceptance: end}}}}}",
        ("emotion", "loss", "stages"),
        "Kubler-Ross grief model",
    ),
    "trust_building": OntologyEntry(
        "trust_building", "psychology",
        "rec X . &{interact: +{positive: &{deepen: X}, negative: +{repair: X, withdraw: end}}}",
        ("emotion", "relationship", "recursive"),
        "Trust building cycle",
    ),
    "learning": OntologyEntry(
        "learning", "education",
        "rec X . &{encounter: +{struggle: &{insight: +{practice: &{mastery: end, plateau: X}}}}}",
        ("education", "growth", "recursive"),
        "Learning process",
    ),

    # -- Art --
    "sonata_form": OntologyEntry(
        "sonata_form", "music",
        "&{exposition: +{development: &{recapitulation: +{coda: end}}}}",
        ("music", "structure", "classical"),
        "Classical sonata form",
    ),
    "hero_journey": OntologyEntry(
        "hero_journey", "narrative",
        "&{call: +{refuse: &{mentor: +{threshold: &{trials: +{abyss: &{transformation: +{return: end}}}}}}}}",
        ("narrative", "archetype", "monomyth"),
        "Campbell's hero's journey",
    ),
    "haiku": OntologyEntry(
        "haiku", "poetry",
        "&{observe: +{juxtapose: &{reveal: end}}}",
        ("poetry", "minimal", "zen"),
        "Haiku structure",
    ),

    # -- Technology --
    "compile": OntologyEntry(
        "compile", "computing",
        "&{lex: +{parse: &{typecheck: +{optimize: &{codegen: end}}}}}",
        ("computing", "transformation", "pipeline"),
        "Compilation pipeline",
    ),
    "deploy": OntologyEntry(
        "deploy", "devops",
        "&{build: +{test: &{stage: +{approve: &{release: +{monitor: end}}}}}}",
        ("devops", "pipeline", "delivery"),
        "CI/CD pipeline",
    ),
    "search": OntologyEntry(
        "search", "computing",
        "rec X . &{query: +{results: &{select: end, refine: X}}}",
        ("computing", "information", "recursive"),
        "Search interaction",
    ),

    # -- Philosophy --
    "dialectic": OntologyEntry(
        "dialectic", "philosophy",
        "rec X . +{thesis: &{antithesis: +{synthesis: X, resolve: end}}}",
        ("philosophy", "reasoning", "recursive"),
        "Hegelian dialectic",
    ),
    "socratic_method": OntologyEntry(
        "socratic_method", "philosophy",
        "rec X . &{question: +{answer: &{challenge: +{refine: X, conclude: end}}}}",
        ("philosophy", "inquiry", "recursive"),
        "Socratic questioning",
    ),
}


# ---------------------------------------------------------------------------
# Lookup and search
# ---------------------------------------------------------------------------


def lookup(name: str) -> OntologyEntry:
    """Look up a repository entry by name.

    Args:
        name: The unique entry name.

    Returns:
        The matching OntologyEntry.

    Raises:
        KeyError: If no entry with that name exists.
    """
    if name not in UNIVERSAL_REPOSITORY:
        raise KeyError(f"Unknown entry: {name!r}")
    return UNIVERSAL_REPOSITORY[name]


def search_by_tag(tag: str) -> list[OntologyEntry]:
    """Return all entries whose tags include *tag*.

    Args:
        tag: The tag to search for.

    Returns:
        List of matching entries, sorted by name.
    """
    return sorted(
        [e for e in UNIVERSAL_REPOSITORY.values() if tag in e.tags],
        key=lambda e: e.name,
    )


def search_by_domain(domain: str) -> list[OntologyEntry]:
    """Return all entries in the given domain.

    Args:
        domain: The domain to filter by.

    Returns:
        List of matching entries, sorted by name.
    """
    return sorted(
        [e for e in UNIVERSAL_REPOSITORY.values() if e.domain == domain],
        key=lambda e: e.name,
    )


def domains() -> list[str]:
    """Return a sorted list of all unique domains in the repository."""
    return sorted({e.domain for e in UNIVERSAL_REPOSITORY.values()})


def tags() -> list[str]:
    """Return a sorted list of all unique tags in the repository."""
    all_tags: set[str] = set()
    for entry in UNIVERSAL_REPOSITORY.values():
        all_tags.update(entry.tags)
    return sorted(all_tags)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose(names: list[str]) -> CompositionResult:
    """Compose multiple named session types via pairwise negotiation.

    Parses each named type, computes pairwise compatibility scores,
    and iteratively negotiates to produce a composed type.

    Args:
        names: List of entry names to compose (at least 2).

    Returns:
        CompositionResult with the composed type and compatibility matrix.

    Raises:
        KeyError: If any name is not in the repository.
        ValueError: If fewer than 2 names are given.
    """
    if len(names) < 2:
        raise ValueError("Need at least 2 entries to compose")

    entries = [lookup(n) for n in names]
    parsed = [parse(e.session_type_str) for e in entries]

    # Build pairwise compatibility matrix
    matrix: dict[tuple[str, str], float] = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                score = compatibility_score(parsed[i], parsed[j])
                matrix[(n1, n2)] = score

    # Iterative negotiation
    composed = parsed[0]
    for t in parsed[1:]:
        composed = negotiate(composed, t)

    composed_str = pretty(composed)

    # Build state space and check lattice
    ss = build_statespace(composed)
    lr = check_lattice(ss)

    return CompositionResult(
        components=tuple(names),
        composed_type_str=composed_str,
        state_count=len(ss.states),
        is_lattice=lr.is_lattice,
        compatibility_matrix=matrix,
    )


# ---------------------------------------------------------------------------
# Analogy finding
# ---------------------------------------------------------------------------


def _describe_similarity(e1: OntologyEntry, e2: OntologyEntry, score: float) -> str:
    """Generate a human-readable description of structural similarity."""
    shared_tags = set(e1.tags) & set(e2.tags)
    parts: list[str] = []
    if shared_tags:
        parts.append(f"shared tags: {', '.join(sorted(shared_tags))}")
    if e1.domain == e2.domain:
        parts.append(f"same domain ({e1.domain})")
    if "recursive" in e1.tags and "recursive" in e2.tags:
        parts.append("both recursive")
    if score >= 0.5:
        parts.append("high structural overlap")
    elif score >= 0.3:
        parts.append("moderate structural overlap")
    return "; ".join(parts) if parts else "weak structural similarity"


def find_analogies(name: str, threshold: float = 0.3) -> list[AnalogResult]:
    """Find entries structurally similar to the given entry.

    Computes the compatibility_score between the named entry and every
    other entry, returning those above *threshold*, sorted by similarity
    descending.

    Args:
        name: The source entry name.
        threshold: Minimum compatibility score to include (default 0.3).

    Returns:
        List of AnalogResult sorted by similarity (descending).

    Raises:
        KeyError: If the name is not in the repository.
    """
    source = lookup(name)
    source_ast = parse(source.session_type_str)

    results: list[AnalogResult] = []
    for target_name, target in UNIVERSAL_REPOSITORY.items():
        if target_name == name:
            continue
        target_ast = parse(target.session_type_str)
        score = compatibility_score(source_ast, target_ast)
        if score >= threshold:
            desc = _describe_similarity(source, target, score)
            results.append(AnalogResult(
                source=name,
                target=target_name,
                similarity=score,
                shared_structure=desc,
            ))

    return sorted(results, key=lambda r: (-r.similarity, r.target))


# ---------------------------------------------------------------------------
# Statistics and verification
# ---------------------------------------------------------------------------


def repository_stats() -> dict[str, int]:
    """Return summary statistics about the repository.

    Returns:
        Dict with keys: total, domains, tags, recursive, non_recursive.
    """
    all_entries = list(UNIVERSAL_REPOSITORY.values())
    total = len(all_entries)
    n_domains = len(domains())
    n_tags = len(tags())
    recursive = sum(1 for e in all_entries if "rec " in e.session_type_str.lower()
                    or "rec " in e.session_type_str)
    non_recursive = total - recursive
    return {
        "total": total,
        "domains": n_domains,
        "tags": n_tags,
        "recursive": recursive,
        "non_recursive": non_recursive,
    }


def all_form_lattices() -> bool:
    """Verify that every entry in the repository forms a lattice.

    Parses each session type string, builds the state space, and checks
    the lattice property.

    Returns:
        True if and only if every entry's state space is a lattice.
    """
    for entry in UNIVERSAL_REPOSITORY.values():
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        if not result.is_lattice:
            return False
    return True
