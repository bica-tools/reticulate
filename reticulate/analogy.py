"""Formal theory of analogy as session type morphism (Step 101).

Formalizes analogy as structural morphism between session type state spaces,
connecting Aristotle's analogical reasoning, Hofstadter's "analogy as core
of cognition," and category-theoretic functors.

The key insight: two processes are *analogous* to the degree that their
session type state spaces share structural properties.  The strength of
an analogy is classified by the strongest morphism that exists between
the state spaces:

    isomorphism > embedding > projection > homomorphism > weak_analogy > false_analogy

This module provides:
    ``analyze_analogy(ss1, ss2)``       -- core analogy analysis between two state spaces.
    ``explain_analogy(result)``         -- rich human-readable explanation.
    ``build_analogy_network(entries)``  -- pairwise analogy network with clustering.
    ``aristotelian_proportion(a, b, c, entries)`` -- A:B :: C:? proportional analogy.
    ``analogy_transitivity(a, b, c, entries)``    -- check transitive analogy.
    ``cross_domain_analogies(repository)``        -- find analogies across domains.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import parse, pretty
from reticulate.statespace import StateSpace, build_statespace
from reticulate.morphism import find_isomorphism, find_embedding
from reticulate.negotiation import compatibility_score
from reticulate.topology import topological_distance


# ---------------------------------------------------------------------------
# Strength constants
# ---------------------------------------------------------------------------

ISOMORPHISM: str = "isomorphism"
EMBEDDING: str = "embedding"
PROJECTION: str = "projection"
HOMOMORPHISM: str = "homomorphism"
WEAK_ANALOGY: str = "weak_analogy"
FALSE_ANALOGY: str = "false_analogy"

_STRENGTH_DESCRIPTIONS: dict[str, str] = {
    ISOMORPHISM: "perfect analogy -- identical interaction structure",
    EMBEDDING: "strong structural analogy -- one fully contains the other",
    PROJECTION: "substantial analogy -- similar size and significant overlap",
    HOMOMORPHISM: "moderate analogy -- meaningful structural overlap",
    WEAK_ANALOGY: "weak analogy -- some shared features but different structures",
    FALSE_ANALOGY: "no meaningful analogy -- superficial similarity only",
}

_STRENGTH_ORDER: dict[str, int] = {
    ISOMORPHISM: 6,
    EMBEDDING: 5,
    PROJECTION: 4,
    HOMOMORPHISM: 3,
    WEAK_ANALOGY: 2,
    FALSE_ANALOGY: 1,
}


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalogyResult:
    """Result of analyzing the analogy between two session type state spaces.

    Attributes:
        source_name: Name of the first entry.
        target_name: Name of the second entry.
        strength: One of the AnalogyStrength values.
        compatibility_score: Jaccard index from negotiation module.
        topological_distance: Distance from topology module.
        state_ratio: min(|S1|, |S2|) / max(|S1|, |S2|) -- size similarity.
        shared_labels: Method names in common between the two state spaces.
        morphism_type: From classify_morphism if a morphism exists, else "none".
        is_perfect: True iff strength is isomorphism.
        is_structural: True iff strength is embedding or better.
        explanation: Human-readable explanation of the analogy.
    """

    source_name: str
    target_name: str
    strength: str
    compatibility_score: float
    topological_distance: float
    state_ratio: float
    shared_labels: tuple[str, ...]
    morphism_type: str
    is_perfect: bool
    is_structural: bool
    explanation: str


@dataclass(frozen=True)
class AnalogyNetwork:
    """Network of pairwise analogies over a set of named state spaces.

    Attributes:
        entries: Names of all entries analyzed.
        edges: (source, target, strength, score) for each pair with score > 0.1.
        clusters: Groups of mutually analogous entries (connected components).
        strongest_analogy: (source, target, score) of the best pair.
        weakest_analogy: (source, target, score) of the weakest pair with score > 0.
    """

    entries: tuple[str, ...]
    edges: tuple[tuple[str, str, str, float], ...]
    clusters: tuple[frozenset[str], ...]
    strongest_analogy: tuple[str, str, float]
    weakest_analogy: tuple[str, str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transition_labels(ss: StateSpace) -> set[str]:
    """Return the set of all transition labels in a state space."""
    return {label for _, label, _ in ss.transitions}


def _compute_compatibility(ss1: StateSpace, ss2: StateSpace) -> float:
    """Compute compatibility score via Jaccard on transition labels.

    Falls back to label-set Jaccard when AST-level negotiation is not
    available (since we operate on state spaces, not ASTs).
    """
    labels1 = _transition_labels(ss1)
    labels2 = _transition_labels(ss2)
    if not labels1 and not labels2:
        return 1.0
    union = labels1 | labels2
    if not union:
        return 1.0
    return len(labels1 & labels2) / len(union)


def _classify_strength(
    iso_found: bool,
    emb_found: bool,
    state_ratio: float,
    compat: float,
) -> str:
    """Classify analogy strength from morphism results and scores."""
    if iso_found:
        return ISOMORPHISM
    if emb_found:
        return EMBEDDING
    if state_ratio > 0.8 and compat > 0.6:
        return PROJECTION
    if compat > 0.4:
        return HOMOMORPHISM
    if compat > 0.1:
        return WEAK_ANALOGY
    return FALSE_ANALOGY


def _generate_explanation(
    source_name: str,
    target_name: str,
    strength: str,
    shared_labels: tuple[str, ...],
    compat: float,
    state_ratio: float,
) -> str:
    """Generate a human-readable explanation of an analogy."""
    desc = _STRENGTH_DESCRIPTIONS.get(strength, "unknown analogy type")

    parts: list[str] = [
        f"{source_name} and {target_name}: {desc}.",
    ]

    if shared_labels:
        parts.append(
            f"Shared methods: {', '.join(shared_labels)}."
        )

    if strength == ISOMORPHISM:
        parts.append(
            "The state spaces are structurally identical "
            "-- a perfect one-to-one correspondence exists."
        )
    elif strength == EMBEDDING:
        parts.append(
            "The state space of one embeds entirely into the other, "
            "preserving all structural relationships."
        )
    elif strength == PROJECTION:
        parts.append(
            f"Similar sizes (ratio {state_ratio:.2f}) and significant "
            f"overlap (compatibility {compat:.2f})."
        )
    elif strength == HOMOMORPHISM:
        parts.append(
            f"Meaningful structural overlap (compatibility {compat:.2f}) "
            "despite different overall shapes."
        )
    elif strength == WEAK_ANALOGY:
        parts.append(
            "Only superficial structural similarities exist."
        )
    else:
        parts.append(
            "No meaningful structural correspondence found."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def analyze_analogy(
    ss1: StateSpace,
    ss2: StateSpace,
    name1: str = "S1",
    name2: str = "S2",
) -> AnalogyResult:
    """Analyze the analogy between two session type state spaces.

    Given two state spaces, classifies the strength of their structural
    analogy by attempting morphisms of decreasing strength and computing
    compatibility metrics.

    Args:
        ss1: First state space.
        ss2: Second state space.
        name1: Human-readable name for the first state space.
        name2: Human-readable name for the second state space.

    Returns:
        An AnalogyResult capturing the strength and details of the analogy.
    """
    # 1. Compatibility score (Jaccard on labels)
    compat = _compute_compatibility(ss1, ss2)

    # 2. Topological distance
    topo_dist = topological_distance(ss1, ss2)

    # 3. State ratio
    n1 = len(ss1.states)
    n2 = len(ss2.states)
    state_ratio = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 1.0

    # 4. Try isomorphism
    iso = find_isomorphism(ss1, ss2)
    iso_found = iso is not None

    # 5. Try embedding (if not iso)
    emb_found = False
    morphism_type = "none"
    if iso_found:
        morphism_type = "isomorphism"
    else:
        emb = find_embedding(ss1, ss2)
        if emb is not None:
            emb_found = True
            morphism_type = emb.kind
        else:
            # Try the other direction
            emb_rev = find_embedding(ss2, ss1)
            if emb_rev is not None:
                emb_found = True
                morphism_type = emb_rev.kind

    # 6. Shared labels
    labels1 = _transition_labels(ss1)
    labels2 = _transition_labels(ss2)
    shared = tuple(sorted(labels1 & labels2))

    # 7. Classify strength
    strength = _classify_strength(iso_found, emb_found, state_ratio, compat)

    # 8. Generate explanation
    explanation = _generate_explanation(
        name1, name2, strength, shared, compat, state_ratio,
    )

    return AnalogyResult(
        source_name=name1,
        target_name=name2,
        strength=strength,
        compatibility_score=compat,
        topological_distance=topo_dist,
        state_ratio=state_ratio,
        shared_labels=shared,
        morphism_type=morphism_type,
        is_perfect=(strength == ISOMORPHISM),
        is_structural=(strength in {ISOMORPHISM, EMBEDDING}),
        explanation=explanation,
    )


def explain_analogy(result: AnalogyResult) -> str:
    """Generate a rich human-readable explanation of an analogy result.

    Args:
        result: An AnalogyResult from analyze_analogy.

    Returns:
        A multi-sentence string explaining the analogy.
    """
    desc = _STRENGTH_DESCRIPTIONS.get(result.strength, "unknown analogy type")

    lines: list[str] = [
        f"Analogy between {result.source_name} and {result.target_name}:",
        f"  Strength: {result.strength} ({desc})",
        f"  Compatibility: {result.compatibility_score:.3f}",
        f"  Topological distance: {result.topological_distance:.3f}",
        f"  State size ratio: {result.state_ratio:.3f}",
    ]

    if result.shared_labels:
        lines.append(f"  Shared methods: {', '.join(result.shared_labels)}")
    else:
        lines.append("  Shared methods: (none)")

    if result.is_perfect:
        lines.append(
            "  Verdict: PERFECT analogy -- the two processes have "
            "identical interaction structure."
        )
    elif result.is_structural:
        lines.append(
            "  Verdict: STRONG structural analogy -- one process "
            "structurally contains the other."
        )
    elif result.strength == PROJECTION:
        lines.append(
            "  Verdict: SUBSTANTIAL analogy -- similar sizes with "
            "significant structural overlap."
        )
    elif result.strength == HOMOMORPHISM:
        lines.append(
            "  Verdict: MODERATE analogy -- meaningful overlap in "
            "interaction patterns."
        )
    elif result.strength == WEAK_ANALOGY:
        lines.append(
            "  Verdict: WEAK analogy -- some shared features but "
            "fundamentally different structures."
        )
    else:
        lines.append(
            "  Verdict: FALSE analogy -- no meaningful structural "
            "correspondence."
        )

    return "\n".join(lines)


def build_analogy_network(
    entries: dict[str, StateSpace],
) -> AnalogyNetwork:
    """Build a pairwise analogy network from named state spaces.

    Computes all-pairs analogies, filters to those with compatibility > 0.1,
    clusters entries into connected components of homomorphism-or-stronger
    analogies, and identifies the strongest and weakest pairs.

    Args:
        entries: Dictionary mapping names to state spaces.

    Returns:
        An AnalogyNetwork with edges, clusters, and extrema.
    """
    names = sorted(entries.keys())
    edges: list[tuple[str, str, str, float]] = []
    all_scores: list[tuple[str, str, float]] = []

    # Compute pairwise analogies (no self-comparisons)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            result = analyze_analogy(entries[n1], entries[n2], n1, n2)
            score = result.compatibility_score
            if score > 0.1:
                edges.append((n1, n2, result.strength, score))
            if score > 0.0:
                all_scores.append((n1, n2, score))

    # Cluster: connected components where edges have strength >= homomorphism
    adj: dict[str, set[str]] = {n: set() for n in names}
    for src, tgt, strength, _ in edges:
        if _STRENGTH_ORDER.get(strength, 0) >= _STRENGTH_ORDER[HOMOMORPHISM]:
            adj[src].add(tgt)
            adj[tgt].add(src)

    visited: set[str] = set()
    clusters: list[frozenset[str]] = []
    for n in names:
        if n not in visited:
            component: set[str] = set()
            stack = [n]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(adj[node] - visited)
            clusters.append(frozenset(component))

    # Sort edges by score descending
    edges.sort(key=lambda e: e[3], reverse=True)

    # Find strongest and weakest
    if all_scores:
        all_scores.sort(key=lambda x: x[2], reverse=True)
        strongest = (all_scores[0][0], all_scores[0][1], all_scores[0][2])
        weakest = (all_scores[-1][0], all_scores[-1][1], all_scores[-1][2])
    else:
        strongest = ("", "", 0.0)
        weakest = ("", "", 0.0)

    return AnalogyNetwork(
        entries=tuple(names),
        edges=tuple(edges),
        clusters=tuple(clusters),
        strongest_analogy=strongest,
        weakest_analogy=weakest,
    )


def aristotelian_proportion(
    a: str,
    b: str,
    c: str,
    entries: dict[str, StateSpace],
) -> tuple[str, float]:
    """Aristotle's proportional analogy: A is to B as C is to ?

    Finds the entry D such that the analogy between C and D is most
    similar to the analogy between A and B, measured by the absolute
    difference in compatibility scores.

    Args:
        a: Name of the first entry in the source pair.
        b: Name of the second entry in the source pair.
        c: Name of the first entry in the target pair.
        entries: Dictionary of named state spaces.

    Returns:
        (best_d_name, similarity_score) where similarity_score is the
        absolute difference in compatibility (lower = better match).

    Raises:
        KeyError: If a, b, or c are not in entries.
    """
    if a not in entries:
        raise KeyError(f"Unknown entry: {a!r}")
    if b not in entries:
        raise KeyError(f"Unknown entry: {b!r}")
    if c not in entries:
        raise KeyError(f"Unknown entry: {c!r}")

    # Compute the reference analogy A:B
    ref = analyze_analogy(entries[a], entries[b], a, b)
    ref_score = ref.compatibility_score

    best_d: str | None = None
    best_diff: float = float("inf")

    for name, ss in entries.items():
        if name == c:
            continue
        candidate = analyze_analogy(entries[c], ss, c, name)
        diff = abs(ref_score - candidate.compatibility_score)
        if diff < best_diff:
            best_diff = diff
            best_d = name

    if best_d is None:
        # Only c was in entries; return c itself as fallback
        return c, 0.0

    return best_d, best_diff


def analogy_transitivity(
    a: str,
    b: str,
    c: str,
    entries: dict[str, StateSpace],
) -> bool:
    """Check if analogy is transitive: if A~B and B~C, does A~C hold?

    Transitivity holds if compatibility(A, C) >= 0.5 * min(compat(A, B), compat(B, C)).

    Args:
        a: Name of the first entry.
        b: Name of the second (bridge) entry.
        c: Name of the third entry.
        entries: Dictionary of named state spaces.

    Returns:
        True if the transitive analogy condition holds.

    Raises:
        KeyError: If any name is not in entries.
    """
    if a not in entries:
        raise KeyError(f"Unknown entry: {a!r}")
    if b not in entries:
        raise KeyError(f"Unknown entry: {b!r}")
    if c not in entries:
        raise KeyError(f"Unknown entry: {c!r}")

    ab = analyze_analogy(entries[a], entries[b], a, b)
    bc = analyze_analogy(entries[b], entries[c], b, c)
    ac = analyze_analogy(entries[a], entries[c], a, c)

    threshold = min(ab.compatibility_score, bc.compatibility_score) * 0.5
    return ac.compatibility_score >= threshold


def cross_domain_analogies(
    repository: dict[str, OntologyEntry],
) -> list[AnalogyResult]:
    """Find analogies between entries in different domains.

    Parses each entry, builds its state space, and computes pairwise
    analogies only between entries from different domains.  Results are
    sorted by compatibility_score descending.

    Args:
        repository: Dictionary mapping names to OntologyEntry objects.

    Returns:
        List of AnalogyResult for cross-domain pairs, sorted descending.
    """
    # Build state spaces
    spaces: dict[str, StateSpace] = {}
    entry_domain: dict[str, str] = {}
    for name, entry in repository.items():
        try:
            ast = parse(entry.session_type_str)
            ss = build_statespace(ast)
            spaces[name] = ss
            entry_domain[name] = entry.domain
        except Exception:
            continue  # Skip entries that fail to parse/build

    # Compute cross-domain pairs
    names = sorted(spaces.keys())
    results: list[AnalogyResult] = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            if entry_domain[n1] == entry_domain[n2]:
                continue
            result = analyze_analogy(spaces[n1], spaces[n2], n1, n2)
            results.append(result)

    results.sort(key=lambda r: r.compatibility_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Import for type annotation
# ---------------------------------------------------------------------------

from reticulate.universal import OntologyEntry  # noqa: E402
