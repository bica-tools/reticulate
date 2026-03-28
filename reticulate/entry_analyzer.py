"""Automated research pipeline: comprehensive analysis for encyclopedia entries.

Takes a session type string and produces a complete multi-dimensional analysis
that can become a step paper.  Aggregates results from topology, information
theory, pendular classification, chaos dynamics, aesthetics, orbital analysis,
and analogy search into a single frozen dataclass.

This module provides:
    ``analyze_entry(name, type_str, ...)``    -- full analysis of one entry.
    ``analyze_all_entries(entries)``           -- batch analysis of many entries.
    ``generate_report(analysis)``             -- human-readable markdown report.
    ``generate_latex_section(analysis)``       -- LaTeX section for batch papers.
    ``compare_entries(a, b)``                 -- pairwise comparison dict.
    ``cluster_by_fingerprint(analyses)``      -- group by structural fingerprint.
    ``summary_statistics(analyses)``          -- aggregate statistics.
    ``analyze_full_encyclopedia()``           -- analyze all encyclopedia modules.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice
from reticulate.topology import classify_topology
from reticulate.information import analyze_information
from reticulate.pendular import classify_alternation
from reticulate.chaos import classify_dynamics
from reticulate.aesthetics import analyze_aesthetics
from reticulate.orbital import analyze_orbital, compute_orbits


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryAnalysis:
    """Complete multi-dimensional analysis of a session type entry.

    Attributes:
        name: Entry identifier.
        domain: Knowledge domain.
        session_type_str: Session type in textual syntax.
        description: Short human-readable description.
        state_count: Number of states in the state space.
        transition_count: Number of transitions in the state space.
        is_lattice: Whether the state space forms a lattice.
        euler_characteristic: V - E for the undirected underlying graph.
        betti_1: First Betti number (independent cycles).
        is_tree: True if acyclic.
        edge_density: Ratio of actual to maximal undirected edges.
        branching_entropy: Average Shannon entropy across non-terminal states.
        channel_capacity: log2(number of distinct top-to-bottom paths).
        is_deterministic: True if no state has duplicate outgoing labels.
        is_pendular: True if strictly pendular (every transition alternates).
        is_graded: True if all top-to-bottom paths have the same length.
        select_ratio: Fraction of non-end states that are Select.
        branch_ratio: Fraction of non-end states that are Branch.
        classification_alternation: Pendular classification string.
        binding_energy: Gravitational binding energy from planetary orbits.
        dynamics_classification: Chaos classification string.
        beauty_score: Aesthetic beauty score.
        aesthetic_classification: Aesthetic classification string.
        shell_count: Number of quantum orbital shells (BFS depth levels).
        symmetry_ratio: States / orbits ratio from group-theoretic analysis.
        orbital_classification: Orbital classification string.
        top_analogies: Top 3 analogies as (name, score) pairs.
        complexity_class: One of trivial, simple, moderate, complex, highly_complex.
        fingerprint: Hashable structural signature for quick comparison.
    """

    name: str
    domain: str
    session_type_str: str
    description: str
    # State space metrics
    state_count: int
    transition_count: int
    is_lattice: bool
    # Topology
    euler_characteristic: int
    betti_1: int
    is_tree: bool
    edge_density: float
    # Information
    branching_entropy: float
    channel_capacity: float
    is_deterministic: bool
    # Pendularity
    is_pendular: bool
    is_graded: bool
    select_ratio: float
    branch_ratio: float
    classification_alternation: str
    # Dynamics
    binding_energy: float
    dynamics_classification: str
    # Aesthetics
    beauty_score: float
    aesthetic_classification: str
    # Orbital
    shell_count: int
    symmetry_ratio: float
    orbital_classification: str
    # Cross-references
    top_analogies: tuple[tuple[str, float], ...]
    # Summary
    complexity_class: str
    fingerprint: tuple[int, ...]


# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------


def _classify_complexity(state_count: int) -> str:
    """Classify protocol complexity by state count."""
    if state_count <= 2:
        return "trivial"
    if state_count <= 5:
        return "simple"
    if state_count <= 10:
        return "moderate"
    if state_count <= 20:
        return "complex"
    return "highly_complex"


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------


def analyze_entry(
    name: str,
    session_type_str: str,
    domain: str = "",
    description: str = "",
    repository: dict[str, str] | None = None,
) -> EntryAnalysis:
    """Produce a complete multi-dimensional analysis of a session type.

    Args:
        name: Entry identifier.
        session_type_str: Session type in textual syntax.
        domain: Knowledge domain (optional).
        description: Short description (optional).
        repository: Optional dict of name -> session_type_str for analogy search.

    Returns:
        EntryAnalysis with all fields populated.
    """
    ast = parse(session_type_str)
    ss = build_statespace(ast)

    # Lattice
    lattice_result = check_lattice(ss)

    # Topology
    topo = classify_topology(ss)

    # Information
    info = analyze_information(ss)

    # Pendular
    pend = classify_alternation(ss)

    # Chaos / dynamics
    chaos = classify_dynamics(ss)

    # Aesthetics
    try:
        aes = analyze_aesthetics(ss)
        beauty = aes.beauty_score
        aes_class = aes.classification
    except Exception:
        beauty = 0.0
        aes_class = "unknown"

    # Orbital
    try:
        orb = analyze_orbital(ss)
        shell_count = orb.quantum.max_principal_quantum + 1
        symmetry_ratio = orb.group.symmetry_ratio
        orb_class = orb.classification
        binding = orb.planetary.binding_energy
    except Exception:
        shell_count = 0
        symmetry_ratio = 0.0
        orb_class = "unknown"
        binding = 0.0

    # Analogies
    top_analogies: list[tuple[str, float]] = []
    if repository:
        try:
            from reticulate.negotiation import compatibility_score

            scores: list[tuple[str, float]] = []
            for other_name, other_type_str in repository.items():
                if other_name == name:
                    continue
                try:
                    other_ast = parse(other_type_str)
                    score = compatibility_score(ast, other_ast)
                    scores.append((other_name, score))
                except Exception:
                    continue
            scores.sort(key=lambda x: x[1], reverse=True)
            top_analogies = scores[:3]
        except ImportError:
            pass

    # Complexity class
    complexity = _classify_complexity(len(ss.states))

    # Fingerprint
    fingerprint = (
        len(ss.states),
        len(ss.transitions),
        topo.euler_characteristic,
        topo.betti_1,
        int(pend.is_pendular),
        int(pend.is_graded),
    )

    return EntryAnalysis(
        name=name,
        domain=domain,
        session_type_str=session_type_str,
        description=description,
        state_count=len(ss.states),
        transition_count=len(ss.transitions),
        is_lattice=lattice_result.is_lattice,
        euler_characteristic=topo.euler_characteristic,
        betti_1=topo.betti_1,
        is_tree=topo.is_tree,
        edge_density=topo.edge_density,
        branching_entropy=info.branching_entropy,
        channel_capacity=info.channel_capacity,
        is_deterministic=info.is_deterministic,
        is_pendular=pend.is_pendular,
        is_graded=pend.is_graded,
        select_ratio=pend.select_ratio,
        branch_ratio=pend.branch_ratio,
        classification_alternation=pend.classification,
        binding_energy=binding,
        dynamics_classification=chaos.classification,
        beauty_score=beauty,
        aesthetic_classification=aes_class,
        shell_count=shell_count,
        symmetry_ratio=symmetry_ratio,
        orbital_classification=orb_class,
        top_analogies=tuple(top_analogies),
        complexity_class=complexity,
        fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------


def analyze_all_entries(
    entries: dict[str, tuple[str, str, str]],
) -> dict[str, EntryAnalysis]:
    """Batch analysis of multiple entries.

    Args:
        entries: Mapping of name -> (session_type_str, domain, description).

    Returns:
        Mapping of name -> EntryAnalysis.
    """
    # Build a flat repository for analogy search across all entries.
    repository: dict[str, str] = {
        name: tup[0] for name, tup in entries.items()
    }
    results: dict[str, EntryAnalysis] = {}
    for entry_name, (type_str, dom, desc) in entries.items():
        try:
            results[entry_name] = analyze_entry(
                entry_name, type_str, dom, desc, repository=repository,
            )
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(analysis: EntryAnalysis) -> str:
    """Generate a human-readable markdown report from an EntryAnalysis.

    Args:
        analysis: The analysis to report on.

    Returns:
        A markdown string.
    """
    a = analysis
    lines: list[str] = [
        f"# Analysis: {a.name}",
        f"**Domain**: {a.domain}" if a.domain else "",
        f"**Session Type**: `{a.session_type_str}`",
        f"**Description**: {a.description}" if a.description else "",
        "",
        "## State Space",
        f"- States: {a.state_count}, Transitions: {a.transition_count}",
        f"- Lattice: {a.is_lattice}",
        f"- Complexity: {a.complexity_class}",
        "",
        "## Topology",
        f"- Euler characteristic: chi = {a.euler_characteristic}",
        f"- Betti number: beta_1 = {a.betti_1}",
        f"- Tree: {a.is_tree}, Edge density: {a.edge_density:.4f}",
        "",
        "## Information Theory",
        f"- Branching entropy: {a.branching_entropy:.4f} bits",
        f"- Channel capacity: {a.channel_capacity:.4f} bits",
        f"- Deterministic: {a.is_deterministic}",
        "",
        "## Dynamics",
        f"- Pendular: {a.is_pendular}, Graded: {a.is_graded}",
        f"- Select ratio: {a.select_ratio:.4f}, Branch ratio: {a.branch_ratio:.4f}",
        f"- Alternation: {a.classification_alternation}",
        f"- Binding energy: {a.binding_energy:.4f}",
        f"- Classification: {a.dynamics_classification}",
        "",
        "## Aesthetics",
        f"- Beauty score: {a.beauty_score:.4f}",
        f"- Classification: {a.aesthetic_classification}",
        "",
        "## Orbital",
        f"- Shells: {a.shell_count}",
        f"- Symmetry ratio: {a.symmetry_ratio:.4f}",
        f"- Classification: {a.orbital_classification}",
        "",
        "## Analogies",
    ]
    if a.top_analogies:
        for other_name, score in a.top_analogies:
            lines.append(f"- {other_name}: {score:.4f}")
    else:
        lines.append("- (none)")
    lines += [
        "",
        "## Fingerprint",
        f"{a.fingerprint}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------


def generate_latex_section(analysis: EntryAnalysis) -> str:
    """Generate a LaTeX subsection from an EntryAnalysis.

    Args:
        analysis: The analysis to convert.

    Returns:
        A LaTeX string suitable for inclusion in a larger document.
    """
    a = analysis
    # Escape underscores and special chars for LaTeX
    safe_name = a.name.replace("_", r"\_")
    safe_type = a.session_type_str.replace("_", r"\_")
    safe_domain = a.domain.replace("_", r"\_") if a.domain else ""
    safe_desc = a.description.replace("_", r"\_") if a.description else ""

    lines: list[str] = [
        f"\\subsection{{{safe_name}}}",
    ]
    if safe_domain:
        lines.append(f"\\textit{{Domain: {safe_domain}}}")
        lines.append("")
    if safe_desc:
        lines.append(f"{safe_desc}")
        lines.append("")
    lines += [
        f"\\paragraph{{Session type.}} \\verb|{a.session_type_str}|",
        "",
        f"\\paragraph{{State space.}} {a.state_count} states, "
        f"{a.transition_count} transitions. "
        f"{'Lattice.' if a.is_lattice else 'Not a lattice.'} "
        f"Complexity: {a.complexity_class}.",
        "",
        f"\\paragraph{{Topology.}} Euler characteristic $\\chi = {a.euler_characteristic}$, "
        f"Betti number $\\beta_1 = {a.betti_1}$, "
        f"{'tree' if a.is_tree else 'cyclic'}, "
        f"edge density ${a.edge_density:.4f}$.",
        "",
        f"\\paragraph{{Information theory.}} Branching entropy "
        f"$H = {a.branching_entropy:.4f}$ bits, "
        f"channel capacity $C = {a.channel_capacity:.4f}$ bits, "
        f"{'deterministic' if a.is_deterministic else 'non-deterministic'}.",
        "",
        f"\\paragraph{{Dynamics.}} "
        f"{'Pendular' if a.is_pendular else 'Non-pendular'}, "
        f"{'graded' if a.is_graded else 'non-graded'}. "
        f"Alternation: {a.classification_alternation}. "
        f"Binding energy ${a.binding_energy:.4f}$. "
        f"Classification: {a.dynamics_classification}.",
        "",
        f"\\paragraph{{Aesthetics.}} Beauty score ${a.beauty_score:.4f}$, "
        f"classification: {a.aesthetic_classification}.",
        "",
        f"\\paragraph{{Orbital.}} {a.shell_count} shells, "
        f"symmetry ratio ${a.symmetry_ratio:.4f}$, "
        f"classification: {a.orbital_classification}.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_entries(a: EntryAnalysis, b: EntryAnalysis) -> dict[str, object]:
    """Compare two entry analyses and return a dict of comparisons.

    Args:
        a: First analysis.
        b: Second analysis.

    Returns:
        Dict with comparison results for each dimension.
    """
    return {
        "names": (a.name, b.name),
        "state_count": {"a": a.state_count, "b": b.state_count,
                        "larger": a.name if a.state_count >= b.state_count else b.name},
        "transition_count": {"a": a.transition_count, "b": b.transition_count,
                             "larger": a.name if a.transition_count >= b.transition_count else b.name},
        "is_lattice": {"a": a.is_lattice, "b": b.is_lattice},
        "euler_characteristic": {"a": a.euler_characteristic, "b": b.euler_characteristic},
        "betti_1": {"a": a.betti_1, "b": b.betti_1},
        "branching_entropy": {"a": a.branching_entropy, "b": b.branching_entropy,
                              "higher": a.name if a.branching_entropy >= b.branching_entropy else b.name},
        "channel_capacity": {"a": a.channel_capacity, "b": b.channel_capacity,
                             "higher": a.name if a.channel_capacity >= b.channel_capacity else b.name},
        "beauty_score": {"a": a.beauty_score, "b": b.beauty_score,
                         "more_beautiful": a.name if a.beauty_score >= b.beauty_score else b.name},
        "complexity_class": {"a": a.complexity_class, "b": b.complexity_class},
        "dynamics_classification": {"a": a.dynamics_classification,
                                    "b": b.dynamics_classification},
        "same_fingerprint": a.fingerprint == b.fingerprint,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_by_fingerprint(
    analyses: dict[str, EntryAnalysis],
) -> dict[tuple[int, ...], list[str]]:
    """Group entries by their structural fingerprint.

    Entries with the same fingerprint are structurally identical and are
    candidates for isomorphism checking.

    Args:
        analyses: Mapping of name -> EntryAnalysis.

    Returns:
        Mapping of fingerprint -> list of entry names sharing that fingerprint.
    """
    clusters: dict[tuple[int, ...], list[str]] = {}
    for entry_name, analysis in analyses.items():
        fp = analysis.fingerprint
        if fp not in clusters:
            clusters[fp] = []
        clusters[fp].append(entry_name)
    return clusters


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summary_statistics(analyses: dict[str, EntryAnalysis]) -> dict[str, object]:
    """Compute aggregate statistics across all analyses.

    Args:
        analyses: Mapping of name -> EntryAnalysis.

    Returns:
        Dict with aggregate stats.
    """
    if not analyses:
        return {
            "total_entries": 0,
            "lattice_fraction": 0.0,
            "avg_states": 0.0,
            "avg_entropy": 0.0,
            "complexity_distribution": {},
            "dynamics_distribution": {},
            "most_common_fingerprint": None,
        }

    all_analyses = list(analyses.values())
    n = len(all_analyses)

    lattice_count = sum(1 for a in all_analyses if a.is_lattice)
    avg_states = sum(a.state_count for a in all_analyses) / n
    avg_entropy = sum(a.branching_entropy for a in all_analyses) / n

    complexity_dist = Counter(a.complexity_class for a in all_analyses)
    dynamics_dist = Counter(a.dynamics_classification for a in all_analyses)
    fp_counter = Counter(a.fingerprint for a in all_analyses)
    most_common_fp = fp_counter.most_common(1)[0][0] if fp_counter else None

    return {
        "total_entries": n,
        "lattice_fraction": lattice_count / n,
        "avg_states": avg_states,
        "avg_entropy": avg_entropy,
        "complexity_distribution": dict(complexity_dist),
        "dynamics_distribution": dict(dynamics_dist),
        "most_common_fingerprint": most_common_fp,
    }


# ---------------------------------------------------------------------------
# Full encyclopedia analysis
# ---------------------------------------------------------------------------


def _collect_encyclopedia_entries() -> dict[str, tuple[str, str, str]]:
    """Import all encyclopedia modules and collect entries.

    Returns:
        Dict of name -> (session_type_str, domain, description).
    """
    entries: dict[str, tuple[str, str, str]] = {}

    # Universal repository
    try:
        from reticulate.universal import UNIVERSAL_REPOSITORY
        for name, entry in UNIVERSAL_REPOSITORY.items():
            entries[name] = (entry.session_type_str, entry.domain, entry.description)
    except ImportError:
        pass

    # Professional library (maps name -> session_type_str only)
    try:
        from reticulate.professional import PROFESSION_LIBRARY
        for name, type_str in PROFESSION_LIBRARY.items():
            entries[f"prof:{name}"] = (type_str, "professional", "")
    except ImportError:
        pass

    # Mythology / archetypes
    try:
        from reticulate.mythology import ARCHETYPE_LIBRARY
        for name, arch in ARCHETYPE_LIBRARY.items():
            entries[f"myth:{name}"] = (
                arch.session_type_str, arch.domain, arch.description,
            )
    except ImportError:
        pass

    # Physics
    try:
        from reticulate.physics_types import PHYSICS_LIBRARY
        for name, entry in PHYSICS_LIBRARY.items():
            entries[f"phys:{name}"] = (
                entry.session_type_str, entry.domain, entry.description,
            )
    except ImportError:
        pass

    # Mathematics
    try:
        from reticulate.mathematics_types import MATH_LIBRARY
        for name, entry in MATH_LIBRARY.items():
            entries[f"math:{name}"] = (
                entry.session_type_str, entry.domain, entry.description,
            )
    except ImportError:
        pass

    # Nature encyclopedia
    try:
        from reticulate.encyclopedia_nature import NATURE_ENCYCLOPEDIA
        for name, entry in NATURE_ENCYCLOPEDIA.items():
            entries[f"nature:{name}"] = (
                entry.session_type_str, entry.domain, entry.description,
            )
    except ImportError:
        pass

    # Human encyclopedia
    try:
        from reticulate.encyclopedia_human import HUMAN_ENCYCLOPEDIA
        for name, entry in HUMAN_ENCYCLOPEDIA.items():
            entries[f"human:{name}"] = (
                entry.session_type_str, entry.domain, entry.description,
            )
    except ImportError:
        pass

    # Civilization encyclopedia
    try:
        from reticulate.encyclopedia_civilization import CIVILIZATION_ENCYCLOPEDIA
        for name, entry in CIVILIZATION_ENCYCLOPEDIA.items():
            entries[f"civ:{name}"] = (
                entry.session_type_str, entry.domain, entry.description,
            )
    except ImportError:
        pass

    return entries


def analyze_full_encyclopedia() -> dict[str, EntryAnalysis]:
    """Analyze every entry across all encyclopedia modules.

    Imports all encyclopedia modules, collects entries, and runs
    analyze_entry on each.  Import errors are handled gracefully.

    Returns:
        Dict of name -> EntryAnalysis for all successfully analyzed entries.
    """
    entries = _collect_encyclopedia_entries()
    return analyze_all_entries(entries)
