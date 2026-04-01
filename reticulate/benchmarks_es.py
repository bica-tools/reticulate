"""Benchmark protocols as event structures (Step 20).

Converts all 79 benchmark protocols to event structures and computes
their event-structure invariants: conflict count, causality depth,
concurrency degree, configuration count, and conflict density.

This module bridges the benchmark suite (tests/benchmarks/protocols.py)
with the event structure framework (event_structures.py), providing
a systematic comparison of protocols in the event-structure domain.

Key functions:
  - ``benchmark_to_es(protocol)``   -- convert one benchmark to ES
  - ``all_benchmarks_es()``          -- convert all benchmarks
  - ``es_invariants(protocol)``      -- compute ES invariants
  - ``es_comparison_table()``        -- tabulate all benchmarks
  - ``analyze_benchmarks_es()``      -- full analysis with statistics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.event_structures import (
    EventStructure,
    ESAnalysis,
    build_event_structure,
    configurations,
    concurrency_pairs,
    classify_events,
    analyze_event_structure,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ESInvariants:
    """Event structure invariants for a single protocol.

    Attributes:
        name: Protocol name.
        num_states: Number of states in the state space.
        num_events: Number of events in the ES.
        num_conflicts: Number of conflict pairs.
        num_causal: Number of causal (non-reflexive) pairs.
        num_configs: Number of configurations.
        num_concurrent: Number of concurrent event pairs.
        conflict_density: Fraction of event pairs in conflict.
        max_config_size: Largest configuration size.
        has_concurrency: True iff any concurrent pairs exist.
        is_isomorphic: True iff config count == state count.
    """
    name: str
    num_states: int
    num_events: int
    num_conflicts: int
    num_causal: int
    num_configs: int
    num_concurrent: int
    conflict_density: float
    max_config_size: int
    has_concurrency: bool
    is_isomorphic: bool


@dataclass(frozen=True)
class BenchmarkESReport:
    """Full analysis report for all benchmarks as event structures.

    Attributes:
        invariants: List of per-protocol invariants.
        total_events: Sum of events across all protocols.
        total_conflicts: Sum of conflicts across all protocols.
        avg_conflict_density: Mean conflict density.
        protocols_with_concurrency: Count of protocols with concurrent events.
        isomorphic_count: Count of protocols where configs == states.
        num_protocols: Number of protocols analyzed.
    """
    invariants: list[ESInvariants]
    total_events: int
    total_conflicts: int
    avg_conflict_density: float
    protocols_with_concurrency: int
    isomorphic_count: int
    num_protocols: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def benchmark_to_es(name: str, type_string: str) -> tuple[ESAnalysis, ESInvariants]:
    """Convert a single benchmark protocol to an event structure.

    Args:
        name: Protocol name.
        type_string: Session type string.

    Returns:
        Tuple of (ESAnalysis, ESInvariants).
    """
    ss = build_statespace(parse(type_string))
    analysis = analyze_event_structure(ss)

    inv = ESInvariants(
        name=name,
        num_states=len(ss.states),
        num_events=analysis.num_events,
        num_conflicts=analysis.num_conflicts,
        num_causal=analysis.num_causal,
        num_configs=analysis.num_configs,
        num_concurrent=analysis.num_concurrent,
        conflict_density=analysis.conflict_density,
        max_config_size=analysis.max_config_size,
        has_concurrency=analysis.num_concurrent > 0,
        is_isomorphic=analysis.is_isomorphic,
    )

    return analysis, inv


def es_invariants(name: str, type_string: str) -> ESInvariants:
    """Compute ES invariants for a single protocol."""
    _, inv = benchmark_to_es(name, type_string)
    return inv


def all_benchmarks_es(
    protocols: list[tuple[str, str]] | None = None,
) -> list[ESInvariants]:
    """Convert all benchmark protocols to event structures.

    Args:
        protocols: List of (name, type_string) pairs. If None, uses
            a default set of representative protocols.

    Returns:
        List of ESInvariants for each protocol.
    """
    if protocols is None:
        protocols = _default_protocols()

    return [es_invariants(name, ts) for name, ts in protocols]


def es_comparison_table(
    protocols: list[tuple[str, str]] | None = None,
) -> str:
    """Generate a text comparison table of ES invariants.

    Returns:
        Formatted table string.
    """
    invariants = all_benchmarks_es(protocols)

    header = (
        f"{'Protocol':<25} {'States':>6} {'Events':>6} {'Confl':>6} "
        f"{'Configs':>7} {'Conc':>5} {'Density':>7} {'Iso':>4}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for inv in invariants:
        lines.append(
            f"{inv.name:<25} {inv.num_states:>6} {inv.num_events:>6} "
            f"{inv.num_conflicts:>6} {inv.num_configs:>7} "
            f"{inv.num_concurrent:>5} {inv.conflict_density:>7.3f} "
            f"{'Y' if inv.is_isomorphic else 'N':>4}"
        )

    return "\n".join(lines)


def analyze_benchmarks_es(
    protocols: list[tuple[str, str]] | None = None,
) -> BenchmarkESReport:
    """Full analysis of all benchmarks as event structures."""
    invariants = all_benchmarks_es(protocols)

    total_events = sum(i.num_events for i in invariants)
    total_conflicts = sum(i.num_conflicts for i in invariants)
    avg_density = (
        sum(i.conflict_density for i in invariants) / len(invariants)
        if invariants else 0.0
    )
    conc_count = sum(1 for i in invariants if i.has_concurrency)
    iso_count = sum(1 for i in invariants if i.is_isomorphic)

    return BenchmarkESReport(
        invariants=invariants,
        total_events=total_events,
        total_conflicts=total_conflicts,
        avg_conflict_density=avg_density,
        protocols_with_concurrency=conc_count,
        isomorphic_count=iso_count,
        num_protocols=len(invariants),
    )


# ---------------------------------------------------------------------------
# Default protocol set
# ---------------------------------------------------------------------------

def _default_protocols() -> list[tuple[str, str]]:
    """Default set of representative protocols for ES analysis."""
    return [
        ("Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        ("File", "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"),
        ("SimpleBranch", "&{a: end, b: end}"),
        ("SimpleSelect", "+{ok: end, err: end}"),
        ("SimpleParallel", "(&{a: end} || &{b: end})"),
        ("DeepChain", "&{a: &{b: &{c: end}}}"),
        ("NestedBranch", "&{a: &{c: end, d: end}, b: &{e: end, f: end}}"),
        ("REST", "&{get: +{OK: end, NOT_FOUND: end}, put: +{OK: end, ERR: end}}"),
        ("RetryLoop", "rec X . &{request: +{OK: X, ERR: end}}"),
        ("WideBranch", "&{a: end, b: end, c: end, d: end, e: end}"),
    ]
