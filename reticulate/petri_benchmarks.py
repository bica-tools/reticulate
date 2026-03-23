"""Petri net benchmark analysis for all session type benchmarks (Step 25).

Runs Petri net construction and analysis on every benchmark protocol,
computing structural properties (places, transitions, free-choice,
occurrence net) and verifying the reachability isomorphism guarantee.

Key results expected:
  - All benchmarks produce valid, 1-safe Petri nets.
  - Reachability graphs are isomorphic to original state spaces.
  - Free-choice property holds for all state-machine-encoded nets.
  - Occurrence net property corresponds to acyclicity (no recursion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.petri import (
    PetriNetResult,
    build_petri_net,
    build_reachability_graph,
    conflict_places,
    session_type_to_petri_net,
)

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PetriBenchmarkResult:
    """Result of Petri net analysis for a single benchmark.

    Attributes:
        name: Benchmark protocol name.
        places: Number of places in the constructed net.
        transitions: Number of transitions in the constructed net.
        is_free_choice: True if the net satisfies the free-choice property.
        is_occurrence_net: True if the net is acyclic (no recursion).
        reachability_isomorphic: True if the reachability graph matches
                                 the original state space.
        reachable_markings: Number of reachable markings.
        ss_states: Number of states in the original state space.
        ss_transitions: Number of transitions in the original state space.
        num_conflict_places: Number of places with multiple consumers.
        is_one_safe: True if all reachable markings have at most 1 token
                     per place.
        uses_parallel: True if the benchmark uses the parallel constructor.
        has_recursion: True if the benchmark has recursive structure.
    """
    name: str
    places: int
    transitions: int
    is_free_choice: bool
    is_occurrence_net: bool
    reachability_isomorphic: bool
    reachable_markings: int
    ss_states: int
    ss_transitions: int
    num_conflict_places: int
    is_one_safe: bool
    uses_parallel: bool
    has_recursion: bool


@dataclass(frozen=True)
class PetriBenchmarkSummary:
    """Aggregate statistics across all benchmark Petri nets.

    Attributes:
        total_benchmarks: Number of benchmarks analysed.
        all_valid: True if all benchmarks produce valid Petri nets.
        all_one_safe: True if all nets are 1-safe.
        all_reachability_iso: True if all reachability graphs match.
        all_free_choice: True if all nets are free-choice.
        num_occurrence_nets: Number of acyclic (occurrence) nets.
        num_with_conflicts: Number of nets with conflict places.
        num_with_parallel: Number of benchmarks using parallel.
        num_with_recursion: Number of benchmarks with recursion.
        avg_places: Average number of places.
        avg_transitions: Average number of transitions.
        max_places: Maximum number of places across all benchmarks.
        max_transitions: Maximum transitions across all benchmarks.
        total_places: Total places across all benchmarks.
        total_transitions: Total transitions across all benchmarks.
        place_to_state_ratios: List of (name, ratio) pairs.
        transition_to_edge_ratios: List of (name, ratio) pairs.
    """
    total_benchmarks: int
    all_valid: bool
    all_one_safe: bool
    all_reachability_iso: bool
    all_free_choice: bool
    num_occurrence_nets: int
    num_with_conflicts: int
    num_with_parallel: int
    num_with_recursion: int
    avg_places: float
    avg_transitions: float
    max_places: int
    max_transitions: int
    total_places: int
    total_transitions: int
    place_to_state_ratios: list[tuple[str, float]]
    transition_to_edge_ratios: list[tuple[str, float]]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _check_one_safe(ss: "StateSpace", result: PetriNetResult) -> bool:
    """Check that every reachable marking has at most 1 token per place."""
    net = result.net
    rg = build_reachability_graph(net)
    for fm in rg.markings:
        for _pid, count in fm:
            if count > 1:
                return False
    return True


def analyze_benchmark(
    name: str,
    type_string: str,
    uses_parallel: bool = False,
) -> PetriBenchmarkResult:
    """Run Petri net analysis on a single benchmark protocol.

    Args:
        name: Human-readable protocol name.
        type_string: Session type string to parse.
        uses_parallel: Whether the protocol uses the parallel constructor.

    Returns:
        PetriBenchmarkResult with all computed properties.
    """
    ast = parse(type_string)
    ss = build_statespace(ast)
    petri_result = session_type_to_petri_net(ss)
    net = petri_result.net

    # Conflict analysis
    conflicts = conflict_places(net)
    num_conflicts = len(conflicts)

    # 1-safety check
    one_safe = _check_one_safe(ss, petri_result)

    # Recursion detection: occurrence net = acyclic = no recursion
    has_recursion = not petri_result.is_occurrence_net

    return PetriBenchmarkResult(
        name=name,
        places=petri_result.num_places,
        transitions=petri_result.num_transitions,
        is_free_choice=petri_result.is_free_choice,
        is_occurrence_net=petri_result.is_occurrence_net,
        reachability_isomorphic=petri_result.reachability_isomorphic,
        reachable_markings=petri_result.num_reachable_markings,
        ss_states=len(ss.states),
        ss_transitions=len(ss.transitions),
        num_conflict_places=num_conflicts,
        is_one_safe=one_safe,
        uses_parallel=uses_parallel,
        has_recursion=has_recursion,
    )


def analyze_all_benchmarks() -> list[PetriBenchmarkResult]:
    """Run Petri net analysis on all binary benchmark protocols.

    Returns:
        List of PetriBenchmarkResult, one per benchmark.
    """
    from tests.benchmarks.protocols import BENCHMARKS

    results: list[PetriBenchmarkResult] = []
    for bp in BENCHMARKS:
        result = analyze_benchmark(
            name=bp.name,
            type_string=bp.type_string,
            uses_parallel=bp.uses_parallel,
        )
        results.append(result)
    return results


def analyze_all_multiparty_benchmarks() -> list[PetriBenchmarkResult]:
    """Run Petri net analysis on all multiparty benchmark protocols.

    Multiparty benchmarks use global types; we build the global state space
    and then construct the Petri net from that.

    Returns:
        List of PetriBenchmarkResult, one per multiparty benchmark.
    """
    from reticulate.global_types import build_global_statespace, parse_global
    from tests.benchmarks.multiparty_protocols import MULTIPARTY_BENCHMARKS

    results: list[PetriBenchmarkResult] = []
    for mb in MULTIPARTY_BENCHMARKS:
        g = parse_global(mb.global_type_string)
        ss = build_global_statespace(g)
        petri_result = session_type_to_petri_net(ss)
        net = petri_result.net

        conflicts = conflict_places(net)
        one_safe = _check_one_safe(ss, petri_result)
        has_recursion = not petri_result.is_occurrence_net

        result = PetriBenchmarkResult(
            name=f"MP:{mb.name}",
            places=petri_result.num_places,
            transitions=petri_result.num_transitions,
            is_free_choice=petri_result.is_free_choice,
            is_occurrence_net=petri_result.is_occurrence_net,
            reachability_isomorphic=petri_result.reachability_isomorphic,
            reachable_markings=petri_result.num_reachable_markings,
            ss_states=len(ss.states),
            ss_transitions=len(ss.transitions),
            num_conflict_places=len(conflicts),
            is_one_safe=one_safe,
            uses_parallel=False,
            has_recursion=has_recursion,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def petri_benchmark_summary(
    results: list[PetriBenchmarkResult],
) -> PetriBenchmarkSummary:
    """Compute aggregate statistics from a list of benchmark results.

    Args:
        results: List of PetriBenchmarkResult from analyze_all_benchmarks().

    Returns:
        PetriBenchmarkSummary with aggregate statistics.
    """
    n = len(results)
    if n == 0:
        return PetriBenchmarkSummary(
            total_benchmarks=0,
            all_valid=True,
            all_one_safe=True,
            all_reachability_iso=True,
            all_free_choice=True,
            num_occurrence_nets=0,
            num_with_conflicts=0,
            num_with_parallel=0,
            num_with_recursion=0,
            avg_places=0.0,
            avg_transitions=0.0,
            max_places=0,
            max_transitions=0,
            total_places=0,
            total_transitions=0,
            place_to_state_ratios=[],
            transition_to_edge_ratios=[],
        )

    all_one_safe = all(r.is_one_safe for r in results)
    all_iso = all(r.reachability_isomorphic for r in results)
    all_fc = all(r.is_free_choice for r in results)

    total_places = sum(r.places for r in results)
    total_trans = sum(r.transitions for r in results)

    place_ratios = [
        (r.name, r.places / r.ss_states if r.ss_states > 0 else 0.0)
        for r in results
    ]
    trans_ratios = [
        (r.name, r.transitions / r.ss_transitions if r.ss_transitions > 0 else 0.0)
        for r in results
    ]

    return PetriBenchmarkSummary(
        total_benchmarks=n,
        all_valid=all_iso,
        all_one_safe=all_one_safe,
        all_reachability_iso=all_iso,
        all_free_choice=all_fc,
        num_occurrence_nets=sum(1 for r in results if r.is_occurrence_net),
        num_with_conflicts=sum(1 for r in results if r.num_conflict_places > 0),
        num_with_parallel=sum(1 for r in results if r.uses_parallel),
        num_with_recursion=sum(1 for r in results if r.has_recursion),
        avg_places=total_places / n,
        avg_transitions=total_trans / n,
        max_places=max(r.places for r in results),
        max_transitions=max(r.transitions for r in results),
        total_places=total_places,
        total_transitions=total_trans,
        place_to_state_ratios=place_ratios,
        transition_to_edge_ratios=trans_ratios,
    )


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_petri_benchmark_table(
    results: list[PetriBenchmarkResult],
) -> str:
    """Format benchmark results as a human-readable table.

    Args:
        results: List of PetriBenchmarkResult.

    Returns:
        Formatted string table with columns for each metric.
    """
    header = (
        f"{'#':>3} {'Name':<35} {'P':>4} {'T':>4} "
        f"{'SS':>4} {'ST':>4} {'FC':>3} {'ON':>3} "
        f"{'1S':>3} {'RI':>3} {'RM':>4} {'CP':>3} "
        f"{'Par':>3} {'Rec':>3}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for i, r in enumerate(results, 1):
        fc = "Y" if r.is_free_choice else "N"
        on = "Y" if r.is_occurrence_net else "N"
        os = "Y" if r.is_one_safe else "N"
        ri = "Y" if r.reachability_isomorphic else "N"
        par = "Y" if r.uses_parallel else ""
        rec = "Y" if r.has_recursion else ""
        name = r.name[:35]
        line = (
            f"{i:>3} {name:<35} {r.places:>4} {r.transitions:>4} "
            f"{r.ss_states:>4} {r.ss_transitions:>4} {fc:>3} {on:>3} "
            f"{os:>3} {ri:>3} {r.reachable_markings:>4} {r.num_conflict_places:>3} "
            f"{par:>3} {rec:>3}"
        )
        lines.append(line)

    lines.append(separator)
    lines.append(
        "Legend: P=Places, T=Transitions, SS=StateSpaceStates, "
        "ST=StateSpaceTransitions,"
    )
    lines.append(
        "  FC=FreeChoice, ON=OccurrenceNet, 1S=OneSafe, "
        "RI=ReachabilityIsomorphic,"
    )
    lines.append(
        "  RM=ReachableMarkings, CP=ConflictPlaces, "
        "Par=Parallel, Rec=Recursion"
    )
    return "\n".join(lines)


def format_summary(summary: PetriBenchmarkSummary) -> str:
    """Format a benchmark summary as human-readable text.

    Args:
        summary: PetriBenchmarkSummary to format.

    Returns:
        Multi-line summary string.
    """
    lines = [
        "Petri Net Benchmark Summary",
        "=" * 40,
        f"Total benchmarks:       {summary.total_benchmarks}",
        f"All valid (iso):        {summary.all_valid}",
        f"All 1-safe:             {summary.all_one_safe}",
        f"All free-choice:        {summary.all_free_choice}",
        f"Occurrence nets:        {summary.num_occurrence_nets}",
        f"With conflicts:         {summary.num_with_conflicts}",
        f"With parallel:          {summary.num_with_parallel}",
        f"With recursion:         {summary.num_with_recursion}",
        f"Avg places:             {summary.avg_places:.1f}",
        f"Avg transitions:        {summary.avg_transitions:.1f}",
        f"Max places:             {summary.max_places}",
        f"Max transitions:        {summary.max_transitions}",
        f"Total places:           {summary.total_places}",
        f"Total transitions:      {summary.total_transitions}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Petri net benchmark analysis and print results."""
    import sys

    print("Analysing binary benchmarks...", file=sys.stderr)
    binary_results = analyze_all_benchmarks()

    print("Analysing multiparty benchmarks...", file=sys.stderr)
    mp_results = analyze_all_multiparty_benchmarks()

    all_results = binary_results + mp_results

    print("\n=== Binary Benchmarks ===\n")
    print(format_petri_benchmark_table(binary_results))

    print("\n=== Multiparty Benchmarks ===\n")
    print(format_petri_benchmark_table(mp_results))

    summary = petri_benchmark_summary(all_results)
    print(f"\n{format_summary(summary)}")


if __name__ == "__main__":
    main()
