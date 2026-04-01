"""Modular toolkit analysis on real-world protocols (Step 30ag).

Applies the modular lattice analysis toolkit — valuations, grading,
Jordan-Hölder dimension — to real-world protocols extracted in Steps
71b (REST APIs) and 97b (Python protocols).

Scientific Method:
  Observe   — Real protocols are mostly modular non-distributive (Step 29d)
  Question  — What does the modular toolkit reveal about real protocols?
  Hypothesis — Valuations and grading provide useful protocol invariants
  Predict   — Modular protocols have rank valuations; non-modular don't
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reticulate.importers import from_openapi
from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice, check_distributive, DistributivityResult
from reticulate.valuation import analyze_valuations, ValuationAnalysis
from reticulate.type_inference import Trace, infer_from_traces

from reticulate.real_world_extraction import REAL_WORLD_SPECS
from reticulate.source_extractor import PYTHON_PROTOCOLS


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolkitResult:
    """Result of applying modular toolkit to a single protocol."""
    name: str
    source: str  # "REST" or "Python"
    num_states: int
    num_transitions: int
    classification: str
    is_graded: bool
    rank_is_valuation: bool
    height_is_valuation: bool
    valuation_dimension: int
    rank_map: dict[int, int]
    height_map: dict[int, int]
    width: int  # max antichain size
    depth: int  # longest chain length


@dataclass(frozen=True)
class ToolkitReport:
    """Aggregate report across all protocols."""
    results: list[ToolkitResult]
    total: int
    graded_count: int
    rank_valuation_count: int
    graded_rate: float
    rank_valuation_rate: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _compute_width(ss: StateSpace) -> int:
    """Compute width (max antichain size) of a state space."""
    # Group states by rank, width = max group size
    from reticulate.valuation import rank_function
    ranks = rank_function(ss)
    if not ranks:
        return 0
    rank_groups: dict[int, int] = {}
    for s, r in ranks.items():
        rank_groups[r] = rank_groups.get(r, 0) + 1
    return max(rank_groups.values()) if rank_groups else 0


def _compute_depth(ss: StateSpace) -> int:
    """Compute depth (longest chain length) of a state space."""
    from reticulate.valuation import height_function
    heights = height_function(ss)
    return max(heights.values()) if heights else 0


def analyze_single(name: str, source: str, st_str: str) -> ToolkitResult:
    """Apply full modular toolkit to a single session type string."""
    ast = parse(st_str)
    ss = build_statespace(ast)
    dr = check_distributive(ss)
    va = analyze_valuations(ss)

    return ToolkitResult(
        name=name,
        source=source,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        classification=dr.classification,
        is_graded=va.is_graded,
        rank_is_valuation=va.rank_is_valuation,
        height_is_valuation=va.height_is_valuation,
        valuation_dimension=va.valuation_dimension_lower,
        rank_map=dict(va.rank),
        height_map=dict(va.height),
        width=_compute_width(ss),
        depth=_compute_depth(ss),
    )


def analyze_all() -> ToolkitReport:
    """Apply modular toolkit to all 20 real-world protocols."""
    results: list[ToolkitResult] = []

    # REST APIs
    for name, spec_fn in REAL_WORLD_SPECS.items():
        spec = spec_fn()
        types = from_openapi(spec)
        for tag, st_str in types.items():
            results.append(analyze_single(name, "REST", st_str))

    # Python protocols
    for name, spec in PYTHON_PROTOCOLS.items():
        traces = [Trace.from_labels(t) for t in spec["traces"]]
        inferred = infer_from_traces(traces)
        st_str = pretty(inferred)
        results.append(analyze_single(name, "Python", st_str))

    total = len(results)
    graded = sum(1 for r in results if r.is_graded)
    rv = sum(1 for r in results if r.rank_is_valuation)

    return ToolkitReport(
        results=results,
        total=total,
        graded_count=graded,
        rank_valuation_count=rv,
        graded_rate=graded / total if total > 0 else 0.0,
        rank_valuation_rate=rv / total if total > 0 else 0.0,
    )


def print_toolkit_report(report: ToolkitReport) -> str:
    """Format a toolkit analysis report."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  MODULAR TOOLKIT ANALYSIS ON REAL-WORLD PROTOCOLS")
    lines.append("=" * 90)
    lines.append(f"  Protocols: {report.total}")
    lines.append(f"  Graded: {report.graded_count}/{report.total} ({report.graded_rate:.0%})")
    lines.append(f"  Rank is valuation: {report.rank_valuation_count}/{report.total} "
                 f"({report.rank_valuation_rate:.0%})")
    lines.append("")

    lines.append(f"  {'Protocol':<22} {'Src':<6} {'St':>3} {'Class':<14} "
                 f"{'Graded':>7} {'RkVal':>6} {'Width':>6} {'Depth':>6} {'Dim':>4}")
    lines.append("  " + "-" * 82)

    for r in report.results:
        g = "YES" if r.is_graded else "NO"
        rv = "YES" if r.rank_is_valuation else "NO"
        lines.append(f"  {r.name:<22} {r.source:<6} {r.num_states:>3} {r.classification:<14} "
                     f"{g:>7} {rv:>6} {r.width:>6} {r.depth:>6} {r.valuation_dimension:>4}")

    # Summary by classification
    lines.append("")
    lines.append("  --- BY CLASSIFICATION ---")
    by_class: dict[str, list[ToolkitResult]] = {}
    for r in report.results:
        by_class.setdefault(r.classification, []).append(r)

    for cls, group in sorted(by_class.items()):
        n = len(group)
        graded = sum(1 for r in group if r.is_graded)
        rv = sum(1 for r in group if r.rank_is_valuation)
        avg_width = sum(r.width for r in group) / n
        avg_depth = sum(r.depth for r in group) / n
        lines.append(f"  {cls:<14}: {n:>2} protocols, graded={graded}/{n}, "
                     f"rank_val={rv}/{n}, avg_width={avg_width:.1f}, avg_depth={avg_depth:.1f}")

    lines.append("")
    lines.append("  --- KEY INSIGHT ---")
    lines.append("  Modular protocols: rank=valuation (measure theory applies)")
    lines.append("  Non-modular protocols: rank≠valuation (only meet/join available)")
    lines.append("  Graded protocols: all maximal chains have equal length (regularity)")
    lines.append("=" * 90)
    return "\n".join(lines)
