"""DoS resistance analysis via lattice width and tropical algebra (Step 89f).

Analyses session type protocols for denial-of-service (DoS) vulnerability
by combining lattice-theoretic width analysis with tropical (max-plus)
spectral theory.  The key insight is that DoS attacks exploit protocol
states where the server must allocate resources proportional to the
attacker's choices, and these states correspond to high-width ranks and
high-branching-factor nodes in the state space.

Five analyses are combined:

1. **DoS surface**: States with high branching factor (out-degree) that an
   attacker can exploit to force resource allocation.
2. **Width exposure**: Lattice width at each rank measures parallelism
   exposure — wider ranks mean more concurrent resources an attacker can
   force the server to hold simultaneously.
3. **Throughput bottleneck**: The tropical eigenvalue (maximum cycle mean)
   identifies the rate-limiting transition in cyclic protocols — the
   bottleneck an attacker can target for amplification.
4. **Kemeny vulnerability**: The Kemeny constant measures expected random
   walk convergence time; low values indicate the protocol can be quickly
   exhausted by random probing.
5. **Combined DoS score**: A composite vulnerability metric.

Key results:
  - High DoS surface + low Kemeny = high vulnerability.
  - Tropical eigenvalue = 1 in cyclic protocols = amplification risk.
  - Width exposure directly bounds server memory under attack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoSResult:
    """Complete DoS resistance analysis result.

    Attributes:
        dos_surface: States with high branching factor (DoS targets).
        dos_surface_score: Fraction of states that are DoS targets.
        width_exposure: Width at each rank (parallelism exposure).
        max_width: Maximum width across all ranks.
        max_width_rank: Rank at which maximum width occurs.
        throughput_bottleneck: Tropical eigenvalue (rate-limiting cycle mean).
        has_amplification: True if tropical eigenvalue > 0 (cyclic protocol).
        kemeny_value: Kemeny constant (convergence time).
        kemeny_vulnerable: True if Kemeny constant is below threshold.
        branching_factors: Out-degree of each state.
        max_branching: Maximum out-degree.
        critical_states: States at intersection of high branching + high rank width.
        vulnerability_score: Composite DoS vulnerability score (0.0 = safe, 1.0 = vulnerable).
        num_states: Number of states in the state space.
    """
    dos_surface: list[int]
    dos_surface_score: float
    width_exposure: list[int]
    max_width: int
    max_width_rank: int
    throughput_bottleneck: float
    has_amplification: bool
    kemeny_value: float
    kemeny_vulnerable: bool
    branching_factors: dict[int, int]
    max_branching: int
    critical_states: list[int]
    vulnerability_score: float
    num_states: int


# ---------------------------------------------------------------------------
# Branching factor computation
# ---------------------------------------------------------------------------

def _branching_factors(ss: "StateSpace") -> dict[int, int]:
    """Compute out-degree (branching factor) for each state."""
    bf: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, _ in ss.transitions:
        bf[src] += 1
    return bf


# ---------------------------------------------------------------------------
# DoS surface: high-branching states
# ---------------------------------------------------------------------------

def dos_surface(ss: "StateSpace", threshold: int = 2) -> list[int]:
    """Identify states with high branching factor (DoS attack targets).

    A state with branching factor >= threshold allows an attacker to force
    the server to handle multiple cases, allocating resources for each.

    Args:
        ss: State space to analyse.
        threshold: Minimum out-degree to be considered a DoS target.

    Returns:
        Sorted list of state IDs with branching factor >= threshold.
    """
    bf = _branching_factors(ss)
    surface = [s for s, deg in bf.items() if deg >= threshold]
    surface.sort()
    return surface


# ---------------------------------------------------------------------------
# Width exposure: lattice width at each rank
# ---------------------------------------------------------------------------

def width_exposure(ss: "StateSpace") -> list[int]:
    """Compute lattice width at each BFS rank from the top.

    Width at rank r = number of states at BFS distance r from the top.
    Higher widths indicate more parallelism exposure — an attacker can
    force the server to maintain more concurrent state at that depth.

    Returns:
        List of widths, indexed by rank.
    """
    from reticulate.resource_leak import width_profile as _wp
    wp = _wp(ss)
    return wp.widths


# ---------------------------------------------------------------------------
# Throughput bottleneck: tropical eigenvalue
# ---------------------------------------------------------------------------

def throughput_bottleneck(ss: "StateSpace") -> float:
    """Identify the throughput bottleneck via tropical eigenvalue.

    The tropical eigenvalue (maximum cycle mean) of the state-space
    adjacency matrix identifies the rate-limiting cycle.  For unit-weight
    edges, this is 1.0 if cycles exist (amplification possible) and 0.0
    if the protocol is acyclic (no amplification).

    An attacker can exploit cycles to generate unbounded load by
    repeatedly traversing the bottleneck cycle.

    Returns:
        Tropical eigenvalue (max cycle mean). 0.0 for acyclic protocols.
    """
    from reticulate.tropical import tropical_eigenvalue
    return tropical_eigenvalue(ss)


# ---------------------------------------------------------------------------
# Kemeny vulnerability
# ---------------------------------------------------------------------------

def kemeny_vulnerability(ss: "StateSpace", threshold: float = 5.0) -> tuple[float, bool]:
    """Assess vulnerability via Kemeny constant.

    The Kemeny constant measures the expected number of steps for a random
    walk on the state-space graph to reach its stationary distribution.
    A low Kemeny constant means the protocol's state space can be
    ``exhausted'' quickly by random probing — an attacker doesn't need
    to know the protocol structure to cause damage.

    Args:
        ss: State space to analyse.
        threshold: Kemeny values below this are considered vulnerable.

    Returns:
        Tuple of (kemeny_constant, is_vulnerable).
    """
    try:
        from reticulate.heat_kernel import kemeny_constant
        k = kemeny_constant(ss)
        if k != k:  # NaN check
            return (float('inf'), False)
        return (k, k < threshold)
    except (ValueError, ZeroDivisionError, ImportError):
        return (float('inf'), False)


# ---------------------------------------------------------------------------
# Critical state identification
# ---------------------------------------------------------------------------

def _compute_ranks(ss: "StateSpace") -> dict[int, int]:
    """BFS rank from top."""
    ranks: dict[int, int] = {ss.top: 0}
    queue = [ss.top]
    head = 0
    while head < len(queue):
        s = queue[head]
        head += 1
        for _, tgt in ss.enabled(s):
            if tgt not in ranks:
                ranks[tgt] = ranks[s] + 1
                queue.append(tgt)
    return ranks


def _critical_states(
    ss: "StateSpace",
    dos_targets: list[int],
    widths: list[int],
) -> list[int]:
    """Find critical states: intersection of high-branching and high-width ranks.

    A critical state is a DoS target whose rank has above-average width.
    These states are the most dangerous: high branching lets the attacker
    force resource allocation, and high width means many concurrent
    resources are already active.
    """
    if not widths or not dos_targets:
        return []

    ranks = _compute_ranks(ss)
    avg_width = sum(widths) / len(widths)

    critical = []
    for s in dos_targets:
        r = ranks.get(s)
        if r is not None and r < len(widths):
            if widths[r] >= avg_width:
                critical.append(s)

    critical.sort()
    return critical


# ---------------------------------------------------------------------------
# Vulnerability score
# ---------------------------------------------------------------------------

def _vulnerability_score(
    dos_surface_score: float,
    max_width: int,
    num_states: int,
    has_amplification: bool,
    kemeny_vulnerable: bool,
) -> float:
    """Compute composite DoS vulnerability score in [0, 1].

    Components:
    - Surface score (0-1): fraction of states that are DoS targets.
    - Width ratio (0-1): max_width / num_states (how "wide" the lattice gets).
    - Amplification (0 or 0.25): whether cycles allow amplification.
    - Kemeny (0 or 0.25): whether random probing is effective.

    The score is a weighted average.
    """
    if num_states <= 1:
        return 0.0

    width_ratio = min(max_width / max(num_states, 1), 1.0)

    score = (
        0.30 * dos_surface_score
        + 0.20 * width_ratio
        + 0.25 * (1.0 if has_amplification else 0.0)
        + 0.25 * (1.0 if kemeny_vulnerable else 0.0)
    )
    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def analyze_dos_resistance(ss: "StateSpace") -> DoSResult:
    """Perform complete DoS resistance analysis.

    Combines DoS surface, width exposure, throughput bottleneck,
    and Kemeny vulnerability into a single composite result.
    """
    bf = _branching_factors(ss)
    max_bf = max(bf.values()) if bf else 0
    surface = dos_surface(ss)
    surface_score = len(surface) / max(len(ss.states), 1)

    widths = width_exposure(ss)
    max_w = max(widths) if widths else 0
    max_w_rank = widths.index(max_w) if widths else 0

    bottleneck = throughput_bottleneck(ss)
    has_amp = bottleneck > 0.0

    kemeny_val, kemeny_vuln = kemeny_vulnerability(ss)

    critical = _critical_states(ss, surface, widths)

    score = _vulnerability_score(
        surface_score, max_w, len(ss.states), has_amp, kemeny_vuln,
    )

    return DoSResult(
        dos_surface=surface,
        dos_surface_score=round(surface_score, 4),
        width_exposure=widths,
        max_width=max_w,
        max_width_rank=max_w_rank,
        throughput_bottleneck=bottleneck,
        has_amplification=has_amp,
        kemeny_value=kemeny_val,
        kemeny_vulnerable=kemeny_vuln,
        branching_factors=bf,
        max_branching=max_bf,
        critical_states=critical,
        vulnerability_score=score,
        num_states=len(ss.states),
    )
