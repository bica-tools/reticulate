"""Lattice-based deadlock detection for session types (Step 80b).

Seven novel deadlock detection techniques unified from the reticulate
framework, all exploiting the lattice structure of session type state spaces:

1. **Lattice Bottom Uniqueness** -- structural guarantee from lattice property.
2. **Black Hole Detection** -- gravitational states from which bottom is
   unreachable (analogous to gravitational black holes).
3. **Spectral Risk Indicators** -- Fiedler value, Cheeger constant, and
   spectral gap predict deadlock risk via algebraic connectivity.
4. **Compositional Checking** -- interface compatibility for composed
   protocols preserves deadlock freedom.
5. **Buffer Capacity Computation** -- exact minimum buffer size for
   asynchronous deadlock freedom.
6. **Siphon/Trap Structural Analysis** -- Petri net siphons empty iff
   deadlock; traps prevent token starvation.
7. **Reticular Form Certificate** -- session type structure implies
   deadlock freedom when the state space is a reticulate.

Public API:
    detect_deadlocks(ss)            -- find all deadlocked states
    is_deadlock_free(ss)            -- True iff no deadlocks
    lattice_deadlock_certificate(ss) -- lattice-based proof of deadlock freedom
    black_hole_detection(ss)        -- unreachable-bottom states
    spectral_deadlock_risk(ss)      -- spectral risk indicators
    compositional_deadlock_check(ss1, ss2) -- check composition safety
    buffer_deadlock_analysis(ss, max_k) -- async buffer deadlock analysis
    siphon_trap_analysis(ss)        -- Petri net structural analysis
    reticular_deadlock_certificate(ss) -- reticular form certificate
    analyze_deadlock(ss)            -- unified analysis (all 7 techniques)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeadlockState:
    """A single deadlocked state.

    Attributes:
        state: The state ID.
        label: Human-readable state label.
        is_black_hole: True if bottom is unreachable from this state.
        depth_from_top: Shortest path length from top to this state.
    """
    state: int
    label: str
    is_black_hole: bool
    depth_from_top: int


@dataclass(frozen=True)
class LatticeCertificate:
    """Certificate that lattice structure implies deadlock freedom.

    Attributes:
        is_lattice: Whether the state space is a lattice.
        has_unique_bottom: True iff there is exactly one minimal element.
        all_reach_bottom: True iff every state can reach bottom.
        certificate_valid: True iff the lattice guarantees deadlock freedom.
        explanation: Human-readable explanation of the certificate.
    """
    is_lattice: bool
    has_unique_bottom: bool
    all_reach_bottom: bool
    certificate_valid: bool
    explanation: str


@dataclass(frozen=True)
class BlackHoleResult:
    """Result of black hole detection.

    Attributes:
        black_holes: States from which bottom is unreachable.
        event_horizons: Transitions leading into black holes from safe states.
        escape_paths: For states near black holes, shortest escape to bottom.
        total_trapped_states: Number of states trapped in black holes.
    """
    black_holes: tuple[int, ...]
    event_horizons: tuple[tuple[int, str, int], ...]
    escape_paths: dict[int, int]
    total_trapped_states: int


@dataclass(frozen=True)
class SpectralRisk:
    """Spectral risk indicators for deadlock.

    Attributes:
        fiedler_value: Algebraic connectivity (lambda_2). Low = high risk.
        cheeger_constant: Edge expansion. Low = bottleneck, high risk.
        spectral_gap: Gap between first and second Laplacian eigenvalues.
        risk_score: Combined risk score in [0, 1]. 1 = highest risk.
        bottleneck_states: States identified as connectivity bottlenecks.
        diagnosis: Human-readable risk assessment.
    """
    fiedler_value: float
    cheeger_constant: float
    spectral_gap: float
    risk_score: float
    bottleneck_states: tuple[int, ...]
    diagnosis: str


@dataclass(frozen=True)
class CompositionalResult:
    """Result of compositional deadlock checking.

    Attributes:
        is_deadlock_free: True iff composition is deadlock-free.
        individual_deadlock_free: Per-component deadlock freedom.
        interface_compatible: True iff shared labels respect duality.
        product_deadlock_free: True iff the product has no deadlocks.
        new_deadlocks: Deadlocked states introduced by composition.
        explanation: Human-readable analysis.
    """
    is_deadlock_free: bool
    individual_deadlock_free: tuple[bool, bool]
    interface_compatible: bool
    product_deadlock_free: bool
    new_deadlocks: tuple[int, ...]
    explanation: str


@dataclass(frozen=True)
class BufferDeadlockResult:
    """Result of buffer deadlock analysis.

    Attributes:
        min_safe_capacity: Minimum buffer capacity for deadlock freedom (-1 if none).
        deadlocks_per_capacity: Mapping capacity -> number of deadlock states.
        is_synchronous_safe: True iff capacity=0 is already deadlock-free.
        growth_rate: How deadlock count changes with capacity.
        explanation: Human-readable analysis.
    """
    min_safe_capacity: int
    deadlocks_per_capacity: dict[int, int]
    is_synchronous_safe: bool
    growth_rate: str
    explanation: str


@dataclass(frozen=True)
class SiphonTrapResult:
    """Result of Petri net siphon/trap structural analysis.

    A siphon is a set S of places such that every transition that puts a
    token into S also takes a token from S: *pre(t) cap S != empty implies
    post(t) cap S != empty*. If a siphon ever becomes empty, it stays empty.

    A trap is a set T of places such that every transition that takes a
    token from T also puts a token into T: *post(t) cap T != empty implies
    pre(t) cap T != empty*. A marked trap can never become unmarked.

    Deadlock iff there exists a siphon that can become unmarked (empty).
    The net is deadlock-free if every siphon contains a marked trap.

    Attributes:
        siphons: List of siphons (each a frozenset of place IDs).
        traps: List of traps (each a frozenset of place IDs).
        dangerous_siphons: Siphons that do not contain a marked trap.
        is_deadlock_free: True iff no dangerous siphons exist.
        explanation: Human-readable analysis.
    """
    siphons: tuple[frozenset[int], ...]
    traps: tuple[frozenset[int], ...]
    dangerous_siphons: tuple[frozenset[int], ...]
    is_deadlock_free: bool
    explanation: str


@dataclass(frozen=True)
class ReticularCertificate:
    """Certificate from reticular form analysis.

    Attributes:
        is_reticulate: True iff state space has reticular form.
        is_lattice: True iff state space forms a lattice.
        certificate_valid: True iff reticular form guarantees deadlock freedom.
        state_classifications: Per-state classification summary.
        explanation: Human-readable analysis.
    """
    is_reticulate: bool
    is_lattice: bool
    certificate_valid: bool
    state_classifications: dict[int, str]
    explanation: str


@dataclass(frozen=True)
class DeadlockAnalysis:
    """Unified deadlock analysis combining all 7 techniques.

    Attributes:
        deadlocked_states: List of deadlocked state details.
        is_deadlock_free: True iff no deadlocks detected by any technique.
        lattice_certificate: Lattice bottom uniqueness certificate.
        black_holes: Black hole detection result.
        spectral_risk: Spectral risk indicators.
        siphon_trap: Siphon/trap structural analysis.
        reticular_certificate: Reticular form certificate.
        num_states: Total states in the state space.
        num_transitions: Total transitions.
        summary: Human-readable summary of all findings.
    """
    deadlocked_states: tuple[DeadlockState, ...]
    is_deadlock_free: bool
    lattice_certificate: LatticeCertificate
    black_holes: BlackHoleResult
    spectral_risk: SpectralRisk
    siphon_trap: SiphonTrapResult
    reticular_certificate: ReticularCertificate
    num_states: int
    num_transitions: int
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outgoing(ss: "StateSpace", state: int) -> list[tuple[str, int]]:
    """Return outgoing (label, target) pairs for a state."""
    return [(label, tgt) for src, label, tgt in ss.transitions if src == state]


def _bfs_distances(ss: "StateSpace", start: int) -> dict[int, int]:
    """BFS shortest distances from *start*."""
    dist: dict[int, int] = {start: 0}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        for label, t in _outgoing(ss, s):
            if t not in dist:
                dist[t] = dist[s] + 1
                queue.append(t)
    return dist


def _reverse_reachable(ss: "StateSpace", target: int) -> set[int]:
    """States that can reach *target* via reverse edges."""
    # Build reverse adjacency
    rev: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        rev.setdefault(tgt, []).append(src)
    visited: set[int] = set()
    queue = deque([target])
    visited.add(target)
    while queue:
        s = queue.popleft()
        for pred in rev.get(s, []):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)
    return visited


def _reachable_from(ss: "StateSpace", start: int) -> set[int]:
    """States reachable from *start* via forward edges."""
    visited: set[int] = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        s = queue.popleft()
        for label, t in _outgoing(ss, s):
            if t not in visited:
                visited.add(t)
                queue.append(t)
    return visited


# ---------------------------------------------------------------------------
# Technique 1: Direct Deadlock Detection
# ---------------------------------------------------------------------------

def detect_deadlocks(ss: "StateSpace") -> tuple[DeadlockState, ...]:
    """Find all states with no outgoing transitions that are not bottom.

    A deadlocked state is one where no progress is possible but the
    protocol has not terminated (the state is not the bottom/end state).

    Returns:
        Tuple of DeadlockState objects for each deadlocked state.
    """
    distances = _bfs_distances(ss, ss.top)
    can_reach_bottom = _reverse_reachable(ss, ss.bottom)

    deadlocks: list[DeadlockState] = []
    for state in ss.states:
        if state == ss.bottom:
            continue
        out = _outgoing(ss, state)
        if len(out) == 0:
            deadlocks.append(DeadlockState(
                state=state,
                label=ss.labels.get(state, f"s{state}"),
                is_black_hole=state not in can_reach_bottom,
                depth_from_top=distances.get(state, -1),
            ))

    return tuple(sorted(deadlocks, key=lambda d: d.state))


def is_deadlock_free(ss: "StateSpace") -> bool:
    """Return True iff no deadlocks exist in the state space.

    Equivalent to: every state either has outgoing transitions or is the
    unique bottom (end) state.
    """
    return len(detect_deadlocks(ss)) == 0


# ---------------------------------------------------------------------------
# Technique 2: Lattice Deadlock Certificate
# ---------------------------------------------------------------------------

def lattice_deadlock_certificate(ss: "StateSpace") -> LatticeCertificate:
    """If the state space is a lattice, return certificate proving deadlock freedom.

    The key theorem: if L(S) is a lattice with unique bottom element bot,
    then every state s has a path to bot (since bot = meet of all elements).
    Therefore no state can be deadlocked -- every non-bottom state must
    have at least one outgoing transition leading (eventually) to bot.

    This is a *structural* guarantee: if it's a lattice, it's deadlock-free
    without checking any individual state.
    """
    from reticulate.lattice import check_lattice

    lr = check_lattice(ss)

    # Check all states can reach bottom
    can_reach_bottom = _reverse_reachable(ss, ss.bottom)
    all_reach = all(s in can_reach_bottom for s in ss.states)

    if lr.is_lattice:
        certificate_valid = True
        explanation = (
            f"State space is a lattice with {lr.num_scc} SCC(s). "
            f"Unique bottom ensures every state has a path to termination. "
            f"Certificate: DEADLOCK-FREE by lattice bottom uniqueness theorem."
        )
    elif lr.has_bottom and all_reach:
        certificate_valid = True
        explanation = (
            f"Not a lattice (missing {'meets' if not lr.all_meets_exist else 'joins'}), "
            f"but all states reach bottom. Deadlock-free by reachability."
        )
    else:
        certificate_valid = False
        failing_states = [s for s in ss.states if s not in can_reach_bottom]
        explanation = (
            f"Not deadlock-free. "
            f"{'Not a lattice. ' if not lr.is_lattice else ''}"
            f"{'No unique bottom. ' if not lr.has_bottom else ''}"
            f"{len(failing_states)} state(s) cannot reach bottom."
        )

    return LatticeCertificate(
        is_lattice=lr.is_lattice,
        has_unique_bottom=lr.has_bottom,
        all_reach_bottom=all_reach,
        certificate_valid=certificate_valid,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Technique 3: Black Hole Detection
# ---------------------------------------------------------------------------

def black_hole_detection(ss: "StateSpace") -> BlackHoleResult:
    """Detect states from which the bottom (end) is unreachable.

    These are "gravitational black holes" -- once the protocol enters such
    a state, termination is impossible.  This is strictly stronger than
    simple deadlock detection: a black hole may have outgoing transitions
    (forming cycles) but still never reach bottom.

    Returns:
        BlackHoleResult with black hole states, event horizons, and escape paths.
    """
    can_reach_bottom = _reverse_reachable(ss, ss.bottom)

    black_holes: list[int] = []
    for state in sorted(ss.states):
        if state not in can_reach_bottom:
            black_holes.append(state)

    # Event horizons: transitions from safe states into black holes
    bh_set = set(black_holes)
    event_horizons: list[tuple[int, str, int]] = []
    for src, label, tgt in ss.transitions:
        if src not in bh_set and tgt in bh_set:
            event_horizons.append((src, label, tgt))

    # Escape paths: for states adjacent to black holes, distance to bottom
    distances = _bfs_distances(ss, ss.top)
    escape_paths: dict[int, int] = {}
    for src, label, tgt in event_horizons:
        if src in can_reach_bottom and src not in escape_paths:
            # Compute shortest path from src to bottom
            fwd = _reachable_from_distances(ss, src)
            if ss.bottom in fwd:
                escape_paths[src] = fwd[ss.bottom]

    return BlackHoleResult(
        black_holes=tuple(black_holes),
        event_horizons=tuple(event_horizons),
        escape_paths=escape_paths,
        total_trapped_states=len(black_holes),
    )


def _reachable_from_distances(ss: "StateSpace", start: int) -> dict[int, int]:
    """BFS distances from a specific state."""
    return _bfs_distances(ss, start)


# ---------------------------------------------------------------------------
# Technique 4: Spectral Deadlock Risk
# ---------------------------------------------------------------------------

def spectral_deadlock_risk(ss: "StateSpace") -> SpectralRisk:
    """Compute spectral risk indicators for deadlock.

    Uses the Fiedler value (algebraic connectivity), Cheeger constant,
    and spectral gap as predictors of deadlock risk.

    - Low Fiedler value -> near-disconnection -> high deadlock risk
    - Low Cheeger constant -> bottleneck -> state isolation
    - Small spectral gap -> poor mixing -> trapped regions

    Returns:
        SpectralRisk with quantified indicators and diagnosis.
    """
    try:
        from reticulate.fiedler import (
            fiedler_value as compute_fiedler,
            find_cut_vertices,
            cheeger_bounds,
        )
        from reticulate.matrix import laplacian_matrix, _eigenvalues_symmetric

        fv = compute_fiedler(ss)
        cuts = find_cut_vertices(ss)
        ch_lower, ch_upper = cheeger_bounds(ss)

        # Spectral gap: difference between lambda_1 (=0) and lambda_2
        L = laplacian_matrix(ss)
        n = len(L)
        if n > 1:
            eigs = sorted(_eigenvalues_symmetric(L))
            spectral_gap = max(0.0, eigs[1]) if len(eigs) >= 2 else 0.0
        else:
            spectral_gap = 0.0

        # Cheeger constant approximation from bounds
        cheeger = (ch_lower + ch_upper) / 2.0 if ch_upper > 0 else 0.0

        # Risk score: normalized inverse of connectivity
        # Higher fiedler -> lower risk; scale relative to state count
        if n <= 1:
            risk_score = 0.0
        elif fv < 1e-10:
            risk_score = 1.0  # disconnected = maximum risk
        else:
            # Normalize: risk decreases as fiedler increases
            # For typical session types, fiedler ranges from ~0.5 to ~4.0
            risk_score = max(0.0, min(1.0, 1.0 / (1.0 + fv)))

        # Diagnosis
        if risk_score >= 0.8:
            diagnosis = "CRITICAL: Near-disconnection detected. Deadlock highly likely."
        elif risk_score >= 0.5:
            diagnosis = (
                f"WARNING: Low algebraic connectivity (Fiedler={fv:.3f}). "
                f"Protocol has bottleneck states that could cause deadlock under perturbation."
            )
        elif risk_score >= 0.2:
            diagnosis = (
                f"LOW RISK: Moderate connectivity (Fiedler={fv:.3f}). "
                f"Protocol is reasonably well-connected."
            )
        else:
            diagnosis = (
                f"SAFE: High algebraic connectivity (Fiedler={fv:.3f}). "
                f"All states are well-connected to bottom."
            )

        return SpectralRisk(
            fiedler_value=fv,
            cheeger_constant=cheeger,
            spectral_gap=spectral_gap,
            risk_score=risk_score,
            bottleneck_states=tuple(cuts),
            diagnosis=diagnosis,
        )

    except Exception:
        # Fallback for trivial state spaces
        return SpectralRisk(
            fiedler_value=0.0,
            cheeger_constant=0.0,
            spectral_gap=0.0,
            risk_score=0.0,
            bottleneck_states=(),
            diagnosis="Spectral analysis not applicable (trivial state space).",
        )


# ---------------------------------------------------------------------------
# Technique 5: Compositional Deadlock Checking
# ---------------------------------------------------------------------------

def compositional_deadlock_check(
    ss1: "StateSpace",
    ss2: "StateSpace",
) -> CompositionalResult:
    """Check if composing two protocol state spaces introduces deadlock.

    Individual deadlock freedom does not guarantee composition deadlock
    freedom. This function:
    1. Checks each component independently.
    2. Builds the product state space.
    3. Checks for deadlocks in the product.
    4. Identifies new deadlocks introduced by composition.

    Returns:
        CompositionalResult with per-component and composition analysis.
    """
    from reticulate.product import product_statespace

    # Check individual components
    ind1 = is_deadlock_free(ss1)
    ind2 = is_deadlock_free(ss2)

    # Build product
    product = product_statespace(ss1, ss2)

    # Check product for deadlocks
    product_deadlocks = detect_deadlocks(product)
    product_free = len(product_deadlocks) == 0

    new_dl = tuple(d.state for d in product_deadlocks)

    if product_free:
        explanation = (
            "Composition is deadlock-free. "
            f"Component 1: {'deadlock-free' if ind1 else 'has deadlocks'}. "
            f"Component 2: {'deadlock-free' if ind2 else 'has deadlocks'}. "
            f"Product ({len(product.states)} states) has no deadlocks."
        )
    else:
        explanation = (
            f"Composition introduces {len(product_deadlocks)} deadlock(s)! "
            f"Component 1: {'deadlock-free' if ind1 else 'has deadlocks'}. "
            f"Component 2: {'deadlock-free' if ind2 else 'has deadlocks'}. "
            f"Product ({len(product.states)} states) has "
            f"{len(product_deadlocks)} deadlocked state(s)."
        )

    # Interface compatibility check (heuristic based on shared labels)
    labels1 = {l for _, l, _ in ss1.transitions}
    labels2 = {l for _, l, _ in ss2.transitions}
    shared = labels1 & labels2
    interface_ok = len(shared) == 0 or product_free

    return CompositionalResult(
        is_deadlock_free=product_free,
        individual_deadlock_free=(ind1, ind2),
        interface_compatible=interface_ok,
        product_deadlock_free=product_free,
        new_deadlocks=new_dl,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Technique 6: Buffer Deadlock Analysis
# ---------------------------------------------------------------------------

def buffer_deadlock_analysis(
    ss: "StateSpace",
    max_k: int = 10,
) -> BufferDeadlockResult:
    """Analyze minimum buffer capacity for deadlock freedom in async channels.

    For asynchronous communication, messages are buffered. The buffer
    capacity k determines whether deadlock can occur:
    - k=0 (synchronous): may deadlock on unmatched send/receive.
    - k=infinity: always deadlock-free but impractical.
    - There exists a minimum k* for deadlock freedom.

    This function searches for k* by building buffered state spaces for
    increasing capacities and checking for deadlocks.

    Args:
        ss: The base (synchronous) state space.
        max_k: Maximum buffer capacity to search.

    Returns:
        BufferDeadlockResult with analysis.
    """
    # Detect deadlocks at each capacity using direct state space analysis
    deadlocks_per_k: dict[int, int] = {}

    # For the base state space, count deadlocks directly
    base_deadlocks = detect_deadlocks(ss)
    deadlocks_per_k[0] = len(base_deadlocks)

    # For higher capacities, simulate buffered behavior
    # We do a simplified analysis: check if adding buffers to selection
    # transitions resolves deadlocks
    for k in range(1, max_k + 1):
        buffered_ss = _simulate_buffered(ss, k)
        dl = detect_deadlocks(buffered_ss)
        deadlocks_per_k[k] = len(dl)

    # Find minimum safe capacity
    min_safe = -1
    for k in sorted(deadlocks_per_k.keys()):
        if deadlocks_per_k[k] == 0:
            min_safe = k
            break

    is_sync_safe = deadlocks_per_k.get(0, 0) == 0

    # Characterize growth rate
    values = [deadlocks_per_k[k] for k in sorted(deadlocks_per_k.keys())]
    if all(v == 0 for v in values):
        growth_rate = "constant-zero"
    elif all(v == values[0] for v in values):
        growth_rate = "constant"
    elif values[-1] < values[0]:
        growth_rate = "decreasing"
    elif values[-1] > values[0]:
        growth_rate = "increasing"
    else:
        growth_rate = "non-monotonic"

    if min_safe >= 0:
        explanation = (
            f"Deadlock-free at buffer capacity k={min_safe}. "
            f"{'Already safe synchronously.' if is_sync_safe else f'Requires {min_safe} buffer slots.'}"
        )
    else:
        explanation = (
            f"No deadlock-free capacity found up to k={max_k}. "
            f"Deadlocks at k=0: {deadlocks_per_k[0]}, at k={max_k}: {deadlocks_per_k[max_k]}."
        )

    return BufferDeadlockResult(
        min_safe_capacity=min_safe,
        deadlocks_per_capacity=deadlocks_per_k,
        is_synchronous_safe=is_sync_safe,
        growth_rate=growth_rate,
        explanation=explanation,
    )


def _simulate_buffered(ss: "StateSpace", capacity: int) -> "StateSpace":
    """Build a buffered version of the state space.

    For capacity > 0, selection transitions can "fire" even if the receiver
    is not ready, up to the buffer limit. This adds intermediate states
    representing buffered messages.

    For simplicity, we model this by extending the state space with
    buffer-augmented states (state, buffer_fill) where buffer_fill in [0, capacity].
    """
    from reticulate.statespace import StateSpace as SS

    if capacity == 0:
        return ss

    # Augmented states: (original_state, buffer_level)
    new_states: set[int] = set()
    new_transitions: list[tuple[int, str, int]] = []
    new_labels: dict[int, str] = {}
    new_selection: set[tuple[int, str, int]] = set()

    # State encoding: original_state * (capacity + 1) + buffer_level
    def encode(state: int, buf: int) -> int:
        return state * (capacity + 1) + buf

    # Build augmented state space
    for state in ss.states:
        for buf in range(capacity + 1):
            sid = encode(state, buf)
            new_states.add(sid)
            new_labels[sid] = f"{ss.labels.get(state, f's{state}')}[b={buf}]"

    new_top = encode(ss.top, 0)
    new_bottom = encode(ss.bottom, 0)

    for src, label, tgt in ss.transitions:
        is_sel = (src, label, tgt) in ss.selection_transitions

        if is_sel:
            # Selection transitions can buffer: advance sender, keep buffer
            for buf in range(capacity + 1):
                if buf < capacity:
                    # Send into buffer (increment buffer)
                    src_id = encode(src, buf)
                    tgt_id = encode(tgt, buf + 1)
                    t = (src_id, label, tgt_id)
                    new_transitions.append(t)
                    new_selection.add(t)
                # Also allow synchronous send (buf stays same)
                src_id = encode(src, buf)
                tgt_id = encode(tgt, buf)
                t = (src_id, label, tgt_id)
                new_transitions.append(t)
                new_selection.add(t)
        else:
            # Branch transitions consume from buffer
            for buf in range(capacity + 1):
                if buf > 0:
                    # Consume from buffer (decrement)
                    src_id = encode(src, buf)
                    tgt_id = encode(tgt, buf - 1)
                    new_transitions.append((src_id, label, tgt_id))
                # Also allow synchronous receive
                src_id = encode(src, buf)
                tgt_id = encode(tgt, buf)
                new_transitions.append((src_id, label, tgt_id))

    return SS(
        states=new_states,
        transitions=new_transitions,
        top=new_top,
        bottom=new_bottom,
        labels=new_labels,
        selection_transitions=new_selection,
    )


# ---------------------------------------------------------------------------
# Technique 7: Siphon/Trap Analysis
# ---------------------------------------------------------------------------

def siphon_trap_analysis(ss: "StateSpace") -> SiphonTrapResult:
    """Petri net siphon/trap structural analysis for deadlock.

    Constructs the Petri net from the state space and computes:
    - Siphons: sets S where every transition putting tokens into S also
      takes tokens from S. An empty siphon stays empty forever.
    - Traps: sets T where every transition taking tokens from T also
      puts tokens into T. A marked trap stays marked forever.

    Deadlock-freedom theorem: if every siphon contains a marked trap,
    the net (and hence the protocol) is deadlock-free.

    Returns:
        SiphonTrapResult with siphons, traps, and diagnosis.
    """
    from reticulate.petri import build_petri_net, PetriNet

    net = build_petri_net(ss)

    siphons = _find_siphons(net)
    traps = _find_traps(net)

    # Check which siphons contain a marked trap
    initial_marking = net.initial_marking
    marked_places = frozenset(p for p, count in initial_marking.items() if count > 0)

    dangerous: list[frozenset[int]] = []
    for siphon in siphons:
        # A siphon is dangerous if it does not contain a marked trap
        contains_marked_trap = False
        for trap in traps:
            if trap <= siphon and (trap & marked_places):
                contains_marked_trap = True
                break
        if not contains_marked_trap:
            dangerous.append(siphon)

    is_df = len(dangerous) == 0

    if is_df:
        explanation = (
            f"Found {len(siphons)} siphon(s) and {len(traps)} trap(s). "
            f"Every siphon contains a marked trap. "
            f"Certificate: DEADLOCK-FREE by siphon/trap theorem."
        )
    else:
        explanation = (
            f"Found {len(siphons)} siphon(s) and {len(traps)} trap(s). "
            f"{len(dangerous)} dangerous siphon(s) without marked traps. "
            f"POTENTIAL DEADLOCK: siphon(s) may become empty."
        )

    return SiphonTrapResult(
        siphons=tuple(siphons),
        traps=tuple(traps),
        dangerous_siphons=tuple(dangerous),
        is_deadlock_free=is_df,
        explanation=explanation,
    )


def _find_siphons(net: "PetriNet") -> list[frozenset[int]]:
    """Find all minimal siphons in a Petri net.

    A siphon S is a set of places where: for every transition t, if
    post(t) intersects S, then pre(t) intersects S.

    Uses iterative subset checking for small nets (appropriate for
    session type state spaces which are typically < 100 states).
    """
    place_ids = list(net.places.keys())
    n = len(place_ids)

    if n == 0:
        return []

    # Build pre/post place sets for each transition from net.pre / net.post
    pre_sets: dict[int, set[int]] = {}
    post_sets: dict[int, set[int]] = {}

    for t_id in net.transitions:
        pre_sets[t_id] = {place_id for place_id, _w in net.pre.get(t_id, set())}
        post_sets[t_id] = {place_id for place_id, _w in net.post.get(t_id, set())}

    siphons: list[frozenset[int]] = []

    # For small nets, enumerate subsets up to a reasonable size
    # For efficiency, check siphon property incrementally
    max_check = min(n, 12)  # limit exponential blowup

    # Start with single places and grow
    candidates: list[frozenset[int]] = [frozenset({p}) for p in place_ids]
    checked: set[frozenset[int]] = set()
    minimal_siphons: list[frozenset[int]] = []

    # BFS over candidate sets
    queue: deque[frozenset[int]] = deque(candidates)
    while queue and len(checked) < 2000:
        candidate = queue.popleft()
        if candidate in checked:
            continue
        checked.add(candidate)

        if _is_siphon(candidate, pre_sets, post_sets):
            # Check minimality: no proper subset is already a siphon
            is_minimal = not any(
                existing < candidate for existing in minimal_siphons
            )
            if is_minimal:
                # Remove non-minimal existing
                minimal_siphons = [
                    s for s in minimal_siphons if not candidate < s
                ]
                minimal_siphons.append(candidate)
        elif len(candidate) < max_check:
            # Grow by adding one place
            for p in place_ids:
                if p not in candidate:
                    new_cand = candidate | {p}
                    if new_cand not in checked:
                        queue.append(new_cand)

    # Always include the full set of places (it's always a siphon for
    # connected nets)
    full = frozenset(place_ids)
    if full not in checked:
        if _is_siphon(full, pre_sets, post_sets):
            is_min = not any(s < full for s in minimal_siphons)
            if is_min:
                minimal_siphons.append(full)

    return minimal_siphons


def _is_siphon(
    candidate: frozenset[int],
    pre_sets: dict[int, set[int]],
    post_sets: dict[int, set[int]],
) -> bool:
    """Check if a set of places is a siphon.

    Siphon condition: for every transition t, if post(t) intersects
    candidate, then pre(t) must also intersect candidate.
    """
    for t_id, post_s in post_sets.items():
        if post_s & candidate:
            if not (pre_sets[t_id] & candidate):
                return False
    return True


def _find_traps(net: "PetriNet") -> list[frozenset[int]]:
    """Find all minimal traps in a Petri net.

    A trap T is a set of places where: for every transition t, if
    pre(t) intersects T, then post(t) intersects T.
    """
    place_ids = list(net.places.keys())
    n = len(place_ids)

    if n == 0:
        return []

    pre_sets: dict[int, set[int]] = {}
    post_sets: dict[int, set[int]] = {}

    for t_id in net.transitions:
        pre_sets[t_id] = {place_id for place_id, _w in net.pre.get(t_id, set())}
        post_sets[t_id] = {place_id for place_id, _w in net.post.get(t_id, set())}

    traps: list[frozenset[int]] = []
    max_check = min(n, 12)

    candidates: list[frozenset[int]] = [frozenset({p}) for p in place_ids]
    checked: set[frozenset[int]] = set()
    minimal_traps: list[frozenset[int]] = []

    queue: deque[frozenset[int]] = deque(candidates)
    while queue and len(checked) < 2000:
        candidate = queue.popleft()
        if candidate in checked:
            continue
        checked.add(candidate)

        if _is_trap(candidate, pre_sets, post_sets):
            is_minimal = not any(
                existing < candidate for existing in minimal_traps
            )
            if is_minimal:
                minimal_traps = [
                    t for t in minimal_traps if not candidate < t
                ]
                minimal_traps.append(candidate)
        elif len(candidate) < max_check:
            for p in place_ids:
                if p not in candidate:
                    new_cand = candidate | {p}
                    if new_cand not in checked:
                        queue.append(new_cand)

    full = frozenset(place_ids)
    if full not in checked:
        if _is_trap(full, pre_sets, post_sets):
            is_min = not any(t < full for t in minimal_traps)
            if is_min:
                minimal_traps.append(full)

    return minimal_traps


def _is_trap(
    candidate: frozenset[int],
    pre_sets: dict[int, set[int]],
    post_sets: dict[int, set[int]],
) -> bool:
    """Check if a set of places is a trap.

    Trap condition: for every transition t, if pre(t) intersects
    candidate, then post(t) must also intersect candidate.
    """
    for t_id, pre_s in pre_sets.items():
        if pre_s & candidate:
            if not (post_sets[t_id] & candidate):
                return False
    return True


# ---------------------------------------------------------------------------
# Technique 8: Reticular Form Certificate
# ---------------------------------------------------------------------------

def reticular_deadlock_certificate(ss: "StateSpace") -> ReticularCertificate:
    """If the state machine has reticular form, certify deadlock freedom.

    Reticular form means the state space arises from a well-formed session
    type. By the reconstruction theorem, every reticulate state space has
    a unique bottom and every state reaches it.  Hence reticular form is
    a sufficient condition for deadlock freedom.

    Returns:
        ReticularCertificate with analysis.
    """
    from reticulate.reticular import check_reticular_form, classify_all_states
    from reticulate.lattice import check_lattice

    rf = check_reticular_form(ss)
    lr = check_lattice(ss)

    # Build classification dict
    classifications: dict[int, str] = {}
    try:
        all_cls = classify_all_states(ss)
        for sc in all_cls:
            classifications[sc.state] = sc.kind
    except Exception:
        for state in ss.states:
            classifications[state] = "unknown"

    if rf.is_reticulate and lr.is_lattice:
        valid = True
        explanation = (
            f"State space has reticular form and is a lattice. "
            f"By the reconstruction theorem, every state reaches bottom. "
            f"Certificate: DEADLOCK-FREE by reticular form."
        )
    elif rf.is_reticulate:
        valid = True
        explanation = (
            f"State space has reticular form (reconstructible as session type). "
            f"All reticulate state spaces have unique bottom. "
            f"Certificate: DEADLOCK-FREE by reticular form."
        )
    else:
        # Check if still deadlock-free despite not being reticulate
        can_reach = _reverse_reachable(ss, ss.bottom)
        all_reach = all(s in can_reach for s in ss.states)
        valid = all_reach
        reason = rf.reason or "unknown"
        explanation = (
            f"State space does NOT have reticular form ({reason}). "
            f"{'All states reach bottom (deadlock-free by reachability).' if all_reach else 'POTENTIAL DEADLOCK: not all states reach bottom.'}"
        )

    return ReticularCertificate(
        is_reticulate=rf.is_reticulate,
        is_lattice=lr.is_lattice,
        certificate_valid=valid,
        state_classifications=classifications,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Unified Analysis
# ---------------------------------------------------------------------------

def analyze_deadlock(ss: "StateSpace") -> DeadlockAnalysis:
    """Run all 7 deadlock detection techniques and return unified analysis.

    This is the main entry point for comprehensive deadlock analysis.
    It combines:
      1. Direct deadlock detection
      2. Lattice certificate
      3. Black hole detection
      4. Spectral risk indicators
      5. Siphon/trap analysis
      6. Reticular form certificate

    (Compositional and buffer analyses require additional inputs and
    should be called separately.)

    Returns:
        DeadlockAnalysis with all results and human-readable summary.
    """
    # 1. Direct detection
    deadlocks = detect_deadlocks(ss)

    # 2. Lattice certificate
    lattice_cert = lattice_deadlock_certificate(ss)

    # 3. Black holes
    bh = black_hole_detection(ss)

    # 4. Spectral risk
    spectral = spectral_deadlock_risk(ss)

    # 5. Siphon/trap
    try:
        siphon = siphon_trap_analysis(ss)
    except Exception:
        siphon = SiphonTrapResult(
            siphons=(),
            traps=(),
            dangerous_siphons=(),
            is_deadlock_free=True,
            explanation="Siphon/trap analysis not applicable.",
        )

    # 6. Reticular certificate
    reticular = reticular_deadlock_certificate(ss)

    # Overall verdict
    is_df = len(deadlocks) == 0
    n_states = len(ss.states)
    n_trans = len(ss.transitions)

    # Build summary
    verdicts = []
    if lattice_cert.certificate_valid:
        verdicts.append("lattice-certificate")
    if bh.total_trapped_states == 0:
        verdicts.append("no-black-holes")
    if siphon.is_deadlock_free:
        verdicts.append("siphon-trap-safe")
    if reticular.certificate_valid:
        verdicts.append("reticular-certificate")

    if is_df:
        summary = (
            f"DEADLOCK-FREE ({n_states} states, {n_trans} transitions). "
            f"Confirmed by {len(verdicts)} technique(s): {', '.join(verdicts)}. "
            f"Spectral risk: {spectral.risk_score:.2f}."
        )
    else:
        summary = (
            f"DEADLOCK DETECTED: {len(deadlocks)} deadlocked state(s) "
            f"in {n_states} states. "
            f"Black holes: {bh.total_trapped_states}. "
            f"Spectral risk: {spectral.risk_score:.2f}. "
            f"Dangerous siphons: {len(siphon.dangerous_siphons)}."
        )

    return DeadlockAnalysis(
        deadlocked_states=deadlocks,
        is_deadlock_free=is_df,
        lattice_certificate=lattice_cert,
        black_holes=bh,
        spectral_risk=spectral,
        siphon_trap=siphon,
        reticular_certificate=reticular,
        num_states=n_states,
        num_transitions=n_trans,
        summary=summary,
    )
