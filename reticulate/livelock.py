"""Livelock detection via tropical analysis for session types (Step 80c).

A livelock occurs when a protocol enters a cycle from which it can never
reach the terminal state (bottom).  Unlike deadlock, where no transitions
are enabled, livelocked states have outgoing transitions but all paths
loop back without ever reaching completion.

Detection strategy:
1. **SCC decomposition**: Compute strongly connected components of the
   state space graph.
2. **Exit analysis**: For each non-trivial SCC (size >= 2 or self-loop),
   check whether any state in the SCC can reach bottom via a path that
   exits the SCC.
3. **Trapped SCCs**: An SCC with no exit path to bottom is a livelock.
4. **Tropical eigenvalue**: The tropical (max-plus) eigenvalue on the
   adjacency submatrix restricted to a trapped SCC measures the cycle
   throughput.  A positive eigenvalue confirms cyclic activity with no
   progress toward termination.
5. **Heat kernel indicator**: The heat kernel diagonal H_t(s,s)
   approaches 1.0 as t -> infinity for states in trapped SCCs,
   indicating probability mass stays trapped.

Public API:
    detect_livelock(ss)           -- find all livelocked states
    livelock_sccs(ss)             -- return the trapped SCCs
    tropical_livelock_score(ss)   -- tropical eigenvalue on trapped SCCs
    heat_kernel_livelock(ss, t)   -- heat kernel livelock indicator
    is_livelock_free(ss)          -- True iff no trapped SCCs
    analyze_livelock(ss)          -- comprehensive LivelockResult
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
class LivelockSCC:
    """A single livelocked strongly connected component.

    Attributes:
        states: The states in the trapped SCC.
        entry_transitions: Transitions from outside that enter this SCC.
        tropical_eigenvalue: Max cycle mean within this SCC (>0 = cycling).
        cycle_labels: Labels that appear on transitions within the SCC.
    """
    states: frozenset[int]
    entry_transitions: tuple[tuple[int, str, int], ...]
    tropical_eigenvalue: float
    cycle_labels: tuple[str, ...]


@dataclass(frozen=True)
class HeatKernelLivelockIndicator:
    """Heat kernel based livelock indicator for a single state.

    Attributes:
        state: The state ID.
        diagonal_value: H_t(state, state) at the given time.
        is_trapped: True if diagonal value is close to 1.0.
    """
    state: int
    diagonal_value: float
    is_trapped: bool


@dataclass(frozen=True)
class LivelockResult:
    """Comprehensive livelock analysis result.

    Attributes:
        is_livelock_free: True iff no trapped SCCs exist.
        livelocked_states: All states that are part of trapped SCCs.
        trapped_sccs: Details of each trapped SCC.
        num_trapped_sccs: Number of trapped SCCs.
        total_trapped_states: Total states in trapped SCCs.
        max_tropical_eigenvalue: Maximum tropical eigenvalue across trapped SCCs.
        heat_indicators: Heat kernel indicators per livelocked state.
        reachable_from_top: States reachable from top that are livelocked.
        num_states: Total states in the state space.
        num_transitions: Total transitions.
        summary: Human-readable summary.
    """
    is_livelock_free: bool
    livelocked_states: frozenset[int]
    trapped_sccs: tuple[LivelockSCC, ...]
    num_trapped_sccs: int
    total_trapped_states: int
    max_tropical_eigenvalue: float
    heat_indicators: tuple[HeatKernelLivelockIndicator, ...]
    reachable_from_top: frozenset[int]
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


def _compute_sccs(ss: "StateSpace") -> list[frozenset[int]]:
    """Compute SCCs using iterative Tarjan's algorithm."""
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    result: list[frozenset[int]] = []

    for start in sorted(ss.states):
        if start in index:
            continue
        call_stack: list[tuple[int, int]] = []
        index[start] = lowlink[start] = index_counter[0]
        index_counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        call_stack.append((start, 0))

        while call_stack:
            v, ni = call_stack[-1]
            neighbors = adj.get(v, [])
            if ni < len(neighbors):
                call_stack[-1] = (v, ni + 1)
                w = neighbors[ni]
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, 0))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                if lowlink[v] == index[v]:
                    scc_members: list[int] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc_members.append(w)
                        if w == v:
                            break
                    result.append(frozenset(scc_members))
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])

    return result


def _is_nontrivial_scc(scc: frozenset[int], ss: "StateSpace") -> bool:
    """Check if an SCC is non-trivial (has cycles: size > 1 or has self-loop)."""
    if len(scc) > 1:
        return True
    # Check for self-loop
    s = next(iter(scc))
    for src, _, tgt in ss.transitions:
        if src == s and tgt == s:
            return True
    return False


def _scc_can_reach_bottom(
    scc: frozenset[int],
    ss: "StateSpace",
    bottom_reachable: set[int],
) -> bool:
    """Check if any state in SCC can reach bottom via a path exiting the SCC."""
    for s in scc:
        for label, t in _outgoing(ss, s):
            if t not in scc and t in bottom_reachable:
                return True
    # Also check if bottom is inside the SCC
    if ss.bottom in scc:
        return True
    return False


def _tropical_eigenvalue_on_subgraph(
    states: frozenset[int],
    ss: "StateSpace",
) -> float:
    """Compute the tropical (max-plus) eigenvalue on a subgraph.

    The tropical eigenvalue is the maximum cycle mean:
        lambda = max over all cycles C of (weight(C) / length(C))

    For unit-weight edges, this is 1.0 if any cycle exists, 0.0 otherwise.
    We use Karp's algorithm for maximum cycle mean.
    """
    if len(states) <= 1:
        # Self-loop check
        s = next(iter(states))
        for src, _, tgt in ss.transitions:
            if src == s and tgt == s:
                return 1.0
        return 0.0

    state_list = sorted(states)
    n = len(state_list)
    idx = {s: i for i, s in enumerate(state_list)}

    # Build adjacency within the SCC
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for src, _, tgt in ss.transitions:
        if src in states and tgt in states:
            adj[idx[src]].append(idx[tgt])

    # Karp's algorithm: compute D[k][v] = max weight of a path of
    # exactly k edges ending at v (using unit weights, so weight = k).
    # lambda = max_v min_k (D[n][v] - D[k][v]) / (n - k)
    NEG_INF = float('-inf')
    D: list[list[float]] = [[NEG_INF] * n for _ in range(n + 1)]

    # Start from all vertices (take max over all starting vertices)
    for s in range(n):
        D[0][s] = 0.0

    for k in range(1, n + 1):
        for v in range(n):
            for u in range(n):
                if D[k - 1][u] != NEG_INF and v in adj.get(u, []):
                    val = D[k - 1][u] + 1.0  # unit weight
                    if val > D[k][v]:
                        D[k][v] = val

    # Compute max cycle mean
    max_mean = NEG_INF
    for v in range(n):
        if D[n][v] == NEG_INF:
            continue
        min_ratio = float('inf')
        for k in range(n):
            if D[k][v] == NEG_INF:
                continue
            ratio = (D[n][v] - D[k][v]) / (n - k)
            min_ratio = min(min_ratio, ratio)
        if min_ratio != float('inf'):
            max_mean = max(max_mean, min_ratio)

    return max_mean if max_mean != NEG_INF else 0.0


def _heat_kernel_diagonal(
    ss: "StateSpace",
    states_of_interest: frozenset[int],
    t: float,
) -> dict[int, float]:
    """Compute heat kernel diagonal H_t(s,s) for states of interest.

    Uses the random walk interpretation: H_t(s,s) is the probability of
    returning to state s after continuous-time random walk of duration t.

    For trapped SCCs, this approaches 1.0 as t -> infinity because
    probability mass cannot escape.

    We approximate using matrix exponential of the sub-Laplacian.
    For small matrices, use Taylor series: exp(-tL) = I - tL + t^2L^2/2! - ...
    """
    if not states_of_interest or not ss.states:
        return {}

    # Build adjacency matrix restricted to states of interest + their neighbors
    relevant = set(states_of_interest)
    state_list = sorted(relevant)
    n = len(state_list)
    idx = {s: i for i, s in enumerate(state_list)}

    if n == 0:
        return {}

    # Build symmetric adjacency (undirected) for Laplacian
    A = [[0.0] * n for _ in range(n)]
    for src, _, tgt in ss.transitions:
        if src in relevant and tgt in relevant:
            si, ti = idx[src], idx[tgt]
            A[si][ti] = 1.0
            A[ti][si] = 1.0  # symmetrize

    # Degree matrix
    D = [sum(A[i]) for i in range(n)]

    # Laplacian L = D - A
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i][j] = D[i] - A[i][j]
            else:
                L[i][j] = -A[i][j]

    # Compute exp(-tL) via Taylor series (truncated)
    # For small matrices, 20 terms is sufficient
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        H[i][i] = 1.0

    # -tL
    tL = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            tL[i][j] = -t * L[i][j]

    # Taylor: H = I + tL + (tL)^2/2! + ...
    term = [[0.0] * n for _ in range(n)]
    for i in range(n):
        term[i][i] = 1.0

    for k in range(1, 20):
        new_term = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for p in range(n):
                    new_term[i][j] += term[i][p] * tL[p][j]
        for i in range(n):
            for j in range(n):
                new_term[i][j] /= k
        term = new_term
        for i in range(n):
            for j in range(n):
                H[i][j] += term[i][j]

    result: dict[int, float] = {}
    for s in states_of_interest:
        if s in idx:
            i = idx[s]
            result[s] = max(0.0, min(1.0, H[i][i]))
        else:
            result[s] = 0.0

    return result


def _get_scc_labels(scc: frozenset[int], ss: "StateSpace") -> tuple[str, ...]:
    """Get the transition labels within an SCC."""
    labels: set[str] = set()
    for src, label, tgt in ss.transitions:
        if src in scc and tgt in scc:
            labels.add(label)
    return tuple(sorted(labels))


def _get_entry_transitions(
    scc: frozenset[int],
    ss: "StateSpace",
) -> tuple[tuple[int, str, int], ...]:
    """Get transitions from outside the SCC into it."""
    entries: list[tuple[int, str, int]] = []
    for src, label, tgt in ss.transitions:
        if src not in scc and tgt in scc:
            entries.append((src, label, tgt))
    return tuple(sorted(entries))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_livelock(ss: "StateSpace") -> frozenset[int]:
    """Find all livelocked states: states in SCCs with no exit to bottom.

    A state is livelocked if it belongs to a non-trivial SCC (has cycles)
    and no state in that SCC can reach bottom.

    Returns:
        Frozenset of livelocked state IDs.
    """
    sccs = _compute_sccs(ss)
    bottom_reachable = _reverse_reachable(ss, ss.bottom)

    livelocked: set[int] = set()
    for scc in sccs:
        if not _is_nontrivial_scc(scc, ss):
            continue
        if not _scc_can_reach_bottom(scc, ss, bottom_reachable):
            livelocked |= scc

    return frozenset(livelocked)


def livelock_sccs(ss: "StateSpace") -> list[frozenset[int]]:
    """Return the SCCs that are livelocked (trapped cycles).

    An SCC is livelocked if it is non-trivial (has cycles) and no state
    in it can reach bottom via a path that exits the SCC.

    Returns:
        List of frozensets, each a trapped SCC.
    """
    sccs = _compute_sccs(ss)
    bottom_reachable = _reverse_reachable(ss, ss.bottom)

    trapped: list[frozenset[int]] = []
    for scc in sccs:
        if not _is_nontrivial_scc(scc, ss):
            continue
        if not _scc_can_reach_bottom(scc, ss, bottom_reachable):
            trapped.append(scc)

    return trapped


def tropical_livelock_score(ss: "StateSpace") -> float:
    """Compute tropical eigenvalue on trapped SCCs.

    Returns the maximum tropical eigenvalue across all trapped SCCs.
    A positive score (> 0) indicates livelock: there is cyclic activity
    with no progress toward termination.

    Returns 0.0 if no trapped SCCs exist (livelock-free).
    """
    trapped = livelock_sccs(ss)
    if not trapped:
        return 0.0

    max_eigenvalue = 0.0
    for scc in trapped:
        ev = _tropical_eigenvalue_on_subgraph(scc, ss)
        max_eigenvalue = max(max_eigenvalue, ev)

    return max_eigenvalue


def heat_kernel_livelock(
    ss: "StateSpace",
    t: float = 10.0,
) -> list[HeatKernelLivelockIndicator]:
    """Heat kernel based livelock indicators.

    For each livelocked state, compute H_t(s,s). A value close to 1.0
    indicates that probability mass stays trapped at that state,
    confirming livelock.

    Args:
        ss: The state space to analyze.
        t: Time parameter for the heat kernel. Larger t makes trapped
           states more clearly identifiable (H_t -> 1.0).

    Returns:
        List of HeatKernelLivelockIndicator for each livelocked state.
    """
    livelocked = detect_livelock(ss)
    if not livelocked:
        return []

    diag = _heat_kernel_diagonal(ss, livelocked, t)

    indicators: list[HeatKernelLivelockIndicator] = []
    for s in sorted(livelocked):
        val = diag.get(s, 0.0)
        indicators.append(HeatKernelLivelockIndicator(
            state=s,
            diagonal_value=val,
            is_trapped=(val > 0.5),
        ))

    return indicators


def is_livelock_free(ss: "StateSpace") -> bool:
    """Return True iff no trapped SCCs exist.

    A state space is livelock-free if every non-trivial SCC has at least
    one exit path to the bottom state.
    """
    return len(livelock_sccs(ss)) == 0


def analyze_livelock(ss: "StateSpace") -> LivelockResult:
    """Comprehensive livelock analysis combining all techniques.

    Performs SCC decomposition, exit analysis, tropical eigenvalue
    computation, and heat kernel analysis.

    Returns:
        LivelockResult with complete analysis.
    """
    sccs = _compute_sccs(ss)
    bottom_reachable = _reverse_reachable(ss, ss.bottom)
    top_reachable = _reachable_from(ss, ss.top)

    trapped_scc_list: list[LivelockSCC] = []
    all_livelocked: set[int] = set()

    for scc in sccs:
        if not _is_nontrivial_scc(scc, ss):
            continue
        if _scc_can_reach_bottom(scc, ss, bottom_reachable):
            continue

        # This is a trapped SCC
        all_livelocked |= scc
        ev = _tropical_eigenvalue_on_subgraph(scc, ss)
        labels = _get_scc_labels(scc, ss)
        entries = _get_entry_transitions(scc, ss)

        trapped_scc_list.append(LivelockSCC(
            states=scc,
            entry_transitions=entries,
            tropical_eigenvalue=ev,
            cycle_labels=labels,
        ))

    livelocked_frozen = frozenset(all_livelocked)
    reachable_livelocked = frozenset(all_livelocked & top_reachable)

    # Heat kernel analysis
    heat_indicators: list[HeatKernelLivelockIndicator] = []
    if all_livelocked:
        diag = _heat_kernel_diagonal(ss, livelocked_frozen, 10.0)
        for s in sorted(all_livelocked):
            val = diag.get(s, 0.0)
            heat_indicators.append(HeatKernelLivelockIndicator(
                state=s,
                diagonal_value=val,
                is_trapped=(val > 0.5),
            ))

    max_ev = max((scc.tropical_eigenvalue for scc in trapped_scc_list), default=0.0)

    is_free = len(trapped_scc_list) == 0

    # Build summary
    if is_free:
        summary = "Livelock-free: all SCCs have exit paths to bottom."
    else:
        summary = (
            f"LIVELOCK DETECTED: {len(trapped_scc_list)} trapped SCC(s) "
            f"containing {len(all_livelocked)} state(s). "
            f"Max tropical eigenvalue: {max_ev:.2f}. "
            f"{len(reachable_livelocked)} livelocked state(s) reachable from top."
        )

    return LivelockResult(
        is_livelock_free=is_free,
        livelocked_states=livelocked_frozen,
        trapped_sccs=tuple(trapped_scc_list),
        num_trapped_sccs=len(trapped_scc_list),
        total_trapped_states=len(all_livelocked),
        max_tropical_eigenvalue=max_ev,
        heat_indicators=tuple(heat_indicators),
        reachable_from_top=reachable_livelocked,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        summary=summary,
    )
