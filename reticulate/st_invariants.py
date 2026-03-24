"""S-invariants and T-invariants for session-type Petri nets (Step 23b).

S-invariants (place invariants) characterise token conservation laws:
an S-invariant is a non-negative integer vector y such that y^T * C >= 0
(semi-positive invariant) or y^T * C = 0 (strict invariant).  A set of
places covered by an S-invariant forms a *state machine component* — a
subnet where every transition has exactly one input and one output place
from that component.

T-invariants (transition invariants) characterise reproducible firing
sequences: a T-invariant is a non-negative integer vector x such that
C * x = 0, meaning that firing the multiset of transitions x returns the
net to its original marking.  T-invariants represent cyclic (repeatable)
behaviours in the protocol.

For session-type Petri nets:
  - S-invariants reveal state machine decomposition (one component per
    orthogonal protocol dimension, e.g. each parallel branch).
  - T-invariants reveal the cyclic sub-protocols induced by recursion
    (rec X . S).  Non-recursive session types have no T-invariants.

Key results:
  - Every session-type net is covered by S-invariants (Theorem 1).
  - Parallel (||) introduces independent S-invariant components whose
    product reconstructs the full state space (Theorem 2).
  - T-invariants exist iff the session type contains productive recursion,
    and each T-invariant's support corresponds to a strongly connected
    component in the state space (Theorem 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.petri import PetriNet


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SInvariant:
    """An S-invariant (place invariant) of a Petri net.

    Attributes:
        weights: Mapping from place ID to integer coefficient.
        support: Set of place IDs with non-zero weight.
        is_strict: True if y^T * C = 0 (not just >= 0).
        token_count: The constant weighted token sum (if strict).
        is_semi_positive: True if all weights >= 0.
    """
    weights: dict[int, int]
    support: frozenset[int]
    is_strict: bool = True
    token_count: int | None = None
    is_semi_positive: bool = True


@dataclass(frozen=True)
class TInvariant:
    """A T-invariant (transition invariant) of a Petri net.

    Attributes:
        firing_counts: Mapping from transition ID to firing count.
        support: Set of transition IDs with non-zero count.
        is_realizable: True if the firing sequence can actually execute
                       from the initial marking (not just algebraically valid).
        transition_labels: Labels of the transitions in the support.
    """
    firing_counts: dict[int, int]
    support: frozenset[int]
    is_realizable: bool = False
    transition_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuralInvariantResult:
    """Result of full structural invariant analysis.

    Attributes:
        s_invariants: List of S-invariants found.
        t_invariants: List of T-invariants found.
        num_s_invariants: Number of S-invariants.
        num_t_invariants: Number of T-invariants.
        s_covered_places: Places covered by at least one S-invariant.
        t_covered_transitions: Transitions covered by at least one T-invariant.
        is_s_covered: True if every place is covered by an S-invariant.
        is_t_covered: True if every transition is covered by a T-invariant.
        is_consistent: True if there exists a T-invariant covering every
                       transition (the net can return to its initial marking).
        is_conservative: True if there exists an S-invariant covering every
                         place (weighted token count is bounded).
        incidence_matrix: The incidence matrix C[p][t].
        has_recursion: True if T-invariants exist (indicates cyclic behaviour).
    """
    s_invariants: list[SInvariant]
    t_invariants: list[TInvariant]
    num_s_invariants: int
    num_t_invariants: int
    s_covered_places: frozenset[int]
    t_covered_transitions: frozenset[int]
    is_s_covered: bool
    is_t_covered: bool
    is_consistent: bool
    is_conservative: bool
    incidence_matrix: dict[int, dict[int, int]]
    has_recursion: bool


# ---------------------------------------------------------------------------
# Incidence matrix (reuses place_invariants but kept self-contained)
# ---------------------------------------------------------------------------

def _build_incidence_matrix(
    net: "PetriNet",
) -> dict[int, dict[int, int]]:
    """Build incidence matrix C[p,t] = post(t)(p) - pre(t)(p)."""
    place_ids = sorted(net.places.keys())
    trans_ids = sorted(net.transitions.keys())

    matrix: dict[int, dict[int, int]] = {p: {} for p in place_ids}

    for tid in trans_ids:
        pre_map: dict[int, int] = {}
        for pid, w in net.pre.get(tid, set()):
            pre_map[pid] = pre_map.get(pid, 0) + w

        post_map: dict[int, int] = {}
        for pid, w in net.post.get(tid, set()):
            post_map[pid] = post_map.get(pid, 0) + w

        all_places = set(pre_map.keys()) | set(post_map.keys())
        for pid in all_places:
            val = post_map.get(pid, 0) - pre_map.get(pid, 0)
            if val != 0:
                matrix[pid][tid] = val

    return matrix


def _matrix_to_dense(
    matrix: dict[int, dict[int, int]],
    row_ids: list[int],
    col_ids: list[int],
) -> list[list[int]]:
    """Convert sparse matrix to dense list-of-lists."""
    rows: list[list[int]] = []
    for r in row_ids:
        row = []
        for c in col_ids:
            row.append(matrix.get(r, {}).get(c, 0))
        rows.append(row)
    return rows


def _transpose_dense(rows: list[list[int]]) -> list[list[int]]:
    """Transpose a dense matrix."""
    if not rows:
        return []
    ncols = len(rows[0])
    return [[rows[r][c] for r in range(len(rows))] for c in range(ncols)]


# ---------------------------------------------------------------------------
# Integer null space computation
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    """Least common multiple."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def _integer_null_space(rows: list[list[int]], ncols: int) -> list[list[int]]:
    """Compute null space of integer matrix via Gaussian elimination.

    Given m x n matrix A, find all x in Z^n with A * x = 0.
    Returns basis vectors.
    """
    if not rows:
        basis = []
        for i in range(ncols):
            v = [0] * ncols
            v[i] = 1
            basis.append(v)
        return basis

    nrows = len(rows)
    mat = [row[:] for row in rows]

    pivot_row_for_col: dict[int, int] = {}

    current_row = 0
    for col in range(ncols):
        if current_row >= nrows:
            break

        # Find pivot
        pivot = -1
        for r in range(current_row, nrows):
            if mat[r][col] != 0:
                pivot = r
                break

        if pivot == -1:
            continue

        if pivot != current_row:
            mat[pivot], mat[current_row] = mat[current_row], mat[pivot]

        pivot_row_for_col[col] = current_row

        # Eliminate
        pivot_val = mat[current_row][col]
        for r in range(nrows):
            if r == current_row or mat[r][col] == 0:
                continue
            factor = mat[r][col]
            for c in range(ncols):
                mat[r][c] = mat[r][c] * pivot_val - factor * mat[current_row][c]
            # Reduce by GCD
            row_gcd = 0
            for c in range(ncols):
                if mat[r][c] != 0:
                    row_gcd = gcd(row_gcd, abs(mat[r][c]))
            if row_gcd > 1:
                for c in range(ncols):
                    mat[r][c] //= row_gcd

        current_row += 1

    # Free variables
    pivot_cols = set(pivot_row_for_col.keys())
    free_cols = [c for c in range(ncols) if c not in pivot_cols]

    basis: list[list[int]] = []
    for fc in free_cols:
        vec = [0] * ncols
        vec[fc] = 1

        scale = 1
        for pc, r in pivot_row_for_col.items():
            if mat[r][fc] != 0:
                scale = _lcm(scale, abs(mat[r][pc]))

        vec[fc] = scale
        for pc, r in pivot_row_for_col.items():
            if mat[r][fc] != 0:
                vec[pc] = -(mat[r][fc] * (scale // mat[r][pc]))

        # Normalize
        vec_gcd = 0
        for v in vec:
            if v != 0:
                vec_gcd = gcd(vec_gcd, abs(v))
        if vec_gcd > 1:
            vec = [v // vec_gcd for v in vec]

        # Make first non-zero positive
        for v in vec:
            if v != 0:
                if v < 0:
                    vec = [-x for x in vec]
                break

        basis.append(vec)

    return basis


# ---------------------------------------------------------------------------
# S-invariant computation
# ---------------------------------------------------------------------------

def compute_s_invariants(net: "PetriNet") -> list[SInvariant]:
    """Compute S-invariants (place invariants) of the net.

    An S-invariant is a vector y such that y^T * C = 0.
    This means the weighted sum of tokens is conserved across all firings.

    For session-type state-machine nets, the all-ones vector is always
    an S-invariant (exactly one token at any time).

    For parallel compositions, independent S-invariant components emerge
    corresponding to the orthogonal protocol dimensions.

    Args:
        net: A PetriNet.

    Returns:
        List of SInvariant objects.
    """
    place_ids = sorted(net.places.keys())
    trans_ids = sorted(net.transitions.keys())

    if not place_ids:
        return []

    if not trans_ids:
        # No transitions: every unit vector is an invariant
        result: list[SInvariant] = []
        for pid in place_ids:
            weights = {pid: 1}
            tc = net.initial_marking.get(pid, 0)
            result.append(SInvariant(
                weights=weights,
                support=frozenset([pid]),
                is_strict=True,
                token_count=tc,
                is_semi_positive=True,
            ))
        return result

    matrix = _build_incidence_matrix(net)

    # Null space of C^T gives S-invariants (left null space of C)
    ct_rows = []
    for t in trans_ids:
        row = []
        for p in place_ids:
            row.append(matrix.get(p, {}).get(t, 0))
        ct_rows.append(row)

    basis = _integer_null_space(ct_rows, len(place_ids))

    result = []
    for vec in basis:
        weights: dict[int, int] = {}
        for i, pid in enumerate(place_ids):
            if vec[i] != 0:
                weights[pid] = vec[i]
        if not weights:
            continue

        support = frozenset(weights.keys())
        is_semi_pos = all(w >= 0 for w in weights.values())

        # Check strictness: y^T * C = 0 for all transitions
        is_strict = True
        for tid in trans_ids:
            dot = 0
            for pid, w in weights.items():
                dot += w * matrix.get(pid, {}).get(tid, 0)
            if dot != 0:
                is_strict = False
                break

        # Compute token count from initial marking
        tc: int | None = None
        if is_strict:
            tc = sum(
                weights.get(pid, 0) * cnt
                for pid, cnt in net.initial_marking.items()
            )

        result.append(SInvariant(
            weights=weights,
            support=support,
            is_strict=is_strict,
            token_count=tc,
            is_semi_positive=is_semi_pos,
        ))

    return result


def verify_s_invariant(
    net: "PetriNet",
    invariant: SInvariant,
) -> bool:
    """Verify an S-invariant against reachable markings.

    Checks that the weighted token sum is constant across all
    reachable markings.
    """
    from reticulate.petri import build_reachability_graph, _thaw_marking

    rg = build_reachability_graph(net)
    expected: int | None = None
    for fm in rg.markings:
        marking = _thaw_marking(fm)
        total = sum(
            invariant.weights.get(pid, 0) * count
            for pid, count in marking.items()
        )
        if expected is None:
            expected = total
        elif total != expected:
            return False
    return True


# ---------------------------------------------------------------------------
# T-invariant computation
# ---------------------------------------------------------------------------

def compute_t_invariants(net: "PetriNet") -> list[TInvariant]:
    """Compute T-invariants (transition invariants) of the net.

    A T-invariant is a non-negative integer vector x such that C * x = 0,
    meaning that firing the multiset of transitions x returns the marking
    to its original state.

    For session-type nets:
      - Non-recursive types have no T-invariants (acyclic nets).
      - Recursive types (rec X . S) produce T-invariants whose support
        corresponds to the transitions in the recursive loop.

    Args:
        net: A PetriNet.

    Returns:
        List of TInvariant objects.
    """
    place_ids = sorted(net.places.keys())
    trans_ids = sorted(net.transitions.keys())

    if not trans_ids:
        return []

    if not place_ids:
        # No places: every transition vector is a T-invariant
        result: list[TInvariant] = []
        for tid in trans_ids:
            counts = {tid: 1}
            labels = (net.transitions[tid].label,)
            result.append(TInvariant(
                firing_counts=counts,
                support=frozenset([tid]),
                is_realizable=True,
                transition_labels=labels,
            ))
        return result

    matrix = _build_incidence_matrix(net)

    # C * x = 0 means x is in the null space of C (rows=places, cols=transitions)
    c_rows = _matrix_to_dense(matrix, place_ids, trans_ids)

    basis = _integer_null_space(c_rows, len(trans_ids))

    result: list[TInvariant] = []
    for vec in basis:
        # T-invariants must have all non-negative components
        # (they represent multisets of transition firings).
        # Skip vectors with negative entries — these are algebraic
        # null-space vectors but not proper T-invariants.
        if any(v < 0 for v in vec):
            continue

        counts: dict[int, int] = {}
        for i, tid in enumerate(trans_ids):
            if vec[i] != 0:
                counts[tid] = vec[i]
        if not counts:
            continue

        support = frozenset(counts.keys())
        labels = tuple(
            net.transitions[tid].label
            for tid in sorted(support)
        )

        # Check realizability: can this firing sequence actually execute?
        is_realizable = _check_realizability(net, counts)

        result.append(TInvariant(
            firing_counts=counts,
            support=support,
            is_realizable=is_realizable,
            transition_labels=labels,
        ))

    return result


def _check_realizability(
    net: "PetriNet",
    firing_counts: dict[int, int],
) -> bool:
    """Check if a T-invariant is realizable from the initial marking.

    A T-invariant is realizable if there exists an ordering of transitions
    (respecting multiplicities) that can fire from the initial marking
    and returns to it.

    Uses a greedy approach: repeatedly fire any enabled transition from
    the remaining multiset until either all are fired or stuck.
    """
    from reticulate.petri import fire, is_enabled

    remaining = dict(firing_counts)
    # Only consider non-negative counts
    for tid in list(remaining):
        if remaining[tid] <= 0:
            del remaining[tid]

    if not remaining:
        return True

    marking = dict(net.initial_marking)
    max_attempts = sum(remaining.values()) * 2  # prevent infinite loop

    attempts = 0
    while remaining and attempts < max_attempts:
        fired_any = False
        for tid in sorted(remaining.keys()):
            if remaining[tid] <= 0:
                continue
            if is_enabled(net, marking, tid):
                result = fire(net, marking, tid)
                if result.enabled and result.new_marking is not None:
                    marking = result.new_marking
                    remaining[tid] -= 1
                    if remaining[tid] == 0:
                        del remaining[tid]
                    fired_any = True
                    break
        if not fired_any:
            return False
        attempts += 1

    return len(remaining) == 0


def verify_t_invariant(
    net: "PetriNet",
    invariant: TInvariant,
) -> bool:
    """Verify a T-invariant: C * x should equal zero vector.

    This is an algebraic check (does not require reachability).
    """
    place_ids = sorted(net.places.keys())
    matrix = _build_incidence_matrix(net)

    for pid in place_ids:
        dot = 0
        for tid, count in invariant.firing_counts.items():
            dot += matrix.get(pid, {}).get(tid, 0) * count
        if dot != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_structural_invariants(
    net: "PetriNet",
) -> StructuralInvariantResult:
    """Full structural invariant analysis of a session-type Petri net.

    Computes both S-invariants (place invariants) and T-invariants
    (transition invariants), checks coverage and structural properties.

    Key structural properties:
      - Conservative: every place covered by an S-invariant (token-bounded).
      - Consistent: every transition covered by a T-invariant (reproducible).
      - S-covered + T-covered together imply liveness for free-choice nets.

    Args:
        net: A PetriNet (from petri.build_petri_net).

    Returns:
        StructuralInvariantResult with all invariants and coverage info.
    """
    matrix = _build_incidence_matrix(net)

    s_invs = compute_s_invariants(net)
    t_invs = compute_t_invariants(net)

    # S-coverage
    s_covered: set[int] = set()
    for inv in s_invs:
        s_covered.update(inv.support)
    s_covered_frozen = frozenset(s_covered)
    is_s_covered = (len(s_covered_frozen) == len(net.places))

    # T-coverage
    t_covered: set[int] = set()
    for inv in t_invs:
        t_covered.update(inv.support)
    t_covered_frozen = frozenset(t_covered)
    is_t_covered = (len(t_covered_frozen) == len(net.transitions))

    # Conservative: exists S-invariant covering all places
    is_conservative = is_s_covered and len(s_invs) > 0

    # Consistent: exists T-invariant covering all transitions
    is_consistent = is_t_covered and len(t_invs) > 0

    has_recursion = len(t_invs) > 0

    return StructuralInvariantResult(
        s_invariants=s_invs,
        t_invariants=t_invs,
        num_s_invariants=len(s_invs),
        num_t_invariants=len(t_invs),
        s_covered_places=s_covered_frozen,
        t_covered_transitions=t_covered_frozen,
        is_s_covered=is_s_covered,
        is_t_covered=is_t_covered,
        is_consistent=is_consistent,
        is_conservative=is_conservative,
        incidence_matrix=matrix,
        has_recursion=has_recursion,
    )
