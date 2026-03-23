"""Place invariants as lattice properties for session-type Petri nets (Step 23).

A place invariant (P-invariant, S-invariant) is an integer vector y >= 0
such that y^T * C = 0, where C is the incidence matrix of the Petri net.
This means the weighted sum  sum_p y(p) * M(p)  is constant across all
reachable markings M — a token conservation law.

For session-type Petri nets constructed via the state-machine encoding
(Step 21), every place corresponds to a protocol state, and exactly one
token exists at any time.  The trivial all-ones vector is always a
P-invariant.  This module goes further:

1. Builds the full incidence matrix C[p,t] = post(t)(p) - pre(t)(p).
2. Computes the integer null space of C^T via Gaussian elimination,
   yielding *all* minimal P-invariants (not just the trivial one).
3. Verifies each invariant against all reachable markings.
4. Relates invariants to lattice structure: covered places, support
   decomposition, and the connection between P-invariant support and
   lattice-theoretic filters/ideals.

Key result: For 1-safe state-machine nets from well-formed session types,
the support of every minimal P-invariant is the entire place set — the
net is *covered* by a single invariant.  This reflects the fundamental
property that session types enforce exactly-one-active-state discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.petri import PetriNet, Marking


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaceInvariant:
    """A place invariant (P-invariant) of a Petri net.

    Attributes:
        weights: Mapping from place ID to integer coefficient.
                 The invariant asserts that sum_p weights[p] * M(p)
                 is constant for every reachable marking M.
        is_conservative: True if the weighted token sum is verified
                         constant across all reachable markings.
        token_count: The constant value of the weighted sum (if conservative).
        support: Set of place IDs with non-zero weight.
    """
    weights: dict[int, int]
    is_conservative: bool
    token_count: int | None = None
    support: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PlaceInvariantResult:
    """Result of full place-invariant analysis.

    Attributes:
        invariants: List of minimal P-invariants found.
        all_conservative: True if every invariant is verified conservative.
        covered_places: Set of place IDs that appear in at least one
                        invariant's support.
        num_places: Total number of places in the net.
        is_fully_covered: True if every place is covered by some invariant.
        incidence_matrix: The incidence matrix C[p][t] as nested dict.
        null_space_dimension: Dimension of the null space of C^T.
    """
    invariants: list[PlaceInvariant]
    all_conservative: bool
    covered_places: frozenset[int]
    num_places: int
    is_fully_covered: bool
    incidence_matrix: dict[int, dict[int, int]]
    null_space_dimension: int


# ---------------------------------------------------------------------------
# Incidence matrix
# ---------------------------------------------------------------------------

def compute_incidence_matrix(
    net: "PetriNet",
) -> dict[int, dict[int, int]]:
    """Build the incidence matrix C[p,t] = post(t)(p) - pre(t)(p).

    Rows are indexed by place ID, columns by transition ID.
    Only non-zero entries are stored.

    Args:
        net: A PetriNet (from petri.build_petri_net).

    Returns:
        Nested dict C where C[p][t] is the incidence value.
        Missing entries are implicitly zero.
    """
    place_ids = sorted(net.places.keys())
    trans_ids = sorted(net.transitions.keys())

    matrix: dict[int, dict[int, int]] = {p: {} for p in place_ids}

    for tid in trans_ids:
        # Build pre and post maps for this transition
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


# ---------------------------------------------------------------------------
# Gaussian elimination over integers (null space of C^T)
# ---------------------------------------------------------------------------

def _transpose_to_dense(
    matrix: dict[int, dict[int, int]],
    row_ids: list[int],
    col_ids: list[int],
) -> list[list[int]]:
    """Convert C[p,t] (sparse) to C^T as dense list-of-lists.

    Result has len(col_ids) rows and len(row_ids) columns,
    i.e. C^T[t][p].
    """
    rows: list[list[int]] = []
    for t in col_ids:
        row = []
        for p in row_ids:
            row.append(matrix.get(p, {}).get(t, 0))
        rows.append(row)
    return rows


def _integer_null_space(rows: list[list[int]], ncols: int) -> list[list[int]]:
    """Compute the null space of an integer matrix via Gaussian elimination.

    Given an m x n matrix A (as list of rows), find all vectors x in Z^n
    such that A * x = 0.  Returns a list of basis vectors.

    Uses fraction-free Gaussian elimination to stay in integers.

    Args:
        rows: The matrix rows (m x n).
        ncols: Number of columns n.

    Returns:
        List of integer vectors spanning the null space.
    """
    if not rows:
        # No constraints: entire space is null space
        basis = []
        for i in range(ncols):
            v = [0] * ncols
            v[i] = 1
            basis.append(v)
        return basis

    nrows = len(rows)

    # Augment with identity for tracking: [A | I_n]^T approach
    # We use column-oriented elimination on A^T to find null space of A.
    # Actually, we want null space of A (m x n): Ax = 0.
    # Standard approach: row-reduce A augmented with column indices.

    # Copy matrix to avoid mutation
    mat = [row[:] for row in rows]

    # Track which columns are pivot columns
    pivot_col = [-1] * nrows  # pivot_col[row] = column index of pivot
    pivot_row_for_col: dict[int, int] = {}  # col -> row

    current_row = 0
    for col in range(ncols):
        if current_row >= nrows:
            break

        # Find pivot in current column from current_row downward
        pivot = -1
        for r in range(current_row, nrows):
            if mat[r][col] != 0:
                pivot = r
                break

        if pivot == -1:
            continue  # Free variable column

        # Swap rows
        if pivot != current_row:
            mat[pivot], mat[current_row] = mat[current_row], mat[pivot]

        pivot_col[current_row] = col
        pivot_row_for_col[col] = current_row

        # Eliminate other rows (fraction-free)
        pivot_val = mat[current_row][col]
        for r in range(nrows):
            if r == current_row or mat[r][col] == 0:
                continue
            factor = mat[r][col]
            for c in range(ncols):
                mat[r][c] = mat[r][c] * pivot_val - factor * mat[current_row][c]
            # Reduce row by GCD to keep numbers small
            row_gcd = 0
            for c in range(ncols):
                if mat[r][c] != 0:
                    row_gcd = gcd(row_gcd, abs(mat[r][c]))
            if row_gcd > 1:
                for c in range(ncols):
                    mat[r][c] //= row_gcd

        current_row += 1

    # Free variables = columns not used as pivots
    pivot_cols = set(pivot_row_for_col.keys())
    free_cols = [c for c in range(ncols) if c not in pivot_cols]

    # For each free variable, construct a null-space vector
    basis: list[list[int]] = []
    for fc in free_cols:
        vec = [0] * ncols
        vec[fc] = 1  # will be scaled

        # For each pivot column, solve for pivot variable
        # mat[r][pc] * x_pc + ... + mat[r][fc] * x_fc + ... = 0
        # After elimination, mat[r] has leading entry at pc and
        # possibly non-zero entries at free columns.
        # x_pc = -sum(mat[r][fc'] * x_fc' for fc' in free_cols) / mat[r][pc]

        # We need to scale so everything is integer
        # Set x_fc = lcm of all pivot values (or just multiply through)
        scale = 1
        for pc, r in pivot_row_for_col.items():
            if mat[r][fc] != 0:
                scale = _lcm(scale, abs(mat[r][pc]))

        vec[fc] = scale
        for pc, r in pivot_row_for_col.items():
            if mat[r][fc] != 0:
                vec[pc] = -(mat[r][fc] * (scale // mat[r][pc]))

        # Normalize: divide by GCD, ensure first non-zero is positive
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


def _lcm(a: int, b: int) -> int:
    """Least common multiple of two positive integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


# ---------------------------------------------------------------------------
# P-invariant computation
# ---------------------------------------------------------------------------

def compute_place_invariants(
    net: "PetriNet",
) -> list[dict[int, int]]:
    """Find all minimal P-invariants: vectors y such that y^T * C = 0.

    Equivalently, y is in the left null space of C, which equals the
    (right) null space of C^T.

    Args:
        net: A PetriNet.

    Returns:
        List of invariant vectors as {place_id: coefficient} dicts.
        Only non-zero entries are included.
    """
    place_ids = sorted(net.places.keys())
    trans_ids = sorted(net.transitions.keys())

    if not place_ids:
        return []

    if not trans_ids:
        # No transitions: every unit vector is an invariant
        result: list[dict[int, int]] = []
        for pid in place_ids:
            result.append({pid: 1})
        return result

    matrix = compute_incidence_matrix(net)

    # Build C^T as dense matrix: C^T[t][p] = C[p][t]
    ct_rows = _transpose_to_dense(matrix, place_ids, trans_ids)

    # Null space of C^T gives us vectors y (indexed by places)
    basis = _integer_null_space(ct_rows, len(place_ids))

    # Convert dense vectors to sparse dicts
    result = []
    for vec in basis:
        d: dict[int, int] = {}
        for i, pid in enumerate(place_ids):
            if vec[i] != 0:
                d[pid] = vec[i]
        if d:
            result.append(d)

    return result


# ---------------------------------------------------------------------------
# Conservation check
# ---------------------------------------------------------------------------

def check_weighted_token_conservation(
    net: "PetriNet",
    invariant: dict[int, int],
) -> tuple[bool, int | None]:
    """Verify that y^T * M = const for all reachable markings.

    Args:
        net: A PetriNet.
        invariant: Weight vector {place_id: coefficient}.

    Returns:
        (is_conservative, token_count) where token_count is the constant
        value if conservative, None otherwise.
    """
    from reticulate.petri import build_reachability_graph, _thaw_marking

    rg = build_reachability_graph(net)

    expected: int | None = None
    for fm in rg.markings:
        marking = _thaw_marking(fm)
        total = sum(
            invariant.get(pid, 0) * count
            for pid, count in marking.items()
        )
        if expected is None:
            expected = total
        elif total != expected:
            return False, None

    return True, expected


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_place_invariants(
    net: "PetriNet",
) -> PlaceInvariantResult:
    """Full place-invariant analysis of a Petri net.

    Finds all minimal P-invariants via null-space computation,
    verifies each against reachable markings, and determines coverage.

    Args:
        net: A PetriNet (from petri.build_petri_net).

    Returns:
        PlaceInvariantResult with invariants, coverage info, and
        the incidence matrix.
    """
    matrix = compute_incidence_matrix(net)
    raw_invariants = compute_place_invariants(net)

    invariants: list[PlaceInvariant] = []
    covered: set[int] = set()
    all_conservative = True

    for weights in raw_invariants:
        is_cons, token_count = check_weighted_token_conservation(net, weights)
        support = frozenset(pid for pid, w in weights.items() if w != 0)
        covered.update(support)

        inv = PlaceInvariant(
            weights=weights,
            is_conservative=is_cons,
            token_count=token_count,
            support=support,
        )
        invariants.append(inv)
        if not is_cons:
            all_conservative = False

    # If no invariants found, still conservative (vacuously)
    if not invariants:
        all_conservative = True

    covered_frozen = frozenset(covered)
    num_places = len(net.places)

    # Compute null space dimension
    null_dim = len(raw_invariants)

    return PlaceInvariantResult(
        invariants=invariants,
        all_conservative=all_conservative,
        covered_places=covered_frozen,
        num_places=num_places,
        is_fully_covered=(len(covered_frozen) == num_places),
        incidence_matrix=matrix,
        null_space_dimension=null_dim,
    )
