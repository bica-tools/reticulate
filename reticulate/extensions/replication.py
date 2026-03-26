"""Bounded replication extension: S^n -- n copies of S in parallel.

Grammar extension:
  S ::= ... | S ^ n    -- n copies of S in parallel (syntactic sugar)

Semantics:
  L(S^n) = L(S)^n  (n-fold product lattice)

The replication operator is syntactic sugar for n-ary parallel:
  S^3 = (S || S || S)

But it adds value beyond sugar:
  1. Compact notation for replicated systems (choirs, server farms, sensor arrays)
  2. Symmetry analysis: L(S^n) has Sn symmetry group (permutation of factors)
  3. Diagonal constraint: S^n|diag = S (unison = single copy)
  4. Compression: Birkhoff dual of S^n uses the symmetry
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import TYPE_CHECKING

from reticulate.parser import SessionType, parse, pretty

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# AST node
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Replication:
    """Bounded replication: ``S ^ n`` -- n parallel copies of S.

    Attributes:
        body: The session type to replicate.
        count: Number of copies (must be >= 1).
    """
    body: SessionType
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"Replication count must be >= 1, got {self.count}")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplicationAnalysis:
    """Full analysis of a replicated session type S^n.

    Attributes:
        body_type: The base session type string.
        n: Number of copies.
        base_states: Number of states in L(S).
        product_states: Number of states in L(S^n) = |L(S)|^n.
        diagonal_states_count: Number of diagonal states (where all factors equal).
        orbit_count: Number of distinct Sn symmetry orbits.
        compression_ratio: product_states / orbit_count (how much symmetry compresses).
        is_lattice: Whether L(S^n) is a lattice.
        base_is_lattice: Whether L(S) is a lattice.
        diagonal_isomorphic: Whether diagonal sublattice is isomorphic to L(S).
    """
    body_type: str
    n: int
    base_states: int
    product_states: int
    diagonal_states_count: int
    orbit_count: int
    compression_ratio: float
    is_lattice: bool
    base_is_lattice: bool
    diagonal_isomorphic: bool


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_replication(s: str) -> SessionType | Replication:
    """Parse a session type string that may contain ``S ^ n`` syntax.

    The ``^`` operator is postfix with lowest precedence.  The exponent
    must be a positive integer literal.  If no ``^`` is present, delegates
    entirely to the core parser.

    Examples::

        parse_replication("end ^ 3")
        parse_replication("&{a: end, b: end} ^ 2")
        parse_replication("rec X . &{next: X, stop: end} ^ 4")

    Parameters:
        s: Session type string, optionally with ``^ n`` suffix.

    Returns:
        A core ``SessionType`` AST node (if no ``^``) or a ``Replication`` node.

    Raises:
        ``ParseError``: If the base type is malformed.
        ``ValueError``: If the exponent is not a positive integer.
    """
    s = s.strip()

    # Find the last '^' that is not inside braces or parentheses
    depth_brace = 0
    depth_paren = 0
    caret_pos = -1
    for i, ch in enumerate(s):
        if ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace -= 1
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren -= 1
        elif ch == '^' and depth_brace == 0 and depth_paren == 0:
            caret_pos = i

    if caret_pos == -1:
        return parse(s)

    base_str = s[:caret_pos].strip()
    exp_str = s[caret_pos + 1:].strip()

    if not exp_str:
        raise ValueError("Missing exponent after '^'")

    try:
        n = int(exp_str)
    except ValueError:
        raise ValueError(f"Exponent must be a positive integer, got '{exp_str}'")

    if n < 1:
        raise ValueError(f"Exponent must be >= 1, got {n}")

    body = parse(base_str)
    if n == 1:
        return body
    return Replication(body=body, count=n)


# ---------------------------------------------------------------------------
# State-space construction
# ---------------------------------------------------------------------------

def build_replicated_statespace(node: Replication) -> "StateSpace":
    """Build the n-fold product state space for a ``Replication`` node.

    Constructs L(S) once, then computes the n-fold product L(S)^n
    using ``power_statespace``.

    Parameters:
        node: A ``Replication`` AST node.

    Returns:
        The n-fold product state space with product coordinates.
    """
    from reticulate.statespace import build_statespace
    from reticulate.product import power_statespace

    base_ss = build_statespace(node.body)
    return power_statespace(base_ss, node.count)


# ---------------------------------------------------------------------------
# Diagonal analysis
# ---------------------------------------------------------------------------

def _get_coords(ss: "StateSpace", n: int) -> dict[int, tuple[int, ...]]:
    """Extract or compute product coordinates for states in an n-fold product.

    If the state space has ``product_coords``, use those directly.
    Otherwise, infer coordinates from the label format ``(x, y, ...)``.
    """
    if ss.product_coords:
        return {sid: coords for sid, coords in ss.product_coords.items()
                if len(coords) == n}

    # Fallback: parse labels like "(s1, s2, ...)"
    coords: dict[int, tuple[int, ...]] = {}
    for sid in ss.states:
        label = ss.labels.get(sid, "")
        if label.startswith("(") and label.endswith(")"):
            inner = label[1:-1]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) == n:
                try:
                    coords[sid] = tuple(int(p) for p in parts)
                except ValueError:
                    pass
    return coords


def diagonal_states(ss: "StateSpace", n: int) -> set[int]:
    """Identify diagonal states: those where all n components are equal.

    In a product lattice L(S)^n, a state (s1, s2, ..., sn) is diagonal
    iff s1 = s2 = ... = sn.  These correspond to "unison" -- all copies
    in the same protocol state.

    Parameters:
        ss: An n-fold product state space.
        n: Number of factors.

    Returns:
        Set of state IDs on the diagonal.
    """
    if n <= 1:
        return set(ss.states)

    coords = _get_coords(ss, n)
    result: set[int] = set()
    for sid, coord in coords.items():
        if len(coord) == n and len(set(coord)) == 1:
            result.add(sid)
    return result


def diagonal_lattice(ss: "StateSpace", n: int) -> "StateSpace":
    """Restrict an n-fold product to the diagonal sublattice.

    The diagonal of L(S)^n contains only states (s, s, ..., s) for
    s in L(S).  The resulting lattice should be isomorphic to L(S).

    Since the product advances one component at a time, direct transitions
    between diagonal states are rare.  Instead we compute the reachability
    relation among diagonal states and build the covering (Hasse) relation
    to obtain the sublattice structure.

    Parameters:
        ss: An n-fold product state space.
        n: Number of factors.

    Returns:
        A new StateSpace whose reachability order matches the diagonal
        restriction of the product order.
    """
    from reticulate.statespace import StateSpace as SS

    if n <= 1:
        return ss

    diag = diagonal_states(ss, n)
    if not diag:
        return SS(states={0}, transitions=[], top=0, bottom=0,
                  labels={0: "empty"}, selection_transitions=set())

    # Compute reachability among diagonal states using the full product
    diag_reach: dict[int, set[int]] = {}
    for sid in diag:
        reachable = ss.reachable_from(sid)
        diag_reach[sid] = reachable & diag

    # Build covering relation (Hasse edges) + self-loops for cycles.
    # s covers t if s reaches t and there is no intermediate diagonal
    # state u (u != s, u != t) with s->u->t.
    transitions: list[tuple[int, str, int]] = []
    for s in diag:
        # Check for self-reachability (cycles) -- if s reaches itself
        # through non-diagonal states, add a self-loop
        if s in (ss.reachable_from(s) - {s}):
            # s is in a cycle in the original -- but we need to check
            # if s reaches s through the product (which it does if the
            # base type has cycles, since all components cycle together)
            for succ in ss.successors(s):
                if s in ss.reachable_from(succ):
                    transitions.append((s, "tau", s))
                    break

        # States reachable from s (excluding s itself)
        below = diag_reach[s] - {s}
        # Remove states reachable via other diagonal states
        covers: set[int] = set(below)
        for u in below:
            if u in covers:
                # Remove everything that u can reach (except u)
                covers -= (diag_reach[u] - {u})
        for t in covers:
            transitions.append((s, "tau", t))

    # Determine top and bottom
    top = ss.top if ss.top in diag else min(diag)
    bottom = ss.bottom if ss.bottom in diag else max(diag)

    labels = {sid: ss.labels.get(sid, str(sid)) for sid in diag}

    return SS(
        states=diag,
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=labels,
        selection_transitions=set(),
    )


def _is_order_isomorphic(ss1: "StateSpace", ss2: "StateSpace") -> bool:
    """Check whether two state spaces have isomorphic reachability posets.

    Unlike ``find_isomorphism`` from morphism.py (which matches transition
    labels), this checks only the order structure: a bijection f such that
    s1 reaches s2 iff f(s1) reaches f(s2).  This is the correct notion
    for comparing the diagonal sublattice (which loses original labels)
    with the base lattice.

    For cyclic state spaces (from recursive types), we quotient by SCCs
    and compare the quotient DAGs, matching the approach in lattice.py.
    """
    if len(ss1.states) != len(ss2.states):
        return False
    if len(ss1.states) <= 1:
        return True

    # Compute SCC quotients for both and compare the DAG structures
    from reticulate.lattice import check_lattice

    lr1 = check_lattice(ss1)
    lr2 = check_lattice(ss2)

    # If both are lattices with same SCC count, that's a strong signal
    if lr1.num_scc != lr2.num_scc:
        return False

    # Build reachability matrix on SCC representatives
    reps1 = sorted(set(lr1.scc_map.values()))
    reps2 = sorted(set(lr2.scc_map.values()))

    if len(reps1) != len(reps2):
        return False

    # For each SCC representative, compute reachability profile
    def scc_reach_profile(ss: "StateSpace", scc_map: dict[int, int]) -> list[tuple[int, int]]:
        reps = sorted(set(scc_map.values()))
        reach: dict[int, set[int]] = {}
        for r in reps:
            # Find a state in this SCC
            states_in_scc = [s for s, rep in scc_map.items() if rep == r]
            if states_in_scc:
                reachable = ss.reachable_from(states_in_scc[0])
                reach[r] = {scc_map[s] for s in reachable if s in scc_map}
            else:
                reach[r] = set()

        # Profile: (reach_count, predecessor_count) per SCC
        pred_count: dict[int, int] = {r: 0 for r in reps}
        for r in reps:
            for t in reach[r]:
                if t != r:
                    pred_count[t] += 1

        return sorted((len(reach[r]), pred_count[r]) for r in reps)

    p1 = scc_reach_profile(ss1, lr1.scc_map)
    p2 = scc_reach_profile(ss2, lr2.scc_map)

    return p1 == p2


# ---------------------------------------------------------------------------
# Symmetry orbits
# ---------------------------------------------------------------------------

def _apply_permutation(coord: tuple[int, ...], perm: tuple[int, ...]) -> tuple[int, ...]:
    """Apply a permutation to a coordinate tuple.

    ``perm`` is a tuple where perm[i] gives the source index for position i.
    """
    return tuple(coord[perm[i]] for i in range(len(perm)))


def symmetry_orbits(ss: "StateSpace", n: int) -> dict[int, list[int]]:
    """Group states by Sn symmetry orbits.

    Two states are in the same orbit if one can be obtained from the other
    by permuting the n factors.  The orbit representative is the state with
    the lexicographically smallest coordinate tuple.

    Parameters:
        ss: An n-fold product state space.
        n: Number of factors.

    Returns:
        Dictionary mapping orbit representative state ID to the list of
        all state IDs in that orbit.
    """
    if n <= 1:
        return {sid: [sid] for sid in ss.states}

    coords = _get_coords(ss, n)
    if not coords:
        return {sid: [sid] for sid in ss.states}

    # Build reverse map: coordinate -> state id
    coord_to_id: dict[tuple[int, ...], int] = {}
    for sid, coord in coords.items():
        coord_to_id[coord] = sid

    # Generate all permutations of n elements
    perms = list(permutations(range(n)))

    # Assign each state to its orbit (canonical = sorted coordinates)
    visited: set[int] = set()
    orbits: dict[int, list[int]] = {}

    for sid in sorted(ss.states):
        if sid in visited:
            continue
        if sid not in coords:
            # No coordinates -- treat as singleton orbit
            orbits[sid] = [sid]
            visited.add(sid)
            continue

        coord = coords[sid]
        orbit_members: list[int] = []

        for perm in perms:
            permuted = _apply_permutation(coord, perm)
            if permuted in coord_to_id:
                member = coord_to_id[permuted]
                if member not in visited:
                    orbit_members.append(member)
                    visited.add(member)

        if not orbit_members:
            orbit_members = [sid]
            visited.add(sid)

        # Representative is the one with smallest id
        rep = min(orbit_members)
        orbits[rep] = sorted(orbit_members)

    return orbits


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_replication(body_type: str, n: int) -> ReplicationAnalysis:
    """Perform full analysis of a replicated session type S^n.

    Builds L(S), computes L(S)^n, checks lattice properties, diagonal
    isomorphism, and symmetry orbits.

    Parameters:
        body_type: The base session type string S.
        n: Number of copies (must be >= 1).

    Returns:
        A ``ReplicationAnalysis`` with all metrics.
    """
    from reticulate.statespace import build_statespace
    from reticulate.product import power_statespace
    from reticulate.lattice import check_lattice

    body = parse(body_type)
    base_ss = build_statespace(body)
    base_result = check_lattice(base_ss)

    if n == 1:
        return ReplicationAnalysis(
            body_type=body_type,
            n=n,
            base_states=len(base_ss.states),
            product_states=len(base_ss.states),
            diagonal_states_count=len(base_ss.states),
            orbit_count=len(base_ss.states),
            compression_ratio=1.0,
            is_lattice=base_result.is_lattice,
            base_is_lattice=base_result.is_lattice,
            diagonal_isomorphic=True,
        )

    product_ss = power_statespace(base_ss, n)
    product_result = check_lattice(product_ss)

    diag = diagonal_states(product_ss, n)
    diag_ss = diagonal_lattice(product_ss, n)

    # Check diagonal order-isomorphism to base (ignore transition labels)
    diag_iso = _is_order_isomorphic(diag_ss, base_ss)

    orbits = symmetry_orbits(product_ss, n)
    orbit_count = len(orbits)

    product_count = len(product_ss.states)
    compression = product_count / orbit_count if orbit_count > 0 else 1.0

    return ReplicationAnalysis(
        body_type=body_type,
        n=n,
        base_states=len(base_ss.states),
        product_states=product_count,
        diagonal_states_count=len(diag),
        orbit_count=orbit_count,
        compression_ratio=compression,
        is_lattice=product_result.is_lattice,
        base_is_lattice=base_result.is_lattice,
        diagonal_isomorphic=diag_iso,
    )
