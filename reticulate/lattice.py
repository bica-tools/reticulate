"""Lattice property checking for session type state spaces.

Given a StateSpace (labeled transition system), checks whether the
reachability poset forms a lattice by verifying:
  1. A unique top (initial state reaches all others)
  2. A unique bottom (all states reach the terminal state)
  3. Every pair of states has a meet (greatest lower bound)
  4. Every pair of states has a join (least upper bound)

Ordering convention (from the session-type perspective):
  s1 >= s2  iff  there is a directed path from s1 to s2
  Top = initial state (greatest element)
  Bottom = end state (least element)

Cycles (from recursive types) violate antisymmetry, so the algorithm
quotients by strongly connected components (SCCs) before checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatticeResult:
    """Result of a lattice check on a state space.

    Attributes:
        is_lattice: True iff the quotient poset is a lattice.
        has_top: True iff the top (initial state's SCC) reaches all SCCs.
        has_bottom: True iff all SCCs reach the bottom (end state's SCC).
        all_meets_exist: True iff every pair of quotient nodes has a meet.
        all_joins_exist: True iff every pair of quotient nodes has a join.
        num_scc: Number of SCCs in the quotient.
        counterexample: First failing pair ``(a, b, "no_meet"|"no_join")``
            using original state IDs, or None if the poset is a lattice.
        scc_map: Mapping from original state ID to its SCC representative.
    """

    is_lattice: bool
    has_top: bool
    has_bottom: bool
    all_meets_exist: bool
    all_joins_exist: bool
    num_scc: int
    counterexample: tuple[int, int, str] | None
    scc_map: dict[int, int]

    @property
    def scc_groups(self) -> dict[int, frozenset[int]]:
        """Multi-state SCC groups: representative → member states.

        Only includes SCCs with 2+ members (from recursive cycles).
        Single-state SCCs (including self-loops) are excluded.
        """
        groups: dict[int, set[int]] = {}
        for state, rep in self.scc_map.items():
            groups.setdefault(rep, set()).add(state)
        return {rep: frozenset(members) for rep, members in groups.items()
                if len(members) > 1}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_lattice(ss: StateSpace) -> LatticeResult:
    """Check whether the reachability poset of *ss* forms a lattice.

    Quotients by SCCs to handle cycles from recursive types, then checks
    that the quotient DAG has top, bottom, and all pairwise meets/joins.
    """
    q = _build_quotient_poset(ss)

    all_nodes = set(q.nodes)

    has_top = q.fwd_reach[q.top] == all_nodes
    has_bottom = all(q.bottom in q.fwd_reach[n] for n in all_nodes)

    all_meets = True
    all_joins = True
    counterexample: tuple[int, int, str] | None = None

    nodes_list = sorted(q.nodes)
    for i, a in enumerate(nodes_list):
        for b in nodes_list[i + 1:]:
            if all_meets and _meet_on_quotient(q, a, b) is None:
                all_meets = False
                if counterexample is None:
                    counterexample = (q.rep[a], q.rep[b], "no_meet")
            if all_joins and _join_on_quotient(q, a, b) is None:
                all_joins = False
                if counterexample is None:
                    counterexample = (q.rep[a], q.rep[b], "no_join")
            if not all_meets and not all_joins:
                break
        if not all_meets and not all_joins:
            break

    is_lattice = has_top and has_bottom and all_meets and all_joins

    return LatticeResult(
        is_lattice=is_lattice,
        has_top=has_top,
        has_bottom=has_bottom,
        all_meets_exist=all_meets,
        all_joins_exist=all_joins,
        num_scc=len(q.nodes),
        counterexample=counterexample,
        scc_map={s: q.rep[q.state_to_node[s]] for s in q.state_to_node},
    )


def compute_meet(ss: StateSpace, a: int, b: int) -> int | None:
    """Compute the meet (greatest lower bound) of states *a* and *b*.

    Returns a representative original state ID, or None if no meet exists.
    """
    q = _build_quotient_poset(ss)
    qa = q.state_to_node.get(a)
    qb = q.state_to_node.get(b)
    if qa is None or qb is None:
        return None
    m = _meet_on_quotient(q, qa, qb)
    if m is None:
        return None
    return q.rep[m]


def compute_join(ss: StateSpace, a: int, b: int) -> int | None:
    """Compute the join (least upper bound) of states *a* and *b*.

    Returns a representative original state ID, or None if no join exists.
    """
    q = _build_quotient_poset(ss)
    qa = q.state_to_node.get(a)
    qb = q.state_to_node.get(b)
    if qa is None or qb is None:
        return None
    j = _join_on_quotient(q, qa, qb)
    if j is None:
        return None
    return q.rep[j]


# ---------------------------------------------------------------------------
# Internal: quotient poset
# ---------------------------------------------------------------------------

@dataclass
class _QuotientPoset:
    """DAG obtained by quotienting a state space by its SCCs."""

    nodes: set[int]                                # SCC indices
    top: int                                       # SCC containing ss.top
    bottom: int                                    # SCC containing ss.bottom
    fwd_adj: dict[int, set[int]]                   # forward adjacency
    rev_adj: dict[int, set[int]]                   # reverse adjacency
    fwd_reach: dict[int, set[int]]                 # forward reachability (inclusive)
    rev_reach: dict[int, set[int]]                 # reverse reachability (inclusive)
    rep: dict[int, int]                            # SCC index -> representative state
    state_to_node: dict[int, int]                  # original state -> SCC index


def _build_quotient_poset(ss: StateSpace) -> _QuotientPoset:
    """Build the quotient DAG from a state space (steps 1-3)."""

    # Step 1: Compute adjacency and SCCs
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    sccs = _compute_sccs(ss.states, adj)

    # Map each original state to its SCC index
    state_to_scc: dict[int, int] = {}
    rep: dict[int, int] = {}
    for idx, scc in enumerate(sccs):
        r = min(scc)
        rep[idx] = r
        for s in scc:
            state_to_scc[s] = idx

    nodes = set(range(len(sccs)))

    # Step 2: Build quotient DAG edges
    fwd_adj: dict[int, set[int]] = {n: set() for n in nodes}
    rev_adj: dict[int, set[int]] = {n: set() for n in nodes}

    for src, _, tgt in ss.transitions:
        s_scc = state_to_scc[src]
        t_scc = state_to_scc[tgt]
        if s_scc != t_scc:
            fwd_adj[s_scc].add(t_scc)
            rev_adj[t_scc].add(s_scc)

    # Step 3: Compute reachability via topological-order DP
    # Topological order: process in reverse topological order (sinks first)
    # for forward reachability, sources first for reverse reachability.
    topo_order = _topological_sort(nodes, fwd_adj)

    # Forward reachability: process sinks first (reverse topo order)
    fwd_reach: dict[int, set[int]] = {n: {n} for n in nodes}
    for n in reversed(topo_order):
        for succ in fwd_adj[n]:
            fwd_reach[n] |= fwd_reach[succ]

    # Reverse reachability: process sources first (topo order)
    rev_reach: dict[int, set[int]] = {n: {n} for n in nodes}
    for n in topo_order:
        for pred in rev_adj[n]:
            rev_reach[n] |= rev_reach[pred]

    q_top = state_to_scc[ss.top]
    q_bottom = state_to_scc[ss.bottom]

    return _QuotientPoset(
        nodes=nodes,
        top=q_top,
        bottom=q_bottom,
        fwd_adj=fwd_adj,
        rev_adj=rev_adj,
        fwd_reach=fwd_reach,
        rev_reach=rev_reach,
        rep=rep,
        state_to_node=state_to_scc,
    )


# ---------------------------------------------------------------------------
# Internal: SCC computation (iterative Tarjan's algorithm)
# ---------------------------------------------------------------------------

def _compute_sccs(
    states: set[int],
    adj: dict[int, list[int]],
) -> list[frozenset[int]]:
    """Compute SCCs using iterative Tarjan's algorithm.

    Returns SCCs in reverse topological order (sinks first).
    """
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    result: list[frozenset[int]] = []

    for start in sorted(states):
        if start in index:
            continue

        # Iterative DFS using an explicit call stack.
        # Each frame is (node, neighbor_iterator, is_root_call).
        call_stack: list[tuple[int, int]] = []
        # Initialize start node
        index[start] = lowlink[start] = index_counter[0]
        index_counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        call_stack.append((start, 0))

        while call_stack:
            v, ni = call_stack[-1]
            neighbors = adj.get(v, [])

            if ni < len(neighbors):
                # Advance neighbor index
                call_stack[-1] = (v, ni + 1)
                w = neighbors[ni]

                if w not in index:
                    # Tree edge: push w
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, 0))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                # All neighbors processed; check if v is SCC root
                if lowlink[v] == index[v]:
                    scc_members: list[int] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc_members.append(w)
                        if w == v:
                            break
                    result.append(frozenset(scc_members))

                # Pop frame and propagate lowlink to parent
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])

    return result


# ---------------------------------------------------------------------------
# Internal: topological sort (Kahn's algorithm)
# ---------------------------------------------------------------------------

def _topological_sort(
    nodes: set[int],
    fwd_adj: dict[int, set[int]],
) -> list[int]:
    """Topological sort of a DAG. Returns nodes from sources to sinks."""
    in_degree: dict[int, int] = {n: 0 for n in nodes}
    for n in nodes:
        for m in fwd_adj[n]:
            in_degree[m] += 1

    # Use sorted() for determinism
    queue = sorted(n for n in nodes if in_degree[n] == 0)
    order: list[int] = []

    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in sorted(fwd_adj[n]):
            in_degree[m] -= 1
            if in_degree[m] == 0:
                queue.append(m)

    return order


# ---------------------------------------------------------------------------
# Internal: meet and join on the quotient DAG
# ---------------------------------------------------------------------------

def check_distributive(ss: StateSpace) -> DistributivityResult:
    """Check whether the reachability poset of *ss* is a distributive lattice.

    First checks that it is a lattice (via check_lattice), then searches for
    forbidden sublattices M₃ (diamond) and N₅ (pentagon).  By Birkhoff's
    Theorem IX.2, a lattice is distributive iff it contains neither.

    Also classifies the lattice in the Birkhoff hierarchy:
    Boolean ⊂ Distributive ⊂ Modular ⊂ Semi-modular ⊂ Lattice
    """
    lr = check_lattice(ss)
    if not lr.is_lattice:
        return DistributivityResult(
            is_lattice=False,
            is_distributive=False,
            is_modular=False,
            has_m3=False,
            has_n5=False,
            m3_witness=None,
            n5_witness=None,
            classification="not_lattice",
            lattice_result=lr,
        )

    q = _build_quotient_poset(ss)

    # Search for N₅ (pentagon) and M₃ (diamond) sublattices
    n5 = _find_n5(q)
    m3 = _find_m3(q)

    has_n5 = n5 is not None
    has_m3 = m3 is not None

    # Birkhoff's characterization:
    # - Modular iff no N₅
    # - Distributive iff no N₅ and no M₃
    is_modular = not has_n5
    is_distributive = is_modular and not has_m3

    # Classification
    if is_distributive:
        # Check Boolean: every element has a complement
        is_boolean = _check_boolean(q)
        classification = "boolean" if is_boolean else "distributive"
    elif is_modular:
        classification = "modular"
    else:
        classification = "lattice"

    # Map witnesses back to original state IDs
    m3_orig = tuple(q.rep[n] for n in m3) if m3 else None
    n5_orig = tuple(q.rep[n] for n in n5) if n5 else None

    return DistributivityResult(
        is_lattice=True,
        is_distributive=is_distributive,
        is_modular=is_modular,
        has_m3=has_m3,
        has_n5=has_n5,
        m3_witness=m3_orig,
        n5_witness=n5_orig,
        classification=classification,
        lattice_result=lr,
    )


@dataclass(frozen=True)
class DistributivityResult:
    """Result of distributivity check on a state space.

    Attributes:
        is_lattice: True iff the quotient poset is a lattice.
        is_distributive: True iff it is a distributive lattice (no M₃, no N₅).
        is_modular: True iff it is a modular lattice (no N₅).
        has_m3: True iff the lattice contains an M₃ (diamond) sublattice.
        has_n5: True iff the lattice contains an N₅ (pentagon) sublattice.
        m3_witness: 5-tuple (top, a, b, c, bot) of original state IDs, or None.
        n5_witness: 5-tuple (top, a, b, c, bot) of original state IDs, or None.
        classification: One of "boolean", "distributive", "modular", "lattice",
            "not_lattice".
        lattice_result: The underlying LatticeResult.
    """

    is_lattice: bool
    is_distributive: bool
    is_modular: bool
    has_m3: bool
    has_n5: bool
    m3_witness: tuple[int, ...] | None
    n5_witness: tuple[int, ...] | None
    classification: str
    lattice_result: LatticeResult


def _find_n5(q: _QuotientPoset) -> tuple[int, ...] | None:
    """Search for an N₅ (pentagon) sublattice.

    N₅ has 5 elements {0, a, b, c, 1} with:
      1 > a > b > 0, 1 > c > 0, and a ∥ c, b ∥ c
    (c is incomparable with both a and b).

    We search for a chain a > b and an element c incomparable with both,
    where all five share a common upper bound (join) and lower bound (meet).
    """
    nodes = sorted(q.nodes)
    n = len(nodes)

    def le(x: int, y: int) -> bool:
        return y in q.fwd_reach[x]

    def incomparable(x: int, y: int) -> bool:
        return not le(x, y) and not le(y, x)

    # For each pair a > b (chain of length 2), find c incomparable with both
    for i in range(n):
        for j in range(n):
            a, b = nodes[i], nodes[j]
            if a == b or not le(a, b):
                continue
            # a > b in the ordering
            for k in range(n):
                c = nodes[k]
                if c == a or c == b:
                    continue
                if not incomparable(a, c) or not incomparable(b, c):
                    continue
                # Found a, b, c with a > b, c ∥ a, c ∥ b
                # Check they have a common top and bottom
                top = _join_on_quotient(q, a, c)
                bot = _meet_on_quotient(q, b, c)
                if top is not None and bot is not None:
                    # Verify this is actually N₅: top > a > b > bot, top > c > bot
                    # Additional check: the 5 elements must form a sublattice
                    # isomorphic to N₅. In N₅, meet(a,c) = bot (not some
                    # intermediate element).
                    if (le(top, a) and le(a, b) and le(b, bot)
                            and le(top, c) and le(c, bot)):
                        # Verify sublattice: in N₅, meet(a,c)=bot
                    # and join(b,c)=top (not some intermediate element)
                        mac = _meet_on_quotient(q, a, c)
                        jbc = _join_on_quotient(q, b, c)
                        if mac == bot and jbc == top:
                            return (top, a, b, c, bot)
    return None


def _find_m3(q: _QuotientPoset) -> tuple[int, ...] | None:
    """Search for an M₃ (diamond) sublattice.

    M₃ has 5 elements {0, a, b, c, 1} with:
      1 > a, b, c > 0
      a, b, c pairwise incomparable
      meet(a,b) = meet(a,c) = meet(b,c) = 0
      join(a,b) = join(a,c) = join(b,c) = 1
    """
    nodes = sorted(q.nodes)
    n = len(nodes)

    def le(x: int, y: int) -> bool:
        return y in q.fwd_reach[x]

    def incomparable(x: int, y: int) -> bool:
        return not le(x, y) and not le(y, x)

    # For each triple of pairwise incomparable elements
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            if not incomparable(a, b):
                continue
            jab = _join_on_quotient(q, a, b)
            mab = _meet_on_quotient(q, a, b)
            if jab is None or mab is None:
                continue
            for k in range(j + 1, n):
                c = nodes[k]
                if not incomparable(a, c) or not incomparable(b, c):
                    continue
                jac = _join_on_quotient(q, a, c)
                mac = _meet_on_quotient(q, a, c)
                jbc = _join_on_quotient(q, b, c)
                mbc = _meet_on_quotient(q, b, c)
                if (jac == jab == jbc and mac == mab == mbc
                        and jab is not None and mab is not None):
                    return (jab, a, b, c, mab)
    return None


def _check_boolean(q: _QuotientPoset) -> bool:
    """Check if a distributive lattice is Boolean (every element complemented)."""
    def le(x: int, y: int) -> bool:
        return y in q.fwd_reach[x]

    for a in q.nodes:
        has_complement = False
        for b in q.nodes:
            m = _meet_on_quotient(q, a, b)
            j = _join_on_quotient(q, a, b)
            if m == q.bottom and j == q.top:
                has_complement = True
                break
        if not has_complement:
            return False
    return True


def _meet_on_quotient(q: _QuotientPoset, a: int, b: int) -> int | None:
    """Compute the meet (GLB) of quotient nodes *a* and *b*.

    Meet = greatest lower bound.
    Lower bounds of (a, b) = nodes reachable from both a and b.
    The meet is the unique lower bound that reaches all other lower bounds
    (i.e., it is the greatest among them).
    """
    if a == b:
        return a

    lower_bounds = q.fwd_reach[a] & q.fwd_reach[b]
    if not lower_bounds:
        return None

    # Find the greatest lower bound: a node m in lower_bounds such that
    # every other node in lower_bounds is reachable from m.
    for m in lower_bounds:
        if lower_bounds <= q.fwd_reach[m]:
            return m

    return None


def _join_on_quotient(q: _QuotientPoset, a: int, b: int) -> int | None:
    """Compute the join (LUB) of quotient nodes *a* and *b*.

    Join = least upper bound.
    Upper bounds of (a, b) = nodes that can reach both a and b.
    The join is the unique upper bound reachable from all other upper bounds
    (i.e., it is the least among them).
    """
    if a == b:
        return a

    upper_bounds = q.rev_reach[a] & q.rev_reach[b]
    if not upper_bounds:
        return None

    # Find the least upper bound: a node j in upper_bounds such that
    # every other node in upper_bounds can reach j.
    for j in upper_bounds:
        if upper_bounds <= q.rev_reach[j]:
            return j

    return None
