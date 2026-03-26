"""Birkhoff's Representation Theorem for session type lattices (Step 30j).

Birkhoff's fundamental theorem (1937) states that every finite distributive
lattice L is isomorphic to the lattice of downsets (order ideals) of the
poset J(L) of its join-irreducible elements:

    L  ≅  O(J(L))

This module implements:
  - Extraction of the join-irreducible poset J(L) from a session type lattice
  - Construction of the downset lattice O(P) from an arbitrary finite poset
  - Verification that L ≅ O(J(L)) (the Birkhoff representation)
  - Compression ratio: |J(L)| vs |L| — how much smaller the representing poset is
  - Inverse: reconstruct a lattice (as a StateSpace) from a Birkhoff poset

For session types with recursive constructs, the state space has cycles.
The lattice is the SCC quotient (as used by ``check_lattice``).  All
operations in this module work on the quotient DAG so that the join-
irreducible computation and Birkhoff map are well-defined.

Key result for session types: for the ~87% of benchmark protocols whose
lattices are distributive, the join-irreducible poset captures the protocol's
essential structure in a strictly smaller representation.
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
class BirkhoffResult:
    """Result of Birkhoff representation analysis.

    Attributes:
        is_lattice: True iff the state space forms a lattice.
        is_distributive: True iff the lattice is distributive.
        is_representable: True iff the Birkhoff representation holds (= distributive).
        join_irreducibles: Set of join-irreducible representative state IDs.
        ji_poset: Adjacency dict of the join-irreducible poset (Hasse diagram).
        ji_order: Full order relation on join-irreducibles (transitive closure).
        downset_lattice_size: Number of downsets of J(L).
        lattice_size: Number of elements in the lattice (= number of SCC quotient nodes).
        compression_ratio: |J(L)| / |L| — smaller means better compression.
        isomorphism_verified: True iff L ≅ O(J(L)) was verified element-by-element.
        iso_map: Mapping from quotient node → downset (frozenset of join-irreducibles).
    """

    is_lattice: bool
    is_distributive: bool
    is_representable: bool
    join_irreducibles: frozenset[int]
    ji_poset: dict[int, set[int]]
    ji_order: dict[int, frozenset[int]]
    downset_lattice_size: int
    lattice_size: int
    compression_ratio: float
    isomorphism_verified: bool
    iso_map: dict[int, frozenset[int]]


@dataclass(frozen=True)
class DownsetLattice:
    """The downset lattice O(P) of a poset P.

    Each element is a downset (order ideal): a subset D of P such that
    if x in D and y <= x then y in D.  Ordered by subset inclusion.

    Attributes:
        poset: The original poset (element -> set of elements below it in Hasse).
        elements: All downsets, as frozensets.
        order: d1 <= d2 iff d1 is a subset of d2.
        top: The maximum downset (= all elements of P).
        bottom: The minimum downset (= empty set).
        hasse: Covering relation of the downset lattice.
    """

    poset: dict[int, set[int]]
    elements: frozenset[frozenset[int]]
    order: dict[frozenset[int], set[frozenset[int]]]
    top: frozenset[int]
    bottom: frozenset[int]
    hasse: dict[frozenset[int], set[frozenset[int]]]


# ---------------------------------------------------------------------------
# Internal: quotient poset helpers
# ---------------------------------------------------------------------------

def _build_quotient(ss: "StateSpace") -> tuple[
    set[int],                          # nodes (SCC indices)
    int,                               # top node
    int,                               # bottom node
    dict[int, set[int]],               # fwd_adj
    dict[int, set[int]],               # fwd_reach (inclusive)
    dict[int, set[int]],               # rev_reach (inclusive)
    dict[int, int],                    # rep: SCC index -> representative state
]:
    """Build the SCC quotient DAG of a state space.

    Reuses the same SCC + topological-sort logic as lattice.py's
    ``_build_quotient_poset``, but as a standalone function to avoid
    depending on private internals.
    """
    # Adjacency
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    # Iterative Tarjan's SCC
    sccs = _compute_sccs(ss.states, adj)

    state_to_scc: dict[int, int] = {}
    rep: dict[int, int] = {}
    for idx, scc in enumerate(sccs):
        rep[idx] = min(scc)
        for s in scc:
            state_to_scc[s] = idx

    nodes = set(range(len(sccs)))

    fwd_adj: dict[int, set[int]] = {n: set() for n in nodes}
    for src, _, tgt in ss.transitions:
        s_scc = state_to_scc[src]
        t_scc = state_to_scc[tgt]
        if s_scc != t_scc:
            fwd_adj[s_scc].add(t_scc)

    # Topological sort (Kahn's)
    in_degree: dict[int, int] = {n: 0 for n in nodes}
    for n in nodes:
        for m in fwd_adj[n]:
            in_degree[m] += 1
    queue = sorted(n for n in nodes if in_degree[n] == 0)
    topo: list[int] = []
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for m in sorted(fwd_adj[n]):
            in_degree[m] -= 1
            if in_degree[m] == 0:
                queue.append(m)

    # Forward reachability (inclusive)
    fwd_reach: dict[int, set[int]] = {n: {n} for n in nodes}
    for n in reversed(topo):
        for succ in fwd_adj[n]:
            fwd_reach[n] |= fwd_reach[succ]

    # Reverse adjacency and reachability
    rev_adj: dict[int, set[int]] = {n: set() for n in nodes}
    for n in nodes:
        for m in fwd_adj[n]:
            rev_adj[m].add(n)

    rev_reach: dict[int, set[int]] = {n: {n} for n in nodes}
    for n in topo:
        for pred in rev_adj[n]:
            rev_reach[n] |= rev_reach[pred]

    q_top = state_to_scc[ss.top]
    q_bottom = state_to_scc[ss.bottom]

    return nodes, q_top, q_bottom, fwd_adj, fwd_reach, rev_reach, rep


def _compute_sccs(
    states: set[int],
    adj: dict[int, list[int]],
) -> list[frozenset[int]]:
    """Compute SCCs using iterative Tarjan's algorithm."""
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    result: list[frozenset[int]] = []

    for start in sorted(states):
        if start in index:
            continue
        index[start] = lowlink[start] = index_counter[0]
        index_counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        call_stack: list[tuple[int, int]] = [(start, 0)]

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


# ---------------------------------------------------------------------------
# Hasse edges and join-irreducibles on quotient DAG
# ---------------------------------------------------------------------------

def _quotient_hasse(
    nodes: set[int],
    fwd_adj: dict[int, set[int]],
    fwd_reach: dict[int, set[int]],
) -> list[tuple[int, int]]:
    """Compute covering relation on the quotient DAG.

    (a, b) means a covers b: a > b and no c with a > c > b.
    """
    edges: list[tuple[int, int]] = []
    for a in nodes:
        direct = fwd_adj[a]
        for b in direct:
            is_cover = True
            for c in direct:
                if c == b:
                    continue
                if b in fwd_reach[c]:
                    is_cover = False
                    break
            if is_cover:
                edges.append((a, b))
    return edges


def _quotient_join_irreducibles(
    nodes: set[int],
    top: int,
    bottom: int,
    fwd_adj: dict[int, set[int]],
    fwd_reach: dict[int, set[int]],
) -> set[int]:
    """Find join-irreducible elements in the quotient lattice.

    An element j is join-irreducible if it has exactly one lower cover
    in the Hasse diagram.  Bottom is never join-irreducible.
    """
    hasse = _quotient_hasse(nodes, fwd_adj, fwd_reach)
    lower_cover_count: dict[int, int] = {n: 0 for n in nodes}
    for a, b in hasse:
        lower_cover_count[a] += 1
    return {n for n in nodes
            if lower_cover_count[n] == 1 and n != bottom}


def _quotient_ji_poset(
    jis: set[int],
    fwd_reach: dict[int, set[int]],
) -> dict[int, set[int]]:
    """Build the Hasse diagram of the join-irreducible poset (restricted order)."""
    ji_list = sorted(jis)

    # Full order restricted to J(L)
    ji_reach: dict[int, set[int]] = {}
    for j in ji_list:
        ji_reach[j] = {k for k in jis if k in fwd_reach[j] and k != j}

    # Remove transitive edges
    hasse: dict[int, set[int]] = {j: set() for j in ji_list}
    for j in ji_list:
        below = ji_reach[j]
        for k in below:
            is_cover = True
            for m in below:
                if m != k and k in fwd_reach.get(m, set()):
                    is_cover = False
                    break
            if is_cover:
                hasse[j].add(k)
    return hasse


def _ji_order(jis: set[int], hasse: dict[int, set[int]]) -> dict[int, frozenset[int]]:
    """Compute full order on J(L) from Hasse: ji_order[j] = {elements <= j}."""
    order: dict[int, frozenset[int]] = {}
    for j in jis:
        visited: set[int] = set()
        stack = [j]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in hasse.get(u, set()):
                stack.append(v)
        order[j] = frozenset(visited)
    return order


# ---------------------------------------------------------------------------
# Downset lattice construction
# ---------------------------------------------------------------------------

def downset_lattice(poset: dict[int, set[int]]) -> DownsetLattice:
    """Construct the downset lattice O(P) from a finite poset P.

    Args:
        poset: Hasse diagram as adjacency dict: node -> set of nodes it covers
               (i.e., node > covered_node).

    Returns:
        The downset lattice with all downsets, ordering, and covering relation.
    """
    elements = sorted(poset.keys())
    if not elements:
        empty: frozenset[int] = frozenset()
        return DownsetLattice(
            poset=poset,
            elements=frozenset({empty}),
            order={empty: {empty}},
            top=empty,
            bottom=empty,
            hasse={empty: set()},
        )

    # Transitive closure: order_below[x] = all y with y <= x (including x)
    order_below: dict[int, frozenset[int]] = {}
    for x in elements:
        visited: set[int] = set()
        stack = [x]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in poset.get(u, set()):
                stack.append(v)
        order_below[x] = frozenset(visited)

    # Enumerate all downsets (2^n subsets — fine for session type posets)
    all_downsets: list[frozenset[int]] = []
    n = len(elements)
    for mask in range(1 << n):
        subset = frozenset(elements[i] for i in range(n) if mask & (1 << i))
        is_downset = all(order_below[x] <= subset for x in subset)
        if is_downset:
            all_downsets.append(subset)

    # Order by subset inclusion
    order: dict[frozenset[int], set[frozenset[int]]] = {}
    for d1 in all_downsets:
        order[d1] = {d2 for d2 in all_downsets if d1 <= d2}

    top = frozenset(elements)
    bottom: frozenset[int] = frozenset()

    # Covering relation: d2 covers d1 iff d1 ⊂ d2 and |d2 - d1| == 1
    hasse: dict[frozenset[int], set[frozenset[int]]] = {d: set() for d in all_downsets}
    for d1 in all_downsets:
        for d2 in all_downsets:
            if d1 < d2 and len(d2 - d1) == 1:
                hasse[d2].add(d1)

    return DownsetLattice(
        poset=poset,
        elements=frozenset(all_downsets),
        order=order,
        top=top,
        bottom=bottom,
        hasse=hasse,
    )


# ---------------------------------------------------------------------------
# Birkhoff map and isomorphism verification
# ---------------------------------------------------------------------------

def _birkhoff_map(
    nodes: set[int],
    jis: set[int],
    fwd_reach: dict[int, set[int]],
) -> dict[int, frozenset[int]]:
    """Compute the Birkhoff isomorphism phi: L -> O(J(L)).

    For each element x in L, phi(x) = {j in J(L) : j <= x}
    = the set of join-irreducibles below or equal to x.
    """
    iso: dict[int, frozenset[int]] = {}
    for x in nodes:
        below = frozenset(j for j in jis if j in fwd_reach[x])
        iso[x] = below
    return iso


def _verify_isomorphism(
    nodes: set[int],
    fwd_reach: dict[int, set[int]],
    iso: dict[int, frozenset[int]],
    dl: DownsetLattice,
) -> bool:
    """Verify that the Birkhoff map is an order-isomorphism.

    Checks injectivity, surjectivity, and order preservation/reflection.
    """
    images = list(iso.values())
    if len(set(images)) != len(images):
        return False
    if set(images) != set(dl.elements):
        return False
    nlist = sorted(nodes)
    for x in nlist:
        for y in nlist:
            x_ge_y = y in fwd_reach[x]
            phi_x_ge_phi_y = iso[y] <= iso[x]
            if x_ge_y != phi_x_ge_phi_y:
                return False
    return True


# ---------------------------------------------------------------------------
# Reconstruction: poset -> StateSpace
# ---------------------------------------------------------------------------

def reconstruct_from_poset(poset: dict[int, set[int]]) -> "StateSpace":
    """Reconstruct a session type state space from a Birkhoff poset.

    Given a poset P (as its Hasse diagram), constructs the state space whose
    reachability lattice is isomorphic to O(P), the downset lattice of P.

    Each downset becomes a state; transitions go from larger to smaller
    downsets (removing one element at a time).
    """
    from reticulate.statespace import StateSpace

    dl = downset_lattice(poset)
    if not dl.elements:
        ss = StateSpace()
        ss.states = {0}
        ss.top = 0
        ss.bottom = 0
        ss.labels = {0: "end"}
        return ss

    sorted_ds = sorted(dl.elements, key=lambda d: (-len(d), sorted(d) if d else []))
    ds_to_id: dict[frozenset[int], int] = {}
    for i, d in enumerate(sorted_ds):
        ds_to_id[d] = i

    states = set(range(len(sorted_ds)))
    top_id = ds_to_id[dl.top]
    bottom_id = ds_to_id[dl.bottom]

    transitions: list[tuple[int, str, int]] = []
    labels: dict[int, str] = {}

    for d in sorted_ds:
        sid = ds_to_id[d]
        labels[sid] = "{" + ",".join(str(x) for x in sorted(d)) + "}" if d else "end"

    for d2 in sorted_ds:
        for d1 in dl.hasse[d2]:
            removed = d2 - d1
            label = "t" + str(min(removed)) if removed else "tau"
            transitions.append((ds_to_id[d2], label, ds_to_id[d1]))

    ss = StateSpace()
    ss.states = states
    ss.transitions = transitions
    ss.top = top_id
    ss.bottom = bottom_id
    ss.labels = labels
    return ss


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def birkhoff_representation(ss: "StateSpace") -> BirkhoffResult:
    """Compute the Birkhoff representation for a session type lattice.

    For a distributive lattice L, verifies L = O(J(L)) where J(L) is the
    poset of join-irreducible elements and O(P) is the downset lattice.

    Works on the SCC quotient of the state space so that recursive types
    (which introduce cycles) are handled correctly.
    """
    from reticulate.lattice import check_distributive

    dr = check_distributive(ss)

    if not dr.is_lattice:
        return BirkhoffResult(
            is_lattice=False,
            is_distributive=False,
            is_representable=False,
            join_irreducibles=frozenset(),
            ji_poset={},
            ji_order={},
            downset_lattice_size=0,
            lattice_size=len(ss.states),
            compression_ratio=1.0,
            isomorphism_verified=False,
            iso_map={},
        )

    # Build quotient
    nodes, q_top, q_bottom, fwd_adj, fwd_reach, rev_reach, rep = _build_quotient(ss)
    lattice_size = len(nodes)

    # Find join-irreducibles on the quotient
    jis = _quotient_join_irreducibles(nodes, q_top, q_bottom, fwd_adj, fwd_reach)

    # Map JI quotient node IDs to representative state IDs for the result
    jis_rep = {rep[j] for j in jis}

    poset = _quotient_ji_poset(jis, fwd_reach)
    ji_ord = _ji_order(jis, poset)

    # Map poset to representative state IDs
    poset_rep: dict[int, set[int]] = {}
    for j in poset:
        poset_rep[rep[j]] = {rep[k] for k in poset[j]}

    ji_ord_rep: dict[int, frozenset[int]] = {}
    for j in ji_ord:
        ji_ord_rep[rep[j]] = frozenset(rep[k] for k in ji_ord[j])

    if dr.is_distributive:
        dl = downset_lattice(poset)
        iso = _birkhoff_map(nodes, jis, fwd_reach)
        verified = _verify_isomorphism(nodes, fwd_reach, iso, dl)
        dl_size = len(dl.elements)
        # Map iso to representative state IDs
        iso_rep: dict[int, frozenset[int]] = {}
        for n in iso:
            iso_rep[rep[n]] = frozenset(rep[j] for j in iso[n])
    else:
        dl_size = 0
        iso_rep = {}
        verified = False

    compression = len(jis) / lattice_size if lattice_size > 0 else 1.0

    return BirkhoffResult(
        is_lattice=True,
        is_distributive=dr.is_distributive,
        is_representable=dr.is_distributive,
        join_irreducibles=frozenset(jis_rep),
        ji_poset=poset_rep,
        ji_order=ji_ord_rep,
        downset_lattice_size=dl_size,
        lattice_size=lattice_size,
        compression_ratio=compression,
        isomorphism_verified=verified,
        iso_map=iso_rep,
    )


def is_representable(ss: "StateSpace") -> bool:
    """Check if a session type lattice has a Birkhoff representation.

    Equivalent to checking distributivity: a finite lattice is representable
    as a downset lattice iff it is distributive.
    """
    from reticulate.lattice import check_distributive
    dr = check_distributive(ss)
    return dr.is_distributive


def representation_size(ss: "StateSpace") -> tuple[int, int, float]:
    """Compute the sizes of the lattice and its representing poset.

    Returns:
        (lattice_size, poset_size, compression_ratio) where
        compression_ratio = poset_size / lattice_size.
        lattice_size is the number of SCC quotient nodes.
    """
    nodes, q_top, q_bottom, fwd_adj, fwd_reach, _, _ = _build_quotient(ss)
    jis = _quotient_join_irreducibles(nodes, q_top, q_bottom, fwd_adj, fwd_reach)
    lattice_size = len(nodes)
    poset_size = len(jis)
    ratio = poset_size / lattice_size if lattice_size > 0 else 1.0
    return (lattice_size, poset_size, ratio)
