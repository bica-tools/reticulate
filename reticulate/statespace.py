"""State-space construction for session types.

Given a session type AST, constructs the labeled transition system (LTS)
representing all reachable protocol states and transitions.

For sequential types, the state space is a directed graph (finite automaton).
For parallel types (∥), the state space is the product of the component
state spaces (see product.py and docs/specs/parallel-constructor-spec.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reticulate.parser import (
    Branch,
    Continuation,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
)
from reticulate.product import product_statespace


@dataclass
class StateSpace:
    """Labeled transition system for a session type.

    - ``states``: set of integer state IDs
    - ``transitions``: list of ``(source, label, target)`` triples
    - ``top``: initial protocol state (⊤)
    - ``bottom``: terminal state (⊥ = end)
    - ``labels``: human-readable description for each state
    - ``selection_transitions``: set of ``(source, label, target)`` triples
      that are selection (internal choice) transitions
    """

    states: set[int] = field(default_factory=set)
    transitions: list[tuple[int, str, int]] = field(default_factory=list)
    top: int = 0
    bottom: int = 0
    labels: dict[int, str] = field(default_factory=dict)
    selection_transitions: set[tuple[int, str, int]] = field(default_factory=set)
    product_coords: dict[int, tuple[int, ...]] | None = field(default=None)
    product_factors: list["StateSpace"] | None = field(default=None)

    def enabled(self, state: int) -> list[tuple[str, int]]:
        """Return ``(label, target)`` pairs for transitions from *state*."""
        return [(l, t) for s, l, t in self.transitions if s == state]

    def successors(self, state: int) -> set[int]:
        """Direct successor states of *state*."""
        return {t for s, l, t in self.transitions if s == state}

    def reachable_from(self, state: int) -> set[int]:
        """All states reachable from *state* (inclusive)."""
        visited: set[int] = set()
        stack = [state]
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            for t in self.successors(s):
                stack.append(t)
        return visited

    def is_selection(self, src: int, label: str, tgt: int) -> bool:
        """Return True if the transition ``(src, label, tgt)`` is a selection."""
        return (src, label, tgt) in self.selection_transitions

    def enabled_methods(self, state: int) -> list[tuple[str, int]]:
        """Return ``(label, target)`` pairs for METHOD transitions from *state*."""
        return [
            (l, t) for s, l, t in self.transitions
            if s == state and (s, l, t) not in self.selection_transitions
        ]

    def enabled_selections(self, state: int) -> list[tuple[str, int]]:
        """Return ``(label, target)`` pairs for SELECTION transitions from *state*."""
        return [
            (l, t) for s, l, t in self.transitions
            if s == state and (s, l, t) in self.selection_transitions
        ]


def truncate_back_edges(ss: StateSpace) -> tuple[StateSpace, list[tuple[int, str, int]]]:
    """Redirect back-edges to bottom, returning acyclic state space + back-edge list.

    Uses SCC decomposition to identify cycles, then finds a feedback arc set
    within each non-trivial SCC using a DFS spanning tree.  Back-edges are
    transitions that close cycles (target is a DFS ancestor within the SCC).
    Each such edge is redirected to point to ``ss.bottom`` instead.

    The result is guaranteed to be acyclic (a DAG).

    Returns:
        A tuple of (truncated_statespace, back_edges) where back_edges is the
        list of original (src, label, tgt) triples that were redirected.
    """
    # Step 1: Compute SCCs using iterative Tarjan's
    sccs = _compute_sccs_list(ss)
    state_to_scc: dict[int, int] = {}
    for idx, scc in enumerate(sccs):
        for s in scc:
            state_to_scc[s] = idx

    # Step 2: For each non-trivial SCC, find a feedback arc set via DFS
    back_edges: list[tuple[int, str, int]] = []

    # Build adjacency
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss.states}
    for src, lbl, tgt in ss.transitions:
        adj[src].append((lbl, tgt))

    for scc in sccs:
        if len(scc) == 1:
            # Check for self-loop
            s = next(iter(scc))
            for src, lbl, tgt in ss.transitions:
                if src == s and tgt == s:
                    back_edges.append((src, lbl, tgt))
            continue

        # Non-trivial SCC: DFS within the SCC to find back-edges
        scc_set = set(scc)
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[int, int] = {s: WHITE for s in scc_set}

        # Start from the state closest to top (minimum id as heuristic,
        # or the entry point if identifiable)
        start = min(scc_set)

        stack: list[tuple[int, bool]] = [(start, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                color[node] = BLACK
                continue
            if color[node] != WHITE:
                continue
            color[node] = GRAY
            stack.append((node, True))
            for lbl, tgt in adj[node]:
                if tgt not in scc_set:
                    continue  # Skip inter-SCC edges
                if color[tgt] == GRAY:
                    back_edges.append((node, lbl, tgt))
                elif color[tgt] == WHITE:
                    stack.append((tgt, False))
                # BLACK: cross-edge within SCC — also a feedback candidate
                elif color[tgt] == BLACK:
                    back_edges.append((node, lbl, tgt))

    # Step 3: Iteratively remove back-edges until acyclic
    # The DFS-based approach may not find all necessary edges in one pass
    # (especially in product spaces).  Iterate until no cycles remain.
    back_edge_set = set(back_edges)
    remaining = [t for t in ss.transitions if t not in back_edge_set]

    # Check if remaining is acyclic; if not, find more back-edges
    for _ in range(10):  # bounded iterations
        cycle_edges = _find_cycle_edges(ss.states, remaining, ss.top)
        if not cycle_edges:
            break
        back_edges.extend(cycle_edges)
        back_edge_set.update(cycle_edges)
        remaining = [t for t in ss.transitions if t not in back_edge_set]

    # Step 4: Build truncated transitions
    new_transitions: list[tuple[int, str, int]] = []
    new_selections: set[tuple[int, str, int]] = set()

    for src, lbl, tgt in ss.transitions:
        if (src, lbl, tgt) in back_edge_set:
            redirected = (src, lbl, ss.bottom)
            new_transitions.append(redirected)
            if (src, lbl, tgt) in ss.selection_transitions:
                new_selections.add(redirected)
        else:
            new_transitions.append((src, lbl, tgt))
            if (src, lbl, tgt) in ss.selection_transitions:
                new_selections.add((src, lbl, tgt))

    # Step 5: Compute reachable states
    reachable: set[int] = set()
    visit_stack = [ss.top]
    while visit_stack:
        s = visit_stack.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for src, _, tgt in new_transitions:
            if src == s and tgt not in reachable:
                visit_stack.append(tgt)

    truncated = StateSpace(
        states=reachable,
        transitions=[(s, l, t) for s, l, t in new_transitions if s in reachable],
        top=ss.top,
        bottom=ss.bottom,
        labels={s: v for s, v in ss.labels.items() if s in reachable},
        selection_transitions={
            (s, l, t) for s, l, t in new_selections if s in reachable
        },
        product_coords=(
            {s: c for s, c in ss.product_coords.items() if s in reachable}
            if ss.product_coords is not None
            else None
        ),
        product_factors=ss.product_factors,
    )
    return truncated, back_edges


def _compute_sccs_list(ss: StateSpace) -> list[set[int]]:
    """Compute SCCs using iterative Tarjan's. Returns list of SCC sets."""
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    sccs: list[set[int]] = []

    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    def strongconnect(v: int) -> None:
        work_stack: list[tuple[int, int, bool]] = [(v, 0, False)]
        while work_stack:
            node, child_idx, returned = work_stack.pop()
            if not returned and node not in index:
                index[node] = lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)
            if returned:
                child = adj[node][child_idx - 1]
                lowlink[node] = min(lowlink[node], lowlink[child])

            pushed = False
            children = adj[node]
            start = child_idx if not returned else child_idx
            for i in range(start, len(children)):
                w = children[i]
                if w not in index:
                    work_stack.append((node, i + 1, True))
                    work_stack.append((w, 0, False))
                    pushed = True
                    break
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], index[w])

            if not pushed and lowlink[node] == index[node]:
                scc: set[int] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w == node:
                        break
                sccs.append(scc)

    for s in ss.states:
        if s not in index:
            strongconnect(s)

    return sccs


def _find_cycle_edges(
    states: set[int],
    transitions: list[tuple[int, str, int]],
    top: int,
) -> list[tuple[int, str, int]]:
    """Find edges that close cycles in the given transition list. Returns empty if acyclic."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[int, int] = {s: WHITE for s in states}
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in states}
    for src, lbl, tgt in transitions:
        adj[src].append((lbl, tgt))

    cycle_edges: list[tuple[int, str, int]] = []
    stack: list[tuple[int, bool]] = [(top, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            color[node] = BLACK
            continue
        if color[node] != WHITE:
            continue
        color[node] = GRAY
        stack.append((node, True))
        for lbl, tgt in adj[node]:
            if color.get(tgt) == GRAY:
                cycle_edges.append((node, lbl, tgt))
            elif color.get(tgt) == WHITE:
                stack.append((tgt, False))
    return cycle_edges


def build_statespace(session_type: SessionType) -> StateSpace:
    """Construct the labeled transition system for a session type.

    Handles all constructors: ``End``, ``Wait``, ``Var``, ``Branch``,
    ``Select``, ``Rec`` (with cycles), ``Continuation`` (chaining after
    parallel), and ``Parallel`` (product construction via
    :func:`product_statespace`).

    Raises ``ValueError`` on unbound type variables.
    """
    builder = _Builder()
    return builder.build(session_type)


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

class _Builder:
    """Accumulates states and transitions during construction."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._states: dict[int, str] = {}
        self._transitions: list[tuple[int, str, int]] = []
        self._selection_transitions: set[tuple[int, str, int]] = set()
        self._product_coords: dict[int, tuple[int, ...]] | None = None
        self._product_factors: list[StateSpace] | None = None

    def _fresh(self, label: str) -> int:
        sid = self._next_id
        self._next_id += 1
        self._states[sid] = label
        return sid

    # -- public entry point -------------------------------------------------

    def build(self, node: SessionType) -> StateSpace:
        end_id = self._fresh("end")
        top_id = self._build(node, {}, end_id)

        reachable = self._reachable(top_id)
        reachable_transitions = [
            (s, l, t) for s, l, t in self._transitions if s in reachable
        ]
        reachable_selections = {
            (s, l, t) for s, l, t in self._selection_transitions if s in reachable
        }
        # Remap product coords to only include reachable states
        product_coords = None
        if self._product_coords is not None:
            product_coords = {
                s: c for s, c in self._product_coords.items() if s in reachable
            }

        return StateSpace(
            states=reachable,
            transitions=reachable_transitions,
            top=top_id,
            bottom=end_id,
            labels={s: self._states[s] for s in reachable if s in self._states},
            selection_transitions=reachable_selections,
            product_coords=product_coords,
            product_factors=self._product_factors,
        )

    # -- recursive construction ---------------------------------------------

    def _build(
        self,
        node: SessionType,
        env: dict[str, int],
        end_id: int,
    ) -> int:
        """Return the entry state ID for *node*."""
        match node:
            case End():
                return end_id

            case Wait():
                # Wait is semantically terminal (like End) at the
                # state-space level.  The distinction between wait
                # and end is enforced by well-formedness checking.
                return end_id

            case Var(name=name):
                if name not in env:
                    raise ValueError(f"unbound type variable: {name!r}")
                return env[name]

            case Branch(choices=choices):
                if len(choices) == 1:
                    label = choices[0][0]
                else:
                    label = "&{" + ", ".join(l for l, _ in choices) + "}"
                entry = self._fresh(label)
                for lbl, body in choices:
                    target = self._build(body, env, end_id)
                    self._transitions.append((entry, lbl, target))
                return entry

            case Select(choices=choices):
                label = "+{" + ", ".join(l for l, _ in choices) + "}"
                entry = self._fresh(label)
                for lbl, body in choices:
                    target = self._build(body, env, end_id)
                    tr = (entry, lbl, target)
                    self._transitions.append(tr)
                    self._selection_transitions.add(tr)
                return entry

            case Rec(var=var, body=body):
                # Pre-allocate a placeholder for the recursion variable.
                # After building the body, merge the placeholder into
                # the body's actual entry state (creating the cycle).
                placeholder = self._fresh(f"rec_{var}")
                new_env = {**env, var: placeholder}
                body_entry = self._build(body, new_env, end_id)
                if body_entry != placeholder:
                    self._merge(placeholder, body_entry)
                    return body_entry
                return placeholder

            case Continuation(left=left, right=right):
                # Build right side first; left's bottom chains to right's entry.
                right_entry = self._build(right, env, end_id)
                left_entry = self._build(left, env, right_entry)
                return left_entry

            case Parallel(branches=branches):
                return self._build_parallel(branches, end_id)

            case _:
                raise TypeError(f"unknown AST node: {type(node).__name__}")

    # -- recursion helper: merge placeholder into real entry ----------------

    def _merge(self, old_id: int, new_id: int) -> None:
        """Redirect every occurrence of *old_id* to *new_id* in transitions."""
        self._transitions = [
            (
                new_id if s == old_id else s,
                l,
                new_id if t == old_id else t,
            )
            for s, l, t in self._transitions
        ]
        self._selection_transitions = {
            (
                new_id if s == old_id else s,
                l,
                new_id if t == old_id else t,
            )
            for s, l, t in self._selection_transitions
        }
        if old_id in self._states:
            del self._states[old_id]

    # -- parallel: product construction ------------------------------------

    def _build_parallel(
        self,
        branches: tuple[SessionType, ...],
        end_id: int,
    ) -> int:
        """Build each branch independently, compute the n-ary product, embed it."""
        from functools import reduce
        spaces = [build_statespace(b) for b in branches]
        prod = reduce(product_statespace, spaces)

        # Remap product state IDs into this builder's ID space.
        remap: dict[int, int] = {}
        for sid in prod.states:
            if sid == prod.bottom:
                remap[sid] = end_id
            else:
                remap[sid] = self._fresh(prod.labels.get(sid, "?"))

        for src, lbl, tgt in prod.transitions:
            remapped = (remap[src], lbl, remap[tgt])
            self._transitions.append(remapped)
            if prod.is_selection(src, lbl, tgt):
                self._selection_transitions.add(remapped)

        # Propagate product metadata with remapped IDs
        if prod.product_coords is not None:
            self._product_coords = {
                remap[sid]: coord
                for sid, coord in prod.product_coords.items()
            }
            self._product_factors = prod.product_factors

        return remap[prod.top]

    # -- reachability -------------------------------------------------------

    def _reachable(self, start: int) -> set[int]:
        visited: set[int] = set()
        stack = [start]
        while stack:
            s = stack.pop()
            if s in visited:
                continue
            visited.add(s)
            for src, _, tgt in self._transitions:
                if src == s:
                    visited.add(s)
                    stack.append(tgt)
        return visited
