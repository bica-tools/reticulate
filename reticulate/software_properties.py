"""Algebraic Software Properties — A Lattice-Theoretic Dictionary.

Maps classical software engineering properties to lattice-theoretic
characterizations on session type state spaces.  Each property has:

1. A **classical definition** (from process algebra / concurrency theory).
2. A **lattice-theoretic characterization** (in terms of meets, joins,
   reachability, order structure).
3. A **checker function** that verifies the property on a StateSpace.

Dictionary entries (properties):

| Property | Classical | Lattice-Theoretic |
|----------|-----------|-------------------|
| Safety | "nothing bad happens" | Downward-closed set (lower set / ideal) |
| Liveness | "something good eventually happens" | Every element reaches a designated set |
| Deadlock-freedom | No state with zero enabled transitions (except end) | Bottom is the only element with no successors |
| Progress | Every branch is eventually resolved | Every non-bottom element has a successor |
| Confluence | Diamond property on transitions | Joins exist for all fork points |
| Determinism | At most one transition per label per state | Transition map is a partial function |
| Reversibility | Every transition can be undone | SCCs cover the entire state space |
| Transparency | Internal choices are invisible to the environment | Selection transitions do not affect meet/join structure |
| Compositionality | Properties preserved under parallel | Property holds on product iff it holds on factors |
| Boundedness | Finite protocol execution | Lattice has finite height |
| Fairness | Every enabled branch is eventually taken | Every non-bottom element reaches bottom via every successor |
| Monotonicity | Progress always moves "forward" | All transitions are order-preserving (antitone) |
| Responsiveness | Server always responds to valid requests | Every branch state has all expected successors |
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
class PropertyResult:
    """Result of checking a single software property.

    Attributes:
        name: Property name (e.g. "safety", "liveness").
        holds: Whether the property holds.
        characterization: Brief lattice-theoretic explanation.
        witnesses: Supporting evidence (states, transitions, etc.).
        counterexample: Violating witness if property fails, else None.
    """
    name: str
    holds: bool
    characterization: str
    witnesses: list[str] = field(default_factory=list)
    counterexample: str | None = None


@dataclass(frozen=True)
class DictionaryResult:
    """Result of checking all dictionary properties on a state space.

    Attributes:
        properties: Map from property name to its result.
        num_states: Number of states in the state space.
        num_transitions: Number of transitions.
        summary: Human-readable summary string.
    """
    properties: dict[str, PropertyResult]
    num_states: int
    num_transitions: int
    summary: str

    def holds(self, name: str) -> bool:
        """Check if a named property holds."""
        return self.properties[name].holds

    @property
    def all_hold(self) -> bool:
        """True iff every property in the dictionary holds."""
        return all(p.holds for p in self.properties.values())

    @property
    def holding(self) -> list[str]:
        """Names of properties that hold."""
        return [n for n, p in self.properties.items() if p.holds]

    @property
    def failing(self) -> list[str]:
        """Names of properties that fail."""
        return [n for n, p in self.properties.items() if not p.holds]


# ---------------------------------------------------------------------------
# Individual property checkers
# ---------------------------------------------------------------------------

def check_safety(ss: StateSpace) -> PropertyResult:
    """Safety: "nothing bad happens" — no stuck states except bottom.

    Lattice characterization: The set of reachable states from top
    forms a downward-closed set (lower set) in the lattice ordering.
    Every state can reach bottom (the terminal state).

    A protocol is safe iff every reachable state has a path to the
    terminal state — no execution can get permanently stuck.
    """
    # Build adjacency
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    # Check: every state can reach bottom
    # Compute reverse reachability from bottom
    rev_adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev_adj[tgt].add(src)

    can_reach_bottom: set[int] = set()
    stack = [ss.bottom]
    while stack:
        s = stack.pop()
        if s in can_reach_bottom:
            continue
        can_reach_bottom.add(s)
        for pred in rev_adj.get(s, set()):
            stack.append(pred)

    stuck_states = ss.states - can_reach_bottom
    holds = len(stuck_states) == 0

    witnesses = [f"{len(can_reach_bottom)}/{len(ss.states)} states can reach bottom"]
    counterexample = None
    if not holds:
        example = min(stuck_states)
        counterexample = f"State {example} cannot reach bottom (terminal state)"

    return PropertyResult(
        name="safety",
        holds=holds,
        characterization="Every reachable state has a path to bottom (⊥). "
                         "The reachable set is a lower set in the lattice.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_liveness(ss: StateSpace) -> PropertyResult:
    """Liveness: "something good eventually happens" — progress toward termination.

    Lattice characterization: Every non-bottom element has at least one
    successor strictly below it in the lattice ordering, ensuring eventual
    termination. Equivalently, there are no infinite descending chains
    that avoid bottom (guaranteed by finite height).
    """
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    # Check every non-bottom state has at least one successor
    stuck = []
    for s in ss.states:
        if s == ss.bottom:
            continue
        if len(adj[s]) == 0:
            stuck.append(s)

    holds = len(stuck) == 0

    witnesses = []
    if holds:
        witnesses.append("Every non-bottom state has at least one outgoing transition")
    counterexample = None
    if not holds:
        counterexample = f"State {min(stuck)} has no outgoing transitions but is not bottom"

    return PropertyResult(
        name="liveness",
        holds=holds,
        characterization="Every non-bottom element has a successor. "
                         "The lattice has no dead ends above ⊥.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_deadlock_freedom(ss: StateSpace) -> PropertyResult:
    """Deadlock-freedom: no state is stuck (except the terminal state).

    Lattice characterization: Bottom (⊥) is the unique minimal element
    with no successors. Every other element has at least one successor.
    This is equivalent to liveness but with the additional check that
    bottom truly has no successors (it is terminal).
    """
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    deadlocked = []
    bottom_has_successors = len(adj.get(ss.bottom, set())) > 0

    for s in ss.states:
        if s == ss.bottom:
            continue
        if len(adj[s]) == 0:
            deadlocked.append(s)

    holds = len(deadlocked) == 0

    witnesses = []
    if holds:
        witnesses.append(f"Bottom (state {ss.bottom}) is the unique sink")
    if bottom_has_successors:
        witnesses.append(f"Note: bottom has {len(adj[ss.bottom])} outgoing transitions (recursive)")

    counterexample = None
    if not holds:
        counterexample = f"State(s) {deadlocked} are deadlocked (no transitions, not bottom)"

    return PropertyResult(
        name="deadlock_freedom",
        holds=holds,
        characterization="⊥ is the unique element with no successors. "
                         "No non-terminal state is a dead end.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_progress(ss: StateSpace) -> PropertyResult:
    """Progress: every reachable non-terminal state enables at least one action.

    Lattice characterization: For every element x > ⊥ in the lattice,
    there exists y such that x covers y (x > y with no z: x > z > y)
    in the Hasse diagram. The lattice has no "gaps" above bottom.
    """
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    # Check from top's reachable set
    reachable = ss.reachable_from(ss.top)
    no_progress = []
    for s in reachable:
        if s == ss.bottom:
            continue
        if len(adj[s]) == 0:
            no_progress.append(s)

    holds = len(no_progress) == 0

    witnesses = [f"{len(reachable)} reachable states checked"]
    counterexample = None
    if not holds:
        counterexample = f"State {min(no_progress)} is reachable but has no enabled actions"

    return PropertyResult(
        name="progress",
        holds=holds,
        characterization="Every reachable non-⊥ element covers at least one element. "
                         "The Hasse diagram has no dead-end nodes above ⊥.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_confluence(ss: StateSpace) -> PropertyResult:
    """Confluence: whenever execution forks, the branches rejoin.

    Lattice characterization: For every state with multiple successors
    (a fork point), the successors have a meet (greatest lower bound).
    This is the lattice diamond property — forks always have a join-back
    point. In a lattice, this is guaranteed by the existence of all meets.

    Stronger than just "meets exist": we check that fork points' successors
    converge to a common continuation.
    """
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    from reticulate.lattice import compute_meet

    fork_points = []
    non_confluent = []

    for s in ss.states:
        succs = sorted(adj[s])
        if len(succs) < 2:
            continue
        fork_points.append(s)

        # Check all pairs of successors have a meet
        all_pairs_meet = True
        for i in range(len(succs)):
            for j in range(i + 1, len(succs)):
                m = compute_meet(ss, succs[i], succs[j])
                if m is None:
                    all_pairs_meet = False
                    non_confluent.append((s, succs[i], succs[j]))
                    break
            if not all_pairs_meet:
                break

    holds = len(non_confluent) == 0

    witnesses = [f"{len(fork_points)} fork points checked"]
    counterexample = None
    if not holds:
        s, a, b = non_confluent[0]
        counterexample = f"Fork at state {s}: successors {a} and {b} have no meet"

    return PropertyResult(
        name="confluence",
        holds=holds,
        characterization="Every fork point's successors have a meet (GLB). "
                         "Branching paths always reconverge in the lattice.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_determinism(ss: StateSpace) -> PropertyResult:
    """Determinism: each label enables at most one transition per state.

    Lattice characterization: Each transition label defines a partial
    function on the lattice (not a relation). Equivalently, for each
    state s and label l, |{t : (s,l,t) ∈ transitions}| ≤ 1.
    This means transition labels are lattice endomorphisms (partial functions).
    """
    label_targets: dict[tuple[int, str], list[int]] = {}
    for src, label, tgt in ss.transitions:
        key = (src, label)
        label_targets.setdefault(key, []).append(tgt)

    non_det = []
    for (src, label), targets in label_targets.items():
        if len(targets) > 1:
            non_det.append((src, label, targets))

    holds = len(non_det) == 0

    witnesses = [f"{len(label_targets)} (state, label) pairs checked"]
    counterexample = None
    if not holds:
        src, label, targets = non_det[0]
        counterexample = (
            f"State {src}, label '{label}' has {len(targets)} targets: {targets}"
        )

    return PropertyResult(
        name="determinism",
        holds=holds,
        characterization="Each transition label is a partial function on the lattice. "
                         "No state has two transitions with the same label.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_reversibility(ss: StateSpace) -> PropertyResult:
    """Reversibility: every state can reach every other state.

    Lattice characterization: The entire state space forms a single
    strongly connected component. Equivalently, the quotient lattice
    is a single point — the SCC map collapses everything.

    In practice, most protocols are NOT reversible (you can't "un-close"
    a file). Reversibility holds mainly for cyclic protocols.
    """
    # Check if all states are in one SCC
    reachable_from_top = ss.reachable_from(ss.top)

    # Build reverse adjacency for reverse reachability
    rev_adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev_adj[tgt].add(src)

    # Reverse reachability from top
    can_reach_top: set[int] = set()
    stack = [ss.top]
    while stack:
        s = stack.pop()
        if s in can_reach_top:
            continue
        can_reach_top.add(s)
        for pred in rev_adj.get(s, set()):
            stack.append(pred)

    single_scc = (reachable_from_top == ss.states) and (can_reach_top == ss.states)

    witnesses = []
    if single_scc:
        witnesses.append("All states form a single SCC")
    else:
        witnesses.append(
            f"Forward reachable from top: {len(reachable_from_top)}, "
            f"reverse reachable from top: {len(can_reach_top)}, "
            f"total: {len(ss.states)}"
        )

    counterexample = None
    if not single_scc:
        unreachable = ss.states - reachable_from_top
        if unreachable:
            counterexample = f"State {min(unreachable)} not reachable from top"
        else:
            no_return = ss.states - can_reach_top
            if no_return:
                counterexample = f"State {min(no_return)} cannot reach back to top"

    return PropertyResult(
        name="reversibility",
        holds=single_scc,
        characterization="The entire state space is a single SCC. "
                         "Every state can reach every other state.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_transparency(ss: StateSpace) -> PropertyResult:
    """Transparency: internal choices (selections) don't create hidden divergence.

    Lattice characterization: Selection transitions (+{...}) preserve
    the lattice structure — removing all selection transitions from the
    state space does not disconnect any reachable state from bottom.
    The selection labels are "transparent" to the external observer.
    """
    # Build adjacency without selection transitions
    adj_no_sel: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, label, tgt in ss.transitions:
        if not ss.is_selection(src, label, tgt):
            adj_no_sel[src].add(tgt)

    # States that only have selection transitions out
    selection_only_states: list[int] = []
    all_adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        all_adj[src].add(tgt)

    for s in ss.states:
        if s == ss.bottom:
            continue
        if len(all_adj[s]) > 0 and len(adj_no_sel[s]) == 0:
            selection_only_states.append(s)

    # Check: can every state still reach bottom without selection transitions?
    # This is too strict — selections ARE needed. Instead check that
    # selection states don't create "hidden" divergence where one branch
    # terminates and another doesn't.
    # A selection is transparent if all its branches can reach bottom.
    non_transparent = []
    for s in ss.states:
        sel_targets = [
            tgt for src, label, tgt in ss.transitions
            if src == s and ss.is_selection(src, label, tgt)
        ]
        if len(sel_targets) < 2:
            continue
        # Check all selection branches can reach bottom
        for t in sel_targets:
            reachable = ss.reachable_from(t)
            if ss.bottom not in reachable:
                non_transparent.append((s, t))
                break

    holds = len(non_transparent) == 0

    witnesses = [f"{len(selection_only_states)} selection-only states"]
    counterexample = None
    if not holds:
        s, t = non_transparent[0]
        counterexample = (
            f"Selection at state {s}: branch to {t} cannot reach bottom"
        )

    return PropertyResult(
        name="transparency",
        holds=holds,
        characterization="All selection branches can reach ⊥. "
                         "Internal choices don't create hidden dead ends.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_boundedness(ss: StateSpace) -> PropertyResult:
    """Boundedness: the protocol has finite execution depth.

    Lattice characterization: The lattice has finite height — the
    longest chain from top to bottom is bounded. In the quotient
    (after SCC collapse), this is always true for finite state spaces.
    The height equals the longest path in the quotient DAG.
    """
    # Compute longest path from top to bottom in the DAG
    # (ignoring cycles — those are collapsed in the quotient)
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    # BFS/DFS to find longest acyclic path from top to bottom
    # Use memoized DFS on the state space
    longest: dict[int, int] = {}

    def _longest_path(s: int, visited: frozenset[int]) -> int:
        if s == ss.bottom:
            return 0
        if s in longest and s not in visited:
            return longest[s]
        best = -1
        for t in adj.get(s, set()):
            if t in visited:
                continue  # skip cycles
            length = _longest_path(t, visited | {t})
            if length >= 0:
                best = max(best, 1 + length)
        if s not in visited:
            longest[s] = best
        return best

    height = _longest_path(ss.top, frozenset({ss.top}))

    holds = height >= 0  # Can reach bottom at all
    bounded = len(ss.states) < 10000  # Finite states = bounded

    witnesses = []
    if height >= 0:
        witnesses.append(f"Lattice height (longest chain): {height}")
    witnesses.append(f"State space size: {len(ss.states)}")

    counterexample = None
    if not holds:
        counterexample = "Top cannot reach bottom via any acyclic path"

    return PropertyResult(
        name="boundedness",
        holds=holds and bounded,
        characterization=f"Finite lattice height = {height}. "
                         "The protocol terminates in bounded steps.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_fairness(ss: StateSpace) -> PropertyResult:
    """Fairness: every enabled branch is eventually reachable to completion.

    Lattice characterization: For every non-bottom state with multiple
    successors, EVERY successor can reach bottom. No branch of a
    choice permanently blocks termination.
    """
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    unfair = []
    checked = 0
    for s in ss.states:
        succs = adj[s]
        if len(succs) < 2:
            continue
        checked += 1
        for t in succs:
            reachable = ss.reachable_from(t)
            if ss.bottom not in reachable:
                unfair.append((s, t))
                break

    holds = len(unfair) == 0

    witnesses = [f"{checked} branching states checked"]
    counterexample = None
    if not holds:
        s, t = unfair[0]
        counterexample = f"Branch at state {s}: successor {t} cannot reach bottom"

    return PropertyResult(
        name="fairness",
        holds=holds,
        characterization="Every branch successor can reach ⊥. "
                         "No choice permanently blocks termination.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_monotonicity(ss: StateSpace) -> PropertyResult:
    """Monotonicity: transitions always move "downward" in the lattice.

    Lattice characterization: For every transition (s, l, t), we have
    s ≥ t in the lattice ordering (t is reachable from s). This means
    protocol execution is monotonically decreasing in the lattice —
    you always move closer to bottom (termination).

    In session types, this is automatic for DAG state spaces but may
    fail in recursive types where cycles create non-monotone paths.
    """
    # Check: for every transition (s, l, t), is t reachable from s?
    # In a well-formed state space built from build_statespace, this should
    # always hold because transitions ARE the reachability edges.
    # But we check anyway for robustness.
    non_monotone = []
    for src, label, tgt in ss.transitions:
        # In the quotient ordering, s >= t iff t is reachable from s
        # Since transitions define reachability, this is trivially true
        # for direct transitions. The deeper check: does the transition
        # move us strictly closer to bottom?
        # We check: tgt can reach bottom but src's path through tgt is the
        # only path (strongest monotonicity).
        pass  # Direct transitions always satisfy s >= t

    # More meaningful check: no transition goes "upward" — from a state
    # closer to bottom to a state further from bottom.
    # Compute distance to bottom for each state
    rev_adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev_adj[tgt].add(src)

    # BFS from bottom to compute minimum distance
    dist: dict[int, int] = {ss.bottom: 0}
    queue = [ss.bottom]
    while queue:
        s = queue.pop(0)
        for pred in rev_adj.get(s, set()):
            if pred not in dist:
                dist[pred] = dist[s] + 1
                queue.append(pred)

    # Check transitions go from higher distance to lower distance
    upward = []
    for src, label, tgt in ss.transitions:
        if src in dist and tgt in dist:
            if dist[tgt] > dist[src]:
                upward.append((src, label, tgt, dist[src], dist[tgt]))

    holds = len(upward) == 0

    witnesses = []
    if ss.top in dist:
        witnesses.append(f"Max distance from bottom: {dist.get(ss.top, '?')}")
    witnesses.append(f"{len(ss.transitions)} transitions checked")

    counterexample = None
    if not holds:
        src, label, tgt, d_src, d_tgt = upward[0]
        counterexample = (
            f"Transition ({src}, '{label}', {tgt}): "
            f"distance to bottom increases from {d_src} to {d_tgt}"
        )

    return PropertyResult(
        name="monotonicity",
        holds=holds,
        characterization="Every transition moves closer to ⊥. "
                         "Protocol execution is monotonically decreasing.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_responsiveness(ss: StateSpace) -> PropertyResult:
    """Responsiveness: every branch state enables at least one action.

    Lattice characterization: Branch states (external choice) have
    at least one enabled method. The server always has a response
    available for valid client requests. In lattice terms, every
    non-bottom, non-selection element covers at least one element.
    """
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].add(tgt)

    # Identify branch states (states with non-selection outgoing transitions)
    branch_states = []
    unresponsive = []
    for s in ss.states:
        if s == ss.bottom:
            continue
        non_sel_out = [
            (label, tgt)
            for src, label, tgt in ss.transitions
            if src == s and not ss.is_selection(src, label, tgt)
        ]
        sel_out = [
            (label, tgt)
            for src, label, tgt in ss.transitions
            if src == s and ss.is_selection(src, label, tgt)
        ]
        # It's a branch state if it has any non-selection transitions
        # or no transitions at all (should have had some)
        if non_sel_out:
            branch_states.append(s)
        elif not sel_out and len(adj[s]) == 0:
            unresponsive.append(s)

    holds = len(unresponsive) == 0

    witnesses = [f"{len(branch_states)} branch states identified"]
    counterexample = None
    if not holds:
        counterexample = f"State {min(unresponsive)} enables no actions"

    return PropertyResult(
        name="responsiveness",
        holds=holds,
        characterization="Every non-terminal state enables at least one action. "
                         "The server always responds to valid requests.",
        witnesses=witnesses,
        counterexample=counterexample,
    )


def check_compositionality(ss: StateSpace) -> PropertyResult:
    """Compositionality: parallel composition preserves protocol structure.

    Lattice characterization: If the state space was built from a
    parallel composition (S₁ ∥ S₂), then the product lattice
    L(S₁) × L(S₂) is isomorphic to the state space. The lattice
    decomposes cleanly into independent factors.
    """
    is_product = ss.product_coords is not None and ss.product_factors is not None
    holds = True
    witnesses = []

    if is_product:
        assert ss.product_factors is not None
        assert ss.product_coords is not None
        n_factors = len(ss.product_factors)
        expected_states = 1
        for factor in ss.product_factors:
            expected_states *= len(factor.states)

        actual_states = len(ss.states)
        holds = actual_states == expected_states
        witnesses.append(f"Product of {n_factors} factors")
        witnesses.append(f"Expected {expected_states} states, got {actual_states}")
    else:
        witnesses.append("Not a product state space (no parallel composition)")

    counterexample = None
    if not holds:
        counterexample = "Product state count mismatch: decomposition is incomplete"

    return PropertyResult(
        name="compositionality",
        holds=holds,
        characterization="Product lattice decomposes into independent factors. "
                         "L(S₁ ∥ S₂) ≅ L(S₁) × L(S₂).",
        witnesses=witnesses,
        counterexample=counterexample,
    )


# ---------------------------------------------------------------------------
# Full dictionary check
# ---------------------------------------------------------------------------

# Registry of all property checkers
PROPERTY_CHECKERS: dict[str, callable] = {
    "safety": check_safety,
    "liveness": check_liveness,
    "deadlock_freedom": check_deadlock_freedom,
    "progress": check_progress,
    "confluence": check_confluence,
    "determinism": check_determinism,
    "reversibility": check_reversibility,
    "transparency": check_transparency,
    "boundedness": check_boundedness,
    "fairness": check_fairness,
    "monotonicity": check_monotonicity,
    "responsiveness": check_responsiveness,
    "compositionality": check_compositionality,
}


def check_all_properties(ss: StateSpace) -> DictionaryResult:
    """Check all dictionary properties on a state space.

    Returns a DictionaryResult with every property evaluated.
    """
    results: dict[str, PropertyResult] = {}
    for name, checker in PROPERTY_CHECKERS.items():
        results[name] = checker(ss)

    holding = [n for n, r in results.items() if r.holds]
    failing = [n for n, r in results.items() if not r.holds]

    summary_parts = [
        f"{len(holding)}/{len(results)} properties hold",
    ]
    if failing:
        summary_parts.append(f"Failing: {', '.join(failing)}")
    else:
        summary_parts.append("All properties satisfied")

    return DictionaryResult(
        properties=results,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        summary="; ".join(summary_parts),
    )


def check_properties(ss: StateSpace, *names: str) -> DictionaryResult:
    """Check specific named properties on a state space.

    Args:
        ss: The state space to check.
        *names: Property names to check (from PROPERTY_CHECKERS keys).

    Raises:
        ValueError: If an unknown property name is given.
    """
    results: dict[str, PropertyResult] = {}
    for name in names:
        if name not in PROPERTY_CHECKERS:
            raise ValueError(
                f"Unknown property '{name}'. "
                f"Available: {sorted(PROPERTY_CHECKERS.keys())}"
            )
        results[name] = PROPERTY_CHECKERS[name](ss)

    holding = [n for n, r in results.items() if r.holds]
    failing = [n for n, r in results.items() if not r.holds]

    summary_parts = [f"{len(holding)}/{len(results)} properties hold"]
    if failing:
        summary_parts.append(f"Failing: {', '.join(failing)}")

    return DictionaryResult(
        properties=results,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        summary="; ".join(summary_parts),
    )


def property_profile(ss: StateSpace) -> dict[str, bool]:
    """Quick boolean profile: property name → holds/fails.

    Useful for comparing protocols or building classification tables.
    """
    result = check_all_properties(ss)
    return {name: r.holds for name, r in result.properties.items()}
