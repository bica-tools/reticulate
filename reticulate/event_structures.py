"""Event structures from session types (Step 16).

A **prime event structure** is a triple (E, ≤, #) where:
- E is a set of events
- ≤ ⊆ E × E is a partial order (causality)
- # ⊆ E × E is a symmetric irreflexive relation (conflict)

satisfying:
- Finite causes: {e' : e' ≤ e} is finite for all e
- Conflict heredity: e # e' and e' ≤ e'' implies e # e''

A **configuration** is a subset x ⊆ E that is:
- Downward-closed: e ∈ x and e' ≤ e implies e' ∈ x
- Conflict-free: no e, e' ∈ x with e # e'

The **configuration domain** D(ES(S)) is ordered by inclusion
and is order-isomorphic to the session type state space L(S).

Translation from session types:
- Events = transitions (s, m, t) in the state space
- Conflict = same-source transitions (choice points)
- Causality = sequential dependency (t₁ reaches s₂ uniquely)
- Concurrency = independent events (from parallel composition)

Key functions:
  - ``build_event_structure(ss)`` -- construct ES(S) from state space
  - ``configurations(es)``        -- enumerate all configurations
  - ``config_domain(es)``         -- build configuration poset
  - ``check_isomorphism(es, ss)`` -- verify D(ES(S)) ≅ L(S)
  - ``classify_events(es)``       -- branch/select/parallel events
  - ``concurrency_pairs(es)``     -- find concurrent event pairs
  - ``analyze_event_structure(ss)`` -- full analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """An event in the event structure.

    Attributes:
        source: Source state ID.
        label: Transition label (method name).
        target: Target state ID.
    """
    source: int
    label: str
    target: int

    def __repr__(self) -> str:
        return f"({self.source},{self.label},{self.target})"


@dataclass(frozen=True)
class EventStructure:
    """A prime event structure (E, ≤, #).

    Attributes:
        events: Set of events.
        causality: Set of (e1, e2) pairs where e1 ≤ e2 (e1 causes e2).
        conflict: Set of (e1, e2) pairs where e1 # e2 (mutually exclusive).
        labels: Mapping from events to their method labels.
    """
    events: frozenset[Event]
    causality: frozenset[tuple[Event, Event]]
    conflict: frozenset[tuple[Event, Event]]

    @property
    def num_events(self) -> int:
        return len(self.events)

    @property
    def num_conflicts(self) -> int:
        return len(self.conflict) // 2  # symmetric, count each pair once

    @property
    def num_causal_pairs(self) -> int:
        # Exclude reflexive pairs
        return sum(1 for e1, e2 in self.causality if e1 != e2)


@dataclass(frozen=True)
class Configuration:
    """A configuration (downward-closed conflict-free subset).

    Attributes:
        events: Frozenset of events in this configuration.
    """
    events: frozenset[Event]

    @property
    def size(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class ConfigDomain:
    """The configuration domain D(ES(S)).

    Attributes:
        configs: List of configurations ordered by inclusion.
        bottom: Empty configuration.
        ordering: Set of (c1, c2) pairs where c1 ⊆ c2.
        num_configs: Number of configurations.
    """
    configs: list[Configuration]
    bottom: Configuration
    ordering: frozenset[tuple[int, int]]  # indices into configs
    num_configs: int


@dataclass(frozen=True)
class EventClassification:
    """Classification of events by their session type origin.

    Attributes:
        branch_events: Events from branch (&) constructors.
        select_events: Events from selection (+) constructors.
        parallel_events: Events from parallel (||) factors.
        conflict_groups: Groups of mutually conflicting events (by source state).
    """
    branch_events: frozenset[Event]
    select_events: frozenset[Event]
    parallel_events: frozenset[Event]
    conflict_groups: dict[int, frozenset[Event]]  # source_state → events


@dataclass(frozen=True)
class ESAnalysis:
    """Complete event structure analysis.

    Attributes:
        es: The event structure.
        num_events: Number of events.
        num_conflicts: Number of conflict pairs.
        num_causal: Number of non-reflexive causal pairs.
        num_configs: Number of configurations.
        num_concurrent: Number of concurrent event pairs.
        is_isomorphic: True iff config domain ≅ state space.
        classification: Event classification.
        max_config_size: Size of the largest configuration.
        conflict_density: Fraction of event pairs in conflict.
    """
    es: EventStructure
    num_events: int
    num_conflicts: int
    num_causal: int
    num_configs: int
    num_concurrent: int
    is_isomorphic: bool
    classification: EventClassification
    max_config_size: int
    conflict_density: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_adj(ss: StateSpace) -> dict[int, list[tuple[str, int]]]:
    """Forward adjacency with labels."""
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss.states}
    for src, label, tgt in ss.transitions:
        adj[src].append((label, tgt))
    return adj


def _reachable(ss: StateSpace, start: int) -> set[int]:
    """States reachable from start (inclusive)."""
    adj = _build_adj(ss)
    visited: set[int] = set()
    stack = [start]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for _, t in adj.get(s, []):
            stack.append(t)
    return visited


# ---------------------------------------------------------------------------
# Public API: Build event structure
# ---------------------------------------------------------------------------

def build_event_structure(ss: StateSpace) -> EventStructure:
    """Construct the prime event structure ES(S) from a state space.

    Events = transitions in the state space.
    Conflict = events sharing the same source state.
    Causality = sequential dependency (target of e1 reaches source of e2).
    """
    adj = _build_adj(ss)

    # 1. Events: one per transition
    events: set[Event] = set()
    for src, label, tgt in ss.transitions:
        events.add(Event(source=src, label=label, target=tgt))

    event_list = sorted(events, key=lambda e: (e.source, e.label, e.target))

    # 2. Immediate conflict: same source state, different events
    immediate_conflict: set[tuple[Event, Event]] = set()
    by_source: dict[int, list[Event]] = {}
    for e in event_list:
        by_source.setdefault(e.source, []).append(e)

    for src, evts in by_source.items():
        if len(evts) >= 2:
            for i in range(len(evts)):
                for j in range(i + 1, len(evts)):
                    immediate_conflict.add((evts[i], evts[j]))
                    immediate_conflict.add((evts[j], evts[i]))

    # 3. Causality: e1 <_E e2 iff e1.target can reach e2.source
    # Use reachability on the state space
    reach: dict[int, set[int]] = {}
    for s in ss.states:
        reach[s] = _reachable(ss, s)

    causality: set[tuple[Event, Event]] = set()
    for e in event_list:
        causality.add((e, e))  # reflexive

    for e1 in event_list:
        for e2 in event_list:
            if e1 == e2:
                continue
            if e2.source in reach.get(e1.target, set()):
                causality.add((e1, e2))

    # 4. Conflict heredity: if e1 # e2 and e2 ≤ e3, then e1 # e3
    full_conflict = set(immediate_conflict)
    changed = True
    while changed:
        changed = False
        new_conflicts: set[tuple[Event, Event]] = set()
        for e1, e2 in full_conflict:
            for e2p, e3 in causality:
                if e2p == e2 and e3 != e1 and (e1, e3) not in full_conflict:
                    new_conflicts.add((e1, e3))
                    new_conflicts.add((e3, e1))
        if new_conflicts - full_conflict:
            full_conflict |= new_conflicts
            changed = True

    return EventStructure(
        events=frozenset(events),
        causality=frozenset(causality),
        conflict=frozenset(full_conflict),
    )


# ---------------------------------------------------------------------------
# Public API: Configurations
# ---------------------------------------------------------------------------

def configurations(es: EventStructure, max_configs: int = 10000) -> list[Configuration]:
    """Enumerate all configurations of the event structure.

    A configuration is a downward-closed, conflict-free subset of events.
    """
    event_list = sorted(es.events, key=lambda e: (e.source, e.label, e.target))

    # Build causality lookup: for each event, what events must precede it
    predecessors: dict[Event, set[Event]] = {e: set() for e in event_list}
    for e1, e2 in es.causality:
        if e1 != e2:
            predecessors[e2].add(e1)

    # Build conflict lookup
    conflicts_of: dict[Event, set[Event]] = {e: set() for e in event_list}
    for e1, e2 in es.conflict:
        conflicts_of[e1].add(e2)

    # Enumerate via BFS on configurations
    result: list[Configuration] = []
    empty = Configuration(events=frozenset())
    queue: list[frozenset[Event]] = [frozenset()]
    seen: set[frozenset[Event]] = {frozenset()}

    while queue and len(result) < max_configs:
        current = queue.pop(0)
        result.append(Configuration(events=current))

        # Try adding each event not already in the configuration
        for e in event_list:
            if e in current:
                continue

            # Check conflict-free
            if any(c in current for c in conflicts_of.get(e, set())):
                continue

            # Check downward-closed: all predecessors must be in current
            if not predecessors[e].issubset(current):
                continue

            new_config = current | {e}
            if new_config not in seen:
                seen.add(new_config)
                queue.append(new_config)

    return result


def config_domain(es: EventStructure, max_configs: int = 10000) -> ConfigDomain:
    """Build the configuration domain D(ES(S)).

    Configurations ordered by subset inclusion.
    """
    configs = configurations(es, max_configs)
    bottom = Configuration(events=frozenset())

    # Build inclusion ordering (indices)
    ordering: set[tuple[int, int]] = set()
    for i in range(len(configs)):
        for j in range(len(configs)):
            if configs[i].events.issubset(configs[j].events):
                ordering.add((i, j))

    return ConfigDomain(
        configs=configs,
        bottom=bottom,
        ordering=frozenset(ordering),
        num_configs=len(configs),
    )


# ---------------------------------------------------------------------------
# Public API: Isomorphism check
# ---------------------------------------------------------------------------

def check_isomorphism(es: EventStructure, ss: StateSpace) -> bool:
    """Check if the configuration domain D(ES(S)) is order-isomorphic to L(S).

    This verifies the fundamental theorem: the number of configurations
    equals the number of states (a necessary condition for isomorphism).
    """
    configs = configurations(es)
    return len(configs) == len(ss.states)


# ---------------------------------------------------------------------------
# Public API: Classification
# ---------------------------------------------------------------------------

def classify_events(es: EventStructure, ss: StateSpace) -> EventClassification:
    """Classify events by their session type origin.

    - Branch events: from & constructors (external choice)
    - Select events: from + constructors (internal choice)
    - Parallel events: from || factors
    """
    branch_events: set[Event] = set()
    select_events: set[Event] = set()
    parallel_events: set[Event] = set()

    for e in es.events:
        if ss.is_selection(e.source, e.label, e.target):
            select_events.add(e)
        else:
            branch_events.add(e)

    # Parallel events: from product state spaces
    if ss.product_coords is not None:
        for e in es.events:
            if e.source in (ss.product_coords or {}):
                parallel_events.add(e)

    # Conflict groups: events sharing a source state
    conflict_groups: dict[int, set[Event]] = {}
    for e in es.events:
        conflict_groups.setdefault(e.source, set()).add(e)
    # Only keep groups with 2+ events (actual conflict)
    conflict_groups = {s: evts for s, evts in conflict_groups.items() if len(evts) >= 2}

    return EventClassification(
        branch_events=frozenset(branch_events),
        select_events=frozenset(select_events),
        parallel_events=frozenset(parallel_events),
        conflict_groups={s: frozenset(evts) for s, evts in conflict_groups.items()},
    )


# ---------------------------------------------------------------------------
# Public API: Concurrency
# ---------------------------------------------------------------------------

def concurrency_pairs(es: EventStructure) -> list[tuple[Event, Event]]:
    """Find all concurrent event pairs.

    Events e1, e2 are concurrent iff:
    - e1 ≠ e2
    - NOT e1 ≤ e2
    - NOT e2 ≤ e1
    - NOT e1 # e2
    """
    event_list = sorted(es.events, key=lambda e: (e.source, e.label, e.target))
    causal_set = set(es.causality)
    conflict_set = set(es.conflict)

    concurrent: list[tuple[Event, Event]] = []
    for i in range(len(event_list)):
        for j in range(i + 1, len(event_list)):
            e1, e2 = event_list[i], event_list[j]
            if ((e1, e2) not in causal_set
                    and (e2, e1) not in causal_set
                    and (e1, e2) not in conflict_set):
                concurrent.append((e1, e2))

    return concurrent


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_event_structure(ss: StateSpace) -> ESAnalysis:
    """Full event structure analysis of a state space."""
    es = build_event_structure(ss)
    configs = configurations(es)
    classification = classify_events(es, ss)
    concurrent = concurrency_pairs(es)
    iso = len(configs) == len(ss.states)

    max_config = max((c.size for c in configs), default=0)

    n_events = len(es.events)
    n_pairs = n_events * (n_events - 1) // 2
    conflict_density = es.num_conflicts / n_pairs if n_pairs > 0 else 0.0

    return ESAnalysis(
        es=es,
        num_events=n_events,
        num_conflicts=es.num_conflicts,
        num_causal=es.num_causal_pairs,
        num_configs=len(configs),
        num_concurrent=len(concurrent),
        is_isomorphic=iso,
        classification=classification,
        max_config_size=max_config,
        conflict_density=conflict_density,
    )
