"""RPG quest graphs as lattices (Step 59).

Role-playing game quest structures exhibit natural ordering:
main quests form a spine, side quests branch off, and prerequisites
define a partial order.  This module encodes RPG quest structures
as session types and verifies that the resulting state spaces
form lattices.

Key concepts:

- **Quest**: A task with prerequisites, rewards, and completion status.
- **Main quest**: Linear spine of the game's primary storyline.
- **Side quest**: Optional branching content.
- **Prerequisites**: Quests that must be completed before starting another.
- **Quest graph**: DAG of quests ordered by prerequisites.
- **Completability**: Whether there exists a valid completion order.

Quest graphs map to session types:
- Linear quest chains -> nested branches (sequencing)
- Optional quests -> branch choices (external choice)
- Mutually exclusive quests -> selection (internal choice)
- Parallel quest lines -> parallel composition

This module provides:
    ``encode_quest_graph()``      -- build session type from quest graph.
    ``check_completability()``    -- verify all quests can be completed.
    ``quest_lattice()``           -- analyze quest graph lattice structure.
    ``encode_quest_chain()``      -- linear quest sequence as type.
    ``branching_quests()``        -- side quests as branch choices.
    ``quest_dependency_order()``  -- topological ordering of quests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
    parse,
    pretty,
)
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quest:
    """An RPG quest."""
    name: str
    description: str = ""
    quest_type: str = "main"          # "main", "side", "faction", "repeatable"
    prerequisites: tuple[str, ...] = ()  # names of prerequisite quests
    reward: str = ""
    xp: int = 0
    level_required: int = 1
    is_optional: bool = False
    exclusive_with: tuple[str, ...] = ()  # mutually exclusive quests


@dataclass(frozen=True)
class QuestGraph:
    """A complete quest graph for a game or region."""
    title: str
    quests: tuple[Quest, ...]

    @property
    def quest_map(self) -> dict[str, Quest]:
        return {q.name: q for q in self.quests}


@dataclass(frozen=True)
class CompletabilityResult:
    """Result of completability checking."""
    is_completable: bool
    completion_order: tuple[str, ...]  # valid topological order
    unreachable_quests: tuple[str, ...]  # quests with unmet prerequisites
    cycle_detected: bool
    missing_prerequisites: dict[str, tuple[str, ...]]  # quest -> missing prereqs


@dataclass(frozen=True)
class QuestLatticeResult:
    """Result of quest lattice analysis."""
    session_type_str: str
    num_quests: int
    num_main: int
    num_side: int
    state_count: int
    transition_count: int
    is_lattice: bool
    completion_order: tuple[str, ...]
    branching_factor: float  # average number of available quests at each step


@dataclass(frozen=True)
class QuestChainResult:
    """Result of encoding a linear quest chain."""
    session_type_str: str
    chain_length: int
    state_count: int
    is_lattice: bool


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------

def quest_dependency_order(graph: QuestGraph) -> tuple[str, ...]:
    """Compute a valid completion order (topological sort).

    Returns a topological ordering of quest names respecting prerequisites.
    Raises ValueError if a cycle is detected.
    """
    quest_map = graph.quest_map
    in_degree: dict[str, int] = {q.name: 0 for q in graph.quests}
    dependents: dict[str, list[str]] = {q.name: [] for q in graph.quests}

    for quest in graph.quests:
        for prereq in quest.prerequisites:
            if prereq in quest_map:
                in_degree[quest.name] += 1
                dependents[prereq].append(quest.name)

    # Kahn's algorithm
    queue: list[str] = [name for name, deg in in_degree.items() if deg == 0]
    order: list[str] = []

    while queue:
        queue.sort()  # deterministic ordering
        current = queue.pop(0)
        order.append(current)
        for dep in dependents[current]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(order) != len(graph.quests):
        raise ValueError("Cycle detected in quest prerequisites")

    return tuple(order)


# ---------------------------------------------------------------------------
# Completability checking
# ---------------------------------------------------------------------------

def check_completability(graph: QuestGraph) -> CompletabilityResult:
    """Check whether all quests in the graph can be completed.

    Verifies:
    1. No cycles in prerequisites.
    2. All prerequisites exist in the graph.
    3. Mutually exclusive quests don't block required quests.
    """
    quest_map = graph.quest_map
    missing: dict[str, list[str]] = {}

    for quest in graph.quests:
        for prereq in quest.prerequisites:
            if prereq not in quest_map:
                if quest.name not in missing:
                    missing[quest.name] = []
                missing[quest.name].append(prereq)

    missing_frozen = {k: tuple(v) for k, v in missing.items()}

    # Check for cycles
    cycle_detected = False
    try:
        order = quest_dependency_order(graph)
    except ValueError:
        cycle_detected = True
        order = ()

    # Find unreachable quests (prerequisites missing from graph)
    unreachable: list[str] = list(missing.keys())

    is_completable = not cycle_detected and len(unreachable) == 0

    return CompletabilityResult(
        is_completable=is_completable,
        completion_order=order,
        unreachable_quests=tuple(unreachable),
        cycle_detected=cycle_detected,
        missing_prerequisites=missing_frozen,
    )


# ---------------------------------------------------------------------------
# Session type encoding
# ---------------------------------------------------------------------------

def _quest_chain_to_st(names: tuple[str, ...]) -> SessionType:
    """Encode a linear chain of quest names as nested branches."""
    if not names:
        return End()
    current: SessionType = End()
    for name in reversed(names):
        current = Branch(((name, current),))
    return current


def encode_quest_chain(quests: tuple[Quest, ...]) -> QuestChainResult:
    """Encode a linear quest chain as a session type."""
    names = tuple(q.name for q in quests)
    st = _quest_chain_to_st(names)
    type_str = pretty(st)
    parsed = parse(type_str)
    ss = build_statespace(parsed)
    lr = check_lattice(ss)

    return QuestChainResult(
        session_type_str=type_str,
        chain_length=len(quests),
        state_count=len(ss.states),
        is_lattice=lr.is_lattice,
    )


def encode_quest_graph(graph: QuestGraph) -> str:
    """Encode a quest graph as a session type string.

    Strategy:
    - Compute topological order.
    - Main quests form the spine (sequential branches).
    - Side quests at each level become additional branch choices.
    - Mutually exclusive quests become selections.
    """
    completability = check_completability(graph)
    if completability.cycle_detected:
        raise ValueError("Cannot encode quest graph with cycles")

    quest_map = graph.quest_map
    order = completability.completion_order
    if not order:
        return pretty(End())

    # Group quests by their "depth" (number of prerequisites)
    depth: dict[str, int] = {}
    for name in order:
        q = quest_map[name]
        if not q.prerequisites:
            depth[name] = 0
        else:
            depth[name] = max(depth.get(p, 0) for p in q.prerequisites
                              if p in depth) + 1

    # Group by depth level
    levels: dict[int, list[Quest]] = {}
    for name in order:
        d = depth[name]
        if d not in levels:
            levels[d] = []
        levels[d].append(quest_map[name])

    # Build session type bottom-up (deepest level first)
    max_depth_val = max(levels.keys()) if levels else 0
    current: SessionType = End()

    for d in range(max_depth_val, -1, -1):
        quests_at_level = levels.get(d, [])
        if not quests_at_level:
            continue

        # Separate main vs optional quests
        main_quests = [q for q in quests_at_level if not q.is_optional]
        optional_quests = [q for q in quests_at_level if q.is_optional]

        # Check for mutually exclusive groups
        exclusive_groups: dict[str, list[Quest]] = {}
        non_exclusive: list[Quest] = []
        for q in quests_at_level:
            if q.exclusive_with:
                group_key = tuple(sorted(set(q.exclusive_with) | {q.name}))
                key_str = ",".join(group_key)
                if key_str not in exclusive_groups:
                    exclusive_groups[key_str] = []
                exclusive_groups[key_str].append(q)
            else:
                non_exclusive.append(q)

        choices: list[tuple[str, SessionType]] = []

        # Non-exclusive quests become branch choices
        for q in non_exclusive:
            choices.append((q.name, current))

        # Exclusive groups become selections
        for group in exclusive_groups.values():
            if len(group) == 1:
                choices.append((group[0].name, current))
            else:
                sel_choices = tuple((q.name, current) for q in group)
                sel = Select(sel_choices)
                choices.append((f"choose_{group[0].name}", sel))

        if len(choices) == 1:
            current = Branch((choices[0],))
        elif len(choices) > 1:
            current = Branch(tuple(choices))

    return pretty(current)


def quest_lattice(graph: QuestGraph) -> QuestLatticeResult:
    """Analyze the lattice structure of a quest graph.

    Builds the state space and checks lattice properties.
    """
    type_str = encode_quest_graph(graph)
    st = parse(type_str)
    ss = build_statespace(st)
    lr = check_lattice(ss)

    completability = check_completability(graph)

    main_count = sum(1 for q in graph.quests if q.quest_type == "main")
    side_count = sum(1 for q in graph.quests if q.quest_type == "side")

    # Branching factor: average out-degree in state space
    out_degrees: list[int] = []
    for state in ss.states:
        out = sum(1 for s, _, _ in ss.transitions if s == state)
        out_degrees.append(out)
    avg_branch = sum(out_degrees) / len(out_degrees) if out_degrees else 0.0

    return QuestLatticeResult(
        session_type_str=type_str,
        num_quests=len(graph.quests),
        num_main=main_count,
        num_side=side_count,
        state_count=len(ss.states),
        transition_count=len(ss.transitions),
        is_lattice=lr.is_lattice,
        completion_order=completability.completion_order,
        branching_factor=avg_branch,
    )


# ---------------------------------------------------------------------------
# Branching quests helper
# ---------------------------------------------------------------------------

def branching_quests(
    main_quest: str,
    side_quests: tuple[str, ...],
    continuation: Optional[str] = None,
) -> str:
    """Build a session type with a main quest and optional side quests.

    The player must complete the main quest but may optionally
    complete side quests first.
    """
    cont: SessionType = End()
    if continuation:
        cont = Branch(((continuation, End()),))

    choices: list[tuple[str, SessionType]] = [(main_quest, cont)]
    for sq in side_quests:
        # Side quest leads back to the main quest choice
        inner = Branch(((main_quest, cont),))
        choices.append((sq, inner))

    st = Branch(tuple(choices))
    return pretty(st)


# ---------------------------------------------------------------------------
# Pre-built RPG templates
# ---------------------------------------------------------------------------

def classic_rpg_graph() -> QuestGraph:
    """A classic RPG quest graph template."""
    return QuestGraph(
        title="Classic RPG",
        quests=(
            Quest("prologue", quest_type="main", level_required=1),
            Quest("village_trouble", quest_type="main",
                  prerequisites=("prologue",), level_required=2),
            Quest("herb_gathering", quest_type="side",
                  prerequisites=("prologue",), is_optional=True),
            Quest("dungeon_1", quest_type="main",
                  prerequisites=("village_trouble",), level_required=5),
            Quest("lost_sword", quest_type="side",
                  prerequisites=("village_trouble",), is_optional=True),
            Quest("arena_challenge", quest_type="side",
                  prerequisites=("dungeon_1",), is_optional=True),
            Quest("boss_fight", quest_type="main",
                  prerequisites=("dungeon_1",), level_required=10),
            Quest("ending", quest_type="main",
                  prerequisites=("boss_fight",)),
        ),
    )


def branching_narrative_graph() -> QuestGraph:
    """A quest graph with mutually exclusive branching paths."""
    return QuestGraph(
        title="Branching Narrative",
        quests=(
            Quest("intro", quest_type="main"),
            Quest("crossroads", quest_type="main",
                  prerequisites=("intro",)),
            Quest("join_rebels", quest_type="main",
                  prerequisites=("crossroads",),
                  exclusive_with=("join_empire",)),
            Quest("join_empire", quest_type="main",
                  prerequisites=("crossroads",),
                  exclusive_with=("join_rebels",)),
            Quest("final_battle", quest_type="main",
                  prerequisites=("join_rebels", "join_empire")),
        ),
    )


def open_world_graph() -> QuestGraph:
    """An open-world quest graph with many optional paths."""
    return QuestGraph(
        title="Open World",
        quests=(
            Quest("awaken", quest_type="main"),
            Quest("explore_forest", quest_type="side",
                  prerequisites=("awaken",), is_optional=True),
            Quest("explore_caves", quest_type="side",
                  prerequisites=("awaken",), is_optional=True),
            Quest("explore_ruins", quest_type="side",
                  prerequisites=("awaken",), is_optional=True),
            Quest("gather_allies", quest_type="main",
                  prerequisites=("awaken",)),
            Quest("forge_weapon", quest_type="side",
                  prerequisites=("explore_caves",), is_optional=True),
            Quest("ancient_knowledge", quest_type="side",
                  prerequisites=("explore_ruins",), is_optional=True),
            Quest("final_quest", quest_type="main",
                  prerequisites=("gather_allies",)),
        ),
    )
