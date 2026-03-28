"""Mythological session types: archetypes as universal protocols (Step 205).

Jung's archetypes are universal interaction patterns embedded in the
collective unconscious.  They ARE session types -- the same patterns
recurring across every culture, every story, every human life.
Campbell's hero journey is a recursive session type with branching
choices.  Greek myths are protocols with sacrifice points and rebirth
points.  Tarot spreads are state-space explorations.

The key insight: archetypes are structurally identical to session types.
Every mythological narrative has:
- **Branch points** (choices the hero faces),
- **Selection points** (actions the hero takes),
- **Recursion** (cycles of death and rebirth, eternal return),
- **Termination** (the hero's journey ends -- or begins again).

This module provides:
    ``get_archetype(name)``               -- look up an archetype by name.
    ``shadow_of(name)``                   -- the shadow (negated) version.
    ``analyze_myth(type_str)``            -- mythological journey analysis.
    ``mythological_analogy(n1, n2)``      -- compatibility between archetypes.
    ``quest(start, challenges)``          -- compose a mythological quest.
    ``all_archetypes_form_lattices()``    -- verify all archetypes form lattices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

from reticulate.lattice import check_lattice
from reticulate.mechanisms import dialectic, negate_type
from reticulate.negotiation import compatibility_score
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
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Archetype:
    """A mythological archetype expressed as a session type.

    Attributes:
        name: Canonical name (e.g. "hero", "shadow", "trickster").
        domain: Mythological tradition ("jungian", "campbellian", "greek", "tarot").
        session_type_str: The archetype as a session type string.
        shadow_type_str: The negated / dark version (Branch <-> Select).
        description: Short human-readable description.
    """

    name: str
    domain: str
    session_type_str: str
    shadow_type_str: str
    description: str


@dataclass(frozen=True)
class MythAnalysis:
    """Analysis of a session type as a mythological journey.

    Attributes:
        archetype: Best-matching archetype name (or "unknown").
        hero_journey_phase: Campbell phase at the entry point.
        transformation_count: Number of state changes (transitions).
        sacrifice_points: States where branch count decreases vs parent.
        rebirth_points: States where new branches open vs parent.
        is_circular: Whether the journey ends where it began (recursive).
    """

    archetype: str
    hero_journey_phase: str
    transformation_count: int
    sacrifice_points: tuple[int, ...]
    rebirth_points: tuple[int, ...]
    is_circular: bool


# ---------------------------------------------------------------------------
# Campbell's hero journey phases (mapped by depth from initial state)
# ---------------------------------------------------------------------------

_HERO_PHASES: tuple[str, ...] = (
    "ordinary_world",         # depth 0: the starting point
    "call_to_adventure",      # depth 1: first transition
    "refusal_of_the_call",    # depth 2: hesitation
    "meeting_the_mentor",     # depth 3: guidance
    "crossing_the_threshold", # depth 4: commitment
    "tests_allies_enemies",   # depth 5: trials
    "approach_to_inmost_cave",# depth 6: preparation
    "ordeal",                 # depth 7: the abyss
    "reward",                 # depth 8: transformation
    "the_road_back",          # depth 9: return begins
    "resurrection",           # depth 10: final test
    "return_with_elixir",     # depth 11+: homecoming
)


def _phase_at_depth(depth: int) -> str:
    """Map a depth to a Campbell hero journey phase."""
    if depth < 0:
        return _HERO_PHASES[0]
    if depth >= len(_HERO_PHASES):
        return _HERO_PHASES[-1]
    return _HERO_PHASES[depth]


# ---------------------------------------------------------------------------
# The Archetype Library
# ---------------------------------------------------------------------------

def _shadow(type_str: str) -> str:
    """Compute the shadow (negation) of a session type string."""
    ast = parse(type_str)
    return pretty(negate_type(ast))


ARCHETYPE_LIBRARY: dict[str, Archetype] = {}

_ARCHETYPE_DEFS: list[tuple[str, str, str, str]] = [
    # (name, domain, type_str, description)
    ("hero",
     "campbellian",
     "rec X . &{call: +{refuse: &{mentor: +{threshold: &{trials: +{abyss: &{transformation: +{return: X, transcend: end}}}}}}}}",
     "The Hero's Journey: call, refusal, mentor, threshold, trials, abyss, transformation, return or transcendence"),
    ("shadow",
     "jungian",
     "rec X . +{tempt: &{resist: X, surrender: +{consume: end}}}",
     "The Shadow: the dark double that tempts and consumes"),
    ("trickster",
     "jungian",
     "rec X . &{order: +{disrupt: &{chaos: +{new_order: X, destruction: end}}}}",
     "The Trickster: disrupts order, creates chaos, births new order or destruction"),
    ("mother",
     "jungian",
     "rec X . &{need: +{nurture: &{grow: +{release: end, cling: X}}}}",
     "The Great Mother: nurtures, enables growth, then releases or clings"),
    ("father",
     "jungian",
     "&{challenge: +{test: &{pass: +{bestow: end}, fail: +{exile: end}}}}",
     "The Father: challenges, tests, bestows or exiles"),
    ("child",
     "jungian",
     "rec X . &{wonder: +{explore: &{discover: +{integrate: X, overwhelm: end}}}}",
     "The Divine Child: wonder, exploration, discovery, integration or overwhelm"),
    ("wise_old",
     "jungian",
     "&{seek: +{riddle: &{answer: +{wisdom: end, fool: end}}}}",
     "The Wise Old Man: dispenses riddles; the seeker gains wisdom or remains a fool"),
    ("lover",
     "jungian",
     "rec X . &{encounter: +{attraction: &{union: +{transform: X, separate: end}}}}",
     "The Lover: encounter, attraction, union, transformation or separation"),
    ("death_rebirth",
     "campbellian",
     "&{crisis: +{surrender: &{dissolution: +{rebirth: end}}}}",
     "Death-Rebirth: crisis, surrender, dissolution, rebirth"),
    ("ouroboros",
     "greek",
     "rec X . &{create: +{sustain: &{destroy: +{recreate: X, dissolve: end}}}}",
     "The Ouroboros: eternal cycle of creation, sustenance, destruction, recreation or dissolution"),
    ("prometheus",
     "greek",
     "&{steal_fire: +{gift: &{punishment: +{endurance: end}}}}",
     "Prometheus: steals fire, gifts it, suffers punishment, endures"),
    ("orpheus",
     "greek",
     "&{descend: +{seek: &{find: +{look_back: end, trust: end}}}}",
     "Orpheus: descends to the underworld, seeks the beloved, looks back or trusts"),
    ("sisyphus",
     "greek",
     "rec X . &{push: +{summit: &{fall: X, collapse: end}}}",
     "Sisyphus: pushes the boulder, reaches the summit, falls and repeats, or collapses"),
    ("phoenix",
     "greek",
     "rec X . &{burn: +{ashes: &{rise: X, extinguish: end}}}",
     "The Phoenix: burns, becomes ashes, rises again or finally extinguishes"),
    ("labyrinth",
     "greek",
     "rec X . &{enter: +{left: X, right: X, center: end}}",
     "The Labyrinth: at each junction choose left, right, or find the center"),
    ("flood",
     "campbellian",
     "&{corruption: +{cleanse: &{renewal: +{covenant: end}}}}",
     "The Great Flood: corruption, cleansing, renewal, covenant"),
    ("tower_of_babel",
     "campbellian",
     "&{unite: +{build: &{hubris: +{scatter: end}}}}",
     "Tower of Babel: unity, construction, hubris, scattering"),
    ("garden_of_eden",
     "campbellian",
     "&{innocence: +{temptation: &{knowledge: +{exile: end}}}}",
     "Garden of Eden: innocence, temptation, knowledge, exile"),
    ("apocalypse",
     "campbellian",
     "&{signs: +{tribulation: &{judgment: +{new_world: end}}}}",
     "Apocalypse: signs, tribulation, judgment, new world"),
    ("creation",
     "campbellian",
     "&{void: +{word: &{light: +{order: &{life: +{rest: end}}}}}}",
     "Creation: void, word, light, order, life, rest"),
    ("fool",
     "tarot",
     "rec X . &{leap: +{fall: &{learn: +{rise: X, stay: end}}}}",
     "The Fool (Tarot 0): leaps into the unknown, falls, learns, rises or stays"),
    ("magician",
     "tarot",
     "&{will: +{channel: &{manifest: +{transform: end}}}}",
     "The Magician (Tarot I): will, channeling, manifestation, transformation"),
    ("high_priestess",
     "tarot",
     "&{veil: +{intuition: &{mystery: +{reveal: end, conceal: end}}}}",
     "The High Priestess (Tarot II): veil, intuition, mystery, revelation or concealment"),
    ("wheel_of_fortune",
     "tarot",
     "rec X . &{ascend: +{peak: &{descend: +{nadir: X, stillness: end}}}}",
     "Wheel of Fortune (Tarot X): ascend, peak, descend, nadir or stillness"),
]


def _build_library() -> None:
    """Populate the archetype library on module load."""
    for name, domain, type_str, description in _ARCHETYPE_DEFS:
        shadow_str = _shadow(type_str)
        ARCHETYPE_LIBRARY[name] = Archetype(
            name=name,
            domain=domain,
            session_type_str=type_str,
            shadow_type_str=shadow_str,
            description=description,
        )


_build_library()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_archetype(name: str) -> Archetype:
    """Look up an archetype by name.

    Raises ``KeyError`` if the name is not in the library.
    """
    if name not in ARCHETYPE_LIBRARY:
        raise KeyError(f"Unknown archetype: {name!r}")
    return ARCHETYPE_LIBRARY[name]


def shadow_of(name: str) -> str:
    """Return the shadow (negated) version of an archetype's session type.

    The shadow swaps Branch <-> Select at every choice point, representing
    the dark mirror of the archetype.
    """
    archetype = get_archetype(name)
    return archetype.shadow_type_str


def analyze_myth(type_str: str) -> MythAnalysis:
    """Analyze any session type as a mythological journey.

    Maps the type's structure onto Campbell's hero journey phases by
    measuring depth from the initial state.  Identifies sacrifice points
    (states where the number of outgoing branches decreases relative to
    predecessors) and rebirth points (states where new branches open).
    """
    ast = parse(type_str)
    ss = build_statespace(ast)

    # Compute depths via BFS from top.
    depths: dict[int, int] = {ss.top: 0}
    queue = [ss.top]
    visited: set[int] = {ss.top}
    while queue:
        current = queue.pop(0)
        for src, _lbl, tgt in ss.transitions:
            if src == current and tgt not in visited:
                visited.add(tgt)
                depths[tgt] = depths[current] + 1
                queue.append(tgt)

    # Outgoing branch count per state.
    outgoing: dict[int, int] = {}
    for state in ss.states:
        outgoing[state] = sum(1 for s, _, _ in ss.transitions if s == state)

    # Sacrifice points: states where outgoing count < max predecessor outgoing.
    # Rebirth points: states where outgoing count > min predecessor outgoing.
    predecessors: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _lbl, tgt in ss.transitions:
        predecessors[tgt].append(src)

    sacrifice_points: list[int] = []
    rebirth_points: list[int] = []
    for state in sorted(ss.states):
        preds = predecessors[state]
        if not preds:
            continue
        max_pred_out = max(outgoing.get(p, 0) for p in preds)
        min_pred_out = min(outgoing.get(p, 0) for p in preds)
        state_out = outgoing.get(state, 0)
        if state_out < max_pred_out:
            sacrifice_points.append(state)
        if state_out > min_pred_out:
            rebirth_points.append(state)

    # Circularity: does any transition target the top state?
    is_circular = any(tgt == ss.top for _, _, tgt in ss.transitions)

    # Best-matching archetype: try each and pick the one with highest
    # compatibility score (by state space structure).
    best_archetype = "unknown"
    best_score = -1.0
    try:
        for name, arch in ARCHETYPE_LIBRARY.items():
            arch_ast = parse(arch.session_type_str)
            score = compatibility_score(ast, arch_ast)
            if score > best_score:
                best_score = score
                best_archetype = name
    except Exception:
        pass

    # Hero journey phase from max depth.
    max_depth = max(depths.values()) if depths else 0
    phase = _phase_at_depth(min(max_depth, len(_HERO_PHASES) - 1))

    return MythAnalysis(
        archetype=best_archetype,
        hero_journey_phase=phase,
        transformation_count=len(ss.transitions),
        sacrifice_points=tuple(sacrifice_points),
        rebirth_points=tuple(rebirth_points),
        is_circular=is_circular,
    )


def mythological_analogy(name1: str, name2: str) -> float:
    """Compute compatibility score between two archetypes.

    Returns a float in [0.0, 1.0] measuring structural similarity of
    the archetypes' session types.  1.0 = identical structure.
    """
    a1 = get_archetype(name1)
    a2 = get_archetype(name2)
    ast1 = parse(a1.session_type_str)
    ast2 = parse(a2.session_type_str)
    return compatibility_score(ast1, ast2)


def _free_vars(node: SessionType, bound: frozenset[str] = frozenset()) -> set[str]:
    """Collect free type variables in an AST."""
    if isinstance(node, (End, Wait)):
        return set()
    if isinstance(node, Var):
        return {node.name} if node.name not in bound else set()
    if isinstance(node, Branch):
        result: set[str] = set()
        for _, c in node.choices:
            result |= _free_vars(c, bound)
        return result
    if isinstance(node, Select):
        result = set()
        for _, c in node.choices:
            result |= _free_vars(c, bound)
        return result
    if isinstance(node, Parallel):
        result = set()
        for b in node.branches:
            result |= _free_vars(b, bound)
        return result
    if isinstance(node, Rec):
        return _free_vars(node.body, bound | {node.var})
    return set()  # pragma: no cover


def quest(start_archetype: str, challenges: list[str]) -> str:
    """Compose a mythological quest by dialectic synthesis.

    Start with the given archetype's session type, then fold each
    challenge archetype into the accumulated type via Hegelian dialectic.
    Free variables from recursive types are re-wrapped in ``rec``.
    Returns the pretty-printed synthesized type string.
    """
    arch = get_archetype(start_archetype)
    accumulated = parse(arch.session_type_str)
    for challenge_name in challenges:
        challenge_arch = get_archetype(challenge_name)
        challenge_ast = parse(challenge_arch.session_type_str)
        accumulated = dialectic(accumulated, challenge_ast)
    # Re-wrap any free variables that lost their binder during dialectic.
    free = _free_vars(accumulated)
    for var_name in sorted(free):
        accumulated = Rec(var_name, accumulated)
    return pretty(accumulated)


def all_archetypes_form_lattices() -> bool:
    """Verify that all archetypes in the library parse and form lattices.

    Returns True iff every archetype's session type string parses
    successfully, builds a valid state space, and passes the lattice
    check.
    """
    for name, arch in ARCHETYPE_LIBRARY.items():
        try:
            ast = parse(arch.session_type_str)
            ss = build_statespace(ast)
            result = check_lattice(ss)
            if not result.is_lattice:
                return False
        except Exception:
            return False
    return True
