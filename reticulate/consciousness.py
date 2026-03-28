"""Consciousness as meta-session type (Step 200).

The radical idea: if the 8 mechanisms of thought are session type
operations, then CONSCIOUSNESS ITSELF is a meta-session-type — a
protocol for selecting and applying protocols.  Awareness is the state;
attention is the transition; the stream of consciousness is the trace.

Key concepts:

- **Self-type**: consciousness modelled as a recursive session type
  that perceives, attends, processes, and reflects (or acts, or dreams).
- **Stream of consciousness**: a trace through the self-type, recording
  which mechanisms fire at each step and what they attend to.
- **Qualia categorization**: how consciousness *experiences* a session
  type — measuring complexity, beauty, meaning, and emotional valence.
- **Attention filter**: selective attention as state-space abstraction,
  keeping only the transitions consciousness attends to.
- **Metacognition**: thinking about thinking, modelled as nesting the
  self-type within itself to arbitrary depth.
- **Dream**: random metaphorical remapping of a type's labels —
  structurally identical but semantically scrambled.
- **Flow state**: Csikszentmihalyi's flow detected via pendular
  balance, moderate entropy, and intermediate binding energy.

This module provides:
    ``self_type()``                  — the recursive meta-protocol of consciousness.
    ``simulate_stream(types, mechs)``— simulate a stream of consciousness.
    ``categorize_qualia(type_str)``  — experiential categorization of a type.
    ``attention_filter(ss, focus)``  — selective attention as abstraction.
    ``metacognition(type_str, d)``   — thinking about thinking (depth d).
    ``dream(type_str, seed)``        — dream-scramble a type's labels.
    ``flow_state(ss)``               — detect Csikszentmihalyi's flow.
"""

from __future__ import annotations

import math
import random
import re
from collections import deque
from dataclasses import dataclass

from reticulate import parse, build_statespace
from reticulate.chaos import classify_dynamics
from reticulate.information import branching_entropy
from reticulate.mechanisms import metaphor
from reticulate.negotiation import compatibility_score
from reticulate.orbital import compute_orbits, planetary_orbits
from reticulate.origami import prune_unreachable
from reticulate.parser import Branch, End, Rec, Select, SessionType, Var, pretty
from reticulate.pendular import is_pendular
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsciousnessState:
    """A single moment of conscious experience.

    Attributes:
        attention_target: What consciousness is attending to (a type string
            or ``"self"`` for metacognitive moments).
        awareness_level: Intensity of awareness, 0 (unconscious) to 1 (vivid).
        active_mechanisms: Which cognitive mechanisms are currently firing.
        metacognitive_depth: How many levels of self-reflection (0 = none).
        stream_position: Index in the stream of consciousness.
    """

    attention_target: str
    awareness_level: float
    active_mechanisms: tuple[str, ...]
    metacognitive_depth: int
    stream_position: int


@dataclass(frozen=True)
class StreamOfConsciousness:
    """A trace through the meta-session-type of consciousness.

    Attributes:
        states: Sequence of conscious states.
        transitions: Triples ``(from_idx, mechanism_applied, to_idx)``.
        total_entropy: Shannon entropy of the mechanism frequency distribution.
        metacognitive_moments: Number of states where attention is on ``"self"``.
        flow_score: Fraction of transitions using the same or related mechanism
            as the previous transition (smooth continuity).
    """

    states: tuple[ConsciousnessState, ...]
    transitions: tuple[tuple[int, str, int], ...]
    total_entropy: float
    metacognitive_moments: int
    flow_score: float


@dataclass(frozen=True)
class QualiaCategorization:
    """How consciousness *experiences* a session type.

    Attributes:
        raw_type: The session type string being experienced.
        perceived_complexity: Branching entropy of the type's state space.
        perceived_beauty: Symmetry ratio from orbital analysis.
        perceived_meaning: Compatibility score with the self-type.
        emotional_valence: Positive if pendular/ordered, negative if chaotic.
    """

    raw_type: str
    perceived_complexity: float
    perceived_beauty: float
    perceived_meaning: float
    emotional_valence: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELF_TYPE_STR: str = (
    "rec Self . &{perceive: +{attend: &{process: "
    "+{reflect: Self, act: end, dream: Self}}}}"
)

_MECHANISM_NAMES: tuple[str, ...] = (
    "abstraction",
    "composition",
    "negation",
    "recursion",
    "emergence",
    "dialectic",
    "metaphor",
    "analogy",
)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def self_type() -> str:
    """Return the session type of consciousness itself.

    Consciousness is a recursive meta-protocol: perceive the world,
    attend to something, process it, then either reflect (recurse into
    metacognition), act (terminate the stream), or dream (recurse with
    scrambled content).
    """
    return _SELF_TYPE_STR


def simulate_stream(
    types: list[str],
    mechanisms: list[str],
    steps: int = 20,
) -> StreamOfConsciousness:
    """Simulate a stream of consciousness.

    Starting from the first type in *types*, at each step apply the next
    mechanism (cycling through *mechanisms*) to shift attention.  The
    mechanism ``"recursion"`` applied when the current target is the
    self-type triggers a metacognitive moment (attention on ``"self"``).

    Parameters:
        types: Session type strings forming the pool of attention targets.
        mechanisms: Mechanism names to cycle through.
        steps: Number of stream steps to simulate.

    Returns:
        A :class:`StreamOfConsciousness` with computed entropy, metacognitive
        moment count, and flow score.
    """
    if not types:
        types = ["end"]
    if not mechanisms:
        mechanisms = list(_MECHANISM_NAMES)

    states: list[ConsciousnessState] = []
    transitions: list[tuple[int, str, int]] = []

    current_target = types[0]
    type_idx = 0

    for i in range(steps):
        mech = mechanisms[i % len(mechanisms)]
        is_meta = mech == "recursion" and current_target == self_type()
        target = "self" if is_meta else current_target
        depth = 1 if is_meta else 0
        # Awareness decays toward edges, peaks in middle of stream.
        progress = (i + 1) / steps if steps > 0 else 1.0
        awareness = 1.0 - abs(2.0 * progress - 1.0)  # peak at midpoint
        awareness = round(max(0.01, awareness), 4)

        states.append(ConsciousnessState(
            attention_target=target,
            awareness_level=awareness,
            active_mechanisms=(mech,),
            metacognitive_depth=depth,
            stream_position=i,
        ))

        # Transition to next state
        if i < steps - 1:
            next_mech = mechanisms[(i + 1) % len(mechanisms)]
            transitions.append((i, mech, i + 1))
            # Shift attention: cycle through types on each new mechanism
            if next_mech != mech:
                type_idx = (type_idx + 1) % len(types)
                current_target = types[type_idx]

    # Compute entropy of mechanism frequency distribution.
    mech_counts: dict[str, int] = {}
    for s in states:
        for m in s.active_mechanisms:
            mech_counts[m] = mech_counts.get(m, 0) + 1
    total = sum(mech_counts.values())
    entropy = 0.0
    if total > 0:
        for count in mech_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
    entropy = round(entropy, 6)

    metacognitive_moments = sum(
        1 for s in states if s.attention_target == "self"
    )

    # Flow score: fraction of consecutive transitions using the same mechanism.
    smooth = 0
    for j in range(1, len(states)):
        if states[j].active_mechanisms == states[j - 1].active_mechanisms:
            smooth += 1
    flow_score = round(smooth / max(1, len(states) - 1), 6)

    return StreamOfConsciousness(
        states=tuple(states),
        transitions=tuple(transitions),
        total_entropy=entropy,
        metacognitive_moments=metacognitive_moments,
        flow_score=flow_score,
    )


def categorize_qualia(type_str: str) -> QualiaCategorization:
    """Categorize a session type as a conscious experience.

    Computes four dimensions of qualitative experience:

    - **Complexity**: branching entropy of the state space.
    - **Beauty**: symmetry ratio from orbital analysis (higher = more
      beautiful — regular, symmetric structures are perceived as elegant).
    - **Meaning**: compatibility score with the self-type (how well the
      type "resonates" with consciousness itself).
    - **Valence**: positive if the type is pendular (balanced, ordered),
      negative if classified as chaotic, near zero otherwise.
    """
    ast = parse(type_str)
    ss = build_statespace(ast)

    # Complexity: branching entropy.
    complexity = round(branching_entropy(ss), 6)

    # Beauty: symmetry ratio from orbital analysis.
    orbits = compute_orbits(ss)
    beauty = round(orbits.symmetry_ratio, 6) if orbits.symmetry_ratio else 0.0

    # Meaning: compatibility with self-type.
    self_ast = parse(self_type())
    meaning = round(compatibility_score(ast, self_ast), 6)

    # Valence: pendular → positive, chaotic → negative.
    pendular = is_pendular(ss)
    chaos = classify_dynamics(ss)
    if pendular:
        valence = 0.5
    elif chaos.is_chaotic:
        valence = -0.5
    else:
        valence = 0.0

    return QualiaCategorization(
        raw_type=type_str,
        perceived_complexity=complexity,
        perceived_beauty=beauty,
        perceived_meaning=meaning,
        emotional_valence=valence,
    )


def attention_filter(ss: StateSpace, focus: set[str]) -> StateSpace:
    """Model selective attention by filtering transitions.

    Keep only transitions whose labels are in *focus*.  Remove states
    that become unreachable.  This captures the fundamental cognitive
    act of attention: consciousness only processes what it selects.

    Parameters:
        ss: The full state space.
        focus: Set of transition labels to attend to.

    Returns:
        A pruned :class:`StateSpace` containing only attended transitions.
    """
    filtered_transitions = tuple(
        (src, lbl, tgt) for src, lbl, tgt in ss.transitions if lbl in focus
    )
    filtered_selections = frozenset(
        (src, lbl, tgt)
        for src, lbl, tgt in ss.selection_transitions
        if lbl in focus
    )
    intermediate = StateSpace(
        states=ss.states,
        transitions=list(filtered_transitions),
        top=ss.top,
        bottom=ss.bottom,
        labels=ss.labels,
        selection_transitions=set(filtered_selections),
    )
    return prune_unreachable(intermediate)


def metacognition(type_str: str, depth: int = 1) -> str:
    """Think about thinking: nest the self-type around *type_str*.

    At depth 0, returns *type_str* unchanged.  At depth 1, wraps it in
    the self-type's perception-attention-processing cycle.  Each
    additional depth level adds another layer of self-reflection.

    Parameters:
        type_str: The session type to reflect upon.
        depth: Number of metacognitive nesting levels.

    Returns:
        A session type string with *depth* layers of metacognition.
    """
    if depth <= 0:
        return type_str
    # Wrap: perceive → attend → process → {reflect on inner, act, dream}
    inner = metacognition(type_str, depth - 1)
    return (
        f"&{{perceive: +{{attend: &{{process: "
        f"+{{reflect: {inner}, act: end, dream: {inner}}}}}}}}}"
    )


def dream(type_str: str, seed: int = 42) -> str:
    """Dream-scramble a session type's labels.

    Uses a seeded RNG to randomly permute all method/label names in the
    type, producing a structurally identical but semantically scrambled
    "dream version".  The structure (branching, selection, recursion) is
    preserved — only the names change.

    Parameters:
        type_str: The session type to dream about.
        seed: Random seed for reproducibility.

    Returns:
        A session type string with permuted labels.
    """
    ast = parse(type_str)
    labels = _collect_labels(ast)
    if not labels:
        return type_str

    rng = random.Random(seed)
    label_list = sorted(labels)
    shuffled = label_list[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(label_list, shuffled))
    dreamed = metaphor(ast, mapping)
    return pretty(dreamed)


def flow_state(ss: StateSpace) -> bool:
    """Detect Csikszentmihalyi's flow in a state space.

    Flow occurs when:
    1. The type is pendular (balanced challenge/skill alternation).
    2. Branching entropy is moderate (not boring, not overwhelming):
       between 0.3 and 2.5 bits.
    3. Binding energy is between 0.3 and 0.7 (engaged but not trapped).

    Returns:
        True if the state space exhibits flow characteristics.
    """
    if not ss.states or len(ss.states) < 2:
        return False

    # Condition 1: pendular balance.
    if not is_pendular(ss):
        return False

    # Condition 2: moderate entropy.
    entropy = branching_entropy(ss)
    if entropy < 0.3 or entropy > 2.5:
        return False

    # Condition 3: moderate binding energy.
    p = planetary_orbits(ss)
    if p.binding_energy < 0.3 or p.binding_energy > 0.7:
        return False

    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_labels(node: SessionType) -> set[str]:
    """Recursively collect all method/label names from an AST."""
    if isinstance(node, End):
        return set()
    if isinstance(node, Var):
        return set()
    if isinstance(node, Branch):
        labels: set[str] = set()
        for name, cont in node.choices:
            labels.add(name)
            labels |= _collect_labels(cont)
        return labels
    if isinstance(node, Select):
        labels = set()
        for name, cont in node.choices:
            labels.add(name)
            labels |= _collect_labels(cont)
        return labels
    if isinstance(node, Rec):
        return _collect_labels(node.body)
    # Parallel, Continuation, etc.
    result: set[str] = set()
    for child in getattr(node, "__dataclass_fields__", {}).keys():
        val = getattr(node, child)
        if isinstance(val, tuple):
            for item in val:
                if isinstance(item, SessionType):
                    result |= _collect_labels(item)
        elif isinstance(val, SessionType):
            result |= _collect_labels(val)
    return result
