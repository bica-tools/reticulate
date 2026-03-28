"""Natural language as session type composition (Step 206).

Every sentence is a session type.  Grammar IS the session type grammar.
Conversation is a multiparty session type.  Translation is metaphor.
Poetry is aesthetic optimization of session types.

Austin's speech act theory maps directly to session type constructors:
- **Assertions** are Branches (the environment may accept or challenge).
- **Commands** are Selections (the speaker chooses what happens).
- **Questions** are Selections that yield Branch responses.
- **Promises** are Selections with expected future fulfillment.
- **Conversations** are sequential compositions of speech acts.

Grice's cooperative maxims become lattice properties:
- **Quantity**: right number of transitions per state.
- **Quality**: all paths reach ``end`` (truthful, well-formed).
- **Relation**: no unreachable states (everything is relevant).
- **Manner**: the state space is orderly (graded lattice).

This module provides:
    ``classify_speech_act(type_str)``     -- classify a type as a speech act.
    ``compose_conversation(acts)``        -- build a conversation type.
    ``translate(source, target_labels)``  -- relabel a speech act (metaphor).
    ``grice_check(ss)``                   -- check Grice's cooperative maxims.
    ``poetic_score(type_str)``            -- aesthetic beauty of a type.
    ``rhetoric_power(type_str)``          -- persuasive power of a type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from reticulate.lattice import check_lattice
from reticulate.mechanisms import metaphor, metaphor_quality
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
class SentenceType:
    """A sentence expressed as a session type.

    Attributes:
        text: Human-readable description of the speech act.
        session_type_str: The session type string representation.
        speech_act: Austin's speech act category.
        illocutionary_force: The illocutionary force (inform, request, etc.).
    """

    text: str
    session_type_str: str
    speech_act: str
    illocutionary_force: str


@dataclass(frozen=True)
class ConversationType:
    """A conversation as a composite session type.

    Attributes:
        turns: The individual speech acts in order.
        session_type_str: The overall session type of the conversation.
        is_cooperative: Whether Grice's maxims are satisfied.
        depth: Maximum nesting depth of the conversation type.
    """

    turns: tuple[SentenceType, ...]
    session_type_str: str
    is_cooperative: bool
    depth: int


@dataclass(frozen=True)
class TranslationResult:
    """Result of translating (relabeling) a speech act.

    Attributes:
        source: The original speech act.
        target: The translated speech act.
        mapping: Label-to-label mapping used.
        fidelity: Quality of the metaphorical mapping (0.0 to 1.0).
        is_faithful: Whether fidelity exceeds 0.8.
    """

    source: SentenceType
    target: SentenceType
    mapping: dict[str, str]
    fidelity: float
    is_faithful: bool


# ---------------------------------------------------------------------------
# Speech Act Library
# ---------------------------------------------------------------------------

SPEECH_ACT_LIBRARY: dict[str, tuple[str, str, str]] = {
    # name -> (type_str, speech_act, illocutionary_force)
    "assertion": (
        "&{claim: +{accept: end, challenge: &{evidence: +{convince: end, retract: end}}}}",
        "assertion",
        "inform",
    ),
    "question": (
        "+{ask: &{answer: +{follow_up: end, accept: end}}}",
        "question",
        "request",
    ),
    "command": (
        "+{order: &{obey: end, refuse: +{insist: &{comply: end, conflict: end}}}}",
        "command",
        "direct",
    ),
    "promise": (
        "+{commit: &{expect: +{fulfill: end, break: +{repair: end}}}}",
        "promise",
        "commit",
    ),
    "greeting": (
        "&{hello: +{hello_back: end}}",
        "greeting",
        "express",
    ),
    "apology": (
        "+{acknowledge: &{forgive: end, resent: end}}",
        "apology",
        "express",
    ),
    "negotiation": (
        "rec X . +{propose: &{accept: end, counter: X, reject: end}}",
        "negotiation",
        "commit",
    ),
    "story": (
        "&{once_upon: +{complication: &{climax: +{resolution: end}}}}",
        "story",
        "inform",
    ),
    "argument": (
        "rec X . +{claim: &{counter: +{rebut: X, concede: end}}}",
        "argument",
        "inform",
    ),
    "lecture": (
        "rec X . +{explain: &{question: +{clarify: X, continue: X, conclude: end}}}",
        "lecture",
        "inform",
    ),
    "confession": (
        "+{reveal: &{judge: +{absolve: end, condemn: end}}}",
        "confession",
        "express",
    ),
    "prayer": (
        "+{petition: &{silence: +{accept: end, persist: end}}}",
        "prayer",
        "request",
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ast_depth(node: SessionType, seen: frozenset[str] = frozenset()) -> int:
    """Compute nesting depth of a session type AST."""
    if isinstance(node, (End, Wait)):
        return 0
    if isinstance(node, Var):
        return 0
    if isinstance(node, Branch):
        if not node.choices:
            return 0
        return 1 + max(
            _ast_depth(cont, seen) for _, cont in node.choices
        )
    if isinstance(node, Select):
        if not node.choices:
            return 0
        return 1 + max(
            _ast_depth(cont, seen) for _, cont in node.choices
        )
    if isinstance(node, Parallel):
        return 1 + max(
            _ast_depth(b, seen) for b in node.branches
        )
    if isinstance(node, Rec):
        if node.var in seen:
            return 0
        return _ast_depth(node.body, seen | {node.var})
    return 0


def _count_paths(ss: StateSpace) -> int:
    """Count distinct top-to-bottom paths via DFS (with cycle cap)."""
    if ss.top == ss.bottom:
        return 1
    count = 0
    stack: list[tuple[int, frozenset[int]]] = [(ss.top, frozenset())]
    while stack:
        state, visited = stack.pop()
        if state == ss.bottom:
            count += 1
            continue
        if state in visited:
            continue
        new_visited = visited | {state}
        for src, _lbl, tgt in ss.transitions:
            if src == state:
                stack.append((tgt, new_visited))
    return max(count, 1)


def _symmetry_score(ss: StateSpace) -> float:
    """Measure symmetry of the state space.

    Compares outgoing degree distribution across states.  Perfectly
    symmetric = all non-terminal states have the same out-degree.
    """
    out_degrees: list[int] = []
    for state in ss.states:
        deg = sum(1 for s, _, _ in ss.transitions if s == state)
        if deg > 0:
            out_degrees.append(deg)
    if not out_degrees:
        return 1.0
    mean = sum(out_degrees) / len(out_degrees)
    if mean == 0:
        return 1.0
    variance = sum((d - mean) ** 2 for d in out_degrees) / len(out_degrees)
    # Normalize: 0 variance = 1.0 symmetry, high variance = low symmetry.
    return 1.0 / (1.0 + variance)


def _is_graded(ss: StateSpace) -> bool:
    """Check if the state space is graded (all maximal chains same length).

    A graded lattice has a rank function such that covering relations
    increase rank by exactly 1.  We check a weaker condition: all
    paths from top to bottom have the same length.
    """
    if ss.top == ss.bottom:
        return True
    # BFS to find all shortest path lengths from top to bottom.
    lengths: set[int] = set()
    queue: list[tuple[int, int, frozenset[int]]] = [(ss.top, 0, frozenset())]
    while queue:
        state, depth, visited = queue.pop(0)
        if state == ss.bottom:
            lengths.add(depth)
            continue
        if state in visited:
            continue
        new_visited = visited | {state}
        for src, _lbl, tgt in ss.transitions:
            if src == state:
                queue.append((tgt, depth + 1, new_visited))
    if not lengths:
        return True
    return len(lengths) == 1


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def classify_speech_act(type_str: str) -> str:
    """Classify a session type as a speech act category.

    Heuristic classification based on the top-level constructor:
    - Branch at top -> "question" (environment chooses).
    - Select at top -> "command" (speaker chooses).
    - Recursive -> "conversation" (ongoing exchange).
    - End -> "silence" (nothing to say).

    Returns one of: "question", "command", "conversation", "silence".
    """
    ast = parse(type_str)
    # Unwrap Rec to get the body's top-level constructor.
    inner = ast
    is_recursive = False
    while isinstance(inner, Rec):
        is_recursive = True
        inner = inner.body
    if isinstance(inner, (End, Wait)):
        return "silence"
    if is_recursive:
        return "conversation"
    if isinstance(inner, Branch):
        return "question"
    if isinstance(inner, Select):
        return "command"
    return "question"


def compose_conversation(acts: list[str]) -> str:
    """Build a conversation type from speech act names.

    Each named act becomes a turn.  Turns are composed sequentially by
    replacing the ``end`` leaves of one turn with the next turn's type.
    Returns the pretty-printed composite session type string.

    Raises ``KeyError`` if an act name is not in the library.
    """
    if not acts:
        return pretty(End())
    asts: list[SessionType] = []
    for act_name in acts:
        if act_name not in SPEECH_ACT_LIBRARY:
            raise KeyError(f"Unknown speech act: {act_name!r}")
        type_str, _sa, _il = SPEECH_ACT_LIBRARY[act_name]
        asts.append(parse(type_str))

    # Sequential composition: replace end in each AST with the next.
    def _chain(a: SessionType, b: SessionType) -> SessionType:
        return _replace_end(a, b)

    composed = asts[0]
    for nxt in asts[1:]:
        composed = _chain(composed, nxt)
    return pretty(composed)


def _replace_end(node: SessionType, replacement: SessionType) -> SessionType:
    """Replace all End nodes with *replacement*."""
    if isinstance(node, End):
        return replacement
    if isinstance(node, (Var, Wait)):
        return node
    if isinstance(node, Branch):
        return Branch(
            tuple((m, _replace_end(c, replacement)) for m, c in node.choices)
        )
    if isinstance(node, Select):
        return Select(
            tuple((m, _replace_end(c, replacement)) for m, c in node.choices)
        )
    if isinstance(node, Parallel):
        return Parallel(
            tuple(_replace_end(b, replacement) for b in node.branches)
        )
    if isinstance(node, Rec):
        return Rec(node.var, _replace_end(node.body, replacement))
    return node  # pragma: no cover


def translate(source_act: str, target_labels: dict[str, str]) -> TranslationResult:
    """Translate a speech act by relabeling (metaphor).

    Takes a speech act name and a label mapping, applies the relabeling,
    computes fidelity as the quality of the metaphorical mapping.

    Raises ``KeyError`` if the source act is not in the library.
    """
    if source_act not in SPEECH_ACT_LIBRARY:
        raise KeyError(f"Unknown speech act: {source_act!r}")
    type_str, sa, il = SPEECH_ACT_LIBRARY[source_act]
    src_ast = parse(type_str)
    tgt_ast = metaphor(src_ast, target_labels)
    tgt_str = pretty(tgt_ast)

    # Compute fidelity via metaphor_quality on state spaces.
    src_ss = build_statespace(src_ast)
    tgt_ss = build_statespace(tgt_ast)
    fidelity = metaphor_quality(target_labels, src_ss, tgt_ss)

    source_st = SentenceType(
        text=source_act,
        session_type_str=type_str,
        speech_act=sa,
        illocutionary_force=il,
    )
    target_st = SentenceType(
        text=f"translated_{source_act}",
        session_type_str=tgt_str,
        speech_act=sa,
        illocutionary_force=il,
    )

    return TranslationResult(
        source=source_st,
        target=target_st,
        mapping=dict(target_labels),
        fidelity=fidelity,
        is_faithful=fidelity > 0.8,
    )


def grice_check(ss: StateSpace) -> dict[str, bool]:
    """Check Grice's cooperative maxims on a state space.

    Returns a dict with four boolean entries:
    - "quantity": not too many / too few transitions per state
      (average out-degree between 1 and 5).
    - "quality": all paths from top can reach bottom (truthful/well-formed).
    - "relation": no unreachable states (everything is relevant).
    - "manner": the state space is orderly (graded).
    """
    # Quantity: average out-degree in [1, 5].
    out_degrees: list[int] = []
    for state in ss.states:
        deg = sum(1 for s, _, _ in ss.transitions if s == state)
        out_degrees.append(deg)
    avg_out = sum(out_degrees) / max(len(out_degrees), 1)
    quantity = 1.0 <= avg_out <= 5.0

    # Quality: BFS from top reaches bottom, and all states can reach bottom.
    reachable_from_top: set[int] = set()
    queue = [ss.top]
    while queue:
        s = queue.pop(0)
        if s in reachable_from_top:
            continue
        reachable_from_top.add(s)
        for src, _lbl, tgt in ss.transitions:
            if src == s:
                queue.append(tgt)
    # Check bottom reachable from all states reachable from top.
    quality = True
    for state in reachable_from_top:
        reachable: set[int] = set()
        q2 = [state]
        while q2:
            current = q2.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            for src, _lbl, tgt in ss.transitions:
                if src == current:
                    q2.append(tgt)
        if ss.bottom not in reachable:
            quality = False
            break

    # Relation: all states are reachable from top.
    relation = reachable_from_top == ss.states

    # Manner: orderly = graded.
    manner = _is_graded(ss)

    return {
        "quantity": quantity,
        "quality": quality,
        "relation": relation,
        "manner": manner,
    }


def poetic_score(type_str: str) -> float:
    """Compute the aesthetic / poetic beauty of a session type.

    Beauty is the average of:
    - symmetry (uniform branching),
    - brevity (fewer states is more concise, normalized),
    - balance (depth vs width ratio close to golden ratio).

    Returns a float in [0.0, 1.0].
    """
    ast = parse(type_str)
    ss = build_statespace(ast)

    # Try to use the aesthetics module's beauty_score.
    try:
        from reticulate.aesthetics import beauty_score
        return beauty_score(ss)
    except Exception:
        pass

    # Fallback: compute from first principles.
    symmetry = _symmetry_score(ss)
    # Brevity: 1.0 for 2 states, decreasing as states increase.
    n_states = len(ss.states)
    brevity = 1.0 / (1.0 + math.log2(max(n_states, 1)))
    # Balance: depth-to-width ratio vs golden ratio (1.618).
    depth = _ast_depth(ast)
    width = n_states
    if width > 0 and depth > 0:
        ratio = depth / width
        golden = (1 + math.sqrt(5)) / 2
        balance = 1.0 / (1.0 + abs(ratio - 1.0 / golden))
    else:
        balance = 0.5

    return (symmetry + brevity + balance) / 3.0


def rhetoric_power(type_str: str) -> float:
    """Compute the persuasive / rhetorical power of a session type.

    Power is the product of:
    - path diversity (log2 of path count = information capacity),
    - engagement (turn-taking measured by alternation of Branch/Select).

    Returns a non-negative float (higher = more powerful).
    """
    ast = parse(type_str)
    ss = build_statespace(ast)

    # Path diversity = channel capacity.
    try:
        from reticulate.information import channel_capacity
        diversity = channel_capacity(ss)
    except Exception:
        n_paths = _count_paths(ss)
        diversity = math.log2(max(n_paths, 1))

    # Engagement = fraction of states that alternate between Branch and Select.
    alternations = 0
    total_transitions = 0
    for src, _lbl, tgt in ss.transitions:
        src_is_selection = any(
            (s, l, t) in ss.selection_transitions
            for s, l, t in ss.transitions if s == src
        )
        tgt_is_selection = any(
            (s, l, t) in ss.selection_transitions
            for s, l, t in ss.transitions if s == tgt
        )
        total_transitions += 1
        if src_is_selection != tgt_is_selection:
            alternations += 1

    engagement = alternations / max(total_transitions, 1)

    return diversity * max(engagement, 0.1)
