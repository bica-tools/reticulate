"""Pedagogical session types: scaffolding and dialogue (Step 60e).

Connects session type theory to educational/pedagogical concepts:

- **Scaffolding**: A teacher type that is a supertype of the learner type
  provides a scaffold — the teacher offers more methods (ZPD) that the
  learner can progressively acquire.
- **Zone of Proximal Development (ZPD)**: The methods present in the
  teacher type but absent from the learner type.  These are capabilities
  the learner can acquire with guidance.
- **Turn-taking in dialogue**: Pendular session types model alternating
  teacher-learner dialogue (Socratic method).  Branch states represent
  teacher turns (questions), select states represent learner turns (answers).
- **Developmental stages**: A sequence of session types of increasing
  capability, classified as novice/intermediate/advanced/expert relative
  to a target type.

This module provides:
    ``analyze_scaffolding(teacher, learner)`` — scaffolding analysis.
    ``zone_of_proximal_development(teacher, learner)`` — ZPD gap methods.
    ``analyze_dialogue(ss)`` — turn-taking dialogue analysis.
    ``developmental_stages(types)`` — classify a progression of types.
    ``learning_path(teacher, learner)`` — ordered list of methods to acquire.
    ``classify_pedagogy(teacher, learner, ss)`` — combined analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import Branch, End, Rec, Select, SessionType, Var, pretty
from reticulate.pendular import is_pendular
from reticulate.reticular import classify_state
from reticulate.statespace import StateSpace
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaffoldingResult:
    """Result of scaffolding analysis.

    Attributes:
        is_scaffold: True if teacher_type is supertype of learner_type.
        zpd_size: Number of methods in teacher but not in learner.
        shared_methods: Methods common to both teacher and learner.
        scaffold_methods: Methods teacher offers beyond learner.
        learning_progress: Ratio of shared methods to total teacher methods.
        stage: Classification: novice/intermediate/advanced/expert.
    """

    is_scaffold: bool
    zpd_size: int
    shared_methods: tuple[str, ...]
    scaffold_methods: tuple[str, ...]
    learning_progress: float
    stage: str


@dataclass(frozen=True)
class DialogueResult:
    """Result of dialogue analysis.

    Attributes:
        is_turn_taking: True if the state space is pendular.
        teacher_turns: Number of branch states (teacher/environment turns).
        learner_turns: Number of select states (learner/process turns).
        balance: Ratio closer to 0.5 = more balanced dialogue.
        dialogue_depth: Max conversation length (BFS depth from top).
        is_socratic: True if teacher mostly branches (asks) and learner
            mostly selects (answers), i.e. branch_count > select_count.
    """

    is_turn_taking: bool
    teacher_turns: int
    learner_turns: int
    balance: float
    dialogue_depth: int
    is_socratic: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_choices(s: SessionType) -> dict[str, SessionType]:
    """Extract top-level choices from a session type."""
    if isinstance(s, (Branch, Select)):
        return dict(s.choices)
    if isinstance(s, Rec):
        return _get_choices(s.body)
    return {}


def _get_all_method_names(s: SessionType) -> set[str]:
    """Extract all method names reachable from a session type (shallow)."""
    return set(_get_choices(s).keys())


def _classify_stage(progress: float) -> str:
    """Classify a learning progress value into a developmental stage."""
    if progress < 0.25:
        return "novice"
    elif progress < 0.50:
        return "intermediate"
    elif progress < 0.75:
        return "advanced"
    else:
        return "expert"


def _count_branch_select(ss: StateSpace) -> tuple[int, int]:
    """Count branch states and select states in a state space."""
    branch_count = 0
    select_count = 0
    for state in ss.states:
        cls = classify_state(ss, state)
        if cls.kind == "branch":
            branch_count += 1
        elif cls.kind == "select":
            select_count += 1
    return branch_count, select_count


def _compute_max_depth(ss: StateSpace) -> int:
    """Compute BFS depth from top to farthest reachable state."""
    from collections import deque

    if not ss.states:
        return 0

    depths: dict[int, int] = {ss.top: 0}
    queue: deque[int] = deque([ss.top])
    adj: dict[int, list[int]] = {}
    for src, _label, tgt in ss.transitions:
        adj.setdefault(src, []).append(tgt)

    while queue:
        s = queue.popleft()
        for tgt in adj.get(s, []):
            if tgt not in depths:
                depths[tgt] = depths[s] + 1
                queue.append(tgt)

    return max(depths.values()) if depths else 0


def _continuation_depth(s: SessionType, depth: int = 0) -> int:
    """Rough estimate of how deep a session type's choices go."""
    if isinstance(s, (End, Var)):
        return depth
    if isinstance(s, (Branch, Select)):
        if not s.choices:
            return depth
        return max(
            _continuation_depth(cont, depth + 1)
            for _, cont in s.choices
        )
    if isinstance(s, Rec):
        return _continuation_depth(s.body, depth)
    return depth


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def analyze_scaffolding(
    teacher: SessionType, learner: SessionType
) -> ScaffoldingResult:
    """Analyse scaffolding relationship between teacher and learner types.

    The teacher type is a scaffold if it is a supertype of the learner
    (i.e., the learner is a subtype of the teacher in Gay-Hole terms:
    the learner's branch offers at least the methods the teacher offers).

    In session type subtyping, a branch subtype offers MORE methods.
    So teacher is a scaffold (supertype) when learner <: teacher,
    meaning teacher has fewer or equal methods and learner has acquired
    at least those.  But pedagogically, teacher should offer MORE.

    We use the pedagogical interpretation: teacher is a scaffold if
    teacher offers methods the learner doesn't yet have (teacher is
    the "richer" type that the learner aspires to).
    """
    teacher_methods = _get_all_method_names(teacher)
    learner_methods = _get_all_method_names(learner)

    shared = teacher_methods & learner_methods
    scaffold = teacher_methods - learner_methods

    total_teacher = len(teacher_methods) if teacher_methods else 1
    progress = len(shared) / total_teacher

    # Teacher is a scaffold if it has methods beyond the learner
    # AND the learner's methods are a subset of the teacher's
    is_scaffold = learner_methods <= teacher_methods and len(scaffold) > 0

    stage = _classify_stage(progress)

    return ScaffoldingResult(
        is_scaffold=is_scaffold,
        zpd_size=len(scaffold),
        shared_methods=tuple(sorted(shared)),
        scaffold_methods=tuple(sorted(scaffold)),
        learning_progress=round(progress, 4),
        stage=stage,
    )


def zone_of_proximal_development(
    teacher: SessionType, learner: SessionType
) -> tuple[SessionType, ...]:
    """Return the ZPD gap — teacher-only method continuations.

    Extracts the choices from both types and returns the continuations
    of methods that appear in the teacher but not in the learner.
    These represent capabilities the learner can acquire with guidance.
    """
    teacher_choices = _get_choices(teacher)
    learner_choices = _get_choices(learner)

    gap_methods = set(teacher_choices.keys()) - set(learner_choices.keys())

    # Return continuations for teacher-only methods, sorted for determinism
    return tuple(teacher_choices[m] for m in sorted(gap_methods))


def analyze_dialogue(ss: StateSpace) -> DialogueResult:
    """Analyse a state space for turn-taking dialogue structure.

    Branch states represent teacher/environment turns (offering choices).
    Select states represent learner/process turns (making selections).
    A pendular state space has strict turn-taking (alternation).
    A Socratic dialogue has more teacher turns (questions) than learner
    turns (answers).
    """
    turn_taking = is_pendular(ss)
    branch_count, select_count = _count_branch_select(ss)
    total = branch_count + select_count
    if total == 0:
        balance = 0.5
    else:
        balance = min(branch_count, select_count) / total

    depth = _compute_max_depth(ss)
    socratic = branch_count > select_count

    return DialogueResult(
        is_turn_taking=turn_taking,
        teacher_turns=branch_count,
        learner_turns=select_count,
        balance=round(balance, 4),
        dialogue_depth=depth,
        is_socratic=socratic,
    )


def developmental_stages(
    types: list[SessionType],
) -> list[tuple[str, SessionType, float]]:
    """Classify a sequence of session types as developmental stages.

    Each type is compared to the LAST type in the list (the target).
    Returns (stage_name, type, progress_score) for each type.
    """
    if not types:
        return []

    target = types[-1]
    target_methods = _get_all_method_names(target)
    total_target = len(target_methods) if target_methods else 1

    result: list[tuple[str, SessionType, float]] = []
    for t in types:
        t_methods = _get_all_method_names(t)
        shared = t_methods & target_methods
        progress = len(shared) / total_target
        stage = _classify_stage(progress)
        result.append((stage, t, round(progress, 4)))

    return result


def learning_path(
    teacher: SessionType, learner: SessionType
) -> list[str]:
    """Compute an ordered list of methods the learner should acquire next.

    Methods are ordered by complexity: methods with shallower continuations
    in the teacher type come first (simpler sub-protocols are easier to
    learn).
    """
    teacher_choices = _get_choices(teacher)
    learner_choices = _get_choices(learner)

    gap_methods = set(teacher_choices.keys()) - set(learner_choices.keys())

    # Sort by continuation depth (simpler first), then alphabetically
    def sort_key(m: str) -> tuple[int, str]:
        return (_continuation_depth(teacher_choices[m]), m)

    return sorted(gap_methods, key=sort_key)


def classify_pedagogy(
    teacher: SessionType, learner: SessionType, ss: StateSpace
) -> dict:
    """Combined pedagogical analysis.

    Returns a dict with:
        scaffolding_result: ScaffoldingResult
        dialogue_result: DialogueResult
        zpd_methods: tuple of SessionType (ZPD gap continuations)
        learning_path: list of str (ordered methods to acquire)
    """
    return {
        "scaffolding_result": analyze_scaffolding(teacher, learner),
        "dialogue_result": analyze_dialogue(ss),
        "zpd_methods": zone_of_proximal_development(teacher, learner),
        "learning_path": learning_path(teacher, learner),
    }
