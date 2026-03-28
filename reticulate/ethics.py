"""Ethics as subtype checking on session types (Step 201).

Formalizes ethical reasoning as session type subtyping: a morally good
action is one whose session type is a subtype of the "good" type defined
by an ethical framework.  Different ethical frameworks—Kantian, utilitarian,
virtue ethics, care ethics, deontological, and contractarian—define
different principle types, and subtype checking determines whether an
action conforms.

The key insight: **ethical judgment IS subtype checking**.  An action type
S_action is ethical under framework F iff S_action ≤_GH S_principle(F).
This reduces moral reasoning to a decidable, mechanized procedure on
session type ASTs.

This module provides:
    ``ETHICAL_FRAMEWORKS``              — predefined ethical frameworks as session types.
    ``judge_action(action, framework)`` — check if action is ethical under framework.
    ``judge_all_frameworks(action)``    — check action under all frameworks.
    ``ethical_dilemma(a, b, framework)``— which action is more ethical.
    ``all_frameworks_agree(action)``    — True if all frameworks agree.
    ``framework_lattice()``             — subtyping relationships among frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import Branch, End, SessionType, parse
from reticulate.statespace import build_statespace
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EthicalFramework:
    """An ethical framework encoded as a session type principle.

    Attributes:
        name: Human-readable name of the framework.
        principle_type_str: Session type string encoding the ethical principle.
        description: Brief description of the framework's moral stance.
    """

    name: str
    principle_type_str: str
    description: str


@dataclass(frozen=True)
class EthicalJudgment:
    """Result of judging an action under an ethical framework.

    Attributes:
        action: The action type string that was judged.
        framework: Name of the ethical framework used.
        is_ethical: True if action_type ≤_GH principle_type.
        violation_methods: Methods in action not present in principle.
        alignment_score: Compatibility score between 0.0 and 1.0.
    """

    action: str
    framework: str
    is_ethical: bool
    violation_methods: tuple[str, ...]
    alignment_score: float


# ---------------------------------------------------------------------------
# Predefined ethical frameworks
# ---------------------------------------------------------------------------


ETHICAL_FRAMEWORKS: dict[str, EthicalFramework] = {
    "kantian": EthicalFramework(
        name="kantian",
        principle_type_str=(
            "rec X . &{consider_universalizability: "
            "+{universal: &{act: X}, not_universal: end}}"
        ),
        description="Act only on maxims you can universalize.",
    ),
    "utilitarian": EthicalFramework(
        name="utilitarian",
        principle_type_str=(
            "&{calculate_consequences: "
            "+{net_positive: &{act: end}, net_negative: &{abstain: end}}}"
        ),
        description="Maximize total welfare.",
    ),
    "virtue": EthicalFramework(
        name="virtue",
        principle_type_str=(
            "rec X . &{assess_character: "
            "+{virtuous: &{act_with_excellence: X, complete: end}, "
            "deficient: &{cultivate: X, abandon: end}}}"
        ),
        description="Act from virtuous character.",
    ),
    "care": EthicalFramework(
        name="care",
        principle_type_str=(
            "rec X . &{perceive_need: "
            "+{respond_with_care: &{maintain_relationship: X}, withdraw: end}}"
        ),
        description="Maintain caring relationships.",
    ),
    "deontological": EthicalFramework(
        name="deontological",
        principle_type_str=(
            "&{check_duty: "
            "+{duty_present: &{fulfill: end}, no_duty: &{permissible: end}}}"
        ),
        description="Follow moral duties.",
    ),
    "contractarian": EthicalFramework(
        name="contractarian",
        principle_type_str=(
            "&{negotiate_terms: "
            "+{fair_agreement: &{honor_contract: end}, "
            "unfair: &{renegotiate: end}}}"
        ),
        description="Follow fair social contracts.",
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_methods(node: SessionType) -> set[str]:
    """Collect all method/label names reachable from a session type AST."""
    visited: set[int] = set()
    methods: set[str] = set()
    stack: list[SessionType] = [node]
    while stack:
        n = stack.pop()
        nid = id(n)
        if nid in visited:
            continue
        visited.add(nid)
        if isinstance(n, Branch):
            for name, cont in n.choices:
                methods.add(name)
                stack.append(cont)
        elif hasattr(n, "choices"):  # Select
            for name, cont in n.choices:
                methods.add(name)
                stack.append(cont)
        elif hasattr(n, "body"):  # Rec
            stack.append(n.body)  # type: ignore[attr-defined]
        elif hasattr(n, "left") and hasattr(n, "right"):  # Parallel
            stack.append(n.left)  # type: ignore[attr-defined]
            stack.append(n.right)  # type: ignore[attr-defined]
        elif hasattr(n, "first") and hasattr(n, "second"):  # Continuation
            stack.append(n.first)  # type: ignore[attr-defined]
            stack.append(n.second)  # type: ignore[attr-defined]
    return methods


def _compute_alignment(action_ast: SessionType, principle_ast: SessionType) -> float:
    """Compute alignment score between action and principle.

    The score is the fraction of principle methods covered by the action.
    Returns 1.0 if the action is a subtype, otherwise a fraction.
    """
    action_methods = _collect_methods(action_ast)
    principle_methods = _collect_methods(principle_ast)
    if not principle_methods:
        return 1.0
    covered = len(action_methods & principle_methods)
    return round(covered / len(principle_methods), 3)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def judge_action(action_type_str: str, framework: str) -> EthicalJudgment:
    """Judge whether an action is ethical under a given framework.

    Parses both the action type and the framework's principle type,
    checks subtyping (action ≤_GH principle), and computes alignment.

    Args:
        action_type_str: Session type string for the action.
        framework: Name of the ethical framework (key in ETHICAL_FRAMEWORKS).

    Returns:
        EthicalJudgment with the verdict.

    Raises:
        KeyError: If framework is not in ETHICAL_FRAMEWORKS.
        ParseError: If either type string is invalid.
    """
    fw = ETHICAL_FRAMEWORKS[framework]
    action_ast = parse(action_type_str)
    principle_ast = parse(fw.principle_type_str)

    ethical = is_subtype(action_ast, principle_ast)

    action_methods = _collect_methods(action_ast)
    principle_methods = _collect_methods(principle_ast)
    violations = tuple(sorted(action_methods - principle_methods))

    alignment = _compute_alignment(action_ast, principle_ast)

    return EthicalJudgment(
        action=action_type_str,
        framework=framework,
        is_ethical=ethical,
        violation_methods=violations,
        alignment_score=alignment,
    )


def judge_all_frameworks(action_type_str: str) -> dict[str, EthicalJudgment]:
    """Judge an action under all predefined ethical frameworks.

    Args:
        action_type_str: Session type string for the action.

    Returns:
        Dict mapping framework name to EthicalJudgment.
    """
    return {
        name: judge_action(action_type_str, name)
        for name in ETHICAL_FRAMEWORKS
    }


def ethical_dilemma(action_a: str, action_b: str, framework: str) -> str:
    """Compare two actions under a framework and return which is more ethical.

    Args:
        action_a: Session type string for the first action.
        action_b: Session type string for the second action.
        framework: Name of the ethical framework.

    Returns:
        "a" if action_a is more ethical, "b" if action_b is, "equal" if tied.
    """
    ja = judge_action(action_a, framework)
    jb = judge_action(action_b, framework)

    # If one is ethical and the other is not, the ethical one wins
    if ja.is_ethical and not jb.is_ethical:
        return "a"
    if jb.is_ethical and not ja.is_ethical:
        return "b"

    # Both ethical or both unethical: compare alignment scores
    if ja.alignment_score > jb.alignment_score:
        return "a"
    if jb.alignment_score > ja.alignment_score:
        return "b"
    return "equal"


def all_frameworks_agree(action_type_str: str) -> bool:
    """Check whether all ethical frameworks give the same judgment.

    Returns True if all frameworks agree (all ethical or all unethical).
    """
    judgments = judge_all_frameworks(action_type_str)
    verdicts = {j.is_ethical for j in judgments.values()}
    return len(verdicts) == 1


def framework_lattice() -> dict[str, list[str]]:
    """Compute the subtyping lattice among ethical frameworks.

    For each framework, list the frameworks whose principle type it
    subsumes (is a subtype of).

    Returns:
        Dict mapping framework name to list of framework names it subsumes.
    """
    result: dict[str, list[str]] = {}
    frameworks = list(ETHICAL_FRAMEWORKS.items())
    for name_a, fw_a in frameworks:
        subsumes: list[str] = []
        ast_a = parse(fw_a.principle_type_str)
        for name_b, fw_b in frameworks:
            if name_a == name_b:
                continue
            ast_b = parse(fw_b.principle_type_str)
            if is_subtype(ast_a, ast_b):
                subsumes.append(name_b)
        result[name_a] = sorted(subsumes)
    return result
