"""Dreams as random metaphor engine (Step 204).

Dreams apply the metaphor mechanism randomly, creating novel relabelings
and recombinations of waking experience.  Dreams are the brain's way of
exploring the analogy space — sampling random points in the space of all
possible label mappings over a fixed structural skeleton.

Key concepts:

- **Dream**: a random metaphor applied to a session type.  Structure is
  always preserved; only labels change.
- **Lucid dream**: a dream where some labels are consciously fixed.
- **Nightmare**: a dream that also inverts polarity (Branch <-> Select).
- **Dream sequence**: a series of dreams from the same base type.
- **Collective dream**: Jung's collective unconscious — the same random
  metaphor applied to multiple types simultaneously.

Metrics:

- **Novelty**: fraction of labels that changed (0 = identity, 1 = total).
- **Lucidity**: compatibility score between original and dream type.  High
  lucidity means the dream closely resembles waking reality.

This module provides:
    ``dream(type_str, seed)``                  — random metaphor.
    ``lucid_dream(type_str, fixed, seed)``     — dream with fixed labels.
    ``nightmare(type_str, seed)``              — inverted dream.
    ``dream_sequence(type_str, count, seed)``  — series of dreams.
    ``interpret_dream(dream)``                 — dream interpretation text.
    ``collective_dream(types, seed)``          — shared metaphor across types.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from reticulate.mechanisms import metaphor, negate_type
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
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DreamResult:
    """Result of dreaming about a session type.

    Attributes:
        original_type_str: pretty-printed original type.
        dream_type_str: pretty-printed dream type.
        label_mapping: how labels were scrambled (old -> new).
        structural_preservation: always True (metaphor preserves structure).
        novelty_score: fraction of labels that changed (0..1).
        lucidity: compatibility between original and dream (0..1).
    """

    original_type_str: str
    dream_type_str: str
    label_mapping: dict[str, str]
    structural_preservation: bool
    novelty_score: float
    lucidity: float


@dataclass(frozen=True)
class DreamSequence:
    """A series of dreams from the same base type.

    Attributes:
        dreams: the individual dream results.
        total_dreams: number of dreams.
        average_novelty: mean novelty score across dreams.
        average_lucidity: mean lucidity score across dreams.
        recurring_elements: labels appearing in >50% of dreams.
    """

    dreams: tuple[DreamResult, ...]
    total_dreams: int
    average_novelty: float
    average_lucidity: float
    recurring_elements: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_labels(s: SessionType) -> set[str]:
    """Collect all unique method labels from a session type AST."""
    labels: set[str] = set()

    def walk(node: SessionType) -> None:
        if isinstance(node, (End, Wait, Var)):
            return
        if isinstance(node, (Branch, Select)):
            for method, cont in node.choices:
                labels.add(method)
                walk(cont)
        elif isinstance(node, Parallel):
            for b in node.branches:
                walk(b)
        elif isinstance(node, Rec):
            walk(node.body)

    walk(s)
    return labels


def _random_permutation(
    labels: list[str], rng: random.Random, fixed: set[str] | None = None,
) -> dict[str, str]:
    """Create a random permutation mapping for the given labels.

    Labels in *fixed* are kept in place; the rest are shuffled.
    """
    if fixed is None:
        fixed = set()

    movable = [l for l in labels if l not in fixed]
    shuffled = list(movable)
    rng.shuffle(shuffled)
    mapping: dict[str, str] = {}
    for old, new in zip(movable, shuffled):
        mapping[old] = new
    # Fixed labels map to themselves (included for completeness).
    for f in fixed:
        if f in labels:
            mapping[f] = f
    return mapping


def _compute_novelty(mapping: dict[str, str]) -> float:
    """Fraction of labels that changed under the mapping."""
    if not mapping:
        return 0.0
    changed = sum(1 for k, v in mapping.items() if k != v)
    return changed / len(mapping)


def _compute_lucidity(original: SessionType, dreamed: SessionType) -> float:
    """Lucidity = compatibility_score between original and dream."""
    return compatibility_score(original, dreamed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dream(type_str: str, seed: int = 42) -> DreamResult:
    """Apply a random metaphor to a session type.

    Parse the type, collect all unique labels, generate a random
    permutation, and apply it as a relabeling.  Structure is always
    preserved.
    """
    ast = parse(type_str)
    labels = sorted(_collect_labels(ast))

    rng = random.Random(seed)
    mapping = _random_permutation(labels, rng)
    dreamed = metaphor(ast, mapping)

    novelty = _compute_novelty(mapping)
    lucidity = _compute_lucidity(ast, dreamed)

    return DreamResult(
        original_type_str=pretty(ast),
        dream_type_str=pretty(dreamed),
        label_mapping=mapping,
        structural_preservation=True,
        novelty_score=novelty,
        lucidity=lucidity,
    )


def lucid_dream(
    type_str: str, fixed_labels: set[str], seed: int = 42,
) -> DreamResult:
    """Dream with some labels consciously fixed.

    Only non-fixed labels are permuted.  Higher lucidity expected since
    more of the original structure is preserved.
    """
    ast = parse(type_str)
    labels = sorted(_collect_labels(ast))

    rng = random.Random(seed)
    mapping = _random_permutation(labels, rng, fixed=fixed_labels)
    dreamed = metaphor(ast, mapping)

    novelty = _compute_novelty(mapping)
    lucidity = _compute_lucidity(ast, dreamed)

    return DreamResult(
        original_type_str=pretty(ast),
        dream_type_str=pretty(dreamed),
        label_mapping=mapping,
        structural_preservation=True,
        novelty_score=novelty,
        lucidity=lucidity,
    )


def nightmare(type_str: str, seed: int = 42) -> DreamResult:
    """Apply random metaphor AND invert polarity (Branch <-> Select).

    The structure is preserved but the experience is inverted — what was
    your choice becomes forced, what was offered becomes demanded.
    """
    ast = parse(type_str)
    labels = sorted(_collect_labels(ast))

    rng = random.Random(seed)
    mapping = _random_permutation(labels, rng)
    relabeled = metaphor(ast, mapping)
    inverted = negate_type(relabeled)

    novelty = _compute_novelty(mapping)
    lucidity = _compute_lucidity(ast, inverted)

    return DreamResult(
        original_type_str=pretty(ast),
        dream_type_str=pretty(inverted),
        label_mapping=mapping,
        structural_preservation=True,
        novelty_score=novelty,
        lucidity=lucidity,
    )


def dream_sequence(
    type_str: str, count: int = 5, base_seed: int = 42,
) -> DreamSequence:
    """Generate a sequence of dreams from the same base type.

    Each dream uses a different seed (base_seed + i), producing a
    trajectory through the space of possible relabelings.
    """
    dreams: list[DreamResult] = []
    for i in range(count):
        d = dream(type_str, seed=base_seed + i)
        dreams.append(d)

    avg_novelty = (
        sum(d.novelty_score for d in dreams) / len(dreams) if dreams else 0.0
    )
    avg_lucidity = (
        sum(d.lucidity for d in dreams) / len(dreams) if dreams else 0.0
    )

    # Recurring elements: labels that appear in >50% of dream types.
    from collections import Counter
    label_counts: Counter[str] = Counter()
    for d in dreams:
        for lbl in d.label_mapping.values():
            label_counts[lbl] += 1
    threshold = len(dreams) / 2.0
    recurring = tuple(
        sorted(lbl for lbl, cnt in label_counts.items() if cnt > threshold)
    )

    return DreamSequence(
        dreams=tuple(dreams),
        total_dreams=len(dreams),
        average_novelty=avg_novelty,
        average_lucidity=avg_lucidity,
        recurring_elements=recurring,
    )


def interpret_dream(dream_result: DreamResult) -> str:
    """Generate a 'dream interpretation' for a dream result.

    For each changed label, produce a line of the form:
        Your 'old' became 'new' — <interpretation>.

    The interpretation is a fun heuristic based on the new label's first
    letter.
    """
    interpretations_by_letter: dict[str, str] = {
        "a": "perhaps you seek acceptance",
        "b": "a desire for balance emerges",
        "c": "change is calling to you",
        "d": "you are drawn to discovery",
        "e": "perhaps you seek freedom from constraint",
        "f": "a feeling of flight or freedom",
        "g": "growth is on your mind",
        "h": "you long for harmony",
        "i": "introspection beckons",
        "j": "joy is within reach",
        "k": "knowledge awaits you",
        "l": "love or longing surfaces",
        "m": "a message is trying to reach you",
        "n": "something new is forming",
        "o": "openness to possibility",
        "p": "patience will serve you well",
        "q": "a quiet truth is emerging",
        "r": "renewal is at hand",
        "s": "a search for meaning continues",
        "t": "transformation is underway",
        "u": "understanding deepens",
        "v": "a vision crystallizes",
        "w": "wisdom from the unconscious",
        "x": "the unknown beckons",
        "y": "yearning for connection",
        "z": "a zenith approaches",
    }
    default_interpretation = "the unconscious speaks in symbols"

    lines: list[str] = []
    for old, new in sorted(dream_result.label_mapping.items()):
        if old != new:
            first = new[0].lower() if new else ""
            interp = interpretations_by_letter.get(first, default_interpretation)
            lines.append(f"Your '{old}' became '{new}' -- {interp}.")

    if not lines:
        return "A dreamless sleep — nothing changed."
    return "\n".join(lines)


def collective_dream(types: list[str], seed: int = 42) -> DreamResult:
    """Jung's collective unconscious: apply the SAME random metaphor to
    multiple types simultaneously.

    Collects labels from all types, builds one shared permutation, and
    applies it to each type.  Returns a DreamResult for the first type
    (as representative), with the shared mapping.
    """
    all_asts = [parse(t) for t in types]
    all_labels: set[str] = set()
    for ast in all_asts:
        all_labels |= _collect_labels(ast)

    rng = random.Random(seed)
    mapping = _random_permutation(sorted(all_labels), rng)

    # Apply to first type as the representative result.
    representative = all_asts[0] if all_asts else parse("end")
    dreamed = metaphor(representative, mapping)

    novelty = _compute_novelty(mapping)
    lucidity = _compute_lucidity(representative, dreamed)

    return DreamResult(
        original_type_str=pretty(representative),
        dream_type_str=pretty(dreamed),
        label_mapping=mapping,
        structural_preservation=True,
        novelty_score=novelty,
        lucidity=lucidity,
    )
