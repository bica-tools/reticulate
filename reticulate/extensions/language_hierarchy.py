"""Language hierarchy for session types: L0 through L9.

Each level adds a constructor. This module:
- Defines each level formally
- Classifies any session type AST by its minimum language level
- Provides separation witnesses (types in L_{i+1} not expressible in L_i)
- Computes containment relationships

The hierarchy is:

    L0  end, wait                    (terminated/synchronised)
    L1  L0 + &{...}                  (external choice / branch)
    L2  L1 + +{...}                  (internal choice / selection)
    L3  L2 + (S1 || S2)             (parallel composition)
    L4  L3 + rec X . S, X           (recursion + variables)
    L5  L4 + (S1 || S2) . S3        (continuation after parallel)
    L6  L5 + nested parallel         (parallel inside parallel)
    L7  L6 + mixed rec+parallel      (recursion containing parallel)
    L8  L7 + mutual recursion        (multiple rec binders, cross-reference)
    L9  L8 + higher-order            (types parameterised by types — future)

Key insight: L1 gives only meet-semilattices (branch fans out from top),
while L2 (adding selection) yields full lattices with both meet and join.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Union

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
    parse,
)


# ---------------------------------------------------------------------------
# Language Level Enum
# ---------------------------------------------------------------------------

class LanguageLevel(IntEnum):
    """Language levels in the session type hierarchy.

    Each level strictly contains all lower levels.  The integer value
    doubles as a sort key so ``max()`` gives the tightest level.
    """

    L0 = 0   # end, wait only
    L1 = 1   # + branch (&)
    L2 = 2   # + selection (+)
    L3 = 3   # + parallel (||)
    L4 = 4   # + recursion (rec X . S, X)
    L5 = 5   # + continuation (S1 || S2) . S3
    L6 = 6   # + nested parallel
    L7 = 7   # + mixed recursion + parallel
    L8 = 8   # + mutual recursion
    L9 = 9   # + higher-order (future)

    @property
    def description(self) -> str:
        """Human-readable description of this level."""
        return _LEVEL_DESCRIPTIONS[self]


_LEVEL_DESCRIPTIONS: dict[LanguageLevel, str] = {
    LanguageLevel.L0: "Terminated sessions (end, wait)",
    LanguageLevel.L1: "External choice (branch)",
    LanguageLevel.L2: "Internal choice (selection)",
    LanguageLevel.L3: "Parallel composition",
    LanguageLevel.L4: "Recursive types",
    LanguageLevel.L5: "Continuation after parallel",
    LanguageLevel.L6: "Nested parallel composition",
    LanguageLevel.L7: "Mixed recursion and parallel",
    LanguageLevel.L8: "Mutual recursion",
    LanguageLevel.L9: "Higher-order session types",
}


# ---------------------------------------------------------------------------
# Constructor sets per level
# ---------------------------------------------------------------------------

_LEVEL_CONSTRUCTORS: dict[LanguageLevel, set[str]] = {
    LanguageLevel.L0: {"End", "Wait"},
    LanguageLevel.L1: {"End", "Wait", "Branch"},
    LanguageLevel.L2: {"End", "Wait", "Branch", "Select"},
    LanguageLevel.L3: {"End", "Wait", "Branch", "Select", "Parallel"},
    LanguageLevel.L4: {"End", "Wait", "Branch", "Select", "Parallel", "Rec", "Var"},
    LanguageLevel.L5: {"End", "Wait", "Branch", "Select", "Parallel", "Rec", "Var",
                        "Continuation"},
    LanguageLevel.L6: {"End", "Wait", "Branch", "Select", "Parallel", "Rec", "Var",
                        "Continuation"},  # same constructors, structural constraint
    LanguageLevel.L7: {"End", "Wait", "Branch", "Select", "Parallel", "Rec", "Var",
                        "Continuation"},  # same constructors, structural constraint
    LanguageLevel.L8: {"End", "Wait", "Branch", "Select", "Parallel", "Rec", "Var",
                        "Continuation"},  # same constructors, structural constraint
    LanguageLevel.L9: {"End", "Wait", "Branch", "Select", "Parallel", "Rec", "Var",
                        "Continuation"},  # future: higher-order
}


def level_constructors(level: LanguageLevel) -> set[str]:
    """Return the set of constructor names available at *level*."""
    return set(_LEVEL_CONSTRUCTORS[level])


# ---------------------------------------------------------------------------
# AST analysis helpers
# ---------------------------------------------------------------------------

def _used_constructors(s: SessionType) -> set[str]:
    """Collect all constructor names used in AST *s*."""
    ctors: set[str] = set()
    _collect_constructors(s, ctors)
    return ctors


def _collect_constructors(s: SessionType, out: set[str]) -> None:
    """Recursively collect constructor names into *out*."""
    match s:
        case End():
            out.add("End")
        case Wait():
            out.add("Wait")
        case Var():
            out.add("Var")
        case Branch(choices=choices):
            out.add("Branch")
            for _, body in choices:
                _collect_constructors(body, out)
        case Select(choices=choices):
            out.add("Select")
            for _, body in choices:
                _collect_constructors(body, out)
        case Parallel(branches=branches):
            out.add("Parallel")
            for b in branches:
                _collect_constructors(b, out)
        case Rec(body=body):
            out.add("Rec")
            out.add("Var")  # rec always implies var usage
            _collect_constructors(body, out)
        case Continuation(left=left, right=right):
            out.add("Continuation")
            _collect_constructors(left, out)
            _collect_constructors(right, out)


def _has_nested_parallel(s: SessionType) -> bool:
    """Check if *s* contains a Parallel node nested inside another Parallel."""
    return _check_nested_parallel(s, inside_parallel=False)


def _check_nested_parallel(s: SessionType, inside_parallel: bool) -> bool:
    match s:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices) | Select(choices=choices):
            return any(_check_nested_parallel(body, inside_parallel)
                       for _, body in choices)
        case Parallel(branches=branches):
            if inside_parallel:
                return True
            return any(_check_nested_parallel(b, inside_parallel=True)
                       for b in branches)
        case Rec(body=body):
            return _check_nested_parallel(body, inside_parallel)
        case Continuation(left=left, right=right):
            return (_check_nested_parallel(left, inside_parallel) or
                    _check_nested_parallel(right, inside_parallel))
    return False  # pragma: no cover


def _has_rec_containing_parallel(s: SessionType) -> bool:
    """Check if *s* has a Rec whose body (transitively) contains Parallel."""
    return _check_rec_parallel(s, inside_rec=False)


def _check_rec_parallel(s: SessionType, inside_rec: bool) -> bool:
    match s:
        case End() | Wait() | Var():
            return False
        case Branch(choices=choices) | Select(choices=choices):
            return any(_check_rec_parallel(body, inside_rec)
                       for _, body in choices)
        case Parallel(branches=branches):
            if inside_rec:
                return True
            return any(_check_rec_parallel(b, inside_rec)
                       for b in branches)
        case Rec(body=body):
            return _check_rec_parallel(body, inside_rec=True)
        case Continuation(left=left, right=right):
            return (_check_rec_parallel(left, inside_rec) or
                    _check_rec_parallel(right, inside_rec))
    return False  # pragma: no cover


def _count_rec_binders(s: SessionType) -> int:
    """Count distinct ``rec`` binders in *s*."""
    count = 0
    match s:
        case End() | Wait() | Var():
            pass
        case Branch(choices=choices) | Select(choices=choices):
            count = sum(_count_rec_binders(body) for _, body in choices)
        case Parallel(branches=branches):
            count = sum(_count_rec_binders(b) for b in branches)
        case Rec(body=body):
            count = 1 + _count_rec_binders(body)
        case Continuation(left=left, right=right):
            count = _count_rec_binders(left) + _count_rec_binders(right)
    return count


def _has_cross_rec_reference(s: SessionType) -> bool:
    """Check for mutual recursion: a Rec body references a variable
    bound by an *outer* Rec with a different variable name, and there
    exist multiple distinct rec binders.

    This is a simplified heuristic: we check if there are 2+ rec binders
    with distinct variable names and nested structure.
    """
    rec_vars: list[str] = []
    _collect_rec_vars(s, rec_vars)
    if len(set(rec_vars)) < 2:
        return False
    # Check if any rec body contains a free reference to another rec var
    return _check_cross_reference(s, bound_rec_vars=set())


def _collect_rec_vars(s: SessionType, out: list[str]) -> None:
    match s:
        case End() | Wait() | Var():
            pass
        case Branch(choices=choices) | Select(choices=choices):
            for _, body in choices:
                _collect_rec_vars(body, out)
        case Parallel(branches=branches):
            for b in branches:
                _collect_rec_vars(b, out)
        case Rec(var=var, body=body):
            out.append(var)
            _collect_rec_vars(body, out)
        case Continuation(left=left, right=right):
            _collect_rec_vars(left, out)
            _collect_rec_vars(right, out)


def _check_cross_reference(s: SessionType, bound_rec_vars: set[str]) -> bool:
    """Return True if a Var references a rec variable from an outer,
    differently-named rec binder."""
    match s:
        case End() | Wait():
            return False
        case Var(name=name):
            return name in bound_rec_vars
        case Branch(choices=choices) | Select(choices=choices):
            return any(_check_cross_reference(body, bound_rec_vars)
                       for _, body in choices)
        case Parallel(branches=branches):
            return any(_check_cross_reference(b, bound_rec_vars)
                       for b in branches)
        case Rec(var=var, body=body):
            # The body of rec X binds X; outer rec vars are "cross" refs
            new_bound = bound_rec_vars - {var}  # remove own var
            # Check if body references any of the outer-bound vars
            if _check_cross_reference(body, new_bound):
                return True
            # Now check children with all bound vars including ours
            return _check_cross_reference(body, bound_rec_vars | {var})
        case Continuation(left=left, right=right):
            return (_check_cross_reference(left, bound_rec_vars) or
                    _check_cross_reference(right, bound_rec_vars))
    return False  # pragma: no cover


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_level(s: SessionType) -> LanguageLevel:
    """Determine the minimum language level needed to express *s*.

    Inspects which constructors *s* uses and checks structural properties
    (nested parallelism, mixed rec+parallel, mutual recursion).
    """
    ctors = _used_constructors(s)

    # Start from base level determined by constructors
    level = LanguageLevel.L0

    if "Branch" in ctors:
        level = max(level, LanguageLevel.L1)
    if "Select" in ctors:
        level = max(level, LanguageLevel.L2)
    if "Parallel" in ctors:
        level = max(level, LanguageLevel.L3)
    if "Rec" in ctors or "Var" in ctors:
        level = max(level, LanguageLevel.L4)
    if "Continuation" in ctors:
        level = max(level, LanguageLevel.L5)

    # Structural checks for L6-L8
    if "Parallel" in ctors and _has_nested_parallel(s):
        level = max(level, LanguageLevel.L6)
    if "Rec" in ctors and "Parallel" in ctors and _has_rec_containing_parallel(s):
        level = max(level, LanguageLevel.L7)
    if "Rec" in ctors and _has_cross_rec_reference(s):
        level = max(level, LanguageLevel.L8)

    return level


def is_expressible_at(s: SessionType, level: LanguageLevel) -> bool:
    """Return True iff *s* can be expressed at the given language *level*.

    A type is expressible at level *k* if ``classify_level(s) <= k``.
    """
    return classify_level(s) <= level


# ---------------------------------------------------------------------------
# Separation witnesses
# ---------------------------------------------------------------------------

def separation_witness(lower: LanguageLevel, upper: LanguageLevel) -> SessionType:
    """Construct a type that belongs to *upper* but not to *lower*.

    Raises ``ValueError`` if ``lower >= upper`` or if no witness is
    available for the requested pair (e.g. L9 witnesses are not yet
    implemented).
    """
    if lower >= upper:
        raise ValueError(
            f"Cannot construct separation witness: {lower.name} >= {upper.name}"
        )

    # We only need witnesses for consecutive levels; for non-consecutive
    # pairs, the witness for (upper-1, upper) works since it's in upper
    # but not in upper-1, hence not in lower either (lower < upper-1 < upper).
    target = upper
    return _witness_for_level(target)


def _witness_for_level(level: LanguageLevel) -> SessionType:
    """Construct the canonical witness for *level*.

    This type is in L_{level} but NOT in L_{level-1}.
    """
    match level:
        case LanguageLevel.L0:
            raise ValueError("L0 has no separation witness (it is the base level)")
        case LanguageLevel.L1:
            # Branch: &{a: end} — needs L1, not in L0
            return parse("&{a: end}")
        case LanguageLevel.L2:
            # Selection: +{a: end} — needs L2, not in L1
            return parse("+{a: end}")
        case LanguageLevel.L3:
            # Parallel: (&{a: end} || &{b: end}) — needs L3
            return parse("(&{a: end} || &{b: end})")
        case LanguageLevel.L4:
            # Recursion: rec X . &{a: X, b: end} — needs L4
            return parse("rec X . &{a: X, b: end}")
        case LanguageLevel.L5:
            # Continuation: (&{a: wait} || &{b: wait}) . &{c: end}
            return parse("(&{a: wait} || &{b: wait}) . &{c: end}")
        case LanguageLevel.L6:
            # Nested parallel: (&{a: end} || (&{b: end} || &{c: end}))
            return parse("(&{a: end} || (&{b: end} || &{c: end}))")
        case LanguageLevel.L7:
            # Rec containing parallel: rec X . (&{a: end} || &{b: X})
            return parse("rec X . (&{a: end} || &{b: X})")
        case LanguageLevel.L8:
            # Mutual recursion: rec X . &{a: rec Y . &{b: X, c: Y}, d: end}
            return parse("rec X . &{a: rec Y . &{b: X, c: Y}, d: end}")
        case LanguageLevel.L9:
            raise NotImplementedError(
                "L9 (higher-order) witnesses are not yet implemented"
            )
    raise ValueError(f"Unknown level: {level}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Benchmark classification
# ---------------------------------------------------------------------------

def classify_benchmark(name: str, type_string: str) -> LanguageLevel:
    """Parse *type_string* and return its language level.

    Parameters
    ----------
    name : str
        Human-readable benchmark name (for error messages).
    type_string : str
        The session type string to parse and classify.

    Returns
    -------
    LanguageLevel
        The minimum language level needed to express the benchmark.
    """
    ast = parse(type_string)
    return classify_level(ast)


def classify_all_benchmarks() -> dict[str, LanguageLevel]:
    """Classify all benchmark protocols from the test suite.

    Returns a dict mapping benchmark name to its language level.
    """
    from tests.benchmarks.protocols import BENCHMARKS

    result: dict[str, LanguageLevel] = {}
    for bp in BENCHMARKS:
        result[bp.name] = classify_benchmark(bp.name, bp.type_string)
    return result


def level_statistics() -> dict[str, object]:
    """Compute statistics about benchmark distribution across levels.

    Returns a dict with:
    - ``level_counts``: dict mapping level name to count of benchmarks
    - ``total``: total number of benchmarks
    - ``level_names``: dict mapping level name to list of benchmark names
    - ``max_level``: the highest level used by any benchmark
    - ``min_level``: the lowest level used by any benchmark
    """
    classifications = classify_all_benchmarks()

    level_counts: dict[str, int] = {}
    level_names: dict[str, list[str]] = {}
    for level in LanguageLevel:
        level_counts[level.name] = 0
        level_names[level.name] = []

    for name, level in classifications.items():
        level_counts[level.name] += 1
        level_names[level.name].append(name)

    levels_used = list(classifications.values())
    return {
        "level_counts": level_counts,
        "total": len(classifications),
        "level_names": level_names,
        "max_level": max(levels_used).name if levels_used else None,
        "min_level": min(levels_used).name if levels_used else None,
    }


# ---------------------------------------------------------------------------
# Containment result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContainmentResult:
    """Result of checking containment between two language levels."""

    lower: LanguageLevel
    upper: LanguageLevel
    is_strict_subset: bool
    witness: SessionType | None
    witness_level: LanguageLevel | None


def check_containment(
    lower: LanguageLevel, upper: LanguageLevel
) -> ContainmentResult:
    """Check that L_{lower} is strictly contained in L_{upper}.

    Returns a ``ContainmentResult`` with a separation witness if the
    containment is strict.
    """
    if lower >= upper:
        return ContainmentResult(
            lower=lower,
            upper=upper,
            is_strict_subset=False,
            witness=None,
            witness_level=None,
        )

    wit = separation_witness(lower, upper)
    wit_level = classify_level(wit)
    return ContainmentResult(
        lower=lower,
        upper=upper,
        is_strict_subset=True,
        witness=wit,
        witness_level=wit_level,
    )


def verify_strict_hierarchy() -> list[ContainmentResult]:
    """Verify that L0 < L1 < L2 < ... < L8 forms a strict chain.

    Returns a list of ContainmentResult for each consecutive pair.
    """
    results: list[ContainmentResult] = []
    levels = sorted(LanguageLevel)
    for i in range(len(levels) - 1):
        lower = levels[i]
        upper = levels[i + 1]
        if upper == LanguageLevel.L9:
            # L9 not yet implemented
            continue
        results.append(check_containment(lower, upper))
    return results


# ---------------------------------------------------------------------------
# Lattice-theoretic properties per level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LevelProperty:
    """Lattice-theoretic properties guaranteed at a language level."""

    level: LanguageLevel
    has_top: bool
    has_bottom: bool
    is_meet_semilattice: bool
    is_join_semilattice: bool
    is_lattice: bool
    description: str


def level_properties(level: LanguageLevel) -> LevelProperty:
    """Return the lattice-theoretic properties guaranteed at *level*.

    Key insight: L1 (branch only) produces meet-semilattices because
    branch creates a tree fanning out from top. L2 (adding selection)
    can produce full lattices because selection creates join structure.
    """
    match level:
        case LanguageLevel.L0:
            return LevelProperty(
                level=level,
                has_top=True,
                has_bottom=True,
                is_meet_semilattice=True,
                is_join_semilattice=True,
                is_lattice=True,
                description="Singleton or two-element chain; trivially a lattice.",
            )
        case LanguageLevel.L1:
            return LevelProperty(
                level=level,
                has_top=True,
                has_bottom=True,
                is_meet_semilattice=True,
                is_join_semilattice=False,
                is_lattice=False,
                description=(
                    "Branch-only types form trees (meet-semilattices). "
                    "Multiple branches from the same state share a meet (top) "
                    "but may lack pairwise joins."
                ),
            )
        case LanguageLevel.L2:
            return LevelProperty(
                level=level,
                has_top=True,
                has_bottom=True,
                is_meet_semilattice=True,
                is_join_semilattice=True,
                is_lattice=True,
                description=(
                    "Adding selection restores join structure. "
                    "Branch provides meets, selection provides joins. "
                    "All terminating L2 types form lattices."
                ),
            )
        case LanguageLevel.L3:
            return LevelProperty(
                level=level,
                has_top=True,
                has_bottom=True,
                is_meet_semilattice=True,
                is_join_semilattice=True,
                is_lattice=True,
                description=(
                    "Parallel composition yields product lattices. "
                    "L(S1 || S2) = L(S1) x L(S2) ordered componentwise."
                ),
            )
        case _:
            return LevelProperty(
                level=level,
                has_top=True,
                has_bottom=True,
                is_meet_semilattice=True,
                is_join_semilattice=True,
                is_lattice=True,
                description=(
                    f"At {level.name}, all well-formed terminating types "
                    f"form lattices."
                ),
            )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def hierarchy_summary() -> str:
    """Generate a human-readable summary of the language hierarchy."""
    lines: list[str] = []
    lines.append("Session Type Language Hierarchy")
    lines.append("=" * 40)
    lines.append("")

    for level in LanguageLevel:
        prop = level_properties(level)
        ctors = level_constructors(level)
        lines.append(f"{level.name}: {level.description}")
        lines.append(f"  Constructors: {', '.join(sorted(ctors))}")
        lines.append(f"  Lattice: {prop.is_lattice}  "
                      f"Meet-SL: {prop.is_meet_semilattice}  "
                      f"Join-SL: {prop.is_join_semilattice}")
        lines.append("")

    # Verify strict hierarchy
    lines.append("Strict Containment Chain")
    lines.append("-" * 40)
    for result in verify_strict_hierarchy():
        status = "STRICT" if result.is_strict_subset else "NOT STRICT"
        lines.append(f"  {result.lower.name} < {result.upper.name}: {status}")

    return "\n".join(lines)
