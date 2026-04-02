"""Extract session type selections from Java preconditions (Step 97e).

Scans Java source code for precondition patterns:
  - checkState(condition)
  - checkArgument(condition)
  - if (condition) throw ...
  - assert condition

Maps each precondition to a session type selection: the condition
becomes a +{TRUE: ..., FALSE: ...} branch in the protocol.

Key insight: every checkState(count != 0) is a hidden selection.
The method is only safe when the condition holds. The extractor
makes this explicit in the session type.

Refactoring proposal: for each group of methods sharing the same
precondition, propose an enum method that returns the condition result,
enabling the client to select explicitly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Precondition:
    """A precondition extracted from a Java method."""
    method_name: str
    condition: str  # e.g., "count != 0", "count > 1"
    kind: str  # "checkState", "checkArgument", "assert", "if_throw"
    line_number: int
    field_name: str  # the field being tested, e.g., "count"
    operator: str  # "!=", ">", ">=", "==", "<"
    threshold: str  # "0", "1", etc.


@dataclass(frozen=True)
class PhaseGroup:
    """A group of methods sharing the same precondition threshold."""
    condition: str  # e.g., "count != 0"
    field_name: str
    operator: str
    threshold: str
    methods: list[str]  # methods requiring this condition
    phase_name: str  # suggested enum value, e.g., "NON_EMPTY"


@dataclass(frozen=True)
class SelectionProposal:
    """Proposed session type selection from preconditions."""
    class_name: str
    field_name: str
    phases: list[PhaseGroup]
    always_available: list[str]  # methods with no precondition
    enum_name: str  # proposed enum type name
    enum_values: list[str]  # proposed enum values
    selector_method: str  # proposed method name, e.g., "phase()"
    session_type: str  # the L3 session type with selections
    refactored_code: str  # proposed Java code


# ---------------------------------------------------------------------------
# Precondition extraction from Java source
# ---------------------------------------------------------------------------

# Patterns for precondition detection
CHECKSTATE_PATTERN = re.compile(
    r'checkState\s*\(\s*(\w+)\s*(!=|>|>=|==|<|<=)\s*(\w+)\s*\)',
)
CHECKARGUMENT_PATTERN = re.compile(
    r'checkArgument\s*\(\s*(\w+)\s*(!=|>|>=|==|<|<=)\s*(\w+)\s*\)',
)
IF_THROW_PATTERN = re.compile(
    r'if\s*\(\s*(\w+)\s*(==|<|<=)\s*(\w+)\s*\)\s*\{?\s*throw',
)
ASSERT_PATTERN = re.compile(
    r'assert\s+(\w+)\s*(!=|>|>=)\s*(\w+)\s*;',
)


def extract_preconditions(java_source: str) -> list[Precondition]:
    """Extract preconditions from Java source code."""
    preconditions: list[Precondition] = []

    # Find current method context
    method_pattern = re.compile(
        r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?'
        r'(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws[^{]*)?\{',
    )

    methods = list(method_pattern.finditer(java_source))

    for i, method_match in enumerate(methods):
        method_name = method_match.group(1)
        start = method_match.end()
        # Find method body (approximate: next method or end)
        end = methods[i + 1].start() if i + 1 < len(methods) else len(java_source)
        body = java_source[start:end]

        # Count lines to get approximate line number
        line_no = java_source[:start].count('\n') + 1

        for pattern, kind in [
            (CHECKSTATE_PATTERN, "checkState"),
            (CHECKARGUMENT_PATTERN, "checkArgument"),
            (IF_THROW_PATTERN, "if_throw"),
            (ASSERT_PATTERN, "assert"),
        ]:
            for match in pattern.finditer(body):
                field_name = match.group(1)
                operator = match.group(2)
                threshold = match.group(3)

                # For if_throw, the condition is INVERTED
                # if (count == 0) throw → precondition is count != 0
                if kind == "if_throw":
                    operator = _invert_operator(operator)

                preconditions.append(Precondition(
                    method_name=method_name,
                    condition=f"{field_name} {operator} {threshold}",
                    kind=kind,
                    line_number=line_no,
                    field_name=field_name,
                    operator=operator,
                    threshold=threshold,
                ))

    return preconditions


def _invert_operator(op: str) -> str:
    """Invert a comparison operator (for if-throw patterns)."""
    inversions = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}
    return inversions.get(op, op)


# ---------------------------------------------------------------------------
# Phase grouping
# ---------------------------------------------------------------------------

def group_into_phases(
    preconditions: list[Precondition],
    all_public_methods: list[str],
) -> tuple[list[PhaseGroup], list[str]]:
    """Group methods by their precondition thresholds.

    Returns (phase_groups, always_available_methods).
    """
    # Group by (field, operator, threshold)
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    methods_with_preconditions: set[str] = set()

    for pc in preconditions:
        key = (pc.field_name, pc.operator, pc.threshold)
        if pc.method_name not in groups[key]:
            groups[key].append(pc.method_name)
        methods_with_preconditions.add(pc.method_name)

    # Methods without preconditions
    always_available = [m for m in all_public_methods
                        if m not in methods_with_preconditions]

    # Create phase groups with suggested names
    phase_groups: list[PhaseGroup] = []
    for (field_name, op, threshold), methods in sorted(groups.items()):
        phase_name = _suggest_phase_name(field_name, op, threshold)
        phase_groups.append(PhaseGroup(
            condition=f"{field_name} {op} {threshold}",
            field_name=field_name,
            operator=op,
            threshold=threshold,
            methods=methods,
            phase_name=phase_name,
        ))

    # Sort by threshold (ascending) for clean phase ordering
    phase_groups.sort(key=lambda g: (g.field_name, _threshold_value(g.threshold)))

    return phase_groups, always_available


def _suggest_phase_name(field: str, op: str, threshold: str) -> str:
    """Suggest a human-readable phase name."""
    if field == "count" or field == "size":
        if threshold == "0" and op in ("!=", ">"):
            return "NON_EMPTY"
        elif threshold == "1" and op == ">":
            return "MULTI"
        elif threshold == "0" and op == "==":
            return "EMPTY"
    return f"{field.upper()}_{op.replace('!', 'NOT_').replace('=', 'EQ').replace('<', 'LT').replace('>', 'GT')}_{threshold}"


def _threshold_value(threshold: str) -> float:
    """Convert threshold to numeric for sorting."""
    try:
        return float(threshold)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Selection proposal generation
# ---------------------------------------------------------------------------

def propose_selection(
    class_name: str,
    java_source: str,
    public_methods: list[str],
) -> Optional[SelectionProposal]:
    """Analyze a Java class and propose session type selections.

    Extracts preconditions, groups into phases, generates:
    - L3 session type with explicit selections
    - Refactored Java code with enum + selector method
    """
    preconditions = extract_preconditions(java_source)
    if not preconditions:
        return None

    phases, always_available = group_into_phases(preconditions, public_methods)
    if not phases:
        return None

    # Determine the field being tested (use most common)
    field_name = max(set(p.field_name for p in preconditions),
                     key=lambda f: sum(1 for p in preconditions if p.field_name == f))

    # Build enum
    enum_values = ["INITIAL"] + [g.phase_name for g in phases]
    enum_name = f"{class_name}Phase"
    selector_method = "phase"

    # Build session type
    session_type = _build_session_type(phases, always_available, enum_values)

    # Build refactored code
    refactored = _build_refactored_code(class_name, enum_name, enum_values,
                                         selector_method, field_name, phases,
                                         always_available)

    return SelectionProposal(
        class_name=class_name,
        field_name=field_name,
        phases=phases,
        always_available=always_available,
        enum_name=enum_name,
        enum_values=enum_values,
        selector_method=selector_method,
        session_type=session_type,
        refactored_code=refactored,
    )


def _build_session_type(
    phases: list[PhaseGroup],
    always_available: list[str],
    enum_values: list[str],
) -> str:
    """Build L3 session type from phase groups."""
    # Always-available methods form the base (recursive)
    # Phase selector returns enum → each value unlocks more methods

    # Build the selection branches
    selection_branches: list[str] = []

    # INITIAL phase: only always-available methods
    base_methods = [f"{m}: X" for m in always_available if m not in ('add', 'addAll')]
    add_methods = [f"{m}: X" for m in always_available if m in ('add', 'addAll')]

    # Each phase unlocks additional methods
    for i, phase in enumerate(phases):
        # This phase's methods + all previous phases' methods
        phase_methods = list(phase.methods)
        for prev in phases[:i]:
            phase_methods.extend(prev.methods)

        branch_entries = add_methods + [f"{m}: X" for m in phase_methods] + base_methods + ["done: end"]
        branch_body = "&{" + ", ".join(branch_entries) + "}"
        selection_branches.append(f"{phase.phase_name}: {branch_body}")

    # INITIAL: only base methods
    initial_entries = add_methods + base_methods + ["done: end"]
    initial_body = "&{" + ", ".join(initial_entries) + "}"
    selection_branches.insert(0, f"INITIAL: {initial_body}")

    # The selector method returns the enum
    selector_body = "+{" + ", ".join(selection_branches) + "}"

    # Full session type: rec X . &{add: X, ..., phase: +{...}, done: end}
    all_entries = add_methods + [f"phase: {selector_body}"] + base_methods + ["done: end"]
    return "rec X . &{" + ", ".join(all_entries) + "}"


def _build_refactored_code(
    class_name: str,
    enum_name: str,
    enum_values: list[str],
    selector_method: str,
    field_name: str,
    phases: list[PhaseGroup],
    always_available: list[str],
) -> str:
    """Generate proposed refactored Java code."""
    lines: list[str] = []

    # Enum definition
    lines.append(f"    public enum {enum_name} {{")
    lines.append(f"        {', '.join(enum_values)}")
    lines.append(f"    }}")
    lines.append(f"")

    # Selector method
    lines.append(f"    /** Returns the current phase based on {field_name}. */")
    lines.append(f"    public {enum_name} {selector_method}() {{")

    # Build conditions from phases (reverse order for if-else chain)
    for phase in reversed(phases):
        lines.append(f"        if ({phase.condition}) return {enum_name}.{phase.phase_name};")
    lines.append(f"        return {enum_name}.INITIAL;")
    lines.append(f"    }}")
    lines.append(f"")

    # Usage example
    lines.append(f"    // Client usage (session-type-safe):")
    lines.append(f"    // switch (acc.{selector_method}()) {{")
    for phase in phases:
        methods_str = ", ".join(phase.methods[:3])
        lines.append(f"    //     case {phase.phase_name} -> acc.{phase.methods[0]}();  // safe: {methods_str}")
    lines.append(f"    //     case INITIAL -> acc.add(value);  // only add/count available")
    lines.append(f"    // }}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: analyze a Java file
# ---------------------------------------------------------------------------

def analyze_java_file(file_path: str) -> Optional[SelectionProposal]:
    """Analyze a Java file and propose selections."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        source = f.read()

    # Extract class name (must be preceded by public/final/abstract or start of declaration)
    class_match = re.search(r'(?:public|final|abstract)\s+(?:final\s+)?class\s+(\w+)', source)
    if not class_match:
        class_match = re.search(r'^class\s+(\w+)', source, re.MULTILINE)
    if not class_match:
        return None
    class_name = class_match.group(1)

    # Extract public methods
    method_pattern = re.compile(
        r'public\s+(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]*>)?)\s+(\w+)\s*\(',
    )
    public_methods = [m.group(1) for m in method_pattern.finditer(source)]

    return propose_selection(class_name, source, public_methods)
