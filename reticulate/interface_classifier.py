"""Classify interface composition: orthogonal (parallel) vs extending (branch widening).

Given a Java class implementing multiple interfaces, determines whether the
interfaces are ORTHOGONAL (independent protocols, parallel ∥) or EXTENDING
(same protocol, branch widening) by analyzing THREE evidence sources:

1. CLASS STRUCTURE — do methods from I₁ and I₂ share mutable fields?
2. CLIENT USAGE — do clients interleave methods from I₁ and I₂?
3. COMBINED — weighted verdict from both analyses

Key insight: client traces tell us the truth. If 90% of clients never
interleave I₁ and I₂ methods → orthogonal. If they always mix → extending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

try:
    import javalang
    from javalang.tree import (
        ClassDeclaration, InterfaceDeclaration, MethodDeclaration,
        MethodInvocation, FieldDeclaration, MemberReference,
    )
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfaceProfile:
    """Profile of an interface's methods."""
    name: str
    methods: frozenset[str]


@dataclass(frozen=True)
class InterleaveAnalysis:
    """Analysis of how clients use methods from two interfaces."""
    interface_a: str
    interface_b: str
    total_clients: int
    interleaved: int  # clients that mix methods from both in same function
    separated: int    # clients that use only one interface per function
    mixed_but_ordered: int  # use both but in blocks (A...A...B...B)
    interleave_rate: float  # interleaved / total


@dataclass(frozen=True)
class CompositionClassification:
    """Classification of how two interfaces compose on a class."""
    class_name: str
    interface_a: str
    interface_b: str
    classification: str  # "orthogonal", "extending", "ambiguous"
    evidence_structural: str  # "shared_fields", "separate_fields", "unknown"
    evidence_clients: str  # "interleaved", "separated", "mixed"
    interleave_rate: float
    shared_fields: list[str]
    confidence: float  # 0.0 to 1.0
    session_type_suggestion: str  # "parallel", "branch_widening", "needs_investigation"
    explanation: str


# ---------------------------------------------------------------------------
# Client trace interleaving analysis
# ---------------------------------------------------------------------------

def analyze_client_interleaving(
    client_traces: list[list[str]],
    methods_a: set[str],
    methods_b: set[str],
) -> InterleaveAnalysis:
    """Analyze whether client traces interleave methods from two interfaces.

    A trace is INTERLEAVED if it alternates: ..A..B..A.. (method from A,
    then B, then A again). A trace is SEPARATED if all A methods come
    before all B methods (or vice versa), or only uses one interface.
    """
    interleaved = 0
    separated = 0
    mixed_ordered = 0

    for trace in client_traces:
        # Classify each method as A, B, or neither
        sequence = []
        for m in trace:
            if m in methods_a and m not in methods_b:
                sequence.append('A')
            elif m in methods_b and m not in methods_a:
                sequence.append('B')
            elif m in methods_a and m in methods_b:
                sequence.append('AB')  # shared method
            # else: unknown method, skip

        # Check interleaving
        if not sequence:
            continue

        has_a = 'A' in sequence
        has_b = 'B' in sequence

        if not has_a or not has_b:
            separated += 1  # only uses one interface
            continue

        # Check if A and B alternate
        # Remove shared methods for this analysis
        filtered = [s for s in sequence if s in ('A', 'B')]
        if not filtered:
            separated += 1
            continue

        # Count transitions between A and B
        transitions = sum(1 for i in range(len(filtered) - 1)
                         if filtered[i] != filtered[i + 1])

        if transitions == 0:
            separated += 1  # all A then all B (or vice versa)
        elif transitions == 1:
            mixed_ordered += 1  # one block of A then one block of B
        else:
            interleaved += 1  # true interleaving

    total = interleaved + separated + mixed_ordered
    rate = interleaved / total if total > 0 else 0.0

    return InterleaveAnalysis(
        interface_a="I_A",
        interface_b="I_B",
        total_clients=total,
        interleaved=interleaved,
        separated=separated,
        mixed_but_ordered=mixed_ordered,
        interleave_rate=rate,
    )


# ---------------------------------------------------------------------------
# Structural analysis (shared fields)
# ---------------------------------------------------------------------------

def _extract_field_access(method_node: Any) -> set[str]:
    """Extract field names accessed by a method (approximate)."""
    fields = set()
    for _, node in method_node.filter(MemberReference):
        if node.qualifier and node.qualifier == 'this':
            fields.add(node.member)
        elif not node.qualifier:
            # Could be a field or local — include as candidate
            fields.add(node.member)
    return fields


def analyze_shared_fields(
    class_source: str,
    methods_a: set[str],
    methods_b: set[str],
) -> tuple[list[str], str]:
    """Check if methods from I_A and I_B access the same fields.

    Returns (shared_field_names, classification).
    """
    if not HAS_JAVALANG:
        return [], "unknown"

    try:
        tree = javalang.parse.parse(class_source)
    except Exception:
        return [], "unknown"

    for _, cls in tree.filter(ClassDeclaration):
        # Collect fields
        all_fields = set()
        for _, fd in cls.filter(FieldDeclaration):
            for decl in fd.declarators:
                all_fields.add(decl.name)

        # Collect field access per method
        fields_by_method: dict[str, set[str]] = {}
        for _, md in cls.filter(MethodDeclaration):
            accessed = _extract_field_access(md)
            field_access = accessed & all_fields  # only actual fields
            fields_by_method[md.name] = field_access

        # Fields accessed by I_A methods
        a_fields = set()
        for m in methods_a:
            if m in fields_by_method:
                a_fields |= fields_by_method[m]

        # Fields accessed by I_B methods
        b_fields = set()
        for m in methods_b:
            if m in fields_by_method:
                b_fields |= fields_by_method[m]

        shared = a_fields & b_fields
        if shared:
            return list(shared), "shared_fields"
        elif a_fields and b_fields:
            return [], "separate_fields"
        else:
            return [], "unknown"

    return [], "unknown"


# ---------------------------------------------------------------------------
# Combined classification
# ---------------------------------------------------------------------------

def classify_composition(
    class_name: str,
    interface_a: InterfaceProfile,
    interface_b: InterfaceProfile,
    client_traces: list[list[str]],
    class_source: str = "",
) -> CompositionClassification:
    """Classify whether two interfaces are orthogonal or extending.

    Uses both structural (field sharing) and behavioral (client interleaving)
    evidence to determine the composition type.
    """
    methods_a = set(interface_a.methods)
    methods_b = set(interface_b.methods)

    # Analyze client interleaving
    ia = analyze_client_interleaving(client_traces, methods_a, methods_b)

    # Analyze shared fields (if source available)
    shared_fields, structural_evidence = analyze_shared_fields(
        class_source, methods_a, methods_b) if class_source else ([], "unknown")

    # Decision logic
    if ia.total_clients == 0:
        classification = "ambiguous"
        confidence = 0.0
        client_evidence = "no_clients"
    elif ia.interleave_rate > 0.5:
        classification = "extending"
        confidence = min(0.9, ia.interleave_rate)
        client_evidence = "interleaved"
    elif ia.interleave_rate < 0.1 and ia.separated > 3:
        classification = "orthogonal"
        confidence = min(0.9, 1.0 - ia.interleave_rate)
        client_evidence = "separated"
    else:
        classification = "ambiguous"
        confidence = 0.5
        client_evidence = "mixed"

    # Structural evidence adjusts confidence
    if structural_evidence == "shared_fields":
        if classification == "orthogonal":
            classification = "ambiguous"  # conflict
            confidence *= 0.5
        elif classification == "extending":
            confidence = min(1.0, confidence + 0.1)
    elif structural_evidence == "separate_fields":
        if classification == "extending":
            classification = "ambiguous"  # conflict
            confidence *= 0.5
        elif classification == "orthogonal":
            confidence = min(1.0, confidence + 0.1)

    # Session type suggestion
    if classification == "orthogonal":
        suggestion = "parallel"
        explanation = (f"Interfaces {interface_a.name} and {interface_b.name} are used "
                      f"independently by clients ({ia.separated}/{ia.total_clients} "
                      f"separated). Recommend: S({interface_a.name}) ∥ S({interface_b.name})")
    elif classification == "extending":
        suggestion = "branch_widening"
        explanation = (f"Interfaces {interface_a.name} and {interface_b.name} are "
                      f"interleaved by clients ({ia.interleaved}/{ia.total_clients} "
                      f"interleaved). Methods from both share the same state machine. "
                      f"Recommend: merge branches from both interfaces.")
    else:
        suggestion = "needs_investigation"
        explanation = (f"Insufficient evidence to classify ({ia.total_clients} clients, "
                      f"{ia.interleave_rate:.0%} interleaved). "
                      f"Structural: {structural_evidence}. Needs manual review.")

    if shared_fields:
        explanation += f" Shared fields: {shared_fields}."

    return CompositionClassification(
        class_name=class_name,
        interface_a=interface_a.name,
        interface_b=interface_b.name,
        classification=classification,
        evidence_structural=structural_evidence,
        evidence_clients=client_evidence,
        interleave_rate=ia.interleave_rate,
        shared_fields=shared_fields,
        confidence=confidence,
        session_type_suggestion=suggestion,
        explanation=explanation,
    )
