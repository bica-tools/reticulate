"""Client conformance checker against L3 session types (Step 80b).

Given a session type (the protocol) and a client trace (method call
sequence with selection outcomes), checks whether the trace conforms
to the protocol. Reports violations with explanations.

This is the core engine for the conformance-checking product:
  API spec → session type → check ANY client → violation report

Usage:
  from reticulate.conformance_checker import check_conformance
  result = check_conformance("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
                              ["hasNext", "TRUE", "next", "hasNext", "FALSE"])
  print(result)  # ConformanceResult(conforms=True, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from reticulate.parser import parse, pretty, SessionType
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    """A single protocol violation in a client trace."""
    step_index: int
    method: str
    state: int
    available_methods: list[str]
    explanation: str

    def __str__(self) -> str:
        return (f"Step {self.step_index}: '{self.method}' is not valid "
                f"(available: {self.available_methods}). {self.explanation}")


@dataclass(frozen=True)
class ConformanceResult:
    """Result of checking a trace against a session type."""
    conforms: bool
    session_type: str
    trace: list[str]
    trace_length: int
    violations: list[Violation]
    steps_completed: int  # how many steps succeeded before first violation
    final_state: Optional[int]
    is_complete: bool  # True if trace reached bottom (end)
    coverage: float  # fraction of transitions exercised

    def summary(self) -> str:
        if self.conforms and self.is_complete:
            return f"PASS: trace conforms and completes ({self.trace_length} steps, {self.coverage:.0%} coverage)"
        elif self.conforms and not self.is_complete:
            return f"INCOMPLETE: trace conforms but session not completed ({self.steps_completed}/{self.trace_length} steps)"
        else:
            n = len(self.violations)
            return f"FAIL: {n} violation(s) found at step {self.violations[0].step_index}"


@dataclass(frozen=True)
class BatchResult:
    """Result of checking multiple traces against a session type."""
    session_type: str
    total_traces: int
    conforming: int
    violating: int
    incomplete: int
    results: list[ConformanceResult]
    conformance_rate: float
    violation_summary: list[str]


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

def _get_available(ss: StateSpace, state: int) -> dict[str, int]:
    """Get available methods at a state: label → target state."""
    return {label: tgt for src, label, tgt in ss.transitions if src == state}


def _is_selection_transition(ss: StateSpace, src: int, label: str) -> bool:
    """Check if a transition is a selection (internal choice by environment)."""
    return (src, label) in {(s, l) for s, l, _ in ss.transitions
                            if (s, l) in {(s2, l2) for s2, l2 in ss.selection_transitions}}


def check_conformance(session_type: str, trace: list[str]) -> ConformanceResult:
    """Check if a client trace conforms to a session type.

    Parameters
    ----------
    session_type : str
        The protocol as a session type string.
    trace : list[str]
        Client method calls + selection outcomes in order.
        Example: ["hasNext", "TRUE", "next", "hasNext", "FALSE"]
        Selection outcomes (TRUE/FALSE/OK/ERR etc.) must be included.

    Returns
    -------
    ConformanceResult with violations if any.
    """
    ast = parse(session_type)
    ss = build_statespace(ast)

    violations: list[Violation] = []
    state = ss.top
    steps_completed = 0
    transitions_hit: set[tuple[int, str, int]] = set()

    for i, method in enumerate(trace):
        available = _get_available(ss, state)

        if method not in available:
            # Determine violation type
            if not available:
                explanation = "Session has ended — no more methods can be called."
            elif _is_at_selection_point(ss, state):
                explanation = (f"Protocol expects a selection outcome "
                               f"(one of {list(available.keys())}), "
                               f"not a method call. Did you forget to include "
                               f"the return value?")
            else:
                explanation = (f"Method '{method}' is not available in the current "
                               f"protocol state. The protocol expects one of: "
                               f"{list(available.keys())}.")

            violations.append(Violation(
                step_index=i,
                method=method,
                state=state,
                available_methods=list(available.keys()),
                explanation=explanation,
            ))
            # Stop at first violation (can't continue — state is unknown)
            break
        else:
            target = available[method]
            transitions_hit.add((state, method, target))
            state = target
            steps_completed += 1

    # Check completeness
    is_complete = (state == ss.bottom) and not violations
    total_transitions = len(ss.transitions)
    coverage = len(transitions_hit) / total_transitions if total_transitions > 0 else 0.0

    return ConformanceResult(
        conforms=len(violations) == 0,
        session_type=session_type,
        trace=trace,
        trace_length=len(trace),
        violations=violations,
        steps_completed=steps_completed,
        final_state=state,
        is_complete=is_complete,
        coverage=coverage,
    )


def _is_at_selection_point(ss: StateSpace, state: int) -> bool:
    """Check if a state is a selection point (all outgoing are selection transitions)."""
    outgoing = [(src, label) for src, label, _ in ss.transitions if src == state]
    if not outgoing:
        return False
    return all((src, label) in ss.selection_transitions for src, label in outgoing)


# ---------------------------------------------------------------------------
# Batch checking
# ---------------------------------------------------------------------------

def check_batch(session_type: str, traces: list[list[str]]) -> BatchResult:
    """Check multiple traces against a session type."""
    results: list[ConformanceResult] = []
    for trace in traces:
        results.append(check_conformance(session_type, trace))

    conforming = sum(1 for r in results if r.conforms and r.is_complete)
    violating = sum(1 for r in results if not r.conforms)
    incomplete = sum(1 for r in results if r.conforms and not r.is_complete)

    # Collect unique violation explanations
    violation_summary: list[str] = []
    seen: set[str] = set()
    for r in results:
        for v in r.violations:
            key = f"Step {v.step_index}: {v.method}"
            if key not in seen:
                seen.add(key)
                violation_summary.append(str(v))

    total = len(results)
    return BatchResult(
        session_type=session_type,
        total_traces=total,
        conforming=conforming,
        violating=violating,
        incomplete=incomplete,
        results=results,
        conformance_rate=conforming / total if total > 0 else 0.0,
        violation_summary=violation_summary,
    )


# ---------------------------------------------------------------------------
# Convenience: check from L3 extractor API description
# ---------------------------------------------------------------------------

def check_against_api(api_name: str, traces: list[list[str]]) -> BatchResult:
    """Check traces against a protocol from the L3 extractor registry.

    Parameters
    ----------
    api_name : str
        Name of the API in the L3 extractor registry.
    traces : list[list[str]]
        Client traces to check.
    """
    from reticulate.l3_extractor import API_REGISTRY, extract_session_type
    if api_name not in API_REGISTRY:
        raise ValueError(f"Unknown API: {api_name}. Available: {list(API_REGISTRY.keys())}")
    api = API_REGISTRY[api_name]()
    st = extract_session_type(api)
    return check_batch(st, traces)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(batch: BatchResult) -> str:
    """Format a batch conformance report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  CONFORMANCE CHECK REPORT")
    lines.append("=" * 70)
    lines.append(f"  Protocol: {batch.session_type[:80]}...")
    lines.append(f"  Traces checked: {batch.total_traces}")
    lines.append(f"  Conforming: {batch.conforming} ({batch.conformance_rate:.0%})")
    lines.append(f"  Violating: {batch.violating}")
    lines.append(f"  Incomplete: {batch.incomplete}")

    if batch.violation_summary:
        lines.append("")
        lines.append("  --- VIOLATIONS ---")
        for v in batch.violation_summary:
            lines.append(f"  {v}")

    lines.append("")
    lines.append("  --- TRACE DETAILS ---")
    for i, r in enumerate(batch.results):
        status = "PASS" if r.conforms and r.is_complete else "INCOMPLETE" if r.conforms else "FAIL"
        lines.append(f"  Trace {i+1}: [{', '.join(r.trace[:8])}{'...' if len(r.trace) > 8 else ''}] → {status}")

    lines.append("=" * 70)
    return "\n".join(lines)
