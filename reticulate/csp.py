"""CSP trace semantics from session type state spaces (Step 27).

Translates session type state spaces (labeled transition systems) into
Hoare's CSP (Communicating Sequential Processes) semantic models:

- **Traces model**: The set of all finite event sequences (traces) that the
  process can perform, obtained by enumerating paths from the top state.
- **Failures model**: Pairs (trace, refusal_set) where the refusal set is the
  set of events that cannot occur after executing the trace.

Key results:
  - Trace refinement (T₁ ⊆ T₂) corresponds to session type subtyping:
    a subtype can do everything the supertype can.
  - Refusal sets at each state correspond to disabled methods in the
    state space, connecting lattice structure to CSP failures.
  - Complete traces (top → bottom) are the maximal accepted behaviours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

Trace = tuple[str, ...]
"""A CSP trace: a finite sequence of event names."""

Failure = tuple[Trace, frozenset[str]]
"""A CSP failure: a trace paired with a refusal set."""


@dataclass(frozen=True)
class CSPResult:
    """Complete CSP semantic characterisation of a state space.

    Attributes:
        traces: All finite traces (prefixes of paths from top).
        complete_traces: Traces that reach the bottom state (complete executions).
        failures: Set of (trace, refusal_set) pairs.
        alphabet: The full event alphabet (all transition labels).
    """
    traces: frozenset[Trace]
    complete_traces: frozenset[Trace]
    failures: frozenset[Failure]
    alphabet: frozenset[str]


# ---------------------------------------------------------------------------
# Trace extraction
# ---------------------------------------------------------------------------


def extract_traces(ss: StateSpace, max_depth: int = 10) -> frozenset[Trace]:
    """Enumerate all finite traces from the state space.

    A trace is any sequence of transition labels along a path starting
    from ``ss.top``.  The empty trace ``()`` is always included (the
    process that does nothing).  Paths are explored up to *max_depth*
    transitions to ensure termination on cyclic state spaces.

    Returns:
        A frozenset of traces (tuples of event name strings).
    """
    traces: set[Trace] = set()
    _trace_dfs(ss, ss.top, (), traces, max_depth)
    return frozenset(traces)


def _trace_dfs(
    ss: StateSpace,
    state: int,
    current: Trace,
    traces: set[Trace],
    remaining: int,
) -> None:
    """DFS trace enumerator."""
    traces.add(current)
    if remaining <= 0:
        return
    for label, target in ss.enabled(state):
        _trace_dfs(ss, target, current + (label,), traces, remaining - 1)


def extract_complete_traces(
    ss: StateSpace, max_depth: int = 10,
) -> frozenset[Trace]:
    """Enumerate traces that reach the bottom state (complete executions).

    A complete trace is a path from ``ss.top`` to ``ss.bottom``.  These
    correspond to full protocol runs where the session reaches ``end``.

    Returns:
        A frozenset of complete traces.
    """
    traces: set[Trace] = set()
    _complete_dfs(ss, ss.top, (), traces, max_depth)
    return frozenset(traces)


def _complete_dfs(
    ss: StateSpace,
    state: int,
    current: Trace,
    traces: set[Trace],
    remaining: int,
) -> None:
    """DFS for complete traces (paths reaching bottom)."""
    if state == ss.bottom:
        traces.add(current)
        return
    if remaining <= 0:
        return
    for label, target in ss.enabled(state):
        _complete_dfs(ss, target, current + (label,), traces, remaining - 1)


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------


def alphabet(ss: StateSpace) -> frozenset[str]:
    """Extract the event alphabet from a state space.

    The alphabet is the set of all transition labels.
    """
    return frozenset(label for _, label, _ in ss.transitions)


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def failures(ss: StateSpace, max_depth: int = 10) -> frozenset[Failure]:
    """Compute the failure set of a state space.

    A failure ``(t, X)`` means: after performing trace ``t``, the process
    can refuse all events in ``X``.  The refusal set at a state is the
    complement of enabled events relative to the full alphabet.

    At the bottom state (``end``), the process refuses the entire alphabet
    (no further events are possible).

    Returns:
        A frozenset of (trace, refusal_set) pairs.
    """
    alpha = alphabet(ss)
    result: set[Failure] = set()
    _failures_dfs(ss, ss.top, (), result, alpha, max_depth)
    return frozenset(result)


def _failures_dfs(
    ss: StateSpace,
    state: int,
    current: Trace,
    result: set[Failure],
    alpha: frozenset[str],
    remaining: int,
) -> None:
    """DFS for failure set computation."""
    enabled_labels = frozenset(label for label, _ in ss.enabled(state))
    refusal = alpha - enabled_labels
    result.add((current, refusal))
    if remaining <= 0:
        return
    for label, target in ss.enabled(state):
        _failures_dfs(ss, target, current + (label,), result, alpha, remaining - 1)


# ---------------------------------------------------------------------------
# Refinement checks
# ---------------------------------------------------------------------------


def trace_refinement(
    traces1: frozenset[Trace], traces2: frozenset[Trace],
) -> bool:
    """Check trace refinement: traces1 is a trace refinement of traces2.

    In CSP, ``P`` refines ``Q`` in the traces model iff
    ``traces(P) ⊆ traces(Q)``.  A refinement can only *reduce* the set
    of observable behaviours.

    Returns:
        True if traces1 is a subset of traces2 (i.e. process 1 refines process 2).
    """
    return traces1 <= traces2


def check_failures_refinement(
    ss1: StateSpace, ss2: StateSpace, max_depth: int = 10,
) -> FailuresRefinementResult:
    """Check failures refinement between two state spaces.

    ``P`` refines ``Q`` in the failures model iff:
      1. traces(P) ⊆ traces(Q)   — P cannot do more than Q
      2. failures(P) ⊇ failures(Q) — wait, that's wrong.

    Actually in CSP failures refinement: ``P ⊑_F Q`` iff
      failures(P) ⊆ failures(Q).

    This is STRONGER than trace refinement: not only must P's traces be
    contained in Q's, but P must also refuse at least as much as Q at
    every point.

    Returns:
        A FailuresRefinementResult with the verdict and counterexamples.
    """
    traces1 = extract_traces(ss1, max_depth)
    traces2 = extract_traces(ss2, max_depth)
    failures1 = failures(ss1, max_depth)
    failures2 = failures(ss2, max_depth)

    traces_ok = traces1 <= traces2
    failures_ok = failures1 <= failures2

    trace_counterexamples = frozenset(traces1 - traces2) if not traces_ok else frozenset()
    failure_counterexamples = frozenset(failures1 - failures2) if not failures_ok else frozenset()

    return FailuresRefinementResult(
        is_refinement=traces_ok and failures_ok,
        traces_refined=traces_ok,
        failures_refined=failures_ok,
        trace_counterexamples=trace_counterexamples,
        failure_counterexamples=failure_counterexamples,
    )


@dataclass(frozen=True)
class FailuresRefinementResult:
    """Result of a failures refinement check.

    Attributes:
        is_refinement: True if ss1 failures-refines ss2.
        traces_refined: True if traces(ss1) ⊆ traces(ss2).
        failures_refined: True if failures(ss1) ⊆ failures(ss2).
        trace_counterexamples: Traces in ss1 not in ss2.
        failure_counterexamples: Failures in ss1 not in ss2.
    """
    is_refinement: bool
    traces_refined: bool
    failures_refined: bool
    trace_counterexamples: frozenset[Trace] = field(default=frozenset())
    failure_counterexamples: frozenset[Failure] = field(default=frozenset())


# ---------------------------------------------------------------------------
# Convenience: from type string
# ---------------------------------------------------------------------------


def traces_from_type(
    type_string: str, max_depth: int = 10,
) -> frozenset[Trace]:
    """Parse a session type string, build state space, extract traces.

    Convenience function that chains parse → build_statespace → extract_traces.

    Example::

        >>> sorted(traces_from_type("&{a: end, b: end}"))
        [(), ('a',), ('b',)]
    """
    ast = parse(type_string)
    ss = build_statespace(ast)
    return extract_traces(ss, max_depth)


def csp_analysis(
    ss: StateSpace, max_depth: int = 10,
) -> CSPResult:
    """Full CSP semantic analysis of a state space.

    Computes traces, complete traces, failures, and alphabet.

    Returns:
        A CSPResult with all CSP semantic data.
    """
    t = extract_traces(ss, max_depth)
    ct = extract_complete_traces(ss, max_depth)
    f = failures(ss, max_depth)
    a = alphabet(ss)
    return CSPResult(traces=t, complete_traces=ct, failures=f, alphabet=a)


def csp_from_type(
    type_string: str, max_depth: int = 10,
) -> CSPResult:
    """Full CSP analysis from a session type string.

    Convenience function that chains parse → build_statespace → csp_analysis.
    """
    ast = parse(type_string)
    ss = build_statespace(ast)
    return csp_analysis(ss, max_depth)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def pretty_trace(t: Trace) -> str:
    """Format a trace using CSP angle-bracket notation.

    Examples::

        >>> pretty_trace(())
        '⟨⟩'
        >>> pretty_trace(('a', 'b', 'c'))
        '⟨a, b, c⟩'
    """
    if not t:
        return "\u27e8\u27e9"  # ⟨⟩
    return "\u27e8" + ", ".join(t) + "\u27e9"


def pretty_failures(f: frozenset[Failure]) -> str:
    """Format a failure set in CSP notation.

    Each failure is shown as ``(⟨trace⟩, {refusals})``.

    Example::

        >>> pretty_failures(frozenset({(('a',), frozenset({'b'}))}))
        '(⟨a⟩, {b})'
    """
    parts: list[str] = []
    for trace, refusals in sorted(f):
        ref_str = "{" + ", ".join(sorted(refusals)) + "}"
        parts.append(f"({pretty_trace(trace)}, {ref_str})")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Determinism check
# ---------------------------------------------------------------------------


def is_deterministic(ss: StateSpace) -> bool:
    """Check if the state space is deterministic.

    A state space is deterministic if from every state, each label leads
    to at most one target state.  CSP distinguishes deterministic and
    non-deterministic processes; session types with external choice (&)
    are deterministic by construction (each label appears at most once).
    """
    for state in ss.states:
        labels_seen: dict[str, int] = {}
        for label, target in ss.enabled(state):
            if label in labels_seen and labels_seen[label] != target:
                return False
            labels_seen[label] = target
    return True


# ---------------------------------------------------------------------------
# Trace equivalence and preorder
# ---------------------------------------------------------------------------


def trace_equivalent(ss1: StateSpace, ss2: StateSpace, max_depth: int = 10) -> bool:
    """Check trace equivalence: traces(ss1) == traces(ss2).

    Two processes are trace equivalent if they can perform exactly the
    same sequences of events.
    """
    t1 = extract_traces(ss1, max_depth)
    t2 = extract_traces(ss2, max_depth)
    return t1 == t2


def failures_equivalent(
    ss1: StateSpace, ss2: StateSpace, max_depth: int = 10,
) -> bool:
    """Check failures equivalence: failures(ss1) == failures(ss2).

    Stronger than trace equivalence: two processes must also refuse the
    same events at corresponding points.
    """
    f1 = failures(ss1, max_depth)
    f2 = failures(ss2, max_depth)
    return f1 == f2
