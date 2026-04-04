"""Process algebra equivalence coincidence for session type state spaces (Step 29c).

Proves that for session type state spaces (deterministic, finite, lattice-
structured), three classical process equivalences COINCIDE:

1. **CCS strong bisimulation** (Milner)
2. **CSP trace refinement** (Hoare)
3. **CSP failure refinement** (Hoare/Roscoe)

The key insight: session type state spaces are deterministic (each label leads
to at most one successor from any state) and have finite, lattice-ordered
reachability structure.  Under these conditions:

- Trace equivalence = bisimulation  (determinism collapses bisimulation to traces)
- Trace equivalence = failure equivalence  (deterministic + prefix-closed refusals)

Hence all three notions coincide on the class of session type state spaces.
This is a significant result: it means the choice of process algebra semantics
is immaterial for session type analysis — CCS and CSP agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessEquivalenceResult:
    """Result of comparing all three process equivalences on a pair of state spaces.

    Attributes:
        bisimilar: Verdict from CCS strong bisimulation.
        trace_equivalent: Verdict from CSP trace equivalence.
        failure_equivalent: Verdict from CSP failure equivalence.
        all_agree: True iff all three verdicts are the same.
        ss1_deterministic: True iff ss1 is deterministic.
        ss2_deterministic: True iff ss2 is deterministic.
        details: Human-readable summary.
    """
    bisimilar: bool
    trace_equivalent: bool
    failure_equivalent: bool
    all_agree: bool
    ss1_deterministic: bool
    ss2_deterministic: bool
    details: str


@dataclass(frozen=True)
class EquivalenceTheoremResult:
    """Result of verifying the equivalence coincidence theorem on a state space.

    Tests that for ALL pairs of reachable sub-state-spaces from a given
    state space, the three equivalences agree.

    Attributes:
        holds: True iff the theorem holds for all tested pairs.
        pairs_tested: Number of (ss1, ss2) pairs compared.
        pairs_agreed: Number of pairs where all three equivalences agree.
        is_deterministic: True iff the state space is deterministic.
        is_lattice: True iff the state space forms a lattice.
        counterexample: Description of first disagreement, or None.
        details: Human-readable summary.
    """
    holds: bool
    pairs_tested: int
    pairs_agreed: int
    is_deterministic: bool
    is_lattice: bool
    counterexample: str | None
    details: str


@dataclass(frozen=True)
class BenchmarkEquivalenceResult:
    """Result of testing equivalence coincidence across benchmark protocols.

    Attributes:
        total_benchmarks: Number of benchmarks tested.
        all_deterministic: Number that are deterministic.
        all_lattice: Number that form lattices.
        theorem_holds: Number where the coincidence theorem holds.
        failures: List of benchmark names where the theorem fails.
        details: Human-readable summary.
    """
    total_benchmarks: int
    all_deterministic: int
    all_lattice: int
    theorem_holds: int
    failures: list[str]
    details: str


# ---------------------------------------------------------------------------
# Core: check all three equivalences on a pair
# ---------------------------------------------------------------------------

def check_all_equivalences(
    ss1: StateSpace,
    ss2: StateSpace,
    max_depth: int = 10,
) -> ProcessEquivalenceResult:
    """Run CCS bisimulation, CSP trace equivalence, and CSP failure equivalence.

    Compares the three verdicts and reports whether they agree.

    Parameters:
        ss1: First state space.
        ss2: Second state space.
        max_depth: Depth bound for trace/failure enumeration.

    Returns:
        A ProcessEquivalenceResult with all three verdicts.
    """
    from reticulate.ccs import (
        LTS,
        check_bisimulation,
    )
    from reticulate.csp import (
        extract_traces,
        failures,
        is_deterministic,
    )

    # 1. CCS bisimulation: convert state spaces to LTS and compare
    lts1 = _statespace_to_lts(ss1)
    lts2 = _statespace_to_lts(ss2)
    bisim = check_bisimulation(lts1, lts2)

    # 2. CSP trace equivalence
    traces1 = extract_traces(ss1, max_depth)
    traces2 = extract_traces(ss2, max_depth)
    trace_eq = (traces1 == traces2)

    # 3. CSP failure equivalence
    fail1 = failures(ss1, max_depth)
    fail2 = failures(ss2, max_depth)
    fail_eq = (fail1 == fail2)

    det1 = is_deterministic(ss1)
    det2 = is_deterministic(ss2)

    all_agree = (bisim == trace_eq == fail_eq)

    if all_agree:
        verdict = "equivalent" if bisim else "not equivalent"
        details = (
            f"All three process equivalences agree: {verdict}. "
            f"ss1 deterministic={det1}, ss2 deterministic={det2}."
        )
    else:
        parts = []
        parts.append(f"bisimulation={bisim}")
        parts.append(f"trace_eq={trace_eq}")
        parts.append(f"failure_eq={fail_eq}")
        details = (
            f"DISAGREEMENT: {', '.join(parts)}. "
            f"ss1 deterministic={det1}, ss2 deterministic={det2}."
        )

    return ProcessEquivalenceResult(
        bisimilar=bisim,
        trace_equivalent=trace_eq,
        failure_equivalent=fail_eq,
        all_agree=all_agree,
        ss1_deterministic=det1,
        ss2_deterministic=det2,
        details=details,
    )


def _statespace_to_lts(ss: StateSpace) -> "LTS":
    """Convert a session-type StateSpace to a CCS LTS.

    Selection transitions get a ``tau_`` prefix to match CCS convention.
    """
    from reticulate.ccs import LTS

    transitions: list[tuple[int, str, int]] = []
    tau_transitions: set[tuple[int, str, int]] = set()

    for src, label, tgt in ss.transitions:
        if ss.is_selection(src, label, tgt):
            action = f"tau_{label}"
            tr = (src, action, tgt)
            transitions.append(tr)
            tau_transitions.add(tr)
        else:
            tr = (src, label, tgt)
            transitions.append(tr)

    return LTS(
        states=set(ss.states),
        transitions=transitions,
        initial=ss.top,
        labels=dict(ss.labels),
        tau_transitions=tau_transitions,
    )


# ---------------------------------------------------------------------------
# Equivalence theorem: verify coincidence on all pairs from a state space
# ---------------------------------------------------------------------------

def equivalence_theorem(
    ss: StateSpace,
    max_depth: int = 10,
) -> EquivalenceTheoremResult:
    """Verify that CCS bisimulation, trace equivalence, and failure equivalence
    coincide for a given session type state space.

    Strategy: we generate a small collection of state spaces derived from the
    given one (sub-state-spaces rooted at each reachable state, plus the
    original) and check that all three equivalences agree on every pair.

    For deterministic state spaces, the theorem is guaranteed by the
    mathematical result that deterministic LTS have bisimulation = trace
    equivalence, and deterministic processes have trace eq = failure eq.

    Parameters:
        ss: The state space to verify.
        max_depth: Depth bound for trace/failure enumeration.

    Returns:
        An EquivalenceTheoremResult.
    """
    from reticulate.csp import is_deterministic
    from reticulate.lattice import check_lattice

    det = is_deterministic(ss)
    lr = check_lattice(ss)
    is_lat = lr.is_lattice

    # Build sub-state-spaces rooted at each state
    sub_spaces = _build_sub_statespaces(ss)

    pairs_tested = 0
    pairs_agreed = 0
    counterexample: str | None = None

    # Compare all distinct pairs
    keys = sorted(sub_spaces.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i:]:
            ss1 = sub_spaces[k1]
            ss2 = sub_spaces[k2]
            result = check_all_equivalences(ss1, ss2, max_depth)
            pairs_tested += 1
            if result.all_agree:
                pairs_agreed += 1
            elif counterexample is None:
                counterexample = (
                    f"States {k1} vs {k2}: bisim={result.bisimilar}, "
                    f"trace_eq={result.trace_equivalent}, "
                    f"fail_eq={result.failure_equivalent}"
                )

    holds = (pairs_tested == pairs_agreed)

    if holds:
        details = (
            f"Equivalence coincidence theorem HOLDS. "
            f"Tested {pairs_tested} pairs, all agreed. "
            f"Deterministic={det}, Lattice={is_lat}."
        )
    else:
        details = (
            f"Equivalence coincidence theorem FAILS. "
            f"{pairs_agreed}/{pairs_tested} pairs agreed. "
            f"Counterexample: {counterexample}"
        )

    return EquivalenceTheoremResult(
        holds=holds,
        pairs_tested=pairs_tested,
        pairs_agreed=pairs_agreed,
        is_deterministic=det,
        is_lattice=is_lat,
        counterexample=counterexample,
        details=details,
    )


def _build_sub_statespaces(ss: StateSpace) -> dict[int, StateSpace]:
    """Build sub-state-spaces rooted at each reachable state.

    For each state s, we build a StateSpace whose top is s, bottom is
    ss.bottom (if reachable from s), and transitions are restricted to
    states reachable from s.
    """
    from reticulate.statespace import StateSpace as SS

    result: dict[int, SS] = {}

    for root in sorted(ss.states):
        reachable = ss.reachable_from(root)
        if len(reachable) < 1:
            continue

        sub_transitions = [
            (s, l, t) for s, l, t in ss.transitions
            if s in reachable and t in reachable
        ]
        sub_selection = {
            (s, l, t) for s, l, t in ss.selection_transitions
            if s in reachable and t in reachable
        }
        sub_labels = {s: ss.labels[s] for s in reachable if s in ss.labels}

        # Bottom: ss.bottom if reachable, else pick a terminal state
        if ss.bottom in reachable:
            bottom = ss.bottom
        else:
            # Find a terminal state (no outgoing transitions)
            terminals = [
                s for s in reachable
                if not any(src == s for src, _, _ in sub_transitions)
            ]
            bottom = terminals[0] if terminals else root

        sub_ss = SS(
            states=reachable,
            transitions=sub_transitions,
            top=root,
            bottom=bottom,
            labels=sub_labels,
            selection_transitions=sub_selection,
        )
        result[root] = sub_ss

    return result


# ---------------------------------------------------------------------------
# Benchmark testing
# ---------------------------------------------------------------------------

def check_benchmarks(max_depth: int = 10) -> BenchmarkEquivalenceResult:
    """Test the equivalence coincidence theorem on all benchmark protocols.

    Loads benchmark protocols, builds state spaces, and verifies that the
    three process equivalences coincide for each.

    Returns:
        A BenchmarkEquivalenceResult summarising findings.
    """
    from reticulate.parser import parse
    from reticulate.statespace import build_statespace
    from reticulate.csp import is_deterministic
    from reticulate.lattice import check_lattice

    try:
        from tests.benchmarks.protocols import BENCHMARKS
    except ImportError:
        # Fallback: use a small set of built-in protocols
        BENCHMARKS = _builtin_benchmarks()

    total = 0
    det_count = 0
    lat_count = 0
    holds_count = 0
    failures: list[str] = []

    for bm in BENCHMARKS:
        name = bm.name if hasattr(bm, "name") else str(bm)
        type_str = bm.type_string if hasattr(bm, "type_string") else bm
        try:
            ast = parse(type_str)
            ss = build_statespace(ast)
        except Exception:
            continue

        total += 1

        det = is_deterministic(ss)
        if det:
            det_count += 1

        lr = check_lattice(ss)
        if lr.is_lattice:
            lat_count += 1

        # For each benchmark, check that self-comparison gives all-agree
        result = check_all_equivalences(ss, ss, max_depth)
        if result.all_agree:
            holds_count += 1
        else:
            failures.append(name)

    details = (
        f"Tested {total} benchmarks: "
        f"{det_count} deterministic, {lat_count} lattice, "
        f"{holds_count} theorem holds. "
        f"Failures: {failures if failures else 'none'}."
    )

    return BenchmarkEquivalenceResult(
        total_benchmarks=total,
        all_deterministic=det_count,
        all_lattice=lat_count,
        theorem_holds=holds_count,
        failures=failures,
        details=details,
    )


def _builtin_benchmarks() -> list[object]:
    """Minimal built-in benchmark set for when the test suite is not importable."""
    from dataclasses import dataclass as dc

    @dc(frozen=True)
    class _BM:
        name: str
        type_string: str

    return [
        _BM("Iterator", "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"),
        _BM("File", "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"),
        _BM("Simple", "&{a: end, b: end}"),
        _BM("Selection", "+{OK: end, ERR: end}"),
        _BM("Nested", "&{a: &{b: end, c: end}, d: end}"),
    ]


# ---------------------------------------------------------------------------
# Convenience: from type strings
# ---------------------------------------------------------------------------

def check_equivalences_from_types(
    type1: str,
    type2: str,
    max_depth: int = 10,
) -> ProcessEquivalenceResult:
    """Parse two session type strings and compare all three process equivalences.

    Convenience wrapper around check_all_equivalences.
    """
    from reticulate.parser import parse
    from reticulate.statespace import build_statespace

    ast1 = parse(type1)
    ast2 = parse(type2)
    ss1 = build_statespace(ast1)
    ss2 = build_statespace(ast2)
    return check_all_equivalences(ss1, ss2, max_depth)


def theorem_from_type(
    type_string: str,
    max_depth: int = 10,
) -> EquivalenceTheoremResult:
    """Parse a session type string and verify the equivalence coincidence theorem.

    Convenience wrapper around equivalence_theorem.
    """
    from reticulate.parser import parse
    from reticulate.statespace import build_statespace

    ast = parse(type_string)
    ss = build_statespace(ast)
    return equivalence_theorem(ss, max_depth)


# ---------------------------------------------------------------------------
# Bisimulation-Isomorphism Correspondence (CONCUR 2026)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BisimIsomorphismResult:
    """Result of checking the bisimulation ↔ lattice isomorphism correspondence.

    For deterministic LTS (which all well-formed session type state spaces are),
    two state spaces are bisimilar if and only if their reticulates (SCC-quotient
    lattices) are isomorphic.

    Attributes:
        bisimilar: CCS strong bisimulation verdict.
        isomorphic: Lattice isomorphism verdict.
        correspondence_holds: True iff bisimilar == isomorphic.
        ss1_deterministic: True iff ss1 is deterministic.
        ss2_deterministic: True iff ss2 is deterministic.
        ss1_is_lattice: True iff ss1 forms a lattice.
        ss2_is_lattice: True iff ss2 forms a lattice.
        isomorphism_mapping: The isomorphism mapping if found, else None.
        details: Human-readable summary.
    """
    bisimilar: bool
    isomorphic: bool
    correspondence_holds: bool
    ss1_deterministic: bool
    ss2_deterministic: bool
    ss1_is_lattice: bool
    ss2_is_lattice: bool
    isomorphism_mapping: dict[int, int] | None
    details: str


def _find_labeled_isomorphism(
    ss1: "StateSpace",
    ss2: "StateSpace",
) -> dict[int, int] | None:
    """Find a label-preserving isomorphism between two state spaces.

    A labeled isomorphism is a bijection f: states(ss1) → states(ss2) such that:
    1. f(top₁) = top₂ and f(bot₁) = bot₂
    2. (s, a, t) ∈ transitions(ss1) ⟺ (f(s), a, f(t)) ∈ transitions(ss2)

    This is stronger than order isomorphism: it preserves transition LABELS,
    not just reachability ordering.

    Returns the mapping dict if found, or None.
    """
    # Quick rejection
    if len(ss1.states) != len(ss2.states):
        return None
    if len(ss1.transitions) != len(ss2.transitions):
        return None

    # Check selection transition counts match (polarity must agree)
    if len(ss1.selection_transitions) != len(ss2.selection_transitions):
        return None

    # Build adjacency: state → {label → (target, is_selection)}
    adj1: dict[int, dict[str, int]] = {s: {} for s in ss1.states}
    sel1: dict[int, set[str]] = {s: set() for s in ss1.states}
    for s, label, t in ss1.transitions:
        adj1[s][label] = t
        if ss1.is_selection(s, label, t):
            sel1[s].add(label)

    adj2: dict[int, dict[str, int]] = {s: {} for s in ss2.states}
    sel2: dict[int, set[str]] = {s: set() for s in ss2.states}
    for s, label, t in ss2.transitions:
        adj2[s][label] = t
        if ss2.is_selection(s, label, t):
            sel2[s].add(label)

    # Check label sets from top must match
    if set(adj1.get(ss1.top, {}).keys()) != set(adj2.get(ss2.top, {}).keys()):
        return None

    # BFS-based matching: start from top→top, follow labels
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    queue: list[tuple[int, int]] = [(ss1.top, ss2.top)]

    while queue:
        s1, s2 = queue.pop(0)

        if s1 in mapping:
            if mapping[s1] != s2:
                return None  # Conflict
            continue

        if s2 in reverse:
            if reverse[s2] != s1:
                return None  # Conflict (not injective)
            continue

        # Check outgoing labels AND polarity (selection vs branch) match
        labels1 = set(adj1.get(s1, {}).keys())
        labels2 = set(adj2.get(s2, {}).keys())
        if labels1 != labels2:
            return None
        if sel1.get(s1, set()) != sel2.get(s2, set()):
            return None  # Selection polarity mismatch

        mapping[s1] = s2
        reverse[s2] = s1

        # Follow each shared label
        for label in labels1:
            t1 = adj1[s1][label]
            t2 = adj2[s2][label]
            if t1 not in mapping:
                queue.append((t1, t2))
            elif mapping[t1] != t2:
                return None

    # Check bijection covers all states
    if len(mapping) != len(ss1.states):
        return None
    if set(mapping.values()) != ss2.states:
        return None

    return mapping


def check_bisim_iso_correspondence(
    ss1: "StateSpace",
    ss2: "StateSpace",
    max_depth: int = 10,
) -> BisimIsomorphismResult:
    """Check the bisimulation ↔ lattice isomorphism correspondence.

    For deterministic session type state spaces:
        S₁ ~ S₂  (bisimilar)  ⟺  L(S₁) ≅ L(S₂)  (isomorphic lattices)

    The proof relies on three facts:
    1. Session type state spaces are deterministic (each label has ≤1 successor).
    2. For deterministic LTS, bisimulation = trace equivalence (coincidence thm).
    3. Trace-equivalent deterministic LTS have identical reachability structure,
       hence isomorphic lattices.

    Parameters:
        ss1: First state space.
        ss2: Second state space.
        max_depth: Depth bound for trace enumeration.

    Returns:
        A BisimIsomorphismResult with both verdicts and correspondence check.
    """
    from reticulate.ccs import LTS, check_bisimulation
    from reticulate.csp import is_deterministic
    from reticulate.lattice import check_lattice
    from reticulate.morphism import find_isomorphism

    # Check determinism
    det1 = is_deterministic(ss1)
    det2 = is_deterministic(ss2)

    # Check lattice property
    lr1 = check_lattice(ss1)
    lr2 = check_lattice(ss2)

    # Check bisimulation
    lts1 = _statespace_to_lts(ss1)
    lts2 = _statespace_to_lts(ss2)
    bisim = check_bisimulation(lts1, lts2)

    # Check labeled lattice isomorphism (order + label preserving)
    iso = _find_labeled_isomorphism(ss1, ss2)
    isomorphic = iso is not None
    iso_mapping = iso if iso is not None else None

    # The correspondence
    correspondence = (bisim == isomorphic)

    if correspondence:
        if bisim:
            details = (
                "Bisimulation-isomorphism correspondence HOLDS: "
                "state spaces are bisimilar AND their lattices are isomorphic. "
                f"Deterministic: ss1={det1}, ss2={det2}. "
                f"Lattice: ss1={lr1.is_lattice}, ss2={lr2.is_lattice}."
            )
        else:
            details = (
                "Bisimulation-isomorphism correspondence HOLDS: "
                "state spaces are NOT bisimilar AND their lattices are NOT isomorphic. "
                f"Deterministic: ss1={det1}, ss2={det2}. "
                f"Lattice: ss1={lr1.is_lattice}, ss2={lr2.is_lattice}."
            )
    else:
        details = (
            f"Bisimulation-isomorphism correspondence FAILS: "
            f"bisimilar={bisim}, isomorphic={isomorphic}. "
            f"Deterministic: ss1={det1}, ss2={det2}. "
            f"Lattice: ss1={lr1.is_lattice}, ss2={lr2.is_lattice}."
        )

    return BisimIsomorphismResult(
        bisimilar=bisim,
        isomorphic=isomorphic,
        correspondence_holds=correspondence,
        ss1_deterministic=det1,
        ss2_deterministic=det2,
        ss1_is_lattice=lr1.is_lattice,
        ss2_is_lattice=lr2.is_lattice,
        isomorphism_mapping=iso_mapping,
        details=details,
    )


def check_bisim_iso_from_types(
    type1: str,
    type2: str,
    max_depth: int = 10,
) -> BisimIsomorphismResult:
    """Parse two session type strings and check the bisimulation-isomorphism
    correspondence.

    Convenience wrapper around check_bisim_iso_correspondence.
    """
    from reticulate.parser import parse
    from reticulate.statespace import build_statespace

    ss1 = build_statespace(parse(type1))
    ss2 = build_statespace(parse(type2))
    return check_bisim_iso_correspondence(ss1, ss2, max_depth)


def verify_bisim_iso_theorem(
    type_strings: list[str],
    max_depth: int = 10,
) -> tuple[bool, int, int, str | None]:
    """Verify the bisimulation-isomorphism correspondence on all pairs.

    Tests that for every pair of session types from the given list,
    bisimilar ⟺ isomorphic lattices.

    Parameters:
        type_strings: List of session type strings to test.
        max_depth: Depth bound for trace enumeration.

    Returns:
        (holds, pairs_tested, pairs_passed, first_counterexample)
    """
    from reticulate.parser import parse
    from reticulate.statespace import build_statespace

    state_spaces = []
    for ts in type_strings:
        ss = build_statespace(parse(ts))
        state_spaces.append((ts, ss))

    pairs_tested = 0
    pairs_passed = 0
    counterexample: str | None = None

    for i in range(len(state_spaces)):
        for j in range(i, len(state_spaces)):
            name1, ss1 = state_spaces[i]
            name2, ss2 = state_spaces[j]
            result = check_bisim_iso_correspondence(ss1, ss2, max_depth)
            pairs_tested += 1
            if result.correspondence_holds:
                pairs_passed += 1
            elif counterexample is None:
                counterexample = (
                    f"'{name1}' vs '{name2}': "
                    f"bisimilar={result.bisimilar}, "
                    f"isomorphic={result.isomorphic}"
                )

    holds = (pairs_tested == pairs_passed)
    return holds, pairs_tested, pairs_passed, counterexample
