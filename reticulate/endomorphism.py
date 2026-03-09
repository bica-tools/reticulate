"""Transitions as lattice endomorphisms (Step 10).

For each transition label m in L(S), the partial function f_m maps each
state s to the unique target t where s --m--> t (if the transition exists).

This module checks whether these partial functions are:
  1. Order-preserving: s₁ ≥ s₂ implies f_m(s₁) ≥ f_m(s₂)
  2. Meet-preserving: f_m(s₁ ∧ s₂) = f_m(s₁) ∧ f_m(s₂)
  3. Join-preserving: f_m(s₁ ∨ s₂) = f_m(s₁) ∨ f_m(s₂)

A partial function that is order-preserving is a monotone map.
A partial function that preserves meets and joins is a lattice homomorphism.
A lattice endomorphism is a lattice homomorphism from a lattice to itself.

The Step 10 question: do transition labels induce lattice endomorphisms
on the state space?
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.lattice import compute_join, compute_meet
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransitionMap:
    """A partial function induced by a transition label.

    Attributes:
        label: The transition label.
        mapping: Dictionary from source state to target state.
        domain_size: Number of states where this label is enabled.
        is_selection: True if this label is a selection transition.
    """
    label: str
    mapping: dict[int, int]
    domain_size: int
    is_selection: bool


@dataclass(frozen=True)
class EndomorphismResult:
    """Result of checking whether a transition map is an endomorphism.

    Attributes:
        label: The transition label.
        domain_size: Number of states where this label is enabled.
        is_order_preserving: True iff the partial map is monotone.
        is_meet_preserving: True iff f(a ∧ b) = f(a) ∧ f(b) when all defined.
        is_join_preserving: True iff f(a ∨ b) = f(a) ∨ f(b) when all defined.
        is_endomorphism: True iff meet- and join-preserving (lattice homo).
        order_counterexample: Pair (a, b) violating monotonicity, or None.
        meet_counterexample: Pair (a, b) violating meet preservation, or None.
        join_counterexample: Pair (a, b) violating join preservation, or None.
    """
    label: str
    domain_size: int
    is_order_preserving: bool
    is_meet_preserving: bool
    is_join_preserving: bool
    is_endomorphism: bool
    order_counterexample: tuple[int, int] | None = None
    meet_counterexample: tuple[int, int] | None = None
    join_counterexample: tuple[int, int] | None = None


@dataclass(frozen=True)
class EndomorphismSummary:
    """Summary of endomorphism checks for all labels in a state space.

    Attributes:
        results: Per-label endomorphism results.
        all_order_preserving: True iff every label is order-preserving.
        all_meet_preserving: True iff every label is meet-preserving.
        all_join_preserving: True iff every label is join-preserving.
        all_endomorphisms: True iff every label is a lattice endomorphism.
        num_labels: Total number of distinct labels.
    """
    results: tuple[EndomorphismResult, ...]
    all_order_preserving: bool
    all_meet_preserving: bool
    all_join_preserving: bool
    all_endomorphisms: bool
    num_labels: int


# ---------------------------------------------------------------------------
# Transition map extraction
# ---------------------------------------------------------------------------

def extract_transition_maps(ss: StateSpace) -> list[TransitionMap]:
    """Extract the partial function for each transition label.

    For label m, f_m(s) = t iff (s, m, t) ∈ transitions.
    If m is enabled at multiple states, f_m is defined on all of them.
    """
    label_maps: dict[str, dict[int, int]] = {}
    label_is_sel: dict[str, bool] = {}

    for src, label, tgt in ss.transitions:
        if label not in label_maps:
            label_maps[label] = {}
            label_is_sel[label] = False
        label_maps[label][src] = tgt
        if (src, label, tgt) in ss.selection_transitions:
            label_is_sel[label] = True

    return [
        TransitionMap(
            label=label,
            mapping=mapping,
            domain_size=len(mapping),
            is_selection=label_is_sel.get(label, False),
        )
        for label, mapping in sorted(label_maps.items())
    ]


# ---------------------------------------------------------------------------
# Endomorphism checking
# ---------------------------------------------------------------------------

def check_endomorphism(
    ss: StateSpace,
    tmap: TransitionMap,
) -> EndomorphismResult:
    """Check whether a transition map is an order-preserving / lattice endomorphism."""
    f = tmap.mapping
    domain = list(f.keys())

    # Order-preserving check: for all a, b in domain where a ≥ b, f(a) ≥ f(b)
    order_ok = True
    order_cx = None
    for a in domain:
        reach_a = ss.reachable_from(a)
        for b in domain:
            if b in reach_a and b != a:
                # a ≥ b (a reaches b), so f(a) should reach f(b)
                fa, fb = f[a], f[b]
                reach_fa = ss.reachable_from(fa)
                if fb not in reach_fa:
                    order_ok = False
                    order_cx = (a, b)
                    break
        if not order_ok:
            break

    # Meet-preserving check: f(a ∧ b) = f(a) ∧ f(b)
    meet_ok = True
    meet_cx = None
    for i, a in enumerate(domain):
        for b in domain[i + 1:]:
            m_ab = compute_meet(ss, a, b)
            if m_ab is not None and m_ab in f:
                fa, fb = f[a], f[b]
                f_meet = f[m_ab]
                meet_fa_fb = compute_meet(ss, fa, fb)
                if meet_fa_fb != f_meet:
                    meet_ok = False
                    meet_cx = (a, b)
                    break
        if not meet_ok:
            break

    # Join-preserving check: f(a ∨ b) = f(a) ∨ f(b)
    join_ok = True
    join_cx = None
    for i, a in enumerate(domain):
        for b in domain[i + 1:]:
            j_ab = compute_join(ss, a, b)
            if j_ab is not None and j_ab in f:
                fa, fb = f[a], f[b]
                f_join = f[j_ab]
                join_fa_fb = compute_join(ss, fa, fb)
                if join_fa_fb != f_join:
                    join_ok = False
                    join_cx = (a, b)
                    break
        if not join_ok:
            break

    return EndomorphismResult(
        label=tmap.label,
        domain_size=tmap.domain_size,
        is_order_preserving=order_ok,
        is_meet_preserving=meet_ok,
        is_join_preserving=join_ok,
        is_endomorphism=meet_ok and join_ok,
        order_counterexample=order_cx,
        meet_counterexample=meet_cx,
        join_counterexample=join_cx,
    )


def check_all_endomorphisms(ss: StateSpace) -> EndomorphismSummary:
    """Check endomorphism properties for all transition labels."""
    tmaps = extract_transition_maps(ss)
    results = tuple(check_endomorphism(ss, tm) for tm in tmaps)

    return EndomorphismSummary(
        results=results,
        all_order_preserving=all(r.is_order_preserving for r in results),
        all_meet_preserving=all(r.is_meet_preserving for r in results),
        all_join_preserving=all(r.is_join_preserving for r in results),
        all_endomorphisms=all(r.is_endomorphism for r in results),
        num_labels=len(results),
    )
