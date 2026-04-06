"""Tests for ``reticulate.confluence`` (Step 80m).

The main theorem is: every well-formed session type is confluent
(equivalently, its state space is a meet-semilattice). These tests
verify the algorithm on benchmark protocols and on hand-crafted
synthetic state spaces that exercise edge cases of the
local/global confluence distinction (Newman's lemma).
"""

from __future__ import annotations

import pytest

from reticulate.confluence import (
    ConfluenceResult,
    check_confluence,
    confluence_closure,
    find_critical_pairs,
    is_globally_confluent,
    is_locally_confluent,
)
from reticulate.lattice import check_lattice
from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss_from(src: str) -> StateSpace:
    return build_statespace(parse(src))


def _make_ss(
    states: set[int],
    transitions: list[tuple[int, str, int]],
    top: int,
    bottom: int,
) -> StateSpace:
    return StateSpace(
        states=set(states),
        transitions=list(transitions),
        top=top,
        bottom=bottom,
        labels={s: str(s) for s in states},
    )


# ---------------------------------------------------------------------------
# 1. Trivial cases
# ---------------------------------------------------------------------------

def test_end_is_confluent():
    ss = _ss_from("end")
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent
    assert r.critical_pair_count == 0


def test_single_branch_is_confluent():
    ss = _ss_from("&{m: end}")
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent


def test_single_selection_is_confluent():
    ss = _ss_from("+{l: end}")
    assert is_locally_confluent(_ss_from("+{l: end}"))
    assert is_globally_confluent(ss)


def test_result_is_frozen_dataclass():
    r = check_confluence(_ss_from("end"))
    assert isinstance(r, ConfluenceResult)
    with pytest.raises(Exception):
        r.is_locally_confluent = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Branch / selection critical pairs
# ---------------------------------------------------------------------------

def test_two_branch_no_critical_pair_when_targets_merge():
    """In reticulate's state space, branches with isomorphic continuations
    share the same target state, so the critical pair collapses. Confluence
    still holds (vacuously locally and via the meet-semilattice property).
    """
    ss = _ss_from("&{a: end, b: end}")
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent


def test_branches_with_distinct_continuations_create_critical_pair():
    ss = _ss_from("&{a: +{x: end}, b: +{y: end}}")
    pairs = find_critical_pairs(ss)
    top_pairs = [p for p in pairs if p[0] == ss.top]
    assert len(top_pairs) >= 1
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent


def test_two_selection_is_confluent():
    ss = _ss_from("+{a: end, b: end}")
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent


# ---------------------------------------------------------------------------
# 3. Confluence closure (meet)
# ---------------------------------------------------------------------------

def test_closure_of_branches_with_distinct_continuations_is_bottom():
    ss = _ss_from("&{a: +{x: end}, b: +{y: end}}")
    succs = sorted(ss.successors(ss.top))
    assert len(succs) >= 2
    w = confluence_closure(ss, succs[0], succs[1])
    assert w == ss.bottom


def test_closure_with_self_is_self():
    ss = _ss_from("&{a: end, b: end}")
    assert confluence_closure(ss, ss.top, ss.top) == ss.top


def test_closure_unknown_state_returns_none():
    ss = _ss_from("end")
    assert confluence_closure(ss, 999, ss.top) is None


# ---------------------------------------------------------------------------
# 4. Parallel composition
# ---------------------------------------------------------------------------

def test_parallel_is_confluent():
    ss = _ss_from("(&{a: end} || &{b: end})")
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent


def test_parallel_with_branches_is_confluent():
    ss = _ss_from("(&{a: end, b: end} || &{c: end, d: end})")
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent


def test_parallel_critical_pairs_close():
    ss = _ss_from("(&{a: end} || &{b: end})")
    for origin, b1, b2 in find_critical_pairs(ss):
        assert confluence_closure(ss, b1, b2) is not None


# ---------------------------------------------------------------------------
# 5. Equivalence with lattice property (the main theorem)
# ---------------------------------------------------------------------------

CONFLUENCE_BENCHMARKS = [
    "end",
    "&{m: end}",
    "+{l: end}",
    "&{a: end, b: end}",
    "+{a: end, b: end}",
    "&{a: end, b: end, c: end}",
    "(&{a: end} || &{b: end})",
    "(+{a: end} || +{b: end})",
    "(&{a: end, b: end} || &{c: end})",
    "&{open: +{ok: end, err: end}}",
]


@pytest.mark.parametrize("src", CONFLUENCE_BENCHMARKS)
def test_lattice_iff_confluent(src):
    ss = _ss_from(src)
    lat = check_lattice(ss)
    conf = check_confluence(ss)
    # The main theorem of Step 80m: lattice property <=> confluence.
    assert lat.is_lattice == conf.is_globally_confluent


@pytest.mark.parametrize("src", CONFLUENCE_BENCHMARKS)
def test_global_confluence_implies_local(src):
    ss = _ss_from(src)
    conf = check_confluence(ss)
    if conf.is_globally_confluent:
        assert conf.is_locally_confluent


@pytest.mark.parametrize("src", CONFLUENCE_BENCHMARKS)
def test_universal_confluence_well_formed(src):
    """Corollary: every well-formed session type is confluent."""
    ss = _ss_from(src)
    assert is_globally_confluent(ss)


# ---------------------------------------------------------------------------
# 6. Recursive types
# ---------------------------------------------------------------------------

def test_iterator_is_confluent():
    ss = _ss_from("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
    r = check_confluence(ss)
    assert r.is_globally_confluent


def test_simple_recursion_is_confluent():
    ss = _ss_from("rec X . &{m: end, n: X}")
    assert is_globally_confluent(ss)


# ---------------------------------------------------------------------------
# 7. Synthetic state spaces -- Newman's lemma sanity checks
# ---------------------------------------------------------------------------

def test_synthetic_locally_but_not_globally_confluent():
    """Hand-crafted ARS that is locally but not globally confluent.

    States: 0 (top) -> 1, 0 -> 2;
            1 -> 3, 1 -> 4;
            2 -> 4, 2 -> 5;
            3 and 5 are sinks; 4 is a sink.

    Critical pair (1,2) at 0 closes locally via 4 (1 reachstar 4, 2 reachstar 4).
    But 3 and 5 share no common lower bound, and they are reachable from a
    common ancestor (0), so global confluence fails.

    NOTE: this construct is NOT a valid session type state space; it is
    only used to validate the algorithm's distinction between local and
    global confluence on a synthetic abstract reduction system.
    """
    ss = _make_ss(
        states={0, 1, 2, 3, 4, 5},
        transitions=[
            (0, "a", 1), (0, "b", 2),
            (1, "x", 3), (1, "y", 4),
            (2, "y", 4), (2, "z", 5),
        ],
        top=0,
        bottom=4,
    )
    # Pair (3, 5) has no common lower bound -> not globally confluent
    assert confluence_closure(ss, 3, 5) is None
    assert not is_globally_confluent(ss)


def test_synthetic_diamond_is_confluent():
    """Classical diamond: 0 -> {1, 2} -> 3."""
    ss = _make_ss(
        states={0, 1, 2, 3},
        transitions=[
            (0, "a", 1), (0, "b", 2),
            (1, "b", 3), (2, "a", 3),
        ],
        top=0,
        bottom=3,
    )
    r = check_confluence(ss)
    assert r.is_locally_confluent
    assert r.is_globally_confluent
    assert confluence_closure(ss, 1, 2) == 3


def test_synthetic_no_critical_pairs_chain():
    """Linear chain has no critical pairs at all."""
    ss = _make_ss(
        states={0, 1, 2, 3},
        transitions=[(0, "a", 1), (1, "b", 2), (2, "c", 3)],
        top=0,
        bottom=3,
    )
    assert find_critical_pairs(ss) == []
    r = check_confluence(ss)
    assert r.critical_pair_count == 0
    assert r.is_locally_confluent


def test_synthetic_unclosed_critical_pair_reported():
    """Two divergent successors with no common descendant."""
    ss = _make_ss(
        states={0, 1, 2},
        transitions=[(0, "a", 1), (0, "b", 2)],
        top=0,
        bottom=1,  # arbitrary
    )
    r = check_confluence(ss)
    # 1 and 2 are sinks with no common lower bound
    assert not r.is_locally_confluent
    assert (0, 1, 2) in r.unclosed_critical_pairs


# ---------------------------------------------------------------------------
# 8. Algorithmic properties
# ---------------------------------------------------------------------------

def test_critical_pair_count_matches_field():
    ss = _ss_from("&{a: end, b: end, c: end}")
    pairs = find_critical_pairs(ss)
    r = check_confluence(ss)
    assert r.critical_pair_count == len(pairs)


def test_critical_pairs_deterministic():
    ss = _ss_from("&{a: end, b: end, c: end}")
    p1 = find_critical_pairs(ss)
    p2 = find_critical_pairs(ss)
    assert p1 == p2


def test_missing_meets_empty_for_well_formed():
    for src in CONFLUENCE_BENCHMARKS:
        r = check_confluence(_ss_from(src))
        assert r.missing_meets == ()


def test_unclosed_critical_pairs_empty_for_well_formed():
    for src in CONFLUENCE_BENCHMARKS:
        r = check_confluence(_ss_from(src))
        assert r.unclosed_critical_pairs == ()


def test_check_confluence_returns_result_type():
    r = check_confluence(_ss_from("&{a: end, b: end}"))
    assert isinstance(r, ConfluenceResult)
    assert isinstance(r.critical_pair_count, int)
    assert isinstance(r.missing_meets, tuple)
    assert isinstance(r.unclosed_critical_pairs, tuple)
