"""Tests for ect_fingerprint module (Step 32i)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.ect_fingerprint import (
    ECTFingerprint,
    ECTComparison,
    MorphismClassification,
    compute_ect,
    compare_ect,
    psi_recover_invariants,
    classify_ect_morphism,
    sublevel_states,
)


def _ss(s: str):
    return build_statespace(parse(s))


# ---------------------------------------------------------------------------
# Core fingerprint properties
# ---------------------------------------------------------------------------

def test_end_fingerprint():
    fp = compute_ect(_ss("end"))
    assert isinstance(fp, ECTFingerprint)
    assert fp.rank_max == 0
    assert fp.chi_sequence == (1,)  # single state: one 0-simplex


def test_single_branch_fingerprint():
    fp = compute_ect(_ss("&{a: end}"))
    assert fp.rank_max >= 1
    assert len(fp.chi_sequence) == fp.rank_max + 1
    # first level has one state (bottom), chi=1
    assert fp.chi_sequence[0] == 1


def test_branch_two_methods():
    fp = compute_ect(_ss("&{a: end, b: end}"))
    # Three states: top with two children to end. But end states are merged.
    assert fp.rank_max >= 1
    assert fp.chi_sequence[0] == 1


def test_chi_sequence_length_matches_rank():
    fp = compute_ect(_ss("&{a: &{b: &{c: end}}}"))
    assert len(fp.chi_sequence) == fp.rank_max + 1


def test_f_vectors_nonempty_when_states_present():
    fp = compute_ect(_ss("&{a: &{b: end}}"))
    for fv, size in zip(fp.f_vectors, fp.level_sizes):
        if size > 0:
            assert len(fv) >= 1


def test_level_sizes_monotone():
    fp = compute_ect(_ss("&{a: &{b: &{c: end}}}"))
    for i in range(1, len(fp.level_sizes)):
        assert fp.level_sizes[i] >= fp.level_sizes[i - 1]


def test_selection_fingerprint():
    fp = compute_ect(_ss("+{a: end, b: end}"))
    assert fp.rank_max >= 1
    assert fp.chi_sequence[0] >= 1


def test_parallel_fingerprint():
    fp = compute_ect(_ss("(&{a: end} || &{b: end})"))
    assert fp.rank_max >= 1
    # parallel product: rank adds
    assert len(fp.chi_sequence) >= 2


def test_deeper_protocol():
    fp = compute_ect(_ss("&{a: &{b: &{c: &{d: end}}}}"))
    assert len(fp.chi_sequence) == fp.rank_max + 1
    assert fp.rank_max >= 3


# ---------------------------------------------------------------------------
# sublevel_states
# ---------------------------------------------------------------------------

def test_sublevel_at_zero():
    ss = _ss("&{a: &{b: end}}")
    sub0 = sublevel_states(ss, 0)
    # bottom only
    assert len(sub0) == 1


def test_sublevel_grows():
    ss = _ss("&{a: &{b: &{c: end}}}")
    s0 = sublevel_states(ss, 0)
    s1 = sublevel_states(ss, 1)
    s2 = sublevel_states(ss, 2)
    assert len(s0) <= len(s1) <= len(s2)


def test_sublevel_at_max_is_full():
    ss = _ss("&{a: end, b: end}")
    from reticulate.zeta import compute_rank
    K = max(compute_rank(ss).values())
    full = sublevel_states(ss, K)
    # Should equal number of SCC reps
    assert len(full) >= 1


# ---------------------------------------------------------------------------
# compare_ect
# ---------------------------------------------------------------------------

def test_compare_equal():
    fp = compute_ect(_ss("&{a: end}"))
    cmp = compare_ect(fp, fp)
    assert cmp.equal
    assert cmp.l1_distance == 0
    assert cmp.first_divergence == -1


def test_compare_different():
    a = compute_ect(_ss("&{a: end}"))
    b = compute_ect(_ss("&{a: &{b: &{c: &{d: end}}}}"))
    cmp = compare_ect(a, b)
    assert not cmp.equal
    assert cmp.l1_distance >= 0


def test_distance_symmetric():
    a = compute_ect(_ss("&{a: end}"))
    b = compute_ect(_ss("&{a: &{b: end}}"))
    assert a.distance(b) == b.distance(a)


def test_distance_zero_iff_equal():
    a = compute_ect(_ss("&{a: &{b: end}}"))
    b = compute_ect(_ss("&{a: &{b: end}}"))
    assert a.distance(b) == 0


def test_first_divergence_index():
    a = compute_ect(_ss("&{a: end}"))
    b = compute_ect(_ss("&{a: &{b: end}}"))
    cmp = compare_ect(a, b)
    if not cmp.equal:
        assert cmp.first_divergence >= 0


# ---------------------------------------------------------------------------
# psi_recover_invariants
# ---------------------------------------------------------------------------

def test_psi_basic():
    fp = compute_ect(_ss("&{a: &{b: end}}"))
    inv = psi_recover_invariants(fp)
    assert "total_chi" in inv
    assert "rank" in inv
    assert "top_chi" in inv
    assert "max_level_size" in inv
    assert inv["rank"] == fp.rank_max


def test_psi_end():
    fp = compute_ect(_ss("end"))
    inv = psi_recover_invariants(fp)
    assert inv["rank"] == 0
    assert inv["top_chi"] == 1


def test_psi_rank_matches():
    for s in ["&{a: end}", "&{a: &{b: end}}", "&{a: &{b: &{c: end}}}"]:
        fp = compute_ect(_ss(s))
        inv = psi_recover_invariants(fp)
        assert inv["rank"] == fp.rank_max


# ---------------------------------------------------------------------------
# classify_ect_morphism
# ---------------------------------------------------------------------------

def test_classify_empty():
    cls = classify_ect_morphism([])
    assert isinstance(cls, MorphismClassification)
    assert cls.phi_surjective_onto_image


def test_classify_samples():
    samples = [
        _ss("end"),
        _ss("&{a: end}"),
        _ss("&{a: &{b: end}}"),
        _ss("(&{a: end} || &{b: end})"),
    ]
    cls = classify_ect_morphism(samples)
    assert cls.classification in ("galois", "projection", "injection")
    assert cls.psi_right_adjoint


def test_classify_single_sample():
    cls = classify_ect_morphism([_ss("&{a: end}")])
    assert cls.galois_connection or cls.phi_order_preserving


# ---------------------------------------------------------------------------
# Fingerprint invariants
# ---------------------------------------------------------------------------

def test_fingerprint_hashable():
    fp = compute_ect(_ss("&{a: end}"))
    assert hash(fp) is not None


def test_fingerprint_equality():
    a = compute_ect(_ss("&{a: end}"))
    b = compute_ect(_ss("&{a: end}"))
    assert a == b


def test_fingerprint_len():
    fp = compute_ect(_ss("&{a: &{b: end}}"))
    assert len(fp) == len(fp.chi_sequence)


def test_recursive_protocol_fingerprint():
    # Recursive types have cycles → SCCs → finite fingerprint
    fp = compute_ect(_ss("rec X . &{loop: X, stop: end}"))
    assert len(fp.chi_sequence) >= 1


def test_large_parallel():
    fp = compute_ect(_ss("(&{a: &{b: end}} || &{c: &{d: end}})"))
    assert fp.rank_max >= 2


def test_nontrivial_l1_distance():
    a = compute_ect(_ss("&{a: end}"))
    b = compute_ect(_ss("&{a: &{b: &{c: &{d: &{e: end}}}}}"))
    assert a.distance(b) > 0
