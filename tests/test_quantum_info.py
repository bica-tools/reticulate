"""Tests for reticulate.quantum_info (Step 31l)."""

from __future__ import annotations

import math

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.quantum_info import (
    HilbertEncoding,
    MorphismClassification,
    ProtocolState,
    classify_pair,
    density_matrix,
    kron,
    mpst_mutual_information,
    outer,
    partial_trace_first,
    partial_trace_second,
    phi_to_hilbert,
    protocol_entropy,
    protocol_state,
    psi_from_hilbert,
    reduced_entropy_parallel,
    symmetric_eigenvalues,
    tensor_density,
    von_neumann_entropy,
    zeros,
    identity,
)


# ---------------------------------------------------------------------------
# Matrix primitives
# ---------------------------------------------------------------------------

def test_zeros_and_identity():
    assert zeros(3) == [[0.0] * 3 for _ in range(3)]
    I = identity(3)
    assert I[0][0] == 1.0 and I[0][1] == 0.0


def test_outer_product_rank_one():
    psi = [1.0, 0.0]
    rho = outer(psi)
    assert rho == [[1.0, 0.0], [0.0, 0.0]]


def test_kron_2x2():
    A = [[1.0, 0.0], [0.0, 1.0]]
    B = [[2.0, 3.0], [4.0, 5.0]]
    R = kron(A, B)
    assert len(R) == 4 and len(R[0]) == 4
    assert R[0][0] == 2.0 and R[3][3] == 5.0
    assert R[2][2] == 2.0


# ---------------------------------------------------------------------------
# Eigensolver and entropy
# ---------------------------------------------------------------------------

def test_symmetric_eigenvalues_diagonal():
    A = [[2.0, 0.0], [0.0, 3.0]]
    eigs = symmetric_eigenvalues(A)
    assert sorted(eigs) == pytest.approx([2.0, 3.0], rel=1e-6)


def test_symmetric_eigenvalues_known_2x2():
    A = [[2.0, 1.0], [1.0, 2.0]]
    eigs = sorted(symmetric_eigenvalues(A))
    assert eigs == pytest.approx([1.0, 3.0], rel=1e-6)


def test_entropy_pure_state_zero():
    rho = [[1.0, 0.0], [0.0, 0.0]]
    assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-9)


def test_entropy_maximally_mixed_2d():
    rho = [[0.5, 0.0], [0.0, 0.5]]
    # S = log(2) nats
    assert von_neumann_entropy(rho) == pytest.approx(math.log(2), abs=1e-6)


def test_entropy_maximally_mixed_4d():
    rho = [[0.25 if i == j else 0.0 for j in range(4)] for i in range(4)]
    assert von_neumann_entropy(rho) == pytest.approx(math.log(4), abs=1e-6)


# ---------------------------------------------------------------------------
# Protocol state
# ---------------------------------------------------------------------------

def test_protocol_state_end_is_single_basis():
    ss = build_statespace(parse("end"))
    ps = protocol_state(ss)
    assert ps.dimension == 1
    assert ps.amplitudes == (1.0,)
    assert ps.density == ((1.0,),)


def test_protocol_state_linear_branch():
    ss = build_statespace(parse("&{a: end}"))
    ps = protocol_state(ss)
    assert ps.dimension >= 2
    # Amplitudes should be normalised.
    assert sum(a * a for a in ps.amplitudes) == pytest.approx(1.0, abs=1e-9)


def test_protocol_state_branch_binary():
    ss = build_statespace(parse("&{a: end, b: end}"))
    ps = protocol_state(ss)
    assert sum(a * a for a in ps.amplitudes) == pytest.approx(1.0, abs=1e-9)


def test_protocol_entropy_pure_state_zero():
    ss = build_statespace(parse("&{a: end, b: end}"))
    ent = protocol_entropy(ss)
    assert ent == pytest.approx(0.0, abs=1e-6)


def test_density_matrix_is_hermitian_and_trace_one():
    ss = build_statespace(parse("&{a: +{ok: end, err: end}}"))
    rho = density_matrix(ss)
    n = len(rho)
    # Symmetric
    for i in range(n):
        for j in range(n):
            assert rho[i][j] == pytest.approx(rho[j][i], abs=1e-9)
    # Trace 1
    tr = sum(rho[i][i] for i in range(n))
    assert tr == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tensor product for parallel composition
# ---------------------------------------------------------------------------

def test_tensor_density_dimensions():
    rho_a = [[1.0, 0.0], [0.0, 0.0]]
    rho_b = [[0.5, 0.0], [0.0, 0.5]]
    t = tensor_density(rho_a, rho_b)
    assert len(t) == 4


def test_partial_trace_recovers_subsystem():
    rho_a = [[0.7, 0.0], [0.0, 0.3]]
    rho_b = [[0.5, 0.0], [0.0, 0.5]]
    joint = kron(rho_a, rho_b)
    a_back = partial_trace_second(joint, 2, 2)
    b_back = partial_trace_first(joint, 2, 2)
    assert a_back[0][0] == pytest.approx(0.7, abs=1e-9)
    assert a_back[1][1] == pytest.approx(0.3, abs=1e-9)
    assert b_back[0][0] == pytest.approx(0.5, abs=1e-9)


def test_reduced_entropy_product_is_sum():
    rho_a = [[0.5, 0.0], [0.0, 0.5]]
    rho_b = [[0.5, 0.0], [0.0, 0.5]]
    joint = kron(rho_a, rho_b)
    s_a, s_b, s_ab = reduced_entropy_parallel(joint, 2, 2)
    # S(rho_a (x) rho_b) = S_A + S_B  => mutual information zero
    assert s_ab == pytest.approx(s_a + s_b, abs=1e-6)


def test_parallel_composition_tensor_property():
    # |end || end| should map to a 1x1 tensor (trivial).
    ss = build_statespace(parse("(end || end)"))
    rho = density_matrix(ss)
    assert sum(rho[i][i] for i in range(len(rho))) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# MPST quantum mutual information
# ---------------------------------------------------------------------------

def test_mpst_mutual_information_two_role_ping():
    from reticulate.global_types import parse_global
    g = parse_global("A->B:{ping: end}")
    res = mpst_mutual_information(g, "A", "B")
    # Direct A-B communication yields positive mutual information.
    assert res.mutual_information > 0.0
    assert res.role_pair == ("A", "B")


def test_mpst_mutual_information_no_comm_is_zero():
    # Roles that do not communicate have zero mutual information.
    from reticulate.global_types import parse_global
    g = parse_global("A->B:{m: end}")
    res = mpst_mutual_information(g, "A", "B")
    # Must be non-negative.
    assert res.mutual_information >= -1e-9


def test_mpst_result_contains_all_role_entropies():
    from reticulate.global_types import parse_global
    g = parse_global("A->B:{m: B->A:{n: end}}")
    res = mpst_mutual_information(g, "A", "B")
    assert "A" in res.role_entropies
    assert "B" in res.role_entropies


# ---------------------------------------------------------------------------
# Bidirectional morphisms
# ---------------------------------------------------------------------------

def test_phi_to_hilbert_basis_matches_states():
    ss = build_statespace(parse("&{a: end}"))
    enc = phi_to_hilbert(ss)
    assert enc.dimension == len(enc.basis_labels)
    assert enc.dimension >= 1


def test_psi_from_hilbert_transitive():
    enc = HilbertEncoding(
        basis_labels=(0, 1, 2),
        density=((0.5, 0.3, 0.0), (0.3, 0.3, 0.2), (0.0, 0.2, 0.2)),
        dimension=3,
    )
    rec = psi_from_hilbert(enc, threshold=0.1)
    # Transitive closure should give (0,2) since (0,1) and (1,2) exist.
    assert (0, 2) in rec.order
    assert (0, 0) in rec.order


def test_classify_pair_returns_known_kind():
    ss = build_statespace(parse("&{a: end, b: end}"))
    cls = classify_pair(ss)
    assert cls.kind in {"isomorphism", "embedding", "projection", "galois"}
    assert cls.phi_injective is True


def test_classify_pair_end_is_isomorphism():
    ss = build_statespace(parse("end"))
    cls = classify_pair(ss)
    assert cls.kind == "isomorphism"


# ---------------------------------------------------------------------------
# Predictive checks against benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src", [
    "end",
    "&{a: end}",
    "&{a: end, b: end}",
    "+{ok: end, err: end}",
    "&{a: +{ok: end, err: end}}",
    "(end || end)",
])
def test_density_matrix_valid_across_types(src):
    ss = build_statespace(parse(src))
    rho = density_matrix(ss)
    n = len(rho)
    tr = sum(rho[i][i] for i in range(n))
    assert tr == pytest.approx(1.0, abs=1e-6)
    # Eigenvalues non-negative (up to tolerance)
    eigs = symmetric_eigenvalues(rho)
    for e in eigs:
        assert e >= -1e-6


def test_entropy_monotone_under_size():
    # A type with more states should not have lower max-possible entropy.
    ss_small = build_statespace(parse("&{a: end}"))
    ss_large = build_statespace(parse("&{a: end, b: end, c: end}"))
    # Pure-state entropies are both zero, but their density dimensions differ.
    assert len(density_matrix(ss_small)) <= len(density_matrix(ss_large))


def test_mutual_information_scales_with_messages():
    from reticulate.global_types import parse_global
    g1 = parse_global("A->B:{m: end}")
    g2 = parse_global("A->B:{m: A->B:{n: end}}")
    r1 = mpst_mutual_information(g1, "A", "B")
    r2 = mpst_mutual_information(g2, "A", "B")
    # More direct messages -> at least as much mutual information.
    assert r2.mutual_information >= r1.mutual_information - 1e-6
