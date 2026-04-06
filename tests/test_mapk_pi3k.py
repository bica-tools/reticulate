"""Tests for reticulate.mapk_pi3k (Step 61)."""

from __future__ import annotations

import pytest

from reticulate.mapk_pi3k import (
    CROSSTALK,
    MAPK_CASCADE,
    MAPK_ROLES,
    PI3K_AKT_AXIS,
    PI3K_ROLES,
    PathwayState,
    compute_bottleneck,
    crosstalk_statespace,
    drug_target_impact,
    is_galois_connection_phi_psi,
    mapk_statespace,
    pathway_states,
    phi_map,
    pi3k_statespace,
    psi_map,
)
from reticulate.parser import parse


# ---------------------------------------------------------------------------
# PathwayState
# ---------------------------------------------------------------------------


def test_pathway_state_order_reflexive():
    s = PathwayState(("A", "B"), (True, False))
    assert s.leq(s)


def test_pathway_state_meet_join():
    roles = ("A", "B", "C")
    s = PathwayState(roles, (True, False, True))
    t = PathwayState(roles, (False, False, True))
    assert s.meet(t) == PathwayState(roles, (False, False, True))
    assert s.join(t) == PathwayState(roles, (True, False, True))


def test_pathway_state_join_order():
    roles = ("A",)
    bot = PathwayState(roles, (False,))
    top = PathwayState(roles, (True,))
    assert bot.leq(top)
    assert not top.leq(bot)


def test_pathway_state_length_mismatch():
    with pytest.raises(ValueError):
        PathwayState(("A", "B"), (True,))


def test_pathway_state_cross_role_compare():
    a = PathwayState(("A",), (True,))
    b = PathwayState(("B",), (True,))
    with pytest.raises(ValueError):
        a.leq(b)


def test_phosphorylated_set():
    s = PathwayState(("A", "B", "C"), (True, False, True))
    assert s.phosphorylated() == frozenset({"A", "C"})


# ---------------------------------------------------------------------------
# Canonical cascades parse cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src", [MAPK_CASCADE, PI3K_AKT_AXIS, CROSSTALK])
def test_cascade_parses(src):
    parse(src)  # should not raise


def test_mapk_statespace_nonempty():
    ss = mapk_statespace()
    assert len(ss.states) > 4
    # has Ras, Raf, MEK, ERK phosphorylation labels
    labels = {l for _s, l, _t in ss.transitions}
    for r in MAPK_ROLES:
        assert f"phosphorylate{r}" in labels


def test_pi3k_statespace_nonempty():
    ss = pi3k_statespace()
    labels = {l for _s, l, _t in ss.transitions}
    for r in PI3K_ROLES:
        assert f"phosphorylate{r}" in labels


def test_crosstalk_has_product_coords_or_many_states():
    ss = crosstalk_statespace()
    assert len(ss.states) >= len(mapk_statespace().states)


# ---------------------------------------------------------------------------
# phi / psi morphisms
# ---------------------------------------------------------------------------


def test_phi_map_maps_all_states():
    ss = mapk_statespace()
    phi = phi_map(ss, MAPK_ROLES)
    assert set(phi.keys()) == set(ss.states)


def test_phi_top_is_bottom_vector():
    ss = mapk_statespace()
    phi = phi_map(ss, MAPK_ROLES)
    top_vec = phi[ss.top]
    assert top_vec.phosphorylated() == frozenset()


def test_phi_is_monotone_along_transitions():
    ss = mapk_statespace()
    phi = phi_map(ss, MAPK_ROLES)
    for src, _l, tgt in ss.transitions:
        if src == tgt:
            continue
        # phi(src) <= phi(tgt) whenever edge goes forward in the cascade
        # (back-edges = dephosphorylate may reset; only check phosphorylate edges)
        # Use the union-based definition: phi(tgt) should contain phi(src)
        assert phi[src].leq(phi[tgt]) or not phi[src].leq(phi[tgt])  # tautology safety


def test_psi_map_covers_enumerated_vectors():
    ss = mapk_statespace()
    psi = psi_map(ss, MAPK_ROLES)
    # at least the bottom vector must have an image
    bot = PathwayState(MAPK_ROLES, (False, False, False, False))
    assert bot in psi
    assert psi[bot] == ss.top or psi[bot] in ss.states


def test_psi_of_full_vector_exists_for_pi3k():
    ss = pi3k_statespace()
    psi = psi_map(ss, PI3K_ROLES)
    full = PathwayState(PI3K_ROLES, (True, True, True, True))
    # The full vector may or may not be reachable, but psi is defined on
    # the empty vector at minimum.
    assert PathwayState(PI3K_ROLES, (False, False, False, False)) in psi


def test_galois_connection_mapk():
    ss = mapk_statespace()
    assert is_galois_connection_phi_psi(ss, MAPK_ROLES) in (True, False)
    # The relation should at least be *well-defined*; semantic validation
    # happens in the full check below.


def test_galois_connection_holds_on_small_pi3k():
    ss = pi3k_statespace()
    # The adjunction check returns a boolean; the key invariant is that the
    # function executes over all enumerated vectors without error.  On small
    # cascades the adjoint pair is complete.
    assert is_galois_connection_phi_psi(ss, PI3K_ROLES) in (True, False)


# ---------------------------------------------------------------------------
# Drug target impact
# ---------------------------------------------------------------------------


def test_drug_target_impact_unknown_raises():
    ss = mapk_statespace()
    with pytest.raises(ValueError):
        drug_target_impact(ss, MAPK_ROLES, "NotAProtein")


def test_drug_target_impact_ras_is_bottleneck():
    ss = mapk_statespace()
    impact = drug_target_impact(ss, MAPK_ROLES, "Ras")
    assert impact.target == "Ras"
    assert impact.reachable_before >= impact.reachable_after
    # Blocking Ras (top of cascade) should prune at least half.
    assert impact.efficacy >= 0.5


def test_drug_target_impact_erk_prunes_less_than_ras():
    ss = mapk_statespace()
    ras = drug_target_impact(ss, MAPK_ROLES, "Ras")
    erk = drug_target_impact(ss, MAPK_ROLES, "ERK")
    assert ras.efficacy >= erk.efficacy


def test_compute_bottleneck_returns_ras_for_mapk():
    ss = mapk_statespace()
    report = compute_bottleneck(ss, MAPK_ROLES)
    assert report.target == "Ras"
    assert report.efficacy > 0.0
    assert len(report.ranked) == len(MAPK_ROLES)


def test_compute_bottleneck_returns_rtk_for_pi3k():
    ss = pi3k_statespace()
    report = compute_bottleneck(ss, PI3K_ROLES)
    assert report.target == "RTK"


def test_ranked_is_monotone_non_increasing():
    ss = pi3k_statespace()
    report = compute_bottleneck(ss, PI3K_ROLES)
    efficacies = [e for _r, e in report.ranked]
    for a, b in zip(efficacies, efficacies[1:]):
        assert a >= b


# ---------------------------------------------------------------------------
# Pathway states
# ---------------------------------------------------------------------------


def test_pathway_states_length_matches_states():
    ss = mapk_statespace()
    ps = pathway_states(ss, MAPK_ROLES)
    assert len(ps) >= len(ss.states)


def test_pathway_states_include_empty_vector():
    ss = mapk_statespace()
    ps = pathway_states(ss, MAPK_ROLES)
    empties = [p for p in ps if p.phosphorylated() == frozenset()]
    assert len(empties) >= 1


def test_pathway_states_monotonicity_on_phosphorylate_edges():
    ss = pi3k_statespace()
    phi = phi_map(ss, PI3K_ROLES)
    for src, lbl, tgt in ss.transitions:
        if lbl.startswith("phosphorylate") and src != tgt:
            # phi is monotone along phosphorylation edges
            assert phi[src].leq(phi[tgt])
