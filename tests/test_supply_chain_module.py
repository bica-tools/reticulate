"""Tests for reticulate.supply_chain (Step 67 module).

The earlier test_supply_chain.py covers three hand-crafted protocols
(EDI/parallel/recursive).  This file covers the five named scenarios
exposed by the reticulate.supply_chain module and the bidirectional
morphism pair (phi, psi).
"""

from __future__ import annotations

import pytest

from reticulate.supply_chain import (
    ALL_SUPPLY_CHAIN_SCENARIOS,
    SUPPLY_CHAIN_PHASES,
    SUPPLY_CHAIN_ROLES,
    analyse_all,
    analyse_scenario,
    classify_morphism_pair,
    format_analysis,
    format_summary,
    order_fulfilment_scenario,
    payment_scenario,
    phase_histogram,
    phase_of_edge,
    phi_edge,
    psi_phase,
    quality_scenario,
    replenishment_scenario,
    return_scenario,
    round_trip_coverage,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_five_canonical_roles():
    assert SUPPLY_CHAIN_ROLES == (
        "Supplier", "Manufacturer", "Distributor", "Retailer", "Customer",
    )


def test_six_phases():
    assert len(SUPPLY_CHAIN_PHASES) == 6
    assert "Sourcing" in SUPPLY_CHAIN_PHASES
    assert "Aftermarket" in SUPPLY_CHAIN_PHASES


def test_all_scenarios_defined():
    assert len(ALL_SUPPLY_CHAIN_SCENARIOS) == 5
    names = {s.name for s in ALL_SUPPLY_CHAIN_SCENARIOS}
    assert names == {
        "OrderFulfilment", "Return", "Payment",
        "QualityInspection", "Replenishment",
    }


# ---------------------------------------------------------------------------
# Scenario shape
# ---------------------------------------------------------------------------


def test_order_fulfilment_has_five_roles():
    s = order_fulfilment_scenario()
    assert s.expected_roles == frozenset(SUPPLY_CHAIN_ROLES)
    assert "placeOrder" in s.global_type_string


def test_return_scenario_has_returns_desk():
    s = return_scenario()
    assert "ReturnsDesk" in s.expected_roles


def test_payment_scenario_has_bank():
    s = payment_scenario()
    assert "Bank" in s.expected_roles
    assert "PaymentGW" in s.expected_roles


def test_quality_scenario_has_inspector():
    s = quality_scenario()
    assert "QualityInspector" in s.expected_roles


def test_replenishment_is_recursive():
    s = replenishment_scenario()
    assert "rec" in s.global_type_string


def test_compliance_notes_present():
    for s in ALL_SUPPLY_CHAIN_SCENARIOS:
        assert len(s.compliance_notes) >= 2


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", list(ALL_SUPPLY_CHAIN_SCENARIOS))
def test_analyse_every_scenario(scenario):
    a = analyse_scenario(scenario)
    assert a.num_states > 0
    assert a.num_transitions > 0
    assert a.num_roles >= 2


def test_order_fulfilment_is_lattice():
    a = analyse_scenario(order_fulfilment_scenario())
    assert a.is_well_formed
    assert a.num_roles == 5


def test_payment_scenario_transitions():
    a = analyse_scenario(payment_scenario())
    assert a.num_transitions >= 6


def test_quality_scenario_has_accept_and_reject():
    a = analyse_scenario(quality_scenario())
    labels = {lbl for _, lbl, _ in a.state_space.transitions}
    assert any("reject" in l for l in labels)
    assert any("accept" in l for l in labels)


def test_replenishment_has_finite_state_space():
    a = analyse_scenario(replenishment_scenario())
    assert a.num_states > 0
    assert a.num_transitions > 0


def test_order_fulfilment_projections_cover_all_roles():
    a = analyse_scenario(order_fulfilment_scenario())
    for role in SUPPLY_CHAIN_ROLES:
        assert role in a.projections


def test_analyse_all_returns_five():
    results = analyse_all()
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Bidirectional morphisms phi, psi
# ---------------------------------------------------------------------------


def test_phase_of_edge_known():
    assert phase_of_edge("Supplier", "Manufacturer") == "Sourcing"
    assert phase_of_edge("Retailer", "Customer") == "Retail"
    assert phase_of_edge("Customer", "ReturnsDesk") == "Aftermarket"


def test_phi_edge_parses_label():
    assert phi_edge("Supplier->Manufacturer:deliverMaterial") == "Sourcing"
    assert phi_edge("Retailer->Customer:deliverOrder") == "Retail"


def test_phi_edge_unknown_falls_back_to_valid_phase():
    phase = phi_edge("Foo->Bar:baz")
    assert phase in SUPPLY_CHAIN_PHASES


def test_psi_phase_inverts_phi():
    retail_edges = psi_phase("Retail")
    assert ("Retailer", "Customer") in retail_edges
    sourcing_edges = psi_phase("Sourcing")
    assert ("Supplier", "Manufacturer") in sourcing_edges


def test_round_trip_coverage_is_one_for_order_fulfilment():
    a = analyse_scenario(order_fulfilment_scenario())
    cov = round_trip_coverage(a.state_space)
    assert cov == 1.0


@pytest.mark.parametrize("scenario", list(ALL_SUPPLY_CHAIN_SCENARIOS))
def test_round_trip_coverage_is_one_for_all_scenarios(scenario):
    a = analyse_scenario(scenario)
    cov = round_trip_coverage(a.state_space)
    assert cov == 1.0


def test_phase_histogram_positive():
    a = analyse_scenario(order_fulfilment_scenario())
    total = sum(a.phase_histogram.values())
    assert total > 0


def test_classify_morphism_pair():
    assert classify_morphism_pair() == "galois-insertion"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_format_analysis_contains_name_and_lattice():
    a = analyse_scenario(order_fulfilment_scenario())
    text = format_analysis(a)
    assert "OrderFulfilment" in text
    assert "Lattice" in text


def test_format_summary_lists_all_scenarios():
    results = analyse_all()
    text = format_summary(results)
    for s in ALL_SUPPLY_CHAIN_SCENARIOS:
        assert s.name in text
