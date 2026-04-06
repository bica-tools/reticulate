"""Tests for reticulate.defi (Step 74)."""

from __future__ import annotations

import pytest

from reticulate.defi import (
    DeFiAnalysis,
    DeFiScenario,
    EXPLOIT_PATTERNS,
    PATTERN_LOSSES_USD,
    SCENARIOS,
    aave_lend_borrow,
    analyse_all,
    analyse_scenario,
    classify_morphism_pair,
    detect_flash_loan_window,
    detect_oracle_manipulation,
    detect_reentrancy,
    detect_sandwich,
    flash_loan,
    format_analysis,
    format_summary,
    pattern_histogram,
    phi_label,
    phi_state,
    price_oracle_consumer,
    psi_pattern,
    roundtrip_coverage,
    uniswap_swap,
    vulnerable_withdraw,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------


def test_all_scenarios_construct():
    for factory in SCENARIOS:
        sc = factory()
        assert isinstance(sc, DeFiScenario)
        assert sc.name
        assert sc.description
        assert sc.session_type_string
        assert sc.phi_rules
        assert sc.audit_actions


def test_scenario_names_unique():
    names = [f().name for f in SCENARIOS]
    assert len(names) == len(set(names))


def test_uniswap_swap_parses():
    parse(uniswap_swap().session_type_string)


def test_aave_parses():
    parse(aave_lend_borrow().session_type_string)


def test_flash_loan_parses():
    parse(flash_loan().session_type_string)


def test_vulnerable_withdraw_parses():
    parse(vulnerable_withdraw().session_type_string)


def test_price_oracle_parses():
    parse(price_oracle_consumer().session_type_string)


# ---------------------------------------------------------------------------
# Lattice and analysis
# ---------------------------------------------------------------------------


def test_analyse_scenario_returns_analysis():
    a = analyse_scenario(uniswap_swap())
    assert isinstance(a, DeFiAnalysis)
    assert a.state_space.states
    assert a.lattice is not None


def test_all_scenarios_are_lattices():
    for a in analyse_all():
        assert a.is_well_formed, f"{a.scenario.name} failed lattice check"


def test_analyse_all_returns_five():
    assert len(analyse_all()) == 5


def test_state_space_nonempty():
    for a in analyse_all():
        assert len(a.state_space.states) >= 2
        assert len(a.state_space.transitions) >= 1


# ---------------------------------------------------------------------------
# phi / psi morphism pair
# ---------------------------------------------------------------------------


def test_phi_label_basic():
    rules = (("external_call", "Reentrancy"), ("settle", "Safe"))
    assert phi_label("external_call", rules) == "Reentrancy"
    assert phi_label("settle", rules) == "Safe"
    assert phi_label("unknown", rules) == "Safe"  # default


def test_phi_label_prefix_matching():
    rules = (("external_call", "Reentrancy"),)
    assert phi_label("external_call2", rules) == "Reentrancy"


def test_phi_state_worst_severity():
    sc = vulnerable_withdraw()
    a = analyse_scenario(sc)
    # At least one state must be classified Reentrancy
    patterns = {phi_state(a.state_space, s, sc.phi_rules)
                for s in a.state_space.states}
    assert "Reentrancy" in patterns


def test_psi_is_inverse_image_of_phi():
    sc = uniswap_swap()
    a = analyse_scenario(sc)
    for pattern in EXPLOIT_PATTERNS:
        pre = psi_pattern(a.state_space, pattern, sc.phi_rules)
        for s in pre:
            assert phi_state(a.state_space, s, sc.phi_rules) == pattern


def test_roundtrip_coverage_is_one():
    # psi(phi(s)) always contains s by definition of preimage
    for a in analyse_all():
        assert a.roundtrip_coverage == 1.0


def test_pattern_histogram_sums_to_state_count():
    for a in analyse_all():
        total = sum(a.pattern_histogram.values())
        assert total == len(a.state_space.states)


def test_histogram_contains_only_known_patterns():
    for a in analyse_all():
        for p in a.pattern_histogram:
            assert p in EXPLOIT_PATTERNS


# ---------------------------------------------------------------------------
# Exploit detection
# ---------------------------------------------------------------------------


def test_vulnerable_withdraw_detects_reentrancy():
    sc = vulnerable_withdraw()
    a = analyse_scenario(sc)
    assert detect_reentrancy(a.state_space, sc.phi_rules)


def test_uniswap_detects_sandwich():
    sc = uniswap_swap()
    a = analyse_scenario(sc)
    assert detect_sandwich(a.state_space, sc.phi_rules)


def test_aave_detects_oracle_manipulation():
    sc = aave_lend_borrow()
    a = analyse_scenario(sc)
    assert detect_oracle_manipulation(a.state_space, sc.phi_rules)


def test_flash_loan_detects_flash_window():
    sc = flash_loan()
    a = analyse_scenario(sc)
    assert detect_flash_loan_window(a.state_space, sc.phi_rules)


def test_safe_scenarios_have_no_reentrancy():
    sc = uniswap_swap()
    a = analyse_scenario(sc)
    assert detect_reentrancy(a.state_space, sc.phi_rules) == []


def test_price_oracle_detects_multiple_patterns():
    sc = price_oracle_consumer()
    a = analyse_scenario(sc)
    hist = a.pattern_histogram
    # price oracle consumer should trigger at least two distinct
    # exploit classes besides Safe
    nonsafe = sum(1 for k, v in hist.items() if k != "Safe" and v > 0)
    assert nonsafe >= 2


# ---------------------------------------------------------------------------
# Loss exposure economics
# ---------------------------------------------------------------------------


def test_loss_exposure_positive_for_vulnerable():
    a = analyse_scenario(vulnerable_withdraw())
    assert a.total_loss_exposure_usd > 0


def test_pattern_losses_defined_for_all_patterns():
    for p in EXPLOIT_PATTERNS:
        assert p in PATTERN_LOSSES_USD
        assert PATTERN_LOSSES_USD[p] >= 0


def test_total_exposure_at_least_one_billion():
    # The combined modelled exposure should exceed USD 1 billion.
    analyses = analyse_all()
    total = sum(a.total_loss_exposure_usd for a in analyses)
    assert total >= 1_000_000_000


# ---------------------------------------------------------------------------
# Morphism classification
# ---------------------------------------------------------------------------


def test_classify_morphism_pair_mentions_galois():
    s = classify_morphism_pair()
    assert "Galois" in s or "galois" in s.lower()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_format_analysis_contains_fields():
    a = analyse_scenario(uniswap_swap())
    report = format_analysis(a)
    assert "UniswapSwap" in report
    assert "Lattice" in report
    assert "Pattern histogram" in report


def test_format_summary_lists_all_scenarios():
    analyses = analyse_all()
    summary = format_summary(analyses)
    for a in analyses:
        assert a.scenario.name in summary
    assert "Total modelled exposure" in summary


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_analysis_deterministic():
    a1 = analyse_scenario(uniswap_swap())
    a2 = analyse_scenario(uniswap_swap())
    assert a1.pattern_histogram == a2.pattern_histogram
    assert a1.roundtrip_coverage == a2.roundtrip_coverage
    assert a1.total_loss_exposure_usd == a2.total_loss_exposure_usd


def test_every_scenario_has_audit_actions_for_used_patterns():
    for factory in SCENARIOS:
        sc = factory()
        a = analyse_scenario(sc)
        used = {p for p, n in a.pattern_histogram.items() if n > 0}
        for pattern in used:
            # Safe patterns always have a default entry; non-Safe
            # patterns MUST have an explicit audit action.
            if pattern != "Safe":
                assert pattern in sc.audit_actions, (
                    f"{sc.name} missing audit action for {pattern}"
                )
