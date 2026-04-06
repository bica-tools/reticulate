"""Tests for FIX 4.4 / ISO 20022 financial protocol verification (Step 73)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.fix_iso20022 import (
    ALL_FINANCIAL_PROTOCOLS,
    ComplianceAction,
    FinancialAnalysisResult,
    FinancialProtocol,
    camt054_protocol,
    execution_report_protocol,
    financial_to_session_type,
    format_financial_report,
    format_financial_summary,
    is_phi_psi_galois,
    new_order_single_protocol,
    order_cancel_protocol,
    pacs008_protocol,
    pain001_protocol,
    pain008_protocol,
    phi,
    phi_state_to_action,
    psi,
    verify_all_financial_protocols,
    verify_financial_protocol,
)


# ---------------------------------------------------------------------------
# Protocol constructors
# ---------------------------------------------------------------------------

FIX_PROTOCOLS = [
    new_order_single_protocol,
    execution_report_protocol,
    order_cancel_protocol,
]

ISO_PROTOCOLS = [
    pain001_protocol,
    pacs008_protocol,
    camt054_protocol,
    pain008_protocol,
]

ALL_CTORS = FIX_PROTOCOLS + ISO_PROTOCOLS


class TestProtocolConstruction:
    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_is_protocol(self, ctor):
        p = ctor()
        assert isinstance(p, FinancialProtocol)
        assert p.name
        assert p.family in {"FIX", "ISO20022"}
        assert p.session_type_string
        assert p.description
        assert p.messages
        assert p.transitions

    @pytest.mark.parametrize("ctor", FIX_PROTOCOLS)
    def test_fix_family(self, ctor):
        assert ctor().family == "FIX"

    @pytest.mark.parametrize("ctor", ISO_PROTOCOLS)
    def test_iso_family(self, ctor):
        assert ctor().family == "ISO20022"

    def test_registry_length(self):
        assert len(ALL_FINANCIAL_PROTOCOLS) == 7

    def test_registry_has_all_protocols(self):
        names = {p.name for p in ALL_FINANCIAL_PROTOCOLS}
        assert "NewOrderSingle" in names
        assert "pain001" in names
        assert "pacs008" in names
        assert "camt054" in names


# ---------------------------------------------------------------------------
# Parsing + state space
# ---------------------------------------------------------------------------

class TestParsing:
    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_parses(self, ctor):
        ast = financial_to_session_type(ctor())
        assert ast is not None

    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_state_space_nonempty(self, ctor):
        ss = build_statespace(financial_to_session_type(ctor()))
        assert len(ss.states) >= 2
        assert len(ss.transitions) >= 1


# ---------------------------------------------------------------------------
# Lattice properties
# ---------------------------------------------------------------------------

class TestLatticeProperties:
    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_is_lattice(self, ctor):
        ss = build_statespace(financial_to_session_type(ctor()))
        lr = check_lattice(ss)
        assert lr.is_lattice, (
            f"{ctor.__name__} does not form a lattice: "
            f"{lr.counterexample}"
        )

    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_has_top_and_bottom(self, ctor):
        ss = build_statespace(financial_to_session_type(ctor()))
        lr = check_lattice(ss)
        assert lr.has_top
        assert lr.has_bottom


# ---------------------------------------------------------------------------
# Bidirectional morphism phi, psi
# ---------------------------------------------------------------------------

class TestMorphisms:
    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_phi_returns_action_per_state(self, ctor):
        ss = build_statespace(financial_to_session_type(ctor()))
        actions = phi(ss)
        assert len(actions) == len(ss.states)
        for a in actions:
            assert isinstance(a, ComplianceAction)
            assert a.action in {
                "audit", "monitor", "block", "settle",
                "reconcile", "archive",
            }

    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_top_is_audit(self, ctor):
        ss = build_statespace(financial_to_session_type(ctor()))
        a = phi_state_to_action(ss, ss.top)
        assert a.action == "audit"

    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_psi_returns_preimage(self, ctor):
        ss = build_statespace(financial_to_session_type(ctor()))
        # Every state must be in psi(phi(state))
        for s in ss.states:
            a = phi_state_to_action(ss, s).action
            assert s in psi(ss, a)

    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_galois_connection(self, ctor):
        ss = build_statespace(financial_to_session_type(ctor()))
        assert is_phi_psi_galois(ss)

    def test_new_order_single_has_terminal_classification(self):
        """FIX NewOrderSingle bottom classifies under conservative priority."""
        ss = build_statespace(
            financial_to_session_type(new_order_single_protocol())
        )
        # bottom has FILL, REJECT, CANCELLED, CANCELREJECT incoming;
        # priority is reject > settle > cancel, so bottom is block.
        assert phi_state_to_action(ss, ss.bottom).action == "block"

    def test_new_order_single_has_block(self):
        ss = build_statespace(
            financial_to_session_type(new_order_single_protocol())
        )
        block_states = psi(ss, "block")
        assert len(block_states) >= 1

    def test_pacs008_terminal_block(self):
        ss = build_statespace(
            financial_to_session_type(pacs008_protocol())
        )
        # bottom has SETTLED, RETURNED, REJECTCLEAR incoming;
        # priority resolves to block.
        assert phi_state_to_action(ss, ss.bottom).action == "block"

    def test_pain001_block_on_rejected(self):
        ss = build_statespace(
            financial_to_session_type(pain001_protocol())
        )
        assert len(psi(ss, "block")) >= 1

    def test_camt054_reconcile_or_block_on_unmatched(self):
        ss = build_statespace(
            financial_to_session_type(camt054_protocol())
        )
        # UNMATCHED is classified as a reject label
        assert len(psi(ss, "block")) >= 1


# ---------------------------------------------------------------------------
# Full verification pipeline
# ---------------------------------------------------------------------------

class TestVerifyPipeline:
    @pytest.mark.parametrize("ctor", ALL_CTORS)
    def test_verify_returns_result(self, ctor):
        r = verify_financial_protocol(ctor())
        assert isinstance(r, FinancialAnalysisResult)
        assert r.is_well_formed
        assert r.num_states >= 2
        assert r.num_transitions >= 1
        assert r.num_valid_paths >= 1
        assert r.test_source  # JUnit source generated
        assert r.compliance_actions

    def test_verify_all(self):
        rs = verify_all_financial_protocols()
        assert len(rs) == 7
        assert all(r.is_well_formed for r in rs)

    def test_format_report(self):
        r = verify_financial_protocol(new_order_single_protocol())
        out = format_financial_report(r)
        assert "NewOrderSingle" in out
        assert "VERDICT" in out
        assert "WELL-FORMED" in out
        assert "phi" in out

    def test_format_summary(self):
        rs = verify_all_financial_protocols()
        out = format_financial_summary(rs)
        assert "NewOrderSingle" in out
        assert "pain001" in out
        assert "7/7" in out

    def test_distributivity_recorded(self):
        r = verify_financial_protocol(pacs008_protocol())
        # pacs008 is a simple tree, should be distributive
        assert r.distributivity.is_distributive
