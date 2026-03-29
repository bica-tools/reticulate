"""Tests for fiveg_protocols.py -- Step 75.

Verifies 5G NAS and RRC protocol definitions, lattice properties,
subtyping for protocol evolution (4G to 5G), and analysis pipeline.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.fiveg_protocols import (
    ALL_5G_PROTOCOLS,
    ALL_LTE_PROTOCOLS,
    ALL_NAS_PROTOCOLS,
    ALL_RRC_PROTOCOLS,
    FiveGAnalysisResult,
    FiveGProtocol,
    ProtocolEvolutionResult,
    analyze_all_5g,
    analyze_nas_protocol,
    analyze_rrc_protocol,
    check_protocol_evolution,
    fiveg_to_session_type,
    format_5g_report,
    format_5g_summary,
    format_evolution_report,
    lte_attach,
    lte_handover,
    lte_rrc_setup,
    nas_authentication,
    nas_deregistration,
    nas_pdu_session_establishment,
    nas_registration,
    nas_service_request,
    rrc_handover,
    rrc_reconfiguration,
    rrc_release,
    rrc_setup,
    verify_5g_protocol,
)
from reticulate.statespace import build_statespace
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# NAS protocol definition tests
# ---------------------------------------------------------------------------

class TestNASProtocolDefinitions:
    """Test that each NAS protocol factory returns a valid FiveGProtocol."""

    def test_nas_registration_structure(self) -> None:
        proto = nas_registration()
        assert proto.name == "NAS_Registration"
        assert proto.layer == "NAS"
        assert "UE" in proto.roles
        assert "AMF" in proto.roles
        assert len(proto.session_type_string) > 0
        assert "TS 24.501" in proto.spec_reference

    def test_nas_deregistration_structure(self) -> None:
        proto = nas_deregistration()
        assert proto.name == "NAS_Deregistration"
        assert proto.layer == "NAS"
        assert "UE" in proto.roles and "AMF" in proto.roles

    def test_nas_service_request_structure(self) -> None:
        proto = nas_service_request()
        assert proto.name == "NAS_ServiceRequest"
        assert proto.layer == "NAS"
        assert "UE" in proto.roles

    def test_nas_pdu_session_establishment_structure(self) -> None:
        proto = nas_pdu_session_establishment()
        assert proto.name == "NAS_PDUSessionEstablishment"
        assert proto.layer == "NAS"
        assert "SMF" in proto.roles

    def test_nas_authentication_structure(self) -> None:
        proto = nas_authentication()
        assert proto.name == "NAS_Authentication"
        assert proto.layer == "NAS"
        assert "AUSF" in proto.roles
        assert "TS 33.501" in proto.spec_reference


# ---------------------------------------------------------------------------
# RRC protocol definition tests
# ---------------------------------------------------------------------------

class TestRRCProtocolDefinitions:
    """Test that each RRC protocol factory returns a valid FiveGProtocol."""

    def test_rrc_setup_structure(self) -> None:
        proto = rrc_setup()
        assert proto.name == "RRC_Setup"
        assert proto.layer == "RRC"
        assert "UE" in proto.roles and "gNB" in proto.roles
        assert "TS 38.331" in proto.spec_reference

    def test_rrc_reconfiguration_structure(self) -> None:
        proto = rrc_reconfiguration()
        assert proto.name == "RRC_Reconfiguration"
        assert proto.layer == "RRC"
        assert "UE" in proto.roles

    def test_rrc_release_structure(self) -> None:
        proto = rrc_release()
        assert proto.name == "RRC_Release"
        assert proto.layer == "RRC"

    def test_rrc_handover_structure(self) -> None:
        proto = rrc_handover()
        assert proto.name == "RRC_Handover"
        assert proto.layer == "RRC"
        assert "SourceGNB" in proto.roles
        assert "TargetGNB" in proto.roles


# ---------------------------------------------------------------------------
# Parse tests -- every protocol must parse without error
# ---------------------------------------------------------------------------

class TestParsing:
    """Test that every 5G protocol session type string parses correctly."""

    @pytest.mark.parametrize("proto", ALL_5G_PROTOCOLS, ids=lambda p: p.name)
    def test_parse_5g_protocol(self, proto: FiveGProtocol) -> None:
        ast = fiveg_to_session_type(proto)
        assert ast is not None

    @pytest.mark.parametrize("proto", ALL_LTE_PROTOCOLS, ids=lambda p: p.name)
    def test_parse_lte_protocol(self, proto: FiveGProtocol) -> None:
        ast = fiveg_to_session_type(proto)
        assert ast is not None


# ---------------------------------------------------------------------------
# State space tests
# ---------------------------------------------------------------------------

class TestStateSpace:
    """Test state space construction for all 5G protocols."""

    @pytest.mark.parametrize("proto", ALL_5G_PROTOCOLS, ids=lambda p: p.name)
    def test_state_space_nonempty(self, proto: FiveGProtocol) -> None:
        ast = fiveg_to_session_type(proto)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2  # at least top and bottom
        assert len(ss.transitions) >= 1

    def test_registration_state_count(self) -> None:
        ast = fiveg_to_session_type(nas_registration())
        ss = build_statespace(ast)
        assert len(ss.states) == 8
        assert len(ss.transitions) == 9

    def test_handover_state_count(self) -> None:
        ast = fiveg_to_session_type(rrc_handover())
        ss = build_statespace(ast)
        assert len(ss.states) == 11
        assert len(ss.transitions) == 12

    def test_deregistration_state_count(self) -> None:
        ast = fiveg_to_session_type(nas_deregistration())
        ss = build_statespace(ast)
        assert len(ss.states) == 4

    def test_rrc_release_state_count(self) -> None:
        ast = fiveg_to_session_type(rrc_release())
        ss = build_statespace(ast)
        assert len(ss.states) == 5
        assert len(ss.transitions) == 6


# ---------------------------------------------------------------------------
# Lattice property tests
# ---------------------------------------------------------------------------

class TestLatticeProperty:
    """Test that all 5G protocols form lattices."""

    @pytest.mark.parametrize("proto", ALL_5G_PROTOCOLS, ids=lambda p: p.name)
    def test_5g_protocol_is_lattice(self, proto: FiveGProtocol) -> None:
        result = verify_5g_protocol(proto)
        assert result.lattice_result.is_lattice, (
            f"{proto.name} state space is not a lattice: "
            f"{result.lattice_result.counterexample}"
        )

    @pytest.mark.parametrize("proto", ALL_LTE_PROTOCOLS, ids=lambda p: p.name)
    def test_lte_protocol_is_lattice(self, proto: FiveGProtocol) -> None:
        result = verify_5g_protocol(proto)
        assert result.lattice_result.is_lattice

    def test_simple_protocols_are_distributive(self) -> None:
        """Protocols without nested failure paths are distributive."""
        for proto in [nas_registration(), nas_deregistration(),
                      nas_service_request(), nas_pdu_session_establishment(),
                      rrc_setup(), rrc_release()]:
            result = verify_5g_protocol(proto)
            assert result.distributivity.is_distributive, proto.name

    def test_complex_protocols_may_not_be_distributive(self) -> None:
        """Protocols with nested selection (re-establishment) may have N5."""
        # NAS_Authentication, RRC_Reconfiguration, RRC_Handover have N5
        for proto in [nas_authentication(), rrc_reconfiguration(), rrc_handover()]:
            result = verify_5g_protocol(proto)
            assert result.lattice_result.is_lattice  # still a lattice
            assert result.distributivity.has_n5  # contains N5 sublattice


# ---------------------------------------------------------------------------
# Layer-specific analysis tests
# ---------------------------------------------------------------------------

class TestLayerAnalysis:
    """Test layer-specific analysis functions."""

    def test_analyze_nas_protocol(self) -> None:
        result = analyze_nas_protocol(nas_registration())
        assert result.is_well_formed
        assert result.protocol.layer == "NAS"

    def test_analyze_rrc_protocol(self) -> None:
        result = analyze_rrc_protocol(rrc_setup())
        assert result.is_well_formed
        assert result.protocol.layer == "RRC"

    def test_analyze_nas_rejects_rrc(self) -> None:
        with pytest.raises(ValueError, match="Expected NAS"):
            analyze_nas_protocol(rrc_setup())

    def test_analyze_rrc_rejects_nas(self) -> None:
        with pytest.raises(ValueError, match="Expected RRC"):
            analyze_rrc_protocol(nas_registration())

    def test_analyze_all_5g(self) -> None:
        results = analyze_all_5g()
        assert len(results) == len(ALL_5G_PROTOCOLS)
        assert all(r.is_well_formed for r in results)


# ---------------------------------------------------------------------------
# Protocol evolution / subtyping tests
# ---------------------------------------------------------------------------

class TestProtocolEvolution:
    """Test 4G to 5G protocol evolution via subtyping."""

    def test_lte_rrc_setup_subtypes_5g_rrc_setup(self) -> None:
        """LTE RRC Setup <= 5G RRC Setup (identical structure)."""
        old_ast = fiveg_to_session_type(lte_rrc_setup())
        new_ast = fiveg_to_session_type(rrc_setup())
        assert is_subtype(old_ast, new_ast)

    def test_lte_attach_not_subtype_5g_registration(self) -> None:
        """LTE Attach is NOT a subtype of 5G Registration (different structure)."""
        old_ast = fiveg_to_session_type(lte_attach())
        new_ast = fiveg_to_session_type(nas_registration())
        # 5G adds security mode command steps, so structures differ
        assert not is_subtype(old_ast, new_ast)

    def test_lte_handover_not_subtype_5g_handover(self) -> None:
        """LTE Handover is NOT a subtype of 5G Handover (5G adds re-establishment)."""
        old_ast = fiveg_to_session_type(lte_handover())
        new_ast = fiveg_to_session_type(rrc_handover())
        assert not is_subtype(old_ast, new_ast)

    def test_evolution_result_rrc_setup(self) -> None:
        """Full evolution check for RRC Setup."""
        result = check_protocol_evolution(lte_rrc_setup(), rrc_setup())
        assert isinstance(result, ProtocolEvolutionResult)
        assert result.is_backward_compatible
        assert result.old_analysis.is_well_formed
        assert result.new_analysis.is_well_formed

    def test_evolution_result_attach_to_registration(self) -> None:
        """Full evolution check for Attach -> Registration."""
        result = check_protocol_evolution(lte_attach(), nas_registration())
        assert isinstance(result, ProtocolEvolutionResult)
        assert not result.is_backward_compatible

    def test_evolution_result_handover(self) -> None:
        """Full evolution check for LTE Handover -> 5G Handover."""
        result = check_protocol_evolution(lte_handover(), rrc_handover())
        assert isinstance(result, ProtocolEvolutionResult)
        # 5G handover adds re-establishment path
        assert not result.is_backward_compatible


# ---------------------------------------------------------------------------
# Handover-specific tests
# ---------------------------------------------------------------------------

class TestHandoverProtocol:
    """Detailed tests for the 5G handover protocol."""

    def test_handover_has_success_path(self) -> None:
        ast = fiveg_to_session_type(rrc_handover())
        ss = build_statespace(ast)
        labels = {t[1] for t in ss.transitions}
        assert "handoverRequest" in labels
        assert "handoverAck" in labels
        assert "rrcReconfigHandover" in labels
        assert "HO_SUCCESS" in labels
        assert "syncTarget" in labels
        assert "rrcReconfigCompleteHO" in labels

    def test_handover_has_failure_path(self) -> None:
        ast = fiveg_to_session_type(rrc_handover())
        ss = build_statespace(ast)
        labels = {t[1] for t in ss.transitions}
        assert "HO_FAIL" in labels
        assert "rrcReestablishRequest" in labels

    def test_handover_more_states_than_lte(self) -> None:
        ho_5g = build_statespace(fiveg_to_session_type(rrc_handover()))
        ho_4g = build_statespace(fiveg_to_session_type(lte_handover()))
        assert len(ho_5g.states) > len(ho_4g.states)


# ---------------------------------------------------------------------------
# Coverage and test generation tests
# ---------------------------------------------------------------------------

class TestCoverageAndTestGen:
    """Test that coverage and test generation work for 5G protocols."""

    def test_registration_coverage(self) -> None:
        result = verify_5g_protocol(nas_registration())
        assert result.coverage.state_coverage > 0
        assert result.coverage.transition_coverage > 0

    def test_registration_valid_paths(self) -> None:
        result = verify_5g_protocol(nas_registration())
        assert result.num_valid_paths >= 2  # accept + reject + auth_fail

    def test_handover_valid_paths(self) -> None:
        result = verify_5g_protocol(rrc_handover())
        assert result.num_valid_paths >= 2  # success + fail paths

    def test_test_source_generated(self) -> None:
        result = verify_5g_protocol(rrc_setup())
        assert "class" in result.test_source
        assert len(result.test_source) > 100


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------

class TestFormatting:
    """Test report formatting functions."""

    def test_format_5g_report(self) -> None:
        result = verify_5g_protocol(nas_registration())
        report = format_5g_report(result)
        assert "NAS_Registration" in report
        assert "5G PROTOCOL REPORT" in report
        assert "Lattice Analysis" in report
        assert "PASS" in report

    def test_format_evolution_report(self) -> None:
        evo = check_protocol_evolution(lte_rrc_setup(), rrc_setup())
        report = format_evolution_report(evo)
        assert "PROTOCOL EVOLUTION" in report
        assert "COMPATIBLE" in report

    def test_format_5g_summary(self) -> None:
        results = analyze_all_5g()
        summary = format_5g_summary(results)
        assert "5G PROTOCOL VERIFICATION SUMMARY" in summary
        assert "NAS_Registration" in summary
        assert "RRC_Handover" in summary
        assert "All protocols form lattices: YES" in summary


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test protocol registries."""

    def test_all_nas_protocols_count(self) -> None:
        assert len(ALL_NAS_PROTOCOLS) == 5

    def test_all_rrc_protocols_count(self) -> None:
        assert len(ALL_RRC_PROTOCOLS) == 4

    def test_all_5g_equals_nas_plus_rrc(self) -> None:
        assert len(ALL_5G_PROTOCOLS) == len(ALL_NAS_PROTOCOLS) + len(ALL_RRC_PROTOCOLS)

    def test_all_lte_protocols_count(self) -> None:
        assert len(ALL_LTE_PROTOCOLS) == 3

    def test_all_nas_have_nas_layer(self) -> None:
        for p in ALL_NAS_PROTOCOLS:
            assert p.layer == "NAS"

    def test_all_rrc_have_rrc_layer(self) -> None:
        for p in ALL_RRC_PROTOCOLS:
            assert p.layer == "RRC"
