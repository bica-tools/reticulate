"""Tests for autosar_can.py -- Step 76."""

from __future__ import annotations

import pytest

from reticulate.autosar_can import (
    ALL_AUTOSAR_PROTOCOLS,
    ALL_CAN_PROTOCOLS,
    ALL_RUNNABLES,
    ALL_UDS_PROTOCOLS,
    ASIL_LEVELS,
    AutosarAnalysisResult,
    AutosarProtocol,
    SafetyClaim,
    analyze_all_autosar,
    autosar_to_session_type,
    can_fd_exchange,
    can_frame_exchange,
    can_network_management,
    can_tx_confirmation,
    check_asil_decomposition,
    classify_morphism_pair,
    format_autosar_report,
    format_autosar_summary,
    phi_all_states,
    phi_safety_claim,
    psi_claim_to_states,
    runnable_client_server,
    runnable_event_triggered,
    runnable_mode_switch,
    runnable_periodic,
    uds_diagnostic_session,
    verify_autosar_protocol,
)
from reticulate.statespace import build_statespace


class TestRunnableDefinitions:
    def test_periodic_structure(self) -> None:
        p = runnable_periodic()
        assert p.name == "Runnable_Periodic"
        assert p.layer == "RTE"
        assert p.asil == "B"
        assert "SWC_Sensor" in p.ecus

    def test_client_server(self) -> None:
        p = runnable_client_server()
        assert p.layer == "RTE"
        assert p.asil == "C"

    def test_event_triggered(self) -> None:
        p = runnable_event_triggered()
        assert p.asil == "A"

    def test_mode_switch(self) -> None:
        p = runnable_mode_switch()
        assert p.asil == "D"

    def test_all_parse(self) -> None:
        for p in ALL_RUNNABLES:
            ast = autosar_to_session_type(p)
            assert ast is not None


class TestCANDefinitions:
    def test_frame_exchange(self) -> None:
        p = can_frame_exchange()
        assert p.layer == "CAN"
        assert "ISO 11898" in p.spec_reference

    def test_can_fd(self) -> None:
        p = can_fd_exchange()
        assert "CAN-FD" in p.spec_reference or "11898" in p.spec_reference

    def test_tx_confirmation(self) -> None:
        p = can_tx_confirmation()
        assert "Com" in p.ecus

    def test_network_management(self) -> None:
        p = can_network_management()
        assert p.asil == "A"

    def test_all_parse(self) -> None:
        for p in ALL_CAN_PROTOCOLS:
            ast = autosar_to_session_type(p)
            assert ast is not None


class TestUDS:
    def test_diagnostic_session(self) -> None:
        p = uds_diagnostic_session()
        assert p.layer == "UDS"
        assert p.asil == "QM"

    def test_parses(self) -> None:
        for p in ALL_UDS_PROTOCOLS:
            assert autosar_to_session_type(p) is not None


class TestValidation:
    def test_invalid_asil(self) -> None:
        with pytest.raises(ValueError):
            AutosarProtocol(
                name="X", layer="CAN", asil="Z", ecus=(),
                session_type_string="end",
                spec_reference="", description="",
            )

    def test_invalid_layer(self) -> None:
        with pytest.raises(ValueError):
            AutosarProtocol(
                name="X", layer="BOGUS", asil="A", ecus=(),
                session_type_string="end",
                spec_reference="", description="",
            )

    def test_asil_levels(self) -> None:
        assert ASIL_LEVELS == ("QM", "A", "B", "C", "D")


class TestVerification:
    def test_verify_periodic(self) -> None:
        r = verify_autosar_protocol(runnable_periodic())
        assert isinstance(r, AutosarAnalysisResult)
        assert r.is_well_formed
        assert r.num_states > 0
        assert r.num_transitions > 0

    def test_verify_can_frame(self) -> None:
        r = verify_autosar_protocol(can_frame_exchange())
        assert r.is_well_formed

    def test_verify_all(self) -> None:
        results = analyze_all_autosar()
        assert len(results) == len(ALL_AUTOSAR_PROTOCOLS)
        for r in results:
            assert r.is_well_formed, f"{r.protocol.name} is not a lattice"

    def test_safety_claims_populated(self) -> None:
        r = verify_autosar_protocol(can_frame_exchange())
        assert len(r.safety_claims) == r.num_states
        for c in r.safety_claims.values():
            assert isinstance(c, SafetyClaim)


class TestMorphisms:
    def test_phi_bottom_is_safe_state(self) -> None:
        p = runnable_periodic()
        ss = build_statespace(autosar_to_session_type(p))
        c = phi_safety_claim(p, ss, ss.bottom)
        assert c.kind == "SAFE_STATE"

    def test_phi_top_is_ffi(self) -> None:
        p = runnable_periodic()
        ss = build_statespace(autosar_to_session_type(p))
        c = phi_safety_claim(p, ss, ss.top)
        assert c.kind == "FFI"

    def test_phi_all_states_complete(self) -> None:
        p = can_frame_exchange()
        ss = build_statespace(autosar_to_session_type(p))
        claims = phi_all_states(p, ss)
        assert set(claims.keys()) == set(ss.states)

    def test_psi_preimage_correct(self) -> None:
        p = can_frame_exchange()
        ss = build_statespace(autosar_to_session_type(p))
        claims = phi_all_states(p, ss)
        for kind in {c.kind for c in claims.values()}:
            preimage = psi_claim_to_states(p, ss, kind)
            for s in preimage:
                assert phi_safety_claim(p, ss, s).kind == kind

    def test_psi_unknown_claim_is_empty(self) -> None:
        p = runnable_periodic()
        ss = build_statespace(autosar_to_session_type(p))
        assert psi_claim_to_states(p, ss, "NOT_A_CLAIM") == frozenset()

    def test_classify_morphism(self) -> None:
        for p in ALL_AUTOSAR_PROTOCOLS:
            ss = build_statespace(autosar_to_session_type(p))
            cls = classify_morphism_pair(p, ss)
            assert cls in {
                "isomorphism", "embedding", "projection",
                "galois", "section-retraction",
            }

    def test_phi_psi_idempotent_on_image(self) -> None:
        """φ ∘ ψ ∘ φ = φ (section--retraction law)."""
        for p in ALL_AUTOSAR_PROTOCOLS:
            ss = build_statespace(autosar_to_session_type(p))
            for s in ss.states:
                k1 = phi_safety_claim(p, ss, s).kind
                preimage = psi_claim_to_states(p, ss, k1)
                assert s in preimage


class TestASILDecomposition:
    def test_decomposition_runs(self) -> None:
        parent = runnable_mode_switch()
        child_a = runnable_periodic()
        child_b = runnable_event_triggered()
        r = check_asil_decomposition(parent, child_a, child_b)
        assert r is not None


class TestFormatting:
    def test_format_report(self) -> None:
        r = verify_autosar_protocol(can_frame_exchange())
        text = format_autosar_report(r)
        assert "CAN_FrameExchange" in text
        assert "ASIL" in text

    def test_format_summary(self) -> None:
        results = analyze_all_autosar()
        text = format_autosar_summary(results)
        assert "SUMMARY" in text
        assert "CAN_FrameExchange" in text
