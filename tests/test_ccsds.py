"""Tests for ccsds.py -- Step 802."""

from __future__ import annotations

import pytest

from reticulate.ccsds import (
    ALL_AOS_PROTOCOLS,
    ALL_CCSDS_PROTOCOLS,
    ALL_CFDP_PROTOCOLS,
    ALL_RF_PROTOCOLS,
    ALL_TC_PROTOCOLS,
    ALL_TM_PROTOCOLS,
    CCSDS_ROLES,
    LINK_MODES,
    CcsdsAnalysisResult,
    CcsdsProtocol,
    LinkMode,
    analyze_all_ccsds,
    aos_mpdu_bpdu_parallel,
    blackout_autonomous_window,
    ccsds_to_session_type,
    cfdp_class1_put,
    cfdp_class2_put,
    classify_morphism_pair,
    detect_deadlocks,
    format_ccsds_report,
    format_ccsds_summary,
    phi_all_states,
    phi_link_mode,
    psi_mode_to_states,
    rf_acquire_lock,
    tc_cop1_fop_farm,
    tc_space_data_link,
    tm_downlink_virtual_channel,
    verify_ccsds_protocol,
)
from reticulate.statespace import build_statespace


class TestProtocolDefinitions:
    def test_tc_cop1_structure(self) -> None:
        p = tc_cop1_fop_farm()
        assert p.name == "TcCop1FopFarm"
        assert p.layer == "COP1"
        assert "ground_mcc" in p.roles
        assert "spacecraft" in p.roles

    def test_tc_space_data_link(self) -> None:
        p = tc_space_data_link()
        assert p.layer == "TC"

    def test_tm_downlink(self) -> None:
        p = tm_downlink_virtual_channel()
        assert p.layer == "TM"
        assert "rsEncode" in p.session_type_string

    def test_aos_uses_parallel(self) -> None:
        p = aos_mpdu_bpdu_parallel()
        assert p.layer == "AOS"
        assert "||" in p.session_type_string

    def test_cfdp_class1(self) -> None:
        p = cfdp_class1_put()
        assert p.layer == "CFDP"

    def test_cfdp_class2(self) -> None:
        p = cfdp_class2_put()
        assert p.layer == "CFDP"
        assert "retransmit" in p.session_type_string.lower()

    def test_blackout(self) -> None:
        p = blackout_autonomous_window()
        assert p.layer == "RF"
        assert "enterBlackout" in p.session_type_string

    def test_rf_acquire(self) -> None:
        p = rf_acquire_lock()
        assert p.layer == "RF"

    def test_invalid_layer_rejected(self) -> None:
        with pytest.raises(ValueError):
            CcsdsProtocol(
                name="X", layer="BOGUS", roles=("ground_mcc",),
                session_type_string="end", spec_reference="x", description="x",
            )

    def test_all_registries_nonempty(self) -> None:
        assert len(ALL_TC_PROTOCOLS) >= 2
        assert len(ALL_TM_PROTOCOLS) >= 1
        assert len(ALL_AOS_PROTOCOLS) >= 1
        assert len(ALL_CFDP_PROTOCOLS) >= 2
        assert len(ALL_RF_PROTOCOLS) >= 2
        assert len(ALL_CCSDS_PROTOCOLS) == (
            len(ALL_TC_PROTOCOLS)
            + len(ALL_TM_PROTOCOLS)
            + len(ALL_AOS_PROTOCOLS)
            + len(ALL_CFDP_PROTOCOLS)
            + len(ALL_RF_PROTOCOLS)
        )


class TestParsing:
    @pytest.mark.parametrize("proto", list(ALL_CCSDS_PROTOCOLS))
    def test_each_protocol_parses(self, proto: CcsdsProtocol) -> None:
        ast = ccsds_to_session_type(proto)
        assert ast is not None

    @pytest.mark.parametrize("proto", list(ALL_CCSDS_PROTOCOLS))
    def test_each_protocol_state_space(self, proto: CcsdsProtocol) -> None:
        ss = build_statespace(ccsds_to_session_type(proto))
        assert len(ss.states) >= 2


class TestVerification:
    @pytest.mark.parametrize("proto", list(ALL_CCSDS_PROTOCOLS))
    def test_verify_each(self, proto: CcsdsProtocol) -> None:
        result = verify_ccsds_protocol(proto)
        assert isinstance(result, CcsdsAnalysisResult)
        assert result.num_states >= 2
        assert result.num_transitions >= 1

    def test_all_form_lattices(self) -> None:
        results = analyze_all_ccsds()
        for r in results:
            assert r.is_well_formed, (
                f"{r.protocol.name} is not a lattice"
            )

    def test_no_deadlocks(self) -> None:
        results = analyze_all_ccsds()
        for r in results:
            assert not r.deadlock_states, (
                f"{r.protocol.name} has deadlock states {r.deadlock_states}"
            )


class TestPhiPsi:
    def test_phi_returns_link_mode(self) -> None:
        p = tc_cop1_fop_farm()
        ss = build_statespace(ccsds_to_session_type(p))
        m = phi_link_mode(p, ss, ss.bottom)
        assert isinstance(m, LinkMode)
        assert m.kind == "TERMINATED"

    def test_phi_top_is_idle(self) -> None:
        p = tc_cop1_fop_farm()
        ss = build_statespace(ccsds_to_session_type(p))
        assert phi_link_mode(p, ss, ss.top).kind == "IDLE"

    def test_phi_all_states_total(self) -> None:
        p = tm_downlink_virtual_channel()
        ss = build_statespace(ccsds_to_session_type(p))
        modes = phi_all_states(p, ss)
        assert set(modes.keys()) == set(ss.states)

    def test_phi_only_uses_link_modes(self) -> None:
        for p in ALL_CCSDS_PROTOCOLS:
            ss = build_statespace(ccsds_to_session_type(p))
            for s in ss.states:
                assert phi_link_mode(p, ss, s).kind in LINK_MODES

    def test_psi_inverse_property(self) -> None:
        p = cfdp_class2_put()
        ss = build_statespace(ccsds_to_session_type(p))
        modes = phi_all_states(p, ss)
        for kind in {m.kind for m in modes.values()}:
            states = psi_mode_to_states(p, ss, kind)
            for s in states:
                assert phi_link_mode(p, ss, s).kind == kind

    def test_classify_morphism_pair(self) -> None:
        for p in ALL_CCSDS_PROTOCOLS:
            ss = build_statespace(ccsds_to_session_type(p))
            kind = classify_morphism_pair(p, ss)
            assert kind in (
                "isomorphism",
                "embedding",
                "projection",
                "galois",
                "section-retraction",
            )

    def test_retransmit_mode_present_in_cop1(self) -> None:
        p = tc_cop1_fop_farm()
        ss = build_statespace(ccsds_to_session_type(p))
        retransmit = psi_mode_to_states(p, ss, "RETRANSMIT")
        assert len(retransmit) >= 1

    def test_blackout_mode_present(self) -> None:
        p = blackout_autonomous_window()
        ss = build_statespace(ccsds_to_session_type(p))
        blackout = psi_mode_to_states(p, ss, "BLACKOUT")
        assert len(blackout) >= 1

    def test_file_transfer_mode_in_cfdp(self) -> None:
        p = cfdp_class2_put()
        ss = build_statespace(ccsds_to_session_type(p))
        ft = psi_mode_to_states(p, ss, "FILE_TRANSFER")
        assert len(ft) >= 1


class TestDeadlockDetection:
    def test_no_deadlocks_well_formed(self) -> None:
        for p in ALL_CCSDS_PROTOCOLS:
            ss = build_statespace(ccsds_to_session_type(p))
            assert detect_deadlocks(ss) == ()


class TestReportFormatting:
    def test_format_report_contains_name(self) -> None:
        p = tc_cop1_fop_farm()
        r = verify_ccsds_protocol(p)
        text = format_ccsds_report(r)
        assert "TcCop1FopFarm" in text
        assert "Lattice Analysis" in text

    def test_format_summary(self) -> None:
        results = analyze_all_ccsds()
        text = format_ccsds_summary(results)
        assert "CCSDS" in text
        for r in results:
            assert r.protocol.name in text


class TestRoles:
    def test_roles_are_recognised(self) -> None:
        for p in ALL_CCSDS_PROTOCOLS:
            for r in p.roles:
                assert r in CCSDS_ROLES


class TestAosParallelProductLattice:
    def test_aos_states_are_product(self) -> None:
        p = aos_mpdu_bpdu_parallel()
        ss = build_statespace(ccsds_to_session_type(p))
        # Each pipeline has 4 states (init, header-inserted, packetized, downlinked)
        # Product is 4*4 = 16 states.
        assert len(ss.states) == 16
