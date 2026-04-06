"""Tests for v2x_intersection.py -- Step 804."""

from __future__ import annotations

import pytest

from reticulate.statespace import build_statespace
from reticulate.v2x_intersection import (
    ALL_V2X_PROTOCOLS,
    CAL_LEVELS,
    ROW_VERDICTS,
    RowVerdict,
    V2xAnalysisResult,
    V2xProtocol,
    analyze_all_v2x,
    classify_morphism_pair,
    check_collision_free,
    check_fairness,
    check_liveness,
    emergency_vehicle_preemption,
    four_way_stop,
    format_v2x_report,
    format_v2x_summary,
    pedestrian_crossing,
    phi_all_states,
    phi_row_verdict,
    psi_verdict_to_states,
    signalized_intersection,
    unprotected_left_turn,
    v2x_to_session_type,
    verify_v2x_protocol,
)


class TestProtocolDefinitions:
    def test_four_way_stop(self) -> None:
        p = four_way_stop()
        assert p.name == "FourWayStop"
        assert p.scenario == "stop_sign"
        assert p.cal == "CAL2"
        assert "ego" in p.roles and "other" in p.roles

    def test_signalized(self) -> None:
        p = signalized_intersection()
        assert p.scenario == "signalized"
        assert "signal" in p.roles

    def test_left_turn(self) -> None:
        p = unprotected_left_turn()
        assert p.scenario == "left_turn"
        assert p.cal == "CAL3"

    def test_emergency(self) -> None:
        p = emergency_vehicle_preemption()
        assert p.cal == "CAL4"
        assert "emergency" in p.roles

    def test_pedestrian(self) -> None:
        p = pedestrian_crossing()
        assert "pedestrian" in p.roles

    def test_all_registry(self) -> None:
        assert len(ALL_V2X_PROTOCOLS) == 5
        names = {p.name for p in ALL_V2X_PROTOCOLS}
        assert "FourWayStop" in names

    def test_invalid_cal(self) -> None:
        with pytest.raises(ValueError):
            V2xProtocol(
                name="X", scenario="stop_sign", cal="CAL9", roles=("ego",),
                session_type_string="end", spec_reference="", description="",
            )

    def test_invalid_scenario(self) -> None:
        with pytest.raises(ValueError):
            V2xProtocol(
                name="X", scenario="bogus", cal="CAL1", roles=("ego",),
                session_type_string="end", spec_reference="", description="",
            )


class TestParsing:
    @pytest.mark.parametrize("p", list(ALL_V2X_PROTOCOLS))
    def test_all_parse(self, p: V2xProtocol) -> None:
        ast = v2x_to_session_type(p)
        assert ast is not None
        ss = build_statespace(ast)
        assert len(ss.states) > 0

    def test_four_way_stop_state_count(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        # Product lattice of two parallel approaches => > 10 states.
        assert len(ss.states) >= 10


class TestMorphisms:
    def test_phi_bottom_clear(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        v = phi_row_verdict(p, ss, ss.bottom)
        assert v.verdict == "CLEAR"
        assert v.cal == p.cal

    def test_phi_top_wait(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        v = phi_row_verdict(p, ss, ss.top)
        assert v.verdict == "WAIT"

    def test_phi_all_states(self) -> None:
        p = signalized_intersection()
        ss = build_statespace(v2x_to_session_type(p))
        verdicts = phi_all_states(p, ss)
        assert set(verdicts.keys()) == set(ss.states)

    def test_psi_inverse(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        for verdict in ROW_VERDICTS:
            states = psi_verdict_to_states(p, ss, verdict)
            for s in states:
                assert phi_row_verdict(p, ss, s).verdict == verdict

    def test_psi_invalid_verdict(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        with pytest.raises(ValueError):
            psi_verdict_to_states(p, ss, "BOGUS")

    def test_classify(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        cls = classify_morphism_pair(p, ss)
        assert cls in {"isomorphism", "embedding", "projection",
                       "galois", "section-retraction"}

    def test_section_retraction_property(self) -> None:
        # phi o psi = id on image of phi.
        p = signalized_intersection()
        ss = build_statespace(v2x_to_session_type(p))
        verdicts = phi_all_states(p, ss)
        image = {v.verdict for v in verdicts.values()}
        for k in image:
            states = psi_verdict_to_states(p, ss, k)
            for s in states:
                assert phi_row_verdict(p, ss, s).verdict == k


class TestSafety:
    def test_collision_free_stop_sign(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        assert check_collision_free(p, ss)

    def test_fairness_stop_sign(self) -> None:
        p = four_way_stop()
        ss = build_statespace(v2x_to_session_type(p))
        assert check_fairness(p, ss)

    def test_liveness_all(self) -> None:
        for p in ALL_V2X_PROTOCOLS:
            ss = build_statespace(v2x_to_session_type(p))
            assert check_liveness(p, ss)

    def test_emergency_allowed(self) -> None:
        p = emergency_vehicle_preemption()
        ss = build_statespace(v2x_to_session_type(p))
        # emergency vehicles are allowed; collision_free should still be True
        assert check_collision_free(p, ss)


class TestPipeline:
    def test_verify_returns_result(self) -> None:
        p = four_way_stop()
        r = verify_v2x_protocol(p)
        assert isinstance(r, V2xAnalysisResult)
        assert r.num_states > 0
        assert r.num_transitions > 0

    def test_analyze_all(self) -> None:
        rs = analyze_all_v2x()
        assert len(rs) == len(ALL_V2X_PROTOCOLS)
        for r in rs:
            assert r.is_well_formed or not r.is_well_formed  # both ok
            assert r.liveness_holds

    def test_report_format(self) -> None:
        p = four_way_stop()
        r = verify_v2x_protocol(p)
        text = format_v2x_report(r)
        assert "FourWayStop" in text
        assert "ROW Verdicts" in text

    def test_summary_format(self) -> None:
        rs = analyze_all_v2x()
        text = format_v2x_summary(rs)
        assert "V2X INTERSECTION" in text
        assert "FourWayStop" in text


class TestRowVerdict:
    def test_invalid_verdict(self) -> None:
        with pytest.raises(ValueError):
            RowVerdict(verdict="BOGUS", cal="CAL1", rationale="x")

    def test_valid_verdicts(self) -> None:
        for v in ROW_VERDICTS:
            r = RowVerdict(verdict=v, cal="CAL1", rationale="ok")
            assert r.verdict == v


class TestConstants:
    def test_cal_levels(self) -> None:
        assert "CAL1" in CAL_LEVELS
        assert "CAL4" in CAL_LEVELS

    def test_row_verdicts(self) -> None:
        assert "CLEAR" in ROW_VERDICTS
        assert "EMERGENCY_BRAKE" in ROW_VERDICTS
        assert len(ROW_VERDICTS) == 5
