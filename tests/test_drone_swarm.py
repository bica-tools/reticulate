"""Tests for drone_swarm.py -- Step 805."""

from __future__ import annotations

import pytest

from reticulate.statespace import build_statespace
from reticulate.drone_swarm import (
    ALL_SWARM_PROTOCOLS,
    SAL_LEVELS,
    SWARM_VERDICTS,
    SwarmAnalysisResult,
    SwarmProtocol,
    SwarmVerdict,
    analyze_all_swarms,
    classify_morphism_pair,
    check_consensus_progress,
    check_fault_tolerance,
    check_liveness,
    check_no_collision,
    collision_avoidance,
    formation_flying,
    format_swarm_report,
    format_swarm_summary,
    leader_election,
    leader_failover,
    phi_all_states,
    phi_swarm_verdict,
    psi_verdict_to_states,
    swarm_to_session_type,
    task_allocation,
    verify_swarm_protocol,
)


class TestProtocolDefinitions:
    def test_leader_election(self) -> None:
        p = leader_election()
        assert p.name == "LeaderElection"
        assert p.scenario == "election"
        assert p.sal == "SAL2"
        assert "drone1" in p.roles and "drone2" in p.roles

    def test_formation(self) -> None:
        p = formation_flying()
        assert p.scenario == "formation"
        assert p.sal == "SAL3"
        assert "leader" in p.roles and "wingman" in p.roles

    def test_task_alloc(self) -> None:
        p = task_allocation()
        assert p.scenario == "task_alloc"
        assert p.sal == "SAL2"

    def test_collision(self) -> None:
        p = collision_avoidance()
        assert p.scenario == "collision"
        assert p.sal == "SAL4"

    def test_failover(self) -> None:
        p = leader_failover()
        assert p.scenario == "failover"
        assert "successor" in p.roles

    def test_registry_size(self) -> None:
        assert len(ALL_SWARM_PROTOCOLS) == 5

    def test_all_distinct_names(self) -> None:
        names = {p.name for p in ALL_SWARM_PROTOCOLS}
        assert len(names) == 5

    def test_invalid_sal(self) -> None:
        with pytest.raises(ValueError):
            SwarmProtocol(
                name="x", scenario="election", sal="SAL9",
                roles=("a",), session_type_string="end",
                spec_reference="", description="",
            )

    def test_invalid_scenario(self) -> None:
        with pytest.raises(ValueError):
            SwarmProtocol(
                name="x", scenario="bogus", sal="SAL1",
                roles=("a",), session_type_string="end",
                spec_reference="", description="",
            )

    def test_invalid_verdict(self) -> None:
        with pytest.raises(ValueError):
            SwarmVerdict(verdict="WAT", sal="SAL1", rationale="")


class TestStateSpaceConstruction:
    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_parses(self, proto: SwarmProtocol) -> None:
        ast = swarm_to_session_type(proto)
        assert ast is not None

    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_state_space_nonempty(self, proto: SwarmProtocol) -> None:
        ss = build_statespace(swarm_to_session_type(proto))
        assert len(ss.states) > 0
        assert len(ss.transitions) > 0


class TestVerification:
    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_verify_runs(self, proto: SwarmProtocol) -> None:
        r = verify_swarm_protocol(proto)
        assert isinstance(r, SwarmAnalysisResult)
        assert r.num_states > 0
        assert r.num_transitions > 0

    def test_analyze_all(self) -> None:
        results = analyze_all_swarms()
        assert len(results) == 5

    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_liveness(self, proto: SwarmProtocol) -> None:
        r = verify_swarm_protocol(proto)
        assert r.liveness_holds

    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_fault_tolerant(self, proto: SwarmProtocol) -> None:
        r = verify_swarm_protocol(proto)
        assert r.fault_tolerant

    def test_collision_protocol_safe(self) -> None:
        # Collision-avoidance scenario is allowed to expose ABORT.
        r = verify_swarm_protocol(collision_avoidance())
        assert r.no_collision

    def test_formation_no_abort(self) -> None:
        r = verify_swarm_protocol(formation_flying())
        # Formation flying contains 'abortMission' which maps to ABORT —
        # so no_collision is False here. Document the alarm.
        assert r.no_collision is False

    def test_consensus_progress_holds_or_vacuous(self) -> None:
        for p in ALL_SWARM_PROTOCOLS:
            r = verify_swarm_protocol(p)
            assert r.consensus_progress


class TestMorphisms:
    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_phi_all_states_total(self, proto: SwarmProtocol) -> None:
        ss = build_statespace(swarm_to_session_type(proto))
        verdicts = phi_all_states(proto, ss)
        assert set(verdicts.keys()) == set(ss.states)
        for v in verdicts.values():
            assert v.verdict in SWARM_VERDICTS

    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_psi_inverse(self, proto: SwarmProtocol) -> None:
        ss = build_statespace(swarm_to_session_type(proto))
        verdicts = phi_all_states(proto, ss)
        for s, v in verdicts.items():
            states = psi_verdict_to_states(proto, ss, v.verdict)
            assert s in states

    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_section_retraction(self, proto: SwarmProtocol) -> None:
        # φ ∘ ψ = id on the image of φ
        ss = build_statespace(swarm_to_session_type(proto))
        verdicts = phi_all_states(proto, ss)
        image = {v.verdict for v in verdicts.values()}
        for verdict in image:
            states = psi_verdict_to_states(proto, ss, verdict)
            for s in states:
                assert phi_swarm_verdict(proto, ss, s).verdict == verdict

    def test_psi_invalid_verdict(self) -> None:
        proto = leader_election()
        ss = build_statespace(swarm_to_session_type(proto))
        with pytest.raises(ValueError):
            psi_verdict_to_states(proto, ss, "BOGUS")

    @pytest.mark.parametrize("proto", ALL_SWARM_PROTOCOLS)
    def test_classify_morphism(self, proto: SwarmProtocol) -> None:
        ss = build_statespace(swarm_to_session_type(proto))
        kind = classify_morphism_pair(proto, ss)
        assert kind in {"isomorphism", "embedding", "projection",
                        "galois", "section-retraction"}

    def test_bottom_is_idle(self) -> None:
        proto = formation_flying()
        ss = build_statespace(swarm_to_session_type(proto))
        v = phi_swarm_verdict(proto, ss, ss.bottom)
        assert v.verdict == "IDLE"


class TestReports:
    def test_format_report(self) -> None:
        r = verify_swarm_protocol(formation_flying())
        text = format_swarm_report(r)
        assert "FormationFlying" in text
        assert "States" in text

    def test_format_summary(self) -> None:
        text = format_swarm_summary(analyze_all_swarms())
        assert "DRONE SWARM" in text
        for p in ALL_SWARM_PROTOCOLS:
            assert p.name in text


class TestSafetyChecks:
    def test_no_collision_helper(self) -> None:
        p = leader_election()
        ss = build_statespace(swarm_to_session_type(p))
        assert check_no_collision(p, ss)

    def test_consensus_helper(self) -> None:
        p = leader_election()
        ss = build_statespace(swarm_to_session_type(p))
        assert check_consensus_progress(p, ss)

    def test_liveness_helper(self) -> None:
        p = leader_election()
        ss = build_statespace(swarm_to_session_type(p))
        assert check_liveness(p, ss)

    def test_fault_tolerance_helper(self) -> None:
        p = leader_failover()
        ss = build_statespace(swarm_to_session_type(p))
        assert check_fault_tolerance(p, ss)
