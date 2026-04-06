"""Tests for ros2_nav.py -- Step 801."""

from __future__ import annotations

import pytest

from reticulate.ros2_nav import (
    ALL_ACTION_PROTOCOLS,
    ALL_AMCL_PROTOCOLS,
    ALL_BT_PROTOCOLS,
    ALL_ROS2_PROTOCOLS,
    ALL_SENSOR_PROTOCOLS,
    ALL_TF_PROTOCOLS,
    NAV2_NODES,
    SUPERVISOR_MODES,
    Ros2AnalysisResult,
    Ros2Protocol,
    SupervisorMode,
    amcl_localization,
    analyze_all_ros2,
    bt_navigator_fallback,
    classify_morphism_pair,
    compute_path_to_pose,
    detect_deadlocks,
    follow_path,
    format_ros2_report,
    format_ros2_summary,
    navigate_to_pose,
    phi_all_states,
    phi_supervisor_mode,
    psi_mode_to_states,
    recovery_spin,
    ros2_to_session_type,
    sensor_pipeline_parallel,
    tf2_transform_chain,
    verify_ros2_protocol,
)
from reticulate.statespace import build_statespace


class TestProtocolDefinitions:
    def test_navigate_to_pose_structure(self) -> None:
        p = navigate_to_pose()
        assert p.name == "NavigateToPose"
        assert p.layer == "ACTION"
        assert "nav2_bt_navigator" in p.nodes

    def test_compute_path_to_pose_structure(self) -> None:
        p = compute_path_to_pose()
        assert p.layer == "ACTION"
        assert "nav2_planner" in p.nodes

    def test_follow_path_structure(self) -> None:
        p = follow_path()
        assert "nav2_controller" in p.nodes

    def test_sensor_pipeline_parallel_uses_parallel(self) -> None:
        p = sensor_pipeline_parallel()
        assert p.layer == "SENSOR"
        assert "||" in p.session_type_string

    def test_bt_navigator_fallback(self) -> None:
        p = bt_navigator_fallback()
        assert p.layer == "BT"

    def test_amcl_localization(self) -> None:
        p = amcl_localization()
        assert p.layer == "AMCL"
        assert "amcl" in p.nodes

    def test_tf2_transform_chain(self) -> None:
        p = tf2_transform_chain()
        assert p.layer == "TF"

    def test_recovery_spin(self) -> None:
        p = recovery_spin()
        assert p.name == "RecoverySpin"

    def test_invalid_layer_rejected(self) -> None:
        with pytest.raises(ValueError):
            Ros2Protocol(
                name="bad",
                layer="QUUX",
                nodes=("amcl",),
                session_type_string="end",
                spec_reference="-",
                description="-",
            )


class TestRegistries:
    def test_all_protocols_aggregate(self) -> None:
        n = (
            len(ALL_ACTION_PROTOCOLS)
            + len(ALL_BT_PROTOCOLS)
            + len(ALL_SENSOR_PROTOCOLS)
            + len(ALL_TF_PROTOCOLS)
            + len(ALL_AMCL_PROTOCOLS)
        )
        assert len(ALL_ROS2_PROTOCOLS) == n
        assert n >= 8

    def test_supervisor_modes_nonempty(self) -> None:
        assert "NAVIGATING" in SUPERVISOR_MODES
        assert "EMERGENCY_STOP" in SUPERVISOR_MODES

    def test_nav2_nodes_nonempty(self) -> None:
        assert "nav2_planner" in NAV2_NODES


class TestVerification:
    @pytest.mark.parametrize("proto", ALL_ROS2_PROTOCOLS)
    def test_each_protocol_parses(self, proto: Ros2Protocol) -> None:
        ast = ros2_to_session_type(proto)
        assert ast is not None

    @pytest.mark.parametrize("proto", ALL_ROS2_PROTOCOLS)
    def test_each_protocol_is_lattice(self, proto: Ros2Protocol) -> None:
        result = verify_ros2_protocol(proto)
        assert result.is_well_formed, (
            f"{proto.name} did not form a lattice"
        )

    @pytest.mark.parametrize("proto", ALL_ROS2_PROTOCOLS)
    def test_no_deadlocks_in_canonical_protocols(self, proto: Ros2Protocol) -> None:
        result = verify_ros2_protocol(proto)
        assert result.deadlock_states == ()

    def test_analyze_all(self) -> None:
        results = analyze_all_ros2()
        assert len(results) == len(ALL_ROS2_PROTOCOLS)
        assert all(r.is_well_formed for r in results)


class TestProductCompression:
    def test_sensor_product_lattice_size(self) -> None:
        """L(lidar) x L(odom) x L(imu) = 4 x 4 x 4 = 64 product states."""
        result = verify_ros2_protocol(sensor_pipeline_parallel())
        assert result.num_states == 64

    def test_sensor_pipeline_distributive(self) -> None:
        """Product of chains is distributive."""
        result = verify_ros2_protocol(sensor_pipeline_parallel())
        assert result.distributivity.is_distributive


class TestMorphisms:
    def test_phi_returns_supervisor_mode(self) -> None:
        proto = navigate_to_pose()
        ss = build_statespace(ros2_to_session_type(proto))
        mode = phi_supervisor_mode(proto, ss, ss.bottom)
        assert isinstance(mode, SupervisorMode)
        assert mode.kind == "GOAL_REACHED"

    def test_phi_top_is_idle(self) -> None:
        proto = navigate_to_pose()
        ss = build_statespace(ros2_to_session_type(proto))
        mode = phi_supervisor_mode(proto, ss, ss.top)
        assert mode.kind == "IDLE"

    def test_phi_all_states_total(self) -> None:
        proto = navigate_to_pose()
        ss = build_statespace(ros2_to_session_type(proto))
        modes = phi_all_states(proto, ss)
        assert set(modes.keys()) == set(ss.states)

    def test_psi_inverse_image(self) -> None:
        proto = navigate_to_pose()
        ss = build_statespace(ros2_to_session_type(proto))
        states = psi_mode_to_states(proto, ss, "GOAL_REACHED")
        assert ss.bottom in states

    def test_psi_phi_consistent(self) -> None:
        """For every state s, s in psi(phi(s).kind)."""
        proto = bt_navigator_fallback()
        ss = build_statespace(ros2_to_session_type(proto))
        for s in ss.states:
            kind = phi_supervisor_mode(proto, ss, s).kind
            assert s in psi_mode_to_states(proto, ss, kind)

    def test_classify_morphism_pair(self) -> None:
        proto = navigate_to_pose()
        ss = build_statespace(ros2_to_session_type(proto))
        kind = classify_morphism_pair(proto, ss)
        assert kind in (
            "isomorphism",
            "embedding",
            "projection",
            "galois",
            "section-retraction",
        )

    def test_emergency_stop_mode_present(self) -> None:
        proto = navigate_to_pose()
        ss = build_statespace(ros2_to_session_type(proto))
        modes = phi_all_states(proto, ss)
        kinds = {m.kind for m in modes.values()}
        assert "EMERGENCY_STOP" in kinds


class TestDeadlockDetection:
    def test_no_deadlock_in_nav_to_pose(self) -> None:
        proto = navigate_to_pose()
        ss = build_statespace(ros2_to_session_type(proto))
        assert detect_deadlocks(ss) == ()

    def test_deadlock_detector_finds_synthetic(self) -> None:
        from reticulate.statespace import StateSpace
        ss = StateSpace(
            states=frozenset({0, 1, 2}),
            transitions=frozenset({(0, "a", 1)}),
            top=0,
            bottom=2,
            selection_transitions=frozenset(),
        )
        # State 1 has no outgoing and is not bottom -> deadlocked
        assert 1 in detect_deadlocks(ss)


class TestFormatting:
    def test_format_report_contains_name(self) -> None:
        result = verify_ros2_protocol(navigate_to_pose())
        text = format_ros2_report(result)
        assert "NavigateToPose" in text
        assert "phi" in text or "Supervisor" in text

    def test_format_summary_lists_all(self) -> None:
        results = analyze_all_ros2()
        text = format_ros2_summary(results)
        for p in ALL_ROS2_PROTOCOLS:
            assert p.name in text
