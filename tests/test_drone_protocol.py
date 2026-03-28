"""Tests for drone flight protocols as session types (Step 60m)."""

import math
import pytest
from reticulate.drone_protocol import (
    Waypoint,
    straight_line,
    triangle_flight,
    circle_flight,
    square_flight,
    figure_eight,
    refine_trajectory,
    compute_trajectory_length,
    compute_smoothness,
    formation_v,
    build_flight_protocol,
    physical_invariants,
    verify_flight_safety,
    _waypoints_to_type,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace


class TestWaypoints:
    def test_straight_line(self):
        wps = straight_line(Waypoint(0, 0), Waypoint(100, 0), 5)
        assert len(wps) == 5
        assert wps[0].x == 0
        assert wps[-1].x == 100

    def test_triangle(self):
        wps = triangle_flight(100, 3)
        assert len(wps) >= 10  # 3 sides × 3 points + takeoff + land
        assert wps[0].label == "takeoff"
        assert wps[-1].label == "land"

    def test_circle(self):
        wps = circle_flight(50, 8)
        assert len(wps) == 10  # 8 circle + return + land

    def test_square(self):
        wps = square_flight(100, 2)
        assert len(wps) >= 9  # 4 sides × 2 + takeoff + land

    def test_figure_eight(self):
        wps = figure_eight(50, 8)
        assert len(wps) >= 10


class TestTrajectoryToType:
    def test_empty(self):
        assert _waypoints_to_type([]) == "end"

    def test_single(self):
        t = _waypoints_to_type([Waypoint(0, 0, label="wp_0")])
        assert "wp_0" in t

    def test_chain(self):
        wps = [Waypoint(0, 0, label="a"), Waypoint(1, 0, label="b")]
        t = _waypoints_to_type(wps)
        ss = build_statespace(parse(t))
        assert len(ss.states) == 3  # top, middle, bottom

    def test_triangle_parseable(self):
        wps = triangle_flight(100, 2)
        t = _waypoints_to_type(wps)
        ss = build_statespace(parse(t))
        assert len(ss.states) >= len(wps)


class TestRefinement:
    def test_refine_doubles(self):
        wps = straight_line(Waypoint(0, 0), Waypoint(100, 0), 3)
        refined = refine_trajectory(wps, factor=2)
        assert len(refined) > len(wps)

    def test_refine_increases_smoothness(self):
        wps = straight_line(Waypoint(0, 0), Waypoint(100, 0), 3)
        s1 = compute_smoothness(wps)
        refined = refine_trajectory(wps, factor=3)
        s2 = compute_smoothness(refined)
        assert s2 > s1  # More waypoints per distance = smoother

    def test_refine_preserves_endpoints(self):
        wps = [Waypoint(0, 0, label="start"), Waypoint(100, 50, label="end")]
        refined = refine_trajectory(wps, factor=4)
        assert refined[0].x == 0 and refined[0].y == 0
        assert refined[-1].x == 100 and refined[-1].y == 50


class TestTrajectoryLength:
    def test_zero_length(self):
        assert compute_trajectory_length([Waypoint(0, 0)]) == 0.0

    def test_unit_length(self):
        length = compute_trajectory_length([Waypoint(0, 0), Waypoint(1, 0)])
        assert abs(length - 1.0) < 0.01

    def test_triangle_perimeter(self):
        wps = [Waypoint(0, 0), Waypoint(100, 0), Waypoint(50, 86.6), Waypoint(0, 0)]
        length = compute_trajectory_length(wps)
        assert abs(length - 300) < 5  # Approx equilateral triangle perimeter


class TestSmoothness:
    def test_single_point(self):
        assert compute_smoothness([Waypoint(0, 0)]) == 0.0

    def test_coarse_vs_fine(self):
        coarse = straight_line(Waypoint(0, 0), Waypoint(100, 0), 3)
        fine = straight_line(Waypoint(0, 0), Waypoint(100, 0), 20)
        assert compute_smoothness(fine) > compute_smoothness(coarse)


class TestFormation:
    def test_single_drone(self):
        t = formation_v(1)
        ss = build_statespace(parse(t))
        assert len(ss.states) >= 3

    def test_three_drones(self):
        t = formation_v(3)
        ss = build_statespace(parse(t))
        # 3 drones in parallel: product of three 4-state chains
        assert len(ss.states) >= 8

    def test_formation_parseable(self):
        for n in [2, 3, 4]:
            t = formation_v(n)
            ss = build_statespace(parse(t))
            assert len(ss.states) >= 2


class TestBuildProtocol:
    def test_straight_flight(self):
        wps = straight_line(Waypoint(0, 0, 10), Waypoint(100, 0, 10), 5)
        fp = build_flight_protocol("Straight", wps)
        assert fp.name == "Straight"
        assert fp.num_states >= 5
        assert fp.height >= 4
        assert fp.width == 1  # Sequential

    def test_triangle_protocol(self):
        wps = triangle_flight(100, 2)
        fp = build_flight_protocol("Triangle", wps)
        assert fp.height >= 5
        assert fp.total_paths >= 1
        assert fp.trajectory_length > 0
        assert fp.smoothness > 0


class TestPhysicalInvariants:
    def test_straight_invariants(self):
        wps = straight_line(Waypoint(0, 0, 10), Waypoint(100, 0, 10), 4)
        fp = build_flight_protocol("Straight", wps)
        pi = physical_invariants(fp)
        assert pi.duration_steps >= 3
        assert pi.parallelism == 1
        assert len(pi.phase_distribution) >= 1

    def test_formation_invariants(self):
        t = formation_v(2)
        ss = build_statespace(parse(t))
        # Build minimal flight protocol for formation
        fp = build_flight_protocol("Formation",
            [Waypoint(0, 0, 10, "takeoff"), Waypoint(100, 0, 10, "fly"),
             Waypoint(0, 0, 0, "land")])
        pi = physical_invariants(fp)
        assert pi.duration_steps >= 2


class TestSafety:
    def test_straight_safe(self):
        wps = straight_line(Waypoint(0, 0, 10), Waypoint(100, 0, 10), 4)
        wps[-1] = Waypoint(0, 0, 0, wps[-1].label)  # Return to base
        fp = build_flight_protocol("Safe", wps)
        safety = verify_flight_safety(fp)
        assert safety["terminates"] is True
        assert safety["no_deadlock"] is True
        assert safety["bounded_altitude"] is True

    def test_high_altitude_unsafe(self):
        wps = [Waypoint(0, 0, 0, "takeoff"), Waypoint(50, 50, 200, "too_high"),
               Waypoint(0, 0, 0, "land")]
        fp = build_flight_protocol("High", wps)
        safety = verify_flight_safety(fp)
        assert safety["bounded_altitude"] is False  # 200m > 120m limit

    def test_no_return_warning(self):
        wps = [Waypoint(0, 0, 10, "takeoff"), Waypoint(500, 500, 10, "faraway")]
        fp = build_flight_protocol("NoReturn", wps)
        safety = verify_flight_safety(fp)
        assert safety["returns_to_base"] is False


class TestShapes:
    """Test that different geometric shapes produce different lattice properties."""

    def test_triangle_vs_square(self):
        tri = build_flight_protocol("Tri", triangle_flight(100, 2))
        sq = build_flight_protocol("Sq", square_flight(100, 2))
        # Different shapes → different state counts
        assert tri.num_states != sq.num_states or tri.height != sq.height

    def test_circle_smoothness_increases_with_points(self):
        c8 = build_flight_protocol("C8", circle_flight(50, 8))
        c16 = build_flight_protocol("C16", circle_flight(50, 16))
        assert c16.smoothness > c8.smoothness

    def test_refinement_preserves_shape(self):
        """Refining a trajectory doesn't change the geometric shape, only resolution."""
        wps = triangle_flight(100, 2)
        refined = refine_trajectory(wps, factor=2)
        len1 = compute_trajectory_length(wps)
        len2 = compute_trajectory_length(refined)
        # Lengths should be approximately equal (same path, more points)
        assert abs(len1 - len2) / max(len1, 1) < 0.1
