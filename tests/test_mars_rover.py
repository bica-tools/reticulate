"""Tests for mars_rover (Step 803)."""

from __future__ import annotations

import pytest

from reticulate.mars_rover import (
    RoverMode,
    SolPhase,
    SensorPipeline,
    SupervisorState,
    rover_mode_type,
    sol_cycle_type,
    sensor_pipeline_type,
    parallel_sensor_type,
    full_rover_type,
    build_rover_protocol,
    build_minimal_rover,
    naive_interleaving_size,
    product_lattice_size,
    compression_report,
    phi_lattice_to_supervisor,
    psi_supervisor_to_lattice,
    is_galois_pair,
    ContingencyTrigger,
    CONTINGENCY_TABLE,
    find_contingency,
    ground_link_windows,
    verify_rover,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_modes(self):
        assert RoverMode.SAFE.value == "SAFE"
        assert len(list(RoverMode)) == 6

    def test_phases(self):
        assert SolPhase.WAKE.value == "wake"
        assert len(list(SolPhase)) == 3

    def test_sensors(self):
        assert SensorPipeline.IMU.value == "imu"
        assert len(list(SensorPipeline)) == 4


# ---------------------------------------------------------------------------
# SupervisorState ordering
# ---------------------------------------------------------------------------

class TestSupervisorOrder:
    def test_reflexive(self):
        s = SupervisorState(RoverMode.SAFE, SolPhase.WAKE, frozenset())
        assert s <= s
        assert not (s < s)

    def test_chain(self):
        a = SupervisorState(RoverMode.SAFE, SolPhase.WAKE, frozenset())
        b = SupervisorState(RoverMode.MIN_OPS, SolPhase.WAKE, frozenset({SensorPipeline.IMU}))
        c = SupervisorState(RoverMode.NOMINAL_SCIENCE, SolPhase.SLEEP,
                            frozenset({SensorPipeline.IMU, SensorPipeline.HAZCAM}))
        assert a <= b <= c
        assert a < c

    def test_incomparable(self):
        a = SupervisorState(RoverMode.NOMINAL_DRIVE, SolPhase.WAKE,
                            frozenset({SensorPipeline.IMU}))
        b = SupervisorState(RoverMode.MIN_OPS, SolPhase.SLEEP,
                            frozenset({SensorPipeline.IMU}))
        assert not (a <= b)
        assert not (b <= a)


# ---------------------------------------------------------------------------
# Type construction
# ---------------------------------------------------------------------------

class TestTypeStrings:
    def test_mode_type_parses(self):
        from reticulate.parser import parse
        parse(rover_mode_type())

    def test_sol_cycle_parses(self):
        from reticulate.parser import parse
        parse(sol_cycle_type())

    def test_sensor_pipeline_parses(self):
        from reticulate.parser import parse
        parse(sensor_pipeline_type("imu"))

    def test_parallel_sensor_default(self):
        from reticulate.parser import parse
        s = parallel_sensor_type()
        parse(s)
        assert "||" in s

    def test_parallel_sensor_subset(self):
        from reticulate.parser import parse
        s = parallel_sensor_type([SensorPipeline.IMU, SensorPipeline.HAZCAM])
        parse(s)
        assert "imu" in s and "hazcam" in s

    def test_parallel_sensor_empty(self):
        assert parallel_sensor_type([]) == "end"

    def test_parallel_sensor_single(self):
        from reticulate.parser import parse
        s = parallel_sensor_type([SensorPipeline.DRILL])
        parse(s)

    def test_full_rover_parses(self):
        from reticulate.parser import parse
        parse(full_rover_type([SensorPipeline.IMU]))


# ---------------------------------------------------------------------------
# Protocol building
# ---------------------------------------------------------------------------

class TestBuildProtocol:
    def test_minimal_builds(self):
        rp = build_minimal_rover()
        assert rp.num_states > 0
        assert rp.num_transitions > 0
        assert rp.name == "minimal"

    def test_minimal_is_lattice(self):
        rp = build_minimal_rover()
        # Pure parallel composition of finite sequences should be a lattice
        assert rp.is_lattice

    def test_full_protocol_builds(self):
        rp = build_rover_protocol(pipelines=[SensorPipeline.IMU, SensorPipeline.HAZCAM])
        assert rp.num_states > 0

    def test_protocol_name(self):
        rp = build_rover_protocol(name="curiosity",
                                   pipelines=[SensorPipeline.IMU])
        assert rp.name == "curiosity"


# ---------------------------------------------------------------------------
# Compression: naive interleaving vs product lattice
# ---------------------------------------------------------------------------

class TestCompression:
    def test_naive_size_simple(self):
        # 2 pipelines x 1 step each: 2!/(1!*1!) = 2
        assert naive_interleaving_size(1, 2) == 2

    def test_naive_size_grows(self):
        # 4 pipelines x 3 steps: 12!/(3!^4) = 369600
        assert naive_interleaving_size(3, 4) == 369600

    def test_product_lattice_size(self):
        assert product_lattice_size(3, 4) == 4 ** 4
        assert product_lattice_size(1, 2) == 4

    def test_compression_ratio_positive(self):
        rep = compression_report(3, 4)
        assert rep.naive_size > rep.lattice_size
        assert rep.compression_ratio > 1.0

    def test_compression_report_default(self):
        rep = compression_report()
        assert rep.naive_size == 369600
        assert rep.lattice_size == 256


# ---------------------------------------------------------------------------
# Bidirectional morphisms
# ---------------------------------------------------------------------------

class TestMorphisms:
    def test_phi_returns_supervisor(self):
        rp = build_minimal_rover()
        sup = phi_lattice_to_supervisor(0, rp.state_space)
        assert isinstance(sup, SupervisorState)

    def test_phi_bottom_is_safe(self):
        rp = build_minimal_rover()
        sup = phi_lattice_to_supervisor(0, rp.state_space)
        assert sup.mode == RoverMode.SAFE

    def test_phi_top_is_science(self):
        rp = build_minimal_rover()
        n = rp.num_states
        sup = phi_lattice_to_supervisor(n - 1, rp.state_space)
        assert sup.mode == RoverMode.NOMINAL_SCIENCE

    def test_psi_in_range(self):
        rp = build_minimal_rover()
        sup = SupervisorState(RoverMode.NOMINAL_SCIENCE, SolPhase.SLEEP,
                              frozenset(SensorPipeline))
        s = psi_supervisor_to_lattice(sup, rp.state_space)
        assert 0 <= s < rp.num_states

    def test_psi_safe_bottom(self):
        rp = build_minimal_rover()
        sup = SupervisorState(RoverMode.SAFE, SolPhase.WAKE, frozenset())
        s = psi_supervisor_to_lattice(sup, rp.state_space)
        assert s == 0

    def test_galois_pair(self):
        rp = build_minimal_rover()
        assert is_galois_pair(rp.state_space)


# ---------------------------------------------------------------------------
# Contingency triggers
# ---------------------------------------------------------------------------

class TestContingency:
    def test_table_nonempty(self):
        assert len(CONTINGENCY_TABLE) >= 5

    def test_find_battery_low(self):
        ct = find_contingency(RoverMode.NOMINAL_SCIENCE, "battery_low")
        assert ct is not None
        assert ct.to_mode == RoverMode.MIN_OPS

    def test_find_imu_fault(self):
        ct = find_contingency(RoverMode.NOMINAL_DRIVE, "imu_fault")
        assert ct is not None
        assert "reset_imu" in ct.safety_action

    def test_find_missing(self):
        assert find_contingency(RoverMode.SAFE, "nonexistent") is None

    def test_uncategorised_to_safe(self):
        ct = find_contingency(RoverMode.NOMINAL_SCIENCE, "uncategorised")
        assert ct is not None
        assert ct.to_mode == RoverMode.SAFE


# ---------------------------------------------------------------------------
# Ground link windows
# ---------------------------------------------------------------------------

class TestGroundLink:
    def test_windows_count(self):
        ws = ground_link_windows(7)
        assert len(ws) == 21  # 3 phases per sol

    def test_windows_phases(self):
        ws = ground_link_windows(2)
        phases = {p for _, p in ws}
        assert phases == {SolPhase.WAKE, SolPhase.COMM_WINDOW, SolPhase.SLEEP}

    def test_windows_sols(self):
        ws = ground_link_windows(3)
        sols = {s for s, _ in ws}
        assert sols == {0, 1, 2}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class TestVerification:
    def test_verify_minimal(self):
        rp = build_minimal_rover()
        v = verify_rover(rp)
        assert v["is_lattice"]
        assert v["has_safe_bottom"]
        assert v["bounded_states"]

    def test_verify_keys(self):
        rp = build_minimal_rover()
        v = verify_rover(rp)
        assert set(v.keys()) == {
            "is_lattice", "has_safe_bottom",
            "sensors_independent", "bounded_states",
        }
