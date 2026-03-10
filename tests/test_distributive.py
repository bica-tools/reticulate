"""Tests for distributivity checking (Step 6a: Birkhoff-inspired).

Tests the check_distributive function on all 39 benchmarks and on
hand-crafted lattices known to be distributive or non-distributive.
"""

from __future__ import annotations

import pytest

from reticulate.lattice import check_distributive, check_lattice
from reticulate.parser import parse
from reticulate.statespace import build_statespace

from tests.benchmarks.protocols import BENCHMARKS


# ---------------------------------------------------------------------------
# Test: all 34 benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarkDistributivity:
    """Run distributivity check on all 37 benchmark protocols."""

    @pytest.mark.parametrize(
        "bench",
        BENCHMARKS,
        ids=[b.name for b in BENCHMARKS],
    )
    def test_benchmark_is_lattice(self, bench):
        """Verify each benchmark is a lattice (prerequisite)."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, f"{bench.name}: not a lattice ({lr.counterexample})"

    # Benchmarks empirically confirmed as non-distributive (contain N₅):
    NON_DISTRIBUTIVE = {
        "Two-Buyer",              # parallel, has N₅
        "Reticulate Pipeline",    # parallel, has N₅
        "TLS Handshake",          # no parallel, has N₅
        "Saga Orchestrator",      # parallel, has N₅
        "Two-Phase Commit",       # no parallel, has N₅
        "Quantum Measurement",    # no parallel, has N₅ — non-commutativity of observables
        "Ki3 Onboarding",         # parallel, has N₅ — selection+parallel interaction
        "Ki3 CI/CD Pipeline",     # parallel, has N₅ — selection+parallel interaction
    }

    @pytest.mark.parametrize(
        "bench",
        BENCHMARKS,
        ids=[b.name for b in BENCHMARKS],
    )
    def test_benchmark_distributivity(self, bench):
        """Test distributivity for each benchmark."""
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        dr = check_distributive(ss)
        assert dr.is_lattice, f"{bench.name}: not a lattice"
        # Report classification
        print(
            f"  {bench.name}: {dr.classification} "
            f"(states={len(ss.states)}, SCCs={dr.lattice_result.num_scc})"
        )
        if bench.name in self.NON_DISTRIBUTIVE:
            assert not dr.is_distributive, (
                f"{bench.name}: expected non-distributive but got {dr.classification}"
            )
            assert dr.has_n5, f"{bench.name}: expected N₅ witness"
        else:
            assert dr.is_distributive, (
                f"{bench.name}: NOT distributive! "
                f"has_m3={dr.has_m3} (witness={dr.m3_witness}), "
                f"has_n5={dr.has_n5} (witness={dr.n5_witness})"
            )


# ---------------------------------------------------------------------------
# Test: hand-crafted examples
# ---------------------------------------------------------------------------

class TestSimpleCases:
    """Test distributivity on simple session types."""

    def test_end(self):
        """end is trivially distributive."""
        ss = build_statespace(parse("end"))
        dr = check_distributive(ss)
        assert dr.is_distributive
        assert dr.classification == "boolean"

    def test_single_method(self):
        """Single method: a.end is distributive (2-element chain)."""
        ss = build_statespace(parse("&{a: end}"))
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_branch_two(self):
        """Branch with two methods: &{a: end, b: end} is distributive."""
        ss = build_statespace(parse("&{a: end, b: end}"))
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_select_two(self):
        """Selection with two choices: +{a: end, b: end} is distributive."""
        ss = build_statespace(parse("+{a: end, b: end}"))
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_simple_parallel(self):
        """Parallel: (&{a: end} || &{b: end}) is distributive."""
        ss = build_statespace(parse("(&{a: end} || &{b: end})"))
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_nested_branch(self):
        """Nested branch: &{a: &{c: end, d: end}, b: end} is distributive."""
        ss = build_statespace(parse("&{a: &{c: end, d: end}, b: end}"))
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_recursive_iterator(self):
        """rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}} is distributive."""
        ss = build_statespace(parse(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ))
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_parallel_with_continuation(self):
        """(&{a: end} || &{b: end}) . &{c: end} is distributive."""
        ss = build_statespace(parse(
            "(&{a: end} || &{b: end}) . &{c: end}"
        ))
        dr = check_distributive(ss)
        assert dr.is_distributive


class TestClassification:
    """Test the lattice hierarchy classification."""

    def test_two_element_is_boolean(self):
        """2-element chain (top, bottom) is Boolean."""
        ss = build_statespace(parse("&{a: end}"))
        dr = check_distributive(ss)
        assert dr.classification == "boolean"

    def test_end_is_boolean(self):
        """Single element (end) is Boolean."""
        ss = build_statespace(parse("end"))
        dr = check_distributive(ss)
        assert dr.classification == "boolean"

    def test_diamond_parallel_classification(self):
        """2x2 product lattice from parallel is distributive (or boolean)."""
        ss = build_statespace(parse("(&{a: end} || &{b: end})"))
        dr = check_distributive(ss)
        assert dr.is_distributive


class TestModularity:
    """Test modular lattice detection."""

    # 7 benchmarks are known to be non-modular (contain N₅)
    NON_MODULAR = {
        "Two-Buyer", "Reticulate Pipeline", "TLS Handshake",
        "Saga Orchestrator", "Two-Phase Commit", "Quantum Measurement",
        "Ki3 Onboarding",
        "Ki3 CI/CD Pipeline",
    }

    def test_modular_benchmarks(self):
        """32/40 benchmarks should be modular (no N₅)."""
        for bench in BENCHMARKS:
            ast = parse(bench.type_string)
            ss = build_statespace(ast)
            dr = check_distributive(ss)
            if bench.name in self.NON_MODULAR:
                assert not dr.is_modular, (
                    f"{bench.name}: expected non-modular"
                )
            else:
                assert dr.is_modular, (
                    f"{bench.name}: not modular! has_n5={dr.has_n5} "
                    f"(witness={dr.n5_witness})"
                )


class TestSummaryReport:
    """Generate a summary report of distributivity across all benchmarks."""

    def test_print_summary(self):
        """Print classification summary for all 39 benchmarks."""
        results = []
        for bench in BENCHMARKS:
            ast = parse(bench.type_string)
            ss = build_statespace(ast)
            dr = check_distributive(ss)
            results.append((bench.name, dr, len(ss.states),
                            dr.lattice_result.num_scc, bench.uses_parallel))

        print("\n" + "=" * 80)
        print("DISTRIBUTIVITY CONJECTURE — BENCHMARK RESULTS")
        print("=" * 80)
        print(f"{'Protocol':<30} {'Class':<15} {'States':>6} {'SCCs':>5} {'∥':>3}")
        print("-" * 80)

        n_dist = n_mod = n_bool = n_lat = 0
        for name, dr, states, sccs, par in results:
            tag = "Y" if par else ""
            print(f"{name:<30} {dr.classification:<15} {states:>6} {sccs:>5} {tag:>3}")
            if dr.classification == "boolean":
                n_bool += 1
            elif dr.classification == "distributive":
                n_dist += 1
            elif dr.classification == "modular":
                n_mod += 1
            else:
                n_lat += 1

        print("-" * 80)
        total = len(results)
        print(f"Boolean: {n_bool}/{total}, Distributive: {n_dist}/{total}, "
              f"Modular: {n_mod}/{total}, Lattice only: {n_lat}/{total}")
        print(f"CONJECTURE {'CONFIRMED' if n_lat == 0 and n_mod == 0 else 'REFUTED'}: "
              f"all benchmarks distributive = {n_lat == 0 and n_mod == 0}")
        print("=" * 80)

        # Empirical result: 32/40 distributive, 8/40 non-modular (N₅)
        assert n_bool + n_dist == 32, (
            f"Expected 32 distributive, got {n_bool + n_dist}"
        )
        assert n_lat == 8, f"Expected 8 lattice-only, got {n_lat}"
