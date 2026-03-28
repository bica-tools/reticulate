"""Tests for quantum walks on session type lattices (Step 31c)."""

import math
import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.quantum_walk import (
    quantum_evolution,
    return_probability,
    top_to_bottom_probability,
    quantum_spread,
    quantum_mixing_estimate,
    analyze_quantum_walk,
)


def _build(s: str):
    return build_statespace(parse(s))


class TestEvolution:
    def test_end(self):
        """Single state: stays at start."""
        P = quantum_evolution(_build("end"), 1.0)
        assert len(P) == 1
        assert abs(P[0][0] - 1.0) < 0.01

    def test_probabilities_nonneg(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}", "(&{a: end} || &{b: end})"]:
            P = quantum_evolution(_build(typ), 1.0)
            for row in P:
                for val in row:
                    assert val >= -0.01

    def test_rows_sum_approx_one(self):
        """Each row of P should sum to ≈1 (unitary evolution)."""
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            P = quantum_evolution(_build(typ), 1.0)
            for row in P:
                assert abs(sum(row) - 1.0) < 0.1

    def test_at_t_zero(self):
        """At t=0, U=I, so P=I."""
        P = quantum_evolution(_build("&{a: end}"), 0.0)
        n = len(P)
        for i in range(n):
            for j in range(n):
                expected = 1.0 if i == j else 0.0
                assert abs(P[i][j] - expected) < 0.01


class TestReturnProbability:
    def test_end(self):
        assert abs(return_probability(_build("end")) - 1.0) < 0.01

    def test_positive(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            rp = return_probability(_build(typ))
            assert 0.0 <= rp <= 1.01


class TestTopToBottom:
    def test_end(self):
        """Top = bottom for end, so probability = 1."""
        assert abs(top_to_bottom_probability(_build("end")) - 1.0) < 0.01

    def test_range(self):
        for typ in ["&{a: end}", "(&{a: end} || &{b: end})"]:
            p = top_to_bottom_probability(_build(typ))
            assert 0.0 <= p <= 1.01


class TestSpread:
    def test_end(self):
        assert quantum_spread(_build("end")) < 0.01

    def test_nonneg(self):
        for typ in ["&{a: end}", "&{a: &{b: end}}"]:
            s = quantum_spread(_build(typ))
            assert s >= -0.01


class TestMixing:
    def test_end(self):
        assert quantum_mixing_estimate(_build("end")) == 0.0

    def test_positive(self):
        m = quantum_mixing_estimate(_build("&{a: end}"))
        assert m >= 0.0


class TestAnalyze:
    def test_end(self):
        r = analyze_quantum_walk(_build("end"))
        assert r.num_states == 1

    def test_chain(self):
        r = analyze_quantum_walk(_build("&{a: end}"))
        assert r.num_states == 2
        assert 0.0 <= r.return_probability <= 1.01

    def test_parallel(self):
        r = analyze_quantum_walk(_build("(&{a: end} || &{b: end})"))
        assert r.num_states == 4


class TestBenchmarks:
    BENCHMARKS = [
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Nested", "&{a: &{b: &{c: end}}}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_quantum_walk_runs(self, name, typ):
        ss = _build(typ)
        r = analyze_quantum_walk(ss)
        assert r.num_states == len(ss.states)
        assert 0.0 <= r.return_probability <= 1.01
        assert r.spread >= -0.01
