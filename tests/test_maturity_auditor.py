"""Tests for maturity auditor agent."""

import pytest
from reticulate.maturity_auditor import (
    _check_toy_indicators,
    _assess_maturity,
    propose_upgrade,
    audit_step,
)


class TestToyIndicators:
    def test_few_tests(self):
        indicators = _check_toy_indicators("", "", "", num_tests=5, word_count=6000, benchmark_count=5)
        assert any("Only 5 tests" in i for i in indicators)

    def test_few_benchmarks(self):
        indicators = _check_toy_indicators("", "", "", num_tests=50, word_count=6000, benchmark_count=1)
        assert any("benchmark" in i.lower() for i in indicators)

    def test_short_paper(self):
        indicators = _check_toy_indicators("", "", "", num_tests=50, word_count=2000, benchmark_count=5)
        assert any("2000 words" in i for i in indicators)

    def test_good_step_few_indicators(self):
        indicators = _check_toy_indicators(
            "# complexity O(n^2)\nraise ValueError('error')",
            '_build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")\n' * 20 +
            "SMTP\nOAuth\ndef test_edge_case():\ndef test_empty():\ndef test_degenerate():",
            "",
            num_tests=50, word_count=6000, benchmark_count=6,
        )
        # Should have few indicators for a well-developed step
        assert len(indicators) <= 3


class TestMaturityLevel:
    def test_toy(self):
        assert _assess_maturity(["a", "b", "c", "d", "e"]) == "toy"

    def test_prototype(self):
        assert _assess_maturity(["a", "b", "c"]) == "prototype"

    def test_research(self):
        assert _assess_maturity(["a"]) == "research"

    def test_production(self):
        assert _assess_maturity([]) == "production"


class TestUpgradeProposal:
    def test_default(self):
        proposal, research, plan = propose_upgrade("unknown topic")
        assert len(proposal) > 0
        assert len(research) >= 3
        assert len(plan) >= 5

    def test_zeta(self):
        proposal, _, _ = propose_upgrade("zeta matrix")
        assert "benchmark" in proposal.lower() or "SMTP" in proposal

    def test_eigenvalues(self):
        proposal, _, _ = propose_upgrade("eigenvalues analysis")
        assert len(proposal) > 0


class TestAuditStep:
    def test_minimal(self):
        r = audit_step("30a", "Zeta Matrix", num_tests=72, word_count=5001, benchmark_count=3)
        assert r.step_number == "30a"
        assert r.maturity_level in ("toy", "prototype", "research", "production")
        assert len(r.execution_plan) >= 5

    def test_toy_step(self):
        r = audit_step("99", "Toy Example", num_tests=3, word_count=500, benchmark_count=0)
        assert r.maturity_level == "toy"
        assert len(r.toy_indicators) >= 3

    def test_good_step(self):
        r = audit_step(
            "30a", "Zeta Matrix",
            module_source="# O(n^2) complexity\nraise ValueError",
            test_source='_build("rec X . &{a: X}")\nSMTP\nOAuth\ndef test_empty():\ndef test_degenerate():',
            num_tests=72, word_count=5001, benchmark_count=6,
        )
        assert r.maturity_level in ("research", "production")
