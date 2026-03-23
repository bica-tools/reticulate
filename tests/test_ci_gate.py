"""Tests for CI/CD gate module."""

import json
import pytest

from reticulate.ci_gate import ci_gate, GateConfig, GateResult, CheckResult


# ---------------------------------------------------------------------------
# Default config: lattice + termination + coverage
# ---------------------------------------------------------------------------

class TestDefaultGate:
    def test_simple_pass(self):
        result = ci_gate("&{a: end}")
        assert result.passed
        assert result.num_states == 2
        assert len(result.failures) == 0

    def test_branch_select_pass(self):
        result = ci_gate("&{a: +{x: end, y: end}}")
        assert result.passed

    def test_recursive_pass(self):
        result = ci_gate("rec X . &{a: X, b: end}")
        assert result.passed

    def test_parallel_pass(self):
        result = ci_gate("(end || end)")
        assert result.passed

    def test_mcp_protocol(self):
        mcp = ("&{initialize: (rec X . &{callTool: +{RESULT: X, ERROR: X}, "
               "listTools: X, shutdown: end} || "
               "rec Y . +{NOTIFICATION: Y, DONE: end})}")
        # MCP has 94% transition coverage due to parallel; relax threshold
        config = GateConfig(min_transition_coverage=0.9)
        result = ci_gate(mcp, config=config)
        assert result.passed

    def test_elapsed_time(self):
        result = ci_gate("&{a: end}")
        assert result.elapsed_ms >= 0

    def test_summary_format(self):
        result = ci_gate("&{a: end}")
        summary = result.summary()
        assert "SESSION TYPE CI GATE" in summary
        assert "PASSED" in summary
        assert "lattice" in summary


# ---------------------------------------------------------------------------
# Lattice gate
# ---------------------------------------------------------------------------

class TestLatticeGate:
    def test_lattice_passes(self):
        result = ci_gate("&{a: end, b: end}")
        lattice_check = [c for c in result.checks if c.name == "lattice"]
        assert len(lattice_check) == 1
        assert lattice_check[0].passed

    def test_lattice_disabled(self):
        config = GateConfig(require_lattice=False, require_termination=False,
                           require_coverage=False)
        result = ci_gate("&{a: end}", config=config)
        assert result.passed
        assert len(result.checks) == 0


# ---------------------------------------------------------------------------
# Termination gate
# ---------------------------------------------------------------------------

class TestTerminationGate:
    def test_terminating_passes(self):
        result = ci_gate("&{a: end}")
        term_check = [c for c in result.checks if c.name == "termination"]
        assert len(term_check) == 1
        assert term_check[0].passed

    def test_non_terminating_fails(self):
        # rec X . &{a: X} has no exit path
        config = GateConfig(require_coverage=False)
        result = ci_gate("rec X . &{a: X}", config=config)
        term_check = [c for c in result.checks if c.name == "termination"]
        assert len(term_check) == 1
        assert not term_check[0].passed
        assert not result.passed


# ---------------------------------------------------------------------------
# Distributivity gate
# ---------------------------------------------------------------------------

class TestDistributivityGate:
    def test_distributive_passes(self):
        config = GateConfig(require_distributive=True)
        result = ci_gate("&{a: end, b: end}", config=config)
        dist_check = [c for c in result.checks if c.name == "distributivity"]
        assert len(dist_check) == 1
        assert dist_check[0].passed

    def test_non_distributive_fails(self):
        # Morning commute is non-distributive
        commute = ("&{wakeUp: &{checkWeather: +{raining: &{takeUmbrella: "
                   "+{bus: &{arrive: end}, taxi: &{arrive: end}}}, "
                   "sunny: +{walk: &{arrive: end}, bike: &{arrive: end}}}}}")
        config = GateConfig(require_distributive=True)
        result = ci_gate(commute, config=config)
        dist_check = [c for c in result.checks if c.name == "distributivity"]
        assert len(dist_check) == 1
        assert not dist_check[0].passed

    def test_non_distributive_not_required(self):
        commute = ("&{wakeUp: &{checkWeather: +{raining: &{takeUmbrella: "
                   "+{bus: &{arrive: end}, taxi: &{arrive: end}}}, "
                   "sunny: +{walk: &{arrive: end}, bike: &{arrive: end}}}}}")
        config = GateConfig(require_distributive=False)
        result = ci_gate(commute, config=config)
        assert result.passed  # distributivity not required


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------

class TestCoverageGate:
    def test_full_coverage(self):
        result = ci_gate("&{a: end}")
        cov_checks = [c for c in result.checks if "coverage" in c.name]
        assert all(c.passed for c in cov_checks)

    def test_coverage_threshold(self):
        config = GateConfig(min_transition_coverage=0.5, min_state_coverage=0.5)
        result = ci_gate("&{a: end, b: end}", config=config)
        assert result.passed

    def test_coverage_disabled(self):
        config = GateConfig(require_coverage=False)
        result = ci_gate("&{a: end}", config=config)
        cov_checks = [c for c in result.checks if "coverage" in c.name]
        assert len(cov_checks) == 0


# ---------------------------------------------------------------------------
# Subtyping / backward compatibility
# ---------------------------------------------------------------------------

class TestSubtypingGate:
    def test_compatible_passes(self):
        # Adding a branch is subtyping-safe (more methods = subtype)
        old = "&{a: end}"
        new = "&{a: end, b: end}"
        config = GateConfig(require_subtype=True)
        result = ci_gate(new, baseline=old, config=config)
        compat = [c for c in result.checks if c.name == "backward-compatibility"]
        assert len(compat) == 1
        assert compat[0].passed

    def test_breaking_change_fails(self):
        # Removing a branch breaks backward compatibility
        old = "&{a: end, b: end}"
        new = "&{a: end}"
        config = GateConfig(require_subtype=True)
        result = ci_gate(new, baseline=old, config=config)
        compat = [c for c in result.checks if c.name == "backward-compatibility"]
        assert len(compat) == 1
        assert not compat[0].passed
        assert "BREAKING" in compat[0].message

    def test_identical_passes(self):
        t = "&{a: end, b: end}"
        config = GateConfig(require_subtype=True)
        result = ci_gate(t, baseline=t, config=config)
        compat = [c for c in result.checks if c.name == "backward-compatibility"]
        assert compat[0].passed


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------

class TestSizeLimits:
    def test_under_limit(self):
        config = GateConfig(max_states=100, max_transitions=100)
        result = ci_gate("&{a: end}", config=config)
        assert result.passed

    def test_over_state_limit(self):
        config = GateConfig(max_states=2)
        result = ci_gate("&{a: &{b: end}}", config=config)
        state_check = [c for c in result.checks if c.name == "state-count"]
        assert len(state_check) == 1
        assert not state_check[0].passed


# ---------------------------------------------------------------------------
# Reconvergence gate
# ---------------------------------------------------------------------------

class TestReconvergenceGate:
    def test_no_reconvergence_passes(self):
        config = GateConfig(require_no_reconvergence=True)
        result = ci_gate("&{a: end, b: end}", config=config)
        reconv = [c for c in result.checks if c.name == "reconvergence"]
        assert len(reconv) == 1
        assert reconv[0].passed

    def test_reconvergence_fails(self):
        # Customer support has shared subtrees
        support = ("&{openTicket: &{describeIssue: +{billing: &{agentReview: "
                   "+{resolve: end, escalate: end}}, technical: &{agentReview: "
                   "+{resolve: end, escalate: end}}}}}")
        config = GateConfig(require_no_reconvergence=True)
        result = ci_gate(support, config=config)
        reconv = [c for c in result.checks if c.name == "reconvergence"]
        assert len(reconv) == 1
        assert not reconv[0].passed


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_invalid_syntax(self):
        from reticulate.parser import ParseError
        with pytest.raises(ParseError):
            ci_gate("INVALID!!!")


# ---------------------------------------------------------------------------
# GateResult properties
# ---------------------------------------------------------------------------

class TestGateResultProperties:
    def test_failures_property(self):
        config = GateConfig(require_distributive=True)
        commute = ("&{wakeUp: &{checkWeather: +{raining: &{takeUmbrella: "
                   "+{bus: &{arrive: end}, taxi: &{arrive: end}}}, "
                   "sunny: +{walk: &{arrive: end}, bike: &{arrive: end}}}}}")
        result = ci_gate(commute, config=config)
        assert len(result.failures) >= 1
        assert all(not f.passed for f in result.failures)

    def test_warnings_property(self):
        result = ci_gate("&{a: end}")
        # No warnings expected for simple passing type
        assert isinstance(result.warnings, tuple)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_pass(self, capsys):
        from reticulate.ci_gate import main
        with pytest.raises(SystemExit) as exc_info:
            main(["&{a: end}"])
        assert exc_info.value.code == 0

    def test_cli_fail(self, capsys):
        from reticulate.ci_gate import main
        with pytest.raises(SystemExit) as exc_info:
            # Non-distributive: two branches reconverge at shared &{arrive: end}
            main(["&{start: +{a: &{x: &{arrive: end}, y: &{arrive: end}}, b: &{x: &{arrive: end}, y: &{arrive: end}}}}",
                  "--require-distributive"])
        assert exc_info.value.code == 1

    def test_cli_parse_error(self, capsys):
        from reticulate.ci_gate import main
        with pytest.raises(SystemExit) as exc_info:
            main(["INVALID!!!"])
        assert exc_info.value.code == 2

    def test_cli_json(self, capsys):
        from reticulate.ci_gate import main
        with pytest.raises(SystemExit) as exc_info:
            main(["&{a: end}", "--json"])
        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["passed"] is True
        assert "checks" in data

    def test_cli_baseline(self, capsys):
        from reticulate.ci_gate import main
        with pytest.raises(SystemExit) as exc_info:
            main(["&{a: end, b: end}", "--baseline", "&{a: end}"])
        assert exc_info.value.code == 0

    def test_cli_breaking_baseline(self, capsys):
        from reticulate.ci_gate import main
        with pytest.raises(SystemExit) as exc_info:
            main(["&{a: end}", "--baseline", "&{a: end, b: end}"])
        assert exc_info.value.code == 1
