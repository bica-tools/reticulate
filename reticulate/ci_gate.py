"""CI/CD gate for session type protocol verification.

Runs a configurable battery of checks against a session type and returns
a pass/fail verdict suitable for CI pipelines. Optionally compares against
a baseline type for backward-compatibility (subtyping) checks.

Usage (Python):
    from reticulate.ci_gate import ci_gate, GateConfig
    result = ci_gate("&{init: rec X . &{call: +{OK: X}, quit: end}}")
    if not result.passed:
        for f in result.failures:
            print(f"FAIL: {f}")
        sys.exit(1)

Usage (CLI):
    python -m reticulate.ci_gate "TYPE_STRING"
    python -m reticulate.ci_gate "NEW_TYPE" --baseline "OLD_TYPE"
    python -m reticulate.ci_gate -f protocol.st --min-coverage 0.9

Exit codes:
    0 — all gates passed
    1 — one or more gates failed
    2 — parse/resolve error
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateConfig:
    """Configuration for CI gate checks.

    Each ``require_*`` flag enables a gate. Thresholds control pass/fail.
    """
    # Structural checks (fast, always safe to run)
    require_lattice: bool = True
    require_distributive: bool = False
    require_termination: bool = True
    require_realizable: bool = False

    # Coverage thresholds (0.0–1.0)
    require_coverage: bool = True
    min_transition_coverage: float = 1.0
    min_state_coverage: float = 1.0

    # Backward compatibility (requires baseline)
    require_subtype: bool = False

    # Compression / reconvergence
    require_no_reconvergence: bool = False
    max_reconvergence_degree: int = 0

    # Limits
    max_states: int = 0          # 0 = no limit
    max_transitions: int = 0     # 0 = no limit


# ---------------------------------------------------------------------------
# Individual check results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckResult:
    """Result of a single gate check."""
    name: str
    passed: bool
    severity: str  # "critical" | "warning" | "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregate gate result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """Aggregate result of all CI gate checks."""
    passed: bool
    type_string: str
    baseline_string: str | None
    checks: tuple[CheckResult, ...]
    elapsed_ms: float
    num_states: int
    num_transitions: int

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.passed and c.severity == "warning")

    def summary(self) -> str:
        """Human-readable summary for CI output."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  SESSION TYPE CI GATE")
        lines.append("=" * 60)
        lines.append("")

        for c in self.checks:
            icon = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{icon}] {c.name}: {c.message}")

        lines.append("")
        lines.append("-" * 60)
        verdict = "PASSED" if self.passed else "FAILED"
        lines.append(f"  Verdict:  {verdict}")
        lines.append(f"  States:   {self.num_states}")
        lines.append(f"  Trans:    {self.num_transitions}")
        lines.append(f"  Checks:   {len(self.checks)} "
                     f"({len(self.failures)} failed)")
        lines.append(f"  Time:     {self.elapsed_ms:.1f}ms")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core: ci_gate
# ---------------------------------------------------------------------------

def ci_gate(
    type_string: str,
    *,
    baseline: str | None = None,
    config: GateConfig | None = None,
) -> GateResult:
    """Run CI gate checks on a session type.

    Args:
        type_string: The session type to check.
        baseline: Optional previous version for subtyping check.
        config: Gate configuration. Defaults to GateConfig().

    Returns:
        GateResult with pass/fail verdict and per-check details.

    Raises:
        ParseError: If type_string or baseline cannot be parsed.
    """
    from reticulate.parser import parse, pretty
    from reticulate.statespace import build_statespace
    from reticulate.lattice import check_lattice, check_distributive
    from reticulate.termination import check_termination
    from reticulate.testgen import TestGenConfig, enumerate as enum_tests
    from reticulate.coverage import compute_coverage
    from reticulate.compress import analyze_compression

    if config is None:
        config = GateConfig()

    t0 = time.time()
    checks: list[CheckResult] = []

    # Parse
    ast = parse(type_string)
    ss = build_statespace(ast)

    # ── Size limits ──
    if config.max_states > 0:
        ok = len(ss.states) <= config.max_states
        checks.append(CheckResult(
            name="state-count",
            passed=ok,
            severity="critical" if not ok else "info",
            message=f"{len(ss.states)} states (max {config.max_states})",
            details={"states": len(ss.states), "max": config.max_states},
        ))

    if config.max_transitions > 0:
        ok = len(ss.transitions) <= config.max_transitions
        checks.append(CheckResult(
            name="transition-count",
            passed=ok,
            severity="critical" if not ok else "info",
            message=f"{len(ss.transitions)} transitions (max {config.max_transitions})",
            details={"transitions": len(ss.transitions), "max": config.max_transitions},
        ))

    # ── Lattice check ──
    if config.require_lattice:
        try:
            lr = check_lattice(ss)
            msg = "forms a lattice" if lr.is_lattice else "NOT a lattice"
            if not lr.is_lattice and lr.counterexample:
                a, b, kind = lr.counterexample
                msg += f" (states {a},{b} have no {kind})"
            checks.append(CheckResult(
                name="lattice",
                passed=lr.is_lattice,
                severity="critical",
                message=msg,
                details={"is_lattice": lr.is_lattice, "sccs": lr.num_scc},
            ))
        except (KeyError, ValueError):
            checks.append(CheckResult(
                name="lattice",
                passed=False,
                severity="critical",
                message="degenerate state space (no bottom or unreachable states)",
                details={"is_lattice": False},
            ))

    # ── Distributivity ──
    if config.require_distributive:
        try:
            dr = check_distributive(ss)
            msg = f"{dr.classification}"
            if not dr.is_distributive and dr.has_m3:
                msg += " (contains M₃ diamond)"
            if not dr.is_distributive and dr.has_n5:
                msg += " (contains N₅ pentagon)"
            checks.append(CheckResult(
                name="distributivity",
                passed=dr.is_distributive,
                severity="warning",
                message=msg,
                details={
                    "is_distributive": dr.is_distributive,
                    "classification": dr.classification,
                },
            ))
        except (KeyError, ValueError):
            checks.append(CheckResult(
                name="distributivity",
                passed=False,
                severity="warning",
                message="degenerate state space",
                details={"is_distributive": False},
            ))

    # ── Termination ──
    if config.require_termination:
        tr = check_termination(ast)
        msg = "terminating" if tr.is_terminating else f"non-terminating (vars: {', '.join(tr.non_terminating_vars)})"
        checks.append(CheckResult(
            name="termination",
            passed=tr.is_terminating,
            severity="critical",
            message=msg,
            details={"is_terminating": tr.is_terminating},
        ))

    # ── Realizability ──
    if config.require_realizable:
        from reticulate.realizability import check_realizability
        rr = check_realizability(ss)
        msg = "realizable" if rr.is_realizable else f"not realizable ({len(rr.obstructions)} obstructions)"
        checks.append(CheckResult(
            name="realizability",
            passed=rr.is_realizable,
            severity="critical",
            message=msg,
            details={"is_realizable": rr.is_realizable},
        ))

    # ── Coverage ──
    if config.require_coverage:
        tc = TestGenConfig(class_name="CIGate", max_paths=200)
        enum_result = enum_tests(ss, tc)
        cov = compute_coverage(ss, result=enum_result)

        t_ok = cov.transition_coverage >= config.min_transition_coverage
        s_ok = cov.state_coverage >= config.min_state_coverage

        checks.append(CheckResult(
            name="transition-coverage",
            passed=t_ok,
            severity="critical" if not t_ok else "info",
            message=f"{cov.transition_coverage:.1%} (min {config.min_transition_coverage:.1%})",
            details={
                "coverage": cov.transition_coverage,
                "min": config.min_transition_coverage,
                "uncovered": len(cov.uncovered_transitions),
            },
        ))
        checks.append(CheckResult(
            name="state-coverage",
            passed=s_ok,
            severity="critical" if not s_ok else "info",
            message=f"{cov.state_coverage:.1%} (min {config.min_state_coverage:.1%})",
            details={
                "coverage": cov.state_coverage,
                "min": config.min_state_coverage,
                "uncovered": len(cov.uncovered_states),
            },
        ))

    # ── Reconvergence / compression ──
    if config.require_no_reconvergence:
        cr = analyze_compression(ast, min_size=2)
        ok = cr.num_shared <= config.max_reconvergence_degree
        msg = f"{cr.num_shared} shared subtrees (max {config.max_reconvergence_degree})"
        if cr.num_shared > 0:
            msg += f", ratio {cr.ratio:.2f}x"
        checks.append(CheckResult(
            name="reconvergence",
            passed=ok,
            severity="warning",
            message=msg,
            details={
                "num_shared": cr.num_shared,
                "ratio": cr.ratio,
                "max": config.max_reconvergence_degree,
            },
        ))

    # ── Backward compatibility (subtyping) ──
    if config.require_subtype and baseline is not None:
        from reticulate.subtyping import check_subtype
        baseline_ast = parse(baseline)
        sr = check_subtype(ast, baseline_ast)
        msg = "backward compatible" if sr.is_subtype else f"BREAKING CHANGE: {sr.reason}"
        checks.append(CheckResult(
            name="backward-compatibility",
            passed=sr.is_subtype,
            severity="critical",
            message=msg,
            details={"is_subtype": sr.is_subtype, "reason": sr.reason},
        ))

    elapsed = (time.time() - t0) * 1000
    all_passed = all(c.passed for c in checks)

    return GateResult(
        passed=all_passed,
        type_string=type_string,
        baseline_string=baseline,
        checks=tuple(checks),
        elapsed_ms=elapsed,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the CI gate."""
    parser = argparse.ArgumentParser(
        prog="reticulate.ci_gate",
        description="CI/CD gate for session type protocol verification",
    )
    parser.add_argument("type_string", nargs="?", help="Session type to check")
    parser.add_argument("-f", "--file", help="Read session type from file")
    parser.add_argument("--baseline", help="Previous version for backward compatibility")
    parser.add_argument("--baseline-file", help="Read baseline from file")

    # Gate flags
    parser.add_argument("--require-distributive", action="store_true",
                       help="Require distributive lattice")
    parser.add_argument("--require-realizable", action="store_true",
                       help="Require realizability")
    parser.add_argument("--require-no-reconvergence", action="store_true",
                       help="Require no shared subtrees (reconvergence)")
    parser.add_argument("--require-subtype", action="store_true",
                       help="Require backward compatibility with baseline")

    # Thresholds
    parser.add_argument("--min-transition-coverage", type=float, default=1.0,
                       help="Minimum transition coverage (0.0-1.0, default 1.0)")
    parser.add_argument("--min-state-coverage", type=float, default=1.0,
                       help="Minimum state coverage (0.0-1.0, default 1.0)")
    parser.add_argument("--max-states", type=int, default=0,
                       help="Maximum allowed states (0 = no limit)")
    parser.add_argument("--max-transitions", type=int, default=0,
                       help="Maximum allowed transitions (0 = no limit)")

    # Output
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only output on failure")

    args = parser.parse_args(argv)

    # Read type string
    if args.file:
        with open(args.file) as f:
            type_string = f.read().strip()
    elif args.type_string:
        type_string = args.type_string
    else:
        parser.error("Provide a session type string or --file")
        return

    # Read baseline
    baseline = None
    if args.baseline_file:
        with open(args.baseline_file) as f:
            baseline = f.read().strip()
    elif args.baseline:
        baseline = args.baseline

    # Build config
    config = GateConfig(
        require_distributive=args.require_distributive,
        require_realizable=args.require_realizable,
        require_no_reconvergence=args.require_no_reconvergence,
        require_subtype=args.require_subtype or baseline is not None,
        min_transition_coverage=args.min_transition_coverage,
        min_state_coverage=args.min_state_coverage,
        max_states=args.max_states,
        max_transitions=args.max_transitions,
    )

    # Run gate
    from reticulate.parser import ParseError
    from reticulate.resolve import ResolveError

    try:
        result = ci_gate(type_string, baseline=baseline, config=config)
    except (ParseError, ResolveError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Output
    if args.json:
        import json
        output = {
            "passed": result.passed,
            "states": result.num_states,
            "transitions": result.num_transitions,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in result.checks
            ],
        }
        print(json.dumps(output, indent=2))
    elif not args.quiet or not result.passed:
        print(result.summary())

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
