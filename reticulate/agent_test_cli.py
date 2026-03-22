"""CLI for MCP/A2A conformance testing.

Usage:
    python -m reticulate.agent_test --protocol mcp --report
    python -m reticulate.agent_test --protocol a2a --test-gen
    python -m reticulate.agent_test --protocol custom --type "&{a: end}"
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for agent protocol conformance testing."""
    parser = argparse.ArgumentParser(
        prog="reticulate-agent-test",
        description="AI agent protocol conformance testing tool",
    )
    parser.add_argument(
        "--protocol",
        choices=["mcp", "a2a", "custom"],
        required=True,
        help="Protocol to analyze: mcp, a2a, or custom",
    )
    parser.add_argument(
        "--type",
        dest="type_string",
        help="Session type string (required with --protocol custom)",
    )
    parser.add_argument(
        "--name",
        default="CustomProtocol",
        help="Protocol name (used with --protocol custom)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run LIVE conformance tests against a running server",
    )
    parser.add_argument(
        "--target",
        help="Server URL for --live (e.g., http://localhost:3000)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print full conformance report",
    )
    parser.add_argument(
        "--test-gen",
        action="store_true",
        help="Print generated JUnit 5 test source",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Print coverage summary",
    )
    parser.add_argument(
        "--hasse",
        nargs="?",
        const="agent_hasse",
        help="Render Hasse diagram to PATH",
    )
    parser.add_argument(
        "--petri",
        action="store_true",
        help="Print Petri net summary",
    )
    parser.add_argument(
        "--invariants",
        action="store_true",
        help="Print algebraic invariants",
    )
    parser.add_argument(
        "--dot",
        action="store_true",
        help="Print DOT source to stdout",
    )
    parser.add_argument(
        "--fmt",
        choices=["png", "svg", "pdf", "dot"],
        default="svg",
        help="Output format for --hasse (default: svg)",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Java class name for test generation",
    )
    parser.add_argument(
        "--package",
        default=None,
        help="Java package for test generation",
    )

    args = parser.parse_args(argv)

    # Validate custom protocol
    if args.protocol == "custom" and not args.type_string:
        parser.error("--type is required with --protocol custom")

    # Late imports for speed
    from reticulate.mcp_conformance import (
        a2a_protocol,
        conformance_report,
        custom_protocol,
        format_report,
        generate_conformance_tests,
        mcp_protocol,
    )

    # Build protocol model
    try:
        if args.protocol == "mcp":
            protocol = mcp_protocol()
        elif args.protocol == "a2a":
            protocol = a2a_protocol()
        else:
            protocol = custom_protocol(args.name, args.type_string)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle --live mode
    if args.live:
        if not args.target:
            parser.error("--target is required with --live")
        from reticulate.mcp_runtime import ConformanceTester
        tester = ConformanceTester(
            target=args.target,
            protocol=args.protocol if args.protocol != "custom" else "mcp",
        )
        report = tester.run_all()
        print(report.summary())
        sys.exit(0 if report.failed_tests == 0 else 1)

    # Default to --report if no specific output requested
    if not any([args.report, args.test_gen, args.coverage,
                args.hasse, args.petri, args.invariants, args.dot]):
        args.report = True

    try:
        if args.report:
            report = conformance_report(protocol)
            print(format_report(report))

        elif args.test_gen:
            from reticulate.testgen import TestGenConfig
            config = TestGenConfig(
                class_name=args.class_name or f"{protocol.name}ProtocolTest",
                package_name=args.package or "com.agent.conformance",
            )
            source = generate_conformance_tests(protocol, config)
            print(source)

        elif args.coverage:
            from reticulate.testgen import TestGenConfig, enumerate as enum_tests
            from reticulate.coverage import compute_coverage
            ss = __import__("reticulate.statespace", fromlist=["build_statespace"]).build_statespace(protocol.ast)
            config = TestGenConfig(class_name=f"{protocol.name}Test")
            result = enum_tests(ss, config)
            cov = compute_coverage(ss, result=result)
            print(f"{protocol.name} Coverage:")
            print(f"  Transition coverage: {cov.transition_coverage:.1%}")
            print(f"  State coverage:      {cov.state_coverage:.1%}")
            print(f"  Valid paths:         {len(result.valid_paths)}")
            print(f"  Violations:          {len(result.violations)}")

        elif args.petri:
            from reticulate.statespace import build_statespace
            from reticulate.petri import session_type_to_petri_net
            ss = build_statespace(protocol.ast)
            result = session_type_to_petri_net(ss)
            print(f"{protocol.name} Petri Net:")
            print(f"  Places:      {result.num_places}")
            print(f"  Transitions: {result.num_transitions}")
            print(f"  Occurrence:  {'yes' if result.is_occurrence_net else 'no'}")
            print(f"  Free-choice: {'yes' if result.is_free_choice else 'no'}")
            print(f"  Isomorphic:  {'yes' if result.reachability_isomorphic else 'NO'}")

        elif args.invariants:
            from reticulate.statespace import build_statespace
            from reticulate.matrix import algebraic_invariants
            ss = build_statespace(protocol.ast)
            inv = algebraic_invariants(ss)
            print(f"{protocol.name} Algebraic Invariants:")
            print(f"  States:           {inv.num_states}")
            print(f"  Mobius mu(T,B):   {inv.mobius_value}")
            print(f"  Rota polynomial:  {inv.rota_polynomial}")
            print(f"  Spectral radius:  {inv.spectral_radius:.4f}")
            print(f"  Fiedler value:    {inv.fiedler_value:.4f}")
            print(f"  Tropical eigenval:{inv.tropical_eigenvalue:.1f}")
            print(f"  VN entropy:       {inv.von_neumann_entropy:.4f}")

        elif args.dot:
            from reticulate.statespace import build_statespace
            from reticulate.visualize import dot_source
            ss = build_statespace(protocol.ast)
            print(dot_source(ss))

        elif args.hasse:
            from reticulate.statespace import build_statespace
            from reticulate.visualize import render_hasse
            ss = build_statespace(protocol.ast)
            render_hasse(ss, args.hasse, fmt=args.fmt)
            print(f"Hasse diagram written to {args.hasse}.{args.fmt}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
