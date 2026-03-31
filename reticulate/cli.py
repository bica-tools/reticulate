"""Command-line interface for the reticulate session type analyzer.

Usage::

    python -m reticulate.cli <type-string>
    python -m reticulate.cli --hasse <type-string>
    python -m reticulate.cli --dot <type-string>
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from reticulate.lattice import check_distributive, check_lattice, compute_meet, compute_join
from reticulate.parser import ParseError, parse, parse_program, pretty
from reticulate.resolve import ResolveError, resolve
from reticulate.statespace import StateSpace, build_statespace
from reticulate.visualize import dot_source, render_hasse

__version__ = "0.1.0"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Supports two modes:
      - ``reticulate modular <type>``  — modularity analysis (P104)
      - ``reticulate <type>``          — legacy lattice analysis
    """
    raw = argv if argv is not None else sys.argv[1:]

    # Dispatch: if first positional is "modular", use the modular subcommand
    if raw and raw[0] == "modular":
        _main_modular(raw[1:])
        return

    # Legacy interface
    parser = argparse.ArgumentParser(
        prog="reticulate",
        description="Session type analyzer \u2014 parse, build state space, check lattice properties.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "type_string",
        nargs="?",
        default=None,
        help='Session type string, e.g. "&{open: &{read: end, close: end}}"',
    )
    parser.add_argument(
        "-f", "--file",
        default=None,
        metavar="FILE",
        help="Read session type (or equation program) from FILE instead of command line",
    )
    parser.add_argument(
        "--hasse",
        nargs="?",
        const="hasse_output",
        default=None,
        metavar="PATH",
        help="Render Hasse diagram to PATH (default: hasse_output)",
    )
    parser.add_argument(
        "--fmt",
        default="png",
        choices=["png", "svg", "pdf", "dot"],
        help="Output format for --hasse (default: png)",
    )
    parser.add_argument(
        "--dot",
        action="store_true",
        help="Print DOT source to stdout (no graphviz needed)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide state labels on nodes (show IDs only)",
    )
    parser.add_argument(
        "--no-edge-labels",
        action="store_true",
        help="Hide transition labels on edges",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title for the Hasse diagram",
    )
    parser.add_argument(
        "--lattice",
        action="store_true",
        help="Pretty-print the lattice: states, transitions, meet/join tables",
    )
    parser.add_argument(
        "--test-gen",
        action="store_true",
        help="Generate Java test class source to stdout",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Java class name for --test-gen (default: MyProtocol)",
    )
    parser.add_argument(
        "--package",
        default=None,
        help="Java package name for --test-gen",
    )
    parser.add_argument(
        "--framework",
        default="junit",
        choices=["junit", "testng"],
        help="Test framework for --test-gen (default: junit)",
    )
    parser.add_argument(
        "--coverage-hasse",
        default=None,
        metavar="PATH",
        help="Render test coverage Hasse diagram to PATH (use with --test-gen)",
    )
    parser.add_argument(
        "--coverage-storyboard",
        default=None,
        metavar="FILE",
        help="Render coverage storyboard to FILE (HTML) — one frame per test (use with --test-gen)",
    )
    parser.add_argument(
        "--coverage-fmt",
        default="svg",
        choices=["png", "svg", "pdf"],
        help="Output format for coverage diagrams (default: svg)",
    )
    parser.add_argument(
        "--distributive",
        action="store_true",
        help="Check distributivity (Birkhoff classification)",
    )
    parser.add_argument(
        "--grid",
        nargs="?",
        const="grid_output",
        default=None,
        metavar="PATH",
        help="Render 2D grid layout for binary parallel types (default: grid_output)",
    )
    parser.add_argument(
        "--3d",
        nargs="?",
        const="product3d.html",
        default=None,
        dest="three_d",
        metavar="PATH",
        help="Render 3D interactive HTML for ternary parallel types (default: product3d.html)",
    )
    parser.add_argument(
        "--factored",
        nargs="?",
        const="factored_output",
        default=None,
        metavar="PATH",
        help="Render factored side-by-side view of parallel factors (default: factored_output)",
    )

    args = parser.parse_args(raw)

    # Determine source string
    if args.file is not None:
        try:
            source = Path(args.file).read_text()
        except (OSError, IOError) as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.type_string is not None:
        source = args.type_string
    else:
        parser.error("either type_string or --file is required")

    # Parse using parse_program (handles both bare expressions and equations)
    try:
        program = parse_program(source)
        ast = resolve(program)
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)
    except ResolveError as e:
        print(f"Resolve error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build state space
    try:
        ss = build_statespace(ast)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Check lattice properties
    result = check_lattice(ss)

    # Dispatch output mode
    if args.test_gen:
        from reticulate.coverage import compute_coverage
        from reticulate.testgen import TestGenConfig, enumerate as enumerate_paths, generate_test_source
        config = TestGenConfig(
            class_name=args.class_name or "MyProtocol",
            package_name=args.package,
            framework=args.framework,
        )
        print(generate_test_source(ss, config, args.type_string))

        # Compute and report test coverage
        enum_result = enumerate_paths(ss, config)
        coverage = compute_coverage(ss, result=enum_result)
        tc = coverage.transition_coverage * 100
        sc = coverage.state_coverage * 100
        n_covered = len(coverage.covered_transitions)
        n_total = n_covered + len(coverage.uncovered_transitions)
        s_covered = len(coverage.covered_states)
        s_total = s_covered + len(coverage.uncovered_states)
        print(f"\n// Test coverage: transitions {n_covered}/{n_total} ({tc:.0f}%), states {s_covered}/{s_total} ({sc:.0f}%)", file=sys.stderr)

        # Render coverage diagram if requested
        if args.coverage_hasse is not None:
            try:
                out = render_hasse(
                    ss,
                    args.coverage_hasse,
                    fmt=args.coverage_fmt,
                    result=result,
                    title=f"Test Coverage — {args.class_name or 'MyProtocol'}",
                    coverage=coverage,
                )
            except ImportError:
                print(
                    "Error: The 'graphviz' Python package is required for --coverage-hasse.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"Coverage diagram: {out}", file=sys.stderr)

        # Render storyboard if requested
        if args.coverage_storyboard is not None:
            from reticulate.coverage import coverage_storyboard, render_storyboard
            frames = coverage_storyboard(ss, enum_result)
            name = args.class_name or "MyProtocol"
            try:
                out = render_storyboard(
                    ss, frames, args.coverage_storyboard,
                    result=result, title=f"Coverage Storyboard — {name}",
                )
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            print(f"Storyboard ({len(frames) + 1} frames): {out}", file=sys.stderr)
        return

    if args.lattice:
        _print_lattice(ast, ss, result)
        return

    # Product visualization modes (check before --dot so --factored --dot works)
    if args.grid is not None:
        from reticulate.product_viz import render_grid_hasse
        try:
            out = render_grid_hasse(
                ss, args.grid, fmt=args.fmt, result=result,
                title=args.title, labels=not args.no_labels,
                edge_labels=not args.no_edge_labels,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ImportError:
            print(
                "Error: The 'graphviz' Python package is required for --grid.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Rendered grid to {out}")
        return

    if args.three_d is not None:
        from reticulate.product_viz import render_3d_product
        try:
            out = render_3d_product(
                ss, args.three_d, title=args.title,
                labels=not args.no_labels,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Rendered 3D view to {out}")
        return

    if args.factored is not None:
        from reticulate.product_viz import factored_dot_source, factored_dashboard
        if args.fmt == "dot" or args.dot:
            try:
                print(factored_dot_source(
                    ss, title=args.title,
                    labels=not args.no_labels,
                    edge_labels=not args.no_edge_labels,
                ))
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                out = factored_dashboard(
                    ss, args.factored, title=args.title,
                )
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            print(f"Rendered factored view to {out}")
        return

    if args.dot:
        print(dot_source(
            ss,
            result,
            title=args.title,
            labels=not args.no_labels,
            edge_labels=not args.no_edge_labels,
        ))
        return

    if args.hasse is not None:
        try:
            out = render_hasse(
                ss,
                args.hasse,
                fmt=args.fmt,
                result=result,
                title=args.title,
                labels=not args.no_labels,
                edge_labels=not args.no_edge_labels,
            )
        except ImportError:
            print(
                "Error: The 'graphviz' Python package is required for --hasse.\n"
                "Install it with: pip install graphviz\n"
                "You also need the Graphviz system package (dot binary).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Rendered to {out}")
        return

    # Default: text summary
    _print_summary(args.type_string, ast, ss, result, distributive=args.distributive)


def _main_modular(argv: list[str]) -> None:
    """Execute the ``reticulate modular`` subcommand."""
    from reticulate.modular_report import generate_report

    parser = argparse.ArgumentParser(
        prog="reticulate modular",
        description="Analyze protocol modularity: distributivity, coupling, Birkhoff decomposition.",
    )
    parser.add_argument(
        "type_string",
        nargs="?",
        default=None,
        help='Session type string, e.g. "&{open: &{read: end, close: end}}"',
    )
    parser.add_argument(
        "-f", "--file",
        default=None,
        metavar="FILE",
        help="Read session type from FILE",
    )
    parser.add_argument(
        "--name",
        default="Protocol",
        help="Protocol name for the report (default: Protocol)",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json", "dot"],
        dest="report_format",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--hasse",
        nargs="?",
        const="modular_hasse",
        default=None,
        metavar="PATH",
        help="Also render Hasse diagram to PATH",
    )
    parser.add_argument(
        "--fmt",
        default="png",
        choices=["png", "svg", "pdf"],
        help="Hasse diagram format (default: png)",
    )

    args = parser.parse_args(argv)

    # Determine source
    if args.file is not None:
        try:
            source = Path(args.file).read_text()
        except (OSError, IOError) as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.type_string is not None:
        source = args.type_string
    else:
        print("Error: either type_string or --file is required", file=sys.stderr)
        sys.exit(1)

    # Parse
    try:
        program = parse_program(source)
        ast = resolve(program)
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)
    except ResolveError as e:
        print(f"Resolve error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build state space
    try:
        ss = build_statespace(ast)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report = generate_report(source, ss, protocol_name=args.name)

    # Output
    if args.report_format == "json":
        print(report.to_json())
    elif args.report_format == "dot":
        print(report.to_dot())
    else:
        print(report.to_text())

    # Optional Hasse diagram
    if args.hasse is not None:
        try:
            out = render_hasse(
                ss,
                args.hasse,
                fmt=args.fmt,
                result=report.lattice_result,
                title=f"Modularity: {args.name}",
            )
        except ImportError:
            print(
                "Error: The 'graphviz' Python package is required for --hasse.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Hasse diagram: {out}", file=sys.stderr)


def _print_summary(
    type_string: str,
    ast: object,
    ss: object,
    result: object,
    *,
    distributive: bool = False,
) -> None:
    """Print structured text summary to stdout."""
    from reticulate.lattice import LatticeResult
    from reticulate.statespace import StateSpace

    assert isinstance(ss, StateSpace)
    assert isinstance(result, LatticeResult)

    num_states = len(ss.states)
    num_transitions = len(ss.transitions)
    num_sccs = result.num_scc

    check = "\u2713" if result.is_lattice else "\u2717"
    verdict = "IS a lattice" if result.is_lattice else "NOT a lattice"

    top_ok = "\u2713" if result.has_top else "\u2717"
    bot_ok = "\u2713" if result.has_bottom else "\u2717"
    meets_ok = "\u2713" if result.all_meets_exist else "\u2717"
    joins_ok = "\u2713" if result.all_joins_exist else "\u2717"

    print(f"Session type: {pretty(ast)}")
    print(f"States: {num_states}  |  Transitions: {num_transitions}  |  SCCs: {num_sccs}")
    print()
    print(f"Lattice check: {check} {verdict}")
    print(f"  Top reachable:    {top_ok}")
    print(f"  Bottom reachable: {bot_ok}")
    print(f"  All meets exist:  {meets_ok}")
    print(f"  All joins exist:  {joins_ok}")

    if result.counterexample is not None:
        a, b, kind = result.counterexample
        kind_str = "no meet" if kind == "no_meet" else "no join"
        print(f"  Counterexample: states {a} and {b} have {kind_str}")

    if distributive and result.is_lattice:
        dr = check_distributive(ss)
        label = dr.classification.replace("_", " ").title()
        print()
        print(f"Distributivity: {label}")
        if dr.has_n5 and dr.n5_witness is not None:
            print(f"  N\u2085 witness: {dr.n5_witness}")
        if dr.has_m3 and dr.m3_witness is not None:
            print(f"  M\u2083 witness: {dr.m3_witness}")


def _print_lattice(
    ast: object,
    ss: StateSpace,
    result: object,
) -> None:
    """Pretty-print the lattice: states, transitions, and meet/join tables."""
    from reticulate.lattice import LatticeResult

    assert isinstance(result, LatticeResult)

    print(f"Session type: {pretty(ast)}")
    print()

    # States sorted by ID, top first, bottom last
    states_sorted = sorted(ss.states)

    # --- States ---
    print(f"States ({len(states_sorted)}):")
    for s in states_sorted:
        label = ss.labels.get(s, "?")
        markers = []
        if s == ss.top:
            markers.append("\u22a4")
        if s == ss.bottom:
            markers.append("\u22a5")
        marker_str = f" ({', '.join(markers)})" if markers else ""
        print(f"  {s}: {label}{marker_str}")

    print()

    # --- Transitions ---
    print(f"Transitions ({len(ss.transitions)}):")
    for src, lbl, tgt in sorted(ss.transitions):
        kind = " [select]" if ss.is_selection(src, lbl, tgt) else ""
        print(f"  {src} \u2014{lbl}\u2192 {tgt}{kind}")

    print()

    # --- Lattice verdict ---
    check = "\u2713" if result.is_lattice else "\u2717"
    verdict = "IS a lattice" if result.is_lattice else "NOT a lattice"
    print(f"Lattice: {check} {verdict}")

    if not result.is_lattice:
        if result.counterexample is not None:
            a, b, kind = result.counterexample
            kind_str = "no meet" if kind == "no_meet" else "no join"
            print(f"  Counterexample: states {a} and {b} have {kind_str}")
        print()
        return

    print()

    # --- Meet table ---
    # Use SCC representatives for display
    reps = sorted({result.scc_map[s] for s in ss.states})
    col_width = max(len(str(s)) for s in reps) + 1

    print(f"Meet table (\u2293):")
    header = " " * (col_width + 1) + "".join(str(s).rjust(col_width) for s in reps)
    print(header)
    for a in reps:
        row = str(a).rjust(col_width) + " "
        for b in reps:
            m = compute_meet(ss, a, b)
            cell = str(m) if m is not None else "-"
            row += cell.rjust(col_width)
        print(row)

    print()

    # --- Join table ---
    print(f"Join table (\u2294):")
    header = " " * (col_width + 1) + "".join(str(s).rjust(col_width) for s in reps)
    print(header)
    for a in reps:
        row = str(a).rjust(col_width) + " "
        for b in reps:
            j = compute_join(ss, a, b)
            cell = str(j) if j is not None else "-"
            row += cell.rjust(col_width)
        print(row)


if __name__ == "__main__":
    main()
