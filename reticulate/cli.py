"""Command-line interface for the reticulate session type analyzer.

Usage::

    python -m reticulate.cli <type-string>
    python -m reticulate.cli --hasse <type-string>
    python -m reticulate.cli --dot <type-string>
"""

from __future__ import annotations

import argparse
import sys

from reticulate.lattice import check_lattice
from reticulate.parser import ParseError, parse, pretty
from reticulate.statespace import build_statespace
from reticulate.visualize import dot_source, render_hasse


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="reticulate",
        description="Session type analyzer \u2014 parse, build state space, check lattice properties.",
    )
    parser.add_argument(
        "type_string",
        help='Session type string, e.g. "&{m: a.end, n: b.end}"',
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

    args = parser.parse_args(argv)

    # Parse the session type
    try:
        ast = parse(args.type_string)
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
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
    _print_summary(args.type_string, ast, ss, result)


def _print_summary(
    type_string: str,
    ast: object,
    ss: object,
    result: object,
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


if __name__ == "__main__":
    main()
