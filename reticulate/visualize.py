"""Visualization of session type state spaces as Hasse diagrams.

Generates Graphviz DOT representations of state-space transition systems.
Supports counterexample highlighting and optional label/edge-label toggling.
All states and transitions (including recursion cycles) are shown explicitly.

The ``dot_source()`` function uses only the standard library and is always
available.  ``hasse_diagram()`` and ``render_hasse()`` require the ``graphviz``
Python package (lazy-imported).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.coverage import CoverageResult
    from reticulate.lattice import LatticeResult
    from reticulate.statespace import StateSpace

_MAX_LABEL_LEN = 40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dot_source(
    ss: StateSpace,
    result: LatticeResult | None = None,
    *,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
    coverage: CoverageResult | None = None,
) -> str:
    """Return DOT source string for the Hasse diagram of *ss*.

    No external dependencies — always available.
    """
    return _build_dot(ss, result, title=title, labels=labels, edge_labels=edge_labels, coverage=coverage)


def hasse_diagram(
    ss: StateSpace,
    result: LatticeResult | None = None,
    *,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
    coverage: CoverageResult | None = None,
) -> "graphviz.Digraph":  # type: ignore[name-defined]
    """Build a Graphviz ``Digraph`` for the Hasse diagram of *ss*.

    If *result* is provided, highlights counterexample pairs.

    Raises ``ImportError`` if the ``graphviz`` Python package is not installed.
    """
    try:
        import graphviz  # noqa: F811
    except ImportError:
        raise ImportError(
            "The 'graphviz' Python package is required for hasse_diagram(). "
            "Install it with: pip install graphviz"
        ) from None

    src = _build_dot(ss, result, title=title, labels=labels, edge_labels=edge_labels, coverage=coverage)
    return graphviz.Source(src)


def render_hasse(
    ss: StateSpace,
    path: str,
    *,
    fmt: str = "png",
    result: LatticeResult | None = None,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
    coverage: CoverageResult | None = None,
) -> str:
    """Render Hasse diagram to file.  Returns the output file path."""
    try:
        import graphviz  # noqa: F811
    except ImportError:
        raise ImportError(
            "The 'graphviz' Python package is required for render_hasse(). "
            "Install it with: pip install graphviz"
        ) from None

    src = _build_dot(ss, result, title=title, labels=labels, edge_labels=edge_labels, coverage=coverage)
    g = graphviz.Source(src)
    return g.render(filename=path, format=fmt, cleanup=True)


# ---------------------------------------------------------------------------
# Internal: DOT generation
# ---------------------------------------------------------------------------

def _truncate(label: str) -> str:
    if len(label) <= _MAX_LABEL_LEN:
        return label
    return label[:_MAX_LABEL_LEN] + "\u2026"


def _escape_dot(s: str) -> str:
    """Escape a string for use inside DOT double-quoted strings."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _build_dot(
    ss: StateSpace,
    result: LatticeResult | None,
    *,
    title: str | None,
    labels: bool,
    edge_labels: bool,
    coverage: CoverageResult | None = None,
) -> str:
    lines: list[str] = []
    lines.append("digraph {")
    lines.append("    rankdir=TB;")
    lines.append('    node [shape=box, style="filled,rounded", fontname="Helvetica"];')
    lines.append('    edge [fontname="Helvetica", fontsize=10];')

    if title is not None:
        lines.append(f'    label="{_escape_dot(title)}";')
        lines.append("    labelloc=t;")
        lines.append("    fontsize=14;")

    # Determine counterexample states (original IDs)
    counter_states: set[int] = set()
    if result is not None and result.counterexample is not None:
        counter_states.add(result.counterexample[0])
        counter_states.add(result.counterexample[1])

    if result is not None:
        _build_dot_with_result(ss, result, lines, labels, edge_labels, counter_states, coverage)
    else:
        _build_dot_plain(ss, lines, labels, edge_labels, coverage)

    # Pin top and bottom to the top and bottom of the layout
    lines.append(f'    {{rank=source; {ss.top}}}')
    lines.append(f'    {{rank=sink; {ss.bottom}}}')

    lines.append("}")
    return "\n".join(lines)


def _build_dot_plain(
    ss: StateSpace,
    lines: list[str],
    labels: bool,
    edge_labels: bool,
    coverage: CoverageResult | None = None,
) -> None:
    """Add nodes/edges without SCC collapsing."""
    for sid in sorted(ss.states):
        node_label = _node_label(ss, sid, labels)
        attrs = _node_attrs(ss, sid, node_label)
        if coverage is not None and sid in coverage.uncovered_states and sid not in (ss.top, ss.bottom):
            attrs["fillcolor"] = "#fee2e2"
        lines.append(f'    {sid} [{_fmt_attrs(attrs)}];')

    for src, lbl, tgt in ss.transitions:
        edge_attrs: dict[str, str] = {}
        if edge_labels:
            edge_attrs["label"] = lbl
        _apply_coverage_edge_attrs(edge_attrs, coverage, src, lbl, tgt)
        lines.append(f'    {src} -> {tgt}{_fmt_edge_attrs(edge_attrs)};')


def _build_dot_with_result(
    ss: StateSpace,
    result: LatticeResult,
    lines: list[str],
    labels: bool,
    edge_labels: bool,
    counter_states: set[int],
    coverage: CoverageResult | None = None,
) -> None:
    """Add all nodes/edges with counterexample highlighting."""
    for sid in sorted(ss.states):
        node_label = _node_label(ss, sid, labels)
        attrs = _node_attrs(ss, sid, node_label)

        # Counterexample highlight
        if sid in counter_states:
            attrs["color"] = "red"
            attrs["penwidth"] = "2"

        if coverage is not None and sid in coverage.uncovered_states and sid not in (ss.top, ss.bottom):
            attrs["fillcolor"] = "#fee2e2"

        lines.append(f'    {sid} [{_fmt_attrs(attrs)}];')

    for src, lbl, tgt in ss.transitions:
        edge_attrs: dict[str, str] = {}
        if edge_labels:
            edge_attrs["label"] = lbl
        _apply_coverage_edge_attrs(edge_attrs, coverage, src, lbl, tgt)
        lines.append(f'    {src} -> {tgt}{_fmt_edge_attrs(edge_attrs)};')


# ---------------------------------------------------------------------------
# Internal: node helpers
# ---------------------------------------------------------------------------

def _node_label(ss: StateSpace, sid: int, labels: bool) -> str:
    """Compute the display label for a node."""
    if not labels:
        raw = str(sid)
    else:
        raw = ss.labels.get(sid, str(sid))

    if sid == ss.top:
        raw = f"\u22a4 {raw}"
    elif sid == ss.bottom:
        raw = f"\u22a5 {raw}"

    return _truncate(raw)


def _node_attrs(ss: StateSpace, sid: int, label: str) -> dict[str, str]:
    """Build attribute dict for a node."""
    attrs: dict[str, str] = {"label": label}

    if sid == ss.top:
        attrs["fillcolor"] = "#bfdbfe"
    elif sid == ss.bottom:
        attrs["fillcolor"] = "#bbf7d0"
    else:
        attrs["fillcolor"] = "#f8fafc"

    return attrs


def _fmt_attrs(attrs: dict[str, str]) -> str:
    """Format attributes for a DOT node."""
    return ", ".join(f'{k}="{_escape_dot(v)}"' for k, v in attrs.items())


def _fmt_edge_attrs(attrs: dict[str, str]) -> str:
    """Format attributes for a DOT edge (returns empty string if no attrs)."""
    if not attrs:
        return ""
    inner = ", ".join(f'{k}="{_escape_dot(v)}"' for k, v in attrs.items())
    return f" [{inner}]"


def _apply_coverage_edge_attrs(
    edge_attrs: dict[str, str],
    coverage: CoverageResult | None,
    src: int,
    lbl: str,
    tgt: int,
) -> None:
    """Merge coverage coloring into *edge_attrs* in-place."""
    if coverage is None:
        return
    trans = (src, lbl, tgt)
    if trans in coverage.covered_transitions:
        edge_attrs["color"] = "#22c55e"
        edge_attrs["penwidth"] = "2"
    elif trans in coverage.uncovered_transitions:
        edge_attrs["color"] = "#ef4444"
        edge_attrs["style"] = "dashed"
