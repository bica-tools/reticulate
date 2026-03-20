"""Visualization of session type state spaces as Hasse diagrams.

Generates Graphviz DOT representations of state-space transition systems.
Supports counterexample highlighting, optional label/edge-label toggling,
and role-annotated edges for multiparty global types.
All states and transitions (including recursion cycles) are shown explicitly.

The ``dot_source()`` function uses only the standard library and is always
available.  ``hasse_diagram()`` and ``render_hasse()`` require the ``graphviz``
Python package (lazy-imported).

For multiparty global types, use ``role_dot_source()``, ``role_hasse_diagram()``,
and ``role_render_hasse()`` to produce role-colored Hasse diagrams where each
edge is colored by the sender role and labeled with the interaction.
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
    node_style: str = "default",
    scc_clusters: bool = False,
) -> str:
    """Return DOT source string for the Hasse diagram of *ss*.

    No external dependencies — always available.

    Parameters:
        node_style: ``"default"`` for labelled boxes, ``"constructor"`` for
            circles showing session type constructor symbols
            (``&``, ``⊕``, ``∥``, ``end``).
        scc_clusters: When True and *result* is provided, multi-state SCCs
            (from recursive types) are drawn as shaded cluster subgraphs
            with intra-SCC back-edges shown as dashed purple lines.
    """
    return _build_dot(ss, result, title=title, labels=labels,
                      edge_labels=edge_labels, coverage=coverage,
                      node_style=node_style, scc_clusters=scc_clusters)


def hasse_diagram(
    ss: StateSpace,
    result: LatticeResult | None = None,
    *,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
    coverage: CoverageResult | None = None,
    node_style: str = "default",
    scc_clusters: bool = False,
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

    src = _build_dot(ss, result, title=title, labels=labels,
                     edge_labels=edge_labels, coverage=coverage,
                     node_style=node_style, scc_clusters=scc_clusters)
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
    node_style: str = "default",
    scc_clusters: bool = False,
) -> str:
    """Render Hasse diagram to file.  Returns the output file path."""
    try:
        import graphviz  # noqa: F811
    except ImportError:
        raise ImportError(
            "The 'graphviz' Python package is required for render_hasse(). "
            "Install it with: pip install graphviz"
        ) from None

    src = _build_dot(ss, result, title=title, labels=labels,
                     edge_labels=edge_labels, coverage=coverage,
                     node_style=node_style, scc_clusters=scc_clusters)
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
    node_style: str = "default",
    scc_clusters: bool = False,
) -> str:
    lines: list[str] = []
    lines.append("digraph {")
    lines.append("    rankdir=TB;")
    if node_style == "constructor":
        lines.append('    node [shape=circle, style=filled, fontname="Helvetica", '
                      'width=0.4, fixedsize=true, fontsize=14];')
    else:
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

    # Compute SCC cluster info if requested
    scc_groups: dict[int, frozenset[int]] = {}
    clustered_states: dict[int, int] = {}  # state → SCC representative
    if scc_clusters and result is not None:
        scc_groups = result.scc_groups
        for rep, members in scc_groups.items():
            for s in members:
                clustered_states[s] = rep

    if result is not None:
        _build_dot_with_result(ss, result, lines, labels, edge_labels,
                               counter_states, coverage, node_style,
                               scc_groups, clustered_states)
    else:
        _build_dot_plain(ss, lines, labels, edge_labels, coverage,
                         node_style, scc_groups, clustered_states)

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
    node_style: str = "default",
    scc_groups: dict[int, frozenset[int]] | None = None,
    clustered_states: dict[int, int] | None = None,
) -> None:
    """Add nodes/edges without SCC collapsing."""
    if scc_groups is None:
        scc_groups = {}
    if clustered_states is None:
        clustered_states = {}

    _emit_nodes(ss, lines, labels, coverage, node_style, scc_groups,
                clustered_states, counter_states=set())
    _emit_edges(ss, lines, edge_labels, coverage, node_style, clustered_states)


def _build_dot_with_result(
    ss: StateSpace,
    result: LatticeResult,
    lines: list[str],
    labels: bool,
    edge_labels: bool,
    counter_states: set[int],
    coverage: CoverageResult | None = None,
    node_style: str = "default",
    scc_groups: dict[int, frozenset[int]] | None = None,
    clustered_states: dict[int, int] | None = None,
) -> None:
    """Add all nodes/edges with counterexample highlighting."""
    if scc_groups is None:
        scc_groups = {}
    if clustered_states is None:
        clustered_states = {}

    _emit_nodes(ss, lines, labels, coverage, node_style, scc_groups,
                clustered_states, counter_states)
    _emit_edges(ss, lines, edge_labels, coverage, node_style, clustered_states)


# ---------------------------------------------------------------------------
# Internal: node and edge emission (with optional SCC clusters)
# ---------------------------------------------------------------------------

def _emit_nodes(
    ss: StateSpace,
    lines: list[str],
    labels: bool,
    coverage: CoverageResult | None,
    node_style: str,
    scc_groups: dict[int, frozenset[int]],
    clustered_states: dict[int, int],
    counter_states: set[int],
) -> None:
    """Emit DOT node declarations, optionally wrapping SCC members in clusters."""

    def _node_decl(sid: int, indent: str = "    ") -> str:
        if node_style == "constructor":
            attrs = _node_attrs_constructor(ss, sid)
        else:
            node_label = _node_label(ss, sid, labels)
            attrs = _node_attrs(ss, sid, node_label)
        if sid in counter_states:
            attrs["color"] = "red"
            attrs["penwidth"] = "2"
        if coverage is not None and sid in coverage.uncovered_states and sid not in (ss.top, ss.bottom):
            attrs["fillcolor"] = "#fee2e2"
        return f'{indent}{sid} [{_fmt_attrs(attrs)}];'

    # Emit SCC cluster subgraphs
    emitted: set[int] = set()
    for idx, (rep, members) in enumerate(sorted(scc_groups.items())):
        n = len(members)
        lines.append(f'    subgraph cluster_scc{idx} {{')
        lines.append(f'        label="SCC ({n} states, cyclic-equivalent)";')
        lines.append('        style=filled;')
        lines.append('        color="#e9d5ff";')
        lines.append('        fillcolor="#f5f3ff";')
        lines.append('        fontname="Helvetica";')
        lines.append('        fontsize=10;')
        for sid in sorted(members):
            lines.append(_node_decl(sid, indent="        "))
            emitted.add(sid)
        lines.append('    }')

    # Emit remaining nodes outside clusters
    for sid in sorted(ss.states):
        if sid not in emitted:
            lines.append(_node_decl(sid))


def _emit_edges(
    ss: StateSpace,
    lines: list[str],
    edge_labels: bool,
    coverage: CoverageResult | None,
    node_style: str,
    clustered_states: dict[int, int],
) -> None:
    """Emit DOT edge declarations, styling intra-SCC edges as dashed."""
    for src, lbl, tgt in ss.transitions:
        edge_attrs: dict[str, str] = {}
        if edge_labels:
            edge_attrs["label"] = lbl

        # Intra-SCC back-edge: dashed purple
        if (clustered_states
                and src in clustered_states
                and tgt in clustered_states
                and clustered_states[src] == clustered_states[tgt]):
            # Check if this is a back-edge (tgt has lower or equal ID, heuristic)
            # In session type state spaces, back-edges go from higher to lower IDs
            if tgt <= src:
                edge_attrs["style"] = "dashed"
                edge_attrs["color"] = "#7c3aed"

        # In constructor mode, mark selection edges as dashed
        if node_style == "constructor" and ss.is_selection(src, lbl, tgt):
            edge_attrs["style"] = "dashed"

        _apply_coverage_edge_attrs(edge_attrs, coverage, src, lbl, tgt)
        lines.append(f'    {src} -> {tgt}{_fmt_edge_attrs(edge_attrs)};')


# ---------------------------------------------------------------------------
# Internal: node helpers
# ---------------------------------------------------------------------------

# Constructor classification symbols and colors
_CONSTRUCTOR_SYMBOLS: dict[str, str] = {
    "branch": "&",
    "select": "\u2295",    # ⊕
    "parallel": "\u2225",  # ∥
    "end": "\u22a5",       # ⊥
    "top": "\u22a4",       # ⊤
}

_CONSTRUCTOR_COLORS: dict[str, str] = {
    "branch": "#bfdbfe",    # blue
    "select": "#fde68a",    # amber
    "parallel": "#d9f99d",  # lime
    "end": "#bbf7d0",       # green
    "top": "#bfdbfe",       # blue
}


def _classify_constructor(ss: StateSpace, sid: int) -> str:
    """Classify a state by its session type constructor.

    Returns one of: "top", "end", "select", "parallel", "branch".
    """
    if sid == ss.bottom:
        return "end"
    if sid == ss.top:
        return "top"

    label = ss.labels.get(sid, "")

    # Selection: label starts with +{ or any outgoing transition is a selection
    if label.startswith("+{"):
        return "select"
    if any(ss.is_selection(s, l, t) for s, l, t in ss.transitions if s == sid):
        return "select"

    # Product (parallel): label is a tuple like "(left, right)"
    if label.startswith("(") and "," in label:
        return "parallel"

    # Default: branch
    return "branch"


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


def _node_attrs_constructor(ss: StateSpace, sid: int) -> dict[str, str]:
    """Build attribute dict for constructor-style node (circle with symbol)."""
    kind = _classify_constructor(ss, sid)
    symbol = _CONSTRUCTOR_SYMBOLS[kind]
    color = _CONSTRUCTOR_COLORS[kind]

    attrs: dict[str, str] = {
        "label": symbol,
        "fillcolor": color,
    }

    # Make end/top slightly different
    if kind == "end":
        attrs["shape"] = "doublecircle"
        attrs["width"] = "0.35"
    elif kind == "top":
        attrs["penwidth"] = "2"

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


# ---------------------------------------------------------------------------
# Role-annotated Hasse diagrams (multiparty global types)
# ---------------------------------------------------------------------------

# Perceptually distinct color palette for up to 10 roles
_ROLE_COLORS = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#854d0e",  # brown
    "#4f46e5",  # indigo
    "#65a30d",  # lime
]


def _extract_roles_from_transitions(
    ss: "StateSpace",
) -> tuple[dict[str, str], set[str]]:
    """Extract role→color mapping from role-annotated transition labels.

    Labels of the form "sender->receiver:method" are parsed to identify
    sender and receiver roles. Returns (role_colors, all_roles).
    """
    all_roles: set[str] = set()
    for _, lbl, _ in ss.transitions:
        if "->" in lbl and ":" in lbl:
            role_part, _ = lbl.split(":", 1)
            parts = role_part.split("->")
            if len(parts) == 2:
                all_roles.add(parts[0])
                all_roles.add(parts[1])

    role_colors: dict[str, str] = {}
    for i, role in enumerate(sorted(all_roles)):
        role_colors[role] = _ROLE_COLORS[i % len(_ROLE_COLORS)]

    return role_colors, all_roles


def _parse_role_label(lbl: str) -> tuple[str | None, str | None, str]:
    """Parse a role-annotated label "sender->receiver:method".

    Returns (sender, receiver, method). If not role-annotated, returns
    (None, None, lbl).
    """
    if "->" not in lbl or ":" not in lbl:
        return None, None, lbl
    role_part, method = lbl.split(":", 1)
    parts = role_part.split("->")
    if len(parts) != 2:
        return None, None, lbl
    return parts[0], parts[1], method


def role_dot_source(
    ss: "StateSpace",
    *,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
    role_colors: dict[str, str] | None = None,
    show_legend: bool = True,
) -> str:
    """Return DOT source for a role-annotated Hasse diagram.

    Edges are colored by sender role. The legend shows role→color mapping.

    Parameters:
        ss: State space with role-annotated transition labels.
        title: Optional diagram title.
        labels: Show state labels.
        edge_labels: Show transition labels.
        role_colors: Optional custom role→color mapping. If None,
            colors are assigned automatically.
        show_legend: Show role color legend.
    """
    if role_colors is None:
        role_colors, _ = _extract_roles_from_transitions(ss)

    lines: list[str] = []
    lines.append("digraph {")
    lines.append("    rankdir=TB;")
    lines.append('    node [shape=box, style="filled,rounded", fontname="Helvetica"];')
    lines.append('    edge [fontname="Helvetica", fontsize=10];')

    if title is not None:
        lines.append(f'    label="{_escape_dot(title)}";')
        lines.append("    labelloc=t;")
        lines.append("    fontsize=14;")

    # Nodes
    for sid in sorted(ss.states):
        node_label = _node_label(ss, sid, labels)
        attrs = _node_attrs(ss, sid, node_label)
        lines.append(f'    {sid} [{_fmt_attrs(attrs)}];')

    # Edges with role coloring
    for src, lbl, tgt in ss.transitions:
        edge_attrs: dict[str, str] = {}
        sender, receiver, method = _parse_role_label(lbl)

        if edge_labels:
            if sender is not None:
                edge_attrs["label"] = f"{sender}\u2192{receiver}:{method}"
            else:
                edge_attrs["label"] = lbl

        if sender is not None and sender in role_colors:
            edge_attrs["color"] = role_colors[sender]
            edge_attrs["fontcolor"] = role_colors[sender]
            edge_attrs["penwidth"] = "1.5"

        # Mark selection transitions with dashed style
        if ss.is_selection(src, lbl, tgt):
            edge_attrs["style"] = "dashed"

        lines.append(f'    {src} -> {tgt}{_fmt_edge_attrs(edge_attrs)};')

    # Layout hints
    lines.append(f'    {{rank=source; {ss.top}}}')
    lines.append(f'    {{rank=sink; {ss.bottom}}}')

    # Legend
    if show_legend and role_colors:
        lines.append("")
        lines.append("    subgraph cluster_legend {")
        lines.append('        label="Roles";')
        lines.append('        style=rounded;')
        lines.append('        color="#94a3b8";')
        lines.append('        fontname="Helvetica";')
        lines.append('        fontsize=11;')
        lines.append('        node [shape=plaintext, fillcolor=white];')

        legend_rows = []
        for role in sorted(role_colors.keys()):
            color = role_colors[role]
            legend_rows.append(
                f'<TR><TD><FONT COLOR="{color}">\u25CF</FONT></TD>'
                f'<TD ALIGN="LEFT">{_escape_dot(role)}</TD></TR>'
            )
        table = "<TABLE BORDER=\"0\" CELLSPACING=\"0\">" + "".join(legend_rows) + "</TABLE>"
        lines.append(f'        legend [label=<{table}>];')
        lines.append("    }")

    lines.append("}")
    return "\n".join(lines)


def role_hasse_diagram(
    ss: "StateSpace",
    *,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
    role_colors: dict[str, str] | None = None,
    show_legend: bool = True,
) -> "graphviz.Digraph":  # type: ignore[name-defined]
    """Build a role-annotated Hasse diagram as a Graphviz object.

    Raises ``ImportError`` if the ``graphviz`` package is not installed.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "The 'graphviz' Python package is required for role_hasse_diagram()."
        ) from None

    src = role_dot_source(
        ss, title=title, labels=labels, edge_labels=edge_labels,
        role_colors=role_colors, show_legend=show_legend,
    )
    return graphviz.Source(src)


def role_render_hasse(
    ss: "StateSpace",
    path: str,
    *,
    fmt: str = "png",
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
    role_colors: dict[str, str] | None = None,
    show_legend: bool = True,
) -> str:
    """Render role-annotated Hasse diagram to file. Returns output path."""
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "The 'graphviz' Python package is required for role_render_hasse()."
        ) from None

    src = role_dot_source(
        ss, title=title, labels=labels, edge_labels=edge_labels,
        role_colors=role_colors, show_legend=show_legend,
    )
    g = graphviz.Source(src)
    return g.render(filename=path, format=fmt, cleanup=True)
