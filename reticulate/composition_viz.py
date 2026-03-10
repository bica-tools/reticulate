"""Visualization for bottom-up multiparty composition (Step 15).

Generates side-by-side and comparative visualizations showing:

1. **Participant panel**: Each participant's local lattice, color-coded.
2. **Three-level comparison**: Global type vs synchronized product vs free
   product, side-by-side with state counts and reduction ratios.
3. **Synchronized product diagram**: Product states colored by which
   participants are "active" (not at their bottom state), with shared-label
   edges bolded to show synchronization points.
4. **Composition dashboard**: Single HTML page combining all views.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.composition import (
        CompositionResult,
        ComparisonResult,
        SynchronizedResult,
    )
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Color palette for participants
# ---------------------------------------------------------------------------

_PARTICIPANT_COLORS = [
    ("#2563eb", "#dbeafe"),  # blue (border, fill)
    ("#dc2626", "#fee2e2"),  # red
    ("#16a34a", "#dcfce7"),  # green
    ("#9333ea", "#f3e8ff"),  # purple
    ("#ea580c", "#ffedd5"),  # orange
    ("#0891b2", "#cffafe"),  # cyan
    ("#be185d", "#fce7f3"),  # pink
    ("#854d0e", "#fef9c3"),  # brown
]

_LEVEL_COLORS = {
    "global":       ("#1e40af", "#dbeafe"),  # deep blue
    "synchronized": ("#7c3aed", "#ede9fe"),  # violet
    "free":         ("#64748b", "#f1f5f9"),  # slate
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape_dot(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _truncate(label: str, max_len: int = 30) -> str:
    if len(label) <= max_len:
        return label
    return label[:max_len] + "\u2026"


def _dot_to_svg(dot: str) -> str:
    """Convert DOT source to inline SVG."""
    proc = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dot failed: {proc.stderr}")
    svg = proc.stdout
    for prefix in ('<?xml', '<!DOCTYPE'):
        while prefix in svg:
            start = svg.index(prefix)
            end = svg.index('>', start) + 1
            svg = svg[:start] + svg[end:]
    return svg.strip()


# ---------------------------------------------------------------------------
# 1. Participant lattice DOT
# ---------------------------------------------------------------------------

def participant_dot(
    name: str,
    ss: "StateSpace",
    *,
    color_index: int = 0,
    edge_labels: bool = True,
) -> str:
    """DOT source for a single participant's lattice, color-coded."""
    border, fill = _PARTICIPANT_COLORS[color_index % len(_PARTICIPANT_COLORS)]

    lines = [
        "digraph {",
        "    rankdir=TB;",
        f'    label="{_escape_dot(name)}";',
        "    labelloc=t;",
        f'    fontcolor="{border}";',
        "    fontsize=14;",
        '    fontname="Helvetica";',
        f'    node [shape=box, style="filled,rounded", fontname="Helvetica", '
        f'fillcolor="{fill}", color="{border}"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]

    for sid in sorted(ss.states):
        label = _truncate(ss.labels.get(sid, str(sid)))
        if sid == ss.top:
            label = f"\u22a4 {label}"
        elif sid == ss.bottom:
            label = f"\u22a5 {label}"
        attrs = f'label="{_escape_dot(label)}"'
        if sid == ss.top:
            attrs += f', fillcolor="{fill}", penwidth="2"'
        elif sid == ss.bottom:
            attrs += ', fillcolor="#bbf7d0"'
        lines.append(f'    {sid} [{attrs}];')

    for src, lbl, tgt in ss.transitions:
        edge_attr = ""
        if edge_labels:
            edge_attr = f' [label="{_escape_dot(lbl)}", color="{border}"]'
        else:
            edge_attr = f' [color="{border}"]'
        if ss.is_selection(src, lbl, tgt):
            edge_attr = edge_attr[:-1] + ', style="dashed"]'
        lines.append(f'    {src} -> {tgt}{edge_attr};')

    lines.append(f'    {{rank=source; {ss.top}}}')
    lines.append(f'    {{rank=sink; {ss.bottom}}}')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Synchronized product DOT — shared edges bolded
# ---------------------------------------------------------------------------

def synchronized_dot(
    ss: "StateSpace",
    shared_labels: set[str] | None = None,
    *,
    title: str = "Synchronized Product",
    edge_labels: bool = True,
) -> str:
    """DOT for synchronized product with shared transitions bolded."""
    border, fill = _LEVEL_COLORS["synchronized"]
    shared = shared_labels or set()

    lines = [
        "digraph {",
        "    rankdir=TB;",
        f'    label="{_escape_dot(title)}";',
        "    labelloc=t;",
        f'    fontcolor="{border}";',
        "    fontsize=14;",
        '    fontname="Helvetica";',
        f'    node [shape=box, style="filled,rounded", fontname="Helvetica", '
        f'fillcolor="{fill}"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]

    for sid in sorted(ss.states):
        label = _truncate(ss.labels.get(sid, str(sid)))
        if sid == ss.top:
            label = f"\u22a4 {label}"
        elif sid == ss.bottom:
            label = f"\u22a5 {label}"
        attrs = f'label="{_escape_dot(label)}"'
        if sid == ss.top:
            attrs += f', penwidth="2"'
        elif sid == ss.bottom:
            attrs += ', fillcolor="#bbf7d0"'
        lines.append(f'    {sid} [{attrs}];')

    for src, lbl, tgt in ss.transitions:
        parts = []
        if edge_labels:
            parts.append(f'label="{_escape_dot(lbl)}"')
        if lbl in shared:
            parts.append('penwidth="2.5"')
            parts.append('color="#7c3aed"')
            if edge_labels:
                parts.append('fontcolor="#7c3aed"')
        if ss.is_selection(src, lbl, tgt):
            parts.append('style="dashed"')
        edge_attr = f' [{", ".join(parts)}]' if parts else ""
        lines.append(f'    {src} -> {tgt}{edge_attr};')

    lines.append(f'    {{rank=source; {ss.top}}}')
    lines.append(f'    {{rank=sink; {ss.bottom}}}')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Comparison DOT — single state space with level coloring
# ---------------------------------------------------------------------------

def level_dot(
    ss: "StateSpace",
    level: str,
    *,
    title: str | None = None,
    edge_labels: bool = True,
) -> str:
    """DOT for a state space colored by composition level."""
    if level not in _LEVEL_COLORS:
        level = "free"
    border, fill = _LEVEL_COLORS[level]
    display_title = title or level.capitalize()

    lines = [
        "digraph {",
        "    rankdir=TB;",
        f'    label="{_escape_dot(display_title)}";',
        "    labelloc=t;",
        f'    fontcolor="{border}";',
        "    fontsize=14;",
        '    fontname="Helvetica";',
        f'    node [shape=box, style="filled,rounded", fontname="Helvetica", '
        f'fillcolor="{fill}"];',
        '    edge [fontname="Helvetica", fontsize=10];',
    ]

    for sid in sorted(ss.states):
        label = _truncate(ss.labels.get(sid, str(sid)))
        if sid == ss.top:
            label = f"\u22a4 {label}"
        elif sid == ss.bottom:
            label = f"\u22a5 {label}"
        attrs = f'label="{_escape_dot(label)}"'
        if sid == ss.top:
            attrs += ', penwidth="2"'
        elif sid == ss.bottom:
            attrs += ', fillcolor="#bbf7d0"'
        lines.append(f'    {sid} [{attrs}];')

    for src, lbl, tgt in ss.transitions:
        parts = []
        if edge_labels:
            parts.append(f'label="{_escape_dot(lbl)}"')
        parts.append(f'color="{border}"')
        if ss.is_selection(src, lbl, tgt):
            parts.append('style="dashed"')
        edge_attr = f' [{", ".join(parts)}]' if parts else ""
        lines.append(f'    {src} -> {tgt}{edge_attr};')

    lines.append(f'    {{rank=source; {ss.top}}}')
    lines.append(f'    {{rank=sink; {ss.bottom}}}')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Composition dashboard — single HTML page
# ---------------------------------------------------------------------------

def composition_dashboard(
    synced_result: "SynchronizedResult",
    output_path: str,
    *,
    title: str = "Composition Dashboard",
    global_ss: "StateSpace | None" = None,
    comparison: "ComparisonResult | None" = None,
) -> str:
    """Render a composition dashboard as a single HTML file.

    Shows:
    - Participant lattices (color-coded, side by side)
    - Synchronized product (shared edges bolded)
    - Free product (for comparison)
    - Optional: global type state space and three-level hierarchy stats

    Returns *output_path*.
    """
    html = []
    html.append("<!DOCTYPE html>")
    html.append(f"<html><head><meta charset='utf-8'><title>{_html_escape(title)}</title>")
    html.append(_dashboard_css())
    html.append("</head><body>")
    html.append(f'<h1>{_html_escape(title)}</h1>')

    # --- Stats banner ---
    n_synced = len(synced_result.synchronized.states)
    n_free = len(synced_result.free_product.states)
    reduction_pct = (1 - synced_result.reduction_ratio) * 100

    html.append('<div class="stats-banner">')
    html.append(f'<div class="stat"><span class="stat-num">{len(synced_result.participants)}</span>'
                '<span class="stat-label">Participants</span></div>')
    if global_ss is not None:
        html.append(f'<div class="stat"><span class="stat-num">{len(global_ss.states)}</span>'
                    '<span class="stat-label">Global States</span></div>')
    html.append(f'<div class="stat synced"><span class="stat-num">{n_synced}</span>'
                '<span class="stat-label">Synced States</span></div>')
    html.append(f'<div class="stat free"><span class="stat-num">{n_free}</span>'
                '<span class="stat-label">Free States</span></div>')
    html.append(f'<div class="stat reduction"><span class="stat-num">{reduction_pct:.0f}%</span>'
                '<span class="stat-label">Reduction</span></div>')
    html.append('</div>')

    # --- Compatibility matrix ---
    names = list(synced_result.participants.keys())
    if len(names) >= 2:
        html.append('<h2>Compatibility</h2>')
        html.append('<table class="compat-table"><tr><th></th>')
        for n in names:
            html.append(f'<th>{_html_escape(n)}</th>')
        html.append('</tr>')
        for i, n1 in enumerate(names):
            html.append(f'<tr><th>{_html_escape(n1)}</th>')
            for j, n2 in enumerate(names):
                if i == j:
                    html.append('<td class="self">\u2014</td>')
                else:
                    key = (n1, n2) if (n1, n2) in synced_result.compatibility else (n2, n1)
                    if key in synced_result.compatibility:
                        ok = synced_result.compatibility[key]
                        cls = "compat" if ok else "incompat"
                        sym = "\u2713" if ok else "\u2717"
                        html.append(f'<td class="{cls}">{sym}</td>')
                    else:
                        html.append('<td>\u2014</td>')
            html.append('</tr>')
        html.append('</table>')

    # --- Shared labels ---
    if synced_result.shared_labels:
        html.append('<h2>Shared Labels (Synchronization Points)</h2>')
        html.append('<table class="shared-table"><tr><th>Pair</th><th>Shared Labels</th></tr>')
        for (n1, n2), labels in sorted(synced_result.shared_labels.items()):
            labels_str = ", ".join(sorted(labels)) if labels else "\u2014 (none)"
            html.append(f'<tr><td>{_html_escape(n1)} \u2194 {_html_escape(n2)}</td>'
                        f'<td><code>{_html_escape(labels_str)}</code></td></tr>')
        html.append('</table>')

    # --- Participant lattices ---
    html.append('<h2>Participant Lattices</h2>')
    html.append('<div class="panel-row">')
    for i, (name, ss) in enumerate(synced_result.state_spaces.items()):
        dot = participant_dot(name, ss, color_index=i)
        svg = _dot_to_svg(dot)
        html.append(f'<div class="panel"><div class="panel-title">{_html_escape(name)}'
                    f' ({len(ss.states)} states)</div>{svg}</div>')
    html.append('</div>')

    # --- Three-level comparison ---
    html.append('<h2>Composition Hierarchy</h2>')
    html.append('<div class="panel-row">')

    # Global (if available)
    if global_ss is not None:
        dot_g = level_dot(global_ss, "global",
                          title=f"Global Type ({len(global_ss.states)} states)")
        svg_g = _dot_to_svg(dot_g)
        html.append(f'<div class="panel">{svg_g}'
                    '<div class="panel-caption">Choreography: only causally ordered interleavings</div>'
                    '</div>')
        html.append('<div class="arrow">\u2286</div>')

    # Synchronized
    all_shared: set[str] = set()
    for labels in synced_result.shared_labels.values():
        all_shared |= labels
    dot_s = synchronized_dot(
        synced_result.synchronized,
        all_shared,
        title=f"Synchronized ({n_synced} states)",
    )
    svg_s = _dot_to_svg(dot_s)
    html.append(f'<div class="panel">{svg_s}'
                '<div class="panel-caption">Causal: sync on shared labels</div>'
                '</div>')
    html.append('<div class="arrow">\u2286</div>')

    # Free
    dot_f = level_dot(synced_result.free_product, "free",
                      title=f"Free Product ({n_free} states)")
    svg_f = _dot_to_svg(dot_f)
    html.append(f'<div class="panel">{svg_f}'
                '<div class="panel-caption">All interleavings (unrestricted)</div>'
                '</div>')
    html.append('</div>')

    html.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html))

    return output_path


def _dashboard_css() -> str:
    return """<style>
body { font-family: Helvetica, Arial, sans-serif; background: #f8fafc; margin: 2em; color: #1e293b; }
h1 { color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3em; }
h2 { color: #334155; margin-top: 2em; }
.stats-banner { display: flex; gap: 1.5em; margin: 1.5em 0; flex-wrap: wrap; }
.stat { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1em 1.5em;
        text-align: center; min-width: 100px; }
.stat-num { display: block; font-size: 2em; font-weight: bold; color: #0f172a; }
.stat-label { font-size: 0.8em; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.stat.synced { border-color: #7c3aed; }
.stat.synced .stat-num { color: #7c3aed; }
.stat.free .stat-num { color: #64748b; }
.stat.reduction { border-color: #16a34a; }
.stat.reduction .stat-num { color: #16a34a; }
.compat-table { border-collapse: collapse; margin: 1em 0; }
.compat-table th, .compat-table td { border: 1px solid #e2e8f0; padding: 0.5em 1em; text-align: center; }
.compat-table th { background: #f1f5f9; font-weight: 600; }
.compat-table .compat { color: #16a34a; font-weight: bold; }
.compat-table .incompat { color: #dc2626; font-weight: bold; }
.compat-table .self { color: #94a3b8; }
.shared-table { border-collapse: collapse; margin: 1em 0; }
.shared-table th, .shared-table td { border: 1px solid #e2e8f0; padding: 0.5em 1em; text-align: left; }
.shared-table th { background: #f1f5f9; }
.shared-table code { background: #ede9fe; padding: 0.15em 0.4em; border-radius: 3px; color: #7c3aed; }
.panel-row { display: flex; align-items: center; gap: 0.5em; flex-wrap: wrap; margin: 1em 0; }
.panel { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1em;
         display: inline-block; vertical-align: top; }
.panel-title { font-size: 0.85em; font-weight: 600; margin-bottom: 0.5em; color: #475569; }
.panel-caption { font-size: 0.75em; color: #94a3b8; margin-top: 0.5em; text-align: center;
                 font-style: italic; }
.arrow { font-size: 2em; color: #94a3b8; padding: 0 0.3em; align-self: center; }
</style>"""
