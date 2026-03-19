"""Visual HTML board for Protocol Duel.

Generates a self-contained HTML file with an interactive SVG game board.
The Hasse diagram is rendered as the board, with clickable nodes for moves.
"""

from __future__ import annotations

import json
import webbrowser
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


def _compute_ranks(ss: StateSpace) -> dict[int, int]:
    """Assign rank (depth from top) to each state via BFS."""
    ranks: dict[int, int] = {ss.top: 0}
    queue = [ss.top]
    adj: dict[int, list[int]] = defaultdict(list)
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in ranks:
                ranks[neighbor] = ranks[node] + 1
                queue.append(neighbor)

    return ranks


def _layout(ss: StateSpace) -> dict[int, tuple[float, float]]:
    """Compute (x, y) positions for each state."""
    ranks = _compute_ranks(ss)
    max_rank = max(ranks.values()) if ranks else 0

    # Group states by rank
    by_rank: dict[int, list[int]] = defaultdict(list)
    for sid, r in ranks.items():
        by_rank[r].append(sid)

    # Sort states within each rank for consistency
    for r in by_rank:
        by_rank[r].sort()

    positions: dict[int, tuple[float, float]] = {}
    board_width = 800
    y_spacing = 120
    y_offset = 80

    for r in range(max_rank + 1):
        states_at_rank = by_rank.get(r, [])
        n = len(states_at_rank)
        if n == 0:
            continue
        x_spacing = board_width / (n + 1)
        for i, sid in enumerate(states_at_rank):
            x = x_spacing * (i + 1)
            y = y_offset + r * y_spacing
            positions[sid] = (x, y)

    return positions


def _classify_node(ss: StateSpace, sid: int) -> str:
    """Classify a state as branch/select/end/top."""
    if sid == ss.bottom:
        return "end"
    if sid == ss.top:
        return "top"

    outgoing = [(s, l, t) for s, l, t in ss.transitions if s == sid]
    if not outgoing:
        return "end"

    if any(ss.is_selection(s, l, t) for s, l, t in outgoing):
        return "select"

    return "branch"


def generate_board_html(ss: StateSpace, title: str = "Protocol Duel",
                        mode: str = "cooperative", human_role: str = "both") -> str:
    """Generate a self-contained HTML file with interactive game board."""
    positions = _layout(ss)
    max_y = max(y for _, y in positions.values()) if positions else 200
    board_height = max_y + 80

    # Build game data as JSON
    nodes = []
    for sid in sorted(ss.states):
        kind = _classify_node(ss, sid)
        x, y = positions.get(sid, (400, 100))
        label = ss.labels.get(sid, str(sid))
        nodes.append({
            "id": sid,
            "x": x,
            "y": y,
            "label": label,
            "kind": kind,
            "is_top": sid == ss.top,
            "is_bottom": sid == ss.bottom,
        })

    edges = []
    for src, lbl, tgt in ss.transitions:
        is_sel = ss.is_selection(src, lbl, tgt)
        edges.append({
            "src": src,
            "tgt": tgt,
            "label": lbl,
            "is_selection": is_sel,
        })

    game_data = json.dumps({
        "nodes": nodes,
        "edges": edges,
        "top": ss.top,
        "bottom": ss.bottom,
        "mode": mode,
        "human_role": human_role,
        "n_states": len(ss.states),
        "n_transitions": len(ss.transitions),
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    background: #0f172a;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    min-height: 100vh;
    overflow: hidden;
}}

/* Board area */
.board-container {{
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    position: relative;
}}

svg.board {{
    background: radial-gradient(ellipse at center, #1e293b 0%, #0f172a 70%);
    border-radius: 16px;
    border: 1px solid #334155;
    filter: drop-shadow(0 0 40px rgba(59, 130, 246, 0.1));
}}

/* Nodes */
.node-circle {{
    cursor: default;
    transition: all 0.3s ease;
}}
.node-circle.clickable {{
    cursor: pointer;
}}
.node-circle.clickable:hover {{
    filter: brightness(1.3);
}}

.node-label {{
    font-size: 11px;
    fill: #94a3b8;
    text-anchor: middle;
    pointer-events: none;
    font-family: 'Segoe UI', system-ui, sans-serif;
}}

/* Token (player position) */
.token {{
    transition: cx 0.5s ease, cy 0.5s ease;
}}

@keyframes pulse {{
    0%, 100% {{ r: 28; opacity: 0.3; }}
    50% {{ r: 35; opacity: 0.15; }}
}}
.token-glow {{
    animation: pulse 2s ease-in-out infinite;
    transition: cx 0.5s ease, cy 0.5s ease;
}}

/* Edges */
.edge-line {{
    stroke: #334155;
    stroke-width: 1.5;
    fill: none;
    transition: stroke 0.3s, stroke-width 0.3s;
}}
.edge-line.available {{
    stroke: #475569;
    stroke-width: 2;
    cursor: pointer;
}}
.edge-line.available.branch-move {{
    stroke: #3b82f6;
    stroke-width: 2.5;
    filter: drop-shadow(0 0 4px rgba(59, 130, 246, 0.4));
}}
.edge-line.available.select-move {{
    stroke: #f59e0b;
    stroke-width: 2.5;
    filter: drop-shadow(0 0 4px rgba(245, 158, 11, 0.4));
}}
.edge-line.available:hover {{
    stroke-width: 3.5;
}}
.edge-line.traversed {{
    stroke: #22c55e;
    stroke-width: 2;
    stroke-dasharray: 6 3;
    opacity: 0.6;
}}

.edge-label {{
    font-size: 12px;
    fill: #64748b;
    text-anchor: middle;
    pointer-events: none;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 500;
}}
.edge-label.available {{
    fill: #e2e8f0;
    font-weight: 700;
    cursor: pointer;
    pointer-events: auto;
}}

.edge-hitbox {{
    stroke: transparent;
    stroke-width: 20;
    fill: none;
    cursor: pointer;
}}

/* Arrowheads */
.arrow-marker {{
    fill: #334155;
}}
.arrow-marker-active {{
    fill: #3b82f6;
}}

/* Side panel */
.panel {{
    width: 320px;
    background: #1e293b;
    border-left: 1px solid #334155;
    display: flex;
    flex-direction: column;
    padding: 24px;
    overflow-y: auto;
}}

.panel h1 {{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 4px;
    background: linear-gradient(135deg, #3b82f6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.panel .subtitle {{
    font-size: 12px;
    color: #64748b;
    margin-bottom: 20px;
}}

.info-block {{
    background: #0f172a;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid #1e293b;
}}

.info-block h3 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 8px;
}}

.turn-indicator {{
    font-size: 16px;
    font-weight: 600;
    padding: 10px 14px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 12px;
}}
.turn-indicator.client {{
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}}
.turn-indicator.server {{
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}}
.turn-indicator.auto {{
    background: rgba(148, 163, 184, 0.1);
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.2);
}}
.turn-indicator.victory {{
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}}

.moves-list {{
    list-style: none;
    max-height: 200px;
    overflow-y: auto;
}}
.moves-list li {{
    padding: 4px 0;
    font-size: 13px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}}
.moves-list li .move-num {{
    color: #475569;
    margin-right: 6px;
}}
.moves-list li.client-move {{ color: #60a5fa; }}
.moves-list li.server-move {{ color: #fbbf24; }}
.moves-list li.auto-move {{ color: #64748b; }}

.available-moves {{
    list-style: none;
}}
.available-moves li {{
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-family: 'JetBrains Mono', monospace;
    transition: all 0.2s;
}}
.available-moves li.branch-option {{
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    color: #60a5fa;
}}
.available-moves li.branch-option:hover {{
    background: rgba(59, 130, 246, 0.25);
    border-color: rgba(59, 130, 246, 0.5);
}}
.available-moves li.select-option {{
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.2);
    color: #fbbf24;
}}
.available-moves li.select-option:hover {{
    background: rgba(245, 158, 11, 0.25);
    border-color: rgba(245, 158, 11, 0.5);
}}

.stat {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 13px;
}}
.stat .label {{ color: #64748b; }}
.stat .value {{ color: #e2e8f0; font-weight: 600; }}

.legend {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: auto;
    padding-top: 16px;
    border-top: 1px solid #334155;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #64748b;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
}}

.btn-restart {{
    margin-top: 12px;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #334155;
    background: #0f172a;
    color: #94a3b8;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
}}
.btn-restart:hover {{
    background: #1e293b;
    color: #e2e8f0;
    border-color: #475569;
}}

/* Victory overlay */
.victory-overlay {{
    display: none;
    position: absolute;
    inset: 0;
    background: rgba(15, 23, 42, 0.85);
    align-items: center;
    justify-content: center;
    z-index: 10;
    border-radius: 16px;
}}
.victory-overlay.show {{ display: flex; }}
.victory-card {{
    text-align: center;
    padding: 40px;
}}
.victory-card h2 {{
    font-size: 28px;
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}}
.victory-card p {{
    color: #94a3b8;
    margin: 4px 0;
}}
.victory-card .score {{
    font-size: 48px;
    font-weight: 700;
    color: #4ade80;
    margin: 16px 0;
}}
</style>
</head>
<body>

<div class="board-container">
    <svg class="board" id="board" width="800" height="{board_height}">
        <defs>
            <marker id="arrowhead" markerWidth="8" markerHeight="6"
                    refX="8" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" class="arrow-marker"/>
            </marker>
            <filter id="glow-blue">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="glow-green">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <radialGradient id="grad-branch" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#60a5fa"/>
                <stop offset="100%" stop-color="#3b82f6"/>
            </radialGradient>
            <radialGradient id="grad-select" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#fbbf24"/>
                <stop offset="100%" stop-color="#f59e0b"/>
            </radialGradient>
            <radialGradient id="grad-end" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#4ade80"/>
                <stop offset="100%" stop-color="#22c55e"/>
            </radialGradient>
            <radialGradient id="grad-top" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#818cf8"/>
                <stop offset="100%" stop-color="#6366f1"/>
            </radialGradient>
        </defs>
        <!-- edges and nodes rendered by JS -->
    </svg>
    <div class="victory-overlay" id="victoryOverlay">
        <div class="victory-card">
            <h2>Protocol Complete!</h2>
            <p>The token reached the end state.</p>
            <div class="score" id="victoryScore">100%</div>
            <p id="victoryPath"></p>
        </div>
    </div>
</div>

<div class="panel">
    <h1>Protocol Duel</h1>
    <div class="subtitle" id="typeInfo">Loading...</div>

    <div class="turn-indicator" id="turnIndicator">—</div>

    <div class="info-block">
        <h3>Available moves</h3>
        <ul class="available-moves" id="movesList"></ul>
    </div>

    <div class="info-block">
        <h3>Move history</h3>
        <ul class="moves-list" id="historyList"></ul>
    </div>

    <div class="info-block">
        <h3>Stats</h3>
        <div class="stat"><span class="label">States</span><span class="value" id="statStates">—</span></div>
        <div class="stat"><span class="label">Transitions</span><span class="value" id="statTrans">—</span></div>
        <div class="stat"><span class="label">Moves</span><span class="value" id="statMoves">0</span></div>
        <div class="stat"><span class="label">Mode</span><span class="value" id="statMode">—</span></div>
    </div>

    <button class="btn-restart" onclick="restartGame()">Restart</button>

    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#6366f1"></div> Top (start)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div> Branch (you)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div> Select (server)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> End</div>
    </div>
</div>

<script>
const GAME_DATA = {game_data};

// ---- State ----
let current = GAME_DATA.top;
let moveHistory = [];
let turnCount = 0;
let traversedEdges = new Set();

const nodeMap = {{}};
GAME_DATA.nodes.forEach(n => nodeMap[n.id] = n);

function outgoing(sid) {{
    return GAME_DATA.edges.filter(e => e.src === sid);
}}

function whoseTurn(sid) {{
    const moves = outgoing(sid);
    if (moves.length === 0) return "none";
    if (moves.length === 1) return "auto";
    if (moves.some(e => e.is_selection)) return "server";
    return "client";
}}

function isHumanTurn(turn) {{
    const role = GAME_DATA.human_role;
    if (role === "both") return true;
    if (role === "client" && turn === "client") return true;
    if (role === "server" && turn === "server") return true;
    return false;
}}

// ---- AI ----
function bfsDistance(from, to) {{
    if (from === to) return 0;
    const visited = new Set([from]);
    const queue = [[from, 0]];
    while (queue.length > 0) {{
        const [node, dist] = queue.shift();
        for (const e of GAME_DATA.edges.filter(e => e.src === node)) {{
            if (e.tgt === to) return dist + 1;
            if (!visited.has(e.tgt)) {{
                visited.add(e.tgt);
                queue.push([e.tgt, dist + 1]);
            }}
        }}
    }}
    return 999;
}}

function aiMove(sid) {{
    const moves = outgoing(sid);
    const mode = GAME_DATA.mode;
    const turn = whoseTurn(sid);

    if (mode === "adversarial" && turn === "server") {{
        // Server tries longest path
        let best = moves[0], bestDist = -1;
        for (const e of moves) {{
            const d = bfsDistance(e.tgt, GAME_DATA.bottom);
            if (d > bestDist) {{ bestDist = d; best = e; }}
        }}
        return best;
    }}
    // Default: shortest path
    let best = moves[0], bestDist = Infinity;
    for (const e of moves) {{
        const d = bfsDistance(e.tgt, GAME_DATA.bottom);
        if (d < bestDist) {{ bestDist = d; best = e; }}
    }}
    return best;
}}

// ---- Rendering ----
function edgeKey(src, lbl, tgt) {{
    return `${{src}}-${{lbl}}-${{tgt}}`;
}}

function renderBoard() {{
    const svg = document.getElementById("board");
    // Clear dynamic content (keep defs)
    const defs = svg.querySelector("defs");
    svg.innerHTML = "";
    svg.appendChild(defs);

    const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const labelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    svg.appendChild(edgeGroup);
    svg.appendChild(nodeGroup);
    svg.appendChild(labelGroup);

    const availableMoves = outgoing(current);
    const turn = whoseTurn(current);
    const isHuman = isHumanTurn(turn);

    // Draw edges
    GAME_DATA.edges.forEach(e => {{
        const src = nodeMap[e.src];
        const tgt = nodeMap[e.tgt];
        if (!src || !tgt) return;

        const isAvail = e.src === current && availableMoves.length > 1 && isHuman;
        const isTraversed = traversedEdges.has(edgeKey(e.src, e.label, e.tgt));

        // Compute curved path for multiple edges between same ranks
        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const len = Math.sqrt(dx*dx + dy*dy);

        // Shorten line to not overlap node circles
        const r = 20;
        const ratio1 = r / len;
        const ratio2 = (len - r - 6) / len;
        const x1 = src.x + dx * ratio1;
        const y1 = src.y + dy * ratio1;
        const x2 = src.x + dx * ratio2;
        const y2 = src.y + dy * ratio2;

        // Check for parallel edges
        const sameDir = GAME_DATA.edges.filter(oe => oe.src === e.src && oe.tgt === e.tgt);
        const idx = sameDir.indexOf(e);
        const total = sameDir.length;
        let pathD;
        if (total > 1) {{
            const offset = (idx - (total - 1) / 2) * 30;
            const mx = (src.x + tgt.x) / 2 - dy / len * offset;
            const my = (src.y + tgt.y) / 2 + dx / len * offset;
            pathD = `M ${{x1}} ${{y1}} Q ${{mx}} ${{my}} ${{x2}} ${{y2}}`;
        }} else {{
            pathD = `M ${{x1}} ${{y1}} L ${{x2}} ${{y2}}`;
        }}

        // Hitbox for clicking
        if (isAvail) {{
            const hitbox = document.createElementNS("http://www.w3.org/2000/svg", "path");
            hitbox.setAttribute("d", pathD);
            hitbox.setAttribute("class", "edge-hitbox");
            hitbox.addEventListener("click", () => makeMove(e));
            edgeGroup.appendChild(hitbox);
        }}

        const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
        line.setAttribute("d", pathD);
        line.setAttribute("marker-end", "url(#arrowhead)");
        let cls = "edge-line";
        if (isTraversed) cls += " traversed";
        if (isAvail) {{
            cls += " available";
            cls += e.is_selection ? " select-move" : " branch-move";
        }}
        line.setAttribute("class", cls);
        if (isAvail) line.addEventListener("click", () => makeMove(e));
        edgeGroup.appendChild(line);

        // Edge label
        let lx, ly;
        if (total > 1) {{
            const offset = (idx - (total - 1) / 2) * 30;
            lx = (src.x + tgt.x) / 2 - dy / len * offset;
            ly = (src.y + tgt.y) / 2 + dx / len * offset;
        }} else {{
            lx = (src.x + tgt.x) / 2 - dy / len * 12;
            ly = (src.y + tgt.y) / 2 + dx / len * 12;
        }}

        const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        lbl.setAttribute("x", lx);
        lbl.setAttribute("y", ly - 4);
        let lblCls = "edge-label";
        if (isAvail) {{
            lblCls += " available";
            lbl.style.pointerEvents = "auto";
            lbl.style.cursor = "pointer";
            lbl.addEventListener("click", () => makeMove(e));
        }}
        lbl.setAttribute("class", lblCls);
        lbl.textContent = e.label;
        labelGroup.appendChild(lbl);
    }});

    // Draw nodes
    GAME_DATA.nodes.forEach(n => {{
        const isCurrent = n.id === current;
        const isTarget = availableMoves.some(e => e.tgt === n.id) && isHuman;

        // Glow for current position
        if (isCurrent) {{
            const glow = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            glow.setAttribute("cx", n.x);
            glow.setAttribute("cy", n.y);
            glow.setAttribute("r", 28);
            glow.setAttribute("class", "token-glow");
            const glowColor = n.is_bottom ? "#22c55e" : "#6366f1";
            glow.setAttribute("fill", glowColor);
            glow.setAttribute("opacity", "0.3");
            nodeGroup.appendChild(glow);
        }}

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", n.x);
        circle.setAttribute("cy", n.y);
        circle.setAttribute("r", isCurrent ? 22 : 18);

        let grad = "url(#grad-branch)";
        if (n.kind === "top") grad = "url(#grad-top)";
        else if (n.kind === "end") grad = "url(#grad-end)";
        else if (n.kind === "select") grad = "url(#grad-select)";
        circle.setAttribute("fill", grad);

        circle.setAttribute("stroke", isCurrent ? "#e2e8f0" : "none");
        circle.setAttribute("stroke-width", isCurrent ? "2.5" : "0");
        circle.setAttribute("opacity", isCurrent ? "1" : (isTarget ? "0.9" : "0.65"));

        let cls = "node-circle";
        if (isTarget) {{
            cls += " clickable";
            circle.addEventListener("click", () => {{
                const edge = availableMoves.find(e => e.tgt === n.id);
                if (edge) makeMove(edge);
            }});
        }}
        circle.setAttribute("class", cls);
        nodeGroup.appendChild(circle);

        // Node label
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", n.x);
        text.setAttribute("y", n.y + 32);
        text.setAttribute("class", "node-label");

        let displayLabel = n.label;
        if (n.is_top) displayLabel = "\\u22a4 " + displayLabel;
        else if (n.is_bottom) displayLabel = "\\u22a5 end";
        if (displayLabel.length > 20) displayLabel = displayLabel.substring(0, 18) + "\\u2026";
        text.textContent = displayLabel;
        labelGroup.appendChild(text);
    }});
}}

function updatePanel() {{
    const moves = outgoing(current);
    const turn = whoseTurn(current);
    const node = nodeMap[current];
    const isFinished = current === GAME_DATA.bottom;

    // Type info
    document.getElementById("typeInfo").textContent =
        `${{GAME_DATA.n_states}} states, ${{GAME_DATA.n_transitions}} transitions`;

    // Turn indicator
    const ti = document.getElementById("turnIndicator");
    if (isFinished) {{
        ti.textContent = "\\u2705 Protocol Complete!";
        ti.className = "turn-indicator victory";
    }} else if (turn === "auto") {{
        ti.textContent = "\\u26a1 Automatic move";
        ti.className = "turn-indicator auto";
    }} else if (turn === "client") {{
        ti.textContent = "\\ud83d\\udd35 Client's turn (branch)";
        ti.className = "turn-indicator client";
    }} else {{
        ti.textContent = "\\ud83d\\udfe0 Server's turn (select)";
        ti.className = "turn-indicator server";
    }}

    // Available moves
    const ml = document.getElementById("movesList");
    ml.innerHTML = "";
    if (!isFinished && (moves.length > 1 || turn === "auto")) {{
        moves.forEach((e, i) => {{
            const li = document.createElement("li");
            li.textContent = `${{i + 1}}. ${{e.label}}`;
            li.className = e.is_selection ? "select-option" : "branch-option";
            if (isHumanTurn(turn) && moves.length > 1) {{
                li.addEventListener("click", () => makeMove(e));
            }}
            ml.appendChild(li);
        }});
    }} else if (isFinished) {{
        const li = document.createElement("li");
        li.textContent = "Game over";
        li.style.color = "#4ade80";
        li.style.cursor = "default";
        ml.appendChild(li);
    }}

    // History
    const hl = document.getElementById("historyList");
    hl.innerHTML = "";
    moveHistory.forEach((m, i) => {{
        const li = document.createElement("li");
        li.innerHTML = `<span class="move-num">${{i + 1}}.</span>${{m.label}}`;
        li.className = m.type + "-move";
        hl.appendChild(li);
    }});
    hl.scrollTop = hl.scrollHeight;

    // Stats
    document.getElementById("statStates").textContent = GAME_DATA.n_states;
    document.getElementById("statTrans").textContent = GAME_DATA.n_transitions;
    document.getElementById("statMoves").textContent = turnCount;
    document.getElementById("statMode").textContent = GAME_DATA.mode;
}}

// ---- Game logic ----
function makeMove(edge) {{
    const turn = whoseTurn(current);
    const moveType = turn === "auto" ? "auto" : (edge.is_selection ? "server" : "client");

    traversedEdges.add(edgeKey(edge.src, edge.label, edge.tgt));
    moveHistory.push({{ label: edge.label, type: moveType }});
    current = edge.tgt;
    turnCount++;

    renderBoard();
    updatePanel();

    // Check victory
    if (current === GAME_DATA.bottom) {{
        showVictory();
        return;
    }}

    // Auto-move or AI move after a short delay
    const nextTurn = whoseTurn(current);
    const nextMoves = outgoing(current);
    if (nextTurn === "auto" && nextMoves.length === 1) {{
        setTimeout(() => makeMove(nextMoves[0]), 600);
    }} else if (nextMoves.length > 1 && !isHumanTurn(nextTurn)) {{
        // AI move
        setTimeout(() => {{
            const ai = aiMove(current);
            makeMove(ai);
        }}, 800);
    }}
}}

function showVictory() {{
    const efficiency = Math.max(0, 100 - (turnCount - 1) * 10);
    document.getElementById("victoryScore").textContent = efficiency + "%";
    const pathStr = moveHistory.map(m => m.label).join(" \\u2192 ");
    document.getElementById("victoryPath").textContent = "\\u22a4 \\u2192 " + pathStr + " \\u2192 \\u22a5";
    document.getElementById("victoryOverlay").classList.add("show");
}}

function restartGame() {{
    current = GAME_DATA.top;
    moveHistory = [];
    turnCount = 0;
    traversedEdges = new Set();
    document.getElementById("victoryOverlay").classList.remove("show");
    renderBoard();
    updatePanel();
}}

// ---- Init ----
renderBoard();
updatePanel();

// Handle initial auto-move
const initTurn = whoseTurn(current);
const initMoves = outgoing(current);
if (initTurn === "auto" && initMoves.length === 1) {{
    setTimeout(() => makeMove(initMoves[0]), 800);
}}
</script>

</body>
</html>"""
    return html


def open_visual_board(ss: StateSpace, title: str = "Protocol Duel",
                      mode: str = "cooperative", human_role: str = "both",
                      output_path: str | None = None) -> str:
    """Generate the visual board HTML and open it in a browser.

    Returns the path to the generated HTML file.
    """
    html = generate_board_html(ss, title=title, mode=mode, human_role=human_role)

    if output_path:
        path = Path(output_path)
    else:
        fd, tmp = tempfile.mkstemp(suffix=".html", prefix="protocol_duel_")
        path = Path(tmp)
        import os
        os.close(fd)

    path.write_text(html, encoding="utf-8")
    webbrowser.open(f"file://{path.resolve()}")
    return str(path)
