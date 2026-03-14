"""Visualization of parallel product state spaces.

Three visualization modes that preserve the geometric product semantics:

1. **Grid layout** (``--grid``): 2D grid for binary products ``S₁ ∥ S₂``.
2. **3D rendering** (``--3d``): Interactive HTML for ternary products ``S₁ ∥ S₂ ∥ S₃``.
3. **Factored view** (``--factored``): Side-by-side factor lattices.

All modes require ``product_coords`` and ``product_factors`` on the ``StateSpace``.
These are populated automatically by ``product_statespace()`` for parallel types.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.lattice import LatticeResult
    from reticulate.statespace import StateSpace


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


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate between two hex colors.  *t* in [0, 1]."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _dot_to_svg(dot: str, *, engine: str = "dot") -> str:
    """Convert DOT source to inline SVG."""
    proc = subprocess.run(
        [engine, "-Tsvg"],
        input=dot,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{engine} failed: {proc.stderr}")
    svg = proc.stdout
    for prefix in ('<?xml', '<!DOCTYPE'):
        while prefix in svg:
            start = svg.index(prefix)
            end = svg.index('>', start) + 1
            svg = svg[:start] + svg[end:]
    return svg.strip()


def _topo_order(ss: "StateSpace") -> list[int]:
    """Topological sort of states (top first, bottom last)."""
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    in_deg: dict[int, int] = {s: 0 for s in ss.states}
    for src, _, tgt in ss.transitions:
        if src != tgt:
            adj[src].append(tgt)
            in_deg[tgt] = in_deg.get(tgt, 0) + 1
    queue = sorted(s for s in ss.states if in_deg[s] == 0)
    order: list[int] = []
    while queue:
        s = queue.pop(0)
        order.append(s)
        for t in adj[s]:
            in_deg[t] -= 1
            if in_deg[t] == 0:
                queue.append(t)
    # Add any remaining states (cycles)
    for s in sorted(ss.states):
        if s not in order:
            order.append(s)
    return order


def _validate_product(ss: "StateSpace", ndim: int | None = None) -> None:
    """Raise ValueError if ss lacks product metadata or has wrong dimensionality."""
    if ss.product_coords is None or ss.product_factors is None:
        raise ValueError("State space has no product metadata (not a parallel type)")
    if ndim is not None and ss.product_factors:
        actual = len(ss.product_factors)
        if actual != ndim:
            raise ValueError(
                f"Expected {ndim} factors, got {actual}"
            )


# ---------------------------------------------------------------------------
# 4a. Grid layout (2D products)
# ---------------------------------------------------------------------------

def grid_dot_source(
    ss: "StateSpace",
    result: "LatticeResult | None" = None,
    *,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
) -> str:
    """DOT source for a 2D grid layout of a binary product.

    Uses ``neato`` engine with pinned ``pos="x,y!"`` coordinates.
    """
    _validate_product(ss, ndim=2)
    assert ss.product_factors is not None
    assert ss.product_coords is not None

    left, right = ss.product_factors

    # Compute ordinal positions via topological sort of each factor
    left_order = _topo_order(left)
    right_order = _topo_order(right)
    left_rank = {s: i for i, s in enumerate(left_order)}
    right_rank = {s: i for i, s in enumerate(right_order)}

    lines = ['digraph grid {']
    lines.append('  graph [overlap=false, splines=true];')
    lines.append('  node [shape=box, style=filled, fontsize=10];')
    lines.append('  edge [fontsize=8];')
    if title:
        lines.append(f'  label="{_escape_dot(title)}";')
        lines.append('  labelloc=t; fontsize=14;')

    # Place nodes on grid
    for sid in sorted(ss.states):
        coord = ss.product_coords[sid]
        x = right_rank.get(coord[1], 0)
        y = -left_rank.get(coord[0], 0)  # invert Y so top is at top

        lbl = ss.labels.get(sid, str(sid))
        display = _truncate(lbl) if labels else str(sid)

        # Color: blue for top, green for bottom, light gray otherwise
        if sid == ss.top:
            fill = "#dbeafe"
            border = "#2563eb"
        elif sid == ss.bottom:
            fill = "#dcfce7"
            border = "#16a34a"
        else:
            fill = "#f3f4f6"
            border = "#6b7280"

        lines.append(
            f'  {sid} [label="{_escape_dot(display)}", '
            f'pos="{x},{y}!", fillcolor="{fill}", color="{border}"];'
        )

    # Edges
    for src, lbl, tgt in ss.transitions:
        if edge_labels:
            lines.append(f'  {src} -> {tgt} [label="{_escape_dot(lbl)}"];')
        else:
            lines.append(f'  {src} -> {tgt};')

    lines.append('}')
    return '\n'.join(lines)


def render_grid_hasse(
    ss: "StateSpace",
    path: str,
    *,
    fmt: str = "png",
    result: "LatticeResult | None" = None,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
) -> str:
    """Render 2D grid layout to file using ``neato``.

    Returns the output file path.
    """
    dot = grid_dot_source(ss, result, title=title, labels=labels, edge_labels=edge_labels)
    import graphviz
    src = graphviz.Source(dot, engine="neato")
    return src.render(path, format=fmt, cleanup=True)


# ---------------------------------------------------------------------------
# 4b. 3D rendering (3-way products)
# ---------------------------------------------------------------------------

def render_3d_product(
    ss: "StateSpace",
    output_path: str,
    *,
    title: str | None = None,
    labels: bool = True,
) -> str:
    """Generate interactive 3D HTML for a product with 3+ factors.

    The first 3 factors map to spatial X/Y/Z axes.  Extra dimensions
    (4th, 5th, ...) are encoded as a color gradient from warm (top)
    to cool (bottom) so that all product structure remains visible.

    Returns *output_path*.
    """
    _validate_product(ss)
    assert ss.product_factors is not None
    assert ss.product_coords is not None
    if len(ss.product_factors) < 3:
        raise ValueError(
            f"Expected at least 3 factors, got {len(ss.product_factors)}"
        )

    factors = ss.product_factors
    ndim = len(factors)
    orders = [_topo_order(f) for f in factors]
    ranks = [{s: i for i, s in enumerate(order)} for order in orders]
    # max rank per extra dimension (for normalising color)
    extra_maxes = [max(1, len(orders[d]) - 1) for d in range(3, ndim)]

    # Build node data
    nodes = []
    for sid in sorted(ss.states):
        coord = ss.product_coords[sid]
        x = ranks[0].get(coord[0], 0)
        y = ranks[1].get(coord[1], 0)
        z = ranks[2].get(coord[2], 0)
        lbl = ss.labels.get(sid, str(sid)) if labels else str(sid)

        if sid == ss.top:
            color = "#2563eb"
        elif sid == ss.bottom:
            color = "#16a34a"
        elif ndim > 3:
            # Encode extra dims as hue: average normalised position in [0,1]
            # 0 (top) → warm orange, 1 (bottom) → cool purple
            t_vals = [
                ranks[d].get(coord[d], 0) / extra_maxes[d - 3]
                for d in range(3, ndim)
            ]
            t = sum(t_vals) / len(t_vals)
            color = _lerp_color("#f97316", "#7c3aed", t)  # orange→purple
        else:
            color = "#9ca3af"
        nodes.append({"id": sid, "x": x, "y": y, "z": z, "label": lbl, "color": color})

    # Build edge data
    edges = []
    for src, lbl, tgt in ss.transitions:
        edges.append({"src": src, "tgt": tgt, "label": lbl})

    disp_title = _html_escape(title or f"{ndim}D Product Lattice")

    # Factor labels for axes
    axis_labels = []
    for i, f in enumerate(factors):
        n = len(f.states)
        axis_labels.append(f"Factor {i+1} ({n} states)")

    html = []
    html.append("<!DOCTYPE html>")
    html.append(f"<html><head><meta charset='utf-8'><title>{disp_title}</title>")
    html.append("<style>body{margin:0;overflow:hidden;font-family:sans-serif}")
    html.append("#info{position:absolute;top:10px;left:10px;color:#333;background:rgba(255,255,255,0.9);padding:10px;border-radius:6px;font-size:13px}")
    html.append("#tooltip{position:absolute;display:none;background:rgba(0,0,0,0.8);color:#fff;padding:4px 8px;border-radius:4px;font-size:12px;pointer-events:none}")
    html.append("</style></head><body>")
    html.append(f'<div id="info"><b>{disp_title}</b><br>')
    html.append(f'{len(nodes)} states, {len(edges)} transitions<br>')
    html.append(f'X: {_html_escape(axis_labels[0])}<br>')
    html.append(f'Y: {_html_escape(axis_labels[1])}<br>')
    html.append(f'Z: {_html_escape(axis_labels[2])}')
    if ndim > 3:
        extra = ", ".join(_html_escape(axis_labels[d]) for d in range(3, ndim))
        html.append(f'<br>Color: {extra}')
    html.append('</div>')
    html.append('<div id="tooltip"></div>')

    html.append('<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>')
    html.append('<script type="module">')
    html.append('import * as THREE from "three";')
    html.append('import {OrbitControls} from "three/addons/controls/OrbitControls.js";')

    # Embed data
    html.append(f'const nodes = {_js_array(nodes)};')
    html.append(f'const edges = {_js_edges(edges)};')

    html.append("""
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf8f9fa);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);

// Compute center and scale
let cx=0,cy=0,cz=0;
nodes.forEach(n=>{cx+=n.x;cy+=n.y;cz+=n.z});
cx/=nodes.length; cy/=nodes.length; cz/=nodes.length;
let maxR=1;
nodes.forEach(n=>{
  const d=Math.sqrt((n.x-cx)**2+(n.y-cy)**2+(n.z-cz)**2);
  if(d>maxR) maxR=d;
});
const spacing = 2.5;

// Add nodes as spheres
const nodeMeshes = [];
const nodeMap = {};
nodes.forEach(n=>{
  const geo = new THREE.SphereGeometry(0.18, 16, 16);
  const mat = new THREE.MeshPhongMaterial({color: n.color});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set((n.x-cx)*spacing, -(n.y-cy)*spacing, (n.z-cz)*spacing);
  mesh.userData = {label: n.label, id: n.id};
  scene.add(mesh);
  nodeMeshes.push(mesh);
  nodeMap[n.id] = mesh;
});

// Add edges as lines
edges.forEach(e=>{
  const s = nodeMap[e.src], t = nodeMap[e.tgt];
  if(!s||!t) return;
  const geo = new THREE.BufferGeometry().setFromPoints([s.position, t.position]);
  const mat = new THREE.LineBasicMaterial({color: 0x999999, transparent:true, opacity:0.5});
  scene.add(new THREE.Line(geo, mat));
});

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(5, 10, 7);
scene.add(dl);

camera.position.set(maxR*spacing*1.5, maxR*spacing*1.5, maxR*spacing*1.5);
controls.target.set(0, 0, 0);
controls.update();

// Tooltip
const tooltip = document.getElementById('tooltip');
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
renderer.domElement.addEventListener('mousemove', e=>{
  mouse.x = (e.clientX/window.innerWidth)*2-1;
  mouse.y = -(e.clientY/window.innerHeight)*2+1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(nodeMeshes);
  if(hits.length>0){
    tooltip.style.display='block';
    tooltip.style.left=e.clientX+12+'px';
    tooltip.style.top=e.clientY+12+'px';
    tooltip.textContent=hits[0].object.userData.label;
  } else {
    tooltip.style.display='none';
  }
});

window.addEventListener('resize', ()=>{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}
animate();
""")

    html.append('</script></body></html>')

    content = '\n'.join(html)
    if not output_path.endswith('.html'):
        output_path += '.html'
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path


def _js_array(nodes: list[dict]) -> str:
    """Serialize node list to JS array literal."""
    parts = []
    for n in nodes:
        lbl = n["label"].replace("\\", "\\\\").replace('"', '\\"')
        parts.append(
            f'{{id:{n["id"]},x:{n["x"]},y:{n["y"]},z:{n["z"]},'
            f'label:"{lbl}",color:"{n["color"]}"}}'
        )
    return '[' + ','.join(parts) + ']'


def _js_edges(edges: list[dict]) -> str:
    """Serialize edge list to JS array literal."""
    parts = []
    for e in edges:
        lbl = e["label"].replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{{src:{e["src"]},tgt:{e["tgt"]},label:"{lbl}"}}')
    return '[' + ','.join(parts) + ']'


# ---------------------------------------------------------------------------
# 4c. Factored view (side-by-side)
# ---------------------------------------------------------------------------

def factored_dot_source(
    ss: "StateSpace",
    *,
    title: str | None = None,
    labels: bool = True,
    edge_labels: bool = True,
) -> str:
    """DOT source with one ``subgraph cluster`` per factor, side-by-side."""
    _validate_product(ss)
    assert ss.product_factors is not None

    lines = ['digraph factored {']
    lines.append('  rankdir=TB;')
    lines.append('  node [shape=box, style=filled, fontsize=10];')
    lines.append('  edge [fontsize=8];')
    if title:
        lines.append(f'  label="{_escape_dot(title)}";')
        lines.append('  labelloc=t; fontsize=14;')

    for i, factor in enumerate(ss.product_factors):
        cluster_label = f"Factor {i+1} ({len(factor.states)} states)"
        lines.append(f'  subgraph cluster_dim{i} {{')
        lines.append(f'    label="{_escape_dot(cluster_label)}";')
        lines.append(f'    style=rounded; color="#6b7280";')

        for sid in sorted(factor.states):
            lbl = factor.labels.get(sid, str(sid))
            display = _truncate(lbl) if labels else str(sid)
            # Prefix IDs to avoid collision between factors
            node_id = f"f{i}s{sid}"
            if sid == factor.top:
                fill, border = "#dbeafe", "#2563eb"
            elif sid == factor.bottom:
                fill, border = "#dcfce7", "#16a34a"
            else:
                fill, border = "#f3f4f6", "#6b7280"
            lines.append(
                f'    {node_id} [label="{_escape_dot(display)}", '
                f'fillcolor="{fill}", color="{border}"];'
            )

        for src, lbl, tgt in factor.transitions:
            src_id = f"f{i}s{src}"
            tgt_id = f"f{i}s{tgt}"
            if edge_labels:
                lines.append(f'    {src_id} -> {tgt_id} [label="{_escape_dot(lbl)}"];')
            else:
                lines.append(f'    {src_id} -> {tgt_id};')

        lines.append('  }')

    lines.append('}')
    return '\n'.join(lines)


def factored_dashboard(
    ss: "StateSpace",
    output_path: str,
    *,
    title: str | None = None,
) -> str:
    """Render factored side-by-side HTML dashboard with inline SVGs.

    Returns *output_path*.
    """
    _validate_product(ss)
    assert ss.product_factors is not None

    from reticulate.visualize import dot_source as viz_dot_source

    disp_title = _html_escape(title or "Factored Product View")

    html = []
    html.append("<!DOCTYPE html>")
    html.append(f"<html><head><meta charset='utf-8'><title>{disp_title}</title>")
    html.append("<style>")
    html.append("body{font-family:sans-serif;margin:20px;background:#fafafa}")
    html.append("h1{color:#1e293b}")
    html.append(".panel-row{display:flex;gap:20px;flex-wrap:wrap}")
    html.append(".panel{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;flex:1;min-width:300px}")
    html.append(".panel-title{font-weight:bold;margin-bottom:8px;color:#334155}")
    html.append(".panel svg{max-width:100%;height:auto}")
    html.append(".stats{margin-bottom:16px;color:#64748b}")
    html.append("</style></head><body>")
    html.append(f'<h1>{disp_title}</h1>')
    html.append(f'<div class="stats">{len(ss.product_factors)} factors, '
                f'{len(ss.states)} product states, '
                f'{len(ss.transitions)} product transitions</div>')
    html.append('<div class="panel-row">')

    for i, factor in enumerate(ss.product_factors):
        # Generate DOT for each factor using the standard visualize module
        dot = viz_dot_source(factor, title=f"Factor {i+1}")
        svg = _dot_to_svg(dot)
        html.append(
            f'<div class="panel"><div class="panel-title">Factor {i+1} '
            f'({len(factor.states)} states, {len(factor.transitions)} transitions)'
            f'</div>{svg}</div>'
        )

    html.append('</div></body></html>')

    content = '\n'.join(html)
    if not output_path.endswith('.html'):
        output_path += '.html'
    with open(output_path, 'w') as f:
        f.write(content)
    return output_path
