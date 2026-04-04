"""Universe of Lattices — transitive session type analysis (Step 80f).

A program is a universe of lattices: one lattice per tracked object.
When class A uses class B which uses class C, we:
  1. Extract L(C) — session type for C from its clients
  2. Extract L(B) — session type for B, including its usage of C
  3. Extract L(A) — session type for A, including its usage of B
  4. Verify B uses C properly (B's traces on C conform to L(C))
  5. Verify A uses B properly (A's traces on B conform to L(B))

The universe is the collection of all per-class lattices plus the
inter-lattice conformance edges (morphisms between them).

Constructors have different "gravitational mass" in this universe:
  - ∥ (parallel) = heavy — product lattice (exponential growth)
  - & (branch) = medium — tree lattice (linear per branch)
  - + (select) = medium — choice points
  - rec (recursion) = creates cycles (infinite potential)
  - end = singularity (collapses to a point)
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import javalang
    from javalang.tree import (
        ClassDeclaration, InterfaceDeclaration,
        MethodDeclaration, MethodInvocation,
        FieldDeclaration, LocalVariableDeclaration, FormalParameter,
    )
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False

from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive
from reticulate.type_inference import Trace, infer_from_traces
from reticulate.conformance_checker import check_conformance


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CallEdge:
    """An inter-class method call."""
    caller_class: str
    caller_method: str
    callee_class: str
    callee_method: str
    file_path: str
    line_number: int


@dataclass(frozen=True)
class ClassLattice:
    """Session type lattice for one class in the universe."""
    class_name: str
    session_type: str
    num_states: int
    num_transitions: int
    is_lattice: bool
    classification: str
    source: str  # "mined", "preloaded", "inferred"
    trace_count: int
    method_count: int


@dataclass(frozen=True)
class ConformanceEdge:
    """Conformance result for one caller→callee relationship."""
    caller_class: str
    callee_class: str
    num_traces: int
    conforming: int
    violating: int
    incomplete: int
    conformance_rate: float
    first_violation: str  # summary of first violation found


@dataclass(frozen=True)
class UniverseReport:
    """Complete universe-of-lattices analysis for one Java project."""
    project_name: str
    project_dir: str
    java_files: int
    classes_found: int
    classes_parsed: int
    call_edges: int
    classes_with_types: int
    classes_with_lattices: int
    lattice_rate: float
    classification_counts: dict[str, int]
    lattices: dict[str, ClassLattice]
    conformance_edges: list[ConformanceEdge]
    conformance_rate: float
    preloaded_matches: list[str]


# ---------------------------------------------------------------------------
# JDK preloaded session types
# ---------------------------------------------------------------------------

_JDK_TYPES_CACHE: dict[str, str] | None = None


def _preload_jdk_types() -> dict[str, str]:
    """Load session types for known JDK classes."""
    global _JDK_TYPES_CACHE
    if _JDK_TYPES_CACHE is not None:
        return _JDK_TYPES_CACHE

    from reticulate.java_api_extractor import JAVA_API_SPECS, extract_session_type
    result: dict[str, str] = {}
    for api_name in JAVA_API_SPECS:
        try:
            er = extract_session_type(api_name)
            short = api_name.split(".")[-1]
            result[short] = er.inferred_type
        except Exception:
            pass
    _JDK_TYPES_CACHE = result
    return result


# ---------------------------------------------------------------------------
# Phase 1: Scan project, discover classes
# ---------------------------------------------------------------------------

def _scan_java_files(project_dir: str) -> list[tuple[str, Any]]:
    """Parse all .java files, return list of (path, javalang tree)."""
    if not HAS_JAVALANG:
        raise ImportError("javalang required: pip install javalang")

    results: list[tuple[str, Any]] = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in
                   ('target', 'build', '.git', '.gradle', 'node_modules',
                    '.idea', '.mvn', 'generated-sources')]
        for fname in files:
            if not fname.endswith('.java'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    source = f.read()
                tree = javalang.parse.parse(source)
                results.append((fpath, tree))
            except Exception:
                continue
    return results


def _discover_classes(parsed_files: list[tuple[str, Any]]) -> dict[str, dict]:
    """First pass: collect all classes and their public methods."""
    classes: dict[str, dict] = {}
    for fpath, tree in parsed_files:
        pkg = tree.package.name if tree.package else ""
        for _, node in tree.filter(ClassDeclaration):
            methods = []
            for _, m in node.filter(MethodDeclaration):
                mods = m.modifiers or set()
                if 'public' in mods or not mods:
                    methods.append(m.name)
            classes[node.name] = {
                'name': node.name,
                'file_path': fpath,
                'package': pkg,
                'methods': methods,
            }
    return classes


# ---------------------------------------------------------------------------
# Phase 2: Build call graph
# ---------------------------------------------------------------------------

def _build_call_graph(
    parsed_files: list[tuple[str, Any]],
    known_classes: dict[str, dict],
    jdk_types: dict[str, str],
) -> tuple[list[CallEdge], dict[tuple[str, str], list[list[str]]]]:
    """Build inter-class call graph and collect per-edge traces.

    Returns:
        edges: list of CallEdge
        traces: dict mapping (caller_class, callee_class) → list of method call sequences
    """
    all_known = set(known_classes.keys()) | set(jdk_types.keys())
    edges: list[CallEdge] = []
    traces: dict[tuple[str, str], list[list[str]]] = defaultdict(list)

    for fpath, tree in parsed_files:
        try:
            for _, cls_node in tree.filter(ClassDeclaration):
                caller_class = cls_node.name

                # Resolve variable types to known classes
                # Strategy: params, locals, fields, AND constructor calls (new X())
                var_types: dict[str, str] = {}

                for _, param in cls_node.filter(FormalParameter):
                    if hasattr(param.type, 'name') and param.type.name in all_known:
                        var_types[param.name] = param.type.name

                for _, local in cls_node.filter(LocalVariableDeclaration):
                    type_name = getattr(local.type, 'name', None)
                    if type_name and type_name in all_known:
                        for decl in local.declarators:
                            var_types[decl.name] = type_name
                    else:
                        # Check if initializer is `new KnownClass(...)`
                        for decl in local.declarators:
                            init = decl.initializer
                            if init and hasattr(init, 'type'):
                                ctor_type = getattr(init.type, 'name', None)
                                if ctor_type and ctor_type in all_known:
                                    var_types[decl.name] = ctor_type

                for _, fd in cls_node.filter(FieldDeclaration):
                    type_name = getattr(fd.type, 'name', None)
                    if type_name and type_name in all_known:
                        for decl in fd.declarators:
                            var_types[decl.name] = type_name

                # Also track `this` for self-calls when the class itself is known
                if caller_class in all_known:
                    var_types['this'] = caller_class

                # Extract method calls per method body
                for _, method_node in cls_node.filter(MethodDeclaration):
                    calls_by_callee: dict[str, list[str]] = defaultdict(list)
                    caller_method = method_node.name

                    # Method-local variable tracking (re-scan for tighter scope)
                    local_vars = dict(var_types)  # copy class-level vars
                    for _, local in method_node.filter(LocalVariableDeclaration):
                        type_name = getattr(local.type, 'name', None)
                        if type_name and type_name in all_known:
                            for decl in local.declarators:
                                local_vars[decl.name] = type_name
                        else:
                            for decl in local.declarators:
                                init = decl.initializer
                                if init and hasattr(init, 'type'):
                                    ctor_type = getattr(init.type, 'name', None)
                                    if ctor_type and ctor_type in all_known:
                                        local_vars[decl.name] = ctor_type

                    for _, inv in method_node.filter(MethodInvocation):
                        qualifier = inv.qualifier
                        if qualifier and qualifier in local_vars:
                            callee_class = local_vars[qualifier]
                            callee_method = inv.member
                            calls_by_callee[callee_class].append(callee_method)

                            line = inv.position.line if inv.position else 0
                            edges.append(CallEdge(
                                caller_class=caller_class,
                                caller_method=caller_method,
                                callee_class=callee_class,
                                callee_method=callee_method,
                                file_path=fpath,
                                line_number=line,
                            ))

                    for callee, methods in calls_by_callee.items():
                        if methods and callee != caller_class:
                            traces[(caller_class, callee)].append(methods)

        except Exception:
            continue

    return edges, dict(traces)


# ---------------------------------------------------------------------------
# Phase 3: Bottom-up session type extraction
# ---------------------------------------------------------------------------

def _topological_order(
    classes: set[str],
    traces: dict[tuple[str, str], list[list[str]]],
) -> list[str]:
    """Topological sort: callees before callers (bottom-up)."""
    # Build adjacency: caller → set of callees
    callees_of: dict[str, set[str]] = defaultdict(set)
    for (caller, callee) in traces:
        if caller in classes and callee in classes:
            callees_of[caller].add(callee)

    # Kahn's algorithm
    in_degree: dict[str, int] = {c: 0 for c in classes}
    for caller, deps in callees_of.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[caller] = in_degree.get(caller, 0)  # caller depends on dep

    # Recompute: in_degree[x] = number of classes x depends on (callees)
    in_degree = {c: 0 for c in classes}
    for caller, deps in callees_of.items():
        in_degree[caller] = len(deps & classes)

    queue = [c for c in classes if in_degree[c] == 0]
    order: list[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        # Find callers of node and decrease their in-degree
        for caller, deps in callees_of.items():
            if node in deps:
                in_degree[caller] -= 1
                if in_degree[caller] == 0 and caller not in order:
                    queue.append(caller)

    # Add remaining (cycles) at the end
    for c in classes:
        if c not in order:
            order.append(c)

    return order


def _extract_types_bottom_up(
    known_classes: dict[str, dict],
    traces: dict[tuple[str, str], list[list[str]]],
    jdk_types: dict[str, str],
) -> dict[str, ClassLattice]:
    """Extract session types bottom-up following topological order."""
    lattices: dict[str, ClassLattice] = {}

    # Preloaded JDK types
    for short_name, st_str in jdk_types.items():
        try:
            ast = parse(st_str)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            cls = "non-lattice"
            if lr.is_lattice:
                try:
                    dr = check_distributive(ss)
                    cls = dr.classification
                except Exception:
                    cls = "lattice"
            lattices[short_name] = ClassLattice(
                class_name=short_name,
                session_type=st_str,
                num_states=len(ss.states),
                num_transitions=len(ss.transitions),
                is_lattice=lr.is_lattice,
                classification=cls,
                source="preloaded",
                trace_count=0,
                method_count=0,
            )
        except Exception:
            continue

    # Collect all client traces for each class
    class_traces: dict[str, list[list[str]]] = defaultdict(list)
    for (caller, callee), trace_list in traces.items():
        class_traces[callee].extend(trace_list)

    # Topological order
    all_classes = set(known_classes.keys())
    order = _topological_order(all_classes, traces)

    for class_name in order:
        if class_name in lattices:
            continue  # already preloaded

        all_traces = class_traces.get(class_name, [])
        if len(all_traces) < 2:
            continue  # not enough traces

        # Filter: need at least 2 distinct methods
        all_methods = set()
        for t in all_traces:
            all_methods.update(t)
        if len(all_methods) < 2:
            continue

        try:
            trace_objs = [Trace.from_labels(t) for t in all_traces]
            inferred = infer_from_traces(trace_objs)
            st_str = pretty(inferred)

            ast = parse(st_str)
            ss = build_statespace(ast)
            lr = check_lattice(ss)

            cls = "non-lattice"
            if lr.is_lattice:
                try:
                    dr = check_distributive(ss)
                    cls = dr.classification
                except Exception:
                    cls = "lattice"

            lattices[class_name] = ClassLattice(
                class_name=class_name,
                session_type=st_str,
                num_states=len(ss.states),
                num_transitions=len(ss.transitions),
                is_lattice=lr.is_lattice,
                classification=cls,
                source="mined",
                trace_count=len(all_traces),
                method_count=len(all_methods),
            )
        except Exception:
            continue

    return lattices


# ---------------------------------------------------------------------------
# Phase 4: Transitive conformance checking
# ---------------------------------------------------------------------------

def _check_transitive_conformance(
    lattices: dict[str, ClassLattice],
    traces: dict[tuple[str, str], list[list[str]]],
) -> list[ConformanceEdge]:
    """Check that each caller uses its callee according to the callee's type."""
    results: list[ConformanceEdge] = []

    for (caller, callee), trace_list in traces.items():
        if callee not in lattices:
            continue

        callee_type = lattices[callee].session_type
        conforming = 0
        violating = 0
        incomplete = 0
        first_violation = ""

        for trace in trace_list:
            try:
                cr = check_conformance(callee_type, trace)
                if cr.conforms and cr.is_complete:
                    conforming += 1
                elif cr.conforms:
                    incomplete += 1
                else:
                    violating += 1
                    if not first_violation and cr.violations:
                        v = cr.violations[0]
                        first_violation = (
                            f"{caller}.* calls {callee}: "
                            f"'{v.method}' not available at step {v.step_index}"
                        )
            except Exception:
                incomplete += 1

        total = conforming + violating + incomplete
        if total > 0:
            results.append(ConformanceEdge(
                caller_class=caller,
                callee_class=callee,
                num_traces=total,
                conforming=conforming,
                violating=violating,
                incomplete=incomplete,
                conformance_rate=conforming / total if total else 0.0,
                first_violation=first_violation,
            ))

    return results


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze_project(project_dir: str, project_name: str = "") -> UniverseReport:
    """Analyze a Java project as a universe of lattices.

    Pipeline: scan → call graph → bottom-up extraction → conformance.
    """
    if not project_name:
        project_name = os.path.basename(project_dir)

    jdk_types = _preload_jdk_types()

    # Phase 1: Scan
    parsed = _scan_java_files(project_dir)
    classes = _discover_classes(parsed)

    # Phase 2: Call graph
    edges, traces = _build_call_graph(parsed, classes, jdk_types)

    # Phase 3: Bottom-up extraction
    lattices = _extract_types_bottom_up(classes, traces, jdk_types)

    # Phase 4: Conformance
    conf_edges = _check_transitive_conformance(lattices, traces)

    # Statistics
    mined = {k: v for k, v in lattices.items() if v.source != "preloaded"}
    preloaded_matches = [k for k, v in lattices.items() if v.source == "preloaded"
                         and any(k == callee for (_, callee) in traces)]

    lattice_count = sum(1 for v in mined.values() if v.is_lattice)
    total_mined = len(mined)
    lattice_rate = lattice_count / total_mined if total_mined else 0.0

    cls_counts: dict[str, int] = defaultdict(int)
    for v in mined.values():
        cls_counts[v.classification] += 1

    total_conf = sum(e.num_traces for e in conf_edges)
    total_conform = sum(e.conforming for e in conf_edges)
    conf_rate = total_conform / total_conf if total_conf else 0.0

    return UniverseReport(
        project_name=project_name,
        project_dir=project_dir,
        java_files=len(parsed),
        classes_found=len(classes),
        classes_parsed=len(parsed),
        call_edges=len(edges),
        classes_with_types=len(mined),
        classes_with_lattices=lattice_count,
        lattice_rate=lattice_rate,
        classification_counts=dict(cls_counts),
        lattices=lattices,
        conformance_edges=conf_edges,
        conformance_rate=conf_rate,
        preloaded_matches=preloaded_matches,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_universe_report(report: UniverseReport) -> str:
    """Format a universe-of-lattices report."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"  UNIVERSE OF LATTICES — {report.project_name}")
    lines.append("=" * 100)
    lines.append(f"  Java files:           {report.java_files}")
    lines.append(f"  Classes found:        {report.classes_found}")
    lines.append(f"  Call graph edges:     {report.call_edges}")
    lines.append(f"  Session types mined:  {report.classes_with_types}")
    lines.append(f"  Lattices:             {report.classes_with_lattices}")
    lines.append(f"  Lattice rate:         {report.lattice_rate:.0%}")
    lines.append(f"  JDK types matched:    {len(report.preloaded_matches)}")
    lines.append(f"  Conformance edges:    {len(report.conformance_edges)}")
    lines.append(f"  Conformance rate:     {report.conformance_rate:.0%}")
    lines.append("")

    # Classification breakdown
    if report.classification_counts:
        lines.append("  Classification:")
        for cls, count in sorted(report.classification_counts.items()):
            lines.append(f"    {cls}: {count}")
        lines.append("")

    # Lattice details (mined only, sorted by states desc)
    mined = [(k, v) for k, v in report.lattices.items() if v.source != "preloaded"]
    mined.sort(key=lambda x: x[1].num_states, reverse=True)

    if mined:
        lines.append(f"  {'Class':<30} {'States':>6} {'Trans':>6} {'Lattice':>8} "
                      f"{'Class':>14} {'Traces':>7}")
        lines.append("  " + "-" * 75)
        for name, lat in mined[:30]:
            l = "YES" if lat.is_lattice else "NO"
            lines.append(f"  {name:<30} {lat.num_states:>6} {lat.num_transitions:>6} "
                          f"{l:>8} {lat.classification:>14} {lat.trace_count:>7}")
        if len(mined) > 30:
            lines.append(f"  ... and {len(mined) - 30} more")
        lines.append("")

    # Conformance edges with violations
    violations = [e for e in report.conformance_edges if e.violating > 0]
    if violations:
        lines.append(f"  VIOLATIONS DETECTED ({len(violations)} edges):")
        lines.append("  " + "-" * 75)
        for e in violations[:20]:
            lines.append(f"  {e.caller_class} → {e.callee_class}: "
                          f"{e.violating}/{e.num_traces} violating")
            if e.first_violation:
                lines.append(f"    → {e.first_violation}")
        lines.append("")

    lines.append("=" * 100)
    return "\n".join(lines)
