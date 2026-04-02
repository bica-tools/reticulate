"""Deep Java project scanner — extract session types and find N₅ (Step 80g).

Scans real Java projects to:
1. Identify stateful classes (have lifecycle: open/close, init/destroy, etc.)
2. Extract L2 session types from method call patterns across ALL clients
3. Check lattice classification (distributive / modular / non-modular)
4. Find N₅ witnesses and trace them back to source code
5. Propose refactoring when non-modularity is found

This is the full detection cycle: scan → extract → classify → diagnose → propose fix.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import Counter, defaultdict

try:
    import javalang
    from javalang.tree import (
        ClassDeclaration, MethodDeclaration, MethodInvocation,
        FieldDeclaration, LocalVariableDeclaration, FormalParameter,
    )
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive
from reticulate.protocol_mining import mine_from_traces
from reticulate.type_inference import Trace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassProfile:
    """Profile of a Java class discovered during scanning."""
    name: str
    file_path: str
    package: str
    public_methods: list[str]
    has_close: bool
    has_lifecycle: bool  # has init/open + close/destroy pattern
    client_count: int  # how many other classes use this one
    client_traces: list[list[str]]  # method call sequences from clients


@dataclass(frozen=True)
class SessionTypeResult:
    """Result of extracting and analyzing a session type for a class."""
    class_name: str
    session_type: str
    num_states: int
    num_transitions: int
    is_lattice: bool
    classification: str  # "boolean", "distributive", "modular", "lattice", "non-lattice"
    has_m3: bool
    has_n5: bool
    m3_witness: Optional[tuple]
    n5_witness: Optional[tuple]
    client_count: int
    trace_count: int


@dataclass(frozen=True)
class N5Finding:
    """A non-modularity finding with code-level diagnosis."""
    class_name: str
    file_path: str
    session_type: str
    n5_witness: tuple  # (top, a, b, c, bottom) state IDs
    n5_methods: dict  # state ID → method labels at that state
    diagnosis: str  # human-readable explanation
    refactoring_suggestion: str


@dataclass(frozen=True)
class DeepScanReport:
    """Complete report from scanning a Java project."""
    project_name: str
    project_dir: str
    java_files: int
    classes_found: int
    stateful_classes: int
    session_types_extracted: int
    classification_counts: dict[str, int]
    n5_findings: list[N5Finding]
    results: list[SessionTypeResult]


# ---------------------------------------------------------------------------
# Phase 1: Discover classes and their clients
# ---------------------------------------------------------------------------

def _discover_classes(project_dir: str) -> dict[str, ClassProfile]:
    """Scan all Java files to build class profiles."""
    if not HAS_JAVALANG:
        raise ImportError("javalang required: pip install javalang")

    classes: dict[str, ClassProfile] = {}
    all_sources: list[tuple[str, str]] = []  # (path, source)

    # First pass: collect all classes and their methods
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in
                   ('target', 'build', '.git', '.gradle', 'test', 'tests',
                    'testFixtures', 'jmh', 'benchmark')]
        for fname in files:
            if not fname.endswith('.java'):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    source = f.read()
                all_sources.append((fpath, source))
                tree = javalang.parse.parse(source)
                pkg = tree.package.name if tree.package else ""

                for _, node in tree.filter(ClassDeclaration):
                    methods = []
                    for _, m in node.filter(MethodDeclaration):
                        mods = m.modifiers or set()
                        if 'public' in mods or not mods:
                            methods.append(m.name)

                    has_close = any(m in ('close', 'shutdown', 'destroy',
                                         'dispose', 'release', 'disconnect')
                                   for m in methods)
                    has_open = any(m in ('open', 'init', 'initialize', 'connect',
                                        'start', 'begin', 'create', 'acquire')
                                  for m in methods)

                    classes[node.name] = ClassProfile(
                        name=node.name,
                        file_path=fpath,
                        package=pkg,
                        public_methods=methods,
                        has_close=has_close,
                        has_lifecycle=has_open and has_close,
                        client_count=0,
                        client_traces=[],
                    )
            except Exception:
                continue

    # Second pass: find clients (who calls methods on instances of each class)
    class_clients: dict[str, list[list[str]]] = defaultdict(list)
    class_client_counts: dict[str, int] = Counter()

    for fpath, source in all_sources:
        try:
            tree = javalang.parse.parse(source)
            for _, cls_node in tree.filter(ClassDeclaration):
                # Find local variables and parameters typed as known classes
                var_types: dict[str, str] = {}

                for _, param in cls_node.filter(FormalParameter):
                    if hasattr(param.type, 'name') and param.type.name in classes:
                        var_types[param.name] = param.type.name

                for _, local in cls_node.filter(LocalVariableDeclaration):
                    if hasattr(local.type, 'name') and local.type.name in classes:
                        for decl in local.declarators:
                            var_types[decl.name] = local.type.name

                # Extract method calls per function
                for _, method_node in cls_node.filter(MethodDeclaration):
                    calls_by_type: dict[str, list[str]] = defaultdict(list)

                    for _, inv in method_node.filter(MethodInvocation):
                        if inv.qualifier and inv.qualifier in var_types:
                            target_class = var_types[inv.qualifier]
                            calls_by_type[target_class].append(inv.member)

                    for target_class, calls in calls_by_type.items():
                        if calls:
                            class_clients[target_class].append(calls)
                            class_client_counts[target_class] += 1
        except Exception:
            continue

    # Update profiles with client info
    updated: dict[str, ClassProfile] = {}
    for name, profile in classes.items():
        updated[name] = ClassProfile(
            name=profile.name,
            file_path=profile.file_path,
            package=profile.package,
            public_methods=profile.public_methods,
            has_close=profile.has_close,
            has_lifecycle=profile.has_lifecycle,
            client_count=class_client_counts.get(name, 0),
            client_traces=class_clients.get(name, []),
        )

    return updated


# ---------------------------------------------------------------------------
# Phase 2: Extract session types from client usage patterns
# ---------------------------------------------------------------------------

def _extract_session_type(profile: ClassProfile) -> Optional[SessionTypeResult]:
    """Extract session type from client traces and check classification."""
    if not profile.client_traces:
        return None

    try:
        # Use protocol mining to infer session type from traces
        traces = [Trace.from_labels(t) for t in profile.client_traces]
        mining_result = mine_from_traces(traces)
        st_str = mining_result.session_type

        if not st_str or st_str == 'end':
            return None

        ast = parse(st_str)
        ss = build_statespace(ast)
        lr = check_lattice(ss)

        if not lr.is_lattice:
            return SessionTypeResult(
                class_name=profile.name, session_type=st_str,
                num_states=len(ss.states), num_transitions=len(ss.transitions),
                is_lattice=False, classification="non-lattice",
                has_m3=False, has_n5=False, m3_witness=None, n5_witness=None,
                client_count=profile.client_count,
                trace_count=len(profile.client_traces),
            )

        dr = check_distributive(ss)
        return SessionTypeResult(
            class_name=profile.name, session_type=st_str,
            num_states=len(ss.states), num_transitions=len(ss.transitions),
            is_lattice=True, classification=dr.classification,
            has_m3=dr.has_m3, has_n5=dr.has_n5,
            m3_witness=dr.m3_witness, n5_witness=dr.n5_witness,
            client_count=profile.client_count,
            trace_count=len(profile.client_traces),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 3: Diagnose N₅ findings
# ---------------------------------------------------------------------------

def _diagnose_n5(profile: ClassProfile, result: SessionTypeResult) -> Optional[N5Finding]:
    """Diagnose a non-modularity finding with code-level explanation."""
    if not result.has_n5 or not result.n5_witness:
        return None

    try:
        ast = parse(result.session_type)
        ss = build_statespace(ast)

        # Map state IDs to their available methods
        state_methods: dict[int, list[str]] = {}
        for s in ss.states:
            methods = [label for src, label, _ in ss.transitions if src == s]
            state_methods[s] = methods

        w = result.n5_witness
        n5_methods = {s: state_methods.get(s, []) for s in w}

        # Generate diagnosis
        # N₅ = chain a > b > bottom with c incomparable to b
        # This means: some method is available at state a but not at c (asymmetry)
        diagnosis = (
            f"Class {profile.name} has asymmetric method availability (N₅). "
            f"The N₅ witness involves states {w}. "
            f"This means some methods require specific prior calls that other "
            f"methods do not — creating an ordering dependency that breaks modularity. "
        )

        # Check which methods create the asymmetry
        if len(w) >= 3:
            chain_methods = set(state_methods.get(w[1], []))
            side_methods = set(state_methods.get(w[2], []) if len(w) > 2 else [])
            asymmetric = chain_methods - side_methods
            if asymmetric:
                diagnosis += (
                    f"Specifically, methods {asymmetric} are available after one path "
                    f"but not another. "
                )

        refactoring = (
            f"Consider splitting {profile.name} into phases: "
            f"(1) a setup phase with unrestricted methods, "
            f"(2) an active phase where state-dependent methods are grouped, "
            f"(3) a cleanup phase. "
            f"Alternatively, use the Builder pattern to enforce ordering, "
            f"or introduce a state enum that makes the current phase explicit."
        )

        return N5Finding(
            class_name=profile.name,
            file_path=profile.file_path,
            session_type=result.session_type,
            n5_witness=w,
            n5_methods=n5_methods,
            diagnosis=diagnosis,
            refactoring_suggestion=refactoring,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def deep_scan(project_dir: str, project_name: str = "") -> DeepScanReport:
    """Perform a deep scan of a Java project.

    Discovers classes, extracts session types from client usage,
    classifies lattice structure, and diagnoses N₅ findings.
    """
    if not project_name:
        project_name = os.path.basename(project_dir)

    # Phase 1: Discover
    profiles = _discover_classes(project_dir)
    java_files = sum(1 for _ in _walk_java(project_dir))

    # Phase 2: Extract session types for classes with clients
    results: list[SessionTypeResult] = []
    for name, profile in profiles.items():
        if profile.client_count >= 2:  # at least 2 client methods
            r = _extract_session_type(profile)
            if r:
                results.append(r)

    # Phase 3: Diagnose N₅
    n5_findings: list[N5Finding] = []
    for r in results:
        if r.has_n5:
            profile = profiles.get(r.class_name)
            if profile:
                finding = _diagnose_n5(profile, r)
                if finding:
                    n5_findings.append(finding)

    # Classify
    class_counts = Counter(r.classification for r in results)
    stateful = sum(1 for p in profiles.values() if p.has_lifecycle)

    return DeepScanReport(
        project_name=project_name,
        project_dir=project_dir,
        java_files=java_files,
        classes_found=len(profiles),
        stateful_classes=stateful,
        session_types_extracted=len(results),
        classification_counts=dict(class_counts),
        n5_findings=n5_findings,
        results=results,
    )


def _walk_java(project_dir: str):
    """Yield all .java file paths in a project."""
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in
                   ('target', 'build', '.git', '.gradle', 'test', 'tests')]
        for f in files:
            if f.endswith('.java'):
                yield os.path.join(root, f)


def format_deep_report(report: DeepScanReport) -> str:
    """Format a deep scan report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"  DEEP SCAN: {report.project_name}")
    lines.append("=" * 80)
    lines.append(f"  Directory: {report.project_dir}")
    lines.append(f"  Java files: {report.java_files}")
    lines.append(f"  Classes discovered: {report.classes_found}")
    lines.append(f"  Stateful classes (lifecycle): {report.stateful_classes}")
    lines.append(f"  Session types extracted: {report.session_types_extracted}")
    lines.append("")

    if report.classification_counts:
        lines.append("  Classification:")
        for cls, count in sorted(report.classification_counts.items(),
                                  key=lambda x: -x[1]):
            lines.append(f"    {cls}: {count}")

    n5_count = len(report.n5_findings)
    m3_count = sum(1 for r in report.results if r.has_m3)
    lines.append("")
    lines.append(f"  Non-modular (N₅): {n5_count}")
    lines.append(f"  Non-distributive (M₃): {m3_count}")

    if report.n5_findings:
        lines.append("")
        lines.append("  === N₅ FINDINGS ===")
        for f in report.n5_findings:
            lines.append(f"")
            lines.append(f"  {f.class_name} ({f.file_path})")
            lines.append(f"    Session type: {f.session_type[:80]}...")
            lines.append(f"    N₅ witness: {f.n5_witness}")
            lines.append(f"    Diagnosis: {f.diagnosis[:120]}...")
            lines.append(f"    Refactoring: {f.refactoring_suggestion[:120]}...")

    if report.results:
        lines.append("")
        lines.append("  === ALL RESULTS ===")
        lines.append(f"  {'Class':<30} {'St':>3} {'Tr':>3} {'Class':<14} "
                     f"{'M3':>3} {'N5':>3} {'Clients':>8}")
        lines.append("  " + "-" * 70)
        for r in sorted(report.results, key=lambda x: x.classification):
            m3 = "Y" if r.has_m3 else "-"
            n5 = "Y" if r.has_n5 else "-"
            lines.append(f"  {r.class_name:<30} {r.num_states:>3} "
                         f"{r.num_transitions:>3} {r.classification:<14} "
                         f"{m3:>3} {n5:>3} {r.client_count:>8}")

    lines.append("=" * 80)
    return "\n".join(lines)
