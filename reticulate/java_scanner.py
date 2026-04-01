"""Scan Java projects for protocol conformance violations (Step 80e).

Uses the ``javalang`` library to parse Java source files and extract
method call sequences on target types, then checks conformance against
session type protocols.

Pipeline:
  Java project dir → parse .java files → find method calls on target type
  → extract call sequences per method body → check against session type

This works on ANY Java project — no annotations required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import javalang
    from javalang.tree import (
        MethodInvocation,
        MethodDeclaration,
        ClassDeclaration,
        CompilationUnit,
    )
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False

from reticulate.conformance_checker import check_conformance, ConformanceResult


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JavaClient:
    """A method call sequence extracted from a Java method."""
    file_path: str
    class_name: str
    method_name: str
    line_number: int
    target_var: str
    methods: list[str]


@dataclass(frozen=True)
class JavaScanResult:
    """Result of scanning a Java method for conformance."""
    client: JavaClient
    conformance: ConformanceResult
    status: str  # "PASS", "FAIL", "INCOMPLETE"


@dataclass(frozen=True)
class JavaProjectReport:
    """Report from scanning an entire Java project."""
    project_dir: str
    session_type: str
    target_vars: list[str]
    java_files_scanned: int
    classes_found: int
    methods_scanned: int
    clients_found: int
    conforming: int
    violating: int
    incomplete: int
    results: list[JavaScanResult]
    conformance_rate: float
    violation_details: list[str]


# ---------------------------------------------------------------------------
# Java AST extraction
# ---------------------------------------------------------------------------

def _extract_method_calls(method_body: Any, target_vars: list[str]) -> dict[str, list[str]]:
    """Extract method calls on target variables from a Java method AST.

    Walks the AST looking for MethodInvocation nodes where the qualifier
    matches one of the target variable names.
    """
    calls_by_var: dict[str, list[str]] = {v: [] for v in target_vars}

    if method_body is None:
        return calls_by_var

    # Walk all nodes in the method body
    for path, node in method_body:
        if isinstance(node, MethodInvocation):
            qualifier = node.qualifier
            if qualifier and qualifier in target_vars:
                calls_by_var[qualifier].append(node.member)

    return calls_by_var


def extract_clients_from_java_file(file_path: str,
                                    target_vars: list[str]) -> list[JavaClient]:
    """Extract client call sequences from a Java source file."""
    if not HAS_JAVALANG:
        raise ImportError("javalang package required: pip install javalang")

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = javalang.parse.parse(source)
    except (javalang.parser.JavaSyntaxError, Exception):
        return []

    clients: list[JavaClient] = []

    for _, class_node in tree.filter(ClassDeclaration):
        class_name = class_node.name
        for _, method_node in class_node.filter(MethodDeclaration):
            method_name = method_node.name
            line_no = method_node.position.line if method_node.position else 0

            # Extract calls on target vars
            calls_by_var = _extract_method_calls(method_node, target_vars)

            for var, calls in calls_by_var.items():
                if calls:
                    clients.append(JavaClient(
                        file_path=file_path,
                        class_name=class_name,
                        method_name=method_name,
                        line_number=line_no,
                        target_var=var,
                        methods=calls,
                    ))

    return clients


def extract_clients_from_java_dir(project_dir: str,
                                   target_vars: list[str]) -> tuple[list[JavaClient], int, int]:
    """Extract all client traces from all Java files in a directory.

    Returns (clients, files_scanned, classes_found).
    """
    clients: list[JavaClient] = []
    files_scanned = 0
    classes_found = 0

    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')
                   and d not in ('target', 'build', '.gradle', 'node_modules')]
        for fname in files:
            if fname.endswith(".java"):
                fpath = os.path.join(root, fname)
                files_scanned += 1
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        source = f.read()
                    tree = javalang.parse.parse(source)
                    for _, cn in tree.filter(ClassDeclaration):
                        classes_found += 1
                except Exception:
                    pass
                file_clients = extract_clients_from_java_file(fpath, target_vars)
                clients.extend(file_clients)

    return clients, files_scanned, classes_found


# ---------------------------------------------------------------------------
# Project scanning
# ---------------------------------------------------------------------------

def scan_java_project(project_dir: str,
                      target_vars: list[str],
                      session_type: str) -> JavaProjectReport:
    """Scan a Java project for protocol conformance violations.

    Parameters
    ----------
    project_dir : str
        Path to the Java source directory.
    target_vars : list[str]
        Variable names to track (e.g., ["conn", "stmt", "rs"]).
    session_type : str
        L2/L3 session type defining the protocol.
    """
    clients, files_scanned, classes_found = extract_clients_from_java_dir(
        project_dir, target_vars)

    results: list[JavaScanResult] = []
    for client in clients:
        cr = check_conformance(session_type, client.methods)
        status = "PASS" if cr.conforms and cr.is_complete else \
                 "INCOMPLETE" if cr.conforms else "FAIL"
        results.append(JavaScanResult(client=client, conformance=cr, status=status))

    conforming = sum(1 for r in results if r.status == "PASS")
    violating = sum(1 for r in results if r.status == "FAIL")
    incomplete = sum(1 for r in results if r.status == "INCOMPLETE")
    total = len(results)

    violation_details: list[str] = []
    for r in results:
        if r.status == "FAIL":
            for v in r.conformance.violations:
                detail = (f"{r.client.file_path}:{r.client.line_number} "
                          f"{r.client.class_name}.{r.client.method_name}(): {v}")
                violation_details.append(detail)

    return JavaProjectReport(
        project_dir=project_dir,
        session_type=session_type,
        target_vars=target_vars,
        java_files_scanned=files_scanned,
        classes_found=classes_found,
        methods_scanned=total,
        clients_found=total,
        conforming=conforming,
        violating=violating,
        incomplete=incomplete,
        results=results,
        conformance_rate=conforming / total if total > 0 else 1.0,
        violation_details=violation_details,
    )


def format_java_report(report: JavaProjectReport) -> str:
    """Format a Java project scan report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  JAVA PROJECT CONFORMANCE SCAN REPORT")
    lines.append("=" * 80)
    lines.append(f"  Directory: {report.project_dir}")
    lines.append(f"  Target variables: {', '.join(report.target_vars)}")
    lines.append(f"  Protocol: {report.session_type[:70]}...")
    lines.append(f"  Java files scanned: {report.java_files_scanned}")
    lines.append(f"  Classes found: {report.classes_found}")
    lines.append(f"  Client methods found: {report.clients_found}")
    lines.append(f"  Conforming: {report.conforming}")
    lines.append(f"  Violating: {report.violating}")
    lines.append(f"  Incomplete: {report.incomplete}")
    lines.append(f"  Conformance rate: {report.conformance_rate:.0%}")

    if report.violation_details:
        lines.append("")
        lines.append("  --- VIOLATIONS ---")
        for detail in report.violation_details[:30]:
            lines.append(f"  {detail}")
        if len(report.violation_details) > 30:
            lines.append(f"  ... and {len(report.violation_details) - 30} more")

    lines.append("")
    lines.append("  --- PER-METHOD RESULTS ---")
    for r in report.results[:40]:
        c = r.client
        methods_str = ", ".join(c.methods[:6]) + ("..." if len(c.methods) > 6 else "")
        lines.append(f"  {r.status:<10} {c.class_name}.{c.method_name}() "
                     f"[{methods_str}]")
    if len(report.results) > 40:
        lines.append(f"  ... and {len(report.results) - 40} more")

    lines.append("=" * 80)
    return "\n".join(lines)
