"""Scan a Python project for protocol conformance violations (Step 80d).

Given a session type (the protocol for an API) and a directory of Python
source files, statically extract method call sequences from ALL functions
that use the target API, then check each for conformance.

This is the full product pipeline:
  Project directory → find all clients → extract call sequences → check conformance

No runtime needed. Pure static analysis.

Usage:
    from reticulate.project_scanner import scan_project
    results = scan_project(
        project_dir="src/",
        target_type="conn",  # variable name pattern
        session_type="rec X . &{execute: +{OK: X, ERR: X}, commit: X, close: end}",
    )
    for r in results:
        print(r)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from reticulate.conformance_checker import check_conformance, ConformanceResult
from reticulate.source_extractor import MethodCallExtractor


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClientTrace:
    """A method call sequence extracted from a function in source code."""
    file_path: str
    function_name: str
    line_number: int
    target_var: str
    methods: list[str]  # L2 trace (method names only, no selections)


@dataclass(frozen=True)
class ScanResult:
    """Result of scanning a single function for conformance."""
    client: ClientTrace
    conformance: ConformanceResult
    status: str  # "PASS", "FAIL", "INCOMPLETE"


@dataclass(frozen=True)
class ProjectReport:
    """Aggregate result of scanning an entire project."""
    project_dir: str
    session_type: str
    target_var: str
    files_scanned: int
    functions_scanned: int
    clients_found: int
    conforming: int
    violating: int
    incomplete: int
    results: list[ScanResult]
    conformance_rate: float
    violation_details: list[str]


# ---------------------------------------------------------------------------
# AST extraction: find all functions using target variable
# ---------------------------------------------------------------------------

class FunctionCallExtractor(ast.NodeVisitor):
    """Extract method calls per function body on a target variable.

    Returns a list of (function_name, line_number, methods_called).
    """

    def __init__(self, target_vars: list[str]) -> None:
        self.target_vars = target_vars
        self.results: list[tuple[str, int, str, list[str]]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._extract_from_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._extract_from_function(node)
        self.generic_visit(node)

    def _extract_from_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for target_var in self.target_vars:
            # Check if this function uses the target variable
            # (as parameter, local assignment, or with-statement)
            if self._function_uses_var(node, target_var):
                extractor = MethodCallExtractor(target_var)
                extractor.visit(node)
                if extractor.calls:
                    self.results.append(
                        (node.name, node.lineno, target_var, extractor.calls)
                    )

    def _function_uses_var(self, node: ast.AST, var_name: str) -> bool:
        """Check if a function references the target variable."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == var_name:
                return True
            # Check function parameters
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in child.args.args:
                    if arg.arg == var_name:
                        return True
        return False


def extract_clients_from_file(file_path: str, target_vars: list[str]) -> list[ClientTrace]:
    """Extract all client traces from a Python source file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    extractor = FunctionCallExtractor(target_vars)
    extractor.visit(tree)

    return [
        ClientTrace(
            file_path=file_path,
            function_name=func_name,
            line_number=line_no,
            target_var=var,
            methods=methods,
        )
        for func_name, line_no, var, methods in extractor.results
    ]


def extract_clients_from_dir(project_dir: str,
                              target_vars: list[str]) -> tuple[list[ClientTrace], int]:
    """Extract all client traces from all Python files in a directory.

    Returns (clients, files_scanned).
    """
    clients: list[ClientTrace] = []
    files_scanned = 0

    for root, dirs, files in os.walk(project_dir):
        # Skip hidden dirs, __pycache__, .git, venv
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))
                   and d not in ('venv', 'env', 'node_modules', '.tox')]
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                files_scanned += 1
                file_clients = extract_clients_from_file(fpath, target_vars)
                clients.extend(file_clients)

    return clients, files_scanned


# ---------------------------------------------------------------------------
# Conformance scanning
# ---------------------------------------------------------------------------

def scan_project(project_dir: str,
                 target_vars: list[str],
                 session_type: str) -> ProjectReport:
    """Scan a Python project for protocol conformance violations.

    Parameters
    ----------
    project_dir : str
        Path to the project directory to scan.
    target_vars : list[str]
        Variable name patterns to track (e.g., ["conn", "db", "cursor"]).
    session_type : str
        The L3 session type string defining the protocol.

    Returns
    -------
    ProjectReport with per-function conformance results.
    """
    clients, files_scanned = extract_clients_from_dir(project_dir, target_vars)

    results: list[ScanResult] = []
    for client in clients:
        # Check conformance (L2 trace — method names only)
        cr = check_conformance(session_type, client.methods)
        status = "PASS" if cr.conforms and cr.is_complete else \
                 "INCOMPLETE" if cr.conforms else "FAIL"
        results.append(ScanResult(client=client, conformance=cr, status=status))

    conforming = sum(1 for r in results if r.status == "PASS")
    violating = sum(1 for r in results if r.status == "FAIL")
    incomplete = sum(1 for r in results if r.status == "INCOMPLETE")
    total = len(results)

    violation_details: list[str] = []
    for r in results:
        if r.status == "FAIL":
            for v in r.conformance.violations:
                detail = (f"{r.client.file_path}:{r.client.line_number} "
                          f"in {r.client.function_name}(): {v}")
                violation_details.append(detail)

    return ProjectReport(
        project_dir=project_dir,
        session_type=session_type,
        target_var=", ".join(target_vars),
        files_scanned=files_scanned,
        functions_scanned=total,
        clients_found=total,
        conforming=conforming,
        violating=violating,
        incomplete=incomplete,
        results=results,
        conformance_rate=conforming / total if total > 0 else 1.0,
        violation_details=violation_details,
    )


def format_project_report(report: ProjectReport) -> str:
    """Format a project scan report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  PROJECT CONFORMANCE SCAN REPORT")
    lines.append("=" * 78)
    lines.append(f"  Directory: {report.project_dir}")
    lines.append(f"  Target variables: {report.target_var}")
    lines.append(f"  Protocol: {report.session_type[:70]}...")
    lines.append(f"  Files scanned: {report.files_scanned}")
    lines.append(f"  Client functions found: {report.clients_found}")
    lines.append(f"  Conforming: {report.conforming}")
    lines.append(f"  Violating: {report.violating}")
    lines.append(f"  Incomplete: {report.incomplete}")
    lines.append(f"  Conformance rate: {report.conformance_rate:.0%}")

    if report.violation_details:
        lines.append("")
        lines.append("  --- VIOLATIONS ---")
        for detail in report.violation_details[:20]:  # limit output
            lines.append(f"  {detail}")
        if len(report.violation_details) > 20:
            lines.append(f"  ... and {len(report.violation_details) - 20} more")

    lines.append("")
    lines.append("  --- PER-FUNCTION RESULTS ---")
    for r in report.results[:30]:
        c = r.client
        lines.append(f"  {r.status:<10} {c.file_path}:{c.line_number} "
                     f"{c.function_name}() [{', '.join(c.methods[:6])}{'...' if len(c.methods) > 6 else ''}]")
    if len(report.results) > 30:
        lines.append(f"  ... and {len(report.results) - 30} more")

    lines.append("=" * 78)
    return "\n".join(lines)
