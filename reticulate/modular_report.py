"""Modularity report generation for P104.

Generates structured modularity certificates in text, JSON, and DOT formats.
PDF generation delegates to external tools (weasyprint or wkhtmltopdf).

Session type for this component:
    &{generate: +{json: end, pdf: end, text: end, dot: end}}
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from reticulate.lattice import LatticeResult, check_lattice
from reticulate.modularity import (
    ModularityAnalysis,
    ModularityResult,
    analyze_modularity,
)
from reticulate.parser import pretty
from reticulate.statespace import StateSpace
from reticulate.visualize import dot_source


@dataclass(frozen=True)
class ModularityReport:
    """A complete modularity certificate for a session type protocol."""

    protocol_name: str
    type_string: str
    analysis: ModularityAnalysis
    lattice_result: LatticeResult
    state_space: StateSpace

    def to_text(self) -> str:
        """Render as human-readable text report."""
        return _render_text(self)

    def to_json(self) -> str:
        """Render as JSON."""
        return _render_json(self)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for API responses)."""
        return _to_dict(self)

    def to_dot(self) -> str:
        """Render Hasse diagram as DOT source."""
        return dot_source(
            self.state_space,
            self.lattice_result,
            title=f"Modularity: {self.protocol_name}",
        )


def generate_report(
    type_string: str,
    ss: StateSpace,
    *,
    protocol_name: str = "Protocol",
) -> ModularityReport:
    """Build a complete modularity report from a parsed session type."""
    lattice_result = check_lattice(ss)
    analysis = analyze_modularity(ss)
    return ModularityReport(
        protocol_name=protocol_name,
        type_string=type_string,
        analysis=analysis,
        lattice_result=lattice_result,
        state_space=ss,
    )


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 60


def _render_text(report: ModularityReport) -> str:
    """Render a text modularity certificate."""
    lines: list[str] = []
    m = report.analysis.modularity

    lines.append(_SEPARATOR)
    lines.append(f"  MODULARITY CERTIFICATE: {report.protocol_name}")
    lines.append(_SEPARATOR)
    lines.append("")
    lines.append(f"Protocol:  {report.type_string}")
    lines.append(f"States:    {m.num_states}")
    lines.append(f"Modules:   {m.num_modules}")
    lines.append("")

    # Verdict
    if m.is_modular:
        lines.append("Verdict:   MODULAR (distributive lattice)")
    elif m.is_algebraically_modular:
        lines.append("Verdict:   WEAKLY MODULAR (modular but not distributive)")
    elif m.is_lattice:
        lines.append("Verdict:   NOT MODULAR (lattice, not distributive)")
    else:
        lines.append("Verdict:   INVALID (not a lattice)")

    lines.append(f"Class:     {m.classification}")
    lines.append("")

    # Metrics
    lines.append("Metrics:")
    lines.append(f"  Fiedler value:      {m.fiedler_value:.4f}")
    lines.append(f"  Cheeger constant:   {m.cheeger_constant:.4f}")
    lines.append(f"  Compression ratio:  {m.compression_ratio:.4f}")
    lines.append(f"  Coupling score:     {report.analysis.coupling:.4f}")
    lines.append(f"  Interface width:    {report.analysis.interface_width:.4f}")
    lines.append("")

    # Irreducible modules (Birkhoff)
    if report.analysis.irreducibles:
        lines.append(f"Minimal modules (Birkhoff): {len(report.analysis.irreducibles)}")
        for i, irr in enumerate(report.analysis.irreducibles, 1):
            labels_str = ", ".join(sorted(irr.labels)) if irr.labels else "(none)"
            lines.append(
                f"  {i}. state {irr.representative} "
                f"(height {irr.height}): {labels_str}"
            )
        lines.append("")

    # Module boundaries
    if report.analysis.boundaries:
        lines.append(f"Module boundaries: {len(report.analysis.boundaries)}")
        for i, b in enumerate(report.analysis.boundaries, 1):
            lines.append(
                f"  {i}. {len(b.partition_a)} + {len(b.partition_b)} states "
                f"(cut ratio {b.cut_ratio:.2f}, Fiedler {b.fiedler_value:.4f})"
            )
        lines.append("")

    # Diagnosis (if non-modular)
    diag = report.analysis.diagnosis
    if diag is not None:
        lines.append("Non-modularity diagnosis:")
        lines.append(f"  {diag.explanation}")
        if diag.m3_witness:
            lines.append(f"  M3 witness: {diag.m3_witness}")
        if diag.n5_witness:
            lines.append(f"  N5 witness: {diag.n5_witness}")
        lines.append("")

    # Refactoring suggestions
    if report.analysis.refactorings:
        lines.append("Refactoring suggestions:")
        for i, r in enumerate(report.analysis.refactorings, 1):
            lines.append(f"  {i}. [{r.kind}] {r.description}")
            lines.append(f"     Expected improvement: {r.expected_improvement:.0%}")
        lines.append("")

    lines.append(_SEPARATOR)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------

def _safe_value(v: Any) -> Any:
    """Convert non-serializable values to JSON-safe types."""
    if isinstance(v, frozenset):
        return sorted(v)
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def _to_dict(report: ModularityReport) -> dict[str, Any]:
    """Convert report to a JSON-serializable dictionary."""
    m = report.analysis.modularity
    result: dict[str, Any] = {
        "protocol_name": report.protocol_name,
        "type_string": report.type_string,
        "verdict": (
            "MODULAR" if m.is_modular
            else "WEAKLY_MODULAR" if m.is_algebraically_modular
            else "NOT_MODULAR" if m.is_lattice
            else "INVALID"
        ),
        "classification": m.classification,
        "is_lattice": m.is_lattice,
        "is_distributive": m.is_distributive,
        "is_modular": m.is_modular,
        "metrics": {
            "states": m.num_states,
            "modules": m.num_modules,
            "fiedler_value": m.fiedler_value,
            "cheeger_constant": m.cheeger_constant,
            "compression_ratio": m.compression_ratio,
            "coupling": report.analysis.coupling,
            "interface_width": report.analysis.interface_width,
        },
        "irreducibles": [
            {
                "representative": irr.representative,
                "downset": sorted(irr.downset),
                "labels": sorted(irr.labels),
                "height": irr.height,
            }
            for irr in report.analysis.irreducibles
        ],
        "boundaries": [
            {
                "partition_a": sorted(b.partition_a),
                "partition_b": sorted(b.partition_b),
                "cut_ratio": b.cut_ratio,
                "fiedler_value": b.fiedler_value,
            }
            for b in report.analysis.boundaries
        ],
    }

    diag = report.analysis.diagnosis
    if diag is not None:
        result["diagnosis"] = {
            "has_m3": diag.has_m3,
            "has_n5": diag.has_n5,
            "m3_witness": list(diag.m3_witness) if diag.m3_witness else None,
            "n5_witness": list(diag.n5_witness) if diag.n5_witness else None,
            "entanglement_type": diag.entanglement_type,
            "explanation": diag.explanation,
        }

    if report.analysis.refactorings:
        result["refactorings"] = [
            {
                "kind": r.kind,
                "description": r.description,
                "expected_improvement": r.expected_improvement,
            }
            for r in report.analysis.refactorings
        ]

    return result


def _render_json(report: ModularityReport) -> str:
    """Render as formatted JSON string."""
    return json.dumps(_to_dict(report), indent=2)
