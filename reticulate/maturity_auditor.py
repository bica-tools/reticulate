"""Maturity auditor: evaluates step papers for toy vs real-world depth (Agent).

Analyzes a step's implementation and paper to determine whether the
content is "toy" (trivial examples, small benchmarks, proof-of-concept)
or "real-world" (complex protocols, industry-scale, production-ready).

For toy steps, proposes a concrete upgrade path with:
1. Real-world complex version of the concept
2. Research needed to make it production-grade
3. Execution plan with milestones

Maturity levels:
- **Toy**: Only works on 2-5 state examples, no real protocol validation
- **Prototype**: Works on benchmarks but untested on real systems
- **Research-grade**: Validated on real protocols, has formal properties
- **Production-grade**: Handles edge cases, scales, has error recovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Maturity assessment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaturityAssessment:
    """Assessment of a step's maturity level.

    Attributes:
        step_number: Step identifier.
        step_title: Step title.
        maturity_level: toy | prototype | research | production.
        toy_indicators: List of reasons it's considered toy.
        strengths: What's already good.
        upgrade_proposal: Concrete proposal for real-world version.
        research_needed: What research/investigation is required.
        execution_plan: Step-by-step plan to upgrade.
        estimated_new_step: Proposed new step number and title.
    """
    step_number: str
    step_title: str
    maturity_level: str
    toy_indicators: list[str]
    strengths: list[str]
    upgrade_proposal: str
    research_needed: list[str]
    execution_plan: list[str]
    estimated_new_step: str


# ---------------------------------------------------------------------------
# Toy indicators
# ---------------------------------------------------------------------------

def _check_toy_indicators(
    module_source: str,
    test_source: str,
    paper_text: str,
    num_tests: int,
    word_count: int,
    benchmark_count: int,
) -> list[str]:
    """Identify toy indicators in a step's artifacts."""
    indicators: list[str] = []

    # Small test suite
    if num_tests < 20:
        indicators.append(f"Only {num_tests} tests (< 20 threshold)")

    # No real-world benchmarks
    if benchmark_count < 3:
        indicators.append(f"Only {benchmark_count} benchmark protocols tested")

    # Paper too short
    if word_count < 4000:
        indicators.append(f"Paper only {word_count} words (< 4000)")

    # Only trivial examples in code
    trivial_types = ["&{a: end}", "&{a: &{b: end}}", "end"]
    trivial_count = sum(1 for t in trivial_types if t in test_source)
    total_type_strings = test_source.count("_build(")
    if total_type_strings > 0 and trivial_count / total_type_strings > 0.5:
        indicators.append("Over 50% of test types are trivial (end, single branch)")

    # No real protocol names
    real_protocols = ["SMTP", "OAuth", "HTTP", "JDBC", "Iterator", "MCP",
                      "Kafka", "gRPC", "WebSocket", "TLS", "MQTT"]
    found_real = [p for p in real_protocols if p.lower() in test_source.lower()]
    if len(found_real) < 2:
        indicators.append(f"Only {len(found_real)} real protocol names in tests")

    # No error handling
    if "Error" not in module_source and "error" not in module_source:
        indicators.append("No error handling in module")

    # No scalability consideration
    if "complexity" not in module_source.lower() and "O(" not in module_source:
        indicators.append("No algorithmic complexity discussion")

    # No edge cases
    edge_case_keywords = ["empty", "single", "degenerate", "corner", "edge case"]
    found_edge = [k for k in edge_case_keywords if k in test_source.lower()]
    if len(found_edge) < 2:
        indicators.append("Few edge case tests")

    return indicators


def _assess_maturity(indicators: list[str]) -> str:
    """Determine maturity level from indicators."""
    n = len(indicators)
    if n >= 5:
        return "toy"
    elif n >= 3:
        return "prototype"
    elif n >= 1:
        return "research"
    else:
        return "production"


# ---------------------------------------------------------------------------
# Upgrade proposals
# ---------------------------------------------------------------------------

# Domain-specific upgrade templates
_UPGRADE_TEMPLATES: dict[str, dict[str, str]] = {
    "zeta": {
        "proposal": "Apply zeta matrix analysis to real SMTP/OAuth/TLS protocol state spaces with 50+ states. Compute zeta matrices for the full 79 benchmark suite. Develop incremental zeta update for protocol refinement.",
        "research": "Study zeta matrix sparsity patterns in real protocols. Investigate compressed zeta representations for large state spaces. Benchmark against graph database implementations.",
    },
    "mobius": {
        "proposal": "Use Möbius inversion for real protocol diff computation. Given two protocol versions, compute the Möbius-based difference to identify structural changes. Apply to API versioning (OpenAPI evolution).",
        "research": "Investigate Möbius function stability under protocol refinement. Study relationship between Möbius values and backward compatibility. Connect to semantic versioning.",
    },
    "eigenvalues": {
        "proposal": "Spectral analysis of Kubernetes service mesh protocols (100+ microservices). Use eigenvalues to detect communication bottlenecks and suggest topology improvements.",
        "research": "Study spectral properties of scale-free protocol networks. Investigate random matrix theory bounds for session type spectra. Benchmark eigenvalue computation for 1000+ state spaces.",
    },
    "fiedler": {
        "proposal": "Real-time Fiedler monitoring for distributed systems. As services join/leave, recompute algebraic connectivity to warn about fragile configurations. Integrate with Kubernetes health checks.",
        "research": "Develop incremental Fiedler update algorithms. Study Fiedler value distribution across microservice architectures. Connect to fault injection testing.",
    },
    "transfer": {
        "proposal": "Transfer matrix analysis of the full OAuth 2.0 + PKCE flow including error recovery paths. Count all valid execution paths for exhaustive test generation.",
        "research": "Study transfer matrix rank for real protocols. Investigate connection to test coverage metrics. Benchmark against existing test generation tools.",
    },
    "default": {
        "proposal": "Apply this analysis to the full 79 benchmark suite and at least 3 real industry protocols (OAuth, gRPC, MQTT). Validate results against manual protocol analysis.",
        "research": "Study scalability to 100+ state protocols. Identify edge cases from real implementations. Connect to existing tools in the domain.",
    },
}


def propose_upgrade(step_title: str) -> tuple[str, list[str], list[str]]:
    """Propose an upgrade from toy to real-world.

    Returns (proposal, research_needed, execution_plan).
    """
    # Find matching template
    template = _UPGRADE_TEMPLATES.get("default", _UPGRADE_TEMPLATES["default"])
    for key, tmpl in _UPGRADE_TEMPLATES.items():
        if key in step_title.lower():
            template = tmpl
            break

    proposal = template["proposal"]

    research_needed = [
        template["research"],
        "Survey existing tools/papers in this specific area",
        "Identify 3-5 real-world use cases from industry",
        "Define success metrics (coverage, performance, correctness)",
    ]

    execution_plan = [
        "1. Run current implementation on full 79 benchmark suite",
        "2. Identify failures, edge cases, and performance bottlenecks",
        "3. Extend to handle 3 real industry protocols",
        "4. Add comprehensive error handling and validation",
        "5. Write real-world case study section for paper",
        "6. Benchmark against existing tools (if any)",
        "7. Update paper with real-world results and limitations",
    ]

    return proposal, research_needed, execution_plan


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def audit_step(
    step_number: str,
    step_title: str,
    module_source: str = "",
    test_source: str = "",
    paper_text: str = "",
    num_tests: int = 0,
    word_count: int = 0,
    benchmark_count: int = 0,
) -> MaturityAssessment:
    """Audit a step for maturity level and propose upgrades."""
    indicators = _check_toy_indicators(
        module_source, test_source, paper_text,
        num_tests, word_count, benchmark_count,
    )

    maturity = _assess_maturity(indicators)

    strengths: list[str] = []
    if num_tests >= 30:
        strengths.append(f"Good test coverage ({num_tests} tests)")
    if word_count >= 5000:
        strengths.append(f"Thorough paper ({word_count} words)")
    if benchmark_count >= 5:
        strengths.append(f"Tested on {benchmark_count} benchmarks")
    if "parallel" in test_source.lower():
        strengths.append("Tests parallel composition")
    if "recursive" in test_source.lower() or "rec X" in test_source:
        strengths.append("Tests recursive types")

    proposal, research, plan = propose_upgrade(step_title)

    estimated_new = f"{step_number}+: {step_title} — Real-World Application"

    return MaturityAssessment(
        step_number=step_number,
        step_title=step_title,
        maturity_level=maturity,
        toy_indicators=indicators,
        strengths=strengths,
        upgrade_proposal=proposal,
        research_needed=research,
        execution_plan=plan,
        estimated_new_step=estimated_new,
    )


def audit_step_from_disk(
    step_number: str,
    root: Path | str = ".",
) -> MaturityAssessment:
    """Audit a step by reading its artifacts from disk."""
    root = Path(root)

    # Find module
    module_source = ""
    test_source = ""
    paper_text = ""
    num_tests = 0
    word_count = 0
    benchmark_count = 0
    step_title = step_number

    # Search for step directory
    steps_dir = root / "papers" / "steps"
    for d in steps_dir.iterdir() if steps_dir.is_dir() else []:
        if d.name.startswith(f"step{step_number}"):
            step_title = d.name.replace(f"step{step_number}-", "").replace("-", " ").title()
            main_tex = d / "main.tex"
            if main_tex.is_file():
                paper_text = main_tex.read_text()
                import re
                stripped = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", " ", paper_text)
                stripped = re.sub(r"[{}$\\&%]", " ", stripped)
                word_count = len(stripped.split())
            break

    # Search for module and tests
    mod_dir = root / "reticulate" / "reticulate"
    test_dir = root / "reticulate" / "tests"
    if mod_dir.is_dir():
        for f in mod_dir.glob("*.py"):
            if step_title.lower().replace(" ", "_") in f.stem or f.stem in step_title.lower():
                module_source = f.read_text()
                test_file = test_dir / f"test_{f.stem}.py"
                if test_file.is_file():
                    test_source = test_file.read_text()
                    num_tests = test_source.count("def test_")
                break

    # Count benchmark parametrize decorators
    benchmark_count = test_source.count("@pytest.mark.parametrize")

    return audit_step(
        step_number=step_number,
        step_title=step_title,
        module_source=module_source,
        test_source=test_source,
        paper_text=paper_text,
        num_tests=num_tests,
        word_count=word_count,
        benchmark_count=benchmark_count,
    )
