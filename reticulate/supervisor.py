"""Research supervision process for the session types programme.

After each step, builds a perspective against the full programme and proposes:
- New steps (theoretical extensions, tool features, cross-domain applications)
- New tools (MCP tools, CLI features, library modules)
- New papers (venue-targeted, educational, survey)
- Conferences and workshops to submit to

Usage:
    from reticulate.supervisor import supervise, SupervisionReport
    report = supervise()
    print(report.summary())

CLI:
    python -m reticulate.supervisor
    python -m reticulate.supervisor --json
    python -m reticulate.supervisor --after-step 28
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepStatus:
    """Status of a single step in the programme."""
    number: str          # e.g. "23", "70b"
    title: str
    directory: str
    has_paper: bool
    has_proofs: bool
    has_module: bool
    module_name: str
    test_count: int
    word_count: int
    grade: str           # "A+", "A", "B", "incomplete", "missing"


@dataclass(frozen=True)
class Proposal:
    """A proposed action (step, tool, paper, or venue)."""
    category: str        # "step" | "tool" | "paper" | "venue"
    title: str
    rationale: str
    priority: str        # "high" | "medium" | "low"
    depends_on: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgrammeSnapshot:
    """Current state of the research programme."""
    total_steps: int
    complete_steps: int
    total_papers: int
    total_modules: int
    total_tests: int
    total_words: int
    steps: tuple[StepStatus, ...]
    phases: dict[str, list[str]]   # phase name → step numbers
    gaps: tuple[str, ...]          # steps without papers
    recent_commits: tuple[str, ...]


@dataclass(frozen=True)
class SupervisionReport:
    """Complete supervision report after a step."""
    snapshot: ProgrammeSnapshot
    proposals: tuple[Proposal, ...]
    step_proposals: tuple[Proposal, ...]
    tool_proposals: tuple[Proposal, ...]
    paper_proposals: tuple[Proposal, ...]
    venue_proposals: tuple[Proposal, ...]

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("  RESEARCH SUPERVISION REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Programme snapshot
        s = self.snapshot
        lines.append(f"  Programme: {s.total_steps} steps, {s.complete_steps} complete")
        lines.append(f"  Papers: {s.total_papers}, Modules: {s.total_modules}, Tests: {s.total_tests}")
        lines.append(f"  Total words: {s.total_words:,}")
        if s.gaps:
            lines.append(f"  Gaps: {', '.join(s.gaps[:10])}")
        lines.append("")

        # Proposals by category
        for cat, proposals in [
            ("NEW STEPS", self.step_proposals),
            ("NEW TOOLS", self.tool_proposals),
            ("NEW PAPERS", self.paper_proposals),
            ("VENUES", self.venue_proposals),
        ]:
            if proposals:
                lines.append(f"  --- {cat} ({len(proposals)}) ---")
                for p in proposals:
                    icon = {"high": "!!!", "medium": "!! ", "low": "!  "}[p.priority]
                    lines.append(f"  [{icon}] {p.title}")
                    lines.append(f"        {p.rationale}")
                    if p.depends_on:
                        lines.append(f"        Depends: {', '.join(p.depends_on)}")
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Programme scanner
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """Find the SessionTypesResearch root."""
    # Try common locations
    for candidate in [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).parent.parent.parent,
    ]:
        if (candidate / "papers" / "steps").is_dir():
            return candidate
        if (candidate / "reticulate" / "reticulate").is_dir():
            return candidate.parent
    return Path.cwd()


def _scan_steps(root: Path) -> list[StepStatus]:
    """Scan all step directories."""
    steps_dir = root / "papers" / "steps"
    if not steps_dir.is_dir():
        return []

    results = []
    for d in sorted(steps_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("step"):
            continue

        # Extract step number
        match = re.match(r"step(\d+\w*)-", d.name)
        if not match:
            continue
        num = match.group(1)

        # Check files
        has_paper = (d / "main.tex").is_file()
        has_proofs = (d / "proofs.tex").is_file()

        # Word count
        word_count = 0
        if has_paper:
            try:
                text = (d / "main.tex").read_text()
                word_count = len(text.split())
            except OSError:
                pass

        # Extract title from directory name
        title = d.name.replace(f"step{num}-", "").replace("-", " ").title()

        # Check for corresponding module
        module_name = ""
        has_module = False
        test_count = 0

        # Map step to likely module
        module_map = {
            "1": "statespace", "2": "benchmarks", "3": "lattice", "4": "lattice",
            "5": "lattice", "5b": "product", "6": "enumerate_types", "6b": "termination",
            "7": "subtyping", "8": "duality", "9": "reticular", "10": "endomorphism",
            "11": "global_types", "12": "projection", "13": "recursion", "13a": "recursion",
            "14": "context_free", "16": "event_structures", "21": "petri", "22": "marking_lattice",
            "23": "place_invariants", "24": "coverability", "25": "petri_benchmarks",
            "26": "ccs", "27": "csp", "28": "failures",
            "29": "distributive", "29a": "distributive", "29b": "gratzer",
            "30": "matrix", "70": "mcp_conformance", "70b": "mcp_server",
            "70e": "compress",
        }

        mod = module_map.get(num, "")
        if mod:
            mod_path = root / "reticulate" / "reticulate" / f"{mod}.py"
            has_module = mod_path.is_file()
            module_name = mod

            # Count tests
            test_path = root / "reticulate" / "tests" / f"test_{mod}.py"
            if test_path.is_file():
                try:
                    test_text = test_path.read_text()
                    test_count = test_text.count("def test_")
                except OSError:
                    pass

        # Grade
        if not has_paper:
            grade = "missing"
        elif word_count < 3000:
            grade = "incomplete"
        elif word_count < 5000:
            grade = "B"
        elif has_proofs:
            grade = "A+"
        else:
            grade = "A"

        results.append(StepStatus(
            number=num, title=title, directory=d.name,
            has_paper=has_paper, has_proofs=has_proofs,
            has_module=has_module, module_name=module_name,
            test_count=test_count, word_count=word_count,
            grade=grade,
        ))

    return results


def _scan_modules(root: Path) -> int:
    """Count Python modules."""
    mod_dir = root / "reticulate" / "reticulate"
    if not mod_dir.is_dir():
        return 0
    return sum(1 for f in mod_dir.glob("*.py") if not f.name.startswith("_"))


def _count_tests(root: Path) -> int:
    """Count test functions."""
    test_dir = root / "reticulate" / "tests"
    if not test_dir.is_dir():
        return 0
    count = 0
    for f in test_dir.glob("test_*.py"):
        try:
            count += f.read_text().count("def test_")
        except OSError:
            pass
    return count


def _recent_commits(root: Path, n: int = 5) -> list[str]:
    """Get recent commit messages."""
    try:
        result = subprocess.run(
            ["git", "log", f"--oneline", f"-{n}"],
            capture_output=True, text=True, cwd=root,
        )
        return result.stdout.strip().split("\n") if result.returncode == 0 else []
    except (OSError, FileNotFoundError):
        return []


def _build_snapshot(root: Path) -> ProgrammeSnapshot:
    """Build a snapshot of the programme state."""
    steps = _scan_steps(root)
    total_words = sum(s.word_count for s in steps)
    complete = sum(1 for s in steps if s.grade in ("A+", "A"))
    gaps = [s.number for s in steps if not s.has_paper]
    # Also check TODO for planned but not-yet-started steps
    num_modules = _scan_modules(root)
    num_tests = _count_tests(root)

    phases = {
        "Phase I: Ground Truth (1-15)": [s.number for s in steps if s.number.rstrip("ab") in [str(i) for i in range(1, 16)]],
        "Phase II: Intersections (16-30)": [s.number for s in steps if 16 <= int(re.match(r"\d+", s.number).group()) <= 30],
        "Phase III: Applications (50-79)": [s.number for s in steps if 50 <= int(re.match(r"\d+", s.number).group()) <= 79],
        "Phase IV: Extensions (100+)": [s.number for s in steps if int(re.match(r"\d+", s.number).group()) >= 100],
    }

    return ProgrammeSnapshot(
        total_steps=len(steps),
        complete_steps=complete,
        total_papers=sum(1 for s in steps if s.has_paper),
        total_modules=num_modules,
        total_tests=num_tests,
        total_words=total_words,
        steps=tuple(steps),
        phases=phases,
        gaps=tuple(gaps),
        recent_commits=tuple(_recent_commits(root)),
    )


# ---------------------------------------------------------------------------
# Proposal generators
# ---------------------------------------------------------------------------

_VENUES = [
    # Top conferences
    Proposal("venue", "CONCUR 2026 — Intl. Conf. on Concurrency Theory",
             "Flagship venue for session types, bisimulation, process algebra. Submit lattice universality theorem.",
             "high", details={"deadline": "April 2026", "type": "conference"}),
    Proposal("venue", "ECOOP 2026 — European Conf. on Object-Oriented Programming",
             "Session types on objects is core ECOOP topic. Submit BICA Reborn tool paper.",
             "high", details={"deadline": "varies", "type": "conference"}),
    Proposal("venue", "ICE 2026 — Interaction and Concurrency Experience",
             "Workshop co-located with DisCoTec. Perfect for step results, less competitive.",
             "high", details={"deadline": "April 2026", "type": "workshop"}),
    Proposal("venue", "PLACES 2026 — Programming Language Approaches to Concurrency and Communication-cEntric Software",
             "Workshop at ETAPS. Session type tools and theory.",
             "high", details={"deadline": "January 2027", "type": "workshop"}),
    Proposal("venue", "FORTE 2026 — Formal Techniques for Distributed Objects, Components and Systems",
             "DisCoTec conference. Conformance testing, runtime verification.",
             "medium", details={"type": "conference"}),
    Proposal("venue", "TACAS 2027 — Tools and Algorithms for the Construction and Analysis of Systems",
             "Tool paper venue. Submit reticulate as verified session type toolkit.",
             "medium", details={"type": "conference"}),
    Proposal("venue", "POPL 2027 — Principles of Programming Languages",
             "Top PL venue. Submit if universality theorem gets a clean proof.",
             "low", details={"type": "conference"}),
    Proposal("venue", "Journal of Logical and Algebraic Methods in Programming (JLAMP)",
             "Journal for lattice-theoretic session type results. Full treatment.",
             "medium", details={"type": "journal"}),
    Proposal("venue", "Science of Computer Programming (SCP)",
             "Elsevier journal. Tool papers, empirical studies.",
             "medium", details={"type": "journal"}),
    Proposal("venue", "Logical Methods in Computer Science (LMCS)",
             "Open access journal. Theoretical results.",
             "medium", details={"type": "journal"}),
]


def _generate_step_proposals(snapshot: ProgrammeSnapshot) -> list[Proposal]:
    """Generate new step proposals based on programme state."""
    proposals = []
    existing = {s.number for s in snapshot.steps}

    # Check which intersections are missing
    if "26" in existing and "27" in existing and "28" in existing:
        proposals.append(Proposal(
            "step", "Step 29c: Process algebra equivalences across CCS/CSP/failures",
            "CCS bisimulation, CSP trace refinement, and failure refinement are now implemented. "
            "A unifying step should prove they coincide on session type state spaces.",
            "high", depends_on=("26", "27", "28"),
        ))

    if "23" in existing:
        proposals.append(Proposal(
            "step", "Step 23b: S-invariants and T-invariants for session type nets",
            "Place invariants (P-invariants) are done. S-invariants (state machine decomposition) "
            "and T-invariants (reproducible firing sequences) complete the Petri net theory.",
            "medium", depends_on=("23",),
        ))

    # Spectral theory
    proposals.append(Proposal(
        "step", "Step 31a: Spectral clustering of session type lattices",
        "The matrix module computes spectral radius and Fiedler value. "
        "Use spectral graph theory to cluster protocols by structural similarity.",
        "medium", depends_on=("30",),
    ))

    # Categorical
    proposals.append(Proposal(
        "step", "Step 167: Monoidal structure of session type composition",
        "Parallel composition (||) gives a monoidal product on session types. "
        "The unit is 'end'. Coherence conditions should follow from lattice product.",
        "medium", depends_on=("163", "168"),
    ))

    # Industry applications
    proposals.append(Proposal(
        "step", "Step 71: Stateful API Contracts (OpenAPI extension)",
        "Session types as OpenAPI extensions for REST API lifecycle constraints. "
        "Highest market-value application of the theory.",
        "high",
    ))

    proposals.append(Proposal(
        "step", "Step 72: FHIR Clinical Workflow Verification",
        "Healthcare is a $8.6B market for formal verification. "
        "Model HL7 FHIR workflows as session types, verify with reticulate.",
        "high",
    ))

    # Runtime monitoring
    proposals.append(Proposal(
        "step", "Step 80: Runtime monitor generation from session types",
        "Generate middleware that enforces session type constraints at runtime. "
        "Direct application of CI gate + state tracker infrastructure.",
        "high", depends_on=("70b",),
    ))

    # Async channels
    proposals.append(Proposal(
        "step", "Step 158: Buffered channel session types",
        "Model bounded buffers between communicating parties. "
        "Connects to async channel analysis already implemented.",
        "medium", depends_on=("157b",),
    ))

    # ── CROSS-DISCIPLINARY STEPS ──

    proposals.append(Proposal(
        "step", "Step 81: Smart contract lifecycle as session type",
        "Model Ethereum/Solidity smart contract state machines as session types. "
        "Runtime monitors enforce transaction ordering. Targets FC, IEEE Blockchain.",
        "high", depends_on=("71", "80"),
        details={"venues": ["FC", "IEEE Blockchain", "ESEC/FSE"]},
    ))

    proposals.append(Proposal(
        "step", "Step 82: LLM agent protocol verification",
        "Extend MCP/A2A conformance testing to multi-agent LLM orchestration systems. "
        "Session types catch protocol violations in agent-to-agent communication.",
        "high", depends_on=("70", "70b", "70c"),
        details={"venues": ["NeurIPS workshop", "AAAI", "ICML workshop"]},
    ))

    proposals.append(Proposal(
        "step", "Step 83: Protocol mining from observed traces",
        "Use reconstruct() to mine session types from observed state machines and logs. "
        "Connects process mining (BPM/ICPM) to session type theory.",
        "medium", depends_on=("9", "156"),
        details={"venues": ["BPM", "ICPM"]},
    ))

    proposals.append(Proposal(
        "step", "Step 84: Interactive session type pedagogy",
        "Build web-based interactive exercises for teaching concurrency via session types. "
        "Leverages the existing web app + dialogue types from Step 18.",
        "medium", depends_on=("52", "18"),
        details={"venues": ["ITiCSE", "SIGCSE", "Koli Calling"]},
    ))

    proposals.append(Proposal(
        "step", "Step 85: Biological signaling pathways as session types",
        "Model enzyme cascades, ion channels, and cell cycle regulation as session types. "
        "Lattice structure reveals pathway properties (modularity, redundancy).",
        "low", depends_on=("53",),
        details={"venues": ["CMSB", "BioPPN"]},
    ))

    proposals.append(Proposal(
        "step", "Step 86: IoT protocol verification (MQTT, CoAP)",
        "Model MQTT pub/sub and CoAP request/response as session types. "
        "Generate runtime monitors for IoT device protocol compliance.",
        "medium", depends_on=("80",),
        details={"venues": ["IoTDI", "SenSys", "MIDDLEWARE"]},
    ))

    proposals.append(Proposal(
        "step", "Step 87: Legal procedures as session types",
        "Model court procedures, contract negotiations, regulatory compliance as session types. "
        "Lattice structure reveals procedural fairness and deadlock-freedom.",
        "low", depends_on=("53",),
        details={"venues": ["ICAIL", "JURIX"]},
    ))

    proposals.append(Proposal(
        "step", "Step 88: Musical form as session type",
        "Model sonata form, fugue structure, theme and variations as session types. "
        "Parallel composition captures polyphony. Non-distributivity from counterpoint.",
        "low", depends_on=("53",),
        details={"venues": ["ISMIR", "NIME"]},
    ))

    proposals.append(Proposal(
        "step", "Step 89: Security protocol verification via lattice properties",
        "Verify OAuth, TLS, mTLS authentication protocols via lattice embeddings. "
        "Subtyping ensures safe protocol version evolution.",
        "high", depends_on=("7",),
        details={"venues": ["CCS", "S&P", "USENIX Security"]},
    ))

    proposals.append(Proposal(
        "step", "Step 90: Database transaction protocols as session types",
        "Model JDBC connection lifecycle and transaction protocols (begin/commit/rollback). "
        "CI gate prevents transaction protocol regressions in database drivers.",
        "medium", depends_on=("53",),
        details={"venues": ["VLDB", "SIGMOD", "EDBT"]},
    ))

    return proposals


def _generate_tool_proposals(snapshot: ProgrammeSnapshot) -> list[Proposal]:
    """Generate new tool proposals."""
    proposals = []

    proposals.append(Proposal(
        "tool", "MCP tool: subtype_check — backward compatibility verification",
        "Expose is_subtype() as MCP tool for CI pipeline integration. "
        "Check if new protocol version is backward-compatible with old.",
        "high",
    ))

    proposals.append(Proposal(
        "tool", "MCP tool: dual — generate server type from client type",
        "Expose dual() as MCP tool. Given a client protocol, generate the matching server.",
        "high",
    ))

    proposals.append(Proposal(
        "tool", "MCP tool: trace_validate — check if a trace follows a protocol",
        "Given a session type and a sequence of method calls, check if the trace is valid. "
        "Direct application for log analysis and debugging.",
        "high",
    ))

    proposals.append(Proposal(
        "tool", "MCP tool: protocol_diff — compare two protocol versions",
        "Combine subtyping + morphism to show what changed between versions. "
        "Essential for protocol evolution in production systems.",
        "medium",
    ))

    proposals.append(Proposal(
        "tool", "VS Code extension for session type analysis",
        "Inline session type checking, Hasse diagram preview, coverage highlighting. "
        "Leverages the MCP server infrastructure.",
        "medium",
    ))

    proposals.append(Proposal(
        "tool", "GitHub Action for session type CI gate",
        "Packaged GitHub Action wrapping python -m reticulate.ci_gate. "
        "One-line integration for any repository.",
        "high",
    ))

    return proposals


def _generate_paper_proposals(snapshot: ProgrammeSnapshot) -> list[Proposal]:
    """Generate paper proposals."""
    proposals = []

    proposals.append(Proposal(
        "paper", "Survey: Session Types as Algebraic Reticulates — The First 100 Steps",
        "Comprehensive survey of all results so far. Map the theory landscape. "
        "Target: JLAMP or LMCS.",
        "high",
    ))

    proposals.append(Proposal(
        "paper", "Tool paper: Reticulate — A Session Type Lattice Checker",
        "Focused tool paper for TACAS or CAV tool track. "
        "Architecture, benchmarks, performance, comparison with Mungo/Scribble.",
        "high",
    ))

    proposals.append(Proposal(
        "paper", "Industry paper: Session Types for API Lifecycle Management",
        "Non-academic paper for practitioners. Target: ACM Queue, IEEE Software. "
        "Focus on CI gate, backward compatibility, protocol evolution.",
        "medium",
    ))

    proposals.append(Proposal(
        "paper", "The Distributivity Dichotomy: When Protocols Are Modular",
        "Venue paper combining Steps 70c, 70d, 70e results. "
        "Target: CONCUR or FORTE.",
        "high",
    ))

    proposals.append(Proposal(
        "paper", "Monograph: Session Types as Algebraic Reticulates (book)",
        "Book-length treatment of the full theory. "
        "Target: LNCS or Cambridge Tracts in TCS.",
        "low",
    ))

    return proposals


# ---------------------------------------------------------------------------
# Core: supervise
# ---------------------------------------------------------------------------

def supervise(
    root: Path | str | None = None,
    after_step: str | None = None,
) -> SupervisionReport:
    """Run the research supervision process.

    Scans the programme, evaluates state, and generates proposals.

    Args:
        root: Project root directory. Auto-detected if None.
        after_step: If set, focus proposals on what comes after this step.

    Returns:
        SupervisionReport with snapshot and proposals.
    """
    if root is None:
        root = _find_project_root()
    else:
        root = Path(root)

    snapshot = _build_snapshot(root)

    step_proposals = _generate_step_proposals(snapshot)
    tool_proposals = _generate_tool_proposals(snapshot)
    paper_proposals = _generate_paper_proposals(snapshot)

    all_proposals = step_proposals + tool_proposals + paper_proposals + list(_VENUES)

    return SupervisionReport(
        snapshot=snapshot,
        proposals=tuple(all_proposals),
        step_proposals=tuple(step_proposals),
        tool_proposals=tuple(tool_proposals),
        paper_proposals=tuple(paper_proposals),
        venue_proposals=tuple(_VENUES),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="reticulate.supervisor",
        description="Research supervision for the session types programme",
    )
    parser.add_argument("--after-step", help="Focus on what comes after this step")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--proposals-only", action="store_true", help="Only show proposals")

    args = parser.parse_args(argv)
    report = supervise(after_step=args.after_step)

    if args.json:
        import json
        output = {
            "snapshot": {
                "total_steps": report.snapshot.total_steps,
                "complete_steps": report.snapshot.complete_steps,
                "total_papers": report.snapshot.total_papers,
                "total_modules": report.snapshot.total_modules,
                "total_tests": report.snapshot.total_tests,
                "gaps": list(report.snapshot.gaps),
            },
            "proposals": [
                {"category": p.category, "title": p.title,
                 "priority": p.priority, "rationale": p.rationale}
                for p in report.proposals
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(report.summary())


if __name__ == "__main__":
    main()
