"""Research supervision process for the session types programme.

Scans the FULL research programme — filesystem, TODO, status, and planning
documents — to build a complete picture and generate actionable proposals.

Data sources:
    1. papers/steps/         — filesystem scan for existing step directories
    2. step-papers-todo.md   — the planning document (all planned steps)
    3. step-papers-status.md — auto-generated status (grades, word counts)
    4. reticulate/            — module and test existence
    5. git log               — recent commits

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
    number: str          # e.g. "23", "70b", "30ae"
    title: str
    phase: str           # "I", "II", etc.
    status: str          # "complete", "draft", "planned", "not_started"
    has_paper: bool
    has_proofs: bool
    has_module: bool
    module_name: str
    test_count: int
    word_count: int
    grade: str           # "A+", "A", "B", "incomplete", "missing", "planned"
    source: str          # "filesystem", "todo", "both"
    priority: str        # "high", "medium", "low", ""
    depends_on: tuple[str, ...] = ()
    domain: str = ""     # e.g. "Music", "Law", "Healthcare"
    market: str = ""     # e.g. "$8.6B by 2036"


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
    # Counts
    total_steps_on_disk: int
    total_steps_planned: int
    complete_steps: int
    draft_steps: int
    planned_steps: int
    not_started_steps: int
    total_papers: int
    total_modules: int
    total_tests: int
    total_words: int
    # Detailed
    steps: tuple[StepStatus, ...]
    phases: dict[str, list[StepStatus]]
    gaps: tuple[str, ...]          # planned steps without papers
    quick_wins: tuple[str, ...]    # steps with module but no paper
    recent_commits: tuple[str, ...]
    # Programme health
    backlog_depth: int             # planned steps not yet started
    a_plus_count: int
    a_plus_pct: float


@dataclass(frozen=True)
class SupervisionReport:
    """Complete supervision report."""
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

        s = self.snapshot
        lines.append(f"  Programme: {s.total_steps_planned} steps planned, "
                     f"{s.total_steps_on_disk} on disk, "
                     f"{s.complete_steps} complete")
        lines.append(f"  Papers: {s.total_papers}, Modules: {s.total_modules}, "
                     f"Tests: {s.total_tests}")
        lines.append(f"  Total words: {s.total_words:,}")
        lines.append(f"  A+ papers: {s.a_plus_count} ({s.a_plus_pct:.0f}%)")
        lines.append(f"  Backlog: {s.backlog_depth} steps planned but not started")
        lines.append("")

        # Phase breakdown
        lines.append("  --- PHASE BREAKDOWN ---")
        for phase_name, phase_steps in s.phases.items():
            if not phase_steps:
                continue
            done = sum(1 for st in phase_steps if st.grade in ("A+", "A"))
            draft = sum(1 for st in phase_steps if st.grade in ("B", "incomplete"))
            planned = sum(1 for st in phase_steps if st.status == "planned")
            lines.append(f"  {phase_name}: {len(phase_steps)} steps "
                        f"({done} done, {draft} draft, {planned} planned)")
        lines.append("")

        # Quick wins
        if s.quick_wins:
            lines.append(f"  --- QUICK WINS ({len(s.quick_wins)}) ---")
            lines.append(f"  Steps with module but no paper: "
                        f"{', '.join(s.quick_wins[:15])}")
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


def _step_sort_key(num: str) -> tuple[int, str]:
    """Sort key for step numbers like '30ae', '5b', '155b'."""
    m = re.match(r"(\d+)(.*)", num)
    if m:
        return (int(m.group(1)), m.group(2))
    return (9999, num)


def _parse_todo(root: Path) -> dict[str, dict[str, Any]]:
    """Parse step-papers-todo.md to discover ALL planned steps.

    Returns dict keyed by step number with metadata.
    """
    todo_path = root / "docs" / "planning" / "step-papers-todo.md"
    if not todo_path.is_file():
        return {}

    text = todo_path.read_text()
    steps: dict[str, dict[str, Any]] = {}

    # Current phase tracker
    current_phase = ""
    current_domain = ""

    for line in text.split("\n"):
        # Track phases
        if line.startswith("## Phase"):
            m = re.match(r"## Phase (\S+)", line)
            if m:
                current_phase = m.group(1).rstrip(":")

        # Track section headers for domain
        if line.startswith("### ") or line.startswith("## "):
            if "Event Structures" in line:
                current_domain = "Event Structures"
            elif "Petri Net" in line:
                current_domain = "Petri Nets"
            elif "Process Algebra" in line:
                current_domain = "Process Algebra"
            elif "Algebraic" in line:
                current_domain = "Algebra"
            elif "Spectral" in line:
                current_domain = "Spectral Theory"
            elif "Möbius" in line:
                current_domain = "Combinatorics"
            elif "Gratzer" in line:
                current_domain = "Lattice Theory"
            elif "Cross-Domain" in line:
                current_domain = "Cross-Domain"
            elif "Industry" in line:
                current_domain = "Industry"
            elif "Language" in line:
                current_domain = "Formalization"
            elif "Morphism" in line:
                current_domain = "Morphisms"

        # Parse table rows: | step | title | ... | status |
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 4:
            continue

        # Skip header/separator rows
        step_num = cols[1].strip()
        if not step_num or step_num == "Step" or step_num.startswith("-"):
            continue

        # Extract step number
        m = re.match(r"(\d+\w*)", step_num)
        if not m:
            continue
        num = m.group(1)

        title = cols[2].strip() if len(cols) > 2 else ""

        # Detect status from last meaningful column
        status_text = " ".join(cols[3:]).lower()
        if "complete" in status_text or "a+" in status_text:
            status = "complete"
        elif "draft" in status_text:
            status = "draft"
        elif "planned" in status_text or "plan:" in status_text:
            status = "planned"
        elif "not started" in status_text:
            status = "not_started"
        elif "write paper" in status_text:
            status = "not_started"
        elif "no" in status_text and "yes" not in status_text:
            status = "not_started"
        else:
            status = "unknown"

        # Detect priority from table
        priority = ""
        if "**#1**" in line or "**#2**" in line or "**#3**" in line:
            priority = "high"
        elif "**#4**" in line or "**#5**" in line:
            priority = "medium"

        # Detect market/domain from table
        market = ""
        domain = current_domain
        for col in cols:
            if "$" in col and "B" in col:
                market = col.strip().strip("*")
            # Domain-specific titles
            for kw, dom in [
                ("Music", "Music"), ("Legal", "Law"), ("Healthcare", "Healthcare"),
                ("Blockchain", "Blockchain"), ("IoT", "IoT"), ("Finance", "Finance"),
                ("AI Agent", "AI"), ("API Contract", "Cloud"), ("Clinical", "Healthcare"),
                ("Biology", "Biology"), ("Signaling", "Biology"),
                ("Pedagogy", "Education"), ("Game", "Games"),
                ("Film", "Film"), ("Dance", "Dance"), ("Theatre", "Theatre"),
                ("Recipe", "Cooking"), ("Supply Chain", "Logistics"),
                ("Football", "Sports"), ("Narrative", "Humanities"),
                ("Security", "Security"), ("Quantum", "Physics"),
            ]:
                if kw.lower() in title.lower() or kw.lower() in col.lower():
                    domain = dom

        # Detect module name from table
        module = ""
        for col in cols:
            col_s = col.strip()
            m2 = re.match(r"(\w+)\.py", col_s)
            if m2:
                module = m2.group(1)

        steps[num] = {
            "title": title,
            "status": status,
            "phase": current_phase,
            "domain": domain,
            "module": module,
            "priority": priority,
            "market": market,
        }

    return steps


def _scan_steps_on_disk(root: Path) -> dict[str, dict[str, Any]]:
    """Scan papers/steps/ for existing step directories."""
    steps_dir = root / "papers" / "steps"
    if not steps_dir.is_dir():
        return {}

    results: dict[str, dict[str, Any]] = {}
    for d in sorted(steps_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("step"):
            continue

        match = re.match(r"step(\d+\w*)-", d.name)
        if not match:
            continue
        num = match.group(1)

        has_paper = (d / "main.tex").is_file()
        has_proofs = (d / "proofs.tex").is_file()

        word_count = 0
        if has_paper:
            try:
                text = (d / "main.tex").read_text()
                # Strip LaTeX commands for better word count
                stripped = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", " ", text)
                stripped = re.sub(r"[{}$\\&%]", " ", stripped)
                word_count = len(stripped.split())
            except OSError:
                pass

        title = d.name.replace(f"step{num}-", "").replace("-", " ").title()

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

        results[num] = {
            "title": title,
            "directory": d.name,
            "has_paper": has_paper,
            "has_proofs": has_proofs,
            "word_count": word_count,
            "grade": grade,
        }

    return results


def _scan_modules(root: Path) -> tuple[int, set[str]]:
    """Count Python modules and return their names."""
    mod_dir = root / "reticulate" / "reticulate"
    if not mod_dir.is_dir():
        return 0, set()
    modules = set()
    for f in mod_dir.glob("*.py"):
        if not f.name.startswith("_"):
            modules.add(f.stem)
    return len(modules), modules


def _count_tests_per_module(root: Path) -> dict[str, int]:
    """Count test functions per module."""
    test_dir = root / "reticulate" / "tests"
    if not test_dir.is_dir():
        return {}
    counts: dict[str, int] = {}
    for f in test_dir.glob("test_*.py"):
        mod = f.stem.replace("test_", "")
        try:
            counts[mod] = f.read_text().count("def test_")
        except OSError:
            pass
    return counts


def _recent_commits(root: Path, n: int = 5) -> list[str]:
    """Get recent commit messages."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{n}"],
            capture_output=True, text=True, cwd=root,
        )
        return result.stdout.strip().split("\n") if result.returncode == 0 else []
    except (OSError, FileNotFoundError):
        return []


def _assign_phase(num: str) -> str:
    """Assign a phase to a step number."""
    m = re.match(r"(\d+)", num)
    if not m:
        return "Unknown"
    n = int(m.group(1))
    if 1 <= n <= 15:
        return "I: Ground Truth (1-15)"
    elif 16 <= n <= 30:
        return "I: Intersections (16-30)"
    elif 31 <= n <= 49:
        return "I: Algebraic/Spectral (31-49)"
    elif 50 <= n <= 69:
        return "I: Cross-Domain (50-69)"
    elif 70 <= n <= 79:
        return "I: Industry (70-79)"
    elif 80 <= n <= 109:
        return "I: Applications (80-109)"
    elif 151 <= n <= 170:
        return "II: Morphisms (151-170)"
    elif 200 <= n <= 210:
        return "III: Formalization (200-210)"
    elif 301 <= n <= 350:
        return "IV: Lattice Theory (301-350)"
    elif 351 <= n <= 400:
        return "IV: Bisimulation (351-400)"
    elif 401 <= n <= 499:
        return "IV: Language Hierarchy (401-499)"
    elif 500 <= n <= 599:
        return "V: Philosophy (500-599)"
    elif 600 <= n <= 650:
        return "V: Physics & Sciences (600-650)"
    elif 651 <= n <= 800:
        return "VI: Comparisons (651-800)"
    elif 801 <= n <= 950:
        return "VII: Applications (801-950)"
    elif 900 <= n <= 999:
        return "VII: Synthesis (900-999)"
    elif n >= 363:
        return "IV: Lattice Theory (363+)"
    return f"Other ({n})"


def _build_snapshot(root: Path) -> ProgrammeSnapshot:
    """Build a complete snapshot merging all data sources."""
    # Scan all sources
    todo_steps = _parse_todo(root)
    disk_steps = _scan_steps_on_disk(root)
    num_modules, module_names = _scan_modules(root)
    test_counts = _count_tests_per_module(root)
    commits = _recent_commits(root)

    # Merge: union of all known step numbers
    all_nums = sorted(set(todo_steps.keys()) | set(disk_steps.keys()),
                      key=_step_sort_key)

    steps: list[StepStatus] = []
    for num in all_nums:
        todo = todo_steps.get(num, {})
        disk = disk_steps.get(num, {})

        # Determine source
        if num in todo_steps and num in disk_steps:
            source = "both"
        elif num in disk_steps:
            source = "filesystem"
        else:
            source = "todo"

        # Title: prefer disk (more descriptive), fall back to todo
        title = disk.get("title", "") or todo.get("title", f"Step {num}")

        # Module
        module_name = todo.get("module", "")
        has_module = module_name in module_names if module_name else False
        # Also try common patterns
        if not has_module and not module_name:
            for mod in module_names:
                # Check if step title words match module name
                if mod in title.lower().replace(" ", "_"):
                    module_name = mod
                    has_module = True
                    break

        test_count = test_counts.get(module_name, 0) if module_name else 0

        # Paper status from disk
        has_paper = disk.get("has_paper", False)
        has_proofs = disk.get("has_proofs", False)
        word_count = disk.get("word_count", 0)
        grade = disk.get("grade", "planned" if source == "todo" else "missing")

        # Overall status
        if grade in ("A+", "A"):
            status = "complete"
        elif grade in ("B", "incomplete"):
            status = "draft"
        elif grade == "planned" or todo.get("status") == "planned":
            status = "planned"
        elif todo.get("status") == "not_started":
            status = "not_started"
        elif has_paper:
            status = "draft"
        else:
            status = "not_started"

        # Phase
        phase = todo.get("phase", "") or _assign_phase(num)

        steps.append(StepStatus(
            number=num,
            title=title,
            phase=phase,
            status=status,
            has_paper=has_paper,
            has_proofs=has_proofs,
            has_module=has_module,
            module_name=module_name,
            test_count=test_count,
            word_count=word_count,
            grade=grade,
            source=source,
            priority=todo.get("priority", ""),
            depends_on=(),
            domain=todo.get("domain", ""),
            market=todo.get("market", ""),
        ))

    # Build phase map
    phases: dict[str, list[StepStatus]] = {}
    for s in steps:
        phase = s.phase or "Unclassified"
        phases.setdefault(phase, []).append(s)

    # Identify gaps and quick wins
    gaps = tuple(s.number for s in steps if s.status in ("planned", "not_started"))
    quick_wins = tuple(s.number for s in steps
                       if s.has_module and not s.has_paper and s.grade != "A+")

    total_tests = sum(test_counts.values())
    total_words = sum(s.word_count for s in steps)
    complete = sum(1 for s in steps if s.grade in ("A+", "A"))
    drafts = sum(1 for s in steps if s.grade in ("B", "incomplete"))
    planned = sum(1 for s in steps if s.status == "planned")
    not_started = sum(1 for s in steps if s.status == "not_started")
    a_plus = sum(1 for s in steps if s.grade == "A+")
    total_with_paper = sum(1 for s in steps if s.has_paper)

    return ProgrammeSnapshot(
        total_steps_on_disk=len(disk_steps),
        total_steps_planned=len(steps),
        complete_steps=complete,
        draft_steps=drafts,
        planned_steps=planned,
        not_started_steps=not_started,
        total_papers=total_with_paper,
        total_modules=num_modules,
        total_tests=total_tests,
        total_words=total_words,
        steps=tuple(steps),
        phases=phases,
        gaps=gaps,
        quick_wins=quick_wins,
        recent_commits=tuple(commits),
        backlog_depth=planned + not_started,
        a_plus_count=a_plus,
        a_plus_pct=(a_plus / total_with_paper * 100) if total_with_paper else 0,
    )


# ---------------------------------------------------------------------------
# Proposal generators — DATA-DRIVEN from programme state
# ---------------------------------------------------------------------------

_VENUES = [
    Proposal("venue", "CONCUR 2026 — Intl. Conf. on Concurrency Theory",
             "Flagship venue for session types, bisimulation, process algebra. "
             "Abstract Apr 20, paper Apr 27.",
             "high", details={"deadline": "2026-04-20", "type": "conference"}),
    Proposal("venue", "ECOOP 2026 — European Conf. on Object-Oriented Programming",
             "Session types on objects is core ECOOP topic. Submit BICA Reborn tool paper.",
             "high", details={"type": "conference"}),
    Proposal("venue", "ICE 2026 — Interaction and Concurrency Experience",
             "Workshop at DisCoTec (Urbino, Jun 12). Deadline Apr 2.",
             "high", details={"deadline": "2026-04-02", "type": "workshop"}),
    Proposal("venue", "PLACES 2026 — PL Approaches to Concurrency",
             "Workshop at ETAPS. Session type tools and theory.",
             "high", details={"type": "workshop"}),
    Proposal("venue", "FORTE 2026 — Formal Techniques for Distributed Systems",
             "DisCoTec conference. Conformance testing, runtime verification.",
             "medium", details={"type": "conference"}),
    Proposal("venue", "TACAS 2027 — Tools and Algorithms",
             "Tool paper venue. Submit reticulate toolkit.",
             "medium", details={"type": "conference"}),
    Proposal("venue", "POPL 2027 — Principles of Programming Languages",
             "Top PL venue. Submit if universality theorem gets clean proof.",
             "low", details={"type": "conference"}),
    Proposal("venue", "JLAMP — Journal of Logical and Algebraic Methods",
             "Journal for lattice-theoretic session type results.",
             "medium", details={"type": "journal"}),
    Proposal("venue", "SCP — Science of Computer Programming",
             "Elsevier journal. Tool papers, empirical studies.",
             "medium", details={"type": "journal"}),
    Proposal("venue", "LMCS — Logical Methods in Computer Science",
             "Open access journal. Theoretical results.",
             "medium", details={"type": "journal"}),
]


def _generate_step_proposals(snapshot: ProgrammeSnapshot) -> list[Proposal]:
    """Generate step proposals dynamically from programme gaps."""
    proposals: list[Proposal] = []
    existing_complete = {s.number for s in snapshot.steps if s.grade in ("A+", "A")}
    existing_on_disk = {s.number for s in snapshot.steps if s.has_paper}

    for s in snapshot.steps:
        # Skip already complete steps
        if s.grade in ("A+", "A"):
            continue

        # Steps with paper that need upgrade
        if s.has_paper and s.grade in ("B", "incomplete"):
            proposals.append(Proposal(
                "step",
                f"Step {s.number}: Expand {s.title} to A+ ({s.word_count} words → 5000+)",
                f"Paper exists but grade is {s.grade}. "
                f"Needs expansion to 5000+ words"
                f"{' and companion proofs.tex' if not s.has_proofs else ''}.",
                "high" if s.has_module else "medium",
                depends_on=s.depends_on,
                details={"current_grade": s.grade, "word_count": s.word_count},
            ))
            continue

        # Steps with module but no paper (quick wins)
        if s.has_module and not s.has_paper:
            proposals.append(Proposal(
                "step",
                f"Step {s.number}: Write paper for {s.title} (module exists, {s.test_count} tests)",
                f"Module {s.module_name}.py exists with {s.test_count} tests but no paper. Quick win.",
                "high",
                depends_on=s.depends_on,
            ))
            continue

        # Planned steps not yet started — propose based on priority/domain
        if s.status in ("planned", "not_started"):
            priority = s.priority or "medium"
            if not s.priority:
                # Auto-prioritize
                if s.domain in ("AI", "Cloud", "Healthcare", "Security"):
                    priority = "high"
                elif s.domain in ("Blockchain", "IoT", "Finance"):
                    priority = "medium"
                elif s.market:
                    priority = "high"

            rationale = f"{s.title}."
            if s.domain:
                rationale += f" Domain: {s.domain}."
            if s.market:
                rationale += f" Market: {s.market}."
            if not s.has_module:
                rationale += " Needs implementation + paper."
            else:
                rationale += " Module exists, needs paper."

            proposals.append(Proposal(
                "step",
                f"Step {s.number}: {s.title}",
                rationale,
                priority,
                depends_on=s.depends_on,
                details={"domain": s.domain, "market": s.market},
            ))

    # Sort: high priority first, then by step number
    priority_order = {"high": 0, "medium": 1, "low": 2}
    proposals.sort(key=lambda p: (priority_order.get(p.priority, 1),
                                   _step_sort_key(p.title.split(":")[0].replace("Step ", "").strip())))

    # Cap at 30 to keep report readable
    return proposals[:30]


def _generate_tool_proposals(snapshot: ProgrammeSnapshot) -> list[Proposal]:
    """Generate tool proposals based on programme state."""
    proposals: list[Proposal] = []
    root = _find_project_root()
    _, module_names = _scan_modules(root)

    # Detect which MCP tools are already exposed in mcp_server.py so we
    # don't keep proposing tools that already exist.
    existing_mcp_tools: set[str] = set()
    mcp_server_path = root / "reticulate" / "reticulate" / "mcp_server.py"
    if mcp_server_path.is_file():
        try:
            src = mcp_server_path.read_text()
            # Match: @mcp.tool(...)\ndef <name>(
            existing_mcp_tools = set(re.findall(
                r"@mcp\.tool\([^)]*\)\s*\ndef\s+(\w+)\s*\(", src))
        except OSError:
            pass

    # Propose only those MCP tools that are NOT already exposed
    mcp_tools_wanted = [
        ("subtype_check", "Backward compatibility verification",
         "Expose is_subtype() as MCP tool for CI integration.", "high"),
        ("dual", "Generate server type from client type",
         "Expose dual() as MCP tool. Given a client protocol, generate the matching server.", "high"),
        ("trace_validate", "Check if a trace follows a protocol",
         "Given a session type and a method call sequence, check validity.", "high"),
        ("protocol_diff", "Compare two protocol versions",
         "Combine subtyping + morphism to show what changed between versions.", "medium"),
    ]
    for name, title, rationale, priority in mcp_tools_wanted:
        if name in existing_mcp_tools:
            continue
        proposals.append(Proposal("tool", f"MCP tool: {name} — {title}",
                                  rationale, priority))

    # Tooling proposals
    proposals.append(Proposal(
        "tool", "VS Code extension for session type analysis",
        "Inline checking, Hasse diagram preview, coverage highlighting. "
        "Leverages MCP server infrastructure.",
        "medium",
    ))
    proposals.append(Proposal(
        "tool", "GitHub Action for session type CI gate",
        "Packaged GitHub Action wrapping reticulate.ci_gate. "
        "One-line integration for any repository.",
        "high",
    ))

    return proposals


def _generate_paper_proposals(snapshot: ProgrammeSnapshot) -> list[Proposal]:
    """Generate paper proposals based on programme state."""
    proposals: list[Proposal] = []

    # Scale-dependent proposals
    if snapshot.total_steps_on_disk >= 80:
        proposals.append(Proposal(
            "paper", "Survey: Session Types as Algebraic Reticulates — The First 100 Steps",
            f"Comprehensive survey of {snapshot.complete_steps} completed steps. "
            f"Map the theory landscape. Target: JLAMP or LMCS.",
            "high",
        ))

    proposals.append(Proposal(
        "paper", "Tool paper: Reticulate — A Session Type Lattice Checker",
        f"Tool paper for TACAS/CAV. {snapshot.total_modules} modules, "
        f"{snapshot.total_tests} tests, {snapshot.total_words:,} words of documentation.",
        "high",
    ))

    proposals.append(Proposal(
        "paper", "Industry paper: Session Types for API Lifecycle Management",
        "Non-academic paper for practitioners. Target: ACM Queue, IEEE Software.",
        "medium",
    ))

    proposals.append(Proposal(
        "paper", "The Distributivity Dichotomy: When Protocols Are Modular",
        "Venue paper on distributivity results. Target: CONCUR or FORTE.",
        "high",
    ))

    if snapshot.total_words >= 500_000:
        proposals.append(Proposal(
            "paper", "Monograph: Session Types as Algebraic Reticulates",
            f"Book-length treatment ({snapshot.total_words:,} words across programme). "
            "Target: LNCS or Cambridge Tracts.",
            "low",
        ))

    # Per-phase summary papers
    for phase_name, phase_steps in snapshot.phases.items():
        done = [s for s in phase_steps if s.grade in ("A+", "A")]
        if len(done) >= 5 and len(done) == len(phase_steps):
            proposals.append(Proposal(
                "paper", f"Phase summary: {phase_name}",
                f"All {len(done)} steps complete. Write a consolidated phase paper.",
                "medium",
            ))

    return proposals


# ---------------------------------------------------------------------------
# Core: supervise
# ---------------------------------------------------------------------------

def supervise(
    root: Path | str | None = None,
    after_step: str | None = None,
) -> SupervisionReport:
    """Run the full research supervision process.

    Scans the complete programme (filesystem + TODO + status),
    evaluates progress, and generates prioritized proposals.

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
    import json

    parser = argparse.ArgumentParser(
        prog="reticulate.supervisor",
        description="Research supervision for the session types programme",
    )
    parser.add_argument("--after-step", help="Focus on what comes after this step")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--phases", action="store_true", help="Show phase breakdown only")

    args = parser.parse_args(argv)
    report = supervise(after_step=args.after_step)

    if args.json:
        output = {
            "snapshot": {
                "total_steps_on_disk": report.snapshot.total_steps_on_disk,
                "total_steps_planned": report.snapshot.total_steps_planned,
                "complete_steps": report.snapshot.complete_steps,
                "draft_steps": report.snapshot.draft_steps,
                "planned_steps": report.snapshot.planned_steps,
                "not_started_steps": report.snapshot.not_started_steps,
                "total_papers": report.snapshot.total_papers,
                "total_modules": report.snapshot.total_modules,
                "total_tests": report.snapshot.total_tests,
                "total_words": report.snapshot.total_words,
                "a_plus_count": report.snapshot.a_plus_count,
                "a_plus_pct": round(report.snapshot.a_plus_pct, 1),
                "backlog_depth": report.snapshot.backlog_depth,
                "gaps": list(report.snapshot.gaps[:20]),
                "quick_wins": list(report.snapshot.quick_wins),
            },
            "phases": {
                name: {
                    "total": len(steps),
                    "complete": sum(1 for s in steps if s.grade in ("A+", "A")),
                    "draft": sum(1 for s in steps if s.grade in ("B", "incomplete")),
                    "planned": sum(1 for s in steps if s.status == "planned"),
                }
                for name, steps in report.snapshot.phases.items()
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
