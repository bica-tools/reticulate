"""Step evaluator: grades deliverables and identifies gaps/weaknesses.

An Evaluator agent reviews a step's deliverables (paper, proofs, code, tests)
and decides:
  - A+ : accepted, no gaps or weaknesses
  - A  : near-complete, minor issues (e.g. missing proofs, below word count)
  - B  : significant gaps (missing sections, low coverage, no benchmarks)
  - C  : major rework needed (placeholder code, missing paper, failing tests)
  - F  : not started or fundamentally broken

A step is ACCEPTED only at A+. Anything below triggers a fix/gap/weakness
improvement phase with specific actionable items.

Usage:
    from reticulate.evaluator import evaluate_step, EvaluationResult
    result = evaluate_step("23")
    print(result.summary())
    if not result.accepted:
        for fix in result.fixes:
            print(f"  FIX: {fix}")

CLI:
    python -m reticulate.evaluator 23
    python -m reticulate.evaluator --all
    python -m reticulate.evaluator --all --json
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Criteria weights
# ---------------------------------------------------------------------------

# Each criterion has: name, max_points, evaluator function
# Total: 100 points. A+ = 90+, A = 80+, B = 65+, C = 50+, F = <50
#
# Point allocation (rebalanced March 2026):
#   Structural (40): paper exists (10), word count (10), proofs (10), structure (5), bibliography (5)
#   Content quality (35): formal density (10), worked examples (10), benchmarks (8), cross-refs (7)
#   Writing quality (20): visual aids (6), narrative flow (7), active voice (7)
#   Implementation (25): module (10), tests exist (8), tests pass (7)
#   Total raw: 120 points → normalized to 0-100 scale
#
# Paper-only steps: module+tests criteria auto-pass (25 points free).
# Writing criteria reward: diagrams, structured paragraphs, roadmap in intro,
# motivation language, internal cross-references, and active scholarly voice.

_GRADE_THRESHOLDS = {"A+": 90, "A": 80, "B": 65, "C": 50}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Criterion:
    """Result of evaluating one criterion."""
    name: str
    max_points: int
    points: int
    passed: bool
    message: str
    severity: str  # "critical" | "major" | "minor" | "info"


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation of a step."""
    step_number: str
    step_title: str
    grade: str                      # "A+", "A", "B", "C", "F"
    score: int                      # 0-100
    accepted: bool                  # True only if grade == "A+"
    criteria: tuple[Criterion, ...]
    fixes: tuple[str, ...]          # actionable fix items
    strengths: tuple[str, ...]      # what's good
    weaknesses: tuple[str, ...]     # what needs work

    def summary(self) -> str:
        lines: list[str] = []
        lines.append("=" * 65)
        lines.append(f"  STEP {self.step_number} EVALUATION: {self.step_title}")
        lines.append("=" * 65)
        lines.append("")
        lines.append(f"  Grade: {self.grade}  ({self.score}/100)")
        lines.append(f"  Accepted: {'YES' if self.accepted else 'NO — needs improvement'}")
        lines.append("")

        # Criteria
        lines.append("  Criteria:")
        for c in self.criteria:
            icon = "PASS" if c.passed else "FAIL"
            lines.append(f"    [{icon}] {c.name}: {c.points}/{c.max_points} — {c.message}")
        lines.append("")

        if self.strengths:
            lines.append("  Strengths:")
            for s in self.strengths:
                lines.append(f"    + {s}")
            lines.append("")

        if self.weaknesses:
            lines.append("  Weaknesses:")
            for w in self.weaknesses:
                lines.append(f"    - {w}")
            lines.append("")

        if self.fixes:
            lines.append("  Required fixes for A+:")
            for i, f in enumerate(self.fixes, 1):
                lines.append(f"    {i}. {f}")
            lines.append("")

        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    for candidate in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
        if (candidate / "papers" / "steps").is_dir():
            return candidate
        if (candidate / "reticulate" / "reticulate").is_dir():
            return candidate.parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# Step metadata
# ---------------------------------------------------------------------------

_MODULE_MAP: dict[str, str] = {
    "1": "statespace", "3": "lattice", "4": "lattice",
    "5": "lattice", "5b": "product", "6": "enumerate_types", "6b": "termination",
    "7": "subtyping", "8": "duality", "9": "reticular", "10": "endomorphism",
    "11": "global_types", "12": "projection", "13": "recursion", "13a": "recursion",
    "14": "context_free", "21": "petri", "22": "marking_lattice",
    "23": "place_invariants", "24": "coverability", "25": "petri_benchmarks",
    "26": "ccs", "27": "csp", "28": "failures",
    "29": "lattice", "29a": "lattice", "29b": "lattice",
    "30": "matrix", "30i": "irreducibles", "30j": "birkhoff",
    "70": "mcp_conformance", "70b": "mcp_server",
    "70e": "compress",
    "57a": "audio_routing",
    # Paper-only steps (no dedicated module): 2, 29b, 50-53, 70c, 70d, 104, 155b-168, 200*, 900*
}


def _find_step_dir(root: Path, step_num: str) -> Path | None:
    steps_dir = root / "papers" / "steps"
    for d in steps_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"step{step_num}-"):
            return d
    return None


# ---------------------------------------------------------------------------
# Individual criterion evaluators
# ---------------------------------------------------------------------------

# ---- STRUCTURAL criteria (40 points) ----

def _eval_paper_exists(step_dir: Path | None) -> Criterion:
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Paper exists", 10, 0, False, "main.tex not found", "critical")
    return Criterion("Paper exists", 10, 10, True, "main.tex present", "info")


def _eval_word_count(step_dir: Path | None) -> Criterion:
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Word count ≥ 5000", 10, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()
    words = len(text.split())
    if words >= 5000:
        return Criterion("Word count ≥ 5000", 10, 10, True, f"{words} words", "info")
    elif words >= 4000:
        return Criterion("Word count ≥ 5000", 10, 7, False, f"{words} words (need {5000 - words} more)", "minor")
    elif words >= 3000:
        return Criterion("Word count ≥ 5000", 10, 4, False, f"{words} words (need {5000 - words} more)", "major")
    else:
        return Criterion("Word count ≥ 5000", 10, 0, False, f"{words} words (far below target)", "critical")


def _eval_proofs(step_dir: Path | None) -> Criterion:
    if step_dir is None:
        return Criterion("Companion proofs.tex", 10, 0, False, "no step directory", "critical")
    if not (step_dir / "proofs.tex").is_file():
        return Criterion("Companion proofs.tex", 10, 0, False, "proofs.tex missing", "major")
    text = (step_dir / "proofs.tex").read_text()
    words = len(text.split())
    theorems = text.count("\\begin{theorem}") + text.count("\\begin{proposition}") + text.count("\\begin{lemma}")
    if words >= 1000 and theorems >= 2:
        return Criterion("Companion proofs.tex", 10, 10, True, f"{words} words, {theorems} theorems", "info")
    elif words >= 500:
        return Criterion("Companion proofs.tex", 10, 7, False, f"{words} words, {theorems} theorems (thin)", "minor")
    else:
        return Criterion("Companion proofs.tex", 10, 3, False, f"{words} words (stub)", "major")


def _eval_paper_structure(step_dir: Path | None) -> Criterion:
    """Check paper has proper sections: intro, body, conclusion, bibliography."""
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Paper structure", 5, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    checks = {
        "abstract": "\\begin{abstract}" in text,
        "introduction": "\\section{Introduction}" in text or "section{Introduction}" in text.replace("\\", ""),
        "conclusion": "Conclusion" in text and "section{" in text,
        "bibliography": "bibliography" in text.lower() or "thebibliography" in text,
    }
    passed_count = sum(checks.values())
    missing = [k for k, v in checks.items() if not v]

    if passed_count == 4:
        return Criterion("Paper structure", 5, 5, True, "abstract, intro, conclusion, bibliography", "info")
    else:
        return Criterion("Paper structure", 5, passed_count, False, f"missing: {', '.join(missing)}", "minor")


def _eval_bibliography(step_dir: Path | None) -> Criterion:
    """Check bibliography has sufficient references (≥8 for quality research)."""
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Bibliography ≥ 8 refs", 5, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()
    refs = text.count("\\bibitem{")
    if refs == 0:
        # Try \bibliography{} style
        refs = len(re.findall(r"\\cite\{([^}]+)\}", text))
    if refs >= 8:
        return Criterion("Bibliography ≥ 8 refs", 5, 5, True, f"{refs} references", "info")
    elif refs >= 5:
        return Criterion("Bibliography ≥ 8 refs", 5, 3, False, f"{refs} references (need 8+)", "minor")
    else:
        return Criterion("Bibliography ≥ 8 refs", 5, 1, False, f"{refs} references (too few)", "major")


# ---- CONTENT QUALITY criteria (35 points) ----

# Benchmark protocol names to search for in papers
_BENCHMARK_NAMES = [
    "SMTP", "OAuth", "HTTP", "JDBC", "Iterator", "MQTT", "TLS",
    "Two-Buyer", "two-buyer", "MCP", "A2A", "ERC-20", "ERC-721",
    "CoAP", "DNS", "WebSocket", "ATM", "Pub/Sub", "pub-sub",
    "FHIR", "Kerberos", "SAML", "Zigbee",
]


def _eval_formal_density(step_dir: Path | None) -> Criterion:
    """Count formal environments: definition, theorem, lemma, proposition, example.

    A quality research paper should have ≥4 formal environments.
    These structure the contribution and make claims precise.
    """
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Formal density (≥4 envs)", 10, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    envs = {
        "definition": len(re.findall(r"\\begin\{definition\}", text)),
        "theorem": len(re.findall(r"\\begin\{theorem\}", text)),
        "lemma": len(re.findall(r"\\begin\{lemma\}", text)),
        "proposition": len(re.findall(r"\\begin\{proposition\}", text)),
        "corollary": len(re.findall(r"\\begin\{corollary\}", text)),
        "example": len(re.findall(r"\\begin\{example\}", text)),
        "remark": len(re.findall(r"\\begin\{remark\}", text)),
    }
    total = sum(envs.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in envs.items() if v > 0)

    if total >= 6:
        return Criterion("Formal density (≥4 envs)", 10, 10, True, f"{total} environments ({breakdown})", "info")
    elif total >= 4:
        return Criterion("Formal density (≥4 envs)", 10, 8, True, f"{total} environments ({breakdown})", "info")
    elif total >= 2:
        return Criterion("Formal density (≥4 envs)", 10, 5, False, f"{total} environments — need more formal content ({breakdown})", "minor")
    else:
        return Criterion("Formal density (≥4 envs)", 10, 2, False, f"{total} environments — paper lacks formal structure", "major")


def _eval_worked_examples(step_dir: Path | None) -> Criterion:
    """Check for worked examples: \\begin{example} or 'Example' subsections.

    Quality papers demonstrate concepts with concrete worked-through examples,
    not just abstract definitions.
    """
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Worked examples (≥2)", 10, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    # Count formal example environments
    examples = len(re.findall(r"\\begin\{example\}", text))
    # Also count "Example" in section/subsection headings or \paragraph
    examples += len(re.findall(r"\\(?:sub)*section\{[^}]*[Ee]xample", text))
    examples += len(re.findall(r"\\paragraph\{[^}]*[Ee]xample", text))
    # Count "Worked example" paragraphs
    examples += len(re.findall(r"\\paragraph\{Worked", text))

    if examples >= 3:
        return Criterion("Worked examples (≥2)", 10, 10, True, f"{examples} worked examples", "info")
    elif examples >= 2:
        return Criterion("Worked examples (≥2)", 10, 8, True, f"{examples} worked examples", "info")
    elif examples >= 1:
        return Criterion("Worked examples (≥2)", 10, 5, False, f"{examples} example — need at least 2", "minor")
    else:
        return Criterion("Worked examples (≥2)", 10, 0, False, "no worked examples found", "major")


def _eval_benchmark_references(step_dir: Path | None) -> Criterion:
    """Check that the paper references real benchmark protocols.

    Grounding in real protocols (SMTP, OAuth, Iterator, etc.) validates
    that the theory applies to practical systems, not just toy examples.
    """
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Benchmark grounding (≥2)", 8, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    found = [name for name in _BENCHMARK_NAMES if name in text]
    unique = len(set(n.lower() for n in found))

    if unique >= 4:
        return Criterion("Benchmark grounding (≥2)", 8, 8, True, f"{unique} protocols referenced ({', '.join(found[:5])})", "info")
    elif unique >= 2:
        return Criterion("Benchmark grounding (≥2)", 8, 6, True, f"{unique} protocols referenced ({', '.join(found)})", "info")
    elif unique >= 1:
        return Criterion("Benchmark grounding (≥2)", 8, 3, False, f"{unique} protocol referenced ({', '.join(found)}) — need 2+", "minor")
    else:
        return Criterion("Benchmark grounding (≥2)", 8, 0, False, "no benchmark protocols referenced", "minor")


def _eval_cross_references(step_dir: Path | None) -> Criterion:
    """Check that the paper connects to other steps in the programme.

    Each step builds on prior work. Cross-references create coherence
    and show the paper's place in the larger theory.
    """
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Cross-references to steps", 7, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    # Match "Step X", "Step~X", "Steps X-Y", "(Step X)"
    step_refs = re.findall(r"[Ss]tep[~\s]+(\d+\w?)", text)
    unique_steps = set(step_refs)

    # Also check for "reticulate" tool references
    tool_refs = "reticulate" in text.lower() or "\\reticulate" in text

    total = len(unique_steps) + (1 if tool_refs else 0)

    if total >= 4:
        return Criterion("Cross-references to steps", 7, 7, True, f"{len(unique_steps)} steps + tool refs", "info")
    elif total >= 2:
        return Criterion("Cross-references to steps", 7, 5, True, f"{len(unique_steps)} step refs", "info")
    elif total >= 1:
        return Criterion("Cross-references to steps", 7, 3, False, f"{total} refs — connect to more prior work", "minor")
    else:
        return Criterion("Cross-references to steps", 7, 0, False, "no cross-references to other steps", "minor")


# ---- WRITING & NARRATIVE QUALITY criteria (20 points) ----


def _eval_visual_aids(step_dir: Path | None) -> Criterion:
    """Check for figures, diagrams, and tables.

    Visual aids are essential for communicating lattice structures,
    state spaces, and empirical results. A paper without visuals
    forces the reader to build mental pictures from text alone.
    """
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Visual aids (figures/tables)", 6, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    figures = len(re.findall(r"\\begin\{figure\}", text))
    tikz = len(re.findall(r"\\begin\{tikzpicture\}", text))
    tables = len(re.findall(r"\\begin\{(?:table|tabular)\}", text))
    listings = len(re.findall(r"\\begin\{lstlisting\}", text))

    total = figures + tikz + tables + listings
    parts = []
    if figures: parts.append(f"{figures} fig")
    if tikz: parts.append(f"{tikz} tikz")
    if tables: parts.append(f"{tables} tbl")
    if listings: parts.append(f"{listings} lst")
    desc = ", ".join(parts) if parts else "none"

    if total >= 3:
        return Criterion("Visual aids (figures/tables)", 6, 6, True, f"{total} visual elements ({desc})", "info")
    elif total >= 1:
        return Criterion("Visual aids (figures/tables)", 6, 4, False, f"{total} visual element(s) ({desc}) — add diagrams or tables", "minor")
    else:
        return Criterion("Visual aids (figures/tables)", 6, 0, False, "no figures, diagrams, or tables", "major")


def _eval_narrative_flow(step_dir: Path | None) -> Criterion:
    """Check for internal navigation and narrative connective tissue.

    Good papers guide the reader: forward references (Cref/Section X),
    structured paragraphs, and transition phrases between sections.
    A paper that jumps from definition to definition without
    connecting prose reads like a reference manual, not a narrative.
    """
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Narrative flow", 7, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    score = 0
    details = []

    # 1. Internal cross-references (Cref, cref, ref to sections/theorems)
    internal_refs = len(re.findall(r"\\[Cc]ref\{", text)) + len(re.findall(r"\\ref\{", text))
    if internal_refs >= 8:
        score += 2
        details.append(f"{internal_refs} internal refs")
    elif internal_refs >= 3:
        score += 1
        details.append(f"{internal_refs} refs (add more)")

    # 2. Structured paragraphs (\paragraph{} for sub-topics)
    paragraphs = len(re.findall(r"\\paragraph\{", text))
    if paragraphs >= 4:
        score += 2
        details.append(f"{paragraphs} structured paragraphs")
    elif paragraphs >= 2:
        score += 1
        details.append(f"{paragraphs} paragraphs")

    # 3. Section roadmap in intro ("Section X ... Section Y ...")
    intro_match = re.search(
        r"\\section\{Introduction\}(.*?)\\section\{",
        text, re.DOTALL
    )
    has_roadmap = False
    if intro_match:
        intro_text = intro_match.group(1)
        section_refs = len(re.findall(r"[Ss]ection|\\[Cc]ref\{sec:", intro_text))
        if section_refs >= 3:
            has_roadmap = True
            score += 2
            details.append("intro roadmap")
        elif section_refs >= 1:
            score += 1
            details.append("partial roadmap")
    else:
        # No intro found, but check if any roadmap exists
        if re.search(r"is organi[sz]ed as follows", text):
            score += 1
            details.append("roadmap phrase found")

    # 4. Motivation language in introduction
    motivation_words = ["why", "problem", "challenge", "question", "motivation",
                       "gap", "need", "crucial", "essential", "important"]
    if intro_match:
        intro_lower = intro_match.group(1).lower()
        motivation_count = sum(1 for w in motivation_words if w in intro_lower)
        if motivation_count >= 3:
            score += 1
            details.append("strong motivation")

    desc = ", ".join(details) if details else "weak narrative structure"
    passed = score >= 5

    return Criterion("Narrative flow", 7, score, passed, desc, "minor" if not passed else "info")


def _eval_active_voice(step_dir: Path | None) -> Criterion:
    """Check for active scholarly voice vs. passive/impersonal writing.

    Active voice ("We prove that...", "We show...", "This demonstrates...")
    engages the reader and clarifies agency. Excessive passive voice
    ("It is shown that...", "It can be seen...") is a sign of weak writing.
    """
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Active scholarly voice", 7, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()

    # Active patterns (good)
    active_patterns = [
        r"\b[Ww]e\s+(?:show|prove|demonstrate|define|introduce|present|develop|establish|observe|verify|analyze|propose|construct|compute|check)\b",
        r"\b[Tt]his\s+(?:shows|proves|demonstrates|means|implies|establishes|ensures|guarantees)\b",
        r"\b[Oo]ur\s+(?:approach|contribution|result|analysis|tool|method|framework)\b",
    ]
    active_count = sum(len(re.findall(p, text)) for p in active_patterns)

    # Passive patterns (not necessarily bad, but should be minority)
    passive_patterns = [
        r"\b[Ii]t\s+(?:is|was|can be|has been)\s+(?:shown|proved|seen|noted|observed|known)\b",
        r"\bis\s+(?:defined|given|obtained|computed|verified)\s+(?:by|as|in)\b",
    ]
    passive_count = sum(len(re.findall(p, text)) for p in passive_patterns)

    total = active_count + passive_count
    if total == 0:
        return Criterion("Active scholarly voice", 7, 3, False, "no active/passive indicators found", "minor")

    active_ratio = active_count / total if total > 0 else 0

    if active_count >= 8 and active_ratio >= 0.6:
        return Criterion("Active scholarly voice", 7, 7, True,
                        f"{active_count} active, {passive_count} passive ({active_ratio:.0%} active)", "info")
    elif active_count >= 4:
        return Criterion("Active scholarly voice", 7, 5, True,
                        f"{active_count} active, {passive_count} passive — good but could be stronger", "info")
    elif active_count >= 2:
        return Criterion("Active scholarly voice", 7, 3, False,
                        f"{active_count} active, {passive_count} passive — add more 'We show/prove/demonstrate'", "minor")
    else:
        return Criterion("Active scholarly voice", 7, 1, False,
                        f"{active_count} active, {passive_count} passive — paper reads as impersonal", "minor")


# ---- IMPLEMENTATION criteria (25 points) ----

def _eval_module(root: Path, step_num: str) -> Criterion:
    mod_name = _MODULE_MAP.get(step_num, "")
    if not mod_name:
        return Criterion("Implementation module", 10, 10, True, "paper-only step (no module expected)", "info")
    mod_path = root / "reticulate" / "reticulate" / f"{mod_name}.py"
    if not mod_path.is_file():
        return Criterion("Implementation module", 10, 0, False, f"{mod_name}.py not found", "critical")
    lines = len(mod_path.read_text().split("\n"))
    funcs = mod_path.read_text().count("\ndef ")
    if lines >= 100 and funcs >= 3:
        return Criterion("Implementation module", 10, 10, True, f"{mod_name}.py: {lines} lines, {funcs} functions", "info")
    elif lines >= 50:
        return Criterion("Implementation module", 10, 7, False, f"{mod_name}.py: {lines} lines (thin)", "minor")
    else:
        return Criterion("Implementation module", 10, 3, False, f"{mod_name}.py: {lines} lines (stub)", "major")


def _eval_tests(root: Path, step_num: str) -> Criterion:
    mod_name = _MODULE_MAP.get(step_num, "")
    if not mod_name:
        return Criterion("Test suite", 8, 8, True, "paper-only step (no tests expected)", "info")
    test_path = root / "reticulate" / "tests" / f"test_{mod_name}.py"
    if not test_path.is_file():
        return Criterion("Test suite", 8, 0, False, f"test_{mod_name}.py not found", "critical")
    text = test_path.read_text()
    test_count = text.count("def test_")
    if test_count >= 20:
        return Criterion("Test suite", 8, 8, True, f"{test_count} tests", "info")
    elif test_count >= 10:
        return Criterion("Test suite", 8, 5, False, f"{test_count} tests (need 20+)", "minor")
    elif test_count >= 5:
        return Criterion("Test suite", 8, 3, False, f"{test_count} tests (need 20+)", "major")
    else:
        return Criterion("Test suite", 8, 0, False, f"{test_count} tests (far too few)", "critical")


def _eval_tests_pass(root: Path, step_num: str) -> Criterion:
    mod_name = _MODULE_MAP.get(step_num, "")
    if not mod_name:
        return Criterion("Tests pass", 7, 7, True, "paper-only step (no tests expected)", "info")
    test_path = root / "reticulate" / "tests" / f"test_{mod_name}.py"
    if not test_path.is_file():
        return Criterion("Tests pass", 7, 0, False, "no test file", "critical")
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", str(test_path), "-q", "--tb=no", "--no-header"],
            capture_output=True, text=True, timeout=60,
            cwd=root / "reticulate",
        )
        if result.returncode == 0:
            match = re.search(r"(\d+) passed", result.stdout)
            passed = int(match.group(1)) if match else 0
            return Criterion("Tests pass", 7, 7, True, f"{passed} passed", "info")
        else:
            match_fail = re.search(r"(\d+) failed", result.stdout)
            failed = int(match_fail.group(1)) if match_fail else 0
            return Criterion("Tests pass", 7, 0, False, f"{failed} failed", "critical")
    except (subprocess.TimeoutExpired, OSError):
        return Criterion("Tests pass", 7, 0, False, "test run failed/timeout", "critical")


# ---------------------------------------------------------------------------
# Core: evaluate_step
# ---------------------------------------------------------------------------

def evaluate_step(
    step_num: str,
    root: Path | str | None = None,
    run_tests: bool = True,
) -> EvaluationResult:
    """Evaluate a step's deliverables.

    Args:
        step_num: Step number (e.g. "23", "70b").
        root: Project root. Auto-detected if None.
        run_tests: Whether to actually run tests (slower but accurate).

    Returns:
        EvaluationResult with grade, score, and actionable fixes.
    """
    if root is None:
        root = _find_root()
    else:
        root = Path(root)

    step_dir = _find_step_dir(root, step_num)
    title = step_dir.name.replace(f"step{step_num}-", "").replace("-", " ").title() if step_dir else "Unknown"

    # Run all criteria — four groups:
    #   Structural    (40 pts): paper, words, proofs, structure, bibliography
    #   Content       (35 pts): formal density, examples, benchmarks, cross-refs
    #   Writing       (20 pts): visual aids, narrative flow, active voice
    #   Implementation(25 pts): module, tests, tests pass
    #   Total: 120 pts → normalized to 100
    criteria: list[Criterion] = [
        # Structural (40 points)
        _eval_paper_exists(step_dir),
        _eval_word_count(step_dir),
        _eval_proofs(step_dir),
        _eval_paper_structure(step_dir),
        _eval_bibliography(step_dir),
        # Content quality (35 points)
        _eval_formal_density(step_dir),
        _eval_worked_examples(step_dir),
        _eval_benchmark_references(step_dir),
        _eval_cross_references(step_dir),
        # Writing & narrative quality (20 points)
        _eval_visual_aids(step_dir),
        _eval_narrative_flow(step_dir),
        _eval_active_voice(step_dir),
        # Implementation (25 points)
        _eval_module(root, step_num),
        _eval_tests(root, step_num),
    ]

    if run_tests:
        criteria.append(_eval_tests_pass(root, step_num))
    else:
        criteria.append(Criterion("Tests pass", 7, 7, True, "skipped (--no-tests)", "info"))

    # Compute score
    score = sum(c.points for c in criteria)
    max_score = sum(c.max_points for c in criteria)
    normalized = int(score / max_score * 100) if max_score > 0 else 0

    # Grade
    grade = "F"
    for g, threshold in _GRADE_THRESHOLDS.items():
        if normalized >= threshold:
            grade = g
            break

    # Identify fixes, strengths, weaknesses
    fixes: list[str] = []
    strengths: list[str] = []
    weaknesses: list[str] = []

    for c in criteria:
        if c.passed and c.points == c.max_points:
            strengths.append(f"{c.name}: {c.message}")
        elif not c.passed:
            weaknesses.append(f"{c.name}: {c.message}")
            if c.severity == "critical":
                fixes.append(f"[CRITICAL] {c.name}: {c.message}")
            elif c.severity == "major":
                fixes.append(f"[MAJOR] {c.name}: {c.message}")
            elif c.severity == "minor":
                fixes.append(f"[MINOR] {c.name}: {c.message}")

    accepted = grade == "A+"

    return EvaluationResult(
        step_number=step_num,
        step_title=title,
        grade=grade,
        score=normalized,
        accepted=accepted,
        criteria=tuple(criteria),
        fixes=tuple(fixes),
        strengths=tuple(strengths),
        weaknesses=tuple(weaknesses),
    )


def evaluate_all(
    root: Path | str | None = None,
    run_tests: bool = False,
) -> list[EvaluationResult]:
    """Evaluate all steps in the programme."""
    if root is None:
        root = _find_root()
    else:
        root = Path(root)

    steps_dir = root / "papers" / "steps"
    results = []
    for d in sorted(steps_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("step"):
            continue
        match = re.match(r"step(\d+\w*)-", d.name)
        if match:
            results.append(evaluate_step(match.group(1), root=root, run_tests=run_tests))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="reticulate.evaluator",
        description="Evaluate step deliverables for A+ grade",
    )
    parser.add_argument("step", nargs="?", help="Step number to evaluate (e.g. 23, 70b)")
    parser.add_argument("--all", action="store_true", help="Evaluate all steps")
    parser.add_argument("--run-tests", action="store_true", help="Actually run tests (slower)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--accepted-only", action="store_true", help="Only show accepted steps")
    parser.add_argument("--failing-only", action="store_true", help="Only show non-A+ steps")

    args = parser.parse_args(argv)

    if args.all:
        results = evaluate_all(run_tests=args.run_tests)

        if args.accepted_only:
            results = [r for r in results if r.accepted]
        if args.failing_only:
            results = [r for r in results if not r.accepted]

        if args.json:
            import json
            output = [{"step": r.step_number, "title": r.step_title,
                       "grade": r.grade, "score": r.score, "accepted": r.accepted,
                       "fixes": list(r.fixes)} for r in results]
            print(json.dumps(output, indent=2))
        else:
            # Summary table
            print(f"{'Step':>6} {'Title':<40} {'Grade':>5} {'Score':>5} {'Status':<10}")
            print("-" * 75)
            accepted = 0
            for r in results:
                status = "ACCEPTED" if r.accepted else f"{len(r.fixes)} fixes"
                print(f"{r.step_number:>6} {r.step_title:<40} {r.grade:>5} {r.score:>5} {status:<10}")
                if r.accepted:
                    accepted += 1
            print("-" * 75)
            print(f"  {accepted}/{len(results)} accepted (A+)")

    elif args.step:
        result = evaluate_step(args.step, run_tests=args.run_tests)
        if args.json:
            import json
            print(json.dumps({
                "step": result.step_number, "grade": result.grade,
                "score": result.score, "accepted": result.accepted,
                "criteria": [{"name": c.name, "points": c.points, "max": c.max_points,
                             "passed": c.passed, "message": c.message} for c in result.criteria],
                "fixes": list(result.fixes),
            }, indent=2))
        else:
            print(result.summary())
        sys.exit(0 if result.accepted else 1)
    else:
        parser.error("Provide a step number or --all")


if __name__ == "__main__":
    main()
