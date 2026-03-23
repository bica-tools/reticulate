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
    "1": "statespace", "2": "benchmarks", "3": "lattice", "4": "lattice",
    "5": "lattice", "5b": "product", "6": "enumerate_types", "6b": "termination",
    "7": "subtyping", "8": "duality", "9": "reticular", "10": "endomorphism",
    "11": "global_types", "12": "projection", "13": "recursion", "13a": "recursion",
    "14": "context_free", "21": "petri", "22": "marking_lattice",
    "23": "place_invariants", "24": "coverability", "25": "petri_benchmarks",
    "26": "ccs", "27": "csp", "28": "failures",
    "29": "distributive", "29a": "distributive", "29b": "gratzer",
    "30": "matrix", "70": "mcp_conformance", "70b": "mcp_server",
    "70e": "compress",
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

def _eval_paper_exists(step_dir: Path | None) -> Criterion:
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Paper exists", 15, 0, False, "main.tex not found", "critical")
    return Criterion("Paper exists", 15, 15, True, "main.tex present", "info")


def _eval_word_count(step_dir: Path | None) -> Criterion:
    if step_dir is None or not (step_dir / "main.tex").is_file():
        return Criterion("Word count ≥ 5000", 15, 0, False, "no paper", "critical")
    text = (step_dir / "main.tex").read_text()
    words = len(text.split())
    if words >= 5000:
        return Criterion("Word count ≥ 5000", 15, 15, True, f"{words} words", "info")
    elif words >= 4000:
        return Criterion("Word count ≥ 5000", 15, 10, False, f"{words} words (need {5000 - words} more)", "minor")
    elif words >= 3000:
        return Criterion("Word count ≥ 5000", 15, 5, False, f"{words} words (need {5000 - words} more)", "major")
    else:
        return Criterion("Word count ≥ 5000", 15, 0, False, f"{words} words (far below target)", "critical")


def _eval_proofs(step_dir: Path | None) -> Criterion:
    if step_dir is None:
        return Criterion("Companion proofs.tex", 15, 0, False, "no step directory", "critical")
    if not (step_dir / "proofs.tex").is_file():
        return Criterion("Companion proofs.tex", 15, 0, False, "proofs.tex missing", "major")
    text = (step_dir / "proofs.tex").read_text()
    words = len(text.split())
    theorems = text.count("\\begin{theorem}") + text.count("\\begin{proposition}") + text.count("\\begin{lemma}")
    if words >= 1000 and theorems >= 2:
        return Criterion("Companion proofs.tex", 15, 15, True, f"{words} words, {theorems} theorems", "info")
    elif words >= 500:
        return Criterion("Companion proofs.tex", 15, 10, False, f"{words} words, {theorems} theorems (thin)", "minor")
    else:
        return Criterion("Companion proofs.tex", 15, 5, False, f"{words} words (stub)", "major")


def _eval_module(root: Path, step_num: str) -> Criterion:
    mod_name = _MODULE_MAP.get(step_num, "")
    if not mod_name:
        # Some steps are paper-only (e.g. cross-domain applications)
        return Criterion("Implementation module", 15, 10, True, "paper-only step (no module expected)", "info")
    mod_path = root / "reticulate" / "reticulate" / f"{mod_name}.py"
    if not mod_path.is_file():
        return Criterion("Implementation module", 15, 0, False, f"{mod_name}.py not found", "critical")
    lines = len(mod_path.read_text().split("\n"))
    funcs = mod_path.read_text().count("\ndef ")
    if lines >= 100 and funcs >= 3:
        return Criterion("Implementation module", 15, 15, True, f"{mod_name}.py: {lines} lines, {funcs} functions", "info")
    elif lines >= 50:
        return Criterion("Implementation module", 15, 10, False, f"{mod_name}.py: {lines} lines (thin)", "minor")
    else:
        return Criterion("Implementation module", 15, 5, False, f"{mod_name}.py: {lines} lines (stub)", "major")


def _eval_tests(root: Path, step_num: str) -> Criterion:
    mod_name = _MODULE_MAP.get(step_num, "")
    if not mod_name:
        return Criterion("Test suite", 15, 10, True, "paper-only step", "info")
    test_path = root / "reticulate" / "tests" / f"test_{mod_name}.py"
    if not test_path.is_file():
        return Criterion("Test suite", 15, 0, False, f"test_{mod_name}.py not found", "critical")
    text = test_path.read_text()
    test_count = text.count("def test_")
    if test_count >= 20:
        return Criterion("Test suite", 15, 15, True, f"{test_count} tests", "info")
    elif test_count >= 10:
        return Criterion("Test suite", 15, 10, False, f"{test_count} tests (need 20+)", "minor")
    elif test_count >= 5:
        return Criterion("Test suite", 15, 5, False, f"{test_count} tests (need 20+)", "major")
    else:
        return Criterion("Test suite", 15, 0, False, f"{test_count} tests (far too few)", "critical")


def _eval_tests_pass(root: Path, step_num: str) -> Criterion:
    mod_name = _MODULE_MAP.get(step_num, "")
    if not mod_name:
        return Criterion("Tests pass", 10, 8, True, "paper-only step", "info")
    test_path = root / "reticulate" / "tests" / f"test_{mod_name}.py"
    if not test_path.is_file():
        return Criterion("Tests pass", 10, 0, False, "no test file", "critical")
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", str(test_path), "-q", "--tb=no", "--no-header"],
            capture_output=True, text=True, timeout=60,
            cwd=root / "reticulate",
        )
        if result.returncode == 0:
            # Extract pass count
            match = re.search(r"(\d+) passed", result.stdout)
            passed = int(match.group(1)) if match else 0
            return Criterion("Tests pass", 10, 10, True, f"{passed} passed", "info")
        else:
            match_fail = re.search(r"(\d+) failed", result.stdout)
            failed = int(match_fail.group(1)) if match_fail else 0
            return Criterion("Tests pass", 10, 0, False, f"{failed} failed", "critical")
    except (subprocess.TimeoutExpired, OSError):
        return Criterion("Tests pass", 10, 0, False, "test run failed/timeout", "critical")


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
    passed = sum(checks.values())
    missing = [k for k, v in checks.items() if not v]

    if passed == 4:
        return Criterion("Paper structure", 5, 5, True, "abstract, intro, conclusion, bibliography", "info")
    else:
        return Criterion("Paper structure", 5, passed, False, f"missing: {', '.join(missing)}", "minor")


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

    # Run all criteria
    criteria: list[Criterion] = [
        _eval_paper_exists(step_dir),
        _eval_word_count(step_dir),
        _eval_proofs(step_dir),
        _eval_module(root, step_num),
        _eval_tests(root, step_num),
        _eval_paper_structure(step_dir),
    ]

    if run_tests:
        criteria.append(_eval_tests_pass(root, step_num))
    else:
        criteria.append(Criterion("Tests pass", 10, 5, True, "skipped (--no-tests)", "info"))

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
