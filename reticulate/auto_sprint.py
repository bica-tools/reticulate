"""Autonomous sprint runner: evaluate → fix → re-evaluate loop.

Runs continuously, finding steps below A+ and generating fix commands.
The human becomes the research director; the agents handle execution.

Usage:
    # Dry run: show what would be fixed
    python -m reticulate.auto_sprint --dry-run

    # Run one sprint (fix all A-grade steps)
    python -m reticulate.auto_sprint --once

    # Run continuously until all steps are A+
    python -m reticulate.auto_sprint --until-done

    # Generate GitHub issue bodies for all failing steps
    python -m reticulate.auto_sprint --issues
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SprintPlan:
    """Plan for one sprint cycle."""
    sprint_number: int
    a_grade_fixes: list[dict]   # steps needing minor fixes
    b_grade_fixes: list[dict]   # steps needing moderate fixes
    c_grade_fixes: list[dict]   # steps needing major rework
    f_grade_fixes: list[dict]   # steps needing restart

    @property
    def total(self) -> int:
        return len(self.a_grade_fixes) + len(self.b_grade_fixes) + len(self.c_grade_fixes) + len(self.f_grade_fixes)


def _find_root() -> Path:
    for candidate in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
        if (candidate / "papers" / "steps").is_dir():
            return candidate
    return Path.cwd()


def evaluate_all() -> list[dict]:
    """Run evaluator on all steps, return results."""
    from reticulate.evaluator import evaluate_all as _eval_all
    results = _eval_all(run_tests=False)
    return [
        {
            "step": r.step_number,
            "title": r.step_title,
            "grade": r.grade,
            "score": r.score,
            "accepted": r.accepted,
            "fixes": list(r.fixes),
            "weaknesses": list(r.weaknesses),
        }
        for r in results
    ]


def plan_sprint(sprint_number: int = 1) -> SprintPlan:
    """Plan a sprint based on current evaluations."""
    results = evaluate_all()
    failing = [r for r in results if not r["accepted"]]

    return SprintPlan(
        sprint_number=sprint_number,
        a_grade_fixes=[r for r in failing if r["grade"] == "A"],
        b_grade_fixes=[r for r in failing if r["grade"] == "B"],
        c_grade_fixes=[r for r in failing if r["grade"] == "C"],
        f_grade_fixes=[r for r in failing if r["grade"] == "F"],
    )


def fix_description(step: dict) -> str:
    """Generate a human/agent-readable fix description for one step."""
    lines = [f"Step {step['step']} ({step['title']}) — Grade: {step['grade']}"]
    for fix in step["fixes"]:
        lines.append(f"  FIX: {fix}")
    return "\n".join(lines)


def agent_prompt(step: dict) -> str:
    """Generate the prompt an agent would need to fix this step."""
    root = _find_root()
    step_dir = None
    for d in (root / "papers" / "steps").iterdir():
        if d.is_dir() and d.name.startswith(f"step{step['step']}-"):
            step_dir = d
            break

    if step_dir is None:
        return f"Step {step['step']}: directory not found"

    prompts = []
    for fix in step["fixes"]:
        fix_lower = fix.lower()
        if "word count" in fix_lower:
            import re
            need_match = re.search(r"need (\d+) more", fix)
            need = int(need_match.group(1)) if need_match else 500
            prompts.append(
                f"Expand {step_dir}/main.tex by {need + 100} words. "
                f"Read the paper, find the thinnest section, add depth with "
                f"worked examples or benchmark data. Do NOT change structure."
            )
        elif "proofs.tex missing" in fix_lower:
            prompts.append(
                f"Create {step_dir}/proofs.tex with 1200+ words and 3+ "
                f"formal theorems. Read main.tex first to identify key results. "
                f"Use standard LaTeX (amsmath, amsthm)."
            )
        elif "proofs.tex" in fix_lower and "thin" in fix_lower:
            prompts.append(
                f"Expand {step_dir}/proofs.tex to 1200+ words. Add 1-2 new "
                f"propositions and deepen existing proofs with more detail."
            )
        elif "not found" in fix_lower and "module" in fix_lower:
            prompts.append(
                f"Implementation module missing for step {step['step']}. "
                f"This may be a paper-only step or the module map needs updating."
            )
        elif "test" in fix_lower and "not found" in fix_lower:
            prompts.append(
                f"Test file missing for step {step['step']}. "
                f"Create test file with 20+ tests covering the module."
            )
        elif "conclusion" in fix_lower:
            prompts.append(
                f"Add a Conclusion section to {step_dir}/main.tex."
            )
        else:
            prompts.append(f"Fix: {fix}")

    return "\n".join(prompts)


def github_issue_body(step: dict) -> str:
    """Generate a GitHub issue body for a failing step."""
    lines = [
        f"## Step {step['step']}: {step['title']}",
        f"",
        f"**Current grade**: {step['grade']} ({step['score']}/100)",
        f"**Target**: A+ (90+)",
        f"",
        f"### Required fixes",
        f"",
    ]
    for fix in step["fixes"]:
        lines.append(f"- [ ] {fix}")

    lines.extend([
        f"",
        f"### Agent prompt",
        f"```",
        agent_prompt(step),
        f"```",
        f"",
        f"### Labels",
        f"`step-improvement`, `grade-{step['grade'].lower()}`, `automated`",
    ])
    return "\n".join(lines)


def print_sprint_summary(plan: SprintPlan) -> None:
    """Print sprint plan summary."""
    print("=" * 65)
    print(f"  SPRINT {plan.sprint_number} PLAN")
    print("=" * 65)
    print()
    print(f"  Total steps to fix: {plan.total}")
    print(f"    A-grade (minor):  {len(plan.a_grade_fixes)}")
    print(f"    B-grade (moderate): {len(plan.b_grade_fixes)}")
    print(f"    C-grade (major):  {len(plan.c_grade_fixes)}")
    print(f"    F-grade (restart): {len(plan.f_grade_fixes)}")
    print()

    for label, steps in [
        ("A-GRADE (deploy Writer/Prover)", plan.a_grade_fixes),
        ("B-GRADE (deploy Writer + Prover)", plan.b_grade_fixes),
        ("C-GRADE (deploy full pipeline)", plan.c_grade_fixes),
        ("F-GRADE (start from scratch)", plan.f_grade_fixes),
    ]:
        if steps:
            print(f"  --- {label} ---")
            for s in steps:
                fixes = "; ".join(f.split("] ")[1][:45] if "] " in f else f[:45] for f in s["fixes"][:2])
                print(f"    Step {s['step']:>5} ({s['score']:>2}) {fixes}")
            print()

    print("=" * 65)


def print_agent_prompts(plan: SprintPlan) -> None:
    """Print agent prompts for all fixes."""
    all_steps = plan.a_grade_fixes + plan.b_grade_fixes + plan.c_grade_fixes + plan.f_grade_fixes
    for step in all_steps:
        print(f"\n--- Step {step['step']} ({step['grade']}) ---")
        print(agent_prompt(step))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="reticulate.auto_sprint",
        description="Autonomous sprint runner for step improvement",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--once", action="store_true", help="Run one sprint")
    parser.add_argument("--until-done", action="store_true", help="Loop until all A+")
    parser.add_argument("--issues", action="store_true", help="Generate GitHub issue bodies")
    parser.add_argument("--prompts", action="store_true", help="Print agent prompts for all fixes")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--sprint", type=int, default=1, help="Sprint number")

    args = parser.parse_args(argv)
    plan = plan_sprint(args.sprint)

    if args.json:
        output = {
            "sprint": plan.sprint_number,
            "total": plan.total,
            "a_fixes": plan.a_grade_fixes,
            "b_fixes": plan.b_grade_fixes,
            "c_fixes": plan.c_grade_fixes,
            "f_fixes": plan.f_grade_fixes,
        }
        print(json.dumps(output, indent=2))
    elif args.issues:
        all_steps = plan.a_grade_fixes + plan.b_grade_fixes + plan.c_grade_fixes + plan.f_grade_fixes
        for step in all_steps:
            print(github_issue_body(step))
            print("\n---\n")
    elif args.prompts:
        print_agent_prompts(plan)
    else:
        print_sprint_summary(plan)

    if plan.total == 0:
        print("ALL STEPS AT A+. Programme complete.")
        sys.exit(0)
    else:
        sys.exit(1 if not args.dry_run else 0)


if __name__ == "__main__":
    main()
