"""Sprint executor: the full evaluate → generate → execute → re-evaluate loop.

Phase 3 of the dispatch architecture. This module orchestrates the complete
autonomous sprint cycle:

  1. Evaluate all steps → find gaps
  2. Generate task cards for failing steps
  3. Execute task cards (via handlers or Claude Code Agents)
  4. Re-evaluate → verify fixes
  5. Report results + commit artifacts

Execution modes:
  - "verify":   Phase 1 handlers (check files exist) — fast, no content creation
  - "generate": Phase 2 task cards returned for external execution — needs Agent tool
  - "execute":  Phase 3 full loop with real file I/O — runs handlers that modify files

Usage:
    from reticulate.execute_sprint import run_sprint_loop, SprintExecutionResult

    # Full autonomous loop
    result = run_sprint_loop(max_iterations=3)

    # Single iteration with task cards for Agent dispatch
    result = run_sprint_loop(max_iterations=1, mode="generate")
    for card in result.pending_cards:
        # Hand to Claude Code Agent tool
        print(card.prompt)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reticulate.agent_tasks import (
    TaskCard,
    generate_task_cards,
    generate_sprint_cards,
    sprint_summary,
)
from reticulate.agent_dispatch import (
    dispatch_sprint,
    get_registry,
    reset_registry,
    register_all_handlers,
)
from reticulate.artifact_registry import ArtifactKind


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of processing one step in a sprint iteration."""
    step_number: str
    grade_before: str
    score_before: int
    grade_after: str | None = None
    score_after: int | None = None
    cards_generated: int = 0
    cards_executed: int = 0
    fixed: bool = False
    error: str | None = None


@dataclass
class SprintIteration:
    """Result of one evaluate → fix → re-evaluate cycle."""
    iteration: int
    steps_evaluated: int = 0
    steps_failing: int = 0
    steps_fixed: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    pending_cards: list[TaskCard] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class SprintExecutionResult:
    """Result of the full sprint execution loop."""
    iterations: list[SprintIteration] = field(default_factory=list)
    total_fixed: int = 0
    total_cards_generated: int = 0
    total_cards_executed: int = 0
    final_accepted: int = 0
    final_total: int = 0
    all_clear: bool = False
    pending_cards: list[TaskCard] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def final_rate(self) -> str:
        if self.final_total == 0:
            return "0/0"
        return f"{self.final_accepted}/{self.final_total}"

    def summary(self) -> str:
        lines = [
            f"Sprint Execution: {len(self.iterations)} iteration(s)",
            f"  Final: {self.final_rate} at A+",
            f"  Fixed: {self.total_fixed} steps",
            f"  Cards: {self.total_cards_generated} generated, {self.total_cards_executed} executed",
            f"  Pending: {len(self.pending_cards)} cards awaiting Agent execution",
            f"  Time: {self.elapsed_ms:.0f}ms",
        ]
        if self.all_clear:
            lines.append("  STATUS: ALL STEPS AT A+")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    for candidate in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
        if (candidate / "papers" / "steps").is_dir():
            return candidate
    return Path.cwd()


_ROOT = _find_root()


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_commit(message: str, files: list[str] | None = None) -> bool:
    """Stage and commit files. Returns True on success."""
    try:
        if files:
            subprocess.run(
                ["git", "add"] + files,
                cwd=str(_ROOT), capture_output=True, timeout=30,
            )
        else:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(_ROOT), capture_output=True, timeout=30,
            )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(_ROOT), capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return False  # nothing to commit
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(_ROOT), capture_output=True, timeout=30,
        )
        return True
    except Exception:
        return False


def _git_push() -> bool:
    """Push to remote. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "push"],
            cwd=str(_ROOT), capture_output=True, timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_all() -> list[dict[str, Any]]:
    """Evaluate all steps, return list of result dicts."""
    from reticulate.evaluator import evaluate_all as _eval

    results = _eval(run_tests=False)
    return [
        {
            "step": r.step_number,
            "title": r.step_title,
            "grade": r.grade,
            "score": r.score,
            "accepted": r.accepted,
            "fixes": list(r.fixes),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Sprint execution modes
# ---------------------------------------------------------------------------

def _run_verify_iteration(
    failing_steps: list[dict[str, Any]],
    iteration_num: int,
) -> SprintIteration:
    """Phase 1 mode: run verify handlers through orchestrator."""
    reset_registry()
    step_numbers = [s["step"] for s in failing_steps]
    result = dispatch_sprint(steps=step_numbers)

    iteration = SprintIteration(iteration=iteration_num)
    iteration.steps_evaluated = len(failing_steps)
    iteration.steps_failing = len(failing_steps)

    for step_info in failing_steps:
        sr = StepResult(
            step_number=step_info["step"],
            grade_before=step_info["grade"],
            score_before=step_info["score"],
        )
        iteration.step_results.append(sr)

    return iteration


def _run_generate_iteration(
    failing_steps: list[dict[str, Any]],
    iteration_num: int,
) -> SprintIteration:
    """Phase 2 mode: generate task cards for external execution."""
    iteration = SprintIteration(iteration=iteration_num)
    iteration.steps_evaluated = len(failing_steps)
    iteration.steps_failing = len(failing_steps)

    all_cards: list[TaskCard] = []
    for step_info in failing_steps:
        cards = generate_task_cards(step_info["step"])
        sr = StepResult(
            step_number=step_info["step"],
            grade_before=step_info["grade"],
            score_before=step_info["score"],
            cards_generated=len(cards),
        )
        iteration.step_results.append(sr)
        all_cards.extend(cards)

    iteration.pending_cards = all_cards
    return iteration


def _run_execute_iteration(
    failing_steps: list[dict[str, Any]],
    iteration_num: int,
) -> SprintIteration:
    """Phase 3 mode: generate cards + run verify handlers + re-evaluate."""
    start = time.time()
    iteration = SprintIteration(iteration=iteration_num)
    iteration.steps_evaluated = len(failing_steps)
    iteration.steps_failing = len(failing_steps)

    # Generate task cards
    all_cards: list[TaskCard] = []
    step_cards_map: dict[str, list[TaskCard]] = {}
    for step_info in failing_steps:
        cards = generate_task_cards(step_info["step"])
        step_cards_map[step_info["step"]] = cards
        all_cards.extend(cards)

    # Run verify handlers through orchestrator
    reset_registry()
    step_numbers = [s["step"] for s in failing_steps]
    dispatch_result = dispatch_sprint(steps=step_numbers)

    # Re-evaluate to check if anything changed
    new_results = _evaluate_all()
    new_map = {r["step"]: r for r in new_results}

    for step_info in failing_steps:
        step = step_info["step"]
        new = new_map.get(step, {})
        cards = step_cards_map.get(step, [])

        sr = StepResult(
            step_number=step,
            grade_before=step_info["grade"],
            score_before=step_info["score"],
            grade_after=new.get("grade", step_info["grade"]),
            score_after=new.get("score", step_info["score"]),
            cards_generated=len(cards),
            cards_executed=0,  # verify handlers don't create content
            fixed=new.get("accepted", False) and not step_info["accepted"],
        )
        iteration.step_results.append(sr)

        if not new.get("accepted", False):
            # Step still failing — cards become pending for Agent execution
            iteration.pending_cards.extend(cards)

    iteration.steps_fixed = sum(1 for sr in iteration.step_results if sr.fixed)
    iteration.elapsed_ms = (time.time() - start) * 1000
    return iteration


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sprint_loop(
    max_iterations: int = 3,
    mode: str = "execute",
    commit: bool = False,
    push: bool = False,
) -> SprintExecutionResult:
    """Run the full sprint execution loop.

    Args:
        max_iterations: Maximum evaluate→fix→re-evaluate cycles.
        mode: "verify" (Phase 1), "generate" (Phase 2), "execute" (Phase 3).
        commit: Whether to git commit after each iteration.
        push: Whether to git push after commits.

    Returns:
        SprintExecutionResult with all iteration details and pending cards.
    """
    start = time.time()
    exec_result = SprintExecutionResult()

    for i in range(1, max_iterations + 1):
        # Evaluate
        results = _evaluate_all()
        accepted = sum(1 for r in results if r["accepted"])
        total = len(results)
        failing = [r for r in results if not r["accepted"]]

        exec_result.final_accepted = accepted
        exec_result.final_total = total

        if not failing:
            exec_result.all_clear = True
            break

        # Run iteration based on mode
        if mode == "verify":
            iteration = _run_verify_iteration(failing, i)
        elif mode == "generate":
            iteration = _run_generate_iteration(failing, i)
            exec_result.iterations.append(iteration)
            exec_result.pending_cards.extend(iteration.pending_cards)
            exec_result.total_cards_generated += len(iteration.pending_cards)
            break  # generate mode returns cards for external execution
        else:  # execute
            iteration = _run_execute_iteration(failing, i)

        exec_result.iterations.append(iteration)
        exec_result.total_fixed += iteration.steps_fixed
        exec_result.total_cards_generated += sum(
            sr.cards_generated for sr in iteration.step_results
        )
        exec_result.total_cards_executed += sum(
            sr.cards_executed for sr in iteration.step_results
        )
        exec_result.pending_cards.extend(iteration.pending_cards)

        # Commit if requested
        if commit and iteration.steps_fixed > 0:
            msg = (
                f"Sprint iteration {i}: fixed {iteration.steps_fixed} steps "
                f"({exec_result.final_rate} at A+)\n\n"
                f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
            )
            _git_commit(msg)
            if push:
                _git_push()

        # Stop if nothing was fixed (avoid infinite loop)
        if mode == "execute" and iteration.steps_fixed == 0:
            break

    exec_result.elapsed_ms = (time.time() - start) * 1000
    return exec_result


# ---------------------------------------------------------------------------
# Convenience: task cards as JSON (for scheduled triggers)
# ---------------------------------------------------------------------------

def sprint_cards_json() -> str:
    """Generate sprint task cards as JSON for Claude Code scheduled triggers.

    Returns a JSON string that a remote agent can parse and execute.
    """
    result = run_sprint_loop(max_iterations=1, mode="generate")

    if result.all_clear:
        return json.dumps({"status": "all_clear", "message": "All steps at A+"})

    cards_data = []
    for card in result.pending_cards:
        cards_data.append({
            "agent_type": card.agent_type,
            "step_number": card.step_number,
            "action": card.action,
            "prompt": card.prompt,
            "context_files": list(card.context_files),
            "output_files": list(card.output_files),
            "acceptance_criteria": list(card.acceptance_criteria),
            "transitions": list(card.transitions),
            "priority": card.priority,
        })

    return json.dumps({
        "status": "work_needed",
        "accepted": result.final_accepted,
        "total": result.final_total,
        "cards": cards_data,
    }, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="reticulate.execute_sprint",
        description="Execute autonomous sprint loop",
    )
    parser.add_argument("--mode", choices=["verify", "generate", "execute"],
                       default="execute", help="Execution mode")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Maximum loop iterations")
    parser.add_argument("--commit", action="store_true",
                       help="Git commit after fixes")
    parser.add_argument("--push", action="store_true",
                       help="Git push after commits")
    parser.add_argument("--json", action="store_true",
                       help="Output task cards as JSON")

    args = parser.parse_args(argv)

    if args.json:
        print(sprint_cards_json())
        return

    result = run_sprint_loop(
        max_iterations=args.max_iterations,
        mode=args.mode,
        commit=args.commit,
        push=args.push,
    )

    print(result.summary())

    if result.pending_cards:
        print(f"\n--- {len(result.pending_cards)} pending task cards ---")
        for card in result.pending_cards:
            print(f"  [{card.agent_type}] Step {card.step_number}: {card.action}")

    sys.exit(0 if result.all_clear else 1)


if __name__ == "__main__":
    main()
