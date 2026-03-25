"""Live dispatch: execute task cards via Claude Code Agent tool.

This is the bridge between the Python sprint loop and Claude Code's
Agent subprocess spawning. It takes TaskCards from execute_sprint and
produces structured dispatch plans that Claude Code can execute.

The dispatch plan groups cards by independence:
  - Independent cards (different steps, or parallel phases) → run in parallel
  - Dependent cards (same step, sequential phases) → run in order

Usage from Claude Code conversation:
    from reticulate.live_dispatch import plan_dispatch, format_agent_call

    plan = plan_dispatch()  # evaluate → find gaps → generate cards → group
    for batch in plan.batches:
        # Execute each batch in parallel via Agent tool
        for card in batch.cards:
            prompt = format_agent_call(card)
            # → Hand to Claude Code Agent tool

    # After all batches complete:
    from reticulate.live_dispatch import post_dispatch
    post_dispatch()  # re-evaluate, commit, push
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reticulate.agent_tasks import (
    TaskCard,
    generate_task_cards,
)
from reticulate.execute_sprint import (
    run_sprint_loop,
    SprintExecutionResult,
    _evaluate_all,
    _git_commit,
    _git_push,
    _find_root,
)


_ROOT = _find_root()


# ---------------------------------------------------------------------------
# Dispatch plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispatchBatch:
    """A batch of task cards that can run in parallel."""
    batch_number: int
    cards: tuple[TaskCard, ...]
    description: str

    @property
    def size(self) -> int:
        return len(self.cards)


@dataclass
class DispatchPlan:
    """A plan for executing task cards via Claude Code Agents."""
    batches: list[DispatchBatch] = field(default_factory=list)
    steps_failing: list[str] = field(default_factory=list)
    total_cards: int = 0
    accepted_before: int = 0
    total_steps: int = 0

    @property
    def all_clear(self) -> bool:
        return self.total_cards == 0

    def summary(self) -> str:
        if self.all_clear:
            return f"All clear: {self.accepted_before}/{self.total_steps} at A+. Nothing to dispatch."

        lines = [
            f"Dispatch Plan: {self.total_cards} cards in {len(self.batches)} batch(es)",
            f"  Programme: {self.accepted_before}/{self.total_steps} at A+",
            f"  Steps needing work: {', '.join(self.steps_failing)}",
            "",
        ]
        for batch in self.batches:
            lines.append(f"  Batch {batch.batch_number}: {batch.description}")
            for card in batch.cards:
                lines.append(f"    [{card.agent_type}] Step {card.step_number}: {card.action}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

def plan_dispatch() -> DispatchPlan:
    """Evaluate the programme and generate a dispatch plan.

    Groups task cards into batches:
      Batch 1: All Writer + Prover cards (independent, can run in parallel)
      Batch 2: All Implementer cards (can run in parallel)
      Batch 3: All Tester cards (depend on Implementer output)
    """
    results = _evaluate_all()
    accepted = sum(1 for r in results if r["accepted"])
    total = len(results)
    failing = [r for r in results if not r["accepted"]]

    plan = DispatchPlan(
        accepted_before=accepted,
        total_steps=total,
        steps_failing=[r["step"] for r in failing],
    )

    if not failing:
        return plan

    # Generate all task cards
    all_cards: list[TaskCard] = []
    for step_info in failing:
        cards = generate_task_cards(step_info["step"])
        all_cards.extend(cards)

    plan.total_cards = len(all_cards)

    if not all_cards:
        return plan

    # Group into batches by agent type (respecting dependencies)
    # Batch 1: Writers + Provers (independent content creation)
    write_cards = [c for c in all_cards if c.agent_type in ("Writer", "Prover")]
    # Batch 2: Implementers (code creation)
    impl_cards = [c for c in all_cards if c.agent_type == "Implementer"]
    # Batch 3: Testers (depend on implementation)
    test_cards = [c for c in all_cards if c.agent_type == "Tester"]
    # Batch 4: Everything else
    other_cards = [c for c in all_cards
                   if c.agent_type not in ("Writer", "Prover", "Implementer", "Tester")]

    batch_num = 0

    if write_cards:
        batch_num += 1
        plan.batches.append(DispatchBatch(
            batch_number=batch_num,
            cards=tuple(write_cards),
            description=f"Content creation: {len(write_cards)} paper/proof tasks (parallel)",
        ))

    if impl_cards:
        batch_num += 1
        plan.batches.append(DispatchBatch(
            batch_number=batch_num,
            cards=tuple(impl_cards),
            description=f"Implementation: {len(impl_cards)} module tasks (parallel)",
        ))

    if test_cards:
        batch_num += 1
        plan.batches.append(DispatchBatch(
            batch_number=batch_num,
            cards=tuple(test_cards),
            description=f"Testing: {len(test_cards)} test suite tasks (parallel, after impl)",
        ))

    if other_cards:
        batch_num += 1
        plan.batches.append(DispatchBatch(
            batch_number=batch_num,
            cards=tuple(other_cards),
            description=f"Other: {len(other_cards)} tasks",
        ))

    return plan


# ---------------------------------------------------------------------------
# Agent prompt formatting
# ---------------------------------------------------------------------------

def format_agent_call(card: TaskCard) -> str:
    """Format a TaskCard as a complete prompt for Claude Code's Agent tool.

    The returned string is the `prompt` parameter for the Agent tool call.
    It includes all context the agent needs to work autonomously.
    """
    sections = [card.prompt]

    if card.context_files:
        sections.append("\n## Files to read first")
        for f in card.context_files:
            sections.append(f"- {f}")

    if card.output_files:
        sections.append("\n## Files to create/edit")
        for f in card.output_files:
            sections.append(f"- {f}")

    if card.acceptance_criteria:
        sections.append("\n## Acceptance criteria")
        for criterion in card.acceptance_criteria:
            sections.append(f"- {criterion}")

    sections.append(f"\n## Session type transitions to report")
    sections.append(f"On success, report transitions: {' → '.join(card.transitions)}")

    sections.append("\n## Important")
    sections.append("- Read existing files before modifying them")
    sections.append("- Do NOT use placeholder text — every sentence must be substantive")
    sections.append("- Follow the project's code style (see CLAUDE.md)")
    sections.append("- Write the actual content, not instructions about what to write")

    return "\n".join(sections)


def format_batch_description(batch: DispatchBatch) -> str:
    """Human-readable description of a batch for status updates."""
    agent_types = set(c.agent_type for c in batch.cards)
    steps = set(c.step_number for c in batch.cards)
    return (
        f"Batch {batch.batch_number}: {batch.size} {'/'.join(agent_types)} "
        f"agent(s) on step(s) {', '.join(sorted(steps))}"
    )


# ---------------------------------------------------------------------------
# Post-dispatch: re-evaluate and commit
# ---------------------------------------------------------------------------

def post_dispatch(commit: bool = True, push: bool = True) -> dict[str, Any]:
    """Run after all Agent tool calls complete.

    Re-evaluates the programme, reports results, and optionally commits.
    """
    results = _evaluate_all()
    accepted = sum(1 for r in results if r["accepted"])
    total = len(results)
    failing = [r for r in results if not r["accepted"]]

    report = {
        "accepted": accepted,
        "total": total,
        "all_clear": len(failing) == 0,
        "still_failing": [
            {"step": r["step"], "grade": r["grade"], "score": r["score"],
             "fixes": r["fixes"]}
            for r in failing
        ],
    }

    if commit:
        committed = _git_commit(
            f"Sprint dispatch: {accepted}/{total} at A+\n\n"
            f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
        )
        report["committed"] = committed
        if committed and push:
            pushed = _git_push()
            report["pushed"] = pushed

    return report


# ---------------------------------------------------------------------------
# One-shot: evaluate + plan + return for execution
# ---------------------------------------------------------------------------

def dispatch_now() -> dict[str, Any]:
    """Main entry point for live dispatch from Claude Code.

    Returns a dict with:
      - plan: the DispatchPlan summary
      - batches: list of batches, each with formatted agent prompts
      - all_clear: True if nothing to do

    Claude Code should:
      1. Call dispatch_now()
      2. For each batch, spawn Agent tool calls in parallel
      3. After all agents complete, call post_dispatch()
    """
    plan = plan_dispatch()

    if plan.all_clear:
        return {
            "all_clear": True,
            "summary": plan.summary(),
            "accepted": plan.accepted_before,
            "total": plan.total_steps,
        }

    batches_data = []
    for batch in plan.batches:
        batch_data = {
            "batch_number": batch.batch_number,
            "description": batch.description,
            "size": batch.size,
            "agents": [],
        }
        for card in batch.cards:
            batch_data["agents"].append({
                "agent_type": card.agent_type,
                "step_number": card.step_number,
                "action": card.action,
                "prompt": format_agent_call(card),
                "output_files": list(card.output_files),
                "description": f"{card.agent_type} step {card.step_number}",
            })
        batches_data.append(batch_data)

    return {
        "all_clear": False,
        "summary": plan.summary(),
        "accepted": plan.accepted_before,
        "total": plan.total_steps,
        "batches": batches_data,
    }
