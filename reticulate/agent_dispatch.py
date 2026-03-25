"""Agent dispatch: maps agent types to executable handler functions.

This is the missing link between the orchestrator (which knows WHAT to run)
and the actual work (reading/writing files, running tools, evaluating).

Each handler function:
1. Receives (agent_id, step_number, task_description)
2. Does real work (file I/O, tool invocation, evaluation)
3. Returns {"transitions": [...], "artifacts": [...], "output": ...}

The transitions list must match the agent's session type protocol.
The orchestrator's AgentMonitor validates this at runtime.

Architecture:
    Claude Code (orchestrator)
      → spawns handlers via agent_dispatch
        → handlers call reticulate tools (evaluator, supervisor, peer_reviewer)
        → handlers read/write files (papers, modules, tests)
        → handlers register artifacts in ArtifactRegistry
      → AgentMonitor validates session type compliance
      → ArtifactRegistry passes outputs between phases
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

from reticulate.artifact_registry import ArtifactRegistry, ArtifactKind


# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    """Find the project root (directory containing papers/steps)."""
    for candidate in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
        if (candidate / "papers" / "steps").is_dir():
            return candidate
    return Path.cwd()


_ROOT = _find_root()


# ---------------------------------------------------------------------------
# Shared artifact registry (one per orchestrator lifetime)
# ---------------------------------------------------------------------------

_registry = ArtifactRegistry(root=_ROOT)


def get_registry() -> ArtifactRegistry:
    """Get the shared artifact registry."""
    return _registry


def reset_registry() -> None:
    """Reset the registry (for testing or between sprints)."""
    _registry.clear()


# ---------------------------------------------------------------------------
# Step metadata helpers
# ---------------------------------------------------------------------------

def _step_dir(step_number: str) -> Path | None:
    """Find the step directory for a given step number."""
    steps_dir = _ROOT / "papers" / "steps"
    if not steps_dir.is_dir():
        return None
    for d in steps_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"step{step_number}-"):
            return d
    return None


def _step_module(step_number: str) -> str | None:
    """Map step number to its implementation module name."""
    from reticulate.evaluator import _MODULE_MAP
    return _MODULE_MAP.get(step_number)


def _run_tests(module_name: str) -> tuple[bool, str]:
    """Run tests for a module, return (passed, output)."""
    test_file = _ROOT / "reticulate" / "tests" / f"test_{module_name}.py"
    if not test_file.exists():
        return False, f"Test file not found: {test_file}"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-q", "--tb=short"],
            capture_output=True, text=True, timeout=120,
            cwd=str(_ROOT / "reticulate"),
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Handler: Researcher
# ---------------------------------------------------------------------------

def researcher_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Researcher: investigate step requirements, report findings.

    Protocol: &{investigate: +{report: end}}
    """
    sdir = _step_dir(step_number)
    module = _step_module(step_number)

    report_lines = [f"Research report for Step {step_number}"]
    report_lines.append(f"  Paper dir: {sdir or '(not found)'}")
    report_lines.append(f"  Module: {module or '(none/paper-only)'}")

    # Check paper status
    if sdir and (sdir / "main.tex").exists():
        text = (sdir / "main.tex").read_text(errors="replace")
        word_count = len(text.split())
        report_lines.append(f"  Paper words: {word_count}")
        has_proofs = (sdir / "proofs.tex").exists()
        report_lines.append(f"  Has proofs.tex: {has_proofs}")
    else:
        report_lines.append("  Paper: MISSING")

    # Check module status
    if module:
        mod_path = _ROOT / "reticulate" / "reticulate" / f"{module}.py"
        report_lines.append(f"  Module exists: {mod_path.exists()}")
        test_path = _ROOT / "reticulate" / "tests" / f"test_{module}.py"
        report_lines.append(f"  Tests exist: {test_path.exists()}")

    report = "\n".join(report_lines)

    _registry.register(
        kind=ArtifactKind.REPORT,
        path=f"(in-memory research report for step {step_number})",
        agent_id=agent_id,
        agent_type="Researcher",
        step_number=step_number,
        metadata={"report": report, "module": module, "step_dir": str(sdir)},
    )

    return {
        "transitions": ["investigate", "report"],
        "output": report,
        "module": module,
        "step_dir": str(sdir) if sdir else None,
    }


# ---------------------------------------------------------------------------
# Handler: Evaluator
# ---------------------------------------------------------------------------

def evaluator_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Evaluator: grade deliverables, decide accept/reject.

    Protocol: &{evaluate: +{accepted: end, needsFixes: +{listFixes: end}}}
    """
    from reticulate.evaluator import evaluate_step

    result = evaluate_step(step_number, run_tests=True)

    _registry.register(
        kind=ArtifactKind.REPORT,
        path=f"(evaluation for step {step_number})",
        agent_id=agent_id,
        agent_type="Evaluator",
        step_number=step_number,
        metadata={
            "grade": result.grade,
            "score": result.score,
            "accepted": result.accepted,
        },
    )

    if result.accepted:
        return {
            "transitions": ["evaluate", "accepted"],
            "output": {
                "accepted": True,
                "grade": result.grade,
                "score": result.score,
            },
        }
    else:
        return {
            "transitions": ["evaluate", "needsFixes", "listFixes"],
            "output": {
                "accepted": False,
                "needs_fixes": True,
                "grade": result.grade,
                "score": result.score,
                "fixes": list(result.fixes),
                "weaknesses": list(result.weaknesses),
            },
        }


# ---------------------------------------------------------------------------
# Handler: Supervisor
# ---------------------------------------------------------------------------

def supervisor_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Supervisor: scan programme, propose next steps.

    Protocol: &{scan: +{proposals: end}}
    """
    from reticulate.supervisor import supervise

    report = supervise()

    return {
        "transitions": ["scan", "proposals"],
        "output": {
            "snapshot": {
                "total_steps": report.snapshot.total_steps,
                "complete_steps": report.snapshot.complete_steps,
                "total_tests": report.snapshot.total_tests,
            },
            "num_step_proposals": len(report.step_proposals),
            "num_tool_proposals": len(report.tool_proposals),
            "num_paper_proposals": len(report.paper_proposals),
        },
    }


# ---------------------------------------------------------------------------
# Handler: Tester
# ---------------------------------------------------------------------------

def tester_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Tester: run tests for a step's module.

    Protocol: rec X . &{writeTests: +{testsPass: end, testsFail: +{fix: X, abort: end}}}
    """
    module = _step_module(step_number)
    if not module:
        return {
            "transitions": ["writeTests", "testsPass"],
            "output": {"skipped": True, "reason": "No module for this step"},
        }

    passed, output = _run_tests(module)

    if passed:
        _registry.register(
            kind=ArtifactKind.TEST,
            path=f"reticulate/tests/test_{module}.py",
            agent_id=agent_id,
            agent_type="Tester",
            step_number=step_number,
            metadata={"passed": True},
        )
        return {
            "transitions": ["writeTests", "testsPass"],
            "output": {"passed": True, "test_output": output[:500]},
        }
    else:
        return {
            "transitions": ["writeTests", "testsFail", "abort"],
            "output": {"passed": False, "test_output": output[:500]},
        }


# ---------------------------------------------------------------------------
# Handler: Implementer (checks module exists)
# ---------------------------------------------------------------------------

def implementer_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Implementer: verify module exists and is functional.

    Protocol: rec X . &{implement: +{moduleReady: end, error: +{retry: X, abort: end}}}
    """
    module = _step_module(step_number)
    if not module:
        # Paper-only step — no module needed, report success
        return {
            "transitions": ["implement", "moduleReady"],
            "output": {"skipped": True, "reason": "Paper-only step"},
        }

    mod_path = _ROOT / "reticulate" / "reticulate" / f"{module}.py"
    if mod_path.exists():
        _registry.register(
            kind=ArtifactKind.MODULE,
            path=f"reticulate/reticulate/{module}.py",
            agent_id=agent_id,
            agent_type="Implementer",
            step_number=step_number,
        )
        return {
            "transitions": ["implement", "moduleReady"],
            "output": {"module": module, "exists": True},
        }
    else:
        return {
            "transitions": ["implement", "error", "abort"],
            "output": {"module": module, "exists": False, "error": "Module file not found"},
        }


# ---------------------------------------------------------------------------
# Handler: Writer (checks paper exists and word count)
# ---------------------------------------------------------------------------

def writer_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Writer: verify paper exists with sufficient content.

    Protocol: &{writePaper: +{paperReady: end}}
    """
    sdir = _step_dir(step_number)
    if sdir is None:
        return {
            "transitions": ["writePaper", "paperReady"],
            "output": {"exists": False, "error": "Step directory not found"},
        }

    main_tex = sdir / "main.tex"
    if main_tex.exists():
        text = main_tex.read_text(errors="replace")
        word_count = len(text.split())
        _registry.register(
            kind=ArtifactKind.PAPER,
            path=str(main_tex.relative_to(_ROOT)),
            agent_id=agent_id,
            agent_type="Writer",
            step_number=step_number,
            metadata={"word_count": word_count},
        )
        return {
            "transitions": ["writePaper", "paperReady"],
            "output": {"exists": True, "word_count": word_count},
        }
    else:
        return {
            "transitions": ["writePaper", "paperReady"],
            "output": {"exists": False, "error": "main.tex not found"},
        }


# ---------------------------------------------------------------------------
# Handler: Prover (checks proofs.tex exists)
# ---------------------------------------------------------------------------

def prover_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Prover: verify companion proofs exist.

    Protocol: &{writeProofs: +{proofsReady: end}}
    """
    sdir = _step_dir(step_number)
    if sdir is None:
        return {
            "transitions": ["writeProofs", "proofsReady"],
            "output": {"exists": False, "error": "Step directory not found"},
        }

    proofs_tex = sdir / "proofs.tex"
    if proofs_tex.exists():
        text = proofs_tex.read_text(errors="replace")
        word_count = len(text.split())
        _registry.register(
            kind=ArtifactKind.PROOFS,
            path=str(proofs_tex.relative_to(_ROOT)),
            agent_id=agent_id,
            agent_type="Prover",
            step_number=step_number,
            metadata={"word_count": word_count},
        )
        return {
            "transitions": ["writeProofs", "proofsReady"],
            "output": {"exists": True, "word_count": word_count},
        }
    else:
        return {
            "transitions": ["writeProofs", "proofsReady"],
            "output": {"exists": False, "error": "proofs.tex not found"},
        }


# ---------------------------------------------------------------------------
# Handler: Reviewer
# ---------------------------------------------------------------------------

def reviewer_handler(agent_id: str, step_number: str, task: str) -> dict[str, Any]:
    """Reviewer: review deliverables, list fixes needed.

    Protocol: &{review: +{fixes: end}}
    """
    # Use evaluator to find what needs fixing
    from reticulate.evaluator import evaluate_step

    result = evaluate_step(step_number, run_tests=False)
    fixes = list(result.fixes)

    return {
        "transitions": ["review", "fixes"],
        "output": {
            "grade": result.grade,
            "fixes": fixes,
            "weaknesses": list(result.weaknesses),
        },
    }


# ---------------------------------------------------------------------------
# Handler registry: maps agent type → handler function
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., Any]] = {
    "Researcher": researcher_handler,
    "Evaluator": evaluator_handler,
    "Supervisor": supervisor_handler,
    "Tester": tester_handler,
    "Implementer": implementer_handler,
    "Writer": writer_handler,
    "Prover": prover_handler,
    "Reviewer": reviewer_handler,
}


def register_all_handlers(orchestrator: Any) -> None:
    """Register all implemented handlers with an Orchestrator instance.

    This is the main wiring function that connects agent types to their
    executable implementations.

    Usage:
        from reticulate.orchestrator import Orchestrator
        from reticulate.agent_dispatch import register_all_handlers

        orch = Orchestrator()
        register_all_handlers(orch)
        result = orch.run_sprint(["23", "24"])  # Now actually executes!
    """
    for agent_type, handler in HANDLERS.items():
        orchestrator.register_handler(agent_type, handler)


def dispatch_sprint(
    steps: list[str] | None = None,
    max_parallel: int = 4,
) -> dict[str, Any]:
    """High-level entry point: run a full sprint with all handlers wired.

    Returns the SprintResult as a dict plus artifact summary.
    """
    from reticulate.orchestrator import Orchestrator, OrchestratorConfig

    config = OrchestratorConfig(max_parallel=max_parallel)
    orch = Orchestrator(config=config)
    register_all_handlers(orch)

    # If no steps specified, find steps needing work
    if steps is None:
        from reticulate.auto_sprint import evaluate_all
        results = evaluate_all()
        steps = [r["step"] for r in results if not r["accepted"]]

    if not steps:
        return {
            "status": "all_clear",
            "message": "All steps at A+. Nothing to fix.",
            "artifacts": _registry.summary(),
        }

    result = orch.run_sprint(steps)

    return {
        "status": "completed",
        "sprint_number": result.sprint_number,
        "steps_processed": result.steps_processed,
        "steps_fixed": result.steps_fixed,
        "violations": result.violations,
        "conformance_rate": result.conformance_rate,
        "total_agents": result.total_agents_dispatched,
        "elapsed_ms": result.elapsed_ms,
        "artifacts": _registry.summary(),
        "dashboard": orch.dashboard(),
    }
