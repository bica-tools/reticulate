"""Session-type-enforced agent workflow for the research programme.

This is the execution engine that connects three layers:
  1. MCP transport — tool-calling agents (Researcher, Evaluator, Supervisor)
  2. A2A transport — task-delegation agents (Implementer, Tester, Writer, Prover)
  3. AgentMonitor — validates every transition against session types

The workflow follows the orchestrator protocol:
  rec S . &{scan: +{proposals:
    &{research: +{report:
      (&{implement: +{moduleReady: end}} || &{writePaper: +{paperReady: end}}) .
      (&{writeTests: +{testsPass: end}} || &{writeProofs: +{proofsReady: end}}) .
      &{evaluate: +{accepted: +{nextStep: S, allDone: end},
                     needsFixes: &{review: +{fixes: S}}}}}}}}

MCP agents call reticulate tools directly (in-process).
A2A agents delegate to claude CLI subprocess with structured task cards.

Usage:
    python3 -m reticulate.workflow run               # full sprint
    python3 -m reticulate.workflow run --steps 13 15  # specific steps
    python3 -m reticulate.workflow status             # show agent dashboard
    python3 -m reticulate.workflow verify             # verify all agent session types
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reticulate.agent_registry import (
    AGENTS,
    ORCHESTRATOR_TYPE,
    AgentInstance,
    AgentMonitor,
    AgentType,
)
from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


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
# Transport layer: MCP (in-process tool calls)
# ---------------------------------------------------------------------------

class MCPTransport:
    """MCP transport: calls reticulate tools directly in-process.

    Used by: Researcher, Evaluator, Supervisor, FactChecker, etc.
    Protocol: tool call → response (stateless, one-shot).
    """

    def __init__(self, root: Path = _ROOT) -> None:
        self.root = root

    def call_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Call a reticulate tool by name. Returns structured result."""
        if tool_name == "supervise":
            return self._supervise()
        elif tool_name == "evaluate":
            return self._evaluate(kwargs.get("step_number", ""))
        elif tool_name == "research":
            return self._research(kwargs.get("step_number", ""))
        elif tool_name == "analyze":
            return self._analyze(kwargs.get("type_string", "end"))
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _supervise(self) -> dict[str, Any]:
        from reticulate.supervisor import supervise
        report = supervise()
        return {
            "total_steps": report.snapshot.total_steps_planned,
            "complete": report.snapshot.complete_steps,
            "tests": report.snapshot.total_tests,
            "step_proposals": [
                {"step": p.step_number, "title": p.title, "reason": p.reason}
                for p in report.step_proposals[:10]
            ],
        }

    def _evaluate(self, step_number: str) -> dict[str, Any]:
        from reticulate.evaluator import evaluate_step
        result = evaluate_step(step_number, run_tests=True)
        return {
            "step": step_number,
            "grade": result.grade,
            "score": result.score,
            "accepted": result.accepted,
            "fixes": list(result.fixes),
            "weaknesses": list(result.weaknesses),
        }

    def _research(self, step_number: str) -> dict[str, Any]:
        """Research a step: check what exists, what's missing."""
        from reticulate.evaluator import _MODULE_MAP
        steps_dir = self.root / "papers" / "steps"
        sdir = None
        for d in steps_dir.iterdir():
            if d.is_dir() and d.name.startswith(f"step{step_number}-"):
                sdir = d
                break

        module = _MODULE_MAP.get(step_number)
        result: dict[str, Any] = {
            "step": step_number,
            "step_dir": str(sdir) if sdir else None,
            "module": module,
        }

        if sdir and (sdir / "main.tex").exists():
            text = (sdir / "main.tex").read_text(errors="replace")
            result["paper_words"] = len(text.split())
            result["has_proofs"] = (sdir / "proofs.tex").exists()
        else:
            result["paper_words"] = 0
            result["has_proofs"] = False

        if module:
            mod_path = self.root / "reticulate" / "reticulate" / f"{module}.py"
            test_path = self.root / "reticulate" / "tests" / f"test_{module}.py"
            result["module_exists"] = mod_path.exists()
            result["tests_exist"] = test_path.exists()

        return result

    def _analyze(self, type_string: str) -> dict[str, Any]:
        ast = parse(type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        return {
            "type": pretty(ast),
            "states": len(ss.states),
            "transitions": len(ss.transitions),
            "is_lattice": lr.is_lattice,
        }


# ---------------------------------------------------------------------------
# Transport layer: A2A (task delegation via claude CLI)
# ---------------------------------------------------------------------------

class A2ATransport:
    """A2A transport: delegates tasks to claude CLI subprocess.

    Used by: Implementer, Tester, Writer, Prover, Reviewer, etc.
    Protocol: sendTask → WORKING/COMPLETED/FAILED (stateful, async).

    Each task is a structured prompt sent to `claude --print -p <prompt>`.
    The response is parsed and returned as a result dict.
    """

    def __init__(self, root: Path = _ROOT, dry_run: bool = False) -> None:
        self.root = root
        self.dry_run = dry_run
        self._claude_path = shutil.which("claude")

    @property
    def available(self) -> bool:
        """Check if claude CLI is available."""
        return self._claude_path is not None

    def send_task(
        self,
        agent_type: str,
        step_number: str,
        prompt: str,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Send a task to a claude CLI agent. Returns result dict.

        This is the A2A "sendTask" operation. The agent processes the
        task autonomously and returns a structured result.
        """
        if self.dry_run:
            return {
                "status": "dry_run",
                "agent_type": agent_type,
                "step": step_number,
                "prompt_length": len(prompt),
            }

        if not self.available:
            return {
                "status": "error",
                "error": "claude CLI not found in PATH",
            }

        try:
            result = subprocess.run(
                [self._claude_path, "--print", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.root),
            )

            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "agent_type": agent_type,
                "step": step_number,
                "output": result.stdout[:5000],
                "error": result.stderr[:2000] if result.returncode != 0 else None,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": f"Task timed out after {timeout}s",
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
            }

    def get_status(self, task_id: str) -> dict[str, Any]:
        """Get status of a running task (A2A getStatus operation)."""
        # For subprocess-based execution, tasks are synchronous
        return {"task_id": task_id, "status": "completed"}

    def cancel(self, task_id: str) -> dict[str, Any]:
        """Cancel a running task (A2A cancel operation)."""
        return {"task_id": task_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# Workflow step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result from processing one research step."""
    step_number: str
    phases: list[PhaseResult] = field(default_factory=list)
    accepted: bool = False
    grade: str = ""
    fixes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class PhaseResult:
    """Result from one phase of a step."""
    name: str
    agents: list[AgentExecution] = field(default_factory=list)
    parallel: bool = False

    @property
    def all_ok(self) -> bool:
        return all(a.success for a in self.agents)


@dataclass
class AgentExecution:
    """Execution record for one agent."""
    agent_type: str
    agent_id: str
    transport: str  # "MCP" | "A2A"
    success: bool = False
    violated: bool = False
    transitions: list[str] = field(default_factory=list)
    output: Any = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Workflow engine
# ---------------------------------------------------------------------------

class Workflow:
    """Session-type-enforced agent workflow.

    Orchestrates the full step lifecycle:
      1. Scan (Supervisor via MCP)
      2. Research (Researcher via MCP)
      3. Implement || Write (parallel A2A)
      4. Test || Prove (parallel A2A)
      5. Evaluate (Evaluator via MCP)
      6. Review if needed (Reviewer via A2A)
    """

    def __init__(
        self,
        dry_run: bool = False,
        max_parallel: int = 2,
        verbose: bool = True,
    ) -> None:
        self.mcp = MCPTransport(root=_ROOT)
        self.a2a = A2ATransport(root=_ROOT, dry_run=dry_run)
        self.monitor = AgentMonitor()
        self.dry_run = dry_run
        self.max_parallel = max_parallel
        self.verbose = verbose
        self._sprint_count = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def _agent_id(self, agent_type: str, step: str) -> str:
        return f"{agent_type.lower()}-step{step}-{self._sprint_count}"

    # -- MCP agent execution --------------------------------------------------

    def _run_mcp_agent(
        self,
        agent_type: str,
        step: str,
        tool_name: str,
        **tool_kwargs: Any,
    ) -> AgentExecution:
        """Run an MCP-based agent: direct tool call + session type tracking."""
        start = time.time()
        agent_id = self._agent_id(agent_type, step)
        agent_def = AGENTS[agent_type]

        # Start monitoring
        self.monitor.start_agent(agent_type, agent_id, step)
        self._log(f"  [{agent_def.protocol}] {agent_type} started → {agent_id}")

        # Call tool
        result = self.mcp.call_tool(tool_name, step_number=step, **tool_kwargs)

        # Determine transitions from session type
        transitions = self._infer_transitions(agent_type, result)

        # Validate transitions
        violated = False
        for label in transitions:
            ok = self.monitor.transition(agent_id, label)
            if not ok:
                violated = True
                self._log(f"  [!!!] {agent_type} VIOLATED: {label}")
                break

        elapsed = (time.time() - start) * 1000
        success = not violated and "error" not in result

        exec_record = AgentExecution(
            agent_type=agent_type,
            agent_id=agent_id,
            transport="MCP",
            success=success,
            violated=violated,
            transitions=transitions,
            output=result,
            elapsed_ms=elapsed,
        )

        self._log(f"  [{agent_def.protocol}] {agent_type} {'OK' if success else 'FAIL'} ({elapsed:.0f}ms)")
        return exec_record

    # -- A2A agent execution --------------------------------------------------

    def _run_a2a_agent(
        self,
        agent_type: str,
        step: str,
        prompt: str,
    ) -> AgentExecution:
        """Run an A2A-based agent: delegate task to claude CLI."""
        start = time.time()
        agent_id = self._agent_id(agent_type, step)
        agent_def = AGENTS[agent_type]

        # Start monitoring
        self.monitor.start_agent(agent_type, agent_id, step)
        self._log(f"  [{agent_def.protocol}] {agent_type} started → {agent_id}")

        # Send task
        result = self.a2a.send_task(
            agent_type=agent_type,
            step_number=step,
            prompt=prompt,
        )

        # Determine transitions from result
        transitions = self._infer_transitions(agent_type, result)

        # Validate transitions
        violated = False
        for label in transitions:
            ok = self.monitor.transition(agent_id, label)
            if not ok:
                violated = True
                self._log(f"  [!!!] {agent_type} VIOLATED: {label}")
                break

        elapsed = (time.time() - start) * 1000
        success = not violated and result.get("status") != "failed"

        exec_record = AgentExecution(
            agent_type=agent_type,
            agent_id=agent_id,
            transport="A2A",
            success=success,
            violated=violated,
            transitions=transitions,
            output=result,
            elapsed_ms=elapsed,
        )

        self._log(f"  [{agent_def.protocol}] {agent_type} {'OK' if success else 'FAIL'} ({elapsed:.0f}ms)")
        return exec_record

    # -- Transition inference -------------------------------------------------

    def _infer_transitions(
        self, agent_type: str, result: dict[str, Any]
    ) -> list[str]:
        """Infer the session type transitions from an agent's result.

        Maps agent output to the correct path through its session type.
        """
        at = AGENTS.get(agent_type)
        if at is None:
            return []

        # Each agent type has known transition paths
        if agent_type == "Supervisor":
            return ["scan", "proposals"]

        elif agent_type == "Researcher":
            return ["investigate", "report"]

        elif agent_type == "Evaluator":
            if result.get("accepted"):
                return ["evaluate", "accepted"]
            else:
                return ["evaluate", "needsFixes", "listFixes"]

        elif agent_type == "Implementer":
            if result.get("status") == "failed" or result.get("error"):
                return ["implement", "error", "abort"]
            return ["implement", "moduleReady"]

        elif agent_type == "Tester":
            if result.get("status") == "failed":
                return ["writeTests", "testsFail", "abort"]
            return ["writeTests", "testsPass"]

        elif agent_type == "Writer":
            return ["writePaper", "paperReady"]

        elif agent_type == "Prover":
            return ["writeProofs", "proofsReady"]

        elif agent_type == "Reviewer":
            return ["review", "fixes"]

        else:
            # Generic: try first two labels in session type
            return []

    # -- Phase execution (parallel / sequential) ------------------------------

    def _run_phase(
        self,
        name: str,
        agents: list[tuple[str, str, dict[str, Any]]],
        parallel: bool = False,
    ) -> PhaseResult:
        """Run a phase with one or more agents.

        Args:
            name: Phase name for logging
            agents: List of (agent_type, step, kwargs) tuples.
                    kwargs must have 'tool_name' for MCP or 'prompt' for A2A.
            parallel: Whether to run agents concurrently.
        """
        self._log(f"\n  Phase: {name} {'(parallel)' if parallel else '(sequential)'}")
        phase = PhaseResult(name=name, parallel=parallel)

        if parallel and len(agents) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = {}
                for agent_type, step, kwargs in agents:
                    agent_def = AGENTS[agent_type]
                    if agent_def.protocol == "MCP":
                        fut = pool.submit(
                            self._run_mcp_agent, agent_type, step,
                            kwargs.get("tool_name", ""),
                        )
                    else:
                        fut = pool.submit(
                            self._run_a2a_agent, agent_type, step,
                            kwargs.get("prompt", ""),
                        )
                    futures[fut] = agent_type
                for fut in as_completed(futures):
                    phase.agents.append(fut.result())
        else:
            for agent_type, step, kwargs in agents:
                agent_def = AGENTS[agent_type]
                if agent_def.protocol == "MCP":
                    exec_result = self._run_mcp_agent(
                        agent_type, step, kwargs.get("tool_name", ""),
                    )
                else:
                    exec_result = self._run_a2a_agent(
                        agent_type, step, kwargs.get("prompt", ""),
                    )
                phase.agents.append(exec_result)

        return phase

    # -- Task card generation -------------------------------------------------

    def _generate_prompt(self, agent_type: str, step: str, context: dict[str, Any]) -> str:
        """Generate a task prompt for an A2A agent."""
        from reticulate.agent_tasks import (
            writer_task, prover_task, implementer_task, tester_task,
        )

        fixes = context.get("fixes", [])

        if agent_type == "Writer":
            card = writer_task(step, fixes)
            return card.prompt if card else f"Write a 5000+ word educational paper for step {step}."

        elif agent_type == "Prover":
            card = prover_task(step, fixes)
            return card.prompt if card else f"Write companion proofs for step {step}."

        elif agent_type == "Implementer":
            card = implementer_task(step, fixes)
            return card.prompt if card else f"Implement module for step {step}."

        elif agent_type == "Tester":
            card = tester_task(step, fixes)
            return card.prompt if card else f"Write tests for step {step}."

        elif agent_type == "Reviewer":
            return f"Review deliverables for step {step}. Grade: {context.get('grade', '?')}. Fixes needed: {fixes}"

        return f"Execute {agent_type} task for step {step}."

    # -- Step lifecycle -------------------------------------------------------

    def run_step(self, step: str) -> StepResult:
        """Run the full lifecycle for one research step.

        Follows the orchestrator session type:
          research → (implement || write) → (test || prove) → evaluate → [review]
        """
        start = time.time()
        self._log(f"\n{'='*60}")
        self._log(f"  Step {step}: Starting lifecycle")
        self._log(f"{'='*60}")

        result = StepResult(step_number=step)

        # Phase 1: Research (MCP — Researcher)
        p1 = self._run_phase("research", [
            ("Researcher", step, {"tool_name": "research"}),
        ])
        result.phases.append(p1)
        if not p1.all_ok:
            self._log(f"  Step {step}: Research failed, aborting.")
            result.elapsed_ms = (time.time() - start) * 1000
            return result

        research_output = p1.agents[0].output if p1.agents else {}

        # Phase 2: Implement || Write (A2A — parallel)
        impl_prompt = self._generate_prompt("Implementer", step, research_output)
        write_prompt = self._generate_prompt("Writer", step, research_output)

        p2 = self._run_phase("implement || write", [
            ("Implementer", step, {"prompt": impl_prompt}),
            ("Writer", step, {"prompt": write_prompt}),
        ], parallel=True)
        result.phases.append(p2)

        # Phase 3: Test || Prove (A2A — parallel)
        test_prompt = self._generate_prompt("Tester", step, research_output)
        prove_prompt = self._generate_prompt("Prover", step, research_output)

        p3 = self._run_phase("test || prove", [
            ("Tester", step, {"prompt": test_prompt}),
            ("Prover", step, {"prompt": prove_prompt}),
        ], parallel=True)
        result.phases.append(p3)

        # Phase 4: Evaluate (MCP — Evaluator)
        p4 = self._run_phase("evaluate", [
            ("Evaluator", step, {"tool_name": "evaluate"}),
        ])
        result.phases.append(p4)

        if p4.agents:
            eval_output = p4.agents[0].output or {}
            result.accepted = eval_output.get("accepted", False)
            result.grade = eval_output.get("grade", "")
            result.fixes = eval_output.get("fixes", [])

        # Phase 5: Review if needed (A2A — Reviewer)
        if not result.accepted:
            review_prompt = self._generate_prompt("Reviewer", step, {
                "grade": result.grade,
                "fixes": result.fixes,
            })
            p5 = self._run_phase("review", [
                ("Reviewer", step, {"prompt": review_prompt}),
            ])
            result.phases.append(p5)

        # Collect violations
        for phase in result.phases:
            for agent in phase.agents:
                if agent.violated:
                    result.violations.append(
                        f"{agent.agent_type}: {agent.output}"
                    )

        result.elapsed_ms = (time.time() - start) * 1000
        self._log(f"\n  Step {step}: {'ACCEPTED' if result.accepted else 'NEEDS WORK'} "
                  f"({result.grade}) [{result.elapsed_ms:.0f}ms]")
        return result

    # -- Sprint execution -----------------------------------------------------

    def run_sprint(self, steps: list[str] | None = None) -> list[StepResult]:
        """Run a full sprint: process multiple steps sequentially.

        If no steps specified, asks the Supervisor for proposals.
        """
        self._sprint_count += 1
        self._log(f"\n{'#'*60}")
        self._log(f"  SPRINT {self._sprint_count}")
        self._log(f"{'#'*60}")

        if steps is None:
            # Ask Supervisor for proposals
            self._log("\n  Asking Supervisor for step proposals...")
            scan_result = self.mcp.call_tool("supervise")
            proposals = scan_result.get("step_proposals", [])
            steps = [p["step"] for p in proposals[:5]]
            self._log(f"  Supervisor proposed: {steps}")

        results = []
        for step in steps:
            sr = self.run_step(step)
            results.append(sr)

        # Print summary
        self._log(f"\n{'='*60}")
        self._log(f"  SPRINT {self._sprint_count} SUMMARY")
        self._log(f"{'='*60}")
        for sr in results:
            status = "A+" if sr.accepted else sr.grade or "?"
            violations = f" [{len(sr.violations)} violations]" if sr.violations else ""
            self._log(f"  Step {sr.step_number}: {status}{violations}")

        # Conformance report
        report = self.monitor.conformance_report()
        self._log(f"\n  Agents: {report['total']} total, "
                  f"{report['completed']} completed, "
                  f"{report['violated']} violated")
        self._log(f"  Conformance rate: {report['conformance_rate']:.0%}")

        return results

    # -- Dashboard & verification ---------------------------------------------

    def dashboard(self) -> str:
        """Show the agent monitor dashboard."""
        return self.monitor.dashboard()

    def verify_protocols(self) -> dict[str, dict[str, Any]]:
        """Verify that all agent session types form lattices.

        Returns {agent_name: {session_type, states, transitions, is_lattice}}.
        """
        results = {}
        for name, agent in AGENTS.items():
            ast = parse(agent.session_type)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            results[name] = {
                "session_type": agent.session_type,
                "protocol": agent.protocol,
                "states": len(ss.states),
                "transitions": len(ss.transitions),
                "is_lattice": lr.is_lattice,
            }
        # Also verify orchestrator
        ast = parse(ORCHESTRATOR_TYPE)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        results["__Orchestrator__"] = {
            "session_type": ORCHESTRATOR_TYPE,
            "protocol": "hybrid",
            "states": len(ss.states),
            "transitions": len(ss.transitions),
            "is_lattice": lr.is_lattice,
        }
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI for the workflow engine."""
    parser = argparse.ArgumentParser(
        prog="workflow",
        description="Session-type-enforced agent workflow for research programme.",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Run a sprint")
    run_p.add_argument("--steps", nargs="+", help="Step numbers to process")
    run_p.add_argument("--dry-run", action="store_true", help="Don't invoke claude CLI")
    run_p.add_argument("--max-parallel", type=int, default=2, help="Max parallel agents")
    run_p.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # status
    sub.add_parser("status", help="Show agent monitor dashboard")

    # verify
    sub.add_parser("verify", help="Verify all agent session types form lattices")

    # protocols
    sub.add_parser("protocols", help="List all agent protocols")

    args = parser.parse_args(argv)

    if args.command == "run":
        wf = Workflow(
            dry_run=args.dry_run,
            max_parallel=args.max_parallel,
            verbose=not args.quiet,
        )
        results = wf.run_sprint(steps=args.steps)

        # Print final status
        print("\n" + wf.dashboard())

        # Exit code: 0 if all accepted, 1 otherwise
        all_ok = all(r.accepted for r in results)
        sys.exit(0 if all_ok else 1)

    elif args.command == "verify":
        wf = Workflow(dry_run=True, verbose=False)
        results = wf.verify_protocols()

        print(f"{'Agent':<25} {'Protocol':<8} {'States':>6} {'Trans':>6} {'Lattice':>8}")
        print("-" * 60)
        all_lattice = True
        for name, info in sorted(results.items()):
            check = "YES" if info["is_lattice"] else "NO"
            if not info["is_lattice"]:
                all_lattice = False
            print(f"{name:<25} {info['protocol']:<8} {info['states']:>6} {info['transitions']:>6} {check:>8}")

        total = len(results)
        lattices = sum(1 for v in results.values() if v["is_lattice"])
        print(f"\n{lattices}/{total} agent protocols form lattices.")
        sys.exit(0 if all_lattice else 1)

    elif args.command == "protocols":
        for name, agent in sorted(AGENTS.items()):
            print(f"\n{name} ({agent.protocol}):")
            print(f"  Type: {agent.session_type}")
            print(f"  Transport: {agent.transport}")
            print(f"  {agent.description}")

        print(f"\nOrchestrator:")
        print(f"  Type: {ORCHESTRATOR_TYPE}")

    elif args.command == "status":
        wf = Workflow(dry_run=True, verbose=False)
        # Run a verify to populate something
        print("No active sprint. Use 'run' to start one.")
        print("\nAvailable agents:")
        for name, agent in sorted(AGENTS.items()):
            print(f"  {name}: {agent.protocol} via {agent.transport}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
