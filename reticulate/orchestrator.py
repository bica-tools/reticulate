"""Monitored parallel orchestrator for session-typed agent sprints.

Replaces the ad-hoc sprint loop with a proper session-typed orchestrator that:
1. Uses AgentMonitor to track every agent's transitions
2. Dispatches agents in parallel where independent
3. Rejects results from agents that violate their protocol

The orchestrator's own protocol uses parallel constructors:
  rec S . &{scan: +{proposals:
    &{research: +{report:
      (implement || writePaper) .
      (writeTests || writeProofs) .
      &{evaluate: +{accepted: +{nextStep: S, allDone: end},
                     needsFixes: &{review: +{fixes: S}}}}}}}}

This models the real workflow: implementation and paper writing are independent
(Phase 2), and testing and proving are independent (Phase 3).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from reticulate.agent_registry import AgentMonitor, AGENTS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_parallel: int = 4
    monitor_enabled: bool = True
    reject_violations: bool = True
    max_retry: int = 3


@dataclass(frozen=True)
class AgentTask:
    """A task to be dispatched to an agent."""
    agent_type: str
    step_number: str
    task_description: str
    handler: Callable[..., Any] | None = None  # if None, uses default stub


@dataclass
class AgentResult:
    """Result from a single agent execution."""
    agent_id: str
    agent_type: str
    step_number: str
    success: bool
    violated: bool = False
    violation_detail: str | None = None
    output: Any = None
    elapsed_ms: float = 0.0


@dataclass
class ParallelPhaseResult:
    """Result from a parallel phase (multiple agents)."""
    phase_name: str
    agents: list[str]
    results: list[AgentResult] = field(default_factory=list)

    @property
    def all_completed(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def any_violated(self) -> bool:
        return any(r.violated for r in self.results)

    @property
    def succeeded_results(self) -> list[AgentResult]:
        return [r for r in self.results if r.success and not r.violated]


@dataclass
class SprintResult:
    """Result from one complete sprint."""
    sprint_number: int
    steps_processed: list[str] = field(default_factory=list)
    steps_fixed: int = 0
    steps_created: int = 0
    violations: list[str] = field(default_factory=list)
    phase_results: list[ParallelPhaseResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def conformance_rate(self) -> float:
        total = sum(len(pr.results) for pr in self.phase_results)
        violated = sum(1 for pr in self.phase_results for r in pr.results if r.violated)
        if total == 0:
            return 1.0
        return (total - violated) / total

    @property
    def total_agents_dispatched(self) -> int:
        return sum(len(pr.results) for pr in self.phase_results)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Monitored parallel orchestrator for session-typed agent sprints.

    Dispatches agents in phases, using AgentMonitor for runtime protocol
    conformance checking. Independent agents run in parallel; dependent
    agents run sequentially between phases.

    Phases per step:
      1. Research   (sequential — needs to complete first)
      2. Implement || Write   (parallel, independent)
      3. Test || Prove        (parallel, independent)
      4. Evaluate   (sequential — needs all above)
      5. Review     (if not accepted — loops)
    """

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.config = config or OrchestratorConfig()
        self.monitor = AgentMonitor()
        self.sprint_count: int = 0
        self._handlers: dict[str, Callable[..., Any]] = {}

    # -- Handler registration -------------------------------------------------

    def register_handler(self, agent_type: str, handler: Callable[..., Any]) -> None:
        """Register a handler function for an agent type.

        The handler is called as handler(agent_id, step_number, task_description)
        and should return a result dict with at least a 'transitions' key
        listing the labels the agent traverses.
        """
        self._handlers[agent_type] = handler

    # -- Agent lifecycle (monitored) ------------------------------------------

    def _start_monitored_agent(
        self, agent_type: str, step_number: str, task: str
    ) -> str:
        """Start an agent with monitoring. Returns the agent_id."""
        agent_id = f"{agent_type.lower()}-{step_number}-{self.sprint_count}"
        if not self.config.monitor_enabled:
            return agent_id
        self.monitor.start_agent(agent_type, agent_id, step_number)
        return agent_id

    def _execute_agent(
        self,
        agent_type: str,
        step_number: str,
        task: str,
        handler: Callable[..., Any] | None = None,
    ) -> AgentResult:
        """Execute a single agent with full monitoring."""
        start = time.time()
        agent_id = self._start_monitored_agent(agent_type, step_number, task)

        # Resolve handler
        fn = handler or self._handlers.get(agent_type)
        if fn is None:
            # No handler — return a stub result (agent not wired up)
            elapsed = (time.time() - start) * 1000
            return AgentResult(
                agent_id=agent_id,
                agent_type=agent_type,
                step_number=step_number,
                success=False,
                output={"error": f"No handler registered for {agent_type}"},
                elapsed_ms=elapsed,
            )

        try:
            result = fn(agent_id, step_number, task)
        except Exception as exc:
            if self.config.monitor_enabled:
                self.monitor.fail_agent(agent_id, str(exc))
            elapsed = (time.time() - start) * 1000
            return AgentResult(
                agent_id=agent_id,
                agent_type=agent_type,
                step_number=step_number,
                success=False,
                output={"error": str(exc)},
                elapsed_ms=elapsed,
            )

        # Process transitions from the handler result
        violated = False
        violation_detail = None
        if self.config.monitor_enabled and isinstance(result, dict):
            transitions = result.get("transitions", [])
            for label in transitions:
                ok = self.monitor.transition(agent_id, label)
                if not ok:
                    violated = True
                    inst = self.monitor.instances.get(agent_id)
                    violation_detail = inst.violation if inst else f"Bad transition: {label}"
                    break

        elapsed = (time.time() - start) * 1000
        success = not violated if self.config.reject_violations else True

        return AgentResult(
            agent_id=agent_id,
            agent_type=agent_type,
            step_number=step_number,
            success=success,
            violated=violated,
            violation_detail=violation_detail,
            output=result,
            elapsed_ms=elapsed,
        )

    def _complete_agent(self, agent_id: str, result: Any) -> bool:
        """Record agent completion. Returns False if violated."""
        if not self.config.monitor_enabled:
            return True
        inst = self.monitor.instances.get(agent_id)
        if inst is None:
            return False
        return inst.status != "violated"

    # -- Parallel phase -------------------------------------------------------

    def _parallel_phase(
        self,
        phase_name: str,
        agents: list[AgentTask],
    ) -> ParallelPhaseResult:
        """Run multiple agents in parallel, all monitored.

        Uses ThreadPoolExecutor bounded by config.max_parallel.
        """
        phase = ParallelPhaseResult(
            phase_name=phase_name,
            agents=[a.agent_type for a in agents],
        )

        if not agents:
            return phase

        max_workers = min(self.config.max_parallel, len(agents))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures: dict[Future[AgentResult], AgentTask] = {}
            for task in agents:
                fut = pool.submit(
                    self._execute_agent,
                    task.agent_type,
                    task.step_number,
                    task.task_description,
                    task.handler,
                )
                futures[fut] = task

            for fut in as_completed(futures):
                agent_result = fut.result()
                phase.results.append(agent_result)

        return phase

    def _sequential_phase(
        self,
        phase_name: str,
        task: AgentTask,
    ) -> ParallelPhaseResult:
        """Run a single agent sequentially (wrapper for consistency)."""
        result = self._execute_agent(
            task.agent_type,
            task.step_number,
            task.task_description,
            task.handler,
        )
        return ParallelPhaseResult(
            phase_name=phase_name,
            agents=[task.agent_type],
            results=[result],
        )

    # -- Sprint execution -----------------------------------------------------

    def run_sprint(
        self,
        steps: list[str] | None = None,
        scan_result: dict[str, Any] | None = None,
    ) -> SprintResult:
        """Run one sprint with monitored parallel agents.

        Args:
            steps: List of step numbers to process (e.g. ["23", "24"]).
            scan_result: Optional pre-computed scan/evaluation result.

        Returns:
            SprintResult with all phase outcomes.
        """
        self.sprint_count += 1
        start = time.time()
        sprint = SprintResult(sprint_number=self.sprint_count)

        if steps is None:
            steps = []

        sprint.steps_processed = list(steps)

        for step in steps:
            step_phases = self._run_step(step)
            sprint.phase_results.extend(step_phases)

            # Collect violations
            for pr in step_phases:
                for r in pr.results:
                    if r.violated:
                        sprint.violations.append(
                            f"{r.agent_type}@step{step}: {r.violation_detail}"
                        )
                    if r.success and not r.violated:
                        # Count created/fixed based on phase
                        if pr.phase_name.startswith("evaluate"):
                            output = r.output if isinstance(r.output, dict) else {}
                            if output.get("accepted"):
                                sprint.steps_created += 1
                            else:
                                sprint.steps_fixed += 1

        sprint.elapsed_ms = (time.time() - start) * 1000
        return sprint

    def _run_step(self, step: str) -> list[ParallelPhaseResult]:
        """Run all phases for a single step."""
        phases: list[ParallelPhaseResult] = []

        # Phase 1: Research (sequential)
        p1 = self._sequential_phase(
            f"research-{step}",
            AgentTask("Researcher", step, "Investigate step requirements"),
        )
        phases.append(p1)
        if not p1.all_completed:
            return phases

        # Phase 2: Implement || Write (parallel)
        p2 = self._parallel_phase(
            f"implement-write-{step}",
            [
                AgentTask("Implementer", step, "Implement module"),
                AgentTask("Writer", step, "Write educational paper"),
            ],
        )
        phases.append(p2)
        if not p2.all_completed and self.config.reject_violations:
            return phases

        # Phase 3: Test || Prove (parallel)
        p3 = self._parallel_phase(
            f"test-prove-{step}",
            [
                AgentTask("Tester", step, "Write and run tests"),
                AgentTask("Prover", step, "Write companion proofs"),
            ],
        )
        phases.append(p3)
        if not p3.all_completed and self.config.reject_violations:
            return phases

        # Phase 4: Evaluate (sequential)
        p4 = self._sequential_phase(
            f"evaluate-{step}",
            AgentTask("Evaluator", step, "Grade deliverables"),
        )
        phases.append(p4)

        # Phase 5: Review if needed (sequential)
        if p4.results and isinstance(p4.results[0].output, dict):
            if p4.results[0].output.get("needs_fixes"):
                p5 = self._sequential_phase(
                    f"review-{step}",
                    AgentTask("Reviewer", step, "Review and suggest fixes"),
                )
                phases.append(p5)

        return phases

    # -- Dashboard ------------------------------------------------------------

    def dashboard(self) -> str:
        """Show monitored agent status."""
        if not self.config.monitor_enabled:
            return "(monitoring disabled)"
        return self.monitor.dashboard()

    def conformance_report(self) -> dict[str, Any]:
        """Get conformance report from the monitor."""
        if not self.config.monitor_enabled:
            return {"monitoring": False}
        return self.monitor.conformance_report()
