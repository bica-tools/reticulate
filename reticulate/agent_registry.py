"""Agent registry: session types for each agent role + runtime tracking.

Each agent type has a session type it MUST follow. The registry defines
these types, and the AgentMonitor tracks actual operations against them.

The 8 agent types:
  MCP (stateless): Researcher, Evaluator, Supervisor
  A2A (autonomous): Implementer, Tester, Writer, Prover, Reviewer

Usage:
    from reticulate.agent_registry import AGENTS, AgentMonitor
    monitor = AgentMonitor()
    monitor.start_agent("Writer", agent_id="writer-step23")
    monitor.transition("writer-step23", "writePaper")
    monitor.transition("writer-step23", "paperReady")  # → complete
    print(monitor.dashboard())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from reticulate.parser import parse, pretty, SessionType
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Agent type definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentType:
    """Definition of an agent type with its session type."""
    name: str
    protocol: str        # "MCP" | "A2A"
    session_type: str    # session type string
    description: str
    transport: str       # "stdio" | "HTTP" | "Agent()"


# The 8 agent types with their session types
AGENTS: dict[str, AgentType] = {
    "Researcher": AgentType(
        name="Researcher",
        protocol="MCP",
        session_type="&{investigate: +{report: end}}",
        description="Reads code and theory, maps the landscape",
        transport="stdio",
    ),
    "Implementer": AgentType(
        name="Implementer",
        protocol="A2A",
        session_type="rec X . &{implement: +{moduleReady: end, error: +{retry: X, abort: end}}}",
        description="Writes one Python module with clean API",
        transport="Agent()",
    ),
    "Tester": AgentType(
        name="Tester",
        protocol="A2A",
        session_type="rec X . &{writeTests: +{testsPass: end, testsFail: +{fix: X, abort: end}}}",
        description="Writes test suite, runs tests, ensures coverage",
        transport="Agent()",
    ),
    "Writer": AgentType(
        name="Writer",
        protocol="A2A",
        session_type="&{writePaper: +{paperReady: end}}",
        description="Writes 5000+ word educational paper",
        transport="Agent()",
    ),
    "Prover": AgentType(
        name="Prover",
        protocol="A2A",
        session_type="&{writeProofs: +{proofsReady: end}}",
        description="Writes companion proofs with formal theorems",
        transport="Agent()",
    ),
    "Evaluator": AgentType(
        name="Evaluator",
        protocol="MCP",
        session_type="&{evaluate: +{accepted: end, needsFixes: +{listFixes: end}}}",
        description="Grades deliverables, decides accept/reject",
        transport="stdio",
    ),
    "Reviewer": AgentType(
        name="Reviewer",
        protocol="A2A",
        session_type="&{review: +{fixes: end}}",
        description="Reviews code quality, suggests improvements",
        transport="Agent()",
    ),
    "Supervisor": AgentType(
        name="Supervisor",
        protocol="MCP",
        session_type="&{scan: +{proposals: end}}",
        description="Scans programme, proposes next steps",
        transport="stdio",
    ),
    # --- Missing agents (identified 2026-03-23) ---
    "Formalizer": AgentType(
        name="Formalizer",
        protocol="A2A",
        session_type="&{formalize: +{definitions: &{typingRules: +{soundness: end}}}}",
        description="Writes formal syntax, semantics, typing rules BEFORE implementation",
        transport="Agent()",
    ),
    "Benchmarker": AgentType(
        name="Benchmarker",
        protocol="A2A",
        session_type="rec X . &{runBenchmarks: +{allPass: +{report: end}, someFailures: +{analyze: X, accept: end}}}",
        description="Runs cross-cutting empirical analysis across ALL protocols",
        transport="Agent()",
    ),
    "Connector": AgentType(
        name="Connector",
        protocol="A2A",
        session_type="&{findLinks: +{crossRefs: &{proposeIntersections: +{linked: end}}}}",
        description="Links steps together, adds cross-references, maintains programme coherence",
        transport="Agent()",
    ),
    "TechWriter": AgentType(
        name="TechWriter",
        protocol="A2A",
        session_type="&{document: +{readme: end, apiDocs: end, changelog: end}}",
        description="Writes README, API docs, CLAUDE.md, changelogs — not academic papers",
        transport="Agent()",
    ),
    "FrontendDev": AgentType(
        name="FrontendDev",
        protocol="A2A",
        session_type="rec X . &{buildFeature: +{componentReady: &{integrate: +{merged: end, conflicts: X}}, designIssue: +{redesign: X, skip: end}}}",
        description="Angular components, Material UI, routing, API integration for bica-tools.org",
        transport="Agent()",
    ),
    "BackendDev": AgentType(
        name="BackendDev",
        protocol="A2A",
        session_type="rec X . &{buildEndpoint: +{endpointReady: &{test: +{pass: end, fail: X}}, designIssue: +{redesign: X, skip: end}}}",
        description="Spring Boot controllers, services, DTOs, database for bica-tools.org",
        transport="Agent()",
    ),
    "CICDEngineer": AgentType(
        name="CICDEngineer",
        protocol="A2A",
        session_type="rec X . &{configurePipeline: +{pipelineReady: &{runPipeline: +{green: end, red: +{fix: X, disable: end}}}}}",
        description="GitHub Actions, CI gates, deployment pipelines, Docker, packaging",
        transport="Agent()",
    ),
    "Deployer": AgentType(
        name="Deployer",
        protocol="A2A",
        session_type="rec X . &{build: +{success: &{deploy: +{healthy: end, rollback: X}}, failure: +{fix: X, abort: end}}}",
        description="Deploys to bica-tools.org, manages infrastructure, monitors uptime",
        transport="Agent()",
    ),
    "Publisher": AgentType(
        name="Publisher",
        protocol="A2A",
        session_type="&{formatForVenue: +{ready: &{submit: +{accepted: end, revise: &{applyRevisions: +{resubmit: end}}}}}}",
        description="Formats papers for venue submission, manages correspondence",
        transport="Agent()",
    ),
    "PublicationPlanner": AgentType(
        name="PublicationPlanner",
        protocol="MCP",
        session_type="&{scanResults: +{planGenerated: &{matchVenues: +{planReady: end}}}}",
        description="Maintains living publication list, maps results to venues, tracks deadlines",
        transport="stdio",
    ),
    "CopyEditor": AgentType(
        name="CopyEditor",
        protocol="A2A",
        session_type="rec X . &{polish: +{flawless: end, issues: &{fix: X}}}",
        description="Camera-ready polish: clarity, grammar, formatting, diagrams, references. Runs AFTER Strong Accept.",
        transport="Agent()",
    ),
    "SubmissionStrategist": AgentType(
        name="SubmissionStrategist",
        protocol="MCP",
        session_type="&{checkPlan: +{consistent: end, issue: +{citationCycle: end, dualSubmission: end, missingDependency: end, orderingProblem: end}}}",
        description="Checks citation dependency ordering, venue overlap, dual submission rules, self-containedness",
        transport="stdio",
    ),
    "Gatekeeper": AgentType(
        name="Gatekeeper",
        protocol="MCP",
        session_type="&{classify: +{public: end, embargoed: &{checkTimestamp: +{timestamped: end, needsTimestamp: end}}, private: end}}",
        description="Decides public/embargoed/private per publication-access-policy.md",
        transport="stdio",
    ),
    # --- Editorial team ---
    "Editor": AgentType(
        name="Editor",
        protocol="MCP",
        session_type="&{selectTopic: +{approved: &{assignWriter: +{scheduled: end}}, deferred: end, rejected: end}}",
        description="Editorial strategy, topic selection, calendar, tone/voice per editorial-concept.md",
        transport="stdio",
    ),
    "BlogWriter": AgentType(
        name="BlogWriter",
        protocol="A2A",
        session_type="rec X . &{writeDraft: +{draftReady: &{editorialReview: +{approved: end, revisions: &{revise: X}}}}}",
        description="Writes accessible blog posts from research results for bica-tools.org/blog",
        transport="Agent()",
    ),
    # --- Tutorials ---
    "Tutor": AgentType(
        name="Tutor",
        protocol="A2A",
        session_type="&{designLesson: +{outline: &{buildInteractive: +{tutorialReady: &{testWithUser: +{effective: end, redesign: end}}}}}}",
        description="Creates interactive tutorials for bica-tools.org with step-by-step exercises",
        transport="Agent()",
    ),
    # --- Quality: peer review + fact checking ---
    "PeerReviewer": AgentType(
        name="PeerReviewer",
        protocol="A2A",
        # Intentionally non-distributive: review outcomes reconverge (M₃).
        # This is semantically correct — accept/revise/reject are coupled
        # verdicts, not independent dimensions. A paper's fate depends
        # on the interaction of all criteria, not each in isolation.
        session_type="&{reviewPaper: +{accept: end, minorRevisions: +{listMinor: end}, majorRevisions: +{listMajor: end}, reject: +{listReasons: end}}}",
        description="Simulates venue peer review: contribution clarity, proof rigour, evaluation sufficiency",
        transport="Agent()",
    ),
    "FactChecker": AgentType(
        name="FactChecker",
        protocol="MCP",
        session_type="rec X . &{verifyClaim: +{confirmed: +{nextClaim: X, allChecked: end}, refuted: +{reportError: end}}}",
        description="Verifies numerical claims in papers match actual tool output on current benchmarks",
        transport="stdio",
    ),
    # --- Research: literature monitoring ---
    "LiteratureScout": AgentType(
        name="LiteratureScout",
        protocol="A2A",
        session_type="&{searchLiterature: +{findings: &{assessRelevance: +{relevant: +{updateRelatedWork: end}, notRelevant: end}}}}",
        description="Searches arXiv/DBLP/Semantic Scholar APIs for related work, deduplicates by title similarity, compares against .bib entries to find new papers (literature_scout.py)",
        transport="Agent()",
    ),
    "Librarian": AgentType(
        name="Librarian",
        protocol="A2A",
        session_type="&{downloadReferences: +{downloaded: &{organize: +{organized: end}}, someFailed: +{checkMissing: end}}}",
        description="Downloads and organizes reference PDFs from DOI/Unpaywall/arXiv/Semantic Scholar, manages {citationkey}.pdf naming (librarian.py)",
        transport="Agent()",
    ),
    # --- Write: bibliography management ---
    "Archivist": AgentType(
        name="Archivist",
        protocol="MCP",
        session_type="&{checkBibliography: +{consistent: end, issues: +{listIssues: end}}}",
        description="Maintains shared-new-refs.bib, ensures citation consistency across all papers",
        transport="stdio",
    ),
    # --- Operations: community ---
    "CommunityManager": AgentType(
        name="CommunityManager",
        protocol="A2A",
        session_type="rec X . &{triageIssue: +{respond: +{resolved: +{nextIssue: X, queueEmpty: end}, needsEscalation: +{escalate: X}}, defer: X}}",
        description="Manages GitHub issues, PRs, discussions, contributor onboarding, CONTRIBUTING.md",
        transport="Agent()",
    ),
    "VenueMatcher": AgentType(
        name="VenueMatcher",
        protocol="MCP",
        session_type="&{extractKeywords: +{keywords: &{rankVenues: +{rankings: end}}}}",
        description="Evaluates paper-venue fit using keyword analysis and Jaccard similarity (venue_matcher.py)",
        transport="stdio",
    ),
}

# The orchestrator's full recursive protocol with parallel constructors.
# Independent phases run in parallel:
#   Phase 2: implement || writePaper
#   Phase 3: writeTests || writeProofs
# This models real workflow independence and forces lattice structure.
ORCHESTRATOR_TYPE = (
    "rec S . &{scan: +{proposals: "
    "&{research: +{report: "
    "(&{implement: +{moduleReady: end}} || &{writePaper: +{paperReady: end}}) . "
    "(&{writeTests: +{testsPass: end}} || &{writeProofs: +{proofsReady: end}}) . "
    "&{evaluate: +{accepted: +{nextStep: S, allDone: end}, "
    "needsFixes: &{review: +{fixes: S}}}}}}}}"
)

# Phase classification
RESEARCH_AGENTS = ("Researcher", "Formalizer", "LiteratureScout")
BUILD_AGENTS = ("Implementer", "Tester", "Benchmarker")
WRITE_AGENTS = ("Writer", "Prover", "TechWriter", "Archivist", "Librarian")
QUALITY_AGENTS = ("Evaluator", "Reviewer", "Connector", "PeerReviewer", "FactChecker")
MANAGEMENT_AGENTS = ("Supervisor",)
OPERATIONS_AGENTS = ("FrontendDev", "BackendDev", "CICDEngineer", "Deployer", "Publisher", "CommunityManager")
GOVERNANCE_AGENTS = ("Gatekeeper",)
EDITORIAL_AGENTS = ("Editor", "BlogWriter", "Tutor")


# ---------------------------------------------------------------------------
# Agent instance tracking
# ---------------------------------------------------------------------------

@dataclass
class AgentInstance:
    """A running agent instance."""
    agent_id: str
    agent_type: str
    step_number: str
    state: str             # current state in the session type
    transitions: list[tuple[float, str]]  # (timestamp, label)
    started_at: float
    completed_at: float | None = None
    status: str = "running"   # "running" | "completed" | "failed" | "violated"
    violation: str | None = None

    @property
    def elapsed_ms(self) -> float:
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# State machine tracker per agent type
# ---------------------------------------------------------------------------

def _build_state_machine(session_type_str: str) -> dict[str, dict[str, str]]:
    """Build a simple state machine from a session type for runtime tracking.

    Returns: {state_name: {label: next_state_name}}
    Uses state IDs from the state space as state names.
    """
    ast = parse(session_type_str)
    ss = build_statespace(ast)

    machine: dict[str, dict[str, str]] = {}
    for state in ss.states:
        machine[str(state)] = {}

    for src, label, tgt in ss.transitions:
        machine[str(src)][label] = str(tgt)

    return machine


# Pre-build state machines for all agent types
_STATE_MACHINES: dict[str, dict[str, dict[str, str]]] = {}

def _get_machine(agent_type: str) -> dict[str, dict[str, str]]:
    if agent_type not in _STATE_MACHINES:
        at = AGENTS.get(agent_type)
        if at is None:
            return {}
        _STATE_MACHINES[agent_type] = _build_state_machine(at.session_type)
    return _STATE_MACHINES[agent_type]


def _get_initial_state(agent_type: str) -> str:
    """Get the initial state (top) for an agent type."""
    at = AGENTS.get(agent_type)
    if at is None:
        return "0"
    ast = parse(at.session_type)
    ss = build_statespace(ast)
    return str(ss.top)


# ---------------------------------------------------------------------------
# Agent monitor
# ---------------------------------------------------------------------------

class AgentMonitor:
    """Monitors agent operations against their session types.

    Tracks all agent instances, their state transitions, and detects
    protocol violations in real time.
    """

    def __init__(self) -> None:
        self.instances: dict[str, AgentInstance] = {}
        self.history: list[AgentInstance] = []

    def start_agent(
        self,
        agent_type: str,
        agent_id: str,
        step_number: str = "",
    ) -> AgentInstance:
        """Start tracking a new agent instance."""
        if agent_type not in AGENTS:
            raise ValueError(f"Unknown agent type: {agent_type}. Valid: {list(AGENTS.keys())}")

        initial = _get_initial_state(agent_type)
        instance = AgentInstance(
            agent_id=agent_id,
            agent_type=agent_type,
            step_number=step_number,
            state=initial,
            transitions=[],
            started_at=time.time(),
        )
        self.instances[agent_id] = instance
        return instance

    def transition(self, agent_id: str, label: str) -> bool:
        """Record a transition for an agent. Returns True if valid."""
        inst = self.instances.get(agent_id)
        if inst is None:
            raise ValueError(f"Unknown agent: {agent_id}")

        machine = _get_machine(inst.agent_type)
        current_state = inst.state
        valid_transitions = machine.get(current_state, {})

        if label not in valid_transitions:
            inst.status = "violated"
            inst.violation = (
                f"Invalid transition '{label}' from state {current_state}. "
                f"Valid: {list(valid_transitions.keys())}"
            )
            inst.transitions.append((time.time(), f"VIOLATION:{label}"))
            return False

        next_state = valid_transitions[label]
        inst.state = next_state
        inst.transitions.append((time.time(), label))

        # Check if reached terminal state (no outgoing transitions)
        if not machine.get(next_state, {}):
            inst.status = "completed"
            inst.completed_at = time.time()
            self.history.append(inst)

        return True

    def fail_agent(self, agent_id: str, reason: str = "") -> None:
        """Mark an agent as failed."""
        inst = self.instances.get(agent_id)
        if inst:
            inst.status = "failed"
            inst.completed_at = time.time()
            inst.violation = reason or "Agent failed"
            self.history.append(inst)

    def get_status(self, agent_id: str) -> dict[str, Any]:
        """Get current status of an agent."""
        inst = self.instances.get(agent_id)
        if inst is None:
            return {"error": f"Unknown agent: {agent_id}"}

        machine = _get_machine(inst.agent_type)
        valid = list(machine.get(inst.state, {}).keys())

        return {
            "agent_id": inst.agent_id,
            "agent_type": inst.agent_type,
            "step": inst.step_number,
            "status": inst.status,
            "state": inst.state,
            "valid_transitions": valid,
            "transitions_taken": inst.num_transitions,
            "elapsed_ms": round(inst.elapsed_ms, 1),
            "violation": inst.violation,
        }

    @property
    def active_count(self) -> int:
        return sum(1 for i in self.instances.values() if i.status == "running")

    @property
    def completed_count(self) -> int:
        return sum(1 for i in self.instances.values() if i.status == "completed")

    @property
    def violated_count(self) -> int:
        return sum(1 for i in self.instances.values() if i.status == "violated")

    def dashboard(self) -> str:
        """Human-readable dashboard of all agents."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("  AGENT MONITOR DASHBOARD")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Active: {self.active_count}  "
                     f"Completed: {self.completed_count}  "
                     f"Violated: {self.violated_count}  "
                     f"Total: {len(self.instances)}")
        lines.append("")

        for inst in self.instances.values():
            icon = {
                "running": "RUN", "completed": "OK ",
                "failed": "ERR", "violated": "!!!"
            }.get(inst.status, "???")
            machine = _get_machine(inst.agent_type)
            valid = list(machine.get(inst.state, {}).keys())
            labels = [l for _, l in inst.transitions]
            trace = " → ".join(labels) if labels else "(none)"

            lines.append(f"  [{icon}] {inst.agent_id}")
            lines.append(f"        Type: {inst.agent_type} ({AGENTS[inst.agent_type].protocol})")
            lines.append(f"        Step: {inst.step_number}")
            lines.append(f"        State: {inst.state}, Valid next: {valid}")
            lines.append(f"        Trace: {trace}")
            lines.append(f"        Time: {inst.elapsed_ms:.0f}ms")
            if inst.violation:
                lines.append(f"        VIOLATION: {inst.violation}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def conformance_report(self) -> dict[str, Any]:
        """Summary conformance report for all agents."""
        total = len(self.instances)
        return {
            "total": total,
            "active": self.active_count,
            "completed": self.completed_count,
            "violated": self.violated_count,
            "failed": sum(1 for i in self.instances.values() if i.status == "failed"),
            "conformance_rate": (
                self.completed_count / total if total > 0 else 1.0
            ),
            "agents": [self.get_status(aid) for aid in self.instances],
        }
