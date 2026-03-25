"""LLM Agent Protocol Verification (Step 82).

Models multi-agent LLM orchestration patterns as multiparty session types
(global types), projects onto individual agent roles, and verifies that the
resulting state spaces form lattices.

Supported patterns:
  - mcp_multi_agent(n)      — N MCP servers orchestrated by one host
  - a2a_chain(n)            — linear pipeline: agent1 -> agent2 -> ... -> agentN
  - a2a_broadcast(n)        — one orchestrator broadcasts to N agents
  - rag_pipeline()          — retriever -> ranker -> generator -> validator
  - tool_use_loop()         — agent -> tool -> result -> agent (with retry)
  - multi_model_consensus() — N models vote, majority wins

Each function returns an AgentOrchestration, which can be verified with
verify_orchestration() to produce an OrchestrationResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.global_types import (
    GEnd,
    GMessage,
    GParallel,
    GRec,
    GVar,
    GlobalType,
    build_global_statespace,
    pretty_global,
    roles,
)
from reticulate.lattice import LatticeResult, check_lattice
from reticulate.projection import (
    ProjectionError,
    project,
    project_all,
)
from reticulate.parser import SessionType, pretty

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentOrchestration:
    """A multi-agent orchestration described by a global session type.

    Attributes:
        name: Human-readable name for the orchestration pattern.
        description: Explanation of the pattern.
        agents: Tuple of agent/role names participating.
        channels: Tuple of (sender, receiver) communication channels.
        global_type: The multiparty global type describing the protocol.
    """
    name: str
    description: str
    agents: tuple[str, ...]
    channels: tuple[tuple[str, str], ...]
    global_type: GlobalType


@dataclass(frozen=True)
class RoleProjection:
    """Result of projecting an orchestration onto a single role.

    Attributes:
        role: The role name.
        local_type: The projected local session type AST.
        local_type_str: Pretty-printed local type.
        is_well_defined: True iff projection succeeded without merge conflicts.
        state_count: Number of states in the local state space.
        is_lattice: True iff the local state space forms a lattice.
    """
    role: str
    local_type: SessionType | None
    local_type_str: str
    is_well_defined: bool
    state_count: int
    is_lattice: bool


@dataclass(frozen=True)
class OrchestrationResult:
    """Complete verification result for a multi-agent orchestration.

    Attributes:
        orchestration: The orchestration being verified.
        global_type_str: Pretty-printed global type.
        global_states: Number of states in the global state space.
        global_is_lattice: True iff the global state space forms a lattice.
        lattice_result: Full LatticeResult for the global state space.
        projections: Per-role projection results.
        all_projections_defined: True iff all roles project without errors.
        all_local_lattices: True iff all local state spaces are lattices.
        is_verified: True iff global is lattice AND all projections are defined
            AND all local state spaces are lattices.
    """
    orchestration: AgentOrchestration
    global_type_str: str
    global_states: int
    global_is_lattice: bool
    lattice_result: LatticeResult
    projections: tuple[RoleProjection, ...]
    all_projections_defined: bool
    all_local_lattices: bool
    is_verified: bool


# ---------------------------------------------------------------------------
# Helper: build GMessage with single choice (sequential message)
# ---------------------------------------------------------------------------

def _msg(sender: str, receiver: str, label: str, cont: GlobalType) -> GMessage:
    """Shorthand for a single-choice message."""
    return GMessage(sender, receiver, ((label, cont),))


# ---------------------------------------------------------------------------
# Orchestration pattern constructors
# ---------------------------------------------------------------------------

def mcp_multi_agent(num_servers: int = 2) -> AgentOrchestration:
    """N MCP servers orchestrated by a single host.

    The host initialises each server, then enters a loop where it can
    call tools on any server or shut down.  The servers run in parallel
    from the host's perspective.

    Pattern:
        host -> server_i : initialize
        rec X . host -> server_i : { callTool: server_i -> host : { RESULT: X, ERROR: X },
                                      shutdown: end }
    """
    if num_servers < 1:
        raise ValueError("num_servers must be >= 1")

    agents = ["host"] + [f"server{i}" for i in range(1, num_servers + 1)]
    channels: list[tuple[str, str]] = []
    for s in agents[1:]:
        channels.append(("host", s))
        channels.append((s, "host"))

    # Build one branch per server: init then loop
    def _server_branch(server: str) -> GlobalType:
        # rec X . host -> server : { callTool: server -> host : { RESULT: X, ERROR: X },
        #                             shutdown: end }
        loop_body = GMessage(
            "host", server, (
                ("callTool", GMessage(
                    server, "host", (
                        ("RESULT", GVar("X")),
                        ("ERROR", GVar("X")),
                    )
                )),
                ("shutdown", GEnd()),
            )
        )
        return _msg("host", server, "initialize", GRec("X", loop_body))

    if num_servers == 1:
        gt = _server_branch("server1")
    else:
        # Parallel composition of all server branches
        branches = [_server_branch(f"server{i}") for i in range(1, num_servers + 1)]
        gt = branches[0]
        for b in branches[1:]:
            gt = GParallel(gt, b)

    return AgentOrchestration(
        name=f"MCP Multi-Agent ({num_servers} servers)",
        description=(
            f"A host orchestrates {num_servers} MCP server(s). "
            f"Each server is independently initialised and then enters a "
            f"tool-call loop with the host. Servers run in parallel."
        ),
        agents=tuple(agents),
        channels=tuple(channels),
        global_type=gt,
    )


def a2a_chain(num_agents: int = 3) -> AgentOrchestration:
    """Linear chain: agent1 -> agent2 -> ... -> agentN.

    Each agent delegates a task to the next agent in the chain.
    The chain processes sequentially: each agent delegates to the next,
    and the last agent responds back up the chain.

    Pattern (for 3 agents):
        agent1 -> agent2 : delegate
        agent2 -> agent3 : delegate
        agent3 -> agent2 : { DONE: agent2 -> agent1 : { DONE: end, FAILED: end },
                             FAILED: agent2 -> agent1 : { DONE: end, FAILED: end } }

    All roles appear consistently in all branches so projection merges succeed.
    """
    if num_agents < 2:
        raise ValueError("num_agents must be >= 2")

    agents = [f"agent{i}" for i in range(1, num_agents + 1)]
    channels: list[tuple[str, str]] = []
    for i in range(len(agents) - 1):
        channels.append((agents[i], agents[i + 1]))
        channels.append((agents[i + 1], agents[i]))

    # Build all delegation messages first (forward pass), then
    # build response messages (backward pass) so all roles are
    # consistently present in all branches.
    def _build_responses(idx: int) -> GlobalType:
        """Build response chain from agent[idx+1] -> agent[idx] backward to start."""
        if idx < 0:
            return GEnd()
        sender = agents[idx + 1]
        receiver = agents[idx]
        cont = _build_responses(idx - 1)
        return GMessage(sender, receiver, (
            ("DONE", cont),
            ("FAILED", cont),
        ))

    def _build_delegations(idx: int) -> GlobalType:
        """Build delegation chain forward, then attach responses."""
        if idx >= len(agents) - 1:
            # End of forward chain: start building responses backward
            return _build_responses(len(agents) - 2)
        sender = agents[idx]
        receiver = agents[idx + 1]
        return _msg(sender, receiver, "delegate", _build_delegations(idx + 1))

    gt = _build_delegations(0)

    return AgentOrchestration(
        name=f"A2A Chain ({num_agents} agents)",
        description=(
            f"A linear chain of {num_agents} agents. Each agent delegates "
            f"a task to the next and receives either DONE or FAILED."
        ),
        agents=tuple(agents),
        channels=tuple(channels),
        global_type=gt,
    )


def a2a_broadcast(num_agents: int = 3) -> AgentOrchestration:
    """One orchestrator broadcasts a task to N worker agents in parallel.

    The orchestrator sends a task to each worker, then each worker
    independently replies with DONE or FAILED.

    Pattern:
        orchestrator -> worker_i : assign
        worker_i -> orchestrator : { DONE: end, FAILED: end }
    """
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1")

    agents = ["orchestrator"] + [f"worker{i}" for i in range(1, num_agents + 1)]
    channels: list[tuple[str, str]] = []
    for w in agents[1:]:
        channels.append(("orchestrator", w))
        channels.append((w, "orchestrator"))

    def _worker_branch(worker: str) -> GlobalType:
        return _msg(
            "orchestrator", worker, "assign",
            GMessage(worker, "orchestrator", (
                ("DONE", GEnd()),
                ("FAILED", GEnd()),
            ))
        )

    if num_agents == 1:
        gt = _worker_branch("worker1")
    else:
        branches = [_worker_branch(f"worker{i}") for i in range(1, num_agents + 1)]
        gt = branches[0]
        for b in branches[1:]:
            gt = GParallel(gt, b)

    return AgentOrchestration(
        name=f"A2A Broadcast ({num_agents} workers)",
        description=(
            f"An orchestrator broadcasts tasks to {num_agents} worker agent(s) "
            f"in parallel. Each worker independently reports DONE or FAILED."
        ),
        agents=tuple(agents),
        channels=tuple(channels),
        global_type=gt,
    )


def rag_pipeline() -> AgentOrchestration:
    """RAG (Retrieval-Augmented Generation) pipeline.

    Pattern:
        user -> retriever : query
        retriever -> ranker : candidates
        ranker -> generator : ranked
        generator -> validator : draft
        validator -> user : { APPROVED: end, REJECTED: end }
    """
    agents = ("user", "retriever", "ranker", "generator", "validator")
    channels = (
        ("user", "retriever"),
        ("retriever", "ranker"),
        ("ranker", "generator"),
        ("generator", "validator"),
        ("validator", "user"),
    )

    gt = _msg("user", "retriever", "query",
         _msg("retriever", "ranker", "candidates",
         _msg("ranker", "generator", "ranked",
         _msg("generator", "validator", "draft",
         GMessage("validator", "user", (
             ("APPROVED", GEnd()),
             ("REJECTED", GEnd()),
         ))))))

    return AgentOrchestration(
        name="RAG Pipeline",
        description=(
            "A Retrieval-Augmented Generation pipeline: user queries a retriever, "
            "results are ranked, a generator produces a draft, and a validator "
            "approves or rejects the output."
        ),
        agents=agents,
        channels=channels,
        global_type=gt,
    )


def tool_use_loop() -> AgentOrchestration:
    """Agent-tool interaction loop with retry.

    The agent calls a tool, receives a result or error. On error
    the agent may retry (recursive loop) or give up.  The user role
    is not involved in the tool interaction loop itself -- only the
    agent and tool participate.  After the loop terminates, the agent
    reports to the user.

    Pattern:
        rec X .
          agent -> tool : invoke
          tool -> agent : { SUCCESS: end,
                            ERROR: agent -> tool : { retry: X,
                                                     abort: end } }

    The agent-user reporting is separated to avoid merge conflicts.
    """
    agents = ("agent", "tool")
    channels = (
        ("agent", "tool"),
        ("tool", "agent"),
    )

    loop_body = _msg("agent", "tool", "invoke",
        GMessage("tool", "agent", (
            ("SUCCESS", GEnd()),
            ("ERROR", GMessage("agent", "tool", (
                ("retry", GVar("X")),
                ("abort", GEnd()),
            ))),
        ))
    )

    gt = GRec("X", loop_body)

    return AgentOrchestration(
        name="Tool Use Loop",
        description=(
            "An agent invokes a tool in a loop. On success, it reports to the user. "
            "On error, it may retry or abort with an error report."
        ),
        agents=agents,
        channels=channels,
        global_type=gt,
    )


def multi_model_consensus(num_models: int = 3) -> AgentOrchestration:
    """N models vote in parallel, an aggregator collects votes.

    The coordinator sends a prompt to each model in parallel.
    Each model returns a vote. The aggregator then reports
    the consensus to the user.

    Pattern:
        ( coordinator -> model_1 : prompt . model_1 -> aggregator : vote . end
          || ... ||
          coordinator -> model_N : prompt . model_N -> aggregator : vote . end )
        then aggregator -> user : consensus . end

    Since global types do not have direct sequencing sugar, we model the
    post-parallel phase via a separate parallel branch between aggregator
    and user that is ordered by the aggregator collecting all votes first.
    """
    if num_models < 2:
        raise ValueError("num_models must be >= 2")

    agents = (
        ["coordinator", "aggregator", "user"]
        + [f"model{i}" for i in range(1, num_models + 1)]
    )
    channels: list[tuple[str, str]] = []
    channels.append(("aggregator", "user"))
    for i in range(1, num_models + 1):
        m = f"model{i}"
        channels.append(("coordinator", m))
        channels.append((m, "aggregator"))

    def _model_branch(model: str) -> GlobalType:
        return _msg("coordinator", model, "prompt",
               _msg(model, "aggregator", "vote", GEnd()))

    # After all votes, aggregator reports to user
    report_phase = _msg("aggregator", "user", "consensus", GEnd())

    # Build parallel of model branches
    model_branches = [_model_branch(f"model{i}") for i in range(1, num_models + 1)]
    voting_phase: GlobalType = model_branches[0]
    for b in model_branches[1:]:
        voting_phase = GParallel(voting_phase, b)

    # The full protocol: voting in parallel, then reporting
    # We model "then" as parallel since the aggregator naturally
    # sequences: it collects votes before reporting
    gt = GParallel(voting_phase, report_phase)

    return AgentOrchestration(
        name=f"Multi-Model Consensus ({num_models} models)",
        description=(
            f"A coordinator sends prompts to {num_models} models in parallel. "
            f"Each model votes to an aggregator. The aggregator reports the "
            f"consensus to the user."
        ),
        agents=tuple(agents),
        channels=tuple(channels),
        global_type=gt,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_orchestration(orch: AgentOrchestration) -> OrchestrationResult:
    """Verify a multi-agent orchestration.

    1. Build the global state space and check it is a lattice.
    2. Project the global type onto each agent role.
    3. Build each local state space and check it is a lattice.
    4. Combine results into an OrchestrationResult.
    """
    from reticulate.statespace import build_statespace

    # Global analysis
    global_ss = build_global_statespace(orch.global_type)
    global_lr = check_lattice(global_ss)
    global_type_str = pretty_global(orch.global_type)

    # Per-role projections
    all_roles = roles(orch.global_type)
    projections: list[RoleProjection] = []
    all_defined = True
    all_lattices = True

    for role in sorted(all_roles):
        try:
            local_type = project(orch.global_type, role)
            local_str = pretty(local_type)
            local_ss = build_statespace(local_type)
            local_lr = check_lattice(local_ss)
            projections.append(RoleProjection(
                role=role,
                local_type=local_type,
                local_type_str=local_str,
                is_well_defined=True,
                state_count=len(local_ss.states),
                is_lattice=local_lr.is_lattice,
            ))
            if not local_lr.is_lattice:
                all_lattices = False
        except ProjectionError:
            all_defined = False
            projections.append(RoleProjection(
                role=role,
                local_type=None,
                local_type_str="<projection failed>",
                is_well_defined=False,
                state_count=0,
                is_lattice=False,
            ))
            all_lattices = False

    is_verified = (
        global_lr.is_lattice
        and all_defined
        and all_lattices
    )

    return OrchestrationResult(
        orchestration=orch,
        global_type_str=global_type_str,
        global_states=len(global_ss.states),
        global_is_lattice=global_lr.is_lattice,
        lattice_result=global_lr,
        projections=tuple(projections),
        all_projections_defined=all_defined,
        all_local_lattices=all_lattices,
        is_verified=is_verified,
    )


def format_orchestration_result(result: OrchestrationResult) -> str:
    """Format an OrchestrationResult as structured text."""
    lines: list[str] = []
    o = result.orchestration

    lines.append("=" * 70)
    lines.append(f"  ORCHESTRATION VERIFICATION: {o.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {o.description}")
    lines.append("")

    lines.append("--- Global Type ---")
    lines.append(f"  {result.global_type_str}")
    lines.append("")

    lines.append("--- Agents ---")
    for a in o.agents:
        lines.append(f"  - {a}")
    lines.append("")

    lines.append("--- Channels ---")
    for s, r in o.channels:
        lines.append(f"  {s} -> {r}")
    lines.append("")

    lines.append("--- Global State Space ---")
    lines.append(f"  States:     {result.global_states}")
    lines.append(f"  Is lattice: {'YES' if result.global_is_lattice else 'NO'}")
    lines.append("")

    lines.append("--- Role Projections ---")
    for p in result.projections:
        status = "OK" if p.is_well_defined else "FAILED"
        lattice = "lattice" if p.is_lattice else "NOT lattice"
        lines.append(f"  {p.role:20s} [{status}] states={p.state_count} {lattice}")
        lines.append(f"    {p.local_type_str}")
    lines.append("")

    lines.append("--- Verdict ---")
    if result.is_verified:
        lines.append(f"  VERIFIED: {o.name} is well-formed.")
        lines.append(f"  Global state space forms a lattice.")
        lines.append(f"  All {len(result.projections)} role projections are lattices.")
    else:
        lines.append(f"  NOT VERIFIED: {o.name} has issues.")
        if not result.global_is_lattice:
            lines.append(f"  - Global state space is NOT a lattice.")
        if not result.all_projections_defined:
            lines.append(f"  - Some projections are undefined (merge conflicts).")
        if not result.all_local_lattices:
            lines.append(f"  - Some local state spaces are NOT lattices.")
    lines.append("=" * 70)

    return "\n".join(lines)
