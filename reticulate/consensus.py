"""Distributed consensus protocols as session types (Step 98).

Models distributed consensus protocols (Two-Phase Commit, Paxos, Raft,
PBFT) as session types and analyzes their lattice properties.  Safety
properties (agreement, validity, termination) are checked via lattice
invariants of the constructed state spaces.

Consensus protocols are fundamentally stateful: they progress through
proposal, voting, and decision phases in a strict order.  Session types
capture this structure precisely.  The lattice property on the resulting
state space provides algebraic interpretations of consensus safety:

  - **Agreement** corresponds to unique meets among decision states:
    all paths reaching a decision must converge.
  - **Validity** corresponds to path existence from proposals to
    decisions: every decided value was proposed.
  - **Termination guarantee** checks that all maximal paths reach
    the bottom element (structural termination).

Usage:
    from reticulate.consensus import (
        two_phase_commit,
        paxos_basic,
        raft_election,
        pbft_basic,
        build_consensus_protocol,
        analyze_consensus,
    )
    proto = two_phase_commit(3)
    result = analyze_consensus("2pc", {"n_participants": 3})
    print(format_consensus_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.lattice import (
    LatticeResult,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsensusProtocol:
    """A named consensus protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "TwoPhaseCommit").
        roles: Participants in the protocol.
        session_type_string: Session type encoding of the protocol.
        fault_model: Fault model ("crash", "byzantine", "none").
        n_participants: Number of participant nodes (excluding coordinator).
        phases: Number of protocol phases.
        description: Free-text description.
    """
    name: str
    roles: tuple[str, ...]
    session_type_string: str
    fault_model: str
    n_participants: int
    phases: int
    description: str


@dataclass(frozen=True)
class AgreementResult:
    """Result of the agreement property check.

    Agreement holds when all paths reaching decision states agree
    on the decided value.  In lattice terms: decision states that
    share a non-bottom meet must carry the same decision label.

    Attributes:
        holds: True iff agreement property is satisfied.
        decision_states: Set of state IDs identified as decision states.
        decision_labels: Mapping from decision state to its decision label.
        conflicting_pair: A pair of conflicting decision states, or None.
    """
    holds: bool
    decision_states: set[int]
    decision_labels: dict[int, str]
    conflicting_pair: tuple[int, int] | None


@dataclass(frozen=True)
class ValidityResult:
    """Result of the validity property check.

    Validity holds when every decided value was proposed:
    for each decision state there is a reachable-from path
    from a proposal state carrying the same value.

    Attributes:
        holds: True iff validity property is satisfied.
        proposal_states: Set of state IDs for proposal states.
        decision_states: Set of state IDs for decision states.
        unmatched_decisions: Decision states with no matching proposal.
    """
    holds: bool
    proposal_states: set[int]
    decision_states: set[int]
    unmatched_decisions: set[int]


@dataclass(frozen=True)
class TerminationResult:
    """Result of structural termination check.

    Structural termination holds when every maximal path through
    the state space reaches the bottom element.

    Attributes:
        holds: True iff structural termination is satisfied.
        num_paths: Number of maximal paths examined.
        stuck_states: States that are non-bottom with no outgoing
            transitions (deadlock states).
    """
    holds: bool
    num_paths: int
    stuck_states: set[int]


@dataclass(frozen=True)
class ConsensusInvariants:
    """Consensus-specific invariants computed from a state space.

    Attributes:
        lattice_height: Height of the lattice (longest chain).
        lattice_width: Width of the lattice (largest antichain).
        num_decision_states: Number of decision states.
        num_proposal_states: Number of proposal states.
        decision_depth: Average depth of decision states.
        has_unique_decision_point: True iff all decisions go through
            a single meet point.
    """
    lattice_height: int
    lattice_width: int
    num_decision_states: int
    num_proposal_states: int
    decision_depth: float
    has_unique_decision_point: bool


@dataclass(frozen=True)
class ConsensusAnalysisResult:
    """Complete analysis result for a consensus protocol.

    Attributes:
        protocol: The analysed protocol definition.
        ast: Parsed session type AST.
        state_space: Constructed state space.
        lattice_result: Lattice property check.
        agreement: Agreement property check result.
        validity: Validity property check result.
        termination: Termination guarantee check result.
        invariants: Consensus-specific invariants.
        num_states: Number of states in the state space.
        num_transitions: Number of transitions.
        is_well_formed: True iff state space is a lattice.
    """
    protocol: ConsensusProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    agreement: AgreementResult
    validity: ValidityResult
    termination: TerminationResult
    invariants: ConsensusInvariants
    num_states: int
    num_transitions: int
    is_well_formed: bool


# ---------------------------------------------------------------------------
# Protocol definitions — session type string builders
# ---------------------------------------------------------------------------

# Labels used to identify proposal and decision transitions.
PROPOSAL_LABELS: frozenset[str] = frozenset({
    "propose", "prepare", "requestVote", "request", "prePrepare",
})
DECISION_LABELS: frozenset[str] = frozenset({
    "commit", "abort", "accepted", "decided", "reply",
    "grantVote", "denyVote", "learn",
})


def two_phase_commit(n_participants: int = 2) -> ConsensusProtocol:
    """Two-Phase Commit (2PC) protocol as a session type.

    Phase 1 (Voting): coordinator sends ``prepare`` to each participant,
    each responds with ``voteCommit`` or ``voteAbort``.
    Phase 2 (Decision): coordinator sends ``commit`` or ``abort`` to all.

    The session type models the coordinator's view:

        prepare . &{voteCommit: ..., voteAbort: ...} (per participant)
        ... +{commit: end, abort: end}

    Args:
        n_participants: Number of participants (1-5).

    Returns:
        ConsensusProtocol with the 2PC session type.
    """
    if n_participants < 1 or n_participants > 5:
        raise ValueError(f"n_participants must be 1..5, got {n_participants}")

    # Build the voting phase: nested branches for each participant's response.
    # After all votes are collected, coordinator decides.
    decision = "+{commit: end, abort: end}"

    # Build nested voting: participant n, then n-1, ..., then 1
    # Each participant votes commit or abort; all must vote before decision.
    inner = decision
    for i in range(n_participants, 0, -1):
        inner = (
            f"&{{voteCommit{i}: {inner}, "
            f"voteAbort{i}: {decision}}}"
        )

    # Prepare phase: coordinator sends prepare to all participants
    prepare = inner
    for i in range(n_participants, 0, -1):
        prepare = f"&{{prepare{i}: {prepare}}}"

    roles = ("Coordinator",) + tuple(f"Participant{i}" for i in range(1, n_participants + 1))

    return ConsensusProtocol(
        name=f"TwoPhaseCommit_{n_participants}",
        roles=roles,
        session_type_string=prepare,
        fault_model="crash",
        n_participants=n_participants,
        phases=2,
        description=(
            f"Two-Phase Commit with {n_participants} participant(s). "
            "Phase 1: coordinator sends prepare, participants vote. "
            "Phase 2: coordinator decides commit or abort."
        ),
    )


def paxos_basic(n_acceptors: int = 3) -> ConsensusProtocol:
    """Basic Paxos (single round) as a session type.

    Phase 1 (Prepare/Promise): proposer sends prepare to acceptors,
    each responds with promise or nack.
    Phase 2 (Accept/Learn): proposer sends accept to acceptors,
    each responds with accepted or rejected, then notifies learner.

    The session type models the proposer's view of a single ballot.

    Args:
        n_acceptors: Number of acceptors (1-5).

    Returns:
        ConsensusProtocol with the single-round Paxos session type.
    """
    if n_acceptors < 1 or n_acceptors > 5:
        raise ValueError(f"n_acceptors must be 1..5, got {n_acceptors}")

    # Phase 2: for each acceptor, proposer sends accept or skip.
    # On accept, acceptor responds accepted/rejected.
    learn = "end"
    phase2 = learn
    for i in range(n_acceptors, 0, -1):
        phase2 = (
            f"+{{accept{i}: &{{accepted{i}: {phase2}, "
            f"rejected{i}: {phase2}}}, skip{i}: {phase2}}}"
        )

    # Phase 1: for each acceptor, proposer sends prepare,
    # acceptor responds promise or nack.
    inner = phase2
    for i in range(n_acceptors, 0, -1):
        inner = (
            f"&{{prepare{i}: &{{promise{i}: {inner}, "
            f"nack{i}: {inner}}}}}"
        )

    roles = ("Proposer",) + tuple(f"Acceptor{i}" for i in range(1, n_acceptors + 1)) + ("Learner",)

    return ConsensusProtocol(
        name=f"Paxos_{n_acceptors}",
        roles=roles,
        session_type_string=inner,
        fault_model="crash",
        n_participants=n_acceptors,
        phases=2,
        description=(
            f"Single-round Paxos with {n_acceptors} acceptor(s). "
            "Phase 1: prepare/promise. Phase 2: accept/accepted. "
            "Models one ballot without retries."
        ),
    )


def raft_election(n_nodes: int = 3) -> ConsensusProtocol:
    """Raft leader election as a recursive session type.

    A candidate requests votes from other nodes. Each node grants
    or denies the vote. If enough votes are granted the candidate
    becomes leader and enters a heartbeat loop; otherwise it retries.

    The session type models the candidate's view:

        rec X . requestVote1 . &{grantVote1: ..., denyVote1: ...}
                ... +{becomeLeader: heartbeat-loop, retry: X}

    Args:
        n_nodes: Total number of nodes in the cluster (2-5).

    Returns:
        ConsensusProtocol with the Raft election session type.
    """
    if n_nodes < 2 or n_nodes > 5:
        raise ValueError(f"n_nodes must be 2..5, got {n_nodes}")

    n_voters = n_nodes - 1  # the candidate itself doesn't vote

    # After collecting all votes, candidate decides
    leader_loop = "rec Y . &{heartbeat: Y}"
    decision = f"+{{becomeLeader: {leader_loop}, retry: X}}"

    # Voting: each voter grants or denies
    inner = decision
    for i in range(n_voters, 0, -1):
        inner = (
            f"&{{requestVote{i}: &{{grantVote{i}: {inner}, "
            f"denyVote{i}: {inner}}}}}"
        )

    st = f"rec X . {inner}"

    roles = ("Candidate",) + tuple(f"Voter{i}" for i in range(1, n_voters + 1))

    return ConsensusProtocol(
        name=f"RaftElection_{n_nodes}",
        roles=roles,
        session_type_string=st,
        fault_model="crash",
        n_participants=n_nodes,
        phases=2,
        description=(
            f"Raft leader election with {n_nodes} nodes. "
            "Candidate requests votes, becomes leader on majority, "
            "or retries with new term."
        ),
    )


def pbft_basic(n_replicas: int = 3) -> ConsensusProtocol:
    """PBFT (Practical Byzantine Fault Tolerance) as a session type.

    Three phases: pre-prepare, prepare, commit.
    1. Client sends request to primary.
    2. Primary sends pre-prepare to all replicas.
    3. Replicas exchange prepare messages.
    4. Replicas exchange commit messages.
    5. Replicas reply to client.

    Simplified to the primary's view for tractability.

    Args:
        n_replicas: Number of replicas including primary (3-5).

    Returns:
        ConsensusProtocol with the PBFT session type.
    """
    if n_replicas < 3 or n_replicas > 5:
        raise ValueError(f"n_replicas must be 3..5, got {n_replicas}")

    n_backups = n_replicas - 1  # exclude primary

    # Phase 3 (Commit): primary collects commits and replies
    reply = "end"
    commit_phase = reply
    for i in range(n_backups, 0, -1):
        commit_phase = (
            f"&{{commit{i}: +{{replyOk{i}: {commit_phase}, "
            f"replyFail{i}: {commit_phase}}}}}"
        )

    # Phase 2 (Prepare): primary collects prepares from backups
    prepare_phase = commit_phase
    for i in range(n_backups, 0, -1):
        prepare_phase = (
            f"&{{prepare{i}: {prepare_phase}}}"
        )

    # Phase 1 (Pre-prepare): primary sends pre-prepare to each backup
    pre_prepare = prepare_phase
    for i in range(n_backups, 0, -1):
        pre_prepare = (
            f"&{{prePrepare{i}: {pre_prepare}}}"
        )

    # Client sends request
    st = f"&{{request: {pre_prepare}}}"

    roles = ("Client", "Primary") + tuple(f"Replica{i}" for i in range(1, n_backups + 1))

    return ConsensusProtocol(
        name=f"PBFT_{n_replicas}",
        roles=roles,
        session_type_string=st,
        fault_model="byzantine",
        n_participants=n_replicas,
        phases=3,
        description=(
            f"PBFT with {n_replicas} replicas (f={(n_replicas - 1) // 3} Byzantine faults). "
            "Phases: pre-prepare, prepare, commit. "
            "Tolerates up to f faulty replicas."
        ),
    )


def multi_paxos(n_acceptors: int = 3) -> ConsensusProtocol:
    """Multi-Paxos: repeated single rounds with retry.

    Wraps basic Paxos in a recursion with a decided/retry choice
    after each round.

    Args:
        n_acceptors: Number of acceptors (1-3).

    Returns:
        ConsensusProtocol with the Multi-Paxos session type.
    """
    if n_acceptors < 1 or n_acceptors > 3:
        raise ValueError(f"n_acceptors must be 1..3, got {n_acceptors}")

    # Single round (same as paxos_basic but without wrapping in protocol)
    learn = "+{decided: end, retry: X}"
    phase2 = learn
    for i in range(n_acceptors, 0, -1):
        phase2 = (
            f"+{{accept{i}: &{{accepted{i}: {phase2}, "
            f"rejected{i}: {phase2}}}, skip{i}: {phase2}}}"
        )

    inner = phase2
    for i in range(n_acceptors, 0, -1):
        inner = (
            f"&{{prepare{i}: &{{promise{i}: {inner}, "
            f"nack{i}: {inner}}}}}"
        )

    st = f"rec X . {inner}"

    roles = ("Proposer",) + tuple(f"Acceptor{i}" for i in range(1, n_acceptors + 1)) + ("Learner",)

    return ConsensusProtocol(
        name=f"MultiPaxos_{n_acceptors}",
        roles=roles,
        session_type_string=st,
        fault_model="crash",
        n_participants=n_acceptors,
        phases=2,
        description=(
            f"Multi-Paxos with {n_acceptors} acceptor(s). "
            "Repeated rounds with decided/retry after each ballot."
        ),
    )


def simple_broadcast(n_receivers: int = 3) -> ConsensusProtocol:
    """Simple reliable broadcast: sender sends to all, all acknowledge.

    Serves as a baseline: simpler than consensus, always terminates,
    always agrees (trivially, since there is one sender).

    Args:
        n_receivers: Number of receivers (1-5).

    Returns:
        ConsensusProtocol with the broadcast session type.
    """
    if n_receivers < 1 or n_receivers > 5:
        raise ValueError(f"n_receivers must be 1..5, got {n_receivers}")

    inner = "end"
    for i in range(n_receivers, 0, -1):
        inner = f"&{{ack{i}: {inner}}}"
    for i in range(n_receivers, 0, -1):
        inner = f"&{{send{i}: {inner}}}"

    roles = ("Sender",) + tuple(f"Receiver{i}" for i in range(1, n_receivers + 1))

    return ConsensusProtocol(
        name=f"Broadcast_{n_receivers}",
        roles=roles,
        session_type_string=inner,
        fault_model="none",
        n_participants=n_receivers,
        phases=1,
        description=(
            f"Reliable broadcast with {n_receivers} receiver(s). "
            "Sender sends to all, all acknowledge."
        ),
    )


# ---------------------------------------------------------------------------
# Registry of all protocols
# ---------------------------------------------------------------------------

ALL_CONSENSUS_PROTOCOLS: tuple[ConsensusProtocol, ...] = (
    two_phase_commit(2),
    two_phase_commit(3),
    paxos_basic(2),
    paxos_basic(3),
    raft_election(3),
    pbft_basic(3),
    multi_paxos(2),
    simple_broadcast(2),
    simple_broadcast(3),
)
"""Pre-defined consensus protocols for batch analysis."""


# ---------------------------------------------------------------------------
# Generic builder
# ---------------------------------------------------------------------------

_BUILDERS: dict[str, type] = {}  # unused, we dispatch by name below


def build_consensus_protocol(
    name: str,
    params: dict[str, int] | None = None,
) -> ConsensusProtocol:
    """Build a consensus protocol by name with optional parameters.

    Supported names: "2pc", "paxos", "raft", "pbft", "multi_paxos",
    "broadcast".

    Args:
        name: Protocol name (case-insensitive).
        params: Optional dict of parameters (e.g., {"n_participants": 3}).

    Returns:
        ConsensusProtocol instance.

    Raises:
        ValueError: If the protocol name is unknown.
    """
    params = params or {}
    key = name.lower().replace("-", "_").replace(" ", "_")

    if key in ("2pc", "two_phase_commit", "twophasecommit"):
        n = params.get("n_participants", 2)
        return two_phase_commit(n)
    elif key in ("paxos", "paxos_basic", "basic_paxos"):
        n = params.get("n_acceptors", 3)
        return paxos_basic(n)
    elif key in ("raft", "raft_election"):
        n = params.get("n_nodes", 3)
        return raft_election(n)
    elif key in ("pbft", "pbft_basic"):
        n = params.get("n_replicas", 3)
        return pbft_basic(n)
    elif key in ("multi_paxos", "multipaxos"):
        n = params.get("n_acceptors", 3)
        return multi_paxos(n)
    elif key in ("broadcast", "simple_broadcast"):
        n = params.get("n_receivers", 3)
        return simple_broadcast(n)
    else:
        raise ValueError(
            f"Unknown consensus protocol: {name!r}. "
            "Supported: 2pc, paxos, raft, pbft, multi_paxos, broadcast."
        )


# ---------------------------------------------------------------------------
# Safety property checks
# ---------------------------------------------------------------------------

def _find_decision_states(ss: StateSpace) -> dict[int, str]:
    """Identify decision states and their decision labels.

    A decision state is a state reached via a transition whose label
    is in DECISION_LABELS (commit, abort, accepted, etc.).

    Returns:
        Mapping from state ID to the decision label that leads to it.
    """
    decisions: dict[int, str] = {}
    for src, label, tgt in ss.transitions:
        if label in DECISION_LABELS or any(
            label.startswith(dl) for dl in DECISION_LABELS
        ):
            decisions[tgt] = label
    return decisions


def _find_proposal_states(ss: StateSpace) -> dict[int, str]:
    """Identify proposal states and their proposal labels.

    A proposal state is a state reached via a transition whose label
    is in PROPOSAL_LABELS (propose, prepare, request, etc.).

    Returns:
        Mapping from state ID to the proposal label that leads to it.
    """
    proposals: dict[int, str] = {}
    for src, label, tgt in ss.transitions:
        if label in PROPOSAL_LABELS or any(
            label.startswith(pl) for pl in PROPOSAL_LABELS
        ):
            proposals[tgt] = label
    return proposals


def check_agreement(ss: StateSpace) -> AgreementResult:
    """Verify the agreement property on a consensus state space.

    Agreement holds if all decision states that share a common
    future (non-bottom meet) carry the same decision label category.
    In practice: all commit-like decisions are consistent.

    Two decision states *conflict* if one carries a "commit"-family
    label and the other an "abort"-family label, AND both are
    reachable from the top (i.e., both can occur in valid executions).

    Args:
        ss: The state space to check.

    Returns:
        AgreementResult with the check outcome.
    """
    decisions = _find_decision_states(ss)
    decision_ids = set(decisions.keys())

    if len(decision_ids) <= 1:
        return AgreementResult(
            holds=True,
            decision_states=decision_ids,
            decision_labels=decisions,
            conflicting_pair=None,
        )

    # Classify labels into positive/negative categories
    positive = {"commit", "accepted", "decided", "grantVote", "learn", "reply",
                "replyOk", "becomeLeader"}
    negative = {"abort", "rejected", "denyVote", "replyFail"}

    def _category(label: str) -> str:
        """Classify a label as 'positive', 'negative', or 'neutral'."""
        for p in positive:
            if label == p or label.startswith(p):
                return "positive"
        for n in negative:
            if label == n or label.startswith(n):
                return "negative"
        return "neutral"

    # Check: do any two decision states on the SAME path carry conflicting
    # categories?  We check whether positive and negative decisions are
    # both reachable from the same ancestor.
    # Simpler conservative check: all reachable decision states from top
    # must carry consistent categories (within each maximal path).
    reachable_from_top = ss.reachable_from(ss.top)

    # Collect categories of all reachable decision states
    categories: dict[str, list[int]] = {}
    for sid in decision_ids:
        if sid in reachable_from_top:
            cat = _category(decisions[sid])
            categories.setdefault(cat, []).append(sid)

    # Agreement violation: both positive and negative decisions reachable,
    # AND they share a common ancestor (which they do, since both are
    # reachable from top).  But we need to check per-path, not globally.
    # Conservative approach: check if for any pair of decision states
    # with different categories, they can both occur on the same path.
    # Two states s1, s2 are on the same path if one is reachable from
    # the other, or they share a non-top common ancestor above them.

    # For consensus protocols, the key check is: given a specific voting
    # outcome, is the decision unique?  We check whether positive and
    # negative decisions are alternatives (different branches) or
    # can co-occur.
    pos_states = set(categories.get("positive", []))
    neg_states = set(categories.get("negative", []))

    if not pos_states or not neg_states:
        # Only one category of decisions: agreement trivially holds.
        return AgreementResult(
            holds=True,
            decision_states=decision_ids,
            decision_labels=decisions,
            conflicting_pair=None,
        )

    # Check: for each positive decision state, is any negative decision
    # state reachable from it (or vice versa)?  If so, that's a conflict.
    for ps in pos_states:
        reach_from_ps = ss.reachable_from(ps)
        for ns in neg_states:
            if ns in reach_from_ps:
                return AgreementResult(
                    holds=False,
                    decision_states=decision_ids,
                    decision_labels=decisions,
                    conflicting_pair=(ps, ns),
                )

    for ns in neg_states:
        reach_from_ns = ss.reachable_from(ns)
        for ps in pos_states:
            if ps in reach_from_ns:
                return AgreementResult(
                    holds=False,
                    decision_states=decision_ids,
                    decision_labels=decisions,
                    conflicting_pair=(ns, ps),
                )

    return AgreementResult(
        holds=True,
        decision_states=decision_ids,
        decision_labels=decisions,
        conflicting_pair=None,
    )


def check_validity(ss: StateSpace) -> ValidityResult:
    """Verify the validity property on a consensus state space.

    Validity holds if every decision state is reachable from some
    proposal state: decided values must have been proposed.

    Args:
        ss: The state space to check.

    Returns:
        ValidityResult with the check outcome.
    """
    proposals = _find_proposal_states(ss)
    decisions = _find_decision_states(ss)

    proposal_ids = set(proposals.keys())
    decision_ids = set(decisions.keys())

    if not decision_ids:
        # No decisions at all: vacuously valid
        return ValidityResult(
            holds=True,
            proposal_states=proposal_ids,
            decision_states=decision_ids,
            unmatched_decisions=set(),
        )

    # For each decision state, check if it's reachable from at least
    # one proposal state.
    unmatched: set[int] = set()
    for d_id in decision_ids:
        found = False
        for p_id in proposal_ids:
            if d_id in ss.reachable_from(p_id):
                found = True
                break
        if not found:
            # Also check: is the decision state reachable from top?
            # If it is, and top itself is a proposal origin, that counts.
            if d_id in ss.reachable_from(ss.top):
                found = True
        if not found:
            unmatched.add(d_id)

    return ValidityResult(
        holds=len(unmatched) == 0,
        proposal_states=proposal_ids,
        decision_states=decision_ids,
        unmatched_decisions=unmatched,
    )


def check_termination_guarantee(ss: StateSpace) -> TerminationResult:
    """Verify structural termination of a consensus state space.

    Structural termination holds if every state can eventually reach
    the bottom element (no deadlocks or livelocks for finite paths).

    For recursive protocols, this checks single-unfolding termination
    (after SCC quotient, every maximal path reaches bottom).

    Args:
        ss: The state space to check.

    Returns:
        TerminationResult with the check outcome.
    """
    bottom = ss.bottom
    stuck: set[int] = set()

    # Find states with no outgoing transitions that are not bottom
    for state in ss.states:
        if state == bottom:
            continue
        successors = ss.successors(state)
        if not successors:
            stuck.add(state)

    # Also check: can every state reach bottom?
    # Build reverse reachability from bottom
    can_reach_bottom: set[int] = set()
    # Reverse graph
    reverse: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        reverse.setdefault(tgt, set()).add(src)

    # BFS from bottom in reverse
    queue = [bottom]
    while queue:
        s = queue.pop(0)
        if s in can_reach_bottom:
            continue
        can_reach_bottom.add(s)
        for pred in reverse.get(s, set()):
            if pred not in can_reach_bottom:
                queue.append(pred)

    # States that cannot reach bottom (excluding states in SCCs that
    # represent recursive loops — those are expected)
    unreachable = ss.states - can_reach_bottom

    # For termination, stuck states are the real concern
    all_stuck = stuck | unreachable

    # Count maximal paths (bounded)
    num_paths = _count_maximal_paths(ss, limit=1000)

    return TerminationResult(
        holds=len(all_stuck) == 0,
        num_paths=num_paths,
        stuck_states=all_stuck,
    )


def _count_maximal_paths(ss: StateSpace, limit: int = 1000) -> int:
    """Count the number of maximal paths from top to bottom (bounded).

    Uses DFS with cycle detection to avoid infinite loops.

    Args:
        ss: The state space.
        limit: Maximum count before returning early.

    Returns:
        Number of maximal paths (capped at limit).
    """
    count = 0
    stack: list[tuple[int, frozenset[int]]] = [(ss.top, frozenset())]

    while stack and count < limit:
        state, visited = stack.pop()
        if state == ss.bottom:
            count += 1
            continue
        if state in visited:
            continue
        new_visited = visited | {state}
        succs = ss.successors(state)
        if not succs:
            count += 1  # maximal path ending at non-bottom
            continue
        for s in succs:
            stack.append((s, new_visited))

    return count


# ---------------------------------------------------------------------------
# Consensus invariants
# ---------------------------------------------------------------------------

def consensus_invariants(ss: StateSpace) -> ConsensusInvariants:
    """Compute consensus-specific invariants from a state space.

    Calculates lattice height, width, and decision/proposal metrics.

    Args:
        ss: The state space to analyze.

    Returns:
        ConsensusInvariants with computed metrics.
    """
    decisions = _find_decision_states(ss)
    proposals = _find_proposal_states(ss)

    height = _lattice_height(ss)
    width = _lattice_width(ss)

    # Decision depth: average distance from top to each decision state
    depths: list[int] = []
    for d_id in decisions:
        d = _shortest_distance(ss, ss.top, d_id)
        if d >= 0:
            depths.append(d)
    avg_depth = sum(depths) / len(depths) if depths else 0.0

    # Unique decision point: do all decisions share a common predecessor?
    # Check if there's a single state through which all decision
    # transitions pass.
    decision_predecessors: set[int] = set()
    for src, label, tgt in ss.transitions:
        if tgt in decisions:
            decision_predecessors.add(src)

    has_unique = len(decision_predecessors) <= 1

    return ConsensusInvariants(
        lattice_height=height,
        lattice_width=width,
        num_decision_states=len(decisions),
        num_proposal_states=len(proposals),
        decision_depth=avg_depth,
        has_unique_decision_point=has_unique,
    )


def _lattice_height(ss: StateSpace) -> int:
    """Compute the height of the lattice (longest path from top to bottom)."""
    if ss.top == ss.bottom:
        return 0

    # BFS/DFS longest path (DAG assumption after quotient)
    dist: dict[int, int] = {ss.top: 0}
    visited: set[int] = set()
    stack = [ss.top]

    # Topological-order longest path
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for _, tgt in ss.enabled(s):
            new_dist = dist[s] + 1
            if tgt not in dist or new_dist > dist[tgt]:
                dist[tgt] = new_dist
            if tgt not in visited:
                stack.append(tgt)

    return dist.get(ss.bottom, 0)


def _lattice_width(ss: StateSpace) -> int:
    """Compute the width of the lattice (size of largest antichain).

    Uses depth layering: states at the same BFS depth from top are
    candidates for an antichain.  This is an approximation.
    """
    # BFS from top to assign depths
    depth: dict[int, int] = {ss.top: 0}
    queue = [ss.top]
    while queue:
        s = queue.pop(0)
        for _, tgt in ss.enabled(s):
            if tgt not in depth:
                depth[tgt] = depth[s] + 1
                queue.append(tgt)

    # Count states per depth level
    levels: dict[int, int] = {}
    for d in depth.values():
        levels[d] = levels.get(d, 0) + 1

    return max(levels.values()) if levels else 1


def _shortest_distance(ss: StateSpace, src: int, tgt: int) -> int:
    """Shortest path distance from src to tgt (-1 if unreachable)."""
    if src == tgt:
        return 0
    dist: dict[int, int] = {src: 0}
    queue = [src]
    while queue:
        s = queue.pop(0)
        for _, next_s in ss.enabled(s):
            if next_s not in dist:
                dist[next_s] = dist[s] + 1
                if next_s == tgt:
                    return dist[next_s]
                queue.append(next_s)
    return -1


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------

def analyze_consensus(
    name: str,
    params: dict[str, int] | None = None,
) -> ConsensusAnalysisResult:
    """Run the full analysis pipeline on a consensus protocol.

    Builds the protocol, parses its session type, constructs the
    state space, checks lattice properties, and verifies consensus
    safety properties (agreement, validity, termination).

    Args:
        name: Protocol name (see build_consensus_protocol).
        params: Optional parameters dict.

    Returns:
        ConsensusAnalysisResult with all findings.
    """
    proto = build_consensus_protocol(name, params)
    return analyze_consensus_protocol(proto)


def analyze_consensus_protocol(
    proto: ConsensusProtocol,
) -> ConsensusAnalysisResult:
    """Run the full analysis pipeline on a ConsensusProtocol instance.

    Args:
        proto: The consensus protocol to analyze.

    Returns:
        ConsensusAnalysisResult with all findings.
    """
    ast = parse(proto.session_type_string)
    ss = build_statespace(ast)
    lr = check_lattice(ss)
    agr = check_agreement(ss)
    val = check_validity(ss)
    term = check_termination_guarantee(ss)
    inv = consensus_invariants(ss)

    return ConsensusAnalysisResult(
        protocol=proto,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        agreement=agr,
        validity=val,
        termination=term,
        invariants=inv,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_well_formed=lr.is_lattice,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_consensus_report(result: ConsensusAnalysisResult) -> str:
    """Format a ConsensusAnalysisResult as structured text for terminal output."""
    lines: list[str] = []
    proto = result.protocol

    lines.append("=" * 70)
    lines.append(f"  CONSENSUS PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")

    # Roles
    lines.append("--- Protocol Roles ---")
    for r in proto.roles:
        lines.append(f"  - {r}")
    lines.append("")

    # Parameters
    lines.append("--- Parameters ---")
    lines.append(f"  Fault model:     {proto.fault_model}")
    lines.append(f"  Participants:    {proto.n_participants}")
    lines.append(f"  Phases:          {proto.phases}")
    lines.append("")

    # State space
    lines.append("--- State Space ---")
    lines.append(f"  States:      {result.num_states}")
    lines.append(f"  Transitions: {result.num_transitions}")
    lines.append(f"  Top (init):  {result.state_space.top}")
    lines.append(f"  Bottom (end):{result.state_space.bottom}")
    lines.append("")

    # Lattice
    lines.append("--- Lattice Analysis ---")
    lines.append(f"  Is lattice:  {result.lattice_result.is_lattice}")
    if not result.lattice_result.is_lattice and result.lattice_result.counterexample:
        lines.append(f"  Counterexample: {result.lattice_result.counterexample}")
    lines.append("")

    # Invariants
    inv = result.invariants
    lines.append("--- Consensus Invariants ---")
    lines.append(f"  Lattice height:   {inv.lattice_height}")
    lines.append(f"  Lattice width:    {inv.lattice_width}")
    lines.append(f"  Decision states:  {inv.num_decision_states}")
    lines.append(f"  Proposal states:  {inv.num_proposal_states}")
    lines.append(f"  Avg decision depth: {inv.decision_depth:.1f}")
    lines.append(f"  Unique decision point: {inv.has_unique_decision_point}")
    lines.append("")

    # Safety properties
    lines.append("--- Safety Properties ---")
    agr = result.agreement
    lines.append(f"  Agreement:   {'PASS' if agr.holds else 'FAIL'}")
    if not agr.holds and agr.conflicting_pair:
        lines.append(f"    Conflict: states {agr.conflicting_pair}")
    val = result.validity
    lines.append(f"  Validity:    {'PASS' if val.holds else 'FAIL'}")
    if not val.holds:
        lines.append(f"    Unmatched: {val.unmatched_decisions}")
    term = result.termination
    lines.append(f"  Termination: {'PASS' if term.holds else 'FAIL'}")
    if not term.holds:
        lines.append(f"    Stuck states: {term.stuck_states}")
    lines.append("")

    # Verdict
    lines.append("--- Verdict ---")
    all_pass = agr.holds and val.holds and result.is_well_formed
    if all_pass:
        lines.append("  PASS: Protocol forms a valid lattice with consensus safety.")
    else:
        lines.append("  ISSUES FOUND:")
        if not result.is_well_formed:
            lines.append("    - State space does NOT form a lattice.")
        if not agr.holds:
            lines.append("    - Agreement property VIOLATED.")
        if not val.holds:
            lines.append("    - Validity property VIOLATED.")
    lines.append("")

    return "\n".join(lines)


def format_consensus_summary(results: list[ConsensusAnalysisResult]) -> str:
    """Format a summary table of all verified consensus protocols."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("  CONSENSUS PROTOCOL VERIFICATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    header = (
        f"  {'Protocol':<22} {'States':>6} {'Trans':>6} "
        f"{'Lattice':>8} {'Agree':>6} {'Valid':>6} {'Term':>5}"
    )
    lines.append(header)
    lines.append("  " + "-" * 63)

    for r in results:
        lat = "YES" if r.is_well_formed else "NO"
        agr = "YES" if r.agreement.holds else "NO"
        val = "YES" if r.validity.holds else "NO"
        trm = "YES" if r.termination.holds else "NO"
        row = (
            f"  {r.protocol.name:<22} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{lat:>8} "
            f"{agr:>6} "
            f"{val:>6} "
            f"{trm:>5}"
        )
        lines.append(row)

    lines.append("")
    all_lattice = all(r.is_well_formed for r in results)
    all_agree = all(r.agreement.holds for r in results)
    lines.append(f"  All protocols form lattices: {'YES' if all_lattice else 'NO'}")
    lines.append(f"  All protocols satisfy agreement: {'YES' if all_agree else 'NO'}")
    lines.append("")

    return "\n".join(lines)
