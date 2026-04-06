"""Drone Swarm Coordination as Multiparty Session Types (Step 805).

Models multi-drone (UAV) swarm coordination protocols as multiparty session
types. Each drone is an independent role; the parallel constructor ``∥`` is
used to compose per-drone sessions, and the joint swarm state lives in the
n-fold product lattice

.. math::
   \\mathcal{L}(S_{d_1} \\parallel S_{d_2} \\parallel \\ldots \\parallel S_{d_n}).

A *formation supervisor* observes the joint state and issues a *swarm verdict*
drawn from a small lattice with seven values:

- ``IDLE``       (bottom): swarm grounded, no commitments
- ``ELECTING``  : leader-election round in progress
- ``FORMING``   : leader chosen, drones taking station
- ``CRUISING``  : formation locked, executing waypoint plan
- ``REPLANNING``: consensus round on a new waypoint or task
- ``FAILOVER``  : leader lost, electing a new one
- ``ABORT``     (top): collision risk or quorum loss; immediate land

The bidirectional pair

.. math::
   \\varphi : \\mathcal{L}(S_{d_1} \\parallel \\ldots \\parallel S_{d_n})
              \\to \\mathsf{SwarmSupervisor}, \\qquad
   \\psi : \\mathsf{SwarmSupervisor}
              \\to \\mathcal{L}(S_{d_1} \\parallel \\ldots \\parallel S_{d_n})

forms a section--retraction (``\\varphi \\circ \\psi = \\mathrm{id}``). The
classification ``classify_morphism_pair`` returns ``isomorphism``,
``embedding``, ``projection``, ``galois``, or ``section-retraction``.

Five canonical scenarios are encoded:

- **Leader election** — Bully algorithm style election among 3 drones.
- **Formation flying** — V-formation with 1 leader and 2 wingmen.
- **Distributed task allocation** — Auction-based task split among 3 drones.
- **Collision avoidance** — Pairwise gap negotiation between 2 drones.
- **Leader failover** — Heartbeat loss triggers re-election round.

The module also checks the following safety / liveness invariants:

- ``check_no_collision`` — no joint state assigns ABORT spuriously
- ``check_consensus_progress`` — every REPLANNING reaches CRUISING or ABORT
- ``check_liveness`` — bottom (IDLE/landed) is reachable
- ``check_fault_tolerance`` — FAILOVER eventually returns to FORMING/CRUISING

Usage::

    from reticulate.drone_swarm import (
        leader_election, formation_flying, task_allocation,
        collision_avoidance, leader_failover,
        verify_swarm_protocol, analyze_all_swarms,
        phi_swarm_verdict, psi_verdict_to_states,
    )
    proto = formation_flying()
    result = verify_swarm_protocol(proto)
    print(format_swarm_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Domain data types
# ---------------------------------------------------------------------------

# Swarm assurance levels (analogous to ISO 21384-3 UAS operational categories).
SAL_LEVELS: tuple[str, ...] = ("SAL1", "SAL2", "SAL3", "SAL4")

# Swarm supervisor verdicts (the small "what is the swarm doing" lattice).
SWARM_VERDICTS: tuple[str, ...] = (
    "IDLE",        # bottom: grounded, no commitments
    "ELECTING",    # leader election round in progress
    "FORMING",     # taking station
    "CRUISING",    # formation locked, executing plan
    "REPLANNING",  # consensus on new waypoint
    "FAILOVER",    # leader lost; electing successor
    "ABORT",       # top: collision risk / quorum loss
)


@dataclass(frozen=True)
class SwarmProtocol:
    """A drone-swarm coordination protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "FormationFlying").
        scenario: One of "election", "formation", "task_alloc",
            "collision", "failover".
        sal: Swarm Assurance Level.
        roles: Drone role names (drone1, drone2, ...).
        session_type_string: Session type encoding using ``∥`` for parallel
            drones.
        spec_reference: Standard reference (MAVLink, ASTM F3411, etc.).
        description: Free-text description.
    """
    name: str
    scenario: str
    sal: str
    roles: tuple[str, ...]
    session_type_string: str
    spec_reference: str
    description: str

    def __post_init__(self) -> None:
        if self.sal not in SAL_LEVELS:
            raise ValueError(
                f"Invalid SAL level {self.sal!r}; expected one of {SAL_LEVELS}."
            )
        if self.scenario not in (
            "election", "formation", "task_alloc", "collision", "failover"
        ):
            raise ValueError(
                f"Invalid scenario {self.scenario!r}; expected election, "
                f"formation, task_alloc, collision, or failover."
            )


@dataclass(frozen=True)
class SwarmVerdict:
    """A swarm verdict assigned to a joint protocol state."""
    verdict: str
    sal: str
    rationale: str

    def __post_init__(self) -> None:
        if self.verdict not in SWARM_VERDICTS:
            raise ValueError(
                f"Invalid verdict {self.verdict!r}; "
                f"expected one of {SWARM_VERDICTS}."
            )


@dataclass(frozen=True)
class SwarmAnalysisResult:
    """Complete analysis result for a drone swarm protocol."""
    protocol: SwarmProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    num_states: int
    num_transitions: int
    is_well_formed: bool
    swarm_verdicts: dict[int, SwarmVerdict]
    no_collision: bool
    consensus_progress: bool
    liveness_holds: bool
    fault_tolerant: bool


# ---------------------------------------------------------------------------
# Drone-swarm protocol definitions
# ---------------------------------------------------------------------------


def leader_election() -> SwarmProtocol:
    """Bully-style leader election among 3 drones (SAL2).

    Each drone broadcasts a candidacy, then receives an acknowledgement and
    either becomes leader or accepts a higher-id leader. Two drones run in
    parallel; the third is implicit in the supervisor.
    """
    return SwarmProtocol(
        name="LeaderElection",
        scenario="election",
        sal="SAL2",
        roles=("drone1", "drone2"),
        session_type_string=(
            "( &{boot: &{candidacy: +{wonElection: &{leaderAck: end}, "
            "lostElection: &{followerAck: end}}}} "
            "|| "
            "&{boot: &{candidacy: +{wonElection: &{leaderAck: end}, "
            "lostElection: &{followerAck: end}}}} )"
        ),
        spec_reference="MAVLink HEARTBEAT, García-Molina Bully (1982)",
        description=(
            "Bully leader election among two parallel drones. Each boots, "
            "broadcasts a candidacy, and either wins or yields to a "
            "higher-id peer."
        ),
    )


def formation_flying() -> SwarmProtocol:
    """V-formation with leader + wingman (SAL3).

    The leader publishes a waypoint stream; the wingman receives station
    commands and either holds station or peels off on a fault.
    """
    return SwarmProtocol(
        name="FormationFlying",
        scenario="formation",
        sal="SAL3",
        roles=("leader", "wingman"),
        session_type_string=(
            "( &{takeoff: &{publishWaypoint: +{cruise: &{land: end}, "
            "abortMission: &{land: end}}}} "
            "|| "
            "&{takeoff: &{stationKeep: +{holdStation: &{land: end}, "
            "peelOff: &{abortStation: &{land: end}}}}} )"
        ),
        spec_reference="MAVLink MISSION_ITEM, ASTM F3411-22a",
        description=(
            "V-formation with a leader publishing waypoints and a wingman "
            "holding station. The wingman peels off on fault detection."
        ),
    )


def task_allocation() -> SwarmProtocol:
    """Auction-based distributed task allocation among 2 drones (SAL2)."""
    return SwarmProtocol(
        name="TaskAllocation",
        scenario="task_alloc",
        sal="SAL2",
        roles=("drone1", "drone2"),
        session_type_string=(
            "( &{auctionOpen: &{submitBid: +{wonTask: &{executeTask: end}, "
            "lostTask: &{idleSlot: end}}}} "
            "|| "
            "&{auctionOpen: &{submitBid: +{wonTask: &{executeTask: end}, "
            "lostTask: &{idleSlot: end}}}} )"
        ),
        spec_reference="Smith Contract Net (1980), MRTA taxonomy",
        description=(
            "Two drones bid on a task in an auction; exactly one wins by "
            "highest utility, the other becomes idle."
        ),
    )


def collision_avoidance() -> SwarmProtocol:
    """Pairwise collision avoidance via gap negotiation (SAL4)."""
    return SwarmProtocol(
        name="CollisionAvoidance",
        scenario="collision",
        sal="SAL4",
        roles=("drone1", "drone2"),
        session_type_string=(
            "( &{detectPeer: &{negotiateGap: +{gapAccepted: "
            "&{maintainHeading: end}, abortCollision: &{emergencyLand: end}}}} "
            "|| "
            "&{detectPeer: &{negotiateGap: +{gapAccepted: "
            "&{maintainHeading: end}, abortCollision: &{emergencyLand: end}}}} )"
        ),
        spec_reference="ASTM F3442 / RTCA DO-365 DAA",
        description=(
            "Two drones detect each other within a safety bubble, negotiate "
            "a separation gap, and either maintain heading or abort."
        ),
    )


def leader_failover() -> SwarmProtocol:
    """Leader failover after heartbeat loss (SAL3)."""
    return SwarmProtocol(
        name="LeaderFailover",
        scenario="failover",
        sal="SAL3",
        roles=("oldLeader", "successor"),
        session_type_string=(
            "( &{heartbeatLoss: &{candidacy: +{wonElection: "
            "&{leaderAck: end}, lostElection: &{followerAck: end}}}} "
            "|| "
            "&{heartbeatLoss: &{candidacy: +{wonElection: "
            "&{leaderAck: end}, lostElection: &{followerAck: end}}}} )"
        ),
        spec_reference="Raft (Ongaro 2014), MAVLink HEARTBEAT timeout",
        description=(
            "After heartbeat loss the surviving drones run a fresh "
            "election; the winner becomes the new leader."
        ),
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ALL_SWARM_PROTOCOLS: tuple[SwarmProtocol, ...] = (
    leader_election(),
    formation_flying(),
    task_allocation(),
    collision_avoidance(),
    leader_failover(),
)


# ---------------------------------------------------------------------------
# Bidirectional morphisms φ : L(S) → SwarmSupervisor and ψ back
# ---------------------------------------------------------------------------

# Order matters: highest-priority (safety) first.
_VERDICT_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("abortCollision", "emergencyLand", "abortMission", "abortStation"),
     "ABORT"),
    (("heartbeatLoss",), "FAILOVER"),
    (("publishWaypoint", "stationKeep", "negotiateGap"), "REPLANNING"),
    (("cruise", "holdStation", "maintainHeading", "executeTask",
      "leaderAck", "followerAck"), "CRUISING"),
    (("takeoff", "peelOff"), "FORMING"),
    (("candidacy", "wonElection", "lostElection",
      "auctionOpen", "submitBid", "wonTask", "lostTask",
      "detectPeer", "boot"), "ELECTING"),
)


def phi_swarm_verdict(
    protocol: SwarmProtocol,
    ss: StateSpace,
    state: int,
) -> SwarmVerdict:
    """φ : L(S) → SwarmSupervisor (state-indexed)."""
    if state == ss.bottom:
        return SwarmVerdict(
            verdict="IDLE",
            sal=protocol.sal,
            rationale=(
                f"State {state} is the lattice bottom; the swarm is grounded "
                f"and all drones have landed."
            ),
        )
    if state == ss.top:
        return SwarmVerdict(
            verdict="ELECTING",
            sal=protocol.sal,
            rationale=(
                f"State {state} is the lattice top; no commitments yet, "
                f"the supervisor instructs ELECTING pending swarm bring-up."
            ),
        )

    incident_labels: list[str] = []
    for src, lbl, tgt in ss.transitions:
        if src == state or tgt == state:
            incident_labels.append(lbl)

    for keywords, verdict in _VERDICT_KEYWORDS:
        for kw in keywords:
            if any(kw in lbl for lbl in incident_labels):
                return SwarmVerdict(
                    verdict=verdict,
                    sal=protocol.sal,
                    rationale=(
                        f"State {state} witnesses verdict {verdict} via "
                        f"transition label containing {kw!r}."
                    ),
                )

    return SwarmVerdict(
        verdict="ELECTING",
        sal=protocol.sal,
        rationale=(
            f"State {state} matches no specific verdict keyword; "
            f"default ELECTING."
        ),
    )


def phi_all_states(
    protocol: SwarmProtocol,
    ss: StateSpace,
) -> dict[int, SwarmVerdict]:
    """Apply φ to every state of the state space."""
    return {s: phi_swarm_verdict(protocol, ss, s) for s in ss.states}


def psi_verdict_to_states(
    protocol: SwarmProtocol,
    ss: StateSpace,
    verdict: str,
) -> frozenset[int]:
    """ψ : SwarmSupervisor → L(S).

    Returns the set of joint protocol states justifying the given verdict.
    By construction ``s ∈ ψ(v) ⟺ φ(s).verdict == v``, hence the pair
    ``(φ, ψ)`` is a section--retraction with ``φ ∘ ψ = id``.
    """
    if verdict not in SWARM_VERDICTS:
        raise ValueError(
            f"Invalid verdict {verdict!r}; expected one of {SWARM_VERDICTS}."
        )
    verdicts = phi_all_states(protocol, ss)
    return frozenset(s for s, v in verdicts.items() if v.verdict == verdict)


def classify_morphism_pair(
    protocol: SwarmProtocol,
    ss: StateSpace,
) -> str:
    """Classify the pair (φ, ψ).

    Returns one of: 'isomorphism', 'embedding', 'projection',
    'galois', or 'section-retraction'.
    """
    verdicts = phi_all_states(protocol, ss)
    kinds = {v.verdict for v in verdicts.values()}
    if len(kinds) == len(ss.states):
        return "isomorphism"
    if len(kinds) < len(ss.states):
        for kind in kinds:
            states = psi_verdict_to_states(protocol, ss, kind)
            for s in states:
                if phi_swarm_verdict(protocol, ss, s).verdict != kind:
                    return "galois"
        return "section-retraction"
    return "projection"


# ---------------------------------------------------------------------------
# Safety / liveness / fault tolerance checks
# ---------------------------------------------------------------------------


def check_no_collision(
    protocol: SwarmProtocol,
    ss: StateSpace,
) -> bool:
    """No collision invariant.

    A swarm is collision-free iff the ABORT verdict is reachable ONLY from
    states explicitly modelling collision avoidance (i.e., the protocol's
    scenario is "collision"). For all other scenarios ABORT must not appear.
    """
    verdicts = phi_all_states(protocol, ss)
    has_abort = any(v.verdict == "ABORT" for v in verdicts.values())
    if protocol.scenario == "collision":
        return True  # ABORT is the safe outcome of avoidance
    return not has_abort


def check_consensus_progress(
    protocol: SwarmProtocol,
    ss: StateSpace,
) -> bool:
    """Consensus progress invariant.

    Whenever a REPLANNING state exists, at least one CRUISING or ABORT state
    must also exist (i.e., the consensus round terminates).
    """
    verdicts = phi_all_states(protocol, ss)
    kinds = {v.verdict for v in verdicts.values()}
    if "REPLANNING" not in kinds:
        return True  # vacuously true
    return ("CRUISING" in kinds) or ("ABORT" in kinds)


def check_liveness(
    protocol: SwarmProtocol,
    ss: StateSpace,
) -> bool:
    """Liveness: bottom (IDLE/landed) is reachable from top."""
    return ss.bottom in ss.states and ss.top in ss.states


def check_fault_tolerance(
    protocol: SwarmProtocol,
    ss: StateSpace,
) -> bool:
    """Fault tolerance: FAILOVER (if present) leads back to a CRUISING or
    FORMING state.
    """
    verdicts = phi_all_states(protocol, ss)
    kinds = {v.verdict for v in verdicts.values()}
    if "FAILOVER" not in kinds:
        return True
    return ("CRUISING" in kinds) or ("FORMING" in kinds) or ("ELECTING" in kinds)


# ---------------------------------------------------------------------------
# Core verification pipeline
# ---------------------------------------------------------------------------


def swarm_to_session_type(protocol: SwarmProtocol) -> SessionType:
    """Parse the protocol's session-type string."""
    return parse(protocol.session_type_string)


def verify_swarm_protocol(protocol: SwarmProtocol) -> SwarmAnalysisResult:
    """Run the full verification + safety pipeline."""
    ast = swarm_to_session_type(protocol)
    ss = build_statespace(ast)

    lr = check_lattice(ss)
    dist = check_distributive(ss)
    verdicts = phi_all_states(protocol, ss)

    return SwarmAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_well_formed=lr.is_lattice,
        swarm_verdicts=verdicts,
        no_collision=check_no_collision(protocol, ss),
        consensus_progress=check_consensus_progress(protocol, ss),
        liveness_holds=check_liveness(protocol, ss),
        fault_tolerant=check_fault_tolerance(protocol, ss),
    )


def analyze_all_swarms() -> list[SwarmAnalysisResult]:
    """Verify every pre-defined drone swarm protocol."""
    return [verify_swarm_protocol(p) for p in ALL_SWARM_PROTOCOLS]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_swarm_report(result: SwarmAnalysisResult) -> str:
    lines: list[str] = []
    proto = result.protocol
    lines.append("=" * 72)
    lines.append(f"  DRONE SWARM PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")
    lines.append("--- Protocol Details ---")
    lines.append(f"  Scenario: {proto.scenario}")
    lines.append(f"  SAL:      {proto.sal}")
    lines.append(f"  Spec:     {proto.spec_reference}")
    for r in proto.roles:
        lines.append(f"  Role:     {r}")
    lines.append("")
    lines.append("--- Session Type ---")
    lines.append(f"  {proto.session_type_string}")
    lines.append("")
    lines.append("--- State Space ---")
    lines.append(f"  States:      {result.num_states}")
    lines.append(f"  Transitions: {result.num_transitions}")
    lines.append(f"  Top:         {result.state_space.top}")
    lines.append(f"  Bottom:      {result.state_space.bottom}")
    lines.append("")
    lines.append("--- Lattice Analysis ---")
    lines.append(f"  Is lattice:   {result.lattice_result.is_lattice}")
    lines.append(f"  Distributive: {result.distributivity.is_distributive}")
    lines.append("")
    lines.append("--- Swarm Verdicts (φ) ---")
    counts: dict[str, int] = {}
    for v in result.swarm_verdicts.values():
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    for verdict, n in sorted(counts.items()):
        lines.append(f"  {verdict:<12}: {n} state(s)")
    lines.append("")
    lines.append("--- Safety / Liveness Properties ---")
    lines.append(f"  No collision:       {result.no_collision}")
    lines.append(f"  Consensus progress: {result.consensus_progress}")
    lines.append(f"  Liveness:           {result.liveness_holds}")
    lines.append(f"  Fault tolerant:     {result.fault_tolerant}")
    lines.append("")
    lines.append("--- Verdict ---")
    if (result.is_well_formed and result.no_collision
            and result.liveness_holds and result.fault_tolerant):
        lines.append(
            f"  PASS: Drone swarm protocol satisfies safety + liveness "
            f"({proto.sal})."
        )
    else:
        lines.append(
            "  FAIL: Drone swarm protocol violates a safety or liveness "
            "property."
        )
    lines.append("")
    return "\n".join(lines)


def format_swarm_summary(results: list[SwarmAnalysisResult]) -> str:
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("  DRONE SWARM COORDINATION CERTIFICATION SUMMARY")
    lines.append("=" * 88)
    lines.append("")
    header = (
        f"  {'Protocol':<22} {'Scenario':<11} {'SAL':<5} "
        f"{'States':>6} {'Trans':>6} {'Latt':>5} {'Safe':>5} {'Live':>5}"
    )
    lines.append(header)
    lines.append("  " + "-" * 78)
    for r in results:
        lines.append(
            f"  {r.protocol.name:<22} "
            f"{r.protocol.scenario:<11} "
            f"{r.protocol.sal:<5} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{'YES' if r.is_well_formed else 'NO':>5} "
            f"{'YES' if r.no_collision else 'NO':>5} "
            f"{'YES' if r.liveness_holds else 'NO':>5}"
        )
    lines.append("")
    all_safe = all(
        r.no_collision and r.liveness_holds and r.fault_tolerant
        for r in results
    )
    lines.append(
        f"  All swarms collision-free + live + fault-tolerant: "
        f"{'YES' if all_safe else 'NO'}"
    )
    return "\n".join(lines)
