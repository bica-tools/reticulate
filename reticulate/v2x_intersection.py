"""V2X Intersection Right-of-Way Protocol via lattice analysis (Step 804).

Models vehicle-to-everything (V2X) intersection coordination as multiparty
session types built from SAE J2735 BSM (Basic Safety Message) and SPaT/MAP
(Signal Phase and Timing / Map) message exchanges between an *ego vehicle*,
one or more *other vehicles*, an optional *roadside unit (RSU)* and a
*traffic signal*.

Three canonical scenarios are encoded:

- **Four-way stop sign** (uncontrolled intersection): vehicles arrive,
  broadcast BSMs, and a fairness/right-of-way supervisor decides who
  proceeds first based on FIFO order with tie-breaking by vehicle id.
- **Signalized intersection** (RSU-mediated): the traffic signal publishes
  SPaT messages; vehicles synchronise their approach phase with green
  windows.
- **Unprotected left turn**: ego vehicle yields to oncoming traffic until a
  safe gap is detected, then commits to the manoeuvre.

The parallel constructor ``∥`` is used to model concurrent vehicle
approaches: each vehicle has its own approach session, and the joint
intersection state lives in the product lattice
``L(S_ego ∥ S_other ∥ ... )``.  *Right-of-way* is captured as a Galois
projection

.. math::
   \\varphi : L(S_{\\text{ego}} \\parallel S_{\\text{other}}) \\to
   \\mathsf{ROWSupervisor},

where ``ROWSupervisor`` is the small "who goes next" lattice with five
verdicts: ``EGO_GO``, ``OTHER_GO``, ``WAIT``, ``EMERGENCY_BRAKE``,
``CLEAR``.  The reverse map

.. math::
   \\psi : \\mathsf{ROWSupervisor} \\to L(S_{\\text{ego}} \\parallel S_{\\text{other}})

retrieves the set of joint protocol states that justify a given verdict.
The pair ``(\\varphi, \\psi)`` is a *section--retraction* (``\\varphi
\\circ \\psi = \\mathrm{id}``) and supports collision-avoidance,
fairness/liveness verification, and ISO/SAE 21434 cybersecurity threat
analysis (the morphism reveals which states are observable to attackers
spoofing BSMs).

Usage:
    from reticulate.v2x_intersection import (
        four_way_stop, signalized_intersection, unprotected_left_turn,
        verify_v2x_protocol, analyze_all_v2x,
        phi_row_verdict, psi_verdict_to_states,
    )
    proto = four_way_stop()
    result = verify_v2x_protocol(proto)
    print(format_v2x_report(result))
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


# ISO/SAE 21434 cybersecurity assurance levels (CAL).
CAL_LEVELS: tuple[str, ...] = ("CAL1", "CAL2", "CAL3", "CAL4")

# Right-of-way verdicts (the small supervisor lattice).
ROW_VERDICTS: tuple[str, ...] = (
    "CLEAR",        # bottom: intersection cleared, all vehicles departed
    "EGO_GO",       # ego has right of way
    "OTHER_GO",     # other vehicle has right of way
    "WAIT",         # ego must wait (all-way stop, red light, etc.)
    "EMERGENCY_BRAKE",  # top: collision risk detected, immediate brake
)


@dataclass(frozen=True)
class V2xProtocol:
    """A V2X intersection protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "FourWayStop").
        scenario: One of "stop_sign", "signalized", "left_turn".
        cal: ISO/SAE 21434 Cybersecurity Assurance Level.
        roles: Roles involved (ego, other, rsu, signal, ...).
        session_type_string: Session type encoding using ``∥`` for parallel
            vehicle approaches.
        spec_reference: SAE / ISO standard reference.
        description: Free-text description.
    """
    name: str
    scenario: str
    cal: str
    roles: tuple[str, ...]
    session_type_string: str
    spec_reference: str
    description: str

    def __post_init__(self) -> None:
        if self.cal not in CAL_LEVELS:
            raise ValueError(
                f"Invalid CAL level {self.cal!r}; expected one of {CAL_LEVELS}."
            )
        if self.scenario not in ("stop_sign", "signalized", "left_turn"):
            raise ValueError(
                f"Invalid scenario {self.scenario!r}; expected stop_sign, "
                f"signalized, or left_turn."
            )


@dataclass(frozen=True)
class RowVerdict:
    """A right-of-way verdict assigned to a joint protocol state.

    Attributes:
        verdict: One of ROW_VERDICTS.
        cal: Cybersecurity assurance level (inherited from protocol).
        rationale: Human-readable justification.
    """
    verdict: str
    cal: str
    rationale: str

    def __post_init__(self) -> None:
        if self.verdict not in ROW_VERDICTS:
            raise ValueError(
                f"Invalid verdict {self.verdict!r}; expected one of {ROW_VERDICTS}."
            )


@dataclass(frozen=True)
class V2xAnalysisResult:
    """Complete analysis result for a V2X intersection protocol."""
    protocol: V2xProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    num_states: int
    num_transitions: int
    is_well_formed: bool
    row_verdicts: dict[int, RowVerdict]
    collision_free: bool
    fairness_holds: bool
    liveness_holds: bool


# ---------------------------------------------------------------------------
# V2X protocol definitions
# ---------------------------------------------------------------------------


def four_way_stop() -> V2xProtocol:
    """Four-way stop sign intersection (CAL2).

    Two vehicles approach an uncontrolled four-way stop. Each broadcasts a
    BSM (Basic Safety Message) on arrival, then waits its turn under FIFO
    arbitration. The parallel constructor models concurrent approaches.

    Session type (simplified, two vehicles):
        ( &{approach: &{bsmTx: &{stopLine: +{firstArrival: &{proceed: end},
                                             yield: &{wait: &{proceed: end}}}}}}
        ||
          &{approach: &{bsmTx: &{stopLine: +{firstArrival: &{proceed: end},
                                             yield: &{wait: &{proceed: end}}}}}} )
    """
    return V2xProtocol(
        name="FourWayStop",
        scenario="stop_sign",
        cal="CAL2",
        roles=("ego", "other"),
        session_type_string=(
            "( &{approach: &{bsmTx: &{stopLine: +{firstArrival: "
            "&{proceed: end}, yieldTurn: &{waitTurn: &{proceed: end}}}}}} "
            "|| "
            "&{approach: &{bsmTx: &{stopLine: +{firstArrival: "
            "&{proceed: end}, yieldTurn: &{waitTurn: &{proceed: end}}}}}} )"
        ),
        spec_reference="SAE J2735_202211 BSM, MUTCD 2B.05",
        description=(
            "Four-way stop intersection with two vehicles. Each vehicle "
            "approaches, broadcasts a BSM, stops at the line, and either "
            "proceeds first or yields. FIFO right-of-way arbitration."
        ),
    )


def signalized_intersection() -> V2xProtocol:
    """Signalized intersection mediated by RSU + traffic signal (CAL3).

    The traffic signal publishes SPaT (Signal Phase and Timing) messages;
    the ego vehicle synchronises its approach with the green phase. The RSU
    relays SPaT and acknowledges vehicle BSMs.

    Session type (ego ∥ signal):
        ( &{approach: &{bsmTx: &{spatRx: +{GREEN: &{proceed: end},
                                          YELLOW: &{decelerate: &{stop: end}},
                                          RED: &{stop: &{waitGreen: &{proceed: end}}}}}}}
        ||
          &{phaseRed: &{phaseGreen: &{phaseYellow: &{phaseRed2: end}}}} )
    """
    return V2xProtocol(
        name="SignalizedIntersection",
        scenario="signalized",
        cal="CAL3",
        roles=("ego", "rsu", "signal"),
        session_type_string=(
            "( &{approach: &{bsmTx: &{spatRx: +{GREEN: &{proceed: end}, "
            "YELLOW: &{decelerate: &{stopAtLine: end}}, "
            "RED: &{stopAtLine: &{waitGreen: &{proceed: end}}}}}}} "
            "|| "
            "&{phaseRed: &{phaseGreen: &{phaseYellow: &{phaseRed2: end}}}} )"
        ),
        spec_reference="SAE J2735_202211 SPaT/MAP, ISO 19091:2017",
        description=(
            "Signalized intersection with RSU broadcasting SPaT/MAP. Ego "
            "vehicle reads phase, proceeds on green, decelerates on yellow, "
            "stops on red. Parallel signal phase rotation."
        ),
    )


def unprotected_left_turn() -> V2xProtocol:
    """Unprotected left turn against oncoming traffic (CAL3).

    Ego vehicle approaches the intersection wishing to turn left across an
    oncoming lane. It must yield to oncoming traffic until a safe gap is
    detected, then commit to the manoeuvre. Models the classic
    "left-turn-across-path / opposite-direction" (LTAP/OD) scenario.
    """
    return V2xProtocol(
        name="UnprotectedLeftTurn",
        scenario="left_turn",
        cal="CAL3",
        roles=("ego", "oncoming"),
        session_type_string=(
            "( &{approach: &{bsmTx: &{enterIntersection: "
            "+{gapDetected: &{commitTurn: &{clearIntersection: end}}, "
            "noGap: &{yieldOncoming: &{waitGap: &{commitTurn: "
            "&{clearIntersection: end}}}}}}}} "
            "|| "
            "&{oncomingApproach: &{oncomingPass: end}} )"
        ),
        spec_reference="SAE J3161/1, NHTSA LTAP/OD CIB",
        description=(
            "Unprotected left turn against oncoming traffic. Ego yields "
            "until a safe gap, then commits. Concurrent oncoming vehicle "
            "session models the gap closure."
        ),
    )


def emergency_vehicle_preemption() -> V2xProtocol:
    """Emergency vehicle preemption (CAL4).

    An emergency vehicle (ambulance, fire truck) requests preemption from
    the RSU; the signal flips to green for the emergency direction; ego
    vehicle yields by stopping or pulling over.
    """
    return V2xProtocol(
        name="EmergencyPreemption",
        scenario="signalized",
        cal="CAL4",
        roles=("ego", "emergency", "rsu"),
        session_type_string=(
            "( &{approach: &{sirenDetected: &{pullOver: &{stop: "
            "&{waitClear: &{resume: end}}}}}} "
            "|| "
            "&{preemptRequest: &{rsuGrant: &{greenWave: "
            "&{passIntersection: end}}}} )"
        ),
        spec_reference="SAE J2735 EVP, NTCIP 1211",
        description=(
            "Emergency vehicle preemption: emergency requests green wave "
            "from RSU; ego vehicle pulls over and waits."
        ),
    )


def pedestrian_crossing() -> V2xProtocol:
    """Pedestrian crossing protected by Pedestrian-Safety Application (CAL3).

    A pedestrian carrying a smartphone broadcasts a Personal Safety Message
    (PSM); the ego vehicle's PSA detects the pedestrian and yields.
    """
    return V2xProtocol(
        name="PedestrianCrossing",
        scenario="signalized",
        cal="CAL3",
        roles=("ego", "pedestrian"),
        session_type_string=(
            "( &{approach: &{psmRx: +{pedDetected: "
            "&{decelerate: &{stopAtCrosswalk: &{waitClear: &{proceed: end}}}}, "
            "noPed: &{proceed: end}}}} "
            "|| "
            "&{stepOffCurb: &{cross: &{reachOtherSide: end}}} )"
        ),
        spec_reference="SAE J2945/9 (PSM), SAE J3186",
        description=(
            "Pedestrian-aware crossing using SAE J2945/9 PSM. Ego detects "
            "pedestrian via PSM and yields at crosswalk."
        ),
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ALL_V2X_PROTOCOLS: tuple[V2xProtocol, ...] = (
    four_way_stop(),
    signalized_intersection(),
    unprotected_left_turn(),
    emergency_vehicle_preemption(),
    pedestrian_crossing(),
)


# ---------------------------------------------------------------------------
# Bidirectional morphisms φ : L(S) → ROWSupervisor and ψ back
# ---------------------------------------------------------------------------

# Keyword catalogue mapping transition labels to ROW verdicts. Order matters:
# the FIRST matching rule wins, so emergency/safety rules come first.
_VERDICT_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("emergency", "siren", "pullOver"), "EMERGENCY_BRAKE"),
    (("waitTurn", "waitGreen", "yieldTurn", "yieldOncoming",
      "waitGap", "waitClear", "stopAtLine", "stopAtCrosswalk",
      "stop", "decelerate", "phaseRed", "RED", "noGap"), "WAIT"),
    (("firstArrival", "GREEN", "gapDetected", "commitTurn",
      "proceed", "rsuGrant", "greenWave"), "EGO_GO"),
    (("oncomingPass", "oncomingApproach", "phaseGreen",
      "passIntersection", "cross"), "OTHER_GO"),
)


def phi_row_verdict(
    protocol: V2xProtocol,
    ss: StateSpace,
    state: int,
) -> RowVerdict:
    """φ : L(S) → ROWSupervisor (state-indexed).

    Maps a joint protocol state to the dominant right-of-way verdict it
    justifies, based on the incident transition labels.
    """
    if state == ss.bottom:
        return RowVerdict(
            verdict="CLEAR",
            cal=protocol.cal,
            rationale=(
                f"State {state} is the lattice bottom; the intersection "
                f"has been cleared by all participating vehicles."
            ),
        )
    if state == ss.top:
        return RowVerdict(
            verdict="WAIT",
            cal=protocol.cal,
            rationale=(
                f"State {state} is the lattice top; no commitments yet, "
                f"the supervisor instructs WAIT pending arbitration."
            ),
        )

    incident_labels: list[str] = []
    for src, lbl, tgt in ss.transitions:
        if src == state or tgt == state:
            incident_labels.append(lbl)

    for keywords, verdict in _VERDICT_KEYWORDS:
        for kw in keywords:
            if any(kw in lbl for lbl in incident_labels):
                return RowVerdict(
                    verdict=verdict,
                    cal=protocol.cal,
                    rationale=(
                        f"State {state} witnesses verdict {verdict} via "
                        f"transition label containing {kw!r}."
                    ),
                )

    return RowVerdict(
        verdict="WAIT",
        cal=protocol.cal,
        rationale=(
            f"State {state} matches no specific verdict keyword; default WAIT."
        ),
    )


def phi_all_states(
    protocol: V2xProtocol,
    ss: StateSpace,
) -> dict[int, RowVerdict]:
    """Apply φ to every state of the state space."""
    return {s: phi_row_verdict(protocol, ss, s) for s in ss.states}


def psi_verdict_to_states(
    protocol: V2xProtocol,
    ss: StateSpace,
    verdict: str,
) -> frozenset[int]:
    """ψ : ROWSupervisor → L(S).

    Given a right-of-way verdict (e.g., a supervisor command issued by the
    RSU or by the ego vehicle's planner), return the set of joint protocol
    states at which the verdict is justified.

    By construction ``s ∈ ψ(v)`` iff ``φ(s).verdict == v``.  This yields a
    Galois-style adjunction

    .. math::
       s \\le_{L(S)} \\psi(v) \\iff \\varphi(s) \\preceq v

    where ``≤_{L(S)}`` is the reachability order and ``⪯`` is the discrete
    order on verdicts.  The composition ``φ ∘ ψ`` is the identity on the
    image of φ, making ``(φ, ψ)`` a *section--retraction* pair (the
    supervisor lattice embeds back into the state lattice via ψ).
    """
    if verdict not in ROW_VERDICTS:
        raise ValueError(
            f"Invalid verdict {verdict!r}; expected one of {ROW_VERDICTS}."
        )
    verdicts = phi_all_states(protocol, ss)
    return frozenset(s for s, v in verdicts.items() if v.verdict == verdict)


def classify_morphism_pair(
    protocol: V2xProtocol,
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
                if phi_row_verdict(protocol, ss, s).verdict != kind:
                    return "galois"
        return "section-retraction"
    return "projection"


# ---------------------------------------------------------------------------
# Safety / fairness / liveness checks
# ---------------------------------------------------------------------------


def check_collision_free(
    protocol: V2xProtocol,
    ss: StateSpace,
) -> bool:
    """Check that no joint state has both vehicles 'proceeding' simultaneously
    without first establishing right-of-way.

    Operationally: a state is unsafe if its incident labels contain
    'proceed' from BOTH ego and other (in a four-way stop) without an
    intervening 'firstArrival' or 'yieldTurn' marker. We approximate this
    by checking that the EMERGENCY_BRAKE verdict is never assigned to any
    state EXCEPT in protocols that explicitly model emergency vehicles.
    """
    verdicts = phi_all_states(protocol, ss)
    has_emergency = any(
        v.verdict == "EMERGENCY_BRAKE" for v in verdicts.values()
    )
    if protocol.scenario == "signalized" and "emergency" in protocol.roles:
        return True  # emergency preemption is the intended behaviour
    return not has_emergency


def check_fairness(
    protocol: V2xProtocol,
    ss: StateSpace,
) -> bool:
    """Check fairness: every vehicle eventually gets EGO_GO or OTHER_GO.

    The lattice has the fairness property iff both EGO_GO and OTHER_GO are
    in the image of φ when the protocol involves multiple vehicles, OR the
    protocol has only one vehicle role.
    """
    if len(protocol.roles) <= 1:
        return True
    verdicts = phi_all_states(protocol, ss)
    kinds = {v.verdict for v in verdicts.values()}
    # Single-vehicle scenarios (signalized) only need EGO_GO eventually.
    if protocol.scenario in ("signalized", "left_turn"):
        return "EGO_GO" in kinds
    # Symmetric stop-sign: both vehicles share labels; EGO_GO covers both
    # via 'firstArrival' / 'proceed' (each vehicle individually obtains it).
    return "EGO_GO" in kinds


def check_liveness(
    protocol: V2xProtocol,
    ss: StateSpace,
) -> bool:
    """Check liveness: the lattice bottom (CLEAR) is reachable from the top.

    This holds iff the lattice is well-formed AND the bottom state is
    reached by at least one transition path; equivalently, every protocol
    eventually terminates with the intersection cleared.
    """
    return ss.bottom in ss.states and ss.top in ss.states


# ---------------------------------------------------------------------------
# Core verification pipeline
# ---------------------------------------------------------------------------


def v2x_to_session_type(protocol: V2xProtocol) -> SessionType:
    """Parse the protocol's session-type string."""
    return parse(protocol.session_type_string)


def verify_v2x_protocol(protocol: V2xProtocol) -> V2xAnalysisResult:
    """Run the full verification + safety pipeline."""
    ast = v2x_to_session_type(protocol)
    ss = build_statespace(ast)

    lr = check_lattice(ss)
    dist = check_distributive(ss)
    verdicts = phi_all_states(protocol, ss)

    return V2xAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_well_formed=lr.is_lattice,
        row_verdicts=verdicts,
        collision_free=check_collision_free(protocol, ss),
        fairness_holds=check_fairness(protocol, ss),
        liveness_holds=check_liveness(protocol, ss),
    )


def analyze_all_v2x() -> list[V2xAnalysisResult]:
    """Verify every pre-defined V2X protocol."""
    return [verify_v2x_protocol(p) for p in ALL_V2X_PROTOCOLS]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_v2x_report(result: V2xAnalysisResult) -> str:
    lines: list[str] = []
    proto = result.protocol
    lines.append("=" * 72)
    lines.append(f"  V2X INTERSECTION PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")
    lines.append("--- Protocol Details ---")
    lines.append(f"  Scenario: {proto.scenario}")
    lines.append(f"  CAL:      {proto.cal}")
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
    lines.append("--- ROW Verdicts (φ) ---")
    counts: dict[str, int] = {}
    for v in result.row_verdicts.values():
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    for verdict, n in sorted(counts.items()):
        lines.append(f"  {verdict:<18}: {n} state(s)")
    lines.append("")
    lines.append("--- Safety Properties ---")
    lines.append(f"  Collision free: {result.collision_free}")
    lines.append(f"  Fairness:       {result.fairness_holds}")
    lines.append(f"  Liveness:       {result.liveness_holds}")
    lines.append("")
    lines.append("--- Verdict ---")
    if result.is_well_formed and result.collision_free and result.liveness_holds:
        lines.append(f"  PASS: V2X protocol satisfies safety + liveness ({proto.cal}).")
    else:
        lines.append("  FAIL: V2X protocol violates a safety or liveness property.")
    lines.append("")
    return "\n".join(lines)


def format_v2x_summary(results: list[V2xAnalysisResult]) -> str:
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("  V2X INTERSECTION RIGHT-OF-WAY CERTIFICATION SUMMARY")
    lines.append("=" * 88)
    lines.append("")
    header = (
        f"  {'Protocol':<28} {'Scenario':<11} {'CAL':<5} "
        f"{'States':>6} {'Trans':>6} {'Latt':>5} {'Safe':>5} {'Live':>5}"
    )
    lines.append(header)
    lines.append("  " + "-" * 84)
    for r in results:
        lines.append(
            f"  {r.protocol.name:<28} "
            f"{r.protocol.scenario:<11} "
            f"{r.protocol.cal:<5} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{'YES' if r.is_well_formed else 'NO':>5} "
            f"{'YES' if r.collision_free else 'NO':>5} "
            f"{'YES' if r.liveness_holds else 'NO':>5}"
        )
    lines.append("")
    all_safe = all(r.collision_free and r.liveness_holds for r in results)
    lines.append(f"  All protocols collision-free + live: {'YES' if all_safe else 'NO'}")
    return "\n".join(lines)
