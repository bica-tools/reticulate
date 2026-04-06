"""ROS 2 Navigation Stack as Session Types (Step 801).

Models the ROS 2 Nav2 stack -- nav2_planner, nav2_controller,
nav2_bt_navigator, nav2_recoveries, costmap_2d, AMCL -- as multiparty
session types. Action calls (NavigateToPose, ComputePathToPose,
FollowPath), tf2 transforms, and behaviour-tree fallbacks all map to
branch/select/parallel constructors.

The ``||`` constructor is used to model parallel sensor pipelines
(lidar, odometry, IMU) feeding the costmap; the resulting product
lattice ``L(S_lidar) x L(S_odom) x L(S_imu)`` compresses the
combinatorial sensor interleaving into a single sub-lattice indexed by
the state of each pipeline.

The lattice ``L(S)`` is connected to a ``NavSupervisor`` domain through
a *bidirectional pair*

.. math::
   \\varphi : L(S) \\to \\text{NavSupervisor}, \\qquad
   \\psi : \\text{NavSupervisor} \\to L(S).

``\\varphi`` ships every protocol state to the *supervisor mode* the
navigation stack should be in (NAVIGATING, REPLANNING, RECOVERING,
LOCALIZED, SENSOR_FAULT, EMERGENCY_STOP, IDLE).  ``\\psi`` takes a
supervisor mode and returns the set of protocol states in which that
mode is live, supporting deadlock detection in the behaviour tree,
recovery escalation, and sensor-failure mode transitions.

Usage::

    from reticulate.ros2_nav import (
        navigate_to_pose,
        compute_path_to_pose,
        follow_path,
        sensor_pipeline_parallel,
        bt_navigator_fallback,
        verify_ros2_protocol,
        analyze_all_ros2,
        phi_supervisor_mode,
        psi_mode_to_states,
    )
    proto = navigate_to_pose()
    result = verify_ros2_protocol(proto)
    print(format_ros2_report(result))
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


# Nav2 supervisor modes (the codomain of phi).
SUPERVISOR_MODES: tuple[str, ...] = (
    "IDLE",
    "LOCALIZED",
    "NAVIGATING",
    "REPLANNING",
    "RECOVERING",
    "SENSOR_FAULT",
    "EMERGENCY_STOP",
    "GOAL_REACHED",
)


# Recognised Nav2 nodes (parties in the multiparty session).
NAV2_NODES: tuple[str, ...] = (
    "nav2_planner",
    "nav2_controller",
    "nav2_bt_navigator",
    "nav2_recoveries",
    "costmap_2d",
    "amcl",
)


@dataclass(frozen=True)
class Ros2Protocol:
    """A ROS 2 Nav2 protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "NavigateToPose").
        layer: Layer ("ACTION", "BT", "SENSOR", "TF").
        nodes: Nav2 nodes (parties) involved.
        session_type_string: Session-type encoding.
        spec_reference: Nav2/ROS reference.
        description: Free-text description.
    """
    name: str
    layer: str
    nodes: tuple[str, ...]
    session_type_string: str
    spec_reference: str
    description: str

    def __post_init__(self) -> None:
        if self.layer not in ("ACTION", "BT", "SENSOR", "TF", "AMCL"):
            raise ValueError(
                f"Invalid layer {self.layer!r}; expected one of "
                f"ACTION, BT, SENSOR, TF, AMCL."
            )


@dataclass(frozen=True)
class SupervisorMode:
    """A Nav2 supervisor mode (codomain of phi).

    Attributes:
        kind: Mode label from SUPERVISOR_MODES.
        rationale: Human-readable justification.
    """
    kind: str
    rationale: str


@dataclass(frozen=True)
class Ros2AnalysisResult:
    """Complete analysis result for a ROS 2 Nav2 protocol."""
    protocol: Ros2Protocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    num_states: int
    num_transitions: int
    is_well_formed: bool
    supervisor_modes: dict[int, SupervisorMode]
    deadlock_states: tuple[int, ...]


# ---------------------------------------------------------------------------
# ROS 2 Nav2 protocol definitions
# ---------------------------------------------------------------------------


def navigate_to_pose() -> Ros2Protocol:
    """Top-level ``NavigateToPose`` action goal life-cycle.

    Models a goal accepted by ``nav2_bt_navigator``: the BT validates
    the goal, requests a plan from ``nav2_planner``, executes via
    ``nav2_controller``, and either reports success or escalates to
    recoveries.
    """
    return Ros2Protocol(
        name="NavigateToPose",
        layer="ACTION",
        nodes=("nav2_bt_navigator", "nav2_planner", "nav2_controller"),
        session_type_string=(
            "&{sendGoal: +{ACCEPTED: "
            "&{computePath: +{PATH_OK: "
            "&{followPath: +{REACHED: &{succeed: end}, "
            "BLOCKED: &{recover: +{CLEARED: &{succeed: end}, "
            "ABORT: &{abort: end}}}}}, "
            "NO_PATH: &{abort: end}}}, "
            "REJECTED: &{abort: end}}}"
        ),
        spec_reference="Nav2 Humble: NavigateToPose action",
        description=(
            "Top-level NavigateToPose: validate goal, compute path, "
            "follow path, recover on block or abort."
        ),
    )


def compute_path_to_pose() -> Ros2Protocol:
    """``ComputePathToPose`` planner action.

    The BT calls the planner, which queries the costmap, runs A*/
    Theta*, and either returns a path or reports failure.
    """
    return Ros2Protocol(
        name="ComputePathToPose",
        layer="ACTION",
        nodes=("nav2_planner", "costmap_2d"),
        session_type_string=(
            "&{requestPlan: +{COSTMAP_OK: "
            "&{queryCostmap: &{runPlanner: "
            "+{PATH_FOUND: &{returnPath: end}, "
            "NO_PATH: &{planFailure: end}}}}, "
            "COSTMAP_STALE: &{planFailure: end}}}"
        ),
        spec_reference="Nav2 Humble: ComputePathToPose action",
        description=(
            "Planner action: query costmap, run A*/Theta*, return path "
            "or planner failure."
        ),
    )


def follow_path() -> Ros2Protocol:
    """``FollowPath`` controller action.

    The controller (DWB / RPP / MPPI) consumes a path, computes
    velocity commands, and either reaches the goal, gets blocked, or
    detects a collision.
    """
    return Ros2Protocol(
        name="FollowPath",
        layer="ACTION",
        nodes=("nav2_controller", "costmap_2d"),
        session_type_string=(
            "&{loadPath: &{computeVelocity: "
            "+{COMMAND_OK: &{publishCmdVel: "
            "+{GOAL_REACHED: &{succeed: end}, "
            "OBSTACLE: &{stop: &{controllerFailure: end}}}}, "
            "INFEASIBLE: &{controllerFailure: end}}}}"
        ),
        spec_reference="Nav2 Humble: FollowPath action (DWB/MPPI)",
        description=(
            "Controller action: load path, compute velocity, publish "
            "cmd_vel until goal reached, blocked, or infeasible."
        ),
    )


def sensor_pipeline_parallel() -> Ros2Protocol:
    """Parallel sensor pipelines (lidar || odom || imu) feeding the costmap.

    Each sensor independently advances through ``read -> filter ->
    publish``; the parallel composition forms the product lattice
    ``L(lidar) x L(odom) x L(imu)``.  This is the load-bearing example
    for product compression of sensor interleaving.
    """
    return Ros2Protocol(
        name="SensorPipelineParallel",
        layer="SENSOR",
        nodes=("costmap_2d",),
        session_type_string=(
            "((&{lidarRead: &{lidarFilter: &{lidarPublish: end}}} "
            "|| &{odomRead: &{odomFilter: &{odomPublish: end}}}) "
            "|| &{imuRead: &{imuFilter: &{imuPublish: end}}})"
        ),
        spec_reference="Nav2 Humble: costmap_2d sensor sources",
        description=(
            "Three sensor pipelines run in parallel and feed the "
            "costmap. The product lattice compresses the 6! = 720 "
            "interleavings into 4*4*4 = 64 product states."
        ),
    )


def bt_navigator_fallback() -> Ros2Protocol:
    """Behaviour-tree fallback / recovery escalation.

    The BT first attempts the primary navigation pipeline; on failure
    it tries clearing the local costmap, then the global costmap,
    then a spin/back-up recovery, then aborts.  This is the canonical
    Nav2 ``RecoveryNode`` chain.
    """
    return Ros2Protocol(
        name="BtNavigatorFallback",
        layer="BT",
        nodes=("nav2_bt_navigator", "nav2_recoveries"),
        session_type_string=(
            "&{tryPrimary: +{OK: &{succeed: end}, "
            "FAIL: &{clearLocal: +{OK: &{retry: &{succeed: end}}, "
            "FAIL: &{clearGlobal: +{OK: &{retry: &{succeed: end}}, "
            "FAIL: &{spinRecovery: +{OK: &{retry: &{succeed: end}}, "
            "FAIL: &{abort: end}}}}}}}}}"
        ),
        spec_reference="Nav2 Humble: behavior_tree_xml RecoveryNode",
        description=(
            "BT recovery escalation: primary -> clearLocal -> "
            "clearGlobal -> spinRecovery -> abort. Each level retries "
            "the primary on success."
        ),
    )


def amcl_localization() -> Ros2Protocol:
    """AMCL localization life-cycle.

    AMCL initialises from a pose estimate, fuses laser scans, and
    either converges or reports a localisation failure.
    """
    return Ros2Protocol(
        name="AmclLocalization",
        layer="AMCL",
        nodes=("amcl",),
        session_type_string=(
            "&{initPose: &{laserUpdate: "
            "+{CONVERGED: &{publishTransform: end}, "
            "DIVERGED: &{relocalize: "
            "+{CONVERGED: &{publishTransform: end}, "
            "FAILED: &{localizationFailure: end}}}}}}"
        ),
        spec_reference="Nav2 Humble: amcl node (KLD-sampling)",
        description=(
            "AMCL: initialise pose, fuse laser scans, converge or "
            "relocalize; report failure if relocalization diverges."
        ),
    )


def tf2_transform_chain() -> Ros2Protocol:
    """tf2 transform lookup chain ``map -> odom -> base_link``."""
    return Ros2Protocol(
        name="Tf2TransformChain",
        layer="TF",
        nodes=("nav2_bt_navigator",),
        session_type_string=(
            "&{lookupMapOdom: +{OK: "
            "&{lookupOdomBase: +{OK: &{composeTransform: end}, "
            "TIMEOUT: &{tfFailure: end}}}, "
            "TIMEOUT: &{tfFailure: end}}}"
        ),
        spec_reference="ROS 2 Humble: tf2 lookupTransform",
        description=(
            "tf2 chain: lookup map->odom and odom->base_link, compose "
            "to map->base_link or report tf failure."
        ),
    )


def recovery_spin() -> Ros2Protocol:
    """``Spin`` recovery behaviour."""
    return Ros2Protocol(
        name="RecoverySpin",
        layer="ACTION",
        nodes=("nav2_recoveries",),
        session_type_string=(
            "&{startSpin: &{rotate: "
            "+{COMPLETE: &{succeed: end}, "
            "BLOCKED: &{abort: end}}}}"
        ),
        spec_reference="Nav2 Humble: Spin recovery",
        description=(
            "Spin recovery: rotate the robot in place to clear local "
            "costmap, succeed or abort if blocked."
        ),
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ALL_ACTION_PROTOCOLS: tuple[Ros2Protocol, ...] = (
    navigate_to_pose(),
    compute_path_to_pose(),
    follow_path(),
    recovery_spin(),
)

ALL_BT_PROTOCOLS: tuple[Ros2Protocol, ...] = (
    bt_navigator_fallback(),
)

ALL_SENSOR_PROTOCOLS: tuple[Ros2Protocol, ...] = (
    sensor_pipeline_parallel(),
)

ALL_TF_PROTOCOLS: tuple[Ros2Protocol, ...] = (
    tf2_transform_chain(),
)

ALL_AMCL_PROTOCOLS: tuple[Ros2Protocol, ...] = (
    amcl_localization(),
)

ALL_ROS2_PROTOCOLS: tuple[Ros2Protocol, ...] = (
    ALL_ACTION_PROTOCOLS
    + ALL_BT_PROTOCOLS
    + ALL_SENSOR_PROTOCOLS
    + ALL_TF_PROTOCOLS
    + ALL_AMCL_PROTOCOLS
)


# ---------------------------------------------------------------------------
# Bidirectional morphisms phi : L(S) -> NavSupervisor and psi back
# ---------------------------------------------------------------------------

# Keyword catalogue mapping transition labels to supervisor modes. Order
# matters: earlier rules win.
_MODE_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("abort", "controllerFailure", "planFailure",
      "localizationFailure", "tfFailure"), "EMERGENCY_STOP"),
    (("recover", "spinRecovery", "clearLocal", "clearGlobal",
      "relocalize", "retry"), "RECOVERING"),
    (("succeed", "GOAL_REACHED", "publishTransform"), "GOAL_REACHED"),
    (("computePath", "queryCostmap", "runPlanner", "requestPlan"),
     "REPLANNING"),
    (("followPath", "publishCmdVel", "computeVelocity", "loadPath",
      "rotate", "startSpin"), "NAVIGATING"),
    (("initPose", "laserUpdate", "lookupMapOdom",
      "lookupOdomBase", "composeTransform"), "LOCALIZED"),
    (("lidar", "odom", "imu"), "SENSOR_FAULT"),
)


def phi_supervisor_mode(
    protocol: Ros2Protocol,
    ss: StateSpace,
    state: int,
) -> SupervisorMode:
    """phi : L(S) -> NavSupervisor (state-indexed).

    Maps a protocol state to the dominant Nav2 supervisor mode.  Bottom
    is GOAL_REACHED (success terminus); top is IDLE (no work yet);
    intermediate states are classified by their incident transition
    labels.
    """
    if state == ss.bottom:
        return SupervisorMode(
            kind="GOAL_REACHED",
            rationale=(
                f"State {state} is the lattice bottom; the navigation "
                f"goal has terminated successfully (or failed cleanly)."
            ),
        )
    if state == ss.top:
        return SupervisorMode(
            kind="IDLE",
            rationale=(
                f"State {state} is the lattice top; no navigation work "
                f"has yet been issued."
            ),
        )

    incident_labels: list[str] = []
    for src, lbl, tgt in ss.transitions:
        if src == state or tgt == state:
            incident_labels.append(lbl)

    for keywords, kind in _MODE_KEYWORDS:
        for kw in keywords:
            if any(kw in lbl for lbl in incident_labels):
                return SupervisorMode(
                    kind=kind,
                    rationale=(
                        f"State {state} mapped to {kind} via incident "
                        f"label containing {kw!r}."
                    ),
                )

    return SupervisorMode(
        kind="NAVIGATING",
        rationale=f"State {state} has no distinguishing label; default NAVIGATING.",
    )


def phi_all_states(
    protocol: Ros2Protocol,
    ss: StateSpace,
) -> dict[int, SupervisorMode]:
    """Apply phi to every state of the state space."""
    return {s: phi_supervisor_mode(protocol, ss, s) for s in ss.states}


def psi_mode_to_states(
    protocol: Ros2Protocol,
    ss: StateSpace,
    mode_kind: str,
) -> frozenset[int]:
    """psi : NavSupervisor -> L(S).

    Given a supervisor mode kind, return the set of protocol states in
    which that mode is live.  By construction ``s in psi(m)`` iff
    ``phi(s).kind == m.kind``, yielding a section--retraction pair.
    """
    modes = phi_all_states(protocol, ss)
    return frozenset(s for s, m in modes.items() if m.kind == mode_kind)


def classify_morphism_pair(
    protocol: Ros2Protocol,
    ss: StateSpace,
) -> str:
    """Classify the pair (phi, psi).

    Returns one of: 'isomorphism', 'embedding', 'projection',
    'galois', 'section-retraction'.
    """
    modes = phi_all_states(protocol, ss)
    kinds = {m.kind for m in modes.values()}
    if len(kinds) == len(ss.states):
        return "isomorphism"
    if len(kinds) < len(ss.states):
        for kind in kinds:
            states = psi_mode_to_states(protocol, ss, kind)
            for s in states:
                if phi_supervisor_mode(protocol, ss, s).kind != kind:
                    return "galois"
        return "section-retraction"
    return "projection"


# ---------------------------------------------------------------------------
# Deadlock detection in the behaviour tree
# ---------------------------------------------------------------------------


def detect_deadlocks(ss: StateSpace) -> tuple[int, ...]:
    """Detect deadlocked states in the BT state space.

    A state is *deadlocked* iff it has no outgoing transitions and is
    not the lattice bottom.  Such states correspond to BT nodes that
    can never report SUCCESS or FAILURE -- the canonical Nav2 freeze.
    """
    has_outgoing: set[int] = {src for src, _, _ in ss.transitions}
    return tuple(
        sorted(
            s for s in ss.states
            if s not in has_outgoing and s != ss.bottom
        )
    )


# ---------------------------------------------------------------------------
# Core verification pipeline
# ---------------------------------------------------------------------------


def ros2_to_session_type(protocol: Ros2Protocol) -> SessionType:
    """Parse the protocol's session-type string."""
    return parse(protocol.session_type_string)


def verify_ros2_protocol(protocol: Ros2Protocol) -> Ros2AnalysisResult:
    """Run the full verification pipeline for a single protocol."""
    ast = ros2_to_session_type(protocol)
    ss = build_statespace(ast)
    lr = check_lattice(ss)
    dist = check_distributive(ss)
    modes = phi_all_states(protocol, ss)
    deadlocks = detect_deadlocks(ss)
    return Ros2AnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_well_formed=lr.is_lattice,
        supervisor_modes=modes,
        deadlock_states=deadlocks,
    )


def analyze_all_ros2() -> list[Ros2AnalysisResult]:
    """Verify every pre-defined ROS 2 protocol."""
    return [verify_ros2_protocol(p) for p in ALL_ROS2_PROTOCOLS]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_ros2_report(result: Ros2AnalysisResult) -> str:
    lines: list[str] = []
    proto = result.protocol
    lines.append("=" * 72)
    lines.append(f"  ROS 2 NAV2 PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")
    lines.append("--- Protocol Details ---")
    lines.append(f"  Layer: {proto.layer}")
    lines.append(f"  Spec:  {proto.spec_reference}")
    for n in proto.nodes:
        lines.append(f"  Node:  {n}")
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
    lines.append("--- Supervisor Modes (phi) ---")
    counts: dict[str, int] = {}
    for m in result.supervisor_modes.values():
        counts[m.kind] = counts.get(m.kind, 0) + 1
    for kind, n in sorted(counts.items()):
        lines.append(f"  {kind:<18}: {n} state(s)")
    lines.append("")
    lines.append("--- Deadlock Detection ---")
    if result.deadlock_states:
        lines.append(
            f"  WARN: {len(result.deadlock_states)} deadlocked state(s): "
            f"{result.deadlock_states}"
        )
    else:
        lines.append("  OK: no deadlocks detected.")
    lines.append("")
    lines.append("--- Verdict ---")
    if result.is_well_formed and not result.deadlock_states:
        lines.append("  PASS: lattice well-formed and deadlock-free.")
    else:
        lines.append("  FAIL: lattice or deadlock check failed.")
    lines.append("")
    return "\n".join(lines)


def format_ros2_summary(results: list[Ros2AnalysisResult]) -> str:
    lines: list[str] = []
    lines.append("=" * 82)
    lines.append("  ROS 2 NAV2 STACK CERTIFICATION SUMMARY")
    lines.append("=" * 82)
    lines.append("")
    header = (
        f"  {'Protocol':<26} {'Layer':<8} "
        f"{'States':>6} {'Trans':>6} {'Lattice':>8} {'Dist':>6} {'Deadlk':>7}"
    )
    lines.append(header)
    lines.append("  " + "-" * 78)
    for r in results:
        lines.append(
            f"  {r.protocol.name:<26} "
            f"{r.protocol.layer:<8} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{'YES' if r.is_well_formed else 'NO':>8} "
            f"{'YES' if r.distributivity.is_distributive else 'NO':>6} "
            f"{len(r.deadlock_states):>7}"
        )
    lines.append("")
    all_lattice = all(r.is_well_formed for r in results)
    lines.append(f"  All protocols form lattices: {'YES' if all_lattice else 'NO'}")
    return "\n".join(lines)
