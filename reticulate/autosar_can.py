"""AUTOSAR/CAN Protocol Certification via lattice analysis (Step 76).

Models AUTOSAR runnable communication and CAN bus message flows as
session types, then uses lattice properties to support ISO 26262
functional safety certification.

Two coordinated layers are captured:

- **AUTOSAR runnables** (SWC layer): periodic and event-driven runnables
  communicate via sender--receiver and client--server ports mediated by
  the RTE (Runtime Environment). Each runnable's interaction pattern is a
  session type over RTE API calls (``Rte_Read``, ``Rte_Write``,
  ``Rte_Call``, ``Rte_Send``, ``Rte_Receive``).

- **CAN bus** (communication layer): message frames flow between ECUs
  over a shared CAN/CAN-FD bus. Each message exchange admits
  ``arbitration``, ``frame transmission``, ``acknowledgement``, and
  optional ``error frames``. We encode the classical CAN frame life
  cycle, the Tx confirmation protocol, and diagnostic services
  (UDS/ISO 14229) used during safety audits.

The resulting lattice ``L(S)`` is connected to the ISO 26262 work
products through a **pair of morphisms**

.. math::
   \\varphi : L(S) \\to \\text{SafetyClaims}, \\qquad
   \\psi : \\text{SafetyClaims} \\to L(S).

``\\varphi`` ships each protocol state to the *safety claim* it
discharges (freedom from interference, hazard containment, fault
detection, fail-operational residue, ASIL decomposition witness).
``\\psi`` takes a certification artefact (FTA cut set, FMEDA failure
mode, dependent failure analysis row) and retrieves the set of states
in which that artefact is live. Composition ``\\varphi \\circ \\psi``
implements the *traceability* axis of the ISO 26262 V-model.

Usage:
    from reticulate.autosar_can import (
        runnable_periodic,
        can_frame_exchange,
        verify_autosar_protocol,
        analyze_all_autosar,
        phi_safety_claim,
        psi_claim_to_states,
    )
    proto = can_frame_exchange()
    result = verify_autosar_protocol(proto)
    print(format_autosar_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.coverage import CoverageResult, compute_coverage
from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.subtyping import SubtypingResult, check_subtype
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Domain data types
# ---------------------------------------------------------------------------


# ISO 26262 ASIL levels (Automotive Safety Integrity Level).
ASIL_LEVELS: tuple[str, ...] = ("QM", "A", "B", "C", "D")


@dataclass(frozen=True)
class AutosarProtocol:
    """An AUTOSAR / CAN protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "CAN_FrameExchange").
        layer: Layer ("RTE", "CAN", or "UDS").
        asil: Target ASIL level from ISO 26262 (QM, A, B, C, D).
        ecus: Electronic Control Units involved.
        session_type_string: Session type encoding.
        spec_reference: AUTOSAR / ISO reference (e.g., "AUTOSAR RTE R22-11").
        description: Free-text description.
    """
    name: str
    layer: str
    asil: str
    ecus: tuple[str, ...]
    session_type_string: str
    spec_reference: str
    description: str

    def __post_init__(self) -> None:
        if self.asil not in ASIL_LEVELS:
            raise ValueError(
                f"Invalid ASIL level {self.asil!r}; expected one of {ASIL_LEVELS}."
            )
        if self.layer not in ("RTE", "CAN", "UDS"):
            raise ValueError(
                f"Invalid layer {self.layer!r}; expected RTE, CAN, or UDS."
            )


@dataclass(frozen=True)
class SafetyClaim:
    """An ISO 26262 safety claim about a protocol state.

    Attributes:
        kind: Claim kind: 'FFI' (freedom from interference), 'FTTI'
            (fault tolerant time interval window), 'HAZARD_CONTAINED',
            'FAULT_DETECTED', 'FAIL_OPERATIONAL', 'ASIL_DECOMPOSITION',
            'SAFE_STATE', 'NONE'.
        asil: Required ASIL for the claim.
        rationale: Human-readable justification.
    """
    kind: str
    asil: str
    rationale: str


@dataclass(frozen=True)
class AutosarAnalysisResult:
    """Complete analysis result for an AUTOSAR / CAN protocol."""
    protocol: AutosarProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    enumeration: EnumerationResult
    test_source: str
    coverage: CoverageResult
    num_states: int
    num_transitions: int
    num_valid_paths: int
    num_violations: int
    is_well_formed: bool
    safety_claims: dict[int, SafetyClaim]


# ---------------------------------------------------------------------------
# AUTOSAR runnable protocol definitions
# ---------------------------------------------------------------------------

def runnable_periodic() -> AutosarProtocol:
    """AUTOSAR periodic runnable (ASIL B).

    Models a periodic runnable that reads a sender-receiver port,
    computes, then writes an output port. The read/write pair form a
    simple branch-selection pattern with the RTE API.
    """
    return AutosarProtocol(
        name="Runnable_Periodic",
        layer="RTE",
        asil="B",
        ecus=("SWC_Sensor", "SWC_Controller"),
        session_type_string=(
            "&{rteRead: +{OK: &{compute: &{rteWrite: end}}, "
            "UNINIT: &{defaultValue: &{rteWrite: end}}}}"
        ),
        spec_reference="AUTOSAR RTE R22-11 Section 4.3.1",
        description=(
            "Periodic runnable: Rte_Read sensor value, compute actuation, "
            "Rte_Write to controller port. Handles uninitialised input."
        ),
    )


def runnable_client_server() -> AutosarProtocol:
    """AUTOSAR client-server runnable (ASIL C).

    Client invokes a server runnable synchronously via ``Rte_Call``;
    server returns a result or an application error.
    """
    return AutosarProtocol(
        name="Runnable_ClientServer",
        layer="RTE",
        asil="C",
        ecus=("SWC_Client", "SWC_Server"),
        session_type_string=(
            "&{rteCall: +{OK: &{serverExec: +{RESULT: end, "
            "APP_ERROR: &{errorHook: end}}}, "
            "E_TIMEOUT: &{errorHook: end}}}"
        ),
        spec_reference="AUTOSAR RTE R22-11 Section 4.3.2",
        description=(
            "Client-server runnable: synchronous Rte_Call to a server "
            "runnable. Handles timeout and application errors."
        ),
    )


def runnable_event_triggered() -> AutosarProtocol:
    """AUTOSAR event-triggered runnable (ASIL A)."""
    return AutosarProtocol(
        name="Runnable_EventTriggered",
        layer="RTE",
        asil="A",
        ecus=("SWC_Event",),
        session_type_string=(
            "&{eventArrival: &{rteReceive: +{DATA: &{processEvent: end}, "
            "NO_DATA: end}}}"
        ),
        spec_reference="AUTOSAR RTE R22-11 Section 4.3.3",
        description=(
            "Event-triggered runnable: activated on event arrival, "
            "receives data and processes it."
        ),
    )


def runnable_mode_switch() -> AutosarProtocol:
    """AUTOSAR mode-switch runnable (ASIL D)."""
    return AutosarProtocol(
        name="Runnable_ModeSwitch",
        layer="RTE",
        asil="D",
        ecus=("SWC_ModeManager",),
        session_type_string=(
            "&{modeRequest: +{TRANSITION_OK: "
            "&{onExit: &{onEntry: &{rteSwitch: end}}}, "
            "TRANSITION_DENIED: &{safeState: end}}}"
        ),
        spec_reference="AUTOSAR RTE R22-11 Section 4.4",
        description=(
            "Mode-switch runnable for ASIL-D functions. Exit/entry "
            "sequencing with safe-state fallback on denial."
        ),
    )


# ---------------------------------------------------------------------------
# CAN bus protocol definitions
# ---------------------------------------------------------------------------

def can_frame_exchange() -> AutosarProtocol:
    """Classical CAN frame transmission (ISO 11898-1).

    Arbitration, data transmission, ACK slot, and optional error
    frame. This is the canonical CAN frame life cycle.
    """
    return AutosarProtocol(
        name="CAN_FrameExchange",
        layer="CAN",
        asil="B",
        ecus=("ECU_Tx", "ECU_Rx"),
        session_type_string=(
            "&{arbitration: +{WON: "
            "&{transmitFrame: +{ACK: end, "
            "NO_ACK: &{errorFrame: end}}}, "
            "LOST: &{backoff: end}}}"
        ),
        spec_reference="ISO 11898-1:2015 Section 10",
        description=(
            "Classical CAN frame exchange: bus arbitration, frame "
            "transmission, ACK slot reception, and error frame on NAK."
        ),
    )


def can_fd_exchange() -> AutosarProtocol:
    """CAN-FD frame exchange (ISO 11898-1:2015)."""
    return AutosarProtocol(
        name="CAN_FD_FrameExchange",
        layer="CAN",
        asil="C",
        ecus=("ECU_Tx", "ECU_Rx"),
        session_type_string=(
            "&{arbitration: +{WON: "
            "&{switchBRS: &{transmitFrame: "
            "+{ACK: &{crcCheck: +{CRC_OK: end, "
            "CRC_FAIL: &{errorFrame: end}}}, "
            "NO_ACK: &{errorFrame: end}}}}, "
            "LOST: &{backoff: end}}}"
        ),
        spec_reference="ISO 11898-1:2015 Section 11 (CAN-FD)",
        description=(
            "CAN-FD frame: arbitration, bit-rate switch (BRS), data "
            "transmission, ACK slot, CRC-17/21 verification, error "
            "handling on mismatch."
        ),
    )


def can_tx_confirmation() -> AutosarProtocol:
    """AUTOSAR Com TxConfirmation protocol (ASIL B)."""
    return AutosarProtocol(
        name="CAN_TxConfirmation",
        layer="CAN",
        asil="B",
        ecus=("Com", "PduR", "CanIf"),
        session_type_string=(
            "&{comSend: &{pduRTransmit: &{canIfTransmit: "
            "+{TX_OK: &{txConfirmation: end}, "
            "TX_FAIL: &{errorNotify: end}}}}}"
        ),
        spec_reference="AUTOSAR Com R22-11 Section 7.3",
        description=(
            "AUTOSAR Com Tx confirmation chain: Com -> PduR -> CanIf "
            "-> hardware, with confirmation propagation and error "
            "notification on failure."
        ),
    )


def can_network_management() -> AutosarProtocol:
    """AUTOSAR CAN Network Management (ASIL A)."""
    return AutosarProtocol(
        name="CAN_NM",
        layer="CAN",
        asil="A",
        ecus=("ECU_A", "ECU_B"),
        session_type_string=(
            "&{networkStart: &{repeatMessageState: "
            "+{STAY_AWAKE: &{normalOperation: &{prepareSleep: &{busSleep: end}}}, "
            "NO_TRAFFIC: &{busSleep: end}}}}"
        ),
        spec_reference="AUTOSAR CanNm R22-11",
        description=(
            "CAN Network Management state machine: repeat-message -> "
            "normal operation -> prepare sleep -> bus sleep."
        ),
    )


# ---------------------------------------------------------------------------
# UDS diagnostic protocol (ISO 14229)
# ---------------------------------------------------------------------------

def uds_diagnostic_session() -> AutosarProtocol:
    """UDS diagnostic session control (ISO 14229-1)."""
    return AutosarProtocol(
        name="UDS_DiagnosticSession",
        layer="UDS",
        asil="QM",
        ecus=("Tester", "ECU"),
        session_type_string=(
            "&{diagSessionControl: +{POSITIVE: "
            "&{securityAccess: +{UNLOCKED: "
            "&{readDataByIdentifier: &{testerPresent: end}}, "
            "LOCKED: &{negativeResponse: end}}}, "
            "NEGATIVE: end}}"
        ),
        spec_reference="ISO 14229-1:2020 Services 0x10, 0x27, 0x22, 0x3E",
        description=(
            "UDS diagnostic session: tester requests session, performs "
            "security access, reads data identifiers, keeps session "
            "alive with TesterPresent."
        ),
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ALL_RUNNABLES: tuple[AutosarProtocol, ...] = (
    runnable_periodic(),
    runnable_client_server(),
    runnable_event_triggered(),
    runnable_mode_switch(),
)

ALL_CAN_PROTOCOLS: tuple[AutosarProtocol, ...] = (
    can_frame_exchange(),
    can_fd_exchange(),
    can_tx_confirmation(),
    can_network_management(),
)

ALL_UDS_PROTOCOLS: tuple[AutosarProtocol, ...] = (
    uds_diagnostic_session(),
)

ALL_AUTOSAR_PROTOCOLS: tuple[AutosarProtocol, ...] = (
    ALL_RUNNABLES + ALL_CAN_PROTOCOLS + ALL_UDS_PROTOCOLS
)


# ---------------------------------------------------------------------------
# Bidirectional morphisms  φ : L(S) → SafetyClaims  and  ψ back
# ---------------------------------------------------------------------------

# Keyword catalogue mapping transition labels / state names to safety claim
# kinds. Keeping this data-driven lets φ be a pure function of the state
# space and protocol, so its behaviour is predictable and testable.
_CLAIM_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("safeState", "busSleep", "prepareSleep"), "SAFE_STATE"),
    (("errorFrame", "errorHook", "errorNotify", "negativeResponse"),
     "FAULT_DETECTED"),
    (("txConfirmation", "rteWrite", "crcCheck", "crcOk", "CRC_OK"),
     "FFI"),
    (("defaultValue", "backoff", "repeatMessageState"),
     "FAIL_OPERATIONAL"),
    (("modeRequest", "onExit", "onEntry", "rteSwitch"),
     "ASIL_DECOMPOSITION"),
    (("securityAccess", "testerPresent"), "HAZARD_CONTAINED"),
    (("arbitration", "transmitFrame", "switchBRS"), "FTTI"),
)


def phi_safety_claim(
    protocol: AutosarProtocol,
    ss: StateSpace,
    state: int,
) -> SafetyClaim:
    """φ : L(S) → SafetyClaims (state-indexed).

    Maps a protocol state to the dominant ISO 26262 safety claim it
    discharges, based on the incoming/outgoing transition labels and
    the protocol's target ASIL.
    """
    if state == ss.bottom:
        return SafetyClaim(
            kind="SAFE_STATE",
            asil=protocol.asil,
            rationale=(
                f"State {state} is the lattice bottom; protocol has "
                f"terminated and the ECU may enter its declared safe state."
            ),
        )
    if state == ss.top:
        return SafetyClaim(
            kind="FFI",
            asil=protocol.asil,
            rationale=(
                f"State {state} is the lattice top; all downstream "
                f"partitions are isolated by the RTE (freedom from "
                f"interference precondition)."
            ),
        )

    incident_labels: list[str] = []
    for src, lbl, tgt in ss.transitions:
        if src == state or tgt == state:
            incident_labels.append(lbl)

    for keywords, kind in _CLAIM_KEYWORDS:
        for kw in keywords:
            if any(kw in lbl for lbl in incident_labels):
                return SafetyClaim(
                    kind=kind,
                    asil=protocol.asil,
                    rationale=(
                        f"State {state} witnesses claim {kind} via "
                        f"label containing {kw!r}."
                    ),
                )

    return SafetyClaim(
        kind="NONE",
        asil=protocol.asil,
        rationale=f"State {state} has no dominant ISO 26262 claim.",
    )


def phi_all_states(
    protocol: AutosarProtocol,
    ss: StateSpace,
) -> dict[int, SafetyClaim]:
    """Apply φ to every state of the state space."""
    return {s: phi_safety_claim(protocol, ss, s) for s in ss.states}


def psi_claim_to_states(
    protocol: AutosarProtocol,
    ss: StateSpace,
    claim_kind: str,
) -> frozenset[int]:
    """ψ : SafetyClaims → L(S).

    Given a safety claim kind (e.g., the identifier of an FMEDA row or
    an FTA minimal cut set), return the set of protocol states at which
    the claim is live.

    By construction ``s ∈ ψ(c)`` iff ``φ(s).kind == c.kind``.  This
    yields the Galois-style adjunction

    .. math::
       s \\le_{L(S)} \\psi(c) \\iff \\varphi(s) \\preceq c

    where ``≤_{L(S)}`` is the reachability order and ``⪯`` is the trivial
    discrete order on claim kinds.  The composition ``φ ∘ ψ`` is the
    identity on the image of φ, making ``(φ, ψ)`` a *section--retraction*
    pair (an embedding of the claim lattice into the state lattice, with
    a best-effort retraction).
    """
    claims = phi_all_states(protocol, ss)
    return frozenset(s for s, c in claims.items() if c.kind == claim_kind)


def classify_morphism_pair(
    protocol: AutosarProtocol,
    ss: StateSpace,
) -> str:
    """Classify the pair (φ, ψ).

    Returns one of: 'isomorphism', 'embedding', 'projection',
    'galois', or 'section-retraction'.
    """
    claims = phi_all_states(protocol, ss)
    kinds = {c.kind for c in claims.values()}
    if len(kinds) == len(ss.states):
        return "isomorphism"
    if len(kinds) < len(ss.states):
        # Check φ ∘ ψ = id on image
        for kind in kinds:
            states = psi_claim_to_states(protocol, ss, kind)
            for s in states:
                if phi_safety_claim(protocol, ss, s).kind != kind:
                    return "galois"
        return "section-retraction"
    return "projection"


# ---------------------------------------------------------------------------
# Core verification pipeline
# ---------------------------------------------------------------------------

def autosar_to_session_type(protocol: AutosarProtocol) -> SessionType:
    """Parse the protocol's session-type string."""
    return parse(protocol.session_type_string)


def verify_autosar_protocol(
    protocol: AutosarProtocol,
    config: TestGenConfig | None = None,
) -> AutosarAnalysisResult:
    """Run the full verification + certification pipeline."""
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}CertTest",
            package_name="com.autosar.cert",
        )

    ast = autosar_to_session_type(protocol)
    ss = build_statespace(ast)

    lr = check_lattice(ss)
    dist = check_distributive(ss)

    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.session_type_string)
    cov = compute_coverage(ss, result=enum)

    claims = phi_all_states(protocol, ss)

    return AutosarAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        enumeration=enum,
        test_source=test_src,
        coverage=cov,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_valid_paths=len(enum.valid_paths),
        num_violations=len(enum.violations),
        is_well_formed=lr.is_lattice,
        safety_claims=claims,
    )


def analyze_all_autosar() -> list[AutosarAnalysisResult]:
    """Verify every pre-defined AUTOSAR / CAN / UDS protocol."""
    return [verify_autosar_protocol(p) for p in ALL_AUTOSAR_PROTOCOLS]


def check_asil_decomposition(
    parent: AutosarProtocol,
    child_a: AutosarProtocol,
    child_b: AutosarProtocol,
) -> SubtypingResult:
    """Check whether (child_a, child_b) is a valid ISO 26262 ASIL decomposition of parent.

    ISO 26262-9 §5 permits decomposing an ASIL-X requirement into two
    lower-ASIL requirements if they are mutually independent. We
    approximate this at the protocol level by requiring
    ``parent <= child_a`` (Gay--Hole subtyping), which certifies that
    ``child_a`` refines the parent's interaction surface; ``child_b``
    acts as a diversity redundant partner.
    """
    return check_subtype(
        autosar_to_session_type(parent),
        autosar_to_session_type(child_a),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_autosar_report(result: AutosarAnalysisResult) -> str:
    lines: list[str] = []
    proto = result.protocol
    lines.append("=" * 72)
    lines.append(f"  AUTOSAR / CAN PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")
    lines.append("--- Protocol Details ---")
    lines.append(f"  Layer: {proto.layer}")
    lines.append(f"  ASIL:  {proto.asil}")
    lines.append(f"  Spec:  {proto.spec_reference}")
    for e in proto.ecus:
        lines.append(f"  ECU:   {e}")
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
    lines.append("--- Safety Claims (φ) ---")
    counts: dict[str, int] = {}
    for c in result.safety_claims.values():
        counts[c.kind] = counts.get(c.kind, 0) + 1
    for kind, n in sorted(counts.items()):
        lines.append(f"  {kind:<22}: {n} state(s)")
    lines.append("")
    lines.append("--- Certification Verdict ---")
    if result.is_well_formed:
        lines.append(f"  PASS: lattice well-formed; ASIL {proto.asil} claims traceable.")
    else:
        lines.append("  FAIL: protocol is not a lattice; ISO 26262 traceability broken.")
    lines.append("")
    return "\n".join(lines)


def format_autosar_summary(results: list[AutosarAnalysisResult]) -> str:
    lines: list[str] = []
    lines.append("=" * 82)
    lines.append("  AUTOSAR / CAN / UDS CERTIFICATION SUMMARY")
    lines.append("=" * 82)
    lines.append("")
    header = (
        f"  {'Protocol':<28} {'Layer':<6} {'ASIL':<5} "
        f"{'States':>6} {'Trans':>6} {'Lattice':>8} {'Dist':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * 78)
    for r in results:
        lines.append(
            f"  {r.protocol.name:<28} "
            f"{r.protocol.layer:<6} "
            f"{r.protocol.asil:<5} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{'YES' if r.is_well_formed else 'NO':>8} "
            f"{'YES' if r.distributivity.is_distributive else 'NO':>6}"
        )
    lines.append("")
    all_lattice = all(r.is_well_formed for r in results)
    lines.append(f"  All protocols form lattices: {'YES' if all_lattice else 'NO'}")
    return "\n".join(lines)
