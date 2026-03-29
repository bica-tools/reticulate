"""5G NAS/RRC protocol verification via lattice properties (Step 75).

Models 5G NAS (Non-Access Stratum) and RRC (Radio Resource Control)
protocols as session types and verifies their correctness through
lattice analysis, subtyping for protocol evolution (4G to 5G), and
coverage computation.

5G protocols operate at two distinct layers:

- **NAS** (Non-Access Stratum): between UE and AMF (core network).
  Handles registration, authentication, service requests, and PDU
  session management.  NAS messages traverse the radio and transport
  network transparently.

- **RRC** (Radio Resource Control): between UE and gNB (base station).
  Handles connection setup, reconfiguration, release, and handover.
  RRC manages the radio bearer configuration.

Session types capture the strict state machine of each procedure.
The lattice property on the resulting state space guarantees that
every pair of protocol states has a well-defined join (least common
continuation) and meet (greatest common predecessor), which is
critical for protocol recovery and error handling in mobile networks.

Protocol evolution from 4G (LTE) to 5G is modelled via Gay--Hole
subtyping: the 5G version extends the 4G version with additional
capabilities while remaining backward-compatible for the common
subset of interactions.

Usage:
    from reticulate.fiveg_protocols import (
        nas_registration,
        rrc_setup,
        verify_5g_protocol,
        analyze_all_5g,
    )
    proto = nas_registration()
    result = verify_5g_protocol(proto)
    print(format_5g_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass
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
from reticulate.subtyping import SubtypingResult, check_subtype, is_subtype
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FiveGProtocol:
    """A named 5G protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "NAS_Registration").
        layer: Protocol layer ("NAS" or "RRC").
        roles: Participants in the protocol (e.g., ("UE", "AMF")).
        session_type_string: Session type encoding of the protocol.
        spec_reference: 3GPP specification reference.
        description: Free-text description of the protocol.
    """
    name: str
    layer: str
    roles: tuple[str, ...]
    session_type_string: str
    spec_reference: str
    description: str


@dataclass(frozen=True)
class FiveGAnalysisResult:
    """Complete analysis result for a 5G protocol.

    Attributes:
        protocol: The analysed protocol definition.
        ast: Parsed session type AST.
        state_space: Constructed state space (reticulate).
        lattice_result: Lattice property check.
        distributivity: Distributivity check result.
        enumeration: Test path enumeration.
        test_source: Generated JUnit 5 test source.
        coverage: Coverage analysis.
        num_states: Number of states in the state space.
        num_transitions: Number of transitions.
        num_valid_paths: Count of valid execution paths.
        num_violations: Count of protocol violation points.
        is_well_formed: True iff state space is a lattice.
    """
    protocol: FiveGProtocol
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


@dataclass(frozen=True)
class ProtocolEvolutionResult:
    """Result of checking backward compatibility between 4G and 5G versions.

    Attributes:
        old_protocol: The 4G (LTE) protocol version.
        new_protocol: The 5G (NR) protocol version.
        subtyping_result: Gay-Hole subtyping check (old <= new).
        is_backward_compatible: True iff old is subtype of new.
        old_analysis: Full analysis of old protocol.
        new_analysis: Full analysis of new protocol.
    """
    old_protocol: FiveGProtocol
    new_protocol: FiveGProtocol
    subtyping_result: SubtypingResult
    is_backward_compatible: bool
    old_analysis: FiveGAnalysisResult
    new_analysis: FiveGAnalysisResult


# ---------------------------------------------------------------------------
# NAS protocol definitions
# ---------------------------------------------------------------------------

def nas_registration() -> FiveGProtocol:
    """5G NAS Registration procedure (TS 24.501 Section 5.5.1).

    Models the UE initial registration with the AMF:
    1. UE sends Registration Request
    2. AMF initiates authentication (EAP-AKA' or 5G-AKA)
    3. UE responds to authentication challenge
    4. AMF performs security mode command
    5. UE completes security mode
    6. AMF accepts or rejects registration

    Participants: UE (User Equipment), AMF (Access and Mobility Function).
    """
    return FiveGProtocol(
        name="NAS_Registration",
        layer="NAS",
        roles=("UE", "AMF"),
        session_type_string=(
            "&{registrationRequest: &{authRequest: "
            "&{authResponse: +{AUTH_OK: "
            "&{securityModeCommand: &{securityModeComplete: "
            "+{REG_ACCEPT: end, REG_REJECT: end}}}, "
            "AUTH_FAIL: end}}}}"
        ),
        spec_reference="3GPP TS 24.501 Section 5.5.1",
        description=(
            "5G NAS Registration: UE sends Registration Request to AMF, "
            "AMF initiates authentication challenge, UE responds, AMF "
            "establishes security context, then accepts or rejects."
        ),
    )


def nas_deregistration() -> FiveGProtocol:
    """5G NAS Deregistration procedure (TS 24.501 Section 5.5.2).

    Models UE-initiated deregistration:
    1. UE sends Deregistration Request (switch-off or re-registration)
    2. AMF processes request
    3. AMF accepts deregistration

    Also models network-initiated deregistration:
    - AMF sends Deregistration Request to UE
    - UE acknowledges
    """
    return FiveGProtocol(
        name="NAS_Deregistration",
        layer="NAS",
        roles=("UE", "AMF"),
        session_type_string=(
            "&{deregRequest: +{SWITCH_OFF: end, "
            "RE_REG: &{deregAccept: end}}}"
        ),
        spec_reference="3GPP TS 24.501 Section 5.5.2",
        description=(
            "5G NAS Deregistration: UE sends Deregistration Request. "
            "If switch-off, session ends immediately. If re-registration "
            "needed, AMF sends Deregistration Accept."
        ),
    )


def nas_service_request() -> FiveGProtocol:
    """5G NAS Service Request procedure (TS 24.501 Section 5.6.1).

    Models the UE service request to resume a connection:
    1. UE sends Service Request (triggers N2 setup)
    2. AMF validates security context
    3. AMF accepts or rejects the service request
    4. If accepted, PDU sessions are activated

    Used when UE transitions from CM-IDLE to CM-CONNECTED.
    """
    return FiveGProtocol(
        name="NAS_ServiceRequest",
        layer="NAS",
        roles=("UE", "AMF"),
        session_type_string=(
            "&{serviceRequest: +{SERVICE_ACCEPT: "
            "&{pduSessionActivation: end}, "
            "SERVICE_REJECT: end}}"
        ),
        spec_reference="3GPP TS 24.501 Section 5.6.1",
        description=(
            "5G NAS Service Request: UE sends Service Request to resume "
            "connection from CM-IDLE state. AMF validates context and "
            "accepts (activating PDU sessions) or rejects."
        ),
    )


def nas_pdu_session_establishment() -> FiveGProtocol:
    """5G NAS PDU Session Establishment (TS 24.501 Section 6.4.1).

    Models PDU session creation:
    1. UE sends PDU Session Establishment Request
    2. AMF forwards to SMF
    3. SMF allocates resources
    4. Network sends PDU Session Establishment Accept or Reject
    5. If accepted, user plane is established

    Participants: UE, AMF, SMF (Session Management Function).
    """
    return FiveGProtocol(
        name="NAS_PDUSessionEstablishment",
        layer="NAS",
        roles=("UE", "AMF", "SMF"),
        session_type_string=(
            "&{pduSessionRequest: &{smfAllocation: "
            "+{PDU_ACCEPT: &{userPlaneSetup: end}, "
            "PDU_REJECT: end}}}"
        ),
        spec_reference="3GPP TS 24.501 Section 6.4.1",
        description=(
            "5G NAS PDU Session Establishment: UE requests a PDU session, "
            "SMF allocates resources, network accepts (establishing user "
            "plane) or rejects the session."
        ),
    )


def nas_authentication() -> FiveGProtocol:
    """5G NAS Authentication procedure (TS 33.501 Section 6.1).

    Models the 5G-AKA authentication:
    1. AMF sends Authentication Request with RAND, AUTN
    2. UE verifies AUTN (network authentication)
    3. UE computes RES* and sends Authentication Response
    4. AMF verifies RES* (UE authentication)
    5. Authentication succeeds or fails

    Uses 5G-AKA (Authentication and Key Agreement).
    """
    return FiveGProtocol(
        name="NAS_Authentication",
        layer="NAS",
        roles=("UE", "AMF", "AUSF"),
        session_type_string=(
            "&{authRequest: +{AUTN_VALID: "
            "&{authResponse: +{RES_VALID: end, "
            "RES_INVALID: &{authReject: end}}}, "
            "AUTN_INVALID: &{authFailure: end}}}"
        ),
        spec_reference="3GPP TS 33.501 Section 6.1",
        description=(
            "5G-AKA Authentication: AMF sends challenge (RAND, AUTN), "
            "UE verifies network authenticity via AUTN, then responds "
            "with RES*. AMF verifies UE identity. Mutual authentication."
        ),
    )


# ---------------------------------------------------------------------------
# RRC protocol definitions
# ---------------------------------------------------------------------------

def rrc_setup() -> FiveGProtocol:
    """5G RRC Setup procedure (TS 38.331 Section 5.3.3).

    Models the RRC connection establishment:
    1. UE sends RRC Setup Request
    2. gNB sends RRC Setup (with SRB1 configuration)
    3. UE sends RRC Setup Complete (with NAS message)

    Establishes Signalling Radio Bearer 1 (SRB1).
    """
    return FiveGProtocol(
        name="RRC_Setup",
        layer="RRC",
        roles=("UE", "gNB"),
        session_type_string=(
            "&{rrcSetupRequest: +{SETUP_OK: "
            "&{rrcSetup: &{rrcSetupComplete: end}}, "
            "SETUP_REJECT: end}}"
        ),
        spec_reference="3GPP TS 38.331 Section 5.3.3",
        description=(
            "5G RRC Setup: UE sends RRC Setup Request, gNB responds with "
            "RRC Setup (configuring SRB1) or rejects. UE completes setup "
            "by sending RRC Setup Complete with initial NAS message."
        ),
    )


def rrc_reconfiguration() -> FiveGProtocol:
    """5G RRC Reconfiguration procedure (TS 38.331 Section 5.3.5).

    Models radio bearer reconfiguration:
    1. gNB sends RRC Reconfiguration (new radio config)
    2. UE applies configuration
    3. UE sends RRC Reconfiguration Complete or fails

    Used for adding/modifying/releasing radio bearers, handover
    preparation, and measurement configuration.
    """
    return FiveGProtocol(
        name="RRC_Reconfiguration",
        layer="RRC",
        roles=("UE", "gNB"),
        session_type_string=(
            "&{rrcReconfiguration: +{RECONFIG_OK: "
            "&{rrcReconfigComplete: end}, "
            "RECONFIG_FAIL: &{rrcReestablishRequest: "
            "+{REESTABLISH_OK: &{rrcReestablishment: "
            "&{rrcReestablishComplete: end}}, "
            "REESTABLISH_REJECT: end}}}}"
        ),
        spec_reference="3GPP TS 38.331 Section 5.3.5",
        description=(
            "5G RRC Reconfiguration: gNB sends new radio configuration, "
            "UE applies it and confirms, or fails and triggers RRC "
            "re-establishment procedure as fallback."
        ),
    )


def rrc_release() -> FiveGProtocol:
    """5G RRC Release procedure (TS 38.331 Section 5.3.8).

    Models RRC connection release:
    1. gNB sends RRC Release
    2. UE may be redirected to another RAT or cell
    3. UE transitions to RRC_IDLE or RRC_INACTIVE

    The release may include a suspend configuration for fast resumption.
    """
    return FiveGProtocol(
        name="RRC_Release",
        layer="RRC",
        roles=("UE", "gNB"),
        session_type_string=(
            "&{rrcRelease: +{TO_IDLE: end, "
            "TO_INACTIVE: &{suspendConfig: end}, "
            "REDIRECT: &{cellRedirection: end}}}"
        ),
        spec_reference="3GPP TS 38.331 Section 5.3.8",
        description=(
            "5G RRC Release: gNB releases the RRC connection. UE "
            "transitions to RRC_IDLE, RRC_INACTIVE (with suspend config "
            "for fast resumption), or is redirected to another cell/RAT."
        ),
    )


def rrc_handover() -> FiveGProtocol:
    """5G Intra-NR Handover (TS 38.331 Section 5.3.5.4).

    Models the Xn-based handover between gNBs:
    1. Source gNB sends Handover Request to target gNB
    2. Target gNB sends Handover Request Acknowledge
    3. Source gNB sends RRC Reconfiguration (handover command) to UE
    4. UE synchronises with target cell
    5. UE sends RRC Reconfiguration Complete to target gNB

    If handover fails, UE triggers re-establishment.
    """
    return FiveGProtocol(
        name="RRC_Handover",
        layer="RRC",
        roles=("UE", "SourceGNB", "TargetGNB"),
        session_type_string=(
            "&{handoverRequest: &{handoverAck: "
            "&{rrcReconfigHandover: +{HO_SUCCESS: "
            "&{syncTarget: &{rrcReconfigCompleteHO: end}}, "
            "HO_FAIL: &{rrcReestablishRequest: "
            "+{REESTABLISH_OK: &{rrcReestablishment: "
            "&{rrcReestablishComplete: end}}, "
            "REESTABLISH_REJECT: end}}}}}}"
        ),
        spec_reference="3GPP TS 38.331 Section 5.3.5.4",
        description=(
            "5G Intra-NR Handover: source gNB requests handover to target, "
            "target acknowledges, source commands UE via RRC Reconfiguration. "
            "UE synchronises with target and completes. On failure, UE "
            "triggers RRC re-establishment."
        ),
    )


# ---------------------------------------------------------------------------
# 4G (LTE) protocol definitions for evolution comparison
# ---------------------------------------------------------------------------

def lte_attach() -> FiveGProtocol:
    """4G LTE Attach procedure (TS 24.301 Section 5.5.1).

    Simplified 4G attach for comparison with 5G registration.
    4G attach is similar but uses different authentication (EPS-AKA)
    and has a simpler security context setup.
    """
    return FiveGProtocol(
        name="LTE_Attach",
        layer="NAS",
        roles=("UE", "MME"),
        session_type_string=(
            "&{attachRequest: &{authRequest: "
            "&{authResponse: +{AUTH_OK: "
            "+{ATTACH_ACCEPT: end, ATTACH_REJECT: end}, "
            "AUTH_FAIL: end}}}}"
        ),
        spec_reference="3GPP TS 24.301 Section 5.5.1",
        description=(
            "4G LTE Attach: UE sends Attach Request to MME, "
            "MME initiates EPS-AKA authentication, UE responds, "
            "MME accepts or rejects the attach."
        ),
    )


def lte_rrc_setup() -> FiveGProtocol:
    """4G LTE RRC Connection Setup (TS 36.331 Section 5.3.3).

    Simplified 4G RRC setup for comparison with 5G RRC setup.
    """
    return FiveGProtocol(
        name="LTE_RRC_Setup",
        layer="RRC",
        roles=("UE", "eNB"),
        session_type_string=(
            "&{rrcSetupRequest: +{SETUP_OK: "
            "&{rrcSetup: &{rrcSetupComplete: end}}, "
            "SETUP_REJECT: end}}"
        ),
        spec_reference="3GPP TS 36.331 Section 5.3.3",
        description=(
            "4G LTE RRC Connection Setup: UE sends RRC Connection Setup "
            "Request, eNB responds with setup or reject, UE completes."
        ),
    )


def lte_handover() -> FiveGProtocol:
    """4G LTE X2 Handover (TS 36.331 Section 5.3.5.4).

    Simplified 4G handover for comparison with 5G handover.
    """
    return FiveGProtocol(
        name="LTE_Handover",
        layer="RRC",
        roles=("UE", "SourceENB", "TargetENB"),
        session_type_string=(
            "&{handoverRequest: &{handoverAck: "
            "&{rrcReconfigHandover: +{HO_SUCCESS: "
            "&{rrcReconfigCompleteHO: end}, "
            "HO_FAIL: end}}}}"
        ),
        spec_reference="3GPP TS 36.331 Section 5.3.5.4",
        description=(
            "4G LTE X2 Handover: source eNB requests handover to target, "
            "target acknowledges, source commands UE. On success UE "
            "completes; on failure connection is lost (no re-establishment "
            "path in simplified model)."
        ),
    )


# ---------------------------------------------------------------------------
# Registry of all 5G protocols
# ---------------------------------------------------------------------------

ALL_NAS_PROTOCOLS: tuple[FiveGProtocol, ...] = (
    nas_registration(),
    nas_deregistration(),
    nas_service_request(),
    nas_pdu_session_establishment(),
    nas_authentication(),
)
"""All pre-defined 5G NAS protocols."""

ALL_RRC_PROTOCOLS: tuple[FiveGProtocol, ...] = (
    rrc_setup(),
    rrc_reconfiguration(),
    rrc_release(),
    rrc_handover(),
)
"""All pre-defined 5G RRC protocols."""

ALL_5G_PROTOCOLS: tuple[FiveGProtocol, ...] = ALL_NAS_PROTOCOLS + ALL_RRC_PROTOCOLS
"""All pre-defined 5G protocols (NAS + RRC)."""

ALL_LTE_PROTOCOLS: tuple[FiveGProtocol, ...] = (
    lte_attach(),
    lte_rrc_setup(),
    lte_handover(),
)
"""4G LTE protocol versions for evolution comparison."""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def fiveg_to_session_type(protocol: FiveGProtocol) -> SessionType:
    """Parse the protocol's session type string into an AST."""
    return parse(protocol.session_type_string)


def verify_5g_protocol(
    protocol: FiveGProtocol,
    config: TestGenConfig | None = None,
) -> FiveGAnalysisResult:
    """Run the full verification pipeline on a 5G protocol.

    Parses the protocol's session type, builds the state space,
    checks lattice properties, generates conformance tests, and
    computes coverage.

    Args:
        protocol: The 5G protocol to verify.
        config: Optional test generation configuration.

    Returns:
        A complete FiveGAnalysisResult.
    """
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}ProtocolTest",
            package_name="com.fiveg.conformance",
        )

    # 1. Parse and build state space
    ast = fiveg_to_session_type(protocol)
    ss = build_statespace(ast)

    # 2. Lattice analysis
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    # 3. Test generation
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.session_type_string)

    # 4. Coverage
    cov = compute_coverage(ss, result=enum)

    return FiveGAnalysisResult(
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
    )


def analyze_nas_protocol(protocol: FiveGProtocol) -> FiveGAnalysisResult:
    """Analyse a single NAS protocol.

    Convenience wrapper around verify_5g_protocol for NAS layer.

    Args:
        protocol: A NAS-layer protocol to analyse.

    Returns:
        FiveGAnalysisResult for the protocol.

    Raises:
        ValueError: If the protocol is not a NAS protocol.
    """
    if protocol.layer != "NAS":
        raise ValueError(f"Expected NAS protocol, got {protocol.layer}: {protocol.name}")
    return verify_5g_protocol(protocol)


def analyze_rrc_protocol(protocol: FiveGProtocol) -> FiveGAnalysisResult:
    """Analyse a single RRC protocol.

    Convenience wrapper around verify_5g_protocol for RRC layer.

    Args:
        protocol: An RRC-layer protocol to analyse.

    Returns:
        FiveGAnalysisResult for the protocol.

    Raises:
        ValueError: If the protocol is not an RRC protocol.
    """
    if protocol.layer != "RRC":
        raise ValueError(f"Expected RRC protocol, got {protocol.layer}: {protocol.name}")
    return verify_5g_protocol(protocol)


def analyze_all_5g() -> list[FiveGAnalysisResult]:
    """Verify all pre-defined 5G protocols and return results."""
    return [verify_5g_protocol(p) for p in ALL_5G_PROTOCOLS]


def check_protocol_evolution(
    old: FiveGProtocol,
    new: FiveGProtocol,
) -> ProtocolEvolutionResult:
    """Check whether a 5G protocol is backward-compatible with its 4G version.

    Uses Gay-Hole subtyping: old <= new means the new version accepts
    all interactions that the old version accepted (the new version may
    offer additional methods, but must support all old ones).

    For mobile protocols, backward compatibility means that UEs built
    for the 4G version can still interact with 5G network functions.

    Args:
        old: The 4G (LTE) protocol version.
        new: The 5G (NR) protocol version.

    Returns:
        ProtocolEvolutionResult with compatibility verdict.
    """
    old_ast = fiveg_to_session_type(old)
    new_ast = fiveg_to_session_type(new)

    sub_result = check_subtype(old_ast, new_ast)

    old_analysis = verify_5g_protocol(old)
    new_analysis = verify_5g_protocol(new)

    return ProtocolEvolutionResult(
        old_protocol=old,
        new_protocol=new,
        subtyping_result=sub_result,
        is_backward_compatible=sub_result.is_subtype,
        old_analysis=old_analysis,
        new_analysis=new_analysis,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_5g_report(result: FiveGAnalysisResult) -> str:
    """Format a FiveGAnalysisResult as structured text for terminal output."""
    lines: list[str] = []
    proto = result.protocol

    lines.append("=" * 70)
    lines.append(f"  5G PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")

    # Protocol details
    lines.append("--- Protocol Details ---")
    lines.append(f"  Layer: {proto.layer}")
    lines.append(f"  Spec:  {proto.spec_reference}")
    for r in proto.roles:
        lines.append(f"  Role:  {r}")
    lines.append("")

    # Session type
    lines.append("--- Session Type ---")
    lines.append(f"  {proto.session_type_string}")
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
    lines.append(f"  Is lattice:      {result.lattice_result.is_lattice}")
    lines.append(f"  Distributive:    {result.distributivity.is_distributive}")
    if not result.lattice_result.is_lattice and result.lattice_result.counterexample:
        lines.append(f"  Counterexample:  {result.lattice_result.counterexample}")
    lines.append("")

    # Test generation
    lines.append("--- Test Generation ---")
    lines.append(f"  Valid paths:     {result.num_valid_paths}")
    lines.append(f"  Violation points:{result.num_violations}")
    lines.append("")

    # Coverage
    lines.append("--- Coverage ---")
    lines.append(f"  State coverage:      {result.coverage.state_coverage:.1%}")
    lines.append(f"  Transition coverage: {result.coverage.transition_coverage:.1%}")
    lines.append("")

    # Verdict
    lines.append("--- Verdict ---")
    if result.is_well_formed:
        lines.append("  PASS: Protocol forms a valid lattice.")
        lines.append("  Every pair of protocol states has a well-defined")
        lines.append("  join and meet, ensuring unambiguous recovery.")
    else:
        lines.append("  FAIL: Protocol does NOT form a valid lattice.")
        lines.append("  Some protocol states lack a join or meet,")
        lines.append("  which may lead to ambiguous recovery scenarios.")
    lines.append("")

    return "\n".join(lines)


def format_evolution_report(result: ProtocolEvolutionResult) -> str:
    """Format a ProtocolEvolutionResult as structured text."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append(f"  PROTOCOL EVOLUTION: {result.old_protocol.name} -> "
                 f"{result.new_protocol.name}")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"  Old protocol: {result.old_protocol.name}")
    lines.append(f"    States: {result.old_analysis.num_states}, "
                 f"Transitions: {result.old_analysis.num_transitions}")
    lines.append(f"  New protocol: {result.new_protocol.name}")
    lines.append(f"    States: {result.new_analysis.num_states}, "
                 f"Transitions: {result.new_analysis.num_transitions}")
    lines.append("")

    lines.append("--- Subtyping Analysis ---")
    lines.append(f"  {result.old_protocol.name} <= {result.new_protocol.name}: "
                 f"{result.subtyping_result.is_subtype}")
    if result.subtyping_result.reason:
        lines.append(f"  Reason: {result.subtyping_result.reason}")
    lines.append("")

    lines.append("--- Backward Compatibility ---")
    if result.is_backward_compatible:
        lines.append("  COMPATIBLE: The 5G version is backward-compatible with 4G.")
        lines.append("  UEs built for the 4G version can interact with")
        lines.append("  5G network functions for common procedures.")
    else:
        lines.append("  INCOMPATIBLE: The 5G version is NOT backward-compatible.")
        lines.append("  The protocol evolution introduced breaking changes.")
    lines.append("")

    return "\n".join(lines)


def format_5g_summary(results: list[FiveGAnalysisResult]) -> str:
    """Format a summary table of all verified 5G protocols."""
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("  5G PROTOCOL VERIFICATION SUMMARY")
    lines.append("=" * 78)
    lines.append("")

    header = (
        f"  {'Protocol':<30} {'Layer':<6} "
        f"{'States':>6} {'Trans':>6} {'Lattice':>8} {'Dist':>6} {'Paths':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * 74)

    for r in results:
        lattice_str = "YES" if r.is_well_formed else "NO"
        dist_str = "YES" if r.distributivity.is_distributive else "NO"
        row = (
            f"  {r.protocol.name:<30} "
            f"{r.protocol.layer:<6} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{lattice_str:>8} "
            f"{dist_str:>6} "
            f"{r.num_valid_paths:>6}"
        )
        lines.append(row)

    lines.append("")
    all_lattice = all(r.is_well_formed for r in results)
    lines.append(f"  All protocols form lattices: {'YES' if all_lattice else 'NO'}")
    lines.append("")

    return "\n".join(lines)
