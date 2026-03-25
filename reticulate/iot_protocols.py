"""IoT protocol verification via lattice properties (Step 86).

Models IoT protocols (MQTT, CoAP, Zigbee) as session types and verifies
their correctness through lattice analysis, coverage computation, and
runtime monitor generation.

IoT protocols are characterised by constrained devices, unreliable
networks, and quality-of-service (QoS) tiers.  Session types capture
the stateful lifecycle of each protocol precisely, and the lattice
property on the resulting state space guarantees unambiguous protocol
recovery from any reachable state --- a critical requirement for
resource-constrained devices that cannot afford expensive error handling.

MQTT QoS levels map naturally to session type complexity:
- QoS 0 (at most once): fire-and-forget, no acknowledgement.
- QoS 1 (at least once): PUBACK handshake.
- QoS 2 (exactly once): four-step PUBREC/PUBREL/PUBCOMP handshake.

CoAP distinguishes confirmable (CON) from non-confirmable (NON)
messages, with the observe pattern adding subscription semantics.

Zigbee network join involves beacon scanning, association, and
security key exchange --- a multi-phase handshake well-suited to
session type modelling.

Usage:
    from reticulate.iot_protocols import (
        mqtt_publish_subscribe,
        coap_request_response,
        verify_iot_protocol,
    )
    proto = mqtt_publish_subscribe(qos=2)
    result = verify_iot_protocol(proto)
    print(format_iot_report(result))
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
from reticulate.monitor import (
    MonitorTemplate,
    generate_python_monitor,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
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
class IoTProtocol:
    """A named IoT protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "MQTT_QoS2").
        session_type_string: Session type encoding of the protocol.
        qos_level: Quality-of-service level (0, 1, or 2 for MQTT;
            -1 for protocols without QoS tiers).
        description: Free-text description of the protocol.
        transport: Underlying transport ("TCP", "UDP", "IEEE802.15.4").
        constrained: Whether the protocol targets constrained devices.
    """
    name: str
    session_type_string: str
    qos_level: int
    description: str
    transport: str = "TCP"
    constrained: bool = False


@dataclass(frozen=True)
class IoTAnalysisResult:
    """Complete analysis result for an IoT protocol.

    Attributes:
        protocol: The analysed protocol definition.
        ast: Parsed session type AST.
        state_space: Constructed state space (reticulate).
        lattice_result: Lattice property check.
        distributivity: Distributivity check result.
        enumeration: Test path enumeration.
        test_source: Generated JUnit 5 test source.
        coverage: Coverage analysis.
        monitor: Generated Python runtime monitor.
        num_states: Number of states in the state space.
        num_transitions: Number of transitions.
        num_valid_paths: Count of valid execution paths.
        num_violations: Count of protocol violation points.
        is_well_formed: True iff state space is a lattice.
    """
    protocol: IoTProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    enumeration: EnumerationResult
    test_source: str
    coverage: CoverageResult
    monitor: MonitorTemplate
    num_states: int
    num_transitions: int
    num_valid_paths: int
    num_violations: int
    is_well_formed: bool


# ---------------------------------------------------------------------------
# MQTT protocol definitions
# ---------------------------------------------------------------------------

def mqtt_publish_subscribe(qos: int = 0) -> IoTProtocol:
    """MQTT publish/subscribe lifecycle with configurable QoS.

    Models the MQTT client lifecycle:
    1. CONNECT to broker
    2. Broker sends CONNACK (accept or reject)
    3. If accepted, SUBSCRIBE to topics
    4. Broker sends SUBACK
    5. PUBLISH messages with QoS-specific handshake
    6. DISCONNECT

    QoS levels determine the publish acknowledgement pattern:
    - QoS 0: no acknowledgement (fire-and-forget)
    - QoS 1: PUBACK from broker
    - QoS 2: PUBREC -> PUBREL -> PUBCOMP four-step handshake

    Args:
        qos: MQTT QoS level (0, 1, or 2).

    Returns:
        IoTProtocol with the MQTT pub/sub session type.

    Raises:
        ValueError: If qos is not 0, 1, or 2.
    """
    if qos not in (0, 1, 2):
        raise ValueError(f"MQTT QoS must be 0, 1, or 2, got {qos}")

    if qos == 0:
        # QoS 0: fire-and-forget — no publish ack
        st = (
            "&{connect: +{CONNACK_OK: "
            "&{subscribe: &{suback: "
            "&{publish: &{disconnect: end}}}}, "
            "CONNACK_FAIL: end}}"
        )
        desc = (
            "MQTT QoS 0 (at most once): client connects, subscribes to "
            "topics, publishes messages with no delivery guarantee, and "
            "disconnects. Fire-and-forget semantics."
        )
    elif qos == 1:
        # QoS 1: PUBACK handshake — at least once
        st = (
            "&{connect: +{CONNACK_OK: "
            "&{subscribe: &{suback: "
            "&{publish: &{puback: &{disconnect: end}}}}}, "
            "CONNACK_FAIL: end}}"
        )
        desc = (
            "MQTT QoS 1 (at least once): client connects, subscribes, "
            "publishes messages with PUBACK acknowledgement from broker. "
            "Messages may be delivered more than once."
        )
    else:
        # QoS 2: PUBREC/PUBREL/PUBCOMP — exactly once
        st = (
            "&{connect: +{CONNACK_OK: "
            "&{subscribe: &{suback: "
            "&{publish: &{pubrec: &{pubrel: &{pubcomp: "
            "&{disconnect: end}}}}}}}, "
            "CONNACK_FAIL: end}}"
        )
        desc = (
            "MQTT QoS 2 (exactly once): client connects, subscribes, "
            "publishes with the four-step PUBREC/PUBREL/PUBCOMP "
            "handshake ensuring exactly-once delivery."
        )

    return IoTProtocol(
        name=f"MQTT_QoS{qos}",
        session_type_string=st,
        qos_level=qos,
        description=desc,
        transport="TCP",
        constrained=False,
    )


def mqtt_retained_messages() -> IoTProtocol:
    """MQTT retained message flow.

    Models the retained message pattern where:
    1. Publisher connects and publishes a retained message
    2. Subscriber connects later and receives the retained message
    3. Subscriber can clear the retained message with empty payload

    The session type captures the publisher side: connect, set retain
    flag, publish retained, then either update or clear.
    """
    st = (
        "&{connect: +{CONNACK_OK: "
        "&{publishRetained: +{RETAIN_OK: "
        "&{updateRetained: end, clearRetained: end}, "
        "RETAIN_FAIL: end}}, CONNACK_FAIL: end}}"
    )
    return IoTProtocol(
        name="MQTT_Retained",
        session_type_string=st,
        qos_level=1,
        description=(
            "MQTT retained messages: publisher connects, publishes a "
            "retained message (stored by broker for future subscribers), "
            "then either updates or clears the retained value."
        ),
        transport="TCP",
        constrained=False,
    )


def mqtt_last_will() -> IoTProtocol:
    """MQTT Last Will and Testament (LWT) pattern.

    Models the LWT setup where:
    1. Client connects with a will message
    2. Broker accepts with will registration
    3. Client operates normally (publish/subscribe)
    4. On ungraceful disconnect, broker publishes will message
    5. On graceful disconnect, will is discarded
    """
    st = (
        "&{connectWithWill: +{CONNACK_WILL_OK: "
        "&{subscribe: &{suback: "
        "&{publish: +{GRACEFUL: &{disconnect: end}, "
        "UNGRACEFUL: &{willPublished: end}}}}}, "
        "CONNACK_FAIL: end}}"
    )
    return IoTProtocol(
        name="MQTT_LastWill",
        session_type_string=st,
        qos_level=0,
        description=(
            "MQTT Last Will and Testament: client connects with a will "
            "message. On ungraceful disconnect the broker publishes the "
            "will to subscribers; on graceful disconnect the will is discarded."
        ),
        transport="TCP",
        constrained=False,
    )


# ---------------------------------------------------------------------------
# CoAP protocol definitions
# ---------------------------------------------------------------------------

def coap_request_response(confirmable: bool = True) -> IoTProtocol:
    """CoAP request/response interaction.

    Models the Constrained Application Protocol (RFC 7252):
    - Confirmable (CON): requires ACK from server
    - Non-confirmable (NON): fire-and-forget like UDP

    For CON messages, the server may piggyback the response on the ACK
    or send a separate response after an empty ACK.

    Args:
        confirmable: If True, model CON request; otherwise NON.

    Returns:
        IoTProtocol with the CoAP request/response session type.
    """
    if confirmable:
        # CON: request -> ACK with piggyback or separate response
        st = (
            "&{conRequest: +{ACK_PIGGYBACK: end, "
            "ACK_EMPTY: &{separateResponse: end}, "
            "TIMEOUT: &{retransmit: +{ACK_PIGGYBACK: end, "
            "ACK_EMPTY: &{separateResponse: end}, "
            "FAIL: end}}}}"
        )
        desc = (
            "CoAP confirmable request: client sends CON request, server "
            "responds with piggybacked ACK or empty ACK followed by "
            "separate response. On timeout, client retransmits."
        )
        name = "CoAP_CON"
    else:
        # NON: request -> optional response (no ack required)
        st = (
            "&{nonRequest: +{RESPONSE: end, NO_RESPONSE: end}}"
        )
        desc = (
            "CoAP non-confirmable request: client sends NON request. "
            "Server may or may not respond. No acknowledgement required."
        )
        name = "CoAP_NON"

    return IoTProtocol(
        name=name,
        session_type_string=st,
        qos_level=1 if confirmable else 0,
        description=desc,
        transport="UDP",
        constrained=True,
    )


def coap_observe() -> IoTProtocol:
    """CoAP observe pattern (RFC 7641).

    Models the CoAP observation lifecycle:
    1. Client registers observation on a resource
    2. Server accepts or rejects
    3. If accepted, server sends notifications on state changes
    4. Client can deregister at any time

    Uses recursion for the notification loop.
    """
    st = (
        "&{register: +{OBSERVE_OK: "
        "rec X . &{notification: +{CONTINUE: X, "
        "DEREGISTER: end}}, "
        "OBSERVE_REJECT: end}}"
    )
    return IoTProtocol(
        name="CoAP_Observe",
        session_type_string=st,
        qos_level=1,
        description=(
            "CoAP observe pattern: client registers to observe a resource, "
            "server sends notifications on state changes. Client can "
            "deregister at any time, ending the observation."
        ),
        transport="UDP",
        constrained=True,
    )


def coap_block_transfer() -> IoTProtocol:
    """CoAP block-wise transfer (RFC 7959).

    Models the block-wise transfer for large payloads:
    1. Client requests resource with Block2 option
    2. Server responds with first block
    3. Client requests next blocks until complete
    """
    st = (
        "&{requestBlock: &{blockResponse: "
        "rec X . +{MORE_BLOCKS: &{requestNext: &{blockResponse: X}}, "
        "COMPLETE: end}}}"
    )
    return IoTProtocol(
        name="CoAP_BlockTransfer",
        session_type_string=st,
        qos_level=1,
        description=(
            "CoAP block-wise transfer: client requests a large resource "
            "in blocks. Server responds block by block until the entire "
            "payload is transferred."
        ),
        transport="UDP",
        constrained=True,
    )


# ---------------------------------------------------------------------------
# Zigbee protocol definitions
# ---------------------------------------------------------------------------

def zigbee_join() -> IoTProtocol:
    """Zigbee network join protocol.

    Models the Zigbee network joining procedure:
    1. Device performs beacon scan to discover networks
    2. Device selects a network and sends association request
    3. Coordinator accepts or rejects
    4. If accepted, key exchange for network security
    5. Device is now part of the network

    Based on IEEE 802.15.4 MAC association + Zigbee APS security.
    """
    st = (
        "&{beaconScan: &{selectNetwork: "
        "&{associateRequest: +{ASSOC_OK: "
        "&{keyExchange: +{KEY_OK: &{joinComplete: end}, "
        "KEY_FAIL: end}}, "
        "ASSOC_REJECT: end}}}}"
    )
    return IoTProtocol(
        name="Zigbee_Join",
        session_type_string=st,
        qos_level=-1,
        description=(
            "Zigbee network join: device scans for beacons, selects a "
            "network, sends association request to coordinator, performs "
            "security key exchange, and completes the join."
        ),
        transport="IEEE802.15.4",
        constrained=True,
    )


def zigbee_data_transfer() -> IoTProtocol:
    """Zigbee data transfer with acknowledgement.

    Models a Zigbee data frame exchange after network join:
    1. Device sends data frame
    2. Coordinator acknowledges
    3. Repeat or end session
    """
    st = (
        "rec X . &{sendData: +{DATA_ACK: "
        "+{CONTINUE: X, DONE: end}, "
        "DATA_NACK: +{RETRY: X, ABORT: end}}}"
    )
    return IoTProtocol(
        name="Zigbee_DataTransfer",
        session_type_string=st,
        qos_level=-1,
        description=(
            "Zigbee data transfer: device sends data frames with "
            "acknowledgement from coordinator. On NACK, device can "
            "retry or abort. Loop continues until done."
        ),
        transport="IEEE802.15.4",
        constrained=True,
    )


# ---------------------------------------------------------------------------
# Registry of all IoT protocols
# ---------------------------------------------------------------------------

ALL_IOT_PROTOCOLS: tuple[IoTProtocol, ...] = (
    mqtt_publish_subscribe(qos=0),
    mqtt_publish_subscribe(qos=1),
    mqtt_publish_subscribe(qos=2),
    mqtt_retained_messages(),
    mqtt_last_will(),
    coap_request_response(confirmable=True),
    coap_request_response(confirmable=False),
    coap_observe(),
    coap_block_transfer(),
    zigbee_join(),
    zigbee_data_transfer(),
)
"""All pre-defined IoT protocols."""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def iot_to_session_type(protocol: IoTProtocol) -> SessionType:
    """Parse the protocol's session type string into an AST."""
    return parse(protocol.session_type_string)


def verify_iot_protocol(
    protocol: IoTProtocol,
    config: TestGenConfig | None = None,
) -> IoTAnalysisResult:
    """Run the full verification pipeline on an IoT protocol.

    Parses the protocol's session type, builds the state space,
    checks lattice properties, generates conformance tests,
    computes coverage, and generates a runtime monitor.

    Args:
        protocol: The IoT protocol to verify.
        config: Optional test generation configuration.

    Returns:
        A complete IoTAnalysisResult.
    """
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}ProtocolTest",
            package_name="com.iot.conformance",
        )

    # 1. Parse and build state space
    ast = iot_to_session_type(protocol)
    ss = build_statespace(ast)

    # 2. Lattice analysis
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    # 3. Test generation
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.session_type_string)

    # 4. Coverage
    cov = compute_coverage(ss, result=enum)

    # 5. Monitor generation
    mon = generate_python_monitor(ast, protocol.name)

    return IoTAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        enumeration=enum,
        test_source=test_src,
        coverage=cov,
        monitor=mon,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_valid_paths=len(enum.valid_paths),
        num_violations=len(enum.violations),
        is_well_formed=lr.is_lattice,
    )


def verify_all_iot_protocols() -> list[IoTAnalysisResult]:
    """Verify all pre-defined IoT protocols and return results."""
    return [verify_iot_protocol(p) for p in ALL_IOT_PROTOCOLS]


# ---------------------------------------------------------------------------
# QoS comparison
# ---------------------------------------------------------------------------

def compare_qos_levels() -> dict[str, IoTAnalysisResult]:
    """Compare MQTT QoS levels 0, 1, 2 side by side.

    Returns a dict mapping "QoS0", "QoS1", "QoS2" to their
    analysis results, enabling cross-QoS comparison of state-space
    size, lattice properties, and test coverage.
    """
    results: dict[str, IoTAnalysisResult] = {}
    for qos in (0, 1, 2):
        proto = mqtt_publish_subscribe(qos=qos)
        results[f"QoS{qos}"] = verify_iot_protocol(proto)
    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_iot_report(result: IoTAnalysisResult) -> str:
    """Format an IoTAnalysisResult as structured text for terminal output."""
    lines: list[str] = []
    proto = result.protocol

    lines.append("=" * 70)
    lines.append(f"  IOT PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")

    # Transport and constraints
    lines.append("--- Protocol Characteristics ---")
    lines.append(f"  Transport:   {proto.transport}")
    lines.append(f"  QoS Level:   {proto.qos_level if proto.qos_level >= 0 else 'N/A'}")
    lines.append(f"  Constrained: {'Yes' if proto.constrained else 'No'}")
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

    # Monitor
    lines.append("--- Runtime Monitor ---")
    lines.append(f"  Monitor class: {result.monitor.class_name}")
    lines.append(f"  Source lines:  {len(result.monitor.source_code.splitlines())}")
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


def format_iot_summary(results: list[IoTAnalysisResult]) -> str:
    """Format a summary table of all verified IoT protocols."""
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("  IOT PROTOCOL VERIFICATION SUMMARY")
    lines.append("=" * 78)
    lines.append("")

    header = (
        f"  {'Protocol':<22} {'Transport':<10} {'QoS':>4} "
        f"{'States':>6} {'Trans':>6} {'Lattice':>8} {'Dist':>6} {'Paths':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * 74)

    for r in results:
        lattice_str = "YES" if r.is_well_formed else "NO"
        dist_str = "YES" if r.distributivity.is_distributive else "NO"
        qos_str = str(r.protocol.qos_level) if r.protocol.qos_level >= 0 else "N/A"
        row = (
            f"  {r.protocol.name:<22} "
            f"{r.protocol.transport:<10} "
            f"{qos_str:>4} "
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
