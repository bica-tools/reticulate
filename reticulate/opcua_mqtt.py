"""OPC-UA and MQTT industrial safety verification (Step 79).

Models OPC-UA client/server sessions and MQTT publish/subscribe flows
as session types, then verifies lattice properties and derives IEC 62443
industrial-security checks via bidirectional morphisms.

Domain
------
OPC-UA (IEC 62541) is the dominant machine-to-machine protocol in
industrial automation: it carries process variables between PLCs,
SCADA systems, historians, and MES/ERP layers.  MQTT (ISO/IEC 20922)
complements OPC-UA on the edge, carrying telemetry from sensors to
brokers.  Both protocols are referenced by IEC 62443 (industrial
cybersecurity) and must be monitored against stateful attacks such
as session hijacking, replay, and integrity violations.

The stateful handshake structure of OPC-UA (HEL -> ACK -> OPN -> CreateSession
-> ActivateSession -> Read/Write/Browse/Subscribe -> CloseSession) and of
MQTT pub/sub makes them natural session-type targets.  The lattice
property of the state space guarantees that every runtime monitor can
decide recovery or abort from any reachable state.

The bidirectional morphisms

    phi : L(S) -> SafetyAndSecurityChecks
    psi : SafetyAndSecurityChecks -> L(S)

map lattice structure onto IEC 62443 Foundational Requirements (FR1-FR7)
and back, giving the engineer a concrete audit trail.

Usage
-----
    from reticulate.opcua_mqtt import (
        opcua_client_session,
        mqtt_sparkplug_edge,
        verify_industrial_protocol,
        phi_lattice_to_checks,
        psi_checks_to_lattice,
    )
    proto = opcua_client_session()
    result = verify_industrial_protocol(proto)
    print(format_industrial_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# IEC 62443 foundational requirements
# ---------------------------------------------------------------------------

#: IEC 62443-3-3 Foundational Requirements (FR1 .. FR7)
IEC62443_FRS: Tuple[str, ...] = (
    "FR1_IdentificationAndAuthenticationControl",
    "FR2_UseControl",
    "FR3_SystemIntegrity",
    "FR4_DataConfidentiality",
    "FR5_RestrictedDataFlow",
    "FR6_TimelyResponseToEvents",
    "FR7_ResourceAvailability",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndustrialProtocol:
    """A named industrial protocol modelled as a session type.

    Attributes:
        name: Short identifier (e.g. "OPCUA_ClientSession").
        session_type_string: Session-type encoding.
        family: Protocol family ("OPCUA" or "MQTT").
        description: Free-text description.
        iec62443_scope: Set of FR identifiers the protocol touches.
    """
    name: str
    session_type_string: str
    family: str
    description: str
    iec62443_scope: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SafetyCheck:
    """A concrete safety or security check derived from a lattice state.

    A SafetyCheck is the image of a reachable state under phi.  It
    records the IEC 62443 Foundational Requirement, a monitor action,
    and an alerting severity.
    """
    state_id: int
    fr: str
    action: str
    severity: str  # "info" | "warn" | "critical"


@dataclass(frozen=True)
class IndustrialAnalysisResult:
    protocol: IndustrialProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    num_states: int
    num_transitions: int
    safety_checks: Tuple[SafetyCheck, ...]
    phi_total: bool
    psi_total: bool
    galois: bool
    is_well_formed: bool


# ---------------------------------------------------------------------------
# OPC-UA protocol definitions
# ---------------------------------------------------------------------------

def opcua_client_session() -> IndustrialProtocol:
    """Standard OPC-UA binary client session lifecycle.

    Captures the IEC 62541 handshake:
        HEL -> ACK -> OPN (SecureChannel) -> CreateSession
            -> ActivateSession -> { Read | Write | Browse }*
            -> CloseSession -> CLO.

    We abstract the Read/Write/Browse loop into one selection and
    terminate after CloseSession; realistic monitors unfold the loop
    once.
    """
    st = (
        "&{hello: +{ACK: "
        "&{openSecureChannel: +{OPN_OK: "
        "&{createSession: +{SESSION_OK: "
        "&{activateSession: +{ACTIVATED: "
        "+{read: &{closeSession: end}, "
        "write: &{closeSession: end}, "
        "browse: &{closeSession: end}}, "
        "AUTH_FAIL: end}}, "
        "SESSION_REJECT: end}}, "
        "OPN_FAIL: end}}, "
        "HEL_REJECT: end}}"
    )
    return IndustrialProtocol(
        name="OPCUA_ClientSession",
        session_type_string=st,
        family="OPCUA",
        description=(
            "OPC-UA binary client: HEL/ACK, SecureChannel open, "
            "CreateSession, ActivateSession, a single Read/Write/Browse "
            "operation, then CloseSession."
        ),
        iec62443_scope=("FR1_IdentificationAndAuthenticationControl",
                        "FR2_UseControl", "FR3_SystemIntegrity",
                        "FR4_DataConfidentiality"),
    )


def opcua_subscription() -> IndustrialProtocol:
    """OPC-UA subscription / MonitoredItem flow.

    After an activated session, the client creates a subscription,
    adds a MonitoredItem, receives a Publish response, then deletes
    the subscription.
    """
    st = (
        "&{activateSession: "
        "&{createSubscription: +{SUB_OK: "
        "&{createMonitoredItem: &{publish: "
        "&{deleteSubscription: &{closeSession: end}}}}, "
        "SUB_FAIL: &{closeSession: end}}}}"
    )
    return IndustrialProtocol(
        name="OPCUA_Subscription",
        session_type_string=st,
        family="OPCUA",
        description=(
            "OPC-UA subscription lifecycle: create subscription, add "
            "monitored item, receive publish notifications, delete "
            "subscription, close session."
        ),
        iec62443_scope=("FR2_UseControl", "FR3_SystemIntegrity",
                        "FR6_TimelyResponseToEvents"),
    )


def opcua_secure_channel_renew() -> IndustrialProtocol:
    """OPC-UA SecureChannel renew flow (certificate re-key)."""
    st = (
        "&{openSecureChannel: +{OPN_OK: "
        "&{useChannel: &{renewSecureChannel: +{RENEW_OK: "
        "&{useChannel2: &{closeSecureChannel: end}}, "
        "RENEW_FAIL: &{closeSecureChannel: end}}}}, "
        "OPN_FAIL: end}}"
    )
    return IndustrialProtocol(
        name="OPCUA_RenewSecureChannel",
        session_type_string=st,
        family="OPCUA",
        description=(
            "OPC-UA SecureChannel renewal: open, use, renew, use, close. "
            "Models certificate/nonce re-keying for long-lived sessions."
        ),
        iec62443_scope=("FR1_IdentificationAndAuthenticationControl",
                        "FR4_DataConfidentiality"),
    )


# ---------------------------------------------------------------------------
# MQTT industrial definitions
# ---------------------------------------------------------------------------

def mqtt_sparkplug_edge() -> IndustrialProtocol:
    """Sparkplug B edge-of-network node lifecycle.

    Sparkplug B (Eclipse Tahu) is the dominant MQTT profile for
    industrial telemetry.  An Edge Node publishes NBIRTH, then
    DBIRTH for each device, then DDATA telemetry, then NDEATH on
    disconnect (via MQTT LWT).
    """
    st = (
        "&{connect: +{CONNACK_OK: "
        "&{nbirth: &{dbirth: "
        "+{ddata: &{ndeath: end}, "
        "ncmd: &{ndeath: end}}}}, "
        "CONNACK_FAIL: end}}"
    )
    return IndustrialProtocol(
        name="MQTT_SparkplugB_Edge",
        session_type_string=st,
        family="MQTT",
        description=(
            "Sparkplug B edge node lifecycle: CONNECT, NBIRTH, DBIRTH, "
            "DDATA or NCMD, NDEATH (last will)."
        ),
        iec62443_scope=("FR1_IdentificationAndAuthenticationControl",
                        "FR3_SystemIntegrity", "FR6_TimelyResponseToEvents"),
    )


def mqtt_tls_secured() -> IndustrialProtocol:
    """MQTT over TLS with mutual authentication for industrial brokers."""
    st = (
        "&{tlsHandshake: +{TLS_OK: "
        "&{connect: +{CONNACK_OK: "
        "&{subscribe: &{suback: "
        "&{publish: &{puback: "
        "&{disconnect: &{tlsClose: end}}}}}}, "
        "CONNACK_FAIL: &{tlsClose: end}}}, "
        "TLS_FAIL: end}}"
    )
    return IndustrialProtocol(
        name="MQTT_TLS_Mutual",
        session_type_string=st,
        family="MQTT",
        description=(
            "MQTT over TLS with mutual authentication: TLS handshake, "
            "CONNECT with client cert, SUBSCRIBE, PUBLISH with QoS 1, "
            "DISCONNECT, TLS close."
        ),
        iec62443_scope=("FR1_IdentificationAndAuthenticationControl",
                        "FR3_SystemIntegrity",
                        "FR4_DataConfidentiality",
                        "FR5_RestrictedDataFlow"),
    )


def all_industrial_protocols() -> Tuple[IndustrialProtocol, ...]:
    """Canonical benchmark set for Step 79."""
    return (
        opcua_client_session(),
        opcua_subscription(),
        opcua_secure_channel_renew(),
        mqtt_sparkplug_edge(),
        mqtt_tls_secured(),
    )


# ---------------------------------------------------------------------------
# Bidirectional morphisms phi / psi
# ---------------------------------------------------------------------------

#: Keywords in transition labels that map to IEC 62443 FRs.
_LABEL_TO_FR = {
    "tls": "FR1_IdentificationAndAuthenticationControl",
    "open": "FR1_IdentificationAndAuthenticationControl",
    "connect": "FR1_IdentificationAndAuthenticationControl",
    "hello": "FR1_IdentificationAndAuthenticationControl",
    "activate": "FR1_IdentificationAndAuthenticationControl",
    "auth": "FR1_IdentificationAndAuthenticationControl",
    "createsession": "FR1_IdentificationAndAuthenticationControl",
    "session": "FR2_UseControl",
    "read": "FR2_UseControl",
    "write": "FR3_SystemIntegrity",
    "browse": "FR2_UseControl",
    "publish": "FR3_SystemIntegrity",
    "subscribe": "FR5_RestrictedDataFlow",
    "ddata": "FR3_SystemIntegrity",
    "nbirth": "FR6_TimelyResponseToEvents",
    "dbirth": "FR6_TimelyResponseToEvents",
    "ndeath": "FR6_TimelyResponseToEvents",
    "ncmd": "FR2_UseControl",
    "renew": "FR4_DataConfidentiality",
    "channel": "FR4_DataConfidentiality",
    "monitored": "FR6_TimelyResponseToEvents",
    "suback": "FR5_RestrictedDataFlow",
    "puback": "FR3_SystemIntegrity",
    "close": "FR7_ResourceAvailability",
    "disconnect": "FR7_ResourceAvailability",
    "delete": "FR7_ResourceAvailability",
}


def _label_to_fr(label: str) -> str:
    """Map a transition label to an IEC 62443 FR by keyword match."""
    lab = label.lower()
    for key, fr in _LABEL_TO_FR.items():
        if key in lab:
            return fr
    return "FR3_SystemIntegrity"  # default: integrity monitoring


def _label_severity(label: str) -> str:
    lab = label.lower()
    if "fail" in lab or "reject" in lab or "ndeath" in lab:
        return "critical"
    if "renew" in lab or "activate" in lab or "open" in lab:
        return "warn"
    return "info"


def _state_action(state_id: int, label: str) -> str:
    lab = label.lower()
    if "fail" in lab or "reject" in lab:
        return f"alert_and_abort(state={state_id}, reason={label})"
    if "close" in lab or "disconnect" in lab or "death" in lab:
        return f"record_session_end(state={state_id})"
    if "open" in lab or "connect" in lab or "tls" in lab:
        return f"log_auth_event(state={state_id}, event={label})"
    return f"update_monitor(state={state_id}, transition={label})"


def phi_lattice_to_checks(ss: StateSpace) -> Tuple[SafetyCheck, ...]:
    """phi : L(S) -> SafetyAndSecurityChecks.

    For each transition (src -label-> tgt), emit a SafetyCheck keyed
    on the target state.  phi is total: every reachable state receives
    at least one check (via its entering transitions), plus the initial
    state receives a session-start check.
    """
    checks = []
    # Entry check for the initial state (top of the lattice).
    checks.append(SafetyCheck(
        state_id=ss.top,
        fr="FR1_IdentificationAndAuthenticationControl",
        action=f"log_session_start(state={ss.top})",
        severity="info",
    ))
    for src, label, tgt in ss.transitions:
        checks.append(SafetyCheck(
            state_id=tgt,
            fr=_label_to_fr(label),
            action=_state_action(tgt, label),
            severity=_label_severity(label),
        ))
    return tuple(checks)


def psi_checks_to_lattice(
    checks: Tuple[SafetyCheck, ...],
    ss: StateSpace,
) -> Tuple[int, ...]:
    """psi : SafetyAndSecurityChecks -> L(S).

    Recovers the reachable-state subset covered by a bag of checks.
    When applied to phi(L(S)), psi returns every reachable state;
    composition psi o phi = id_{reachable states}.
    """
    covered = set()
    reachable = _reachable_states(ss)
    for c in checks:
        if c.state_id in reachable:
            covered.add(c.state_id)
    return tuple(sorted(covered))


def _reachable_states(ss: StateSpace) -> set:
    return set(ss.reachable_from(ss.top))


def is_phi_total(ss: StateSpace, checks: Tuple[SafetyCheck, ...]) -> bool:
    """Every reachable state is covered by at least one check."""
    covered = {c.state_id for c in checks}
    return _reachable_states(ss).issubset(covered)


def is_psi_total(
    ss: StateSpace, checks: Tuple[SafetyCheck, ...]
) -> bool:
    """psi is total on the image of phi: every check's state is reachable."""
    reachable = _reachable_states(ss)
    return all(c.state_id in reachable for c in checks)


def is_galois_pair(ss: StateSpace) -> bool:
    """Check psi o phi = id on reachable states (Galois insertion).

    We check a necessary property: the composition recovers exactly
    the reachable-state set.  This is the defining property of the
    retraction (psi, phi) on the powerset lattice of states.
    """
    phi_checks = phi_lattice_to_checks(ss)
    recovered = set(psi_checks_to_lattice(phi_checks, ss))
    return recovered == _reachable_states(ss)


# ---------------------------------------------------------------------------
# Verification driver
# ---------------------------------------------------------------------------

def verify_industrial_protocol(
    proto: IndustrialProtocol,
) -> IndustrialAnalysisResult:
    ast = parse(proto.session_type_string)
    ss = build_statespace(ast)
    lat = check_lattice(ss)
    dist = check_distributive(ss)
    checks = phi_lattice_to_checks(ss)
    return IndustrialAnalysisResult(
        protocol=proto,
        ast=ast,
        state_space=ss,
        lattice_result=lat,
        distributivity=dist,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        safety_checks=checks,
        phi_total=is_phi_total(ss, checks),
        psi_total=is_psi_total(ss, checks),
        galois=is_galois_pair(ss),
        is_well_formed=lat.is_lattice,
    )


def format_industrial_report(res: IndustrialAnalysisResult) -> str:
    lines = [
        f"Industrial protocol: {res.protocol.name} ({res.protocol.family})",
        f"  states: {res.num_states}, transitions: {res.num_transitions}",
        f"  lattice: {res.is_well_formed}, distributive: {res.distributivity.is_distributive}",
        f"  phi total: {res.phi_total}, psi total: {res.psi_total}, galois: {res.galois}",
        f"  safety checks emitted: {len(res.safety_checks)}",
        "  IEC 62443 FR coverage:",
    ]
    frs = sorted({c.fr for c in res.safety_checks})
    for fr in frs:
        n = sum(1 for c in res.safety_checks if c.fr == fr)
        lines.append(f"    {fr}: {n}")
    return "\n".join(lines)
