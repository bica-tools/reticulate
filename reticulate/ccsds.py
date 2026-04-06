"""CCSDS Deep-Space Telecommand/Telemetry as Session Types (Step 802).

Models the CCSDS deep-space link stack -- TC Space Data Link Protocol
with COP-1 (FOP/FARM retransmission), TM Space Data Link Protocol,
Advanced Orbiting Systems (AOS), and the CCSDS File Delivery Protocol
(CFDP) -- as session types between a Mission Control Centre (MCC)
ground role and a spacecraft role.

Realistic features modelled:

* COP-1 FOP-1 / FARM-1 retransmission with sliding window and BD/BC frame
  acceptance, including the canonical "wait for CLCW" loop after a frame
  is rejected.
* TM downlink with virtual channels and Reed-Solomon / BCH outer
  framing markers (sync_marker, attached_sync, rs_decode).
* AOS multiplexing of M_PDU and B_PDU paths in parallel.
* CFDP class-1 (unacknowledged) and class-2 (acknowledged) put requests
  with file directives and EOF/Finished handshakes.
* Light-time delay windows where the spacecraft must operate
  autonomously (blackout) and resynchronise on the next ground pass.

The lattice ``L(S)`` is connected to a ``LinkSupervisor`` domain via a
*bidirectional pair*

.. math::
   \\varphi : L(S) \\to \\text{LinkSupervisor}, \\qquad
   \\psi : \\text{LinkSupervisor} \\to L(S).

``\\varphi`` ships every protocol state to the supervisor mode the deep
space link should be in (ACQUIRING, LOCKED, COMMANDING, RETRANSMIT,
DOWNLINKING, BLACKOUT, SAFE_HOLD, FILE_TRANSFER, IDLE).  ``\\psi`` takes
a supervisor mode and returns the set of protocol states in which that
mode is live, supporting link-budget contingency analysis, FOP
retransmission correctness, and blackout-window safety arguments.

Usage::

    from reticulate.ccsds import (
        tc_cop1_fop_farm,
        tm_downlink_virtual_channel,
        aos_mpdu_bpdu_parallel,
        cfdp_class2_put,
        verify_ccsds_protocol,
        analyze_all_ccsds,
        phi_link_mode,
        psi_mode_to_states,
    )
    proto = tc_cop1_fop_farm()
    result = verify_ccsds_protocol(proto)
    print(format_ccsds_report(result))
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


# Deep-space link supervisor modes (the codomain of phi).
LINK_MODES: tuple[str, ...] = (
    "IDLE",
    "ACQUIRING",
    "LOCKED",
    "COMMANDING",
    "RETRANSMIT",
    "DOWNLINKING",
    "BLACKOUT",
    "SAFE_HOLD",
    "FILE_TRANSFER",
    "TERMINATED",
)


# Recognised CCSDS roles (parties in the multiparty session).
CCSDS_ROLES: tuple[str, ...] = (
    "ground_mcc",
    "spacecraft",
    "dsn_station",
    "cfdp_entity",
)


# CCSDS specifications referenced.
CCSDS_SPECS: tuple[str, ...] = (
    "CCSDS 232.0-B-3 (TC Space Data Link Protocol)",
    "CCSDS 232.1-B-2 (TC Synchronization and Channel Coding)",
    "CCSDS 132.0-B-2 (TM Space Data Link Protocol)",
    "CCSDS 131.0-B-3 (TM Synchronization and Channel Coding)",
    "CCSDS 732.0-B-3 (AOS Space Data Link Protocol)",
    "CCSDS 727.0-B-5 (CFDP File Delivery Protocol)",
    "CCSDS 232.0-B-3 Annex C (COP-1)",
)


@dataclass(frozen=True)
class CcsdsProtocol:
    """A CCSDS deep-space protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "TcCop1FopFarm").
        layer: Layer ("TC", "TM", "AOS", "CFDP", "COP1", "RF").
        roles: CCSDS roles (parties) involved.
        session_type_string: Session-type encoding.
        spec_reference: CCSDS Blue/Magenta Book reference.
        description: Free-text description.
    """
    name: str
    layer: str
    roles: tuple[str, ...]
    session_type_string: str
    spec_reference: str
    description: str

    def __post_init__(self) -> None:
        if self.layer not in ("TC", "TM", "AOS", "CFDP", "COP1", "RF"):
            raise ValueError(
                f"Invalid layer {self.layer!r}; expected one of "
                f"TC, TM, AOS, CFDP, COP1, RF."
            )


@dataclass(frozen=True)
class LinkMode:
    """A deep-space link supervisor mode (codomain of phi).

    Attributes:
        kind: Mode label from LINK_MODES.
        rationale: Human-readable justification.
    """
    kind: str
    rationale: str


@dataclass(frozen=True)
class CcsdsAnalysisResult:
    """Complete analysis result for a CCSDS protocol."""
    protocol: CcsdsProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    num_states: int
    num_transitions: int
    is_well_formed: bool
    link_modes: dict[int, LinkMode]
    deadlock_states: tuple[int, ...]


# ---------------------------------------------------------------------------
# CCSDS protocol definitions
# ---------------------------------------------------------------------------


def tc_cop1_fop_farm() -> CcsdsProtocol:
    """COP-1 FOP/FARM telecommand retransmission protocol.

    Models the canonical TC frame uplink: the FOP at the ground side
    sends a frame, waits for the CLCW reported by FARM at the spacecraft.
    Outcomes are ACCEPTED (advance window), REJECTED (retransmit),
    LOCKOUT (enter LOCKOUT, recover via BC frame), or TIMEOUT
    (retransmit on next pass).
    """
    return CcsdsProtocol(
        name="TcCop1FopFarm",
        layer="COP1",
        roles=("ground_mcc", "spacecraft"),
        session_type_string=(
            "&{sendBdFrame: &{awaitClcw: "
            "+{ACCEPTED: &{advanceWindow: end}, "
            "REJECTED: &{retransmit: +{ACCEPTED: &{advanceWindow: end}, "
            "LOCKOUT: &{sendBcUnlock: &{awaitClcw: "
            "+{UNLOCKED: &{advanceWindow: end}, "
            "TIMEOUT: &{abortFop: end}}}}}}, "
            "TIMEOUT: &{abortFop: end}}}}"
        ),
        spec_reference="CCSDS 232.0-B-3 Annex C (COP-1, FOP-1/FARM-1)",
        description=(
            "COP-1 FOP/FARM retransmission: send BD frame, await CLCW, "
            "retransmit on REJECTED, send BC unlock on LOCKOUT, abort on "
            "TIMEOUT. Models the canonical sliding-window correctness."
        ),
    )


def tc_space_data_link() -> CcsdsProtocol:
    """TC Space Data Link Protocol uplink (single TC frame).

    Ground forms a TC transfer frame, applies BCH coding, modulates
    onto the uplink carrier, and either receives an ACK CLCW or a NAK.
    """
    return CcsdsProtocol(
        name="TcSpaceDataLink",
        layer="TC",
        roles=("ground_mcc", "spacecraft"),
        session_type_string=(
            "&{formTcFrame: &{bchEncode: &{modulateUplink: "
            "+{CARRIER_LOCKED: &{transmit: &{awaitClcw: "
            "+{ACK: &{frameAccepted: end}, "
            "NAK: &{frameRejected: end}}}}, "
            "NO_LOCK: &{linkLossAbort: end}}}}}"
        ),
        spec_reference="CCSDS 232.0-B-3 / 232.1-B-2",
        description=(
            "TC SDLP uplink: form frame, BCH encode, modulate, await "
            "CLCW, accept or reject."
        ),
    )


def tm_downlink_virtual_channel() -> CcsdsProtocol:
    """TM Space Data Link Protocol downlink with virtual channel.

    Spacecraft fills a TM transfer frame from a virtual channel, applies
    Reed-Solomon (255,223) outer coding, modulates the downlink carrier,
    and the ground demodulates / decodes / extracts the VCID payload.
    """
    return CcsdsProtocol(
        name="TmDownlinkVirtualChannel",
        layer="TM",
        roles=("spacecraft", "dsn_station", "ground_mcc"),
        session_type_string=(
            "&{fillTmFrame: &{rsEncode: &{attachSyncMarker: "
            "&{modulateDownlink: +{ACQUIRED: "
            "&{demodulate: &{rsDecode: +{CRC_OK: "
            "&{extractVcid: &{deliverPayload: end}}, "
            "CRC_BAD: &{frameDiscard: end}}}}, "
            "NO_SIGNAL: &{linkLossAbort: end}}}}}}"
        ),
        spec_reference="CCSDS 132.0-B-2 / 131.0-B-3",
        description=(
            "TM downlink: fill frame, RS encode, attach sync marker, "
            "modulate; ground demodulates, RS-decodes, extracts VCID."
        ),
    )


def aos_mpdu_bpdu_parallel() -> CcsdsProtocol:
    """AOS Space Data Link with parallel M_PDU and B_PDU paths.

    AOS multiplexes packetised payloads (M_PDU virtual channels) and
    bitstream payloads (B_PDU virtual channels) in parallel.  The
    resulting product lattice ``L(M_PDU) x L(B_PDU)`` compresses the
    interleavings of the two pipelines.
    """
    return CcsdsProtocol(
        name="AosMpduBpduParallel",
        layer="AOS",
        roles=("spacecraft", "dsn_station"),
        session_type_string=(
            "(&{mpduPacketize: &{mpduInsertHeader: "
            "&{mpduDownlink: end}}} "
            "|| &{bpduChunk: &{bpduInsertHeader: "
            "&{bpduDownlink: end}}})"
        ),
        spec_reference="CCSDS 732.0-B-3 (AOS SDLP)",
        description=(
            "AOS parallel pipelines: M_PDU (packets) and B_PDU "
            "(bitstream) downlink simultaneously. Product lattice "
            "4 x 4 = 16 compresses the 4!*C(8,4)=1680 raw interleavings."
        ),
    )


def cfdp_class2_put() -> CcsdsProtocol:
    """CFDP Class 2 (acknowledged) Put.put request.

    Ground sends Metadata + File Data PDUs, then EOF, awaits Finished;
    on NAK, retransmits the missing data PDU; on success closes the
    transaction.
    """
    return CcsdsProtocol(
        name="CfdpClass2Put",
        layer="CFDP",
        roles=("ground_mcc", "cfdp_entity", "spacecraft"),
        session_type_string=(
            "&{sendMetadata: &{sendFileData: &{sendEof: &{awaitFinished: "
            "+{COMPLETE: &{closeTransaction: end}, "
            "NAK_LIST: &{retransmitData: &{awaitFinished: "
            "+{COMPLETE: &{closeTransaction: end}, "
            "TIMEOUT: &{abortCfdp: end}}}}}}}}}"
        ),
        spec_reference="CCSDS 727.0-B-5 (CFDP Class 2)",
        description=(
            "CFDP class-2 acknowledged put: metadata, file data, EOF, "
            "Finished; retransmit on NAK_LIST; abort on TIMEOUT."
        ),
    )


def cfdp_class1_put() -> CcsdsProtocol:
    """CFDP Class 1 (unreliable) Put.put request."""
    return CcsdsProtocol(
        name="CfdpClass1Put",
        layer="CFDP",
        roles=("ground_mcc", "cfdp_entity"),
        session_type_string=(
            "&{sendMetadata: &{sendFileData: "
            "&{sendEof: &{closeTransaction: end}}}}"
        ),
        spec_reference="CCSDS 727.0-B-5 (CFDP Class 1)",
        description=(
            "CFDP class-1 unacknowledged put: metadata, file data, EOF, "
            "close. No retransmission, fire-and-forget."
        ),
    )


def blackout_autonomous_window() -> CcsdsProtocol:
    """Spacecraft autonomous operations during a blackout window.

    During Mars-class light-time delay or solar-conjunction blackout,
    the spacecraft must run housekeeping, science, and safe-mode checks
    autonomously, then resynchronise on the next ground pass.
    """
    return CcsdsProtocol(
        name="BlackoutAutonomousWindow",
        layer="RF",
        roles=("spacecraft",),
        session_type_string=(
            "&{enterBlackout: &{runHousekeeping: "
            "+{NOMINAL: &{collectScience: &{awaitNextPass: "
            "+{REACQUIRED: &{resyncCop1: end}, "
            "MISSED: &{enterSafeHold: end}}}}, "
            "ANOMALY: &{enterSafeHold: end}}}}"
        ),
        spec_reference="Mars-class operations: solar-conjunction blackout",
        description=(
            "Autonomous blackout window: housekeeping, science, await "
            "next pass; safe-hold on anomaly or missed pass."
        ),
    )


def rf_acquire_lock() -> CcsdsProtocol:
    """Carrier and symbol acquisition before any frame transfer."""
    return CcsdsProtocol(
        name="RfAcquireLock",
        layer="RF",
        roles=("dsn_station", "spacecraft"),
        session_type_string=(
            "&{searchCarrier: +{CARRIER_LOCKED: "
            "&{symbolSync: +{SYMBOL_LOCKED: "
            "&{frameSync: +{FRAME_LOCKED: &{linkUp: end}, "
            "NO_FRAME_SYNC: &{acquireFail: end}}}, "
            "NO_SYMBOL_SYNC: &{acquireFail: end}}}, "
            "NO_CARRIER: &{acquireFail: end}}}"
        ),
        spec_reference="CCSDS 131.0-B-3 / 232.1-B-2 (acquisition)",
        description=(
            "RF acquisition: carrier search, symbol sync, frame sync, "
            "or fail at any stage."
        ),
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

ALL_TC_PROTOCOLS: tuple[CcsdsProtocol, ...] = (
    tc_cop1_fop_farm(),
    tc_space_data_link(),
)

ALL_TM_PROTOCOLS: tuple[CcsdsProtocol, ...] = (
    tm_downlink_virtual_channel(),
)

ALL_AOS_PROTOCOLS: tuple[CcsdsProtocol, ...] = (
    aos_mpdu_bpdu_parallel(),
)

ALL_CFDP_PROTOCOLS: tuple[CcsdsProtocol, ...] = (
    cfdp_class1_put(),
    cfdp_class2_put(),
)

ALL_RF_PROTOCOLS: tuple[CcsdsProtocol, ...] = (
    rf_acquire_lock(),
    blackout_autonomous_window(),
)

ALL_CCSDS_PROTOCOLS: tuple[CcsdsProtocol, ...] = (
    ALL_TC_PROTOCOLS
    + ALL_TM_PROTOCOLS
    + ALL_AOS_PROTOCOLS
    + ALL_CFDP_PROTOCOLS
    + ALL_RF_PROTOCOLS
)


# ---------------------------------------------------------------------------
# Bidirectional morphisms phi : L(S) -> LinkSupervisor and psi back
# ---------------------------------------------------------------------------


_MODE_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("abortFop", "abortCfdp", "linkLossAbort", "acquireFail",
      "frameDiscard", "frameRejected"), "SAFE_HOLD"),
    (("retransmit", "sendBcUnlock"), "RETRANSMIT"),
    (("enterBlackout", "runHousekeeping", "collectScience",
      "awaitNextPass", "enterSafeHold"), "BLACKOUT"),
    (("sendMetadata", "sendFileData", "sendEof", "awaitFinished",
      "closeTransaction", "retransmitData"), "FILE_TRANSFER"),
    (("rsEncode", "rsDecode", "attachSyncMarker", "modulateDownlink",
      "demodulate", "extractVcid", "deliverPayload",
      "fillTmFrame", "mpduPacketize", "mpduInsertHeader",
      "mpduDownlink", "bpduChunk", "bpduInsertHeader", "bpduDownlink"),
     "DOWNLINKING"),
    (("sendBdFrame", "awaitClcw", "advanceWindow",
      "formTcFrame", "bchEncode", "modulateUplink", "transmit",
      "frameAccepted"), "COMMANDING"),
    (("searchCarrier", "symbolSync", "frameSync", "linkUp"), "ACQUIRING"),
    (("resyncCop1",), "LOCKED"),
)


def phi_link_mode(
    protocol: CcsdsProtocol,
    ss: StateSpace,
    state: int,
) -> LinkMode:
    """phi : L(S) -> LinkSupervisor (state-indexed).

    Maps a protocol state to the dominant link supervisor mode.
    """
    if state == ss.bottom:
        return LinkMode(
            kind="TERMINATED",
            rationale=(
                f"State {state} is the lattice bottom; the deep-space "
                f"session has terminated (success or clean abort)."
            ),
        )
    if state == ss.top:
        return LinkMode(
            kind="IDLE",
            rationale=(
                f"State {state} is the lattice top; no command has yet "
                f"been issued."
            ),
        )

    incident_labels: list[str] = []
    for src, lbl, tgt in ss.transitions:
        if src == state or tgt == state:
            incident_labels.append(lbl)

    for keywords, kind in _MODE_KEYWORDS:
        for kw in keywords:
            if any(kw in lbl for lbl in incident_labels):
                return LinkMode(
                    kind=kind,
                    rationale=(
                        f"State {state} mapped to {kind} via incident "
                        f"label containing {kw!r}."
                    ),
                )

    return LinkMode(
        kind="LOCKED",
        rationale=f"State {state} has no distinguishing label; default LOCKED.",
    )


def phi_all_states(
    protocol: CcsdsProtocol,
    ss: StateSpace,
) -> dict[int, LinkMode]:
    """Apply phi to every state of the state space."""
    return {s: phi_link_mode(protocol, ss, s) for s in ss.states}


def psi_mode_to_states(
    protocol: CcsdsProtocol,
    ss: StateSpace,
    mode_kind: str,
) -> frozenset[int]:
    """psi : LinkSupervisor -> L(S)."""
    modes = phi_all_states(protocol, ss)
    return frozenset(s for s, m in modes.items() if m.kind == mode_kind)


def classify_morphism_pair(
    protocol: CcsdsProtocol,
    ss: StateSpace,
) -> str:
    """Classify the pair (phi, psi)."""
    modes = phi_all_states(protocol, ss)
    kinds = {m.kind for m in modes.values()}
    if len(kinds) == len(ss.states):
        return "isomorphism"
    if len(kinds) < len(ss.states):
        for kind in kinds:
            states = psi_mode_to_states(protocol, ss, kind)
            for s in states:
                if phi_link_mode(protocol, ss, s).kind != kind:
                    return "galois"
        return "section-retraction"
    return "projection"


# ---------------------------------------------------------------------------
# Deadlock detection
# ---------------------------------------------------------------------------


def detect_deadlocks(ss: StateSpace) -> tuple[int, ...]:
    """Detect deadlocked states (no outgoing transitions, not bottom)."""
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


def ccsds_to_session_type(protocol: CcsdsProtocol) -> SessionType:
    """Parse the protocol's session-type string."""
    return parse(protocol.session_type_string)


def verify_ccsds_protocol(protocol: CcsdsProtocol) -> CcsdsAnalysisResult:
    """Run the full verification pipeline for a single CCSDS protocol."""
    ast = ccsds_to_session_type(protocol)
    ss = build_statespace(ast)
    lr = check_lattice(ss)
    dist = check_distributive(ss)
    modes = phi_all_states(protocol, ss)
    deadlocks = detect_deadlocks(ss)
    return CcsdsAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_well_formed=lr.is_lattice,
        link_modes=modes,
        deadlock_states=deadlocks,
    )


def analyze_all_ccsds() -> list[CcsdsAnalysisResult]:
    """Verify every pre-defined CCSDS protocol."""
    return [verify_ccsds_protocol(p) for p in ALL_CCSDS_PROTOCOLS]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_ccsds_report(result: CcsdsAnalysisResult) -> str:
    lines: list[str] = []
    proto = result.protocol
    lines.append("=" * 72)
    lines.append(f"  CCSDS DEEP-SPACE PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")
    lines.append("--- Protocol Details ---")
    lines.append(f"  Layer: {proto.layer}")
    lines.append(f"  Spec:  {proto.spec_reference}")
    for r in proto.roles:
        lines.append(f"  Role:  {r}")
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
    lines.append("--- Link Modes (phi) ---")
    counts: dict[str, int] = {}
    for m in result.link_modes.values():
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


def format_ccsds_summary(results: list[CcsdsAnalysisResult]) -> str:
    lines: list[str] = []
    lines.append("=" * 82)
    lines.append("  CCSDS DEEP-SPACE STACK CERTIFICATION SUMMARY")
    lines.append("=" * 82)
    lines.append("")
    header = (
        f"  {'Protocol':<28} {'Layer':<6} "
        f"{'States':>6} {'Trans':>6} {'Lattice':>8} {'Dist':>6} {'Deadlk':>7}"
    )
    lines.append(header)
    lines.append("  " + "-" * 78)
    for r in results:
        lines.append(
            f"  {r.protocol.name:<28} "
            f"{r.protocol.layer:<6} "
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
