"""FIX 4.4 and ISO 20022 financial protocol verification (Step 73).

Models financial messaging protocols (FIX 4.4 order lifecycle and ISO 20022
payment initiation / clearing / cash reporting) as session types, enabling
lattice-based verification, audit-trail generation and regulatory
conformance testing.

FIX (Financial Information eXchange) 4.4 is the dominant pre-trade and
trade protocol in global equity / FX / futures markets.  ISO 20022 is
the ISO standard for financial services messaging adopted by SWIFT,
SEPA, CHAPS, Fedwire and most modern real-time gross settlement (RTGS)
systems.  Both define strict message-state machines: once an order is
FILLED, it cannot transition to CANCELLED; once a pacs.008 credit
transfer is SETTLED, no further pacs.002 status updates are accepted.

This module encodes each protocol as a session type, builds the
state space (reticulate), checks lattice properties, and produces a
bidirectional morphism pair into a ComplianceActions domain of
audit / monitoring / regulatory testing operations.

Usage:
    from reticulate.fix_iso20022 import (
        new_order_single_protocol,
        pain001_protocol,
        verify_financial_protocol,
        format_financial_report,
    )
    p = new_order_single_protocol()
    result = verify_financial_protocol(p)
    print(format_financial_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.coverage import compute_coverage, CoverageResult
from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinancialProtocol:
    """A named financial messaging protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., ``NewOrderSingle``).
        family: Protocol family, ``FIX`` or ``ISO20022``.
        messages: Tuple of message-type codes used in this protocol.
        transitions: Human-readable transition descriptions.
        session_type_string: Session-type encoding.
        description: Free-text description of the protocol purpose.
    """
    name: str
    family: str
    messages: tuple[str, ...]
    transitions: tuple[str, ...]
    session_type_string: str
    description: str


@dataclass(frozen=True)
class ComplianceAction:
    """A compliance / audit action attached to a state-space point.

    Represents what a bank, regulator or monitoring system should do
    when the protocol reaches a particular state.  Used by the
    bidirectional morphism ``phi`` and its pseudo-inverse ``psi``.

    Attributes:
        state_id: State identifier in the reticulate.
        action: Kind of action (``audit``, ``monitor``, ``block``,
            ``settle``, ``reconcile``, ``archive``).
        rationale: Natural-language explanation of why.
    """
    state_id: int
    action: str
    rationale: str


@dataclass(frozen=True)
class FinancialAnalysisResult:
    """Complete analysis result for a financial protocol."""
    protocol: FinancialProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    enumeration: EnumerationResult
    test_source: str
    coverage: CoverageResult
    compliance_actions: tuple[ComplianceAction, ...]
    num_states: int
    num_transitions: int
    num_valid_paths: int
    num_violations: int
    is_well_formed: bool


# ---------------------------------------------------------------------------
# FIX 4.4 protocol definitions
# ---------------------------------------------------------------------------

def new_order_single_protocol() -> FinancialProtocol:
    """FIX 4.4 New Order Single (D) order lifecycle.

    Client submits a NewOrderSingle (D); the exchange responds with
    an ExecutionReport (8) that is either ACK, REJECT, FILL, or
    PARTIAL.  In the PARTIAL case the order remains open for further
    fills or cancellation.
    """
    return FinancialProtocol(
        name="NewOrderSingle",
        family="FIX",
        messages=("D", "8", "F", "9"),
        transitions=(
            "newOrderSingle: Tag 35=D submission",
            "ack: Tag 35=8 ExecutionReport OrdStatus=0",
            "reject: Tag 35=8 ExecType=8",
            "fill: Tag 35=8 ExecType=F OrdStatus=2",
            "partial: Tag 35=8 ExecType=F OrdStatus=1",
            "cancelRequest: Tag 35=F",
            "cancelled: Tag 35=8 ExecType=4",
            "cancelReject: Tag 35=9",
        ),
        session_type_string=(
            "&{newOrderSingle: "
            "+{ACK: &{cancelRequest: +{CANCELLED: end, CANCELREJECT: end}}, "
            "REJECT: end, "
            "FILL: end, "
            "PARTIAL: &{cancelRequest: +{CANCELLED: end, CANCELREJECT: end}}}}"
        ),
        description=(
            "FIX 4.4 NewOrderSingle lifecycle: submission, exchange "
            "response (ack/reject/fill/partial), optional cancel "
            "request with confirmation or reject."
        ),
    )


def execution_report_protocol() -> FinancialProtocol:
    """FIX 4.4 ExecutionReport multi-fill sequence.

    Models a working order that receives multiple partial fills and
    then either a final fill or a cancellation.  The monitoring
    angle: every ExecutionReport must be mapped to an audit event
    and reconciled against the order book.
    """
    return FinancialProtocol(
        name="ExecutionReport",
        family="FIX",
        messages=("8",),
        transitions=(
            "report: Receive ExecutionReport",
            "PARTIAL: Partial fill",
            "DONE: Final fill / done-for-day",
            "EXPIRED: Order expired",
        ),
        session_type_string=(
            "&{report: +{PARTIAL: &{report: +{DONE: end, EXPIRED: end}}, "
            "DONE: end, EXPIRED: end}}"
        ),
        description=(
            "FIX 4.4 ExecutionReport sequence: a working order "
            "receives a first report that is either PARTIAL, DONE, "
            "or EXPIRED; if PARTIAL, a second report terminates."
        ),
    )


def order_cancel_protocol() -> FinancialProtocol:
    """FIX 4.4 OrderCancelRequest (F) / Reject (9) protocol."""
    return FinancialProtocol(
        name="OrderCancelRequest",
        family="FIX",
        messages=("F", "8", "9"),
        transitions=(
            "cancelRequest: Tag 35=F submission",
            "CANCELLED: ExecutionReport ExecType=4",
            "CANCELREJECT: OrderCancelReject Tag 35=9",
        ),
        session_type_string=(
            "&{cancelRequest: +{CANCELLED: end, CANCELREJECT: end}}"
        ),
        description=(
            "FIX 4.4 order cancel request: submit cancel, receive "
            "either a CANCELLED ExecutionReport or an "
            "OrderCancelReject."
        ),
    )


# ---------------------------------------------------------------------------
# ISO 20022 protocol definitions
# ---------------------------------------------------------------------------

def pain001_protocol() -> FinancialProtocol:
    """ISO 20022 pain.001 CustomerCreditTransferInitiation.

    A corporate customer submits a pain.001 payment initiation.  The
    debtor bank validates and responds with a pain.002
    PaymentStatusReport that is either ACCEPTED (ACSP) or REJECTED
    (RJCT).  Accepted initiations proceed to interbank clearing.
    """
    return FinancialProtocol(
        name="pain001",
        family="ISO20022",
        messages=("pain.001", "pain.002"),
        transitions=(
            "initiate: Submit pain.001",
            "validate: Bank validation",
            "ACCEPTED: pain.002 status=ACSP",
            "REJECTED: pain.002 status=RJCT",
            "clear: Hand off to pacs.008 clearing",
            "SETTLED: Settlement confirmation",
            "RETURNED: pacs.004 payment return",
        ),
        session_type_string=(
            "&{initiate: &{validate: +{ACCEPTED: &{clear: "
            "+{SETTLED: end, RETURNED: end}}, REJECTED: end}}}"
        ),
        description=(
            "ISO 20022 pain.001 customer credit transfer: initiate, "
            "bank validation (accept or reject), clearing and "
            "settlement (or return)."
        ),
    )


def pacs008_protocol() -> FinancialProtocol:
    """ISO 20022 pacs.008 FIToFICustomerCreditTransfer (interbank)."""
    return FinancialProtocol(
        name="pacs008",
        family="ISO20022",
        messages=("pacs.008", "pacs.002", "pacs.004"),
        transitions=(
            "send: pacs.008 interbank credit transfer",
            "ackClear: pacs.002 ACSC (accepted settled clearing)",
            "rejectClear: pacs.002 RJCT (rejected)",
            "RETURNED: pacs.004 payment return",
            "SETTLED: Settlement finalised",
        ),
        session_type_string=(
            "&{send: +{ACKCLEAR: +{SETTLED: end, RETURNED: end}, "
            "REJECTCLEAR: end}}"
        ),
        description=(
            "ISO 20022 pacs.008 interbank clearing: send credit "
            "transfer, receive pacs.002 status (accept or reject), "
            "finalise or return."
        ),
    )


def camt054_protocol() -> FinancialProtocol:
    """ISO 20022 camt.054 BankToCustomerDebitCreditNotification."""
    return FinancialProtocol(
        name="camt054",
        family="ISO20022",
        messages=("camt.054",),
        transitions=(
            "notify: camt.054 debit/credit notification",
            "reconcile: Customer reconciles against ledger",
            "MATCHED: Entry matched to open item",
            "UNMATCHED: Entry kept in suspense",
        ),
        session_type_string=(
            "&{notify: &{reconcile: +{MATCHED: end, UNMATCHED: end}}}"
        ),
        description=(
            "ISO 20022 camt.054 debit/credit notification: the bank "
            "notifies a customer of a movement; the customer "
            "reconciles against its ledger."
        ),
    )


def pain008_protocol() -> FinancialProtocol:
    """ISO 20022 pain.008 CustomerDirectDebitInitiation (SEPA DD)."""
    return FinancialProtocol(
        name="pain008",
        family="ISO20022",
        messages=("pain.008", "pain.002", "pacs.003"),
        transitions=(
            "initiateDD: pain.008 direct debit initiation",
            "validateDD: Creditor bank validation",
            "ACCEPTED: pain.002 ACSP",
            "REJECTED: pain.002 RJCT",
            "collect: pacs.003 interbank collection",
            "SETTLED: Funds settled",
            "RETURNED: pacs.004 return (R-transaction)",
        ),
        session_type_string=(
            "&{initiateDD: &{validateDD: +{ACCEPTED: &{collect: "
            "+{SETTLED: end, RETURNED: end}}, REJECTED: end}}}"
        ),
        description=(
            "ISO 20022 pain.008 direct debit: initiate, validate, "
            "collect and either settle or return (R-transaction)."
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_FINANCIAL_PROTOCOLS: tuple[FinancialProtocol, ...] = (
    new_order_single_protocol(),
    execution_report_protocol(),
    order_cancel_protocol(),
    pain001_protocol(),
    pacs008_protocol(),
    camt054_protocol(),
    pain008_protocol(),
)
"""All pre-defined financial messaging protocols."""


# ---------------------------------------------------------------------------
# Bidirectional morphism: L(S) <-> ComplianceActions
# ---------------------------------------------------------------------------

#: Transition labels that mark a terminal settlement / finalisation.
_TERMINAL_SETTLE_LABELS: frozenset[str] = frozenset({
    "FILL", "SETTLED", "MATCHED",
})

#: Labels that indicate a reject / block / return.
_TERMINAL_REJECT_LABELS: frozenset[str] = frozenset({
    "REJECT", "REJECTED", "REJECTCLEAR", "RJCT",
    "CANCELREJECT", "UNMATCHED", "EXPIRED",
})

#: Labels that indicate a cancel / return.
_TERMINAL_CANCEL_LABELS: frozenset[str] = frozenset({
    "CANCELLED", "RETURNED",
})


def phi_state_to_action(
    ss: StateSpace,
    state_id: int,
) -> ComplianceAction:
    """Forward morphism phi: L(S) -> ComplianceActions.

    Maps a state-space point to the compliance / audit action the
    bank or regulator should take when the protocol reaches it.

    The mapping is determined by the incoming transition labels to
    ``state_id`` and whether the state is terminal (``bottom``) or
    initial (``top``):

    * ``top``: ``audit`` the incoming request (KYC, pre-trade check).
    * terminal reached by a settle label: ``settle`` and ``archive``.
    * terminal reached by a reject label: ``block``.
    * terminal reached by a cancel / return label: ``reconcile``.
    * non-terminal, non-top states: ``monitor``.
    """
    if state_id == ss.top:
        return ComplianceAction(
            state_id=state_id,
            action="audit",
            rationale=(
                "Initial state: run pre-trade / KYC / sanctions "
                "checks on incoming message."
            ),
        )

    incoming = [lbl for (s, lbl, t) in ss.transitions if t == state_id]

    is_terminal = state_id == ss.bottom or not any(
        src == state_id for (src, _, _) in ss.transitions
    )

    if is_terminal:
        # Priority: reject > settle > cancel, reflecting conservative
        # compliance posture (block first, reconcile last).
        if any(lbl in _TERMINAL_REJECT_LABELS for lbl in incoming):
            return ComplianceAction(
                state_id=state_id,
                action="block",
                rationale=(
                    "Terminal reject state: record reject reason, "
                    "notify upstream, no settlement booking."
                ),
            )
        if any(lbl in _TERMINAL_SETTLE_LABELS for lbl in incoming):
            return ComplianceAction(
                state_id=state_id,
                action="settle",
                rationale=(
                    "Terminal settle state: finalise books, archive "
                    "for regulatory retention (MiFID II RTS 22, "
                    "ISO 20022 retention)."
                ),
            )
        if any(lbl in _TERMINAL_CANCEL_LABELS for lbl in incoming):
            return ComplianceAction(
                state_id=state_id,
                action="reconcile",
                rationale=(
                    "Terminal cancel / return state: reconcile "
                    "pending items and release reservation."
                ),
            )
        return ComplianceAction(
            state_id=state_id,
            action="archive",
            rationale=(
                "Terminal state: archive audit trail."
            ),
        )

    return ComplianceAction(
        state_id=state_id,
        action="monitor",
        rationale=(
            "Intermediate state: monitor SLA timers and surveillance "
            "rules until next transition."
        ),
    )


def phi(ss: StateSpace) -> tuple[ComplianceAction, ...]:
    """Forward morphism applied to every state in the reticulate."""
    return tuple(phi_state_to_action(ss, s) for s in sorted(ss.states))


def psi(
    ss: StateSpace,
    action: str,
) -> frozenset[int]:
    """Backward morphism psi: ComplianceActions -> 2^L(S).

    Returns the set of states mapped to ``action`` by ``phi``.  This
    is a Galois-style pseudo-inverse: ``psi`` recovers a preimage
    set rather than a single point because ``phi`` is generally not
    injective.
    """
    return frozenset(
        s for s in ss.states if phi_state_to_action(ss, s).action == action
    )


def is_phi_psi_galois(ss: StateSpace) -> bool:
    """Check that ``(phi, psi)`` form a Galois connection.

    Since ``phi`` is a function L(S) -> ComplianceActions and ``psi``
    returns the preimage, the pair is trivially a Galois insertion:
    for every state ``s``, ``s in psi(phi(s))``; and for every
    action ``a`` that appears in the image, ``phi(psi(a)) = {a}``.
    We verify both conditions.
    """
    actions = tuple(phi_state_to_action(ss, s) for s in ss.states)
    action_names = {a.action for a in actions}

    for s in ss.states:
        a = phi_state_to_action(ss, s).action
        if s not in psi(ss, a):
            return False

    for a in action_names:
        pre = psi(ss, a)
        if not pre:
            continue
        images = {phi_state_to_action(ss, s).action for s in pre}
        if images != {a}:
            return False

    return True


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def financial_to_session_type(protocol: FinancialProtocol) -> SessionType:
    """Parse the protocol's session-type string into an AST."""
    return parse(protocol.session_type_string)


def verify_financial_protocol(
    protocol: FinancialProtocol,
    config: TestGenConfig | None = None,
) -> FinancialAnalysisResult:
    """Run the full verification pipeline on a financial protocol."""
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}ConformanceTest",
            package_name=f"com.{protocol.family.lower()}.conformance",
        )

    ast = financial_to_session_type(protocol)
    ss = build_statespace(ast)

    lr = check_lattice(ss)
    dist = check_distributive(ss)

    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.session_type_string)
    cov = compute_coverage(ss, result=enum)

    actions = phi(ss)

    return FinancialAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        enumeration=enum,
        test_source=test_src,
        coverage=cov,
        compliance_actions=actions,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_valid_paths=len(enum.valid_paths),
        num_violations=len(enum.violations),
        is_well_formed=lr.is_lattice,
    )


def verify_all_financial_protocols() -> list[FinancialAnalysisResult]:
    """Verify all pre-defined financial protocols."""
    return [verify_financial_protocol(p) for p in ALL_FINANCIAL_PROTOCOLS]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_financial_report(result: FinancialAnalysisResult) -> str:
    """Format a FinancialAnalysisResult as structured text."""
    lines: list[str] = []
    p = result.protocol

    lines.append("=" * 72)
    lines.append(f"  FINANCIAL PROTOCOL REPORT: {p.family} / {p.name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {p.description}")
    lines.append("")

    lines.append("--- Messages ---")
    for m in p.messages:
        lines.append(f"  - {m}")
    lines.append("")

    lines.append("--- Session Type ---")
    lines.append(f"  {p.session_type_string}")
    lines.append("")

    lines.append("--- State Space ---")
    lines.append(f"  States:      {result.num_states}")
    lines.append(f"  Transitions: {result.num_transitions}")
    lines.append(f"  Top:         {result.state_space.top}")
    lines.append(f"  Bottom:      {result.state_space.bottom}")
    lines.append("")

    lattice_str = "YES" if result.lattice_result.is_lattice else "NO"
    dist_str = "yes" if result.distributivity.is_distributive else "no"
    lines.append("--- Lattice Properties ---")
    lines.append(f"  Is lattice:     {lattice_str}")
    lines.append(f"  Has top:        {result.lattice_result.has_top}")
    lines.append(f"  Has bottom:     {result.lattice_result.has_bottom}")
    lines.append(f"  Distributive:   {dist_str}")
    lines.append("")

    lines.append("--- Compliance Actions (phi) ---")
    for a in result.compliance_actions:
        lines.append(f"  state {a.state_id:>3d} -> {a.action:<10s}  {a.rationale}")
    lines.append("")

    lines.append("--- Conformance Tests ---")
    lines.append(f"  Valid paths:          {result.num_valid_paths}")
    lines.append(f"  Violation points:     {result.num_violations}")
    lines.append(f"  Transition coverage:  {result.coverage.transition_coverage:.1%}")
    lines.append(f"  State coverage:       {result.coverage.state_coverage:.1%}")
    lines.append("")

    lines.append("=" * 72)
    if result.is_well_formed:
        lines.append(f"  VERDICT: {p.name} protocol is WELL-FORMED")
    else:
        lines.append(f"  VERDICT: {p.name} protocol has ISSUES")
    lines.append("=" * 72)

    return "\n".join(lines)


def format_financial_summary(
    results: list[FinancialAnalysisResult],
) -> str:
    """Summary table across multiple financial protocol analyses."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  FINANCIAL PROTOCOL VERIFICATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"  {'Protocol':<20s} {'Family':<10s} {'States':>6s} "
        f"{'Trans':>6s} {'Lat':>4s} {'Dist':>5s} {'Paths':>6s} "
        f"{'Viols':>6s}"
    )
    lines.append("  " + "-" * 70)

    for r in results:
        latt = "YES" if r.is_well_formed else "NO"
        dist = "yes" if r.distributivity.is_distributive else "no"
        lines.append(
            f"  {r.protocol.name:<20s} {r.protocol.family:<10s} "
            f"{r.num_states:>6d} {r.num_transitions:>6d} "
            f"{latt:>4s} {dist:>5s} {r.num_valid_paths:>6d} "
            f"{r.num_violations:>6d}"
        )

    lines.append("")
    total = len(results)
    n_lat = sum(1 for r in results if r.is_well_formed)
    lines.append(f"  {n_lat}/{total} protocols form lattices.")
    lines.append("=" * 80)

    return "\n".join(lines)
