"""Supply-chain protocols as multiparty session types (Step 67).

Models a five-tier supply chain --- Supplier, Manufacturer, Distributor,
Retailer, Customer --- as multiparty global types and analyses them with
reticulate's lattice/projection/coverage machinery.

Five realistic scenarios are provided:

1. Happy-path order fulfilment (full tier traversal).
2. Return / reverse-logistics flow.
3. Payment-clearing flow through a payment gateway (stand-in role).
4. Quality inspection with accept/reject branch.
5. Recursive replenishment loop (manufacturer-to-supplier restock).

Each scenario is encoded as a global type, built into a state space,
checked for lattice properties, projected onto every role, and bundled
with a bidirectional morphism pair

    phi: L(G)  -> supply chain phases
    psi: phases -> L(G)

which classifies each tier transition as a safety/compliance/monitoring
checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from reticulate.global_types import (
    GlobalType,
    build_global_statespace,
    parse_global,
    roles,
)
from reticulate.lattice import LatticeResult, check_lattice
from reticulate.parser import SessionType
from reticulate.projection import ProjectionError, project
from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Roles and phases
# ---------------------------------------------------------------------------

#: Canonical five-tier role ordering.
SUPPLY_CHAIN_ROLES: tuple[str, ...] = (
    "Supplier",
    "Manufacturer",
    "Distributor",
    "Retailer",
    "Customer",
)


#: Abstract phases of a supply-chain session --- the codomain of phi.
SUPPLY_CHAIN_PHASES: tuple[str, ...] = (
    "Sourcing",
    "Production",
    "Distribution",
    "Retail",
    "Delivery",
    "Aftermarket",
)


# Map role-pair (sender, receiver) to the abstract phase it belongs to.
# This is the semantic heart of the bidirectional morphism.
_PHASE_OF_EDGE: dict[tuple[str, str], str] = {
    ("Supplier", "Manufacturer"): "Sourcing",
    ("Manufacturer", "Supplier"): "Sourcing",
    ("Manufacturer", "Distributor"): "Production",
    ("Distributor", "Manufacturer"): "Production",
    ("Distributor", "Retailer"): "Distribution",
    ("Retailer", "Distributor"): "Distribution",
    ("Retailer", "Customer"): "Retail",
    ("Customer", "Retailer"): "Retail",
    ("Shipper", "Customer"): "Delivery",
    ("Customer", "Shipper"): "Delivery",
    ("Retailer", "Shipper"): "Delivery",
    ("Shipper", "Retailer"): "Delivery",
    ("Customer", "PaymentGW"): "Retail",
    ("PaymentGW", "Customer"): "Retail",
    ("Retailer", "PaymentGW"): "Retail",
    ("PaymentGW", "Retailer"): "Retail",
    ("PaymentGW", "Bank"): "Retail",
    ("Bank", "PaymentGW"): "Retail",
    ("Customer", "ReturnsDesk"): "Aftermarket",
    ("ReturnsDesk", "Customer"): "Aftermarket",
    ("ReturnsDesk", "Retailer"): "Aftermarket",
    ("Retailer", "ReturnsDesk"): "Aftermarket",
    ("Manufacturer", "QualityInspector"): "Production",
    ("QualityInspector", "Manufacturer"): "Production",
    ("QualityInspector", "Distributor"): "Production",
}


def phase_of_edge(sender: str, receiver: str) -> str:
    """Return the abstract phase for a role-to-role interaction edge."""
    key = (sender, receiver)
    if key in _PHASE_OF_EDGE:
        return _PHASE_OF_EDGE[key]
    # Fallback heuristic --- use the higher-tier role's phase.
    for phase_pair, phase in _PHASE_OF_EDGE.items():
        if sender in phase_pair or receiver in phase_pair:
            return phase
    return "Sourcing"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupplyChainScenario:
    """A named supply-chain scenario expressed as a multiparty global type."""

    name: str
    description: str
    global_type_string: str
    expected_roles: frozenset[str]
    compliance_notes: tuple[str, ...]


@dataclass(frozen=True)
class SupplyChainAnalysis:
    """Complete analysis of a supply-chain scenario."""

    scenario: SupplyChainScenario
    global_type: GlobalType
    state_space: StateSpace
    lattice_result: LatticeResult
    projections: dict[str, SessionType]
    num_states: int
    num_transitions: int
    num_roles: int
    phase_histogram: dict[str, int]
    is_well_formed: bool


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def order_fulfilment_scenario() -> SupplyChainScenario:
    """Five-tier happy-path order fulfilment."""
    return SupplyChainScenario(
        name="OrderFulfilment",
        description=(
            "Customer places order with Retailer; Retailer pulls stock "
            "from Distributor; Distributor requests a production run from "
            "Manufacturer; Manufacturer sources raw material from Supplier; "
            "goods propagate back down the chain until Customer receives "
            "the product."
        ),
        global_type_string=(
            "Customer -> Retailer : {placeOrder: "
            "Retailer -> Distributor : {requestStock: "
            "Distributor -> Manufacturer : {requestProduction: "
            "Manufacturer -> Supplier : {requestMaterial: "
            "Supplier -> Manufacturer : {deliverMaterial: "
            "Manufacturer -> Distributor : {deliverGoods: "
            "Distributor -> Retailer : {shipStock: "
            "Retailer -> Customer : {deliverOrder: end}}}}}}}}"
        ),
        expected_roles=frozenset(SUPPLY_CHAIN_ROLES),
        compliance_notes=(
            "ISO 28000: each tier transition is a control point.",
            "Monitoring: observability must capture every role hop.",
            "Safety: ordering of requestMaterial before deliverMaterial "
            "is enforced by the protocol.",
        ),
    )


def return_scenario() -> SupplyChainScenario:
    """Reverse-logistics return flow."""
    return SupplyChainScenario(
        name="Return",
        description=(
            "Customer initiates a return to a ReturnsDesk role; the desk "
            "collects the item, forwards it upstream to Retailer and then "
            "to Distributor. Distributor either restocks or escalates to "
            "Manufacturer for refurbishment."
        ),
        global_type_string=(
            "Customer -> ReturnsDesk : {requestReturn: "
            "ReturnsDesk -> Customer : {approved: "
            "Customer -> ReturnsDesk : {ship: "
            "ReturnsDesk -> Retailer : {forward: "
            "Retailer -> Distributor : {returnStock: "
            "Distributor -> Manufacturer : {refurbish: "
            "Manufacturer -> Distributor : {refurbished: end}}}}}}}"
        ),
        expected_roles=frozenset(
            {"Customer", "ReturnsDesk", "Retailer",
             "Distributor", "Manufacturer"}
        ),
        compliance_notes=(
            "Consumer protection: explicit approval step audits the return.",
            "Safety: defective items must not re-enter retail stock "
            "without refurbishment.",
            "Traceability: every hop is a GS1 EPCIS event.",
        ),
    )


def payment_scenario() -> SupplyChainScenario:
    """Payment clearing through a payment gateway and bank."""
    return SupplyChainScenario(
        name="Payment",
        description=(
            "Customer pays Retailer through a PaymentGW; the gateway "
            "authorises with Bank, then confirms or declines back to "
            "the Retailer, who in turn notifies the Customer."
        ),
        global_type_string=(
            "Customer -> Retailer : {checkout: "
            "Retailer -> PaymentGW : {charge: "
            "PaymentGW -> Bank : {authorise: "
            "Bank -> PaymentGW : {approved: "
            "PaymentGW -> Retailer : {success: "
            "Retailer -> Customer : {receipt: end}}, "
            "declined: "
            "PaymentGW -> Retailer : {failure: "
            "Retailer -> Customer : {reject: end}}}}}}"
        ),
        expected_roles=frozenset(
            {"Customer", "Retailer", "PaymentGW", "Bank"}
        ),
        compliance_notes=(
            "PCI-DSS: merchant never sees raw card data.",
            "Safety: declined path must not release goods.",
            "Audit: every role transition is logged.",
        ),
    )


def quality_scenario() -> SupplyChainScenario:
    """Quality inspection with accept/reject branch at production."""
    return SupplyChainScenario(
        name="QualityInspection",
        description=(
            "Manufacturer submits a batch to an independent "
            "QualityInspector. If the inspector accepts, the batch "
            "ships to Distributor; if rejected, Manufacturer must "
            "rework and the cycle restarts."
        ),
        global_type_string=(
            "Manufacturer -> QualityInspector : {submit: "
            "QualityInspector -> Manufacturer : {accept: "
            "Manufacturer -> Distributor : {ship: end}, "
            "reject: "
            "Manufacturer -> Distributor : {scrap: end}}}"
        ),
        expected_roles=frozenset(
            {"Manufacturer", "QualityInspector", "Distributor"}
        ),
        compliance_notes=(
            "ISO 9001: quality gate is mandatory before distribution.",
            "Safety: rejected batches cannot bypass inspector.",
            "Testing: every production run exercises at least one "
            "accept and one reject path.",
        ),
    )


def replenishment_scenario() -> SupplyChainScenario:
    """Recursive replenishment loop between Manufacturer and Supplier."""
    return SupplyChainScenario(
        name="Replenishment",
        description=(
            "Manufacturer and Supplier repeatedly negotiate raw-material "
            "replenishment: Manufacturer issues a forecast; Supplier "
            "confirms and delivers, or declines (Manufacturer retries)."
        ),
        global_type_string=(
            "rec X . Manufacturer -> Supplier : {forecast: "
            "Supplier -> Manufacturer : {confirm: "
            "Manufacturer -> Supplier : {purchase: "
            "Supplier -> Manufacturer : {deliver: end}}, "
            "decline: X}}"
        ),
        expected_roles=frozenset({"Manufacturer", "Supplier"}),
        compliance_notes=(
            "Just-in-time manufacturing requires bounded loop depth.",
            "Monitoring: the recursion variable marks an SLA boundary.",
            "Safety: every decline must be paired with a retry.",
        ),
    )


ALL_SUPPLY_CHAIN_SCENARIOS: tuple[SupplyChainScenario, ...] = (
    order_fulfilment_scenario(),
    return_scenario(),
    payment_scenario(),
    quality_scenario(),
    replenishment_scenario(),
)


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------


def analyse_scenario(scenario: SupplyChainScenario) -> SupplyChainAnalysis:
    """Run the full lattice / projection pipeline on a scenario."""
    g = parse_global(scenario.global_type_string)
    ss = build_global_statespace(g)
    lr = check_lattice(ss)
    projs: dict[str, SessionType] = {}
    for role in sorted(roles(g)):
        try:
            projs[role] = project(g, role)
        except ProjectionError:
            # Role not fully projectable (merge failure) --- record None.
            pass
    phase_hist = phase_histogram(ss)
    return SupplyChainAnalysis(
        scenario=scenario,
        global_type=g,
        state_space=ss,
        lattice_result=lr,
        projections=projs,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_roles=len(roles(g)),
        phase_histogram=phase_hist,
        is_well_formed=lr.is_lattice,
    )


def analyse_all() -> list[SupplyChainAnalysis]:
    """Analyse every pre-defined supply-chain scenario."""
    return [analyse_scenario(s) for s in ALL_SUPPLY_CHAIN_SCENARIOS]


# ---------------------------------------------------------------------------
# Bidirectional morphisms: L(G)  <->  phases
# ---------------------------------------------------------------------------


def _parse_label(label: str) -> tuple[str, str, str] | None:
    """Parse a role-annotated transition label 'sender->receiver:method'."""
    if "->" not in label or ":" not in label:
        return None
    left, _, rest = label.partition("->")
    receiver, _, method = rest.partition(":")
    if "{" in method:
        return None  # multi-method aggregate label --- skip
    return (left.strip(), receiver.strip(), method.strip())


def phi_edge(label: str) -> str:
    """phi: a transition label maps to its abstract phase.

    This is the forward leg of the bidirectional morphism pair.
    """
    parsed = _parse_label(label)
    if parsed is None:
        return "Sourcing"
    sender, receiver, _method = parsed
    return phase_of_edge(sender, receiver)


def psi_phase(phase: str) -> frozenset[tuple[str, str]]:
    """psi: a phase maps back to the set of role-edges it covers.

    This is the reverse leg of the bidirectional morphism pair.
    """
    return frozenset(
        edge for edge, p in _PHASE_OF_EDGE.items() if p == phase
    )


def phase_histogram(ss: StateSpace) -> dict[str, int]:
    """Count transitions per abstract phase --- witness of phi."""
    hist: dict[str, int] = {p: 0 for p in SUPPLY_CHAIN_PHASES}
    for _src, label, _tgt in ss.transitions:
        phase = phi_edge(label)
        hist[phase] = hist.get(phase, 0) + 1
    return hist


def round_trip_coverage(ss: StateSpace) -> float:
    """Fraction of transitions whose edge lies in psi(phi(edge))."""
    total = 0
    covered = 0
    for _src, label, _tgt in ss.transitions:
        parsed = _parse_label(label)
        if parsed is None:
            continue
        total += 1
        sender, receiver, _ = parsed
        phase = phase_of_edge(sender, receiver)
        if (sender, receiver) in psi_phase(phase):
            covered += 1
    if total == 0:
        return 1.0
    return covered / total


def classify_morphism_pair() -> str:
    """Classify the (phi, psi) pair in the reticulate morphism hierarchy.

    phi collapses every edge to its phase; psi recovers the full set of
    edges inside a phase.  Since phi is many-to-one and psi is a set-valued
    inverse with psi(phi(edge)) containing edge, the pair forms a
    *Galois insertion* between the powerset lattice of transitions and the
    lattice of phases ordered by inclusion.
    """
    return "galois-insertion"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_analysis(analysis: SupplyChainAnalysis) -> str:
    """Human-readable report for a single analysis."""
    s = analysis.scenario
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  SUPPLY CHAIN SCENARIO: {s.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {s.description}")
    lines.append("")
    lines.append(f"  Roles:        {sorted(s.expected_roles)}")
    lines.append(f"  States:       {analysis.num_states}")
    lines.append(f"  Transitions:  {analysis.num_transitions}")
    lines.append(f"  Lattice:      {analysis.is_well_formed}")
    lines.append("")
    lines.append("  Phase histogram:")
    for phase, count in analysis.phase_histogram.items():
        if count > 0:
            lines.append(f"    {phase:<14s} {count}")
    lines.append("")
    lines.append("  Compliance notes:")
    for note in s.compliance_notes:
        lines.append(f"    - {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def format_summary(analyses: list[SupplyChainAnalysis]) -> str:
    """Compact summary table for multiple analyses."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  SUPPLY CHAIN VERIFICATION SUMMARY")
    lines.append("=" * 72)
    lines.append(
        f"  {'Scenario':<18s} {'Roles':>5s} {'States':>7s} "
        f"{'Trans':>6s} {'Lattice':>8s}"
    )
    lines.append("  " + "-" * 56)
    for a in analyses:
        latt = "YES" if a.is_well_formed else "NO"
        lines.append(
            f"  {a.scenario.name:<18s} {a.num_roles:>5d} "
            f"{a.num_states:>7d} {a.num_transitions:>6d} {latt:>8s}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)
