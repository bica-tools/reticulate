"""Dogfooding: extract session types FROM the reticulate checker modules (Step 80i).

Each checker module in reticulate (parser, statespace, lattice, subtyping, ...)
exposes a public entry point whose internal control flow follows a protocol:
    parser:     tokenize -> parse_ast -> validate
    statespace: collect_states -> add_transitions -> close
    lattice:    quotient_sccs -> reach_dag -> check_top_bottom -> check_pairs
    subtyping:  normalise -> coinductive_check

Each such protocol is a session type. This module encodes those protocols as
``SessionType`` ASTs, builds their state spaces, checks lattice properties,
and computes bidirectional morphisms between the module call graph
(a poset of phases under "happens-before") and the protocol lattice L(S).

Result: the reticulate checkers dogfood the theory — they themselves satisfy
session-type contracts, and the contracts form lattices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import LatticeResult, check_lattice


# ---------------------------------------------------------------------------
# Checker protocol registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckerProtocol:
    """A session-type description of a reticulate checker module.

    Attributes:
        module_name: Python module name (e.g. ``"parser"``).
        entry_point: Public function name that drives the protocol.
        phases: Tuple of phase names in happens-before order.
        session_type: Session-type string in reticulate grammar.
        edges: Call-graph edges ``(caller_phase, callee_phase)`` as strings.
    """

    module_name: str
    entry_point: str
    phases: Tuple[str, ...]
    session_type: str
    edges: Tuple[Tuple[str, str], ...]


#: The canonical set of dogfooded checker protocols.
CHECKER_PROTOCOLS: Tuple[CheckerProtocol, ...] = (
    CheckerProtocol(
        module_name="parser",
        entry_point="parse",
        phases=("tokenize", "parse_ast", "validate"),
        # tokenize -> parse_ast -> {ok -> validate -> end , error -> end}
        session_type="&{tokenize: &{parse_ast: +{ok: &{validate: end}, error: end}}}",
        edges=(("tokenize", "parse_ast"), ("parse_ast", "validate")),
    ),
    CheckerProtocol(
        module_name="statespace",
        entry_point="build_statespace",
        phases=("collect", "transitions", "close"),
        session_type="&{collect: &{transitions: &{close: end}}}",
        edges=(("collect", "transitions"), ("transitions", "close")),
    ),
    CheckerProtocol(
        module_name="lattice",
        entry_point="check_lattice",
        phases=("quotient", "reach", "bounds", "pairs"),
        # After bounds we may find a counterexample or proceed to pairs.
        session_type=(
            "&{quotient: &{reach: &{bounds: "
            "+{hasBounds: &{pairs: +{ok: end, counterexample: end}},"
            " noBounds: end}}}}"
        ),
        edges=(
            ("quotient", "reach"),
            ("reach", "bounds"),
            ("bounds", "pairs"),
        ),
    ),
    CheckerProtocol(
        module_name="subtyping",
        entry_point="is_subtype",
        phases=("normalise", "coinductive_check"),
        session_type="&{normalise: &{coinductive_check: +{yes: end, no: end}}}",
        edges=(("normalise", "coinductive_check"),),
    ),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DogfoodResult:
    """Result of dogfooding a single checker.

    Attributes:
        protocol: The checker protocol analysed.
        statespace: The state space built from ``protocol.session_type``.
        lattice_result: Lattice check for the state space.
        phi_valid: True if ``phi`` (phase -> state) is order-preserving.
        psi_valid: True if ``psi`` (state -> phase) is order-preserving.
        bidirectional: True if both φ and ψ are order-preserving on their domains.
    """

    protocol: CheckerProtocol
    statespace: StateSpace
    lattice_result: LatticeResult
    phi_valid: bool
    psi_valid: bool
    bidirectional: bool


# ---------------------------------------------------------------------------
# Call graph poset
# ---------------------------------------------------------------------------


def call_graph_reachability(
    phases: Tuple[str, ...],
    edges: Tuple[Tuple[str, str], ...],
) -> Dict[str, FrozenSet[str]]:
    """Return reachability closure over the call-graph edges.

    ``result[p]`` is the set of phases reachable from ``p`` (including ``p``).
    """

    reach: Dict[str, set] = {p: {p} for p in phases}
    # Iterate to a fixed point — cheap: N small.
    changed = True
    while changed:
        changed = False
        for src, tgt in edges:
            before = len(reach[src])
            reach[src] |= reach[tgt]
            if len(reach[src]) > before:
                changed = True
    return {p: frozenset(s) for p, s in reach.items()}


def call_graph_leq(
    phases: Tuple[str, ...],
    edges: Tuple[Tuple[str, str], ...],
) -> Callable[[str, str], bool]:
    """Return the ≤ relation of the call-graph poset (p ≤ q iff q reachable from p)."""

    reach = call_graph_reachability(phases, edges)

    def leq(p: str, q: str) -> bool:
        return q in reach[p]

    return leq


# ---------------------------------------------------------------------------
# Bidirectional morphisms
# ---------------------------------------------------------------------------


def build_phi(
    protocol: CheckerProtocol,
    ss: StateSpace,
) -> Dict[str, int]:
    """Map each phase name to the state reached by executing its prefix.

    We walk transitions from the top state, matching phase names to transition
    labels. For selection/branch forks after a phase, we stop at the state
    *before* the fork resolves — the phase is identified with the decision point.
    """

    mapping: Dict[str, int] = {}
    current = ss.top
    for phase in protocol.phases:
        mapping[phase] = current
        # Find an outgoing transition whose label matches ``phase``.
        next_state: Optional[int] = None
        for src, label, tgt in ss.transitions:
            if src == current and label == phase:
                next_state = tgt
                break
        if next_state is None:
            # Phase not found as direct label; stop walk.
            break
        current = next_state
    return mapping


def build_psi(
    protocol: CheckerProtocol,
    ss: StateSpace,
    phi: Dict[str, int],
) -> Dict[int, str]:
    """Inverse map: each state in the image of φ is labelled by its phase.

    For states not in the image of φ we assign the *nearest* preceding phase
    (the phase whose φ-image is ≤ this state under reachability).
    """

    # Order phases by depth (φ image order).
    phase_order = list(phi.items())
    psi: Dict[int, str] = {}
    # Reachability within state space.
    reachable: Dict[int, set] = {s: {s} for s in ss.states}
    changed = True
    while changed:
        changed = False
        for src, _lbl, tgt in ss.transitions:
            before = len(reachable[src])
            reachable[src] |= reachable[tgt]
            if len(reachable[src]) > before:
                changed = True

    for state in ss.states:
        # Find deepest phase whose φ-image reaches ``state``.
        best: Optional[str] = None
        for phase, ps in phase_order:
            if state in reachable[ps]:
                best = phase
        if best is not None:
            psi[state] = best
    return psi


def check_phi_order_preserving(
    protocol: CheckerProtocol,
    ss: StateSpace,
    phi: Dict[str, int],
) -> bool:
    """φ is order-preserving if p ≤_cg q implies φ(p) ≤_L φ(q)."""

    leq = call_graph_leq(protocol.phases, protocol.edges)
    # Reachability in the state space.
    reach: Dict[int, set] = {s: {s} for s in ss.states}
    changed = True
    while changed:
        changed = False
        for src, _lbl, tgt in ss.transitions:
            before = len(reach[src])
            reach[src] |= reach[tgt]
            if len(reach[src]) > before:
                changed = True

    for p in protocol.phases:
        for q in protocol.phases:
            if leq(p, q) and p in phi and q in phi:
                if phi[q] not in reach[phi[p]]:
                    return False
    return True


def check_psi_order_preserving(
    protocol: CheckerProtocol,
    ss: StateSpace,
    psi: Dict[int, str],
) -> bool:
    """ψ is order-preserving if s ≤_L t implies ψ(s) ≤_cg ψ(t)."""

    leq = call_graph_leq(protocol.phases, protocol.edges)
    reach: Dict[int, set] = {s: {s} for s in ss.states}
    changed = True
    while changed:
        changed = False
        for src, _lbl, tgt in ss.transitions:
            before = len(reach[src])
            reach[src] |= reach[tgt]
            if len(reach[src]) > before:
                changed = True

    for s in ss.states:
        for t in ss.states:
            if t in reach[s] and s in psi and t in psi:
                if not leq(psi[s], psi[t]):
                    return False
    return True


# ---------------------------------------------------------------------------
# Top-level dogfood driver
# ---------------------------------------------------------------------------


def dogfood_checker(protocol: CheckerProtocol) -> DogfoodResult:
    """Build state space, check lattice, and verify bidirectional morphisms."""

    ast = parse(protocol.session_type)
    ss = build_statespace(ast)
    lattice_result = check_lattice(ss)
    phi = build_phi(protocol, ss)
    psi = build_psi(protocol, ss, phi)
    phi_valid = check_phi_order_preserving(protocol, ss, phi)
    psi_valid = check_psi_order_preserving(protocol, ss, psi)
    return DogfoodResult(
        protocol=protocol,
        statespace=ss,
        lattice_result=lattice_result,
        phi_valid=phi_valid,
        psi_valid=psi_valid,
        bidirectional=phi_valid and psi_valid,
    )


def dogfood_all() -> Tuple[DogfoodResult, ...]:
    """Dogfood every registered checker protocol."""

    return tuple(dogfood_checker(p) for p in CHECKER_PROTOCOLS)


def dogfood_summary() -> Dict[str, object]:
    """Aggregate statistics: how many checkers pass, how many yield lattices."""

    results = dogfood_all()
    return {
        "total": len(results),
        "lattices": sum(1 for r in results if r.lattice_result.is_lattice),
        "bidirectional": sum(1 for r in results if r.bidirectional),
        "phi_ok": sum(1 for r in results if r.phi_valid),
        "psi_ok": sum(1 for r in results if r.psi_valid),
    }
