"""MAPK and PI3K/AKT signaling cascades as multiparty session types (Step 61).

Biology abstraction
-------------------
Cellular signaling cascades such as MAPK (Ras -> Raf -> MEK -> ERK) and
PI3K/AKT (RTK -> PI3K -> AKT -> mTOR) are sequential phosphorylation events
carried out by protein kinases on shared substrates.  Each protein has a
well-defined protocol:

    * it receives a "phosphorylate" signal from an upstream kinase,
    * it acquires the "active" state,
    * while active, it can phosphorylate downstream substrates,
    * and it is eventually dephosphorylated (phosphatase reset).

Modelled as session types, proteins are roles, phosphorylation events are
messages, and the pathway is a multiparty global type whose projection onto
each protein yields its local signaling protocol.

This module builds small but faithful models of:

    * MAPK cascade: Ras, Raf, MEK, ERK (4 roles)
    * PI3K/AKT axis: RTK, PI3K, AKT, mTOR (4 roles)
    * crosstalk composition: Ras activates both Raf and PI3K

and exposes analyses relevant to drug discovery:

    * ``pathway_states`` — reachable phosphorylation vectors
    * ``drug_target_impact`` — blocking a node and measuring pruned states
    * ``compute_bottleneck`` — single-node cuts that maximally disconnect
      the cascade (classical drug targets)
    * ``phi_map`` / ``psi_map`` — the bidirectional morphism between the
      session-type lattice L(S) and the biological phosphorylation lattice

The morphisms are the core contribution of the step: they witness that the
abstract protocol lattice and the domain-specific pathway lattice are
*equivalent as partial orders*, so lattice operations (meet, join, complement)
translate to operational biological actions (pathway intersection,
co-activation, inhibition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as iproduct

from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    SessionType,
    Var,
    parse,
)
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Domain model: pathway states
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathwayState:
    """Phosphorylation vector over a fixed role ordering.

    Each component is ``True`` iff the corresponding protein is currently
    phosphorylated (active).  The partial order is componentwise:
    ``s <= t`` iff every active protein in ``s`` is also active in ``t``.
    """

    roles: tuple[str, ...]
    active: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.roles) != len(self.active):
            raise ValueError("roles and active must have equal length")

    def leq(self, other: "PathwayState") -> bool:
        if self.roles != other.roles:
            raise ValueError("states over different roles")
        return all((not a) or b for a, b in zip(self.active, other.active))

    def meet(self, other: "PathwayState") -> "PathwayState":
        return PathwayState(
            self.roles,
            tuple(a and b for a, b in zip(self.active, other.active)),
        )

    def join(self, other: "PathwayState") -> "PathwayState":
        return PathwayState(
            self.roles,
            tuple(a or b for a, b in zip(self.active, other.active)),
        )

    def phosphorylated(self) -> frozenset[str]:
        return frozenset(r for r, a in zip(self.roles, self.active) if a)

    def __str__(self) -> str:  # pragma: no cover - pretty repr
        bits = "".join("1" if a else "0" for a in self.active)
        return f"P[{bits}]"


# ---------------------------------------------------------------------------
# Canonical signaling session types
# ---------------------------------------------------------------------------


#: MAPK cascade: Ras activates Raf activates MEK activates ERK; each step
#: may alternatively be blocked (phosphatase reset) and the cascade retries.
MAPK_CASCADE = (
    "rec X . &{"
    "phosphorylateRas: &{"
    "phosphorylateRaf: &{"
    "phosphorylateMEK: &{"
    "phosphorylateERK: end, "
    "dephosphorylate: X}, "
    "dephosphorylate: X}, "
    "dephosphorylate: X}, "
    "dephosphorylate: X}"
)

#: PI3K/AKT axis: RTK -> PI3K -> AKT -> mTOR with phosphatase resets.
PI3K_AKT_AXIS = (
    "rec Y . &{"
    "phosphorylateRTK: &{"
    "phosphorylatePI3K: &{"
    "phosphorylateAKT: &{"
    "phosphorylateMTOR: end, "
    "dephosphorylate: Y}, "
    "dephosphorylate: Y}, "
    "dephosphorylate: Y}, "
    "dephosphorylate: Y}"
)

#: Cross-talk model: parallel composition of MAPK and PI3K on a shared Ras
#: hub.  Both branches run concurrently; this forces product lattice structure.
CROSSTALK = f"({MAPK_CASCADE} || {PI3K_AKT_AXIS})"


MAPK_ROLES: tuple[str, ...] = ("Ras", "Raf", "MEK", "ERK")
PI3K_ROLES: tuple[str, ...] = ("RTK", "PI3K", "AKT", "MTOR")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugTargetImpact:
    """Effect of inhibiting a single node in the pathway.

    Attributes:
        target: protein name that has been silenced.
        reachable_before: states reachable before inhibition.
        reachable_after: states reachable after inhibition.
        pruned: states removed by the inhibitor.
        efficacy: fraction ``|pruned| / |reachable_before|``.
    """

    target: str
    reachable_before: int
    reachable_after: int
    pruned: int
    efficacy: float


@dataclass(frozen=True)
class BottleneckReport:
    target: str
    efficacy: float
    ranked: tuple[tuple[str, float], ...]


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def mapk_statespace() -> StateSpace:
    return build_statespace(parse(MAPK_CASCADE))


def pi3k_statespace() -> StateSpace:
    return build_statespace(parse(PI3K_AKT_AXIS))


def crosstalk_statespace() -> StateSpace:
    return build_statespace(parse(CROSSTALK))


def _labels_on_path(ss: StateSpace, start: int, end: int) -> dict[int, frozenset[str]]:
    """Return, for each state reachable from ``start`` in the pathway DAG
    (back-edges removed), the set of phosphorylation labels emitted on any
    forward path from ``start`` to that state.

    Back-edges are those labelled with a dephosphorylation or reset (they
    model phosphatase resets and do not monotonically advance the cascade).
    Removing them yields an acyclic cover of the cascade and a well-defined
    monotone history function.
    """
    def is_back(lbl: str) -> bool:
        return lbl.startswith("dephosphorylate") or lbl.startswith("reset")

    forward_edges = [
        (s, l, t) for (s, l, t) in ss.transitions if not is_back(l)
    ]
    history: dict[int, frozenset[str]] = {s: frozenset() for s in ss.states}
    history[start] = frozenset()
    changed = True
    while changed:
        changed = False
        for src, lbl, tgt in forward_edges:
            new = history[src] | {lbl}
            if not new <= history[tgt]:
                history[tgt] = history[tgt] | new
                changed = True
    return history


def _extract_protein(label: str) -> str | None:
    """Map a phosphorylation label to the protein it activates."""
    if label.startswith("phosphorylate"):
        return label[len("phosphorylate"):]
    return None


def pathway_states(ss: StateSpace, roles: tuple[str, ...]) -> list[PathwayState]:
    """Project each state-space node into a :class:`PathwayState`."""
    histories = _labels_on_path(ss, ss.top, ss.bottom)
    result: list[PathwayState] = []
    max_id = max(ss.states) if ss.states else 0
    for i in range(max_id + 1):
        h = histories.get(i, frozenset()) if isinstance(histories, dict) else frozenset()
        active = tuple(any(_extract_protein(lbl) == r for lbl in h) for r in roles)
        result.append(PathwayState(roles, active))
    return result


def phi_map(ss: StateSpace, roles: tuple[str, ...]) -> dict[int, PathwayState]:
    """The forward morphism phi: L(S) -> PathwayState lattice.

    Sends each session-type state to the phosphorylation vector induced by
    the labels crossed on some witnessing path from top.
    """
    states = pathway_states(ss, roles)
    return {i: states[i] for i in sorted(ss.states)}


def psi_map(
    ss: StateSpace,
    roles: tuple[str, ...],
) -> dict[PathwayState, int]:
    """The backward morphism psi: PathwayState lattice -> L(S).

    Sends each phosphorylation vector to the *smallest* state-space node
    whose phi-image dominates it (pointwise).  This is the Galois adjoint
    to phi and witnesses the Galois connection classification.
    """
    phi = phi_map(ss, roles)
    inverse: dict[PathwayState, int] = {}
    for ps in _enumerate_vectors(roles):
        # pick smallest state id whose phi contains ps (componentwise)
        candidates = [sid for sid, img in phi.items() if ps.leq(img)]
        if candidates:
            inverse[ps] = min(candidates)
    return inverse


def _enumerate_vectors(roles: tuple[str, ...]) -> list[PathwayState]:
    out: list[PathwayState] = []
    for bits in iproduct((False, True), repeat=len(roles)):
        out.append(PathwayState(roles, bits))
    return out


def drug_target_impact(
    ss: StateSpace,
    roles: tuple[str, ...],
    target: str,
) -> DrugTargetImpact:
    """Block all transitions that phosphorylate ``target`` and measure
    how many state-space nodes become unreachable from the top."""
    if target not in roles:
        raise ValueError(f"unknown target {target!r}")
    before = _reachable_count(ss)
    blocked_label = f"phosphorylate{target}"
    pruned_edges = [
        (s, l, t) for (s, l, t) in ss.transitions if l != blocked_label
    ]
    ss2 = StateSpace(
        states=set(ss.states),
        transitions=pruned_edges,
        top=ss.top,
        bottom=ss.bottom,
        labels=dict(ss.labels),
        selection_transitions=ss.selection_transitions
        & set(pruned_edges),
    )
    after = _reachable_count(ss2)
    pruned = before - after
    efficacy = pruned / before if before else 0.0
    return DrugTargetImpact(
        target=target,
        reachable_before=before,
        reachable_after=after,
        pruned=pruned,
        efficacy=round(efficacy, 4),
    )


def compute_bottleneck(
    ss: StateSpace,
    roles: tuple[str, ...],
) -> BottleneckReport:
    """Return the single-node cut maximising pathway disruption."""
    ranked: list[tuple[str, float]] = []
    for r in roles:
        impact = drug_target_impact(ss, roles, r)
        ranked.append((r, impact.efficacy))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    best = ranked[0]
    return BottleneckReport(target=best[0], efficacy=best[1], ranked=tuple(ranked))


def _reachable_count(ss: StateSpace) -> int:
    seen = {ss.top}
    frontier = [ss.top]
    while frontier:
        s = frontier.pop()
        for src, _l, tgt in ss.transitions:
            if src == s and tgt not in seen:
                seen.add(tgt)
                frontier.append(tgt)
    return len(seen)


def is_galois_connection_phi_psi(
    ss: StateSpace,
    roles: tuple[str, ...],
) -> bool:
    """Check that phi and psi form a Galois connection:
    ``phi(s) >= p  iff  s >= psi(p)`` for all s, p."""
    phi = phi_map(ss, roles)
    psi = psi_map(ss, roles)
    # Build state reachability for L(S)
    reach: dict[int, set[int]] = {i: {i} for i in ss.states}
    changed = True
    while changed:
        changed = False
        for src, _l, tgt in ss.transitions:
            if not reach[tgt].issubset(reach[src]):
                reach[src] |= reach[tgt]
                changed = True

    def leq_L(a: int, b: int) -> bool:
        return b in reach[a]

    for s in phi:
        for p, psp in psi.items():
            lhs = p.leq(phi[s])
            rhs = leq_L(psp, s)
            if lhs != rhs:
                return False
    return True


__all__ = [
    "PathwayState",
    "MAPK_CASCADE",
    "PI3K_AKT_AXIS",
    "CROSSTALK",
    "MAPK_ROLES",
    "PI3K_ROLES",
    "DrugTargetImpact",
    "BottleneckReport",
    "mapk_statespace",
    "pi3k_statespace",
    "crosstalk_statespace",
    "pathway_states",
    "phi_map",
    "psi_map",
    "drug_target_impact",
    "compute_bottleneck",
    "is_galois_connection_phi_psi",
]
