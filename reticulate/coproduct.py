"""Coproducts in SessLat — categorical coproduct analysis (Step 164).

Investigates whether the category SessLat (session-type lattices with
lattice homomorphisms) has coproducts.  A coproduct of L₁ and L₂ would
be a lattice C with injections ι₁: L₁→C, ι₂: L₂→C satisfying the
universal property: for every pair f: L₁→Y, g: L₂→Y there exists a
unique [f,g]: C→Y with [f,g]∘ι₁ = f and [f,g]∘ι₂ = g.

**Main result**: SessLat does NOT have general coproducts.  The
realizability constraint (reticular form) prevents arbitrary lattice
coproducts from being session types.  This module constructs three
candidate coproduct constructions, tests them, and documents the
negative result with concrete counterexamples.

Candidate constructions:

1. **Coalesced sum** — identify bottoms, add fresh top.
2. **Separated sum** — fresh top and fresh bottom, factors side by side.
3. **Linear sum** — place L₁ above L₂ (ordinal sum).

Public API:

- :func:`coalesced_sum` — shared bottom + fresh top
- :func:`separated_sum` — fresh top + fresh bottom
- :func:`linear_sum` — L₁ stacked above L₂
- :func:`check_coproduct` — try all candidates, return best or counterexample
- :func:`is_coproduct` — boolean wrapper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.lattice import check_lattice, compute_meet, compute_join
from reticulate.morphism import is_order_preserving
from reticulate.reticular import is_reticulate

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionMap:
    """An injection ι: factor → candidate."""

    factor_index: int
    mapping: dict[int, int]
    is_injective: bool
    is_order_preserving: bool


@dataclass(frozen=True)
class CoproductCandidate:
    """A candidate coproduct construction with analysis results."""

    statespace: "StateSpace"
    injections: list[InjectionMap]
    construction: str  # "coalesced", "separated", "linear"
    is_lattice: bool
    is_realizable: bool


@dataclass(frozen=True)
class CoproductResult:
    """Result of coproduct analysis for a pair of lattices."""

    has_coproduct: bool
    best_candidate: CoproductCandidate | None
    all_candidates: list[CoproductCandidate]
    counterexample: str | None


# ---------------------------------------------------------------------------
# Construction functions
# ---------------------------------------------------------------------------

def coalesced_sum(left: "StateSpace", right: "StateSpace") -> "StateSpace":
    """Coalesced sum: identify bottoms of L₁, L₂, add fresh top.

    Result has ``|L₁| + |L₂| - 1 + 1`` states (shared bottom, fresh top).
    Fresh top has edges to both L₁.top and L₂.top.
    """
    from reticulate.statespace import StateSpace as SS

    # Remap left states starting from 0
    next_id = 0
    left_remap: dict[int, int] = {}
    labels: dict[int, str] = {}
    for s in sorted(left.states):
        left_remap[s] = next_id
        labels[next_id] = f"L₁:{left.labels.get(s, str(s))}"
        next_id += 1

    # Remap right states, sharing bottom with left's bottom
    right_remap: dict[int, int] = {}
    for s in sorted(right.states):
        if s == right.bottom:
            right_remap[s] = left_remap[left.bottom]
        else:
            right_remap[s] = next_id
            labels[next_id] = f"L₂:{right.labels.get(s, str(s))}"
            next_id += 1

    # Add fresh top
    fresh_top = next_id
    labels[fresh_top] = "⊤_coprod"
    next_id += 1

    shared_bottom = left_remap[left.bottom]

    # Build states
    states: set[int] = set(left_remap.values()) | set(right_remap.values()) | {fresh_top}

    # Build transitions
    transitions: list[tuple[int, str, int]] = []
    sel_trans: set[tuple[int, str, int]] = set()

    # Left transitions
    for src, lbl, tgt in left.transitions:
        tr = (left_remap[src], lbl, left_remap[tgt])
        transitions.append(tr)
        if left.is_selection(src, lbl, tgt):
            sel_trans.add(tr)

    # Right transitions
    for src, lbl, tgt in right.transitions:
        tr = (right_remap[src], lbl, right_remap[tgt])
        transitions.append(tr)
        if right.is_selection(src, lbl, tgt):
            sel_trans.add(tr)

    # Fresh top edges
    transitions.append((fresh_top, "ι₁", left_remap[left.top]))
    transitions.append((fresh_top, "ι₂", right_remap[right.top]))

    return SS(
        states=states,
        transitions=transitions,
        top=fresh_top,
        bottom=shared_bottom,
        labels=labels,
        selection_transitions=sel_trans,
    )


def separated_sum(left: "StateSpace", right: "StateSpace") -> "StateSpace":
    """Separated sum: L₁ and L₂ side by side, plus fresh top and bottom.

    Result has ``|L₁| + |L₂| + 2`` states.
    Fresh top → {L₁.top, L₂.top}, {L₁.bottom, L₂.bottom} → fresh bottom.
    """
    from reticulate.statespace import StateSpace as SS

    next_id = 0
    left_remap: dict[int, int] = {}
    labels: dict[int, str] = {}
    for s in sorted(left.states):
        left_remap[s] = next_id
        labels[next_id] = f"L₁:{left.labels.get(s, str(s))}"
        next_id += 1

    right_remap: dict[int, int] = {}
    for s in sorted(right.states):
        right_remap[s] = next_id
        labels[next_id] = f"L₂:{right.labels.get(s, str(s))}"
        next_id += 1

    fresh_top = next_id
    labels[fresh_top] = "⊤_coprod"
    next_id += 1

    fresh_bottom = next_id
    labels[fresh_bottom] = "⊥_coprod"
    next_id += 1

    states = set(left_remap.values()) | set(right_remap.values()) | {fresh_top, fresh_bottom}

    transitions: list[tuple[int, str, int]] = []
    sel_trans: set[tuple[int, str, int]] = set()

    # Left transitions
    for src, lbl, tgt in left.transitions:
        tr = (left_remap[src], lbl, left_remap[tgt])
        transitions.append(tr)
        if left.is_selection(src, lbl, tgt):
            sel_trans.add(tr)

    # Right transitions
    for src, lbl, tgt in right.transitions:
        tr = (right_remap[src], lbl, right_remap[tgt])
        transitions.append(tr)
        if right.is_selection(src, lbl, tgt):
            sel_trans.add(tr)

    # Fresh top edges
    transitions.append((fresh_top, "ι₁", left_remap[left.top]))
    transitions.append((fresh_top, "ι₂", right_remap[right.top]))

    # Bottom edges: factor bottoms → fresh bottom
    transitions.append((left_remap[left.bottom], "⊥₁", fresh_bottom))
    transitions.append((right_remap[right.bottom], "⊥₂", fresh_bottom))

    return SS(
        states=states,
        transitions=transitions,
        top=fresh_top,
        bottom=fresh_bottom,
        labels=labels,
        selection_transitions=sel_trans,
    )


def linear_sum(left: "StateSpace", right: "StateSpace") -> "StateSpace":
    """Linear (ordinal) sum: L₁ stacked above L₂.

    Identify L₁.bottom with L₂.top.  NOT symmetric.
    Result has ``|L₁| + |L₂| - 1`` states.
    """
    from reticulate.statespace import StateSpace as SS

    next_id = 0
    left_remap: dict[int, int] = {}
    labels: dict[int, str] = {}
    for s in sorted(left.states):
        left_remap[s] = next_id
        labels[next_id] = f"L₁:{left.labels.get(s, str(s))}"
        next_id += 1

    right_remap: dict[int, int] = {}
    for s in sorted(right.states):
        if s == right.top:
            # Identify L₂.top with L₁.bottom
            right_remap[s] = left_remap[left.bottom]
            # Update label to show the join point
            labels[left_remap[left.bottom]] = "L₁⊥=L₂⊤"
        else:
            right_remap[s] = next_id
            labels[next_id] = f"L₂:{right.labels.get(s, str(s))}"
            next_id += 1

    states = set(left_remap.values()) | set(right_remap.values())

    transitions: list[tuple[int, str, int]] = []
    sel_trans: set[tuple[int, str, int]] = set()

    for src, lbl, tgt in left.transitions:
        tr = (left_remap[src], lbl, left_remap[tgt])
        transitions.append(tr)
        if left.is_selection(src, lbl, tgt):
            sel_trans.add(tr)

    for src, lbl, tgt in right.transitions:
        tr = (right_remap[src], lbl, right_remap[tgt])
        transitions.append(tr)
        if right.is_selection(src, lbl, tgt):
            sel_trans.add(tr)

    return SS(
        states=states,
        transitions=transitions,
        top=left_remap[left.top],
        bottom=right_remap[right.bottom],
        labels=labels,
        selection_transitions=sel_trans,
    )


# ---------------------------------------------------------------------------
# Injection builder
# ---------------------------------------------------------------------------

def _build_injections(
    candidate: "StateSpace",
    left: "StateSpace",
    right: "StateSpace",
    left_remap: dict[int, int],
    right_remap: dict[int, int],
) -> list[InjectionMap]:
    """Build injection maps from factor remap tables."""
    result: list[InjectionMap] = []
    for i, (factor, remap) in enumerate([(left, left_remap), (right, right_remap)]):
        mapping = dict(remap)
        injective = len(set(mapping.values())) == len(mapping)
        order_pres = is_order_preserving(factor, candidate, mapping)
        result.append(InjectionMap(
            factor_index=i,
            mapping=mapping,
            is_injective=injective,
            is_order_preserving=order_pres,
        ))
    return result


def _compute_remaps(
    left: "StateSpace",
    right: "StateSpace",
    candidate: "StateSpace",
    construction: str,
) -> tuple[dict[int, int], dict[int, int]]:
    """Recompute the remap tables for a candidate construction.

    This reconstructs the mapping by re-running the construction logic
    in a deterministic way (sorted state iteration).
    """
    if construction == "coalesced":
        next_id = 0
        left_remap: dict[int, int] = {}
        for s in sorted(left.states):
            left_remap[s] = next_id
            next_id += 1
        right_remap: dict[int, int] = {}
        for s in sorted(right.states):
            if s == right.bottom:
                right_remap[s] = left_remap[left.bottom]
            else:
                right_remap[s] = next_id
                next_id += 1
        return left_remap, right_remap

    elif construction == "separated":
        next_id = 0
        left_remap = {}
        for s in sorted(left.states):
            left_remap[s] = next_id
            next_id += 1
        right_remap = {}
        for s in sorted(right.states):
            right_remap[s] = next_id
            next_id += 1
        return left_remap, right_remap

    elif construction == "linear":
        next_id = 0
        left_remap = {}
        for s in sorted(left.states):
            left_remap[s] = next_id
            next_id += 1
        right_remap = {}
        for s in sorted(right.states):
            if s == right.top:
                right_remap[s] = left_remap[left.bottom]
            else:
                right_remap[s] = next_id
                next_id += 1
        return left_remap, right_remap

    else:
        raise ValueError(f"unknown construction: {construction}")


# ---------------------------------------------------------------------------
# Homomorphism enumeration (small lattices)
# ---------------------------------------------------------------------------

def _find_all_homomorphisms(
    source: "StateSpace",
    target: "StateSpace",
) -> list[dict[int, int]]:
    """Enumerate all lattice homomorphisms from source to target.

    Uses backtracking with pruning.  Only practical for small lattices
    (< ~15 states each).  A lattice homomorphism must:
    - Map top to top
    - Map bottom to bottom
    - Preserve order (reachability)
    - Preserve meets and joins
    """
    src_lattice = check_lattice(source)
    tgt_lattice = check_lattice(target)
    if not src_lattice.is_lattice or not tgt_lattice.is_lattice:
        return []

    # Edge case: if source top == bottom but target top ≠ bottom, no hom
    if source.top == source.bottom and target.top != target.bottom:
        return []
    if source.top != source.bottom and target.top == target.bottom:
        # All source states must map to the single target state
        # Check this is order-preserving (always true) and preserves meets/joins
        mapping = {s: target.top for s in source.states}
        scc_tgt = tgt_lattice.scc_map
        # Trivially preserves everything since all map to same element
        return [mapping]

    src_reach: dict[int, set[int]] = {s: source.reachable_from(s) for s in source.states}
    tgt_reach: dict[int, set[int]] = {s: target.reachable_from(s) for s in target.states}

    results: list[dict[int, int]] = []
    mapping: dict[int, int] = {}

    # Fix top→top, bottom→bottom
    mapping[source.top] = target.top
    if source.bottom != source.top:
        mapping[source.bottom] = target.bottom

    remaining = sorted(s for s in source.states if s not in mapping)

    tgt_states_sorted = sorted(target.states)

    def _order_compatible(s1: int, t1: int) -> bool:
        """Check that assigning s1→t1 is order-compatible with partial mapping."""
        for a, fa in mapping.items():
            if a in src_reach[s1] and fa not in tgt_reach[t1]:
                return False
            if s1 in src_reach[a] and t1 not in tgt_reach[fa]:
                return False
        return True

    def _check_meets_joins(full_map: dict[int, int]) -> bool:
        """Verify meet/join preservation for the complete mapping."""
        scc_src = src_lattice.scc_map
        scc_tgt = tgt_lattice.scc_map
        reps = sorted({scc_src[s] for s in source.states})
        for i, a in enumerate(reps):
            for b in reps[i:]:
                m_src = compute_meet(source, a, b)
                if m_src is not None:
                    m_tgt = compute_meet(target, full_map[a], full_map[b])
                    if m_tgt is None:
                        return False
                    if scc_tgt.get(full_map[m_src]) != scc_tgt.get(m_tgt):
                        return False
                j_src = compute_join(source, a, b)
                if j_src is not None:
                    j_tgt = compute_join(target, full_map[a], full_map[b])
                    if j_tgt is None:
                        return False
                    if scc_tgt.get(full_map[j_src]) != scc_tgt.get(j_tgt):
                        return False
        return True

    def backtrack(idx: int) -> None:
        if idx == len(remaining):
            if _check_meets_joins(mapping):
                results.append(dict(mapping))
            return

        s = remaining[idx]
        for t in tgt_states_sorted:
            if _order_compatible(s, t):
                mapping[s] = t
                backtrack(idx + 1)
                del mapping[s]

    backtrack(0)
    return results


# ---------------------------------------------------------------------------
# Universal property check
# ---------------------------------------------------------------------------

def _find_mediating_morphisms(
    candidate: "StateSpace",
    left: "StateSpace",
    right: "StateSpace",
    inj_left: dict[int, int],
    inj_right: dict[int, int],
    f: dict[int, int],
    g: dict[int, int],
    target: "StateSpace",
) -> list[dict[int, int]]:
    """Find all lattice homomorphisms h: candidate → target with h∘ι₁=f, h∘ι₂=g.

    The mediating morphism must be a lattice homomorphism (preserving
    meets and joins), not merely order-preserving.  This is required
    because morphisms in SessLat are lattice homomorphisms.
    """
    # Find all lattice homomorphisms from candidate to target
    all_homos = _find_all_homomorphisms(candidate, target)

    # Filter by commutativity: h∘ι₁ = f and h∘ι₂ = g
    valid: list[dict[int, int]] = []
    for h in all_homos:
        # Check h∘ι₁ = f
        ok = True
        for s_left, c_state in inj_left.items():
            if h.get(c_state) != f[s_left]:
                ok = False
                break
        if not ok:
            continue
        # Check h∘ι₂ = g
        for s_right, c_state in inj_right.items():
            if h.get(c_state) != g[s_right]:
                ok = False
                break
        if ok:
            valid.append(h)

    return valid


def check_coproduct_universal_property(
    candidate: "StateSpace",
    left: "StateSpace",
    right: "StateSpace",
    inj_left: dict[int, int],
    inj_right: dict[int, int],
    test_targets: list["StateSpace"] | None = None,
) -> tuple[bool, str | None]:
    """Verify the universal property for a coproduct candidate.

    For each test target Y and each pair (f: L₁→Y, g: L₂→Y) of
    lattice homomorphisms, checks:
    1. A mediating morphism [f,g]: C→Y exists (as a lattice hom)
    2. [f,g] is unique
    3. [f,g]∘ι₁ = f and [f,g]∘ι₂ = g  (commutativity)
    """
    from reticulate.product import product_statespace

    if test_targets is None:
        # Default targets: factors, candidate, and the product L₁×L₂
        # The product is a key target because it has enough structure
        # to witness UP failures.
        prod = product_statespace(left, right)
        test_targets = [left, right, candidate, prod]

    for target in test_targets:
        tgt_lattice = check_lattice(target)
        if not tgt_lattice.is_lattice:
            continue

        # Find all lattice homomorphisms from each factor to target
        homos_left = _find_all_homomorphisms(left, target)
        homos_right = _find_all_homomorphisms(right, target)

        for f in homos_left:
            for g in homos_right:
                # Find all valid mediating lattice homomorphisms
                mediators = _find_mediating_morphisms(
                    candidate, left, right,
                    inj_left, inj_right, f, g, target,
                )

                if len(mediators) == 0:
                    return False, (
                        f"no mediating morphism for "
                        f"target with {len(target.states)} states"
                    )

                if len(mediators) > 1:
                    return False, (
                        f"mediating morphism not unique "
                        f"({len(mediators)} found) for "
                        f"target with {len(target.states)} states"
                    )

    return True, None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def check_coproduct(
    left: "StateSpace",
    right: "StateSpace",
) -> CoproductResult:
    """Check whether L₁ + L₂ exists as a coproduct in SessLat.

    Tries all three candidate constructions (coalesced, separated, linear),
    checks lattice property, realizability, injections, and universal property.
    Returns the best candidate if any passes, or a counterexample.
    """
    candidates: list[CoproductCandidate] = []
    constructions = [
        ("coalesced", coalesced_sum),
        ("separated", separated_sum),
        ("linear", linear_sum),
    ]

    for name, build_fn in constructions:
        cand_ss = build_fn(left, right)
        lr = check_lattice(cand_ss)
        realizable = is_reticulate(cand_ss)

        # Compute remap tables for injections
        left_remap, right_remap = _compute_remaps(left, right, cand_ss, name)
        injections = _build_injections(cand_ss, left, right, left_remap, right_remap)

        candidates.append(CoproductCandidate(
            statespace=cand_ss,
            injections=injections,
            construction=name,
            is_lattice=lr.is_lattice,
            is_realizable=realizable,
        ))

    # Check universal property for lattice candidates
    best: CoproductCandidate | None = None
    for cand in candidates:
        if not cand.is_lattice:
            continue

        left_remap, right_remap = _compute_remaps(
            left, right, cand.statespace, cand.construction,
        )

        up_ok, up_cx = check_coproduct_universal_property(
            cand.statespace, left, right,
            left_remap, right_remap,
        )

        if up_ok:
            best = cand
            break

    if best is not None:
        return CoproductResult(
            has_coproduct=True,
            best_candidate=best,
            all_candidates=candidates,
            counterexample=None,
        )

    # Build counterexample description
    reasons: list[str] = []
    for cand in candidates:
        if not cand.is_lattice:
            reasons.append(f"{cand.construction}: not a lattice")
        else:
            left_remap, right_remap = _compute_remaps(
                left, right, cand.statespace, cand.construction,
            )
            _, cx = check_coproduct_universal_property(
                cand.statespace, left, right,
                left_remap, right_remap,
            )
            reasons.append(f"{cand.construction}: {cx}")

    return CoproductResult(
        has_coproduct=False,
        best_candidate=None,
        all_candidates=candidates,
        counterexample="; ".join(reasons),
    )


def is_coproduct(
    candidate: "StateSpace",
    left: "StateSpace",
    right: "StateSpace",
    inj_left: dict[int, int],
    inj_right: dict[int, int],
) -> bool:
    """Check if candidate is a coproduct of left and right."""
    cand_lattice = check_lattice(candidate)
    if not cand_lattice.is_lattice:
        return False

    up_ok, _ = check_coproduct_universal_property(
        candidate, left, right, inj_left, inj_right,
    )
    return up_ok
