"""Coequalizers in SessLat — categorical coequalizer analysis (Step 166).

A **coequalizer** of two morphisms f, g: L → R is a lattice Q with a
morphism q: R → Q such that q ∘ f = q ∘ g, and Q is universal:
any h: R → Y with h ∘ f = h ∘ g factors uniquely through Q.

The standard construction: quotient R by the smallest congruence
identifying f(s) with g(s) for all s ∈ L.

**Main result**: SessLat does NOT have general coequalizers.
The quotient R/θ is always a lattice (standard lattice theory), but
it may not be realizable as a session type (reticular form violation).
This parallels the coproduct failure (Step 164).

**Corollary**: SessLat is NOT finitely cocomplete.  Combined with
Step 165: SessLat has all finite limits but not all finite colimits —
a fundamental asymmetry.

Public API:

- :func:`coequalizer_seed_pairs` — {(f(s), g(s)) : s ∈ L}
- :func:`congruence_closure` — smallest congruence containing seeds
- :func:`quotient_lattice` — build R/θ
- :func:`check_coequalizer` — full coequalizer analysis
- :func:`check_coequalizer_universal_property` — verify universal property
- :func:`is_coequalizer` — boolean wrapper
- :func:`check_finite_cocompleteness` — coproducts + coequalizers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.lattice import check_lattice, compute_meet, compute_join
from reticulate.category import is_lattice_homomorphism
from reticulate.reticular import is_reticulate

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Congruence:
    """A congruence relation on a lattice."""

    pairs: frozenset[tuple[int, int]]
    classes: dict[int, frozenset[int]]  # representative → class members
    num_classes: int
    is_trivial: bool    # all in one class
    is_discrete: bool   # each element alone


@dataclass(frozen=True)
class CoequalizerResult:
    """Result of coequalizer analysis for a pair of morphisms."""

    has_coequalizer: bool
    quotient_space: "StateSpace | None"
    quotient_map: dict[int, int] | None  # q: R → Q
    congruence: Congruence | None
    f_mapping: dict[int, int]
    g_mapping: dict[int, int]
    is_lattice: bool
    is_realizable: bool
    universal_property_holds: bool
    counterexample: str | None


@dataclass(frozen=True)
class FiniteCocompletenessResult:
    """Result of checking finite cocompleteness of SessLat."""

    has_coproducts: bool
    has_coequalizers: bool
    is_finitely_cocomplete: bool
    num_spaces_tested: int
    counterexample: str | None


# ---------------------------------------------------------------------------
# Seed pairs
# ---------------------------------------------------------------------------

def coequalizer_seed_pairs(
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
) -> set[tuple[int, int]]:
    """Compute {(f(s), g(s)) : s ∈ source}.

    These are the pairs in the target that must be identified
    in the coequalizer quotient.
    """
    return {(f[s], g[s]) for s in source.states}


# ---------------------------------------------------------------------------
# Union-Find for congruence closure
# ---------------------------------------------------------------------------

class _UnionFind:
    """Simple union-find (disjoint set) data structure."""

    def __init__(self, elements: set[int]) -> None:
        self.parent: dict[int, int] = {e: e for e in elements}
        self.rank: dict[int, int] = {e: 0 for e in elements}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Merge sets containing x and y.  Returns True if a merge happened."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def classes(self) -> dict[int, frozenset[int]]:
        """Return representative → frozenset of class members."""
        groups: dict[int, set[int]] = {}
        for e in self.parent:
            r = self.find(e)
            groups.setdefault(r, set()).add(e)
        return {r: frozenset(members) for r, members in groups.items()}


# ---------------------------------------------------------------------------
# Congruence closure
# ---------------------------------------------------------------------------

def congruence_closure(
    target: "StateSpace",
    seed_pairs: set[tuple[int, int]],
) -> Congruence:
    """Compute the smallest congruence on target containing seed_pairs.

    Algorithm:
    1. Initialize union-find with seed pairs
    2. Repeat until fixed point:
       a. For all pairs (a₁,a₂), (b₁,b₂) in θ:
          merge(meet(a₁,b₁), meet(a₂,b₂))
          merge(join(a₁,b₁), join(a₂,b₂))
       b. Transitive closure via union-find
    3. Extract equivalence classes
    """
    uf = _UnionFind(target.states)

    # Seed the union-find
    for a, b in seed_pairs:
        uf.union(a, b)

    # Iterate meet/join closure until fixed point
    changed = True
    while changed:
        changed = False
        # Collect current equivalence classes
        cls = uf.classes()
        # Get all pairs of representatives
        reps = sorted(cls.keys())
        # For each pair of classes, check meet/join compatibility
        for i, r1 in enumerate(reps):
            for r2 in reps[i:]:
                members1 = cls[r1]
                members2 = cls[r2]
                # Take representative pairs from each class
                for a1 in members1:
                    for a2 in members1:
                        if a1 == a2:
                            continue
                        for b1 in members2:
                            for b2 in members2:
                                if b1 == b2:
                                    continue
                                # If (a1,a2) are in same class and (b1,b2) in same class
                                # then meet(a1,b1) and meet(a2,b2) must be in same class
                                m1 = compute_meet(target, a1, b1)
                                m2 = compute_meet(target, a2, b2)
                                if m1 is not None and m2 is not None:
                                    if uf.union(m1, m2):
                                        changed = True
                                j1 = compute_join(target, a1, b1)
                                j2 = compute_join(target, a2, b2)
                                if j1 is not None and j2 is not None:
                                    if uf.union(j1, j2):
                                        changed = True

        # Also: for all pairs within a class, close under meet/join
        cls = uf.classes()
        for rep, members in cls.items():
            member_list = sorted(members)
            for i, a in enumerate(member_list):
                for b in member_list[i + 1:]:
                    # a and b are in the same class.
                    # For each other element c, meet(a,c) and meet(b,c)
                    # must be in the same class.
                    for c in target.states:
                        ma = compute_meet(target, a, c)
                        mb = compute_meet(target, b, c)
                        if ma is not None and mb is not None:
                            if uf.union(ma, mb):
                                changed = True
                        ja = compute_join(target, a, c)
                        jb = compute_join(target, b, c)
                        if ja is not None and jb is not None:
                            if uf.union(ja, jb):
                                changed = True

    cls = uf.classes()
    # Build pairs set
    pairs: set[tuple[int, int]] = set()
    for members in cls.values():
        for a in members:
            for b in members:
                pairs.add((a, b))

    num_classes = len(cls)
    is_trivial = num_classes == 1
    is_discrete = all(len(members) == 1 for members in cls.values())

    return Congruence(
        pairs=frozenset(pairs),
        classes=cls,
        num_classes=num_classes,
        is_trivial=is_trivial,
        is_discrete=is_discrete,
    )


# ---------------------------------------------------------------------------
# Quotient lattice construction
# ---------------------------------------------------------------------------

def quotient_lattice(
    target: "StateSpace",
    congruence: Congruence,
) -> "StateSpace":
    """Build the quotient lattice R/θ.

    States = one representative per congruence class (min element).
    Order: [a] ≥ [b] iff a ≥ b in R (well-defined for congruences).
    Covering relation computed on the quotient poset.
    """
    from reticulate.statespace import StateSpace as SS

    # Choose representative per class: minimum element
    rep_map: dict[int, int] = {}  # original state → representative
    for rep, members in congruence.classes.items():
        canonical = min(members)
        for m in members:
            rep_map[m] = canonical

    # Representatives (quotient states), renumbered
    canonical_reps = sorted(set(rep_map.values()))
    remap: dict[int, int] = {}
    for i, c in enumerate(canonical_reps):
        remap[c] = i

    states: set[int] = set(remap.values())
    labels: dict[int, str] = {}
    for old, new in remap.items():
        cls_members = [
            m for m in target.states if rep_map[m] == old
        ]
        if len(cls_members) == 1:
            labels[new] = target.labels.get(old, str(old))
        else:
            labels[new] = "[" + ",".join(
                target.labels.get(m, str(m)) for m in sorted(cls_members)
            ) + "]"

    # Compute reachability in target to determine order on quotient
    tgt_reach: dict[int, set[int]] = {
        s: target.reachable_from(s) for s in target.states
    }

    # Order on quotient: [a] ≥ [b] iff ∃ a' ∈ [a], b' ∈ [b] with a' ≥ b'
    # (For congruences, this is well-defined: equivalent to ∀ a' ∈ [a] ∃ b' ∈ [b])
    q_reach: dict[int, set[int]] = {c: set() for c in canonical_reps}
    for a in canonical_reps:
        a_class = [m for m in target.states if rep_map[m] == a]
        for b in canonical_reps:
            if a == b:
                continue
            b_class = [m for m in target.states if rep_map[m] == b]
            # Check if any a' reaches any b'
            if any(b_m in tgt_reach[a_m] for a_m in a_class for b_m in b_class):
                q_reach[a].add(b)

    # Build covering relation on quotient
    transitions: list[tuple[int, str, int]] = []
    sel_trans: set[tuple[int, str, int]] = set()

    for a in canonical_reps:
        for b in canonical_reps:
            if a == b:
                continue
            if b not in q_reach[a]:
                continue
            # Check covering: no intermediate c
            is_cover = True
            for c in canonical_reps:
                if c == a or c == b:
                    continue
                if c in q_reach[a] and b in q_reach[c]:
                    is_cover = False
                    break
            if is_cover:
                label = _find_label(target, a, b, rep_map)
                tr = (remap[a], label, remap[b])
                transitions.append(tr)
                if _is_selection_path(target, a, b, rep_map):
                    sel_trans.add(tr)

    # Top and bottom
    top_rep = rep_map[target.top]
    bottom_rep = rep_map[target.bottom]

    return SS(
        states=states,
        transitions=transitions,
        top=remap[top_rep],
        bottom=remap[bottom_rep],
        labels=labels,
        selection_transitions=sel_trans,
    )


def _find_label(
    target: "StateSpace",
    a: int,
    b: int,
    rep_map: dict[int, int],
) -> str:
    """Find a transition label between classes [a] and [b]."""
    a_class = [m for m in target.states if rep_map[m] == a]
    b_class = [m for m in target.states if rep_map[m] == b]
    # Direct transition from any member of [a] to any member of [b]?
    for src, lbl, tgt in target.transitions:
        if rep_map.get(src) == a and rep_map.get(tgt) == b:
            return lbl
    # Indirect: first label on a path
    for src, lbl, tgt in target.transitions:
        if rep_map.get(src) == a:
            reachable = target.reachable_from(tgt)
            if any(m in reachable or m == tgt for m in b_class):
                return lbl
    return "τ"


def _is_selection_path(
    target: "StateSpace",
    a: int,
    b: int,
    rep_map: dict[int, int],
) -> bool:
    """Check if path from [a] to [b] starts with a selection transition."""
    a_class = [m for m in target.states if rep_map[m] == a]
    b_class = [m for m in target.states if rep_map[m] == b]
    for src, lbl, tgt in target.transitions:
        if rep_map.get(src) == a:
            if rep_map.get(tgt) == b or any(
                m in target.reachable_from(tgt) for m in b_class
            ):
                return target.is_selection(src, lbl, tgt)
    return False


# ---------------------------------------------------------------------------
# Quotient map
# ---------------------------------------------------------------------------

def _build_quotient_map(
    target: "StateSpace",
    congruence: Congruence,
    q_ss: "StateSpace",
) -> dict[int, int]:
    """Build the quotient map q: R → Q (target state → quotient state)."""
    # rep_map: original → canonical representative
    rep_map: dict[int, int] = {}
    for rep, members in congruence.classes.items():
        canonical = min(members)
        for m in members:
            rep_map[m] = canonical

    canonical_reps = sorted(set(rep_map.values()))
    remap: dict[int, int] = {}
    for i, c in enumerate(canonical_reps):
        remap[c] = i

    return {s: remap[rep_map[s]] for s in target.states}


# ---------------------------------------------------------------------------
# Homomorphism enumeration (copied from equalizer.py)
# ---------------------------------------------------------------------------

def _find_all_homomorphisms(
    source: "StateSpace",
    target: "StateSpace",
) -> list[dict[int, int]]:
    """Enumerate all lattice homomorphisms from source to target.

    Uses backtracking with pruning.  Only practical for small lattices
    (< ~15 states each).
    """
    src_lattice = check_lattice(source)
    tgt_lattice = check_lattice(target)
    if not src_lattice.is_lattice or not tgt_lattice.is_lattice:
        return []

    if source.top == source.bottom and target.top != target.bottom:
        return []
    if source.top != source.bottom and target.top == target.bottom:
        mapping = {s: target.top for s in source.states}
        return [mapping]

    src_reach: dict[int, set[int]] = {
        s: source.reachable_from(s) for s in source.states
    }
    tgt_reach: dict[int, set[int]] = {
        s: target.reachable_from(s) for s in target.states
    }

    results: list[dict[int, int]] = []
    mapping: dict[int, int] = {}

    mapping[source.top] = target.top
    if source.bottom != source.top:
        mapping[source.bottom] = target.bottom

    remaining = sorted(s for s in source.states if s not in mapping)
    tgt_states_sorted = sorted(target.states)

    scc_src = src_lattice.scc_map
    scc_tgt = tgt_lattice.scc_map

    def _order_compatible(s1: int, t1: int) -> bool:
        for a, fa in mapping.items():
            if a in src_reach[s1] and fa not in tgt_reach[t1]:
                return False
            if s1 in src_reach[a] and t1 not in tgt_reach[fa]:
                return False
        return True

    def _check_meets_joins(full_map: dict[int, int]) -> bool:
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
# Universal property
# ---------------------------------------------------------------------------

def check_coequalizer_universal_property(
    q_space: "StateSpace",
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
    q_map: dict[int, int],
    test_targets: list["StateSpace"] | None = None,
) -> tuple[bool, str | None]:
    """Verify the universal property for a coequalizer candidate.

    For each test target Y and each lattice homomorphism h: R → Y
    with h ∘ f = h ∘ g, checks:
    1. A mediating morphism u: Q → Y exists with u ∘ q = h
    2. u is unique
    """
    if test_targets is None:
        test_targets = [target, q_space]

    for test_tgt in test_targets:
        tgt_lattice = check_lattice(test_tgt)
        if not tgt_lattice.is_lattice:
            continue

        # Find all lattice homs h: target → test_tgt
        all_h = _find_all_homomorphisms(target, test_tgt)

        for h in all_h:
            # Check if h ∘ f = h ∘ g (h coequalizes f and g)
            coequalizes = all(h[f[s]] == h[g[s]] for s in source.states)
            if not coequalizes:
                continue

            # h coequalizes — find mediating morphism u: Q → test_tgt
            # with u ∘ q = h
            all_u = _find_all_homomorphisms(q_space, test_tgt)

            # Filter: u ∘ q = h
            valid: list[dict[int, int]] = []
            for u in all_u:
                if all(u[q_map[s]] == h[s] for s in target.states):
                    valid.append(u)

            if len(valid) == 0:
                return False, (
                    f"no mediating morphism for test target "
                    f"with {len(test_tgt.states)} states"
                )

            if len(valid) > 1:
                return False, (
                    f"mediating morphism not unique "
                    f"({len(valid)} found) for test target "
                    f"with {len(test_tgt.states)} states"
                )

    return True, None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def check_coequalizer(
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
) -> CoequalizerResult:
    """Check whether the coequalizer of f, g: source → target exists in SessLat.

    Constructs the quotient R/θ where θ is the smallest congruence
    identifying f(s) with g(s) for all s.  Checks lattice property,
    realizability, and universal property.
    """
    # Validate inputs
    src_lr = check_lattice(source)
    tgt_lr = check_lattice(target)
    if not src_lr.is_lattice:
        return CoequalizerResult(
            has_coequalizer=False,
            quotient_space=None,
            quotient_map=None,
            congruence=None,
            f_mapping=f,
            g_mapping=g,
            is_lattice=False,
            is_realizable=False,
            universal_property_holds=False,
            counterexample="source is not a lattice",
        )
    if not tgt_lr.is_lattice:
        return CoequalizerResult(
            has_coequalizer=False,
            quotient_space=None,
            quotient_map=None,
            congruence=None,
            f_mapping=f,
            g_mapping=g,
            is_lattice=False,
            is_realizable=False,
            universal_property_holds=False,
            counterexample="target is not a lattice",
        )

    # Compute seed pairs and congruence closure
    seeds = coequalizer_seed_pairs(source, target, f, g)
    cong = congruence_closure(target, seeds)

    # Build quotient lattice
    q_ss = quotient_lattice(target, cong)
    q_map = _build_quotient_map(target, cong, q_ss)

    # Check the quotient is a lattice
    q_lr = check_lattice(q_ss)
    is_lat = q_lr.is_lattice

    # Check realizability
    realizable = is_reticulate(q_ss) if is_lat else False

    # Verify q is a lattice homomorphism
    if is_lat and not is_lattice_homomorphism(target, q_ss, q_map):
        return CoequalizerResult(
            has_coequalizer=False,
            quotient_space=q_ss,
            quotient_map=q_map,
            congruence=cong,
            f_mapping=f,
            g_mapping=g,
            is_lattice=is_lat,
            is_realizable=realizable,
            universal_property_holds=False,
            counterexample="quotient map is not a lattice homomorphism",
        )

    # Verify q ∘ f = q ∘ g
    for s in source.states:
        if q_map[f[s]] != q_map[g[s]]:
            return CoequalizerResult(
                has_coequalizer=False,
                quotient_space=q_ss,
                quotient_map=q_map,
                congruence=cong,
                f_mapping=f,
                g_mapping=g,
                is_lattice=is_lat,
                is_realizable=realizable,
                universal_property_holds=False,
                counterexample="q ∘ f ≠ q ∘ g",
            )

    # Check universal property
    up_ok, up_cx = check_coequalizer_universal_property(
        q_ss, source, target, f, g, q_map,
    )

    has_coeq = is_lat and up_ok
    cx = up_cx if not up_ok else (
        "quotient is not a lattice" if not is_lat else None
    )

    return CoequalizerResult(
        has_coequalizer=has_coeq,
        quotient_space=q_ss,
        quotient_map=q_map,
        congruence=cong,
        f_mapping=f,
        g_mapping=g,
        is_lattice=is_lat,
        is_realizable=realizable,
        universal_property_holds=up_ok,
        counterexample=cx,
    )


def is_coequalizer(
    quotient: "StateSpace",
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
    q_map: dict[int, int],
) -> bool:
    """Check if quotient with q_map is the coequalizer of f and g."""
    q_lr = check_lattice(quotient)
    if not q_lr.is_lattice:
        return False

    # Check q ∘ f = q ∘ g
    for s in source.states:
        if q_map[f[s]] != q_map[g[s]]:
            return False

    up_ok, _ = check_coequalizer_universal_property(
        quotient, source, target, f, g, q_map,
    )
    return up_ok


# ---------------------------------------------------------------------------
# Finite cocompleteness
# ---------------------------------------------------------------------------

def check_finite_cocompleteness(
    spaces: list["StateSpace"],
) -> FiniteCocompletenessResult:
    """Verify whether SessLat is finitely cocomplete: coproducts + coequalizers.

    Tests that:
    1. Coproducts exist (from Step 164) — known to FAIL
    2. Coequalizers exist for all homomorphism pairs on the given spaces

    **Expected result**: FAILS — coproducts already fail (Step 164).
    """
    from reticulate.coproduct import check_coproduct

    lattices = [ss for ss in spaces if check_lattice(ss).is_lattice]

    # 1. Check coproducts
    has_coproducts = True
    coprod_cx: str | None = None
    for i, l1 in enumerate(lattices):
        for l2 in lattices[i:]:
            result = check_coproduct(l1, l2)
            if not result.has_coproduct:
                has_coproducts = False
                coprod_cx = (
                    f"coproduct fails for "
                    f"{len(l1.states)}+{len(l2.states)}: "
                    f"{result.counterexample}"
                )
                break
        if not has_coproducts:
            break

    # 2. Check coequalizers
    has_coequalizers = True
    coeq_cx: str | None = None
    for source in lattices:
        for target_ss in lattices:
            homos = _find_all_homomorphisms(source, target_ss)
            for fi in range(len(homos)):
                for gi in range(fi, len(homos)):
                    result = check_coequalizer(
                        source, target_ss, homos[fi], homos[gi],
                    )
                    if not result.has_coequalizer:
                        has_coequalizers = False
                        coeq_cx = (
                            f"coequalizer fails for "
                            f"{len(source.states)}->"
                            f"{len(target_ss.states)}: "
                            f"{result.counterexample}"
                        )
                        break
                if not has_coequalizers:
                    break
            if not has_coequalizers:
                break
        if not has_coequalizers:
            break

    is_fcc = has_coproducts and has_coequalizers
    cx = coprod_cx or coeq_cx

    return FiniteCocompletenessResult(
        has_coproducts=has_coproducts,
        has_coequalizers=has_coequalizers,
        is_finitely_cocomplete=is_fcc,
        num_spaces_tested=len(lattices),
        counterexample=cx,
    )
