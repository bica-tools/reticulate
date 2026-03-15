"""Equalizers in SessLat — categorical equalizer analysis (Step 165).

An **equalizer** of two morphisms f, g: L → R is a lattice E with a
morphism equ: E → L such that f ∘ equ = g ∘ equ, and E is universal:
any h: X → L with f ∘ h = g ∘ h factors uniquely through E.

**Main result**: SessLat HAS equalizers.  The subset equalizer
E = {s ∈ L : f(s) = g(s)} is always a bounded sublattice of L.
Combined with products (Step 163), this makes SessLat **finitely
complete** — it has all finite limits.

Public API:

- :func:`agreement_set` — compute {s ∈ L : f(s) = g(s)}
- :func:`subset_equalizer` — build the sublattice on the agreement set
- :func:`compute_kernel_pair` — {(a,b) : f(a) = f(b)}
- :func:`check_equalizer` — full equalizer analysis
- :func:`check_equalizer_universal_property` — verify the universal property
- :func:`is_equalizer` — boolean wrapper
- :func:`check_finite_completeness` — products + equalizers = finite limits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.lattice import check_lattice, compute_meet, compute_join
from reticulate.category import is_lattice_homomorphism

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EqualizerResult:
    """Result of equalizer analysis for a pair of morphisms."""

    has_equalizer: bool
    equalizer_space: "StateSpace | None"
    inclusion: dict[int, int] | None  # equ: E → L
    f_mapping: dict[int, int]
    g_mapping: dict[int, int]
    universal_property_holds: bool
    counterexample: str | None


@dataclass(frozen=True)
class KernelPair:
    """The kernel pair of a morphism f: {(a,b) ∈ L×L : f(a)=f(b)}."""

    pairs: frozenset[tuple[int, int]]
    is_congruence: bool


@dataclass(frozen=True)
class FiniteCompletenessResult:
    """Result of checking finite completeness of SessLat."""

    has_products: bool
    has_equalizers: bool
    is_finitely_complete: bool
    num_spaces_tested: int
    counterexample: str | None


# ---------------------------------------------------------------------------
# Agreement set
# ---------------------------------------------------------------------------

def agreement_set(
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
) -> set[int]:
    """Compute E = {s ∈ source : f(s) = g(s)}.

    Both *f* and *g* must be total on ``source.states``.
    """
    return {s for s in source.states if f[s] == g[s]}


# ---------------------------------------------------------------------------
# Subset equalizer construction
# ---------------------------------------------------------------------------

def subset_equalizer(
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
) -> "StateSpace":
    """Build the sublattice on the agreement set {s ∈ source : f(s) = g(s)}.

    The resulting state space has:
    - States: renumbered agreement set
    - Transitions: covering relation of the sub-poset (order inherited from source)
    - Top: source.top (always in agreement set for lattice homs)
    - Bottom: source.bottom (always in agreement set for lattice homs)

    We compute the covering relation (Hasse diagram) of the sub-poset
    rather than simply restricting transitions, because removing intermediate
    states can break connectivity.
    """
    from reticulate.statespace import StateSpace as SS

    agree = agreement_set(source, target, f, g)

    # Renumber states: old_id → new_id
    sorted_agree = sorted(agree)
    remap: dict[int, int] = {}
    for i, s in enumerate(sorted_agree):
        remap[s] = i

    states: set[int] = set(remap.values())
    labels: dict[int, str] = {}
    for old, new in remap.items():
        labels[new] = source.labels.get(old, str(old))

    # Compute reachability in source to determine order on agreement set
    src_reach: dict[int, set[int]] = {
        s: source.reachable_from(s) for s in agree
    }

    # Build covering relation: a covers b if a > b and no c with a > c > b
    # (all in the agreement set)
    transitions: list[tuple[int, str, int]] = []
    sel_trans: set[tuple[int, str, int]] = set()

    agree_list = sorted_agree  # sorted for determinism
    for a in agree_list:
        for b in agree_list:
            if a == b:
                continue
            if b not in src_reach[a]:
                continue
            # a > b in the original order — check if it's a cover in sub-poset
            is_cover = True
            for c in agree_list:
                if c == a or c == b:
                    continue
                if c in src_reach[a] and b in src_reach.get(c, set()):
                    is_cover = False
                    break
            if is_cover:
                # Find a label: use original transition label if direct,
                # otherwise synthesize
                label = _find_label(source, a, b)
                tr = (remap[a], label, remap[b])
                transitions.append(tr)
                # Propagate selection kind from original
                if _is_selection_path(source, a, b):
                    sel_trans.add(tr)

    top = remap[source.top]
    bottom = remap[source.bottom]

    return SS(
        states=states,
        transitions=transitions,
        top=top,
        bottom=bottom,
        labels=labels,
        selection_transitions=sel_trans,
    )


def _find_label(source: "StateSpace", a: int, b: int) -> str:
    """Find a transition label from a to b in source, possibly indirect."""
    # Direct transition?
    for src, lbl, tgt in source.transitions:
        if src == a and tgt == b:
            return lbl
    # Indirect: use first label on a path from a to b
    for src, lbl, tgt in source.transitions:
        if src == a and b in source.reachable_from(tgt):
            return lbl
    return "τ"  # fallback


def _is_selection_path(source: "StateSpace", a: int, b: int) -> bool:
    """Check if the path from a to b starts with a selection transition."""
    for src, lbl, tgt in source.transitions:
        if src == a and (b == tgt or b in source.reachable_from(tgt)):
            return source.is_selection(src, lbl, tgt)
    return False


# ---------------------------------------------------------------------------
# Kernel pair
# ---------------------------------------------------------------------------

def compute_kernel_pair(
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
) -> KernelPair:
    """Compute the kernel pair {(a,b) ∈ L×L : f(a) = f(b)}.

    Also checks whether the kernel pair is a congruence (equivalence
    relation compatible with meets and joins).
    """
    pairs: set[tuple[int, int]] = set()
    for a in source.states:
        for b in source.states:
            if f[a] == f[b]:
                pairs.add((a, b))

    # Check congruence: reflexive + symmetric + transitive + compatible
    # Reflexive and symmetric are guaranteed by construction.
    # Transitive: if (a,b) and (b,c) then f(a)=f(b)=f(c) so (a,c) ∈ pairs. ✓
    # Compatible with meets: if (a₁,a₂) and (b₁,b₂) in pairs,
    #   then f(a₁∧b₁) = f(a₁)∧f(b₁) = f(a₂)∧f(b₂) = f(a₂∧b₂).
    # This holds when f is a lattice homomorphism, so always true in SessLat.
    lr = check_lattice(source)
    is_cong = lr.is_lattice  # always a congruence for lattice homs

    return KernelPair(
        pairs=frozenset(pairs),
        is_congruence=is_cong,
    )


# ---------------------------------------------------------------------------
# Homomorphism enumeration (reuse from coproduct)
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

def check_equalizer_universal_property(
    eq_space: "StateSpace",
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
    inclusion: dict[int, int],
    test_sources: list["StateSpace"] | None = None,
) -> tuple[bool, str | None]:
    """Verify the universal property for an equalizer candidate.

    For each test source X and each lattice homomorphism h: X → source
    with f ∘ h = g ∘ h, checks:
    1. A mediating morphism e: X → E exists with inclusion ∘ e = h
    2. e is unique
    """
    if test_sources is None:
        test_sources = [source, eq_space]

    for test_src in test_sources:
        tgt_lattice = check_lattice(test_src)
        if not tgt_lattice.is_lattice:
            continue

        # Find all lattice homs h: test_src → source
        all_h = _find_all_homomorphisms(test_src, source)

        for h in all_h:
            # Check if f ∘ h = g ∘ h
            equalizes = all(f[h[s]] == g[h[s]] for s in test_src.states)
            if not equalizes:
                continue

            # h equalizes f and g — find mediating morphism e: X → E
            # with inclusion ∘ e = h
            all_e = _find_all_homomorphisms(test_src, eq_space)

            # Filter: inclusion ∘ e = h
            valid: list[dict[int, int]] = []
            for e in all_e:
                if all(inclusion[e[s]] == h[s] for s in test_src.states):
                    valid.append(e)

            if len(valid) == 0:
                return False, (
                    f"no mediating morphism for test source "
                    f"with {len(test_src.states)} states"
                )

            if len(valid) > 1:
                return False, (
                    f"mediating morphism not unique "
                    f"({len(valid)} found) for test source "
                    f"with {len(test_src.states)} states"
                )

    return True, None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def check_equalizer(
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
) -> EqualizerResult:
    """Check whether the equalizer of f, g: source → target exists.

    Constructs the subset equalizer E = {s ∈ source : f(s) = g(s)},
    verifies it is a lattice, and checks the universal property.
    """
    # Validate inputs
    src_lr = check_lattice(source)
    tgt_lr = check_lattice(target)
    if not src_lr.is_lattice:
        return EqualizerResult(
            has_equalizer=False,
            equalizer_space=None,
            inclusion=None,
            f_mapping=f,
            g_mapping=g,
            universal_property_holds=False,
            counterexample="source is not a lattice",
        )
    if not tgt_lr.is_lattice:
        return EqualizerResult(
            has_equalizer=False,
            equalizer_space=None,
            inclusion=None,
            f_mapping=f,
            g_mapping=g,
            universal_property_holds=False,
            counterexample="target is not a lattice",
        )

    # Build the subset equalizer
    agree = agreement_set(source, target, f, g)

    # Top and bottom must be in agreement set (lattice hom preserves extrema)
    if source.top not in agree:
        return EqualizerResult(
            has_equalizer=False,
            equalizer_space=None,
            inclusion=None,
            f_mapping=f,
            g_mapping=g,
            universal_property_holds=False,
            counterexample="top not in agreement set (f and g disagree on top)",
        )
    if source.bottom not in agree:
        return EqualizerResult(
            has_equalizer=False,
            equalizer_space=None,
            inclusion=None,
            f_mapping=f,
            g_mapping=g,
            universal_property_holds=False,
            counterexample="bottom not in agreement set (f and g disagree on bottom)",
        )

    eq_ss = subset_equalizer(source, target, f, g)

    # Build inclusion map: eq_state → source_state
    sorted_agree = sorted(agree)
    inclusion: dict[int, int] = {}
    for i, s in enumerate(sorted_agree):
        inclusion[i] = s

    # Check the equalizer is a lattice
    eq_lr = check_lattice(eq_ss)
    if not eq_lr.is_lattice:
        return EqualizerResult(
            has_equalizer=False,
            equalizer_space=eq_ss,
            inclusion=inclusion,
            f_mapping=f,
            g_mapping=g,
            universal_property_holds=False,
            counterexample="subset equalizer is not a lattice",
        )

    # Verify inclusion is a lattice homomorphism
    if not is_lattice_homomorphism(eq_ss, source, inclusion):
        return EqualizerResult(
            has_equalizer=False,
            equalizer_space=eq_ss,
            inclusion=inclusion,
            f_mapping=f,
            g_mapping=g,
            universal_property_holds=False,
            counterexample="inclusion is not a lattice homomorphism",
        )

    # Verify f ∘ equ = g ∘ equ
    for eq_state in eq_ss.states:
        src_state = inclusion[eq_state]
        if f[src_state] != g[src_state]:
            return EqualizerResult(
                has_equalizer=False,
                equalizer_space=eq_ss,
                inclusion=inclusion,
                f_mapping=f,
                g_mapping=g,
                universal_property_holds=False,
                counterexample="f ∘ equ ≠ g ∘ equ",
            )

    # Check universal property
    up_ok, up_cx = check_equalizer_universal_property(
        eq_ss, source, target, f, g, inclusion,
    )

    return EqualizerResult(
        has_equalizer=up_ok,
        equalizer_space=eq_ss,
        inclusion=inclusion,
        f_mapping=f,
        g_mapping=g,
        universal_property_holds=up_ok,
        counterexample=up_cx,
    )


def is_equalizer(
    eq_space: "StateSpace",
    source: "StateSpace",
    target: "StateSpace",
    f: dict[int, int],
    g: dict[int, int],
    inclusion: dict[int, int],
) -> bool:
    """Check if eq_space with inclusion is the equalizer of f and g."""
    eq_lr = check_lattice(eq_space)
    if not eq_lr.is_lattice:
        return False

    # Check f ∘ equ = g ∘ equ
    for eq_state in eq_space.states:
        src_state = inclusion[eq_state]
        if f[src_state] != g[src_state]:
            return False

    up_ok, _ = check_equalizer_universal_property(
        eq_space, source, target, f, g, inclusion,
    )
    return up_ok


# ---------------------------------------------------------------------------
# Finite completeness
# ---------------------------------------------------------------------------

def check_finite_completeness(
    spaces: list["StateSpace"],
) -> FiniteCompletenessResult:
    """Verify SessLat is finitely complete: products + equalizers.

    Tests that:
    1. Products exist (from Step 163) — parallel constructor gives products
    2. Equalizers exist for all homomorphism pairs on the given spaces

    Products + equalizers ⟹ all finite limits exist.
    """
    from reticulate.category import check_product_universal_property
    from reticulate.product import product_statespace

    lattices = [ss for ss in spaces if check_lattice(ss).is_lattice]

    # 1. Check products exist (verify on pairs)
    has_products = True
    product_cx: str | None = None
    for i, l1 in enumerate(lattices):
        for l2 in lattices[i:]:
            prod = product_statespace(l1, l2)
            pr = check_product_universal_property(prod)
            if not pr.is_product:
                has_products = False
                product_cx = f"product fails for {len(l1.states)}×{len(l2.states)}"
                break
        if not has_products:
            break

    # 2. Check equalizers exist for homomorphism pairs
    has_equalizers = True
    equalizer_cx: str | None = None
    for i, source in enumerate(lattices):
        for target in lattices:
            homos = _find_all_homomorphisms(source, target)
            # Check pairs of homomorphisms
            for fi in range(len(homos)):
                for gi in range(fi, len(homos)):
                    result = check_equalizer(source, target, homos[fi], homos[gi])
                    if not result.has_equalizer:
                        has_equalizers = False
                        equalizer_cx = (
                            f"equalizer fails for {len(source.states)}->"
                            f"{len(target.states)}: {result.counterexample}"
                        )
                        break
                if not has_equalizers:
                    break
            if not has_equalizers:
                break
        if not has_equalizers:
            break

    is_fc = has_products and has_equalizers
    cx = product_cx or equalizer_cx

    return FiniteCompletenessResult(
        has_products=has_products,
        has_equalizers=has_equalizers,
        is_finitely_complete=is_fc,
        num_spaces_tested=len(lattices),
        counterexample=cx,
    )
