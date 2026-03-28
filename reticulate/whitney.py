"""Whitney number analysis for session type lattices (Step 30g).

Whitney numbers describe the rank-level structure of a lattice:

- **W_k (second kind)**: number of elements at rank k
- **w_k (first kind)**: Σ_{ρ(x)=k} μ(⊤,x) — signed rank-level counts
- **Rank profile**: the sequence [W_0, W_1, ..., W_h]
- **Unimodality**: W_0 ≤ W_1 ≤ ... ≤ W_m ≥ ... ≥ W_h (single peak)
- **Sperner property**: max antichain size = max W_k
- **Log-concavity**: W_k² ≥ W_{k-1}·W_{k+1} (implies unimodality)
- **Rank symmetry**: W_k = W_{h-k} (symmetric rank profile)
- **Composition**: product lattice rank profile is convolution

Whitney numbers of the second kind are the f-vector of the lattice
viewed as a graded poset.  Whitney numbers of the first kind are
the h-vector (characteristic polynomial coefficients).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.characteristic import (
    _compute_corank,
    whitney_numbers_first as _w_first,
    whitney_numbers_second as _w_second,
    check_log_concave,
)
from reticulate.zeta import _compute_sccs


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WhitneyResult:
    """Complete Whitney number analysis.

    Attributes:
        height: Height of the lattice.
        whitney_first: First kind w_k (signed, from Möbius function).
        whitney_second: Second kind W_k (unsigned, element counts).
        rank_profile: [W_0, W_1, ..., W_h] as a list.
        max_rank_level: Maximum W_k and the rank k where it occurs.
        is_unimodal: True iff W_k sequence is unimodal.
        is_log_concave_second: True iff W_k sequence is log-concave.
        is_log_concave_first: True iff |w_k| sequence is log-concave.
        is_rank_symmetric: True iff W_k = W_{h-k} for all k.
        sperner_width: Maximum antichain size (= max W_k for Sperner lattices).
        is_sperner: True iff max antichain = max W_k.
        total_elements: Total elements (sum of W_k).
    """
    height: int
    whitney_first: dict[int, int]
    whitney_second: dict[int, int]
    rank_profile: list[int]
    max_rank_level: tuple[int, int]  # (k, W_k)
    is_unimodal: bool
    is_log_concave_second: bool
    is_log_concave_first: bool
    is_rank_symmetric: bool
    sperner_width: int
    is_sperner: bool
    total_elements: int


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def rank_profile(ss: "StateSpace") -> list[int]:
    """Compute [W_0, W_1, ..., W_h] — elements per corank level.

    W_k = number of elements at corank k (distance k from top in quotient).
    """
    W = _w_second(ss)
    if not W:
        return [1]
    h = max(W.keys())
    return [W.get(k, 0) for k in range(h + 1)]


def max_rank_level(ss: "StateSpace") -> tuple[int, int]:
    """Find the rank level with most elements: (k, W_k)."""
    W = _w_second(ss)
    if not W:
        return (0, 1)
    k_max = max(W, key=lambda k: W[k])
    return (k_max, W[k_max])


# ---------------------------------------------------------------------------
# Unimodality and log-concavity
# ---------------------------------------------------------------------------

def is_unimodal(profile: list[int]) -> bool:
    """Check if a sequence is unimodal (increases then decreases).

    A sequence a_0, a_1, ..., a_n is unimodal iff there exists m
    such that a_0 ≤ a_1 ≤ ... ≤ a_m ≥ ... ≥ a_n.
    """
    n = len(profile)
    if n <= 2:
        return True
    # Find peak
    increasing = True
    for i in range(1, n):
        if profile[i] < profile[i - 1]:
            increasing = False
        elif profile[i] > profile[i - 1] and not increasing:
            return False  # Increased after decreasing
    return True


def is_rank_symmetric(ss: "StateSpace") -> bool:
    """Check if W_k = W_{h-k} for all k (rank symmetry).

    Self-dual lattices have symmetric rank profiles.
    """
    prof = rank_profile(ss)
    h = len(prof) - 1
    for k in range(h + 1):
        if prof[k] != prof[h - k]:
            return False
    return True


# ---------------------------------------------------------------------------
# Sperner property
# ---------------------------------------------------------------------------

def sperner_width(ss: "StateSpace") -> int:
    """Compute the Sperner width (maximum antichain size).

    For Sperner lattices (e.g., Boolean lattices), this equals max W_k.
    For general lattices, the actual maximum antichain may differ.

    We compute the actual maximum antichain size using the width
    computation from the zeta module.
    """
    from reticulate.zeta import compute_width
    return compute_width(ss)


def is_sperner(ss: "StateSpace") -> bool:
    """Check if the lattice has the Sperner property.

    A graded poset has the Sperner property iff the maximum antichain
    size equals the maximum Whitney number of the second kind.
    """
    prof = rank_profile(ss)
    max_W = max(prof) if prof else 0
    actual_width = sperner_width(ss)
    return actual_width == max_W


# ---------------------------------------------------------------------------
# Convolution (composition under product)
# ---------------------------------------------------------------------------

def convolve_profiles(p1: list[int], p2: list[int]) -> list[int]:
    """Convolve two rank profiles: (p1 * p2)[k] = Σ_{i+j=k} p1[i]·p2[j].

    This gives the rank profile of the product lattice L₁ × L₂.
    """
    if not p1 or not p2:
        return []
    n = len(p1) + len(p2) - 1
    result = [0] * n
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i + j] += p1[i] * p2[j]
    return result


def verify_profile_convolution(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
) -> bool:
    """Verify rank_profile(L₁×L₂) = rank_profile(L₁) * rank_profile(L₂)."""
    p_left = rank_profile(ss_left)
    p_right = rank_profile(ss_right)
    p_product = rank_profile(ss_product)
    p_expected = convolve_profiles(p_left, p_right)
    return p_product == p_expected


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_whitney(ss: "StateSpace") -> WhitneyResult:
    """Complete Whitney number analysis."""
    w1 = _w_first(ss)
    w2 = _w_second(ss)
    prof = rank_profile(ss)
    h = len(prof) - 1

    k_max, W_max = max_rank_level(ss)

    unimodal = is_unimodal(prof)
    lc_second = check_log_concave(prof)

    # Log-concavity of |w_k|
    w1_seq = [w1.get(k, 0) for k in range(h + 1)]
    lc_first = check_log_concave(w1_seq)

    sym = is_rank_symmetric(ss)
    sw = sperner_width(ss)
    sp = (sw == max(prof) if prof else True)

    scc_map, _ = _compute_sccs(ss)
    total = len(set(scc_map.values()))

    return WhitneyResult(
        height=h,
        whitney_first=w1,
        whitney_second=w2,
        rank_profile=prof,
        max_rank_level=(k_max, W_max),
        is_unimodal=unimodal,
        is_log_concave_second=lc_second,
        is_log_concave_first=lc_first,
        is_rank_symmetric=sym,
        sperner_width=sw,
        is_sperner=sp,
        total_elements=total,
    )
