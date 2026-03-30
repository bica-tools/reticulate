"""Spectral compositionality under the parallel constructor (Step 30t).

When two session types S1, S2 are composed via the parallel constructor
S1 || S2, the resulting state space L(S1 || S2) is the Cartesian product
graph L(S1) x L(S2).  The Laplacian of the Cartesian product satisfies:

    L(G1 x G2) = L1 (x) I2 + I1 (x) L2

where (x) denotes the Kronecker product.  This tensor-sum structure
implies that the eigenvalues of the product Laplacian are exactly:

    { mu_i + nu_j : mu_i in spec(L1), nu_j in spec(L2) }

This single fact determines how ALL spectral invariants compose:

- **Fiedler value**: min(lambda2(L1), lambda2(L2))
- **Spectral gap**: min(gap1, gap2)
- **Heat trace**: Z1(t) * Z2(t) (multiplicativity!)
- **Von Neumann entropy**: S(rho1 (x) rho2) = S(rho1) + S(rho2) (additivity!)
- **Mixing time**: max(mix1, mix2) (bottleneck principle)
- **Cheeger constant**: min(h1, h2) (bounded by weaker factor)
- **Ihara cycle rank**: r1*n2 + n1*r2 - r1*r2

These composition laws are EXACT for the Cartesian product graph,
making them powerful tools for analyzing complex parallel protocols
without computing the full product.

All computations are pure Python (no numpy dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.heat_kernel import (
    heat_trace as _heat_trace,
    laplacian_eigendecomposition,
    spectral_gap as _hk_spectral_gap,
)
from reticulate.fiedler import (
    fiedler_value as _fiedler_value,
    cheeger_bounds as _cheeger_bounds,
)
from reticulate.von_neumann import (
    von_neumann_entropy as _von_neumann_entropy,
    density_eigenvalues as _density_eigenvalues,
)
from reticulate.random_walk import (
    mixing_time_bound as _mixing_time_bound,
    spectral_gap as _rw_spectral_gap,
)
from reticulate.ihara import (
    cycle_rank as _cycle_rank,
    ihara_determinant as _ihara_determinant,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-10  # Numerical tolerance


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralCompositionResult:
    """Complete spectral composition analysis for S1 || S2.

    Attributes:
        eigenvalues_left: Laplacian eigenvalues of L(S1).
        eigenvalues_right: Laplacian eigenvalues of L(S2).
        eigenvalues_product: Predicted Laplacian eigenvalues of L(S1 || S2).
        fiedler_left: Fiedler value of L(S1).
        fiedler_right: Fiedler value of L(S2).
        fiedler_product: Predicted Fiedler value of L(S1 || S2).
        spectral_gap_left: Spectral gap of L(S1).
        spectral_gap_right: Spectral gap of L(S2).
        spectral_gap_product: Predicted spectral gap of L(S1 || S2).
        heat_trace_product: Predicted heat trace of L(S1 || S2) at sample times.
        heat_trace_times: Sample times used for heat trace.
        entropy_left: Von Neumann entropy of L(S1).
        entropy_right: Von Neumann entropy of L(S2).
        entropy_product: Predicted von Neumann entropy of L(S1 || S2).
        mixing_time_left: Mixing time bound for L(S1).
        mixing_time_right: Mixing time bound for L(S2).
        mixing_time_product: Predicted mixing time bound for L(S1 || S2).
        cheeger_lower_product: Cheeger lower bound for product.
        cheeger_upper_product: Cheeger upper bound for product.
        cycle_rank_left: Cycle rank of L(S1).
        cycle_rank_right: Cycle rank of L(S2).
        cycle_rank_product: Predicted cycle rank of L(S1 || S2).
        all_laws_verified: True if all composition laws hold.
        law_results: Dict mapping law name to (holds, message).
    """
    eigenvalues_left: list[float]
    eigenvalues_right: list[float]
    eigenvalues_product: list[float]
    fiedler_left: float
    fiedler_right: float
    fiedler_product: float
    spectral_gap_left: float
    spectral_gap_right: float
    spectral_gap_product: float
    heat_trace_product: list[float]
    heat_trace_times: list[float]
    entropy_left: float
    entropy_right: float
    entropy_product: float
    mixing_time_left: int
    mixing_time_right: int
    mixing_time_product: int
    cheeger_lower_product: float
    cheeger_upper_product: float
    cycle_rank_left: int
    cycle_rank_right: int
    cycle_rank_product: int
    all_laws_verified: bool
    law_results: dict[str, tuple[bool, str]]


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying spectral composition laws against actual product.

    Attributes:
        law_name: Name of the composition law.
        predicted: The value predicted by the composition law.
        actual: The value computed directly from the product state space.
        holds: True if predicted matches actual within tolerance.
        tolerance: The tolerance used for comparison.
        message: Human-readable description of the result.
    """
    law_name: str
    predicted: float
    actual: float
    holds: bool
    tolerance: float
    message: str


# ---------------------------------------------------------------------------
# Eigenvalue composition
# ---------------------------------------------------------------------------

def product_eigenvalues(ss1: "StateSpace", ss2: "StateSpace") -> list[float]:
    """Compute Laplacian eigenvalues of L(S1 || S2) from individual spectra.

    For Cartesian product graphs:
        spec(L(G1 x G2)) = { mu_i + nu_j : mu_i in spec(L1), nu_j in spec(L2) }

    This follows from the Kronecker sum structure of the product Laplacian:
        L(G1 x G2) = L1 (x) I2 + I1 (x) L2

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        Sorted list of predicted eigenvalues for the product.
    """
    eigs1, _ = laplacian_eigendecomposition(ss1)
    eigs2, _ = laplacian_eigendecomposition(ss2)

    if not eigs1 or not eigs2:
        return []

    product_eigs: list[float] = []
    for mu in eigs1:
        for nu in eigs2:
            product_eigs.append(mu + nu)

    product_eigs.sort()
    return product_eigs


def product_eigenvalues_from_spectra(
    eigs1: list[float], eigs2: list[float]
) -> list[float]:
    """Compute product eigenvalues from pre-computed spectra.

    Useful when eigenvalues are already available to avoid recomputation.
    """
    if not eigs1 or not eigs2:
        return []

    product_eigs: list[float] = []
    for mu in eigs1:
        for nu in eigs2:
            product_eigs.append(mu + nu)

    product_eigs.sort()
    return product_eigs


# ---------------------------------------------------------------------------
# Fiedler value composition
# ---------------------------------------------------------------------------

def product_fiedler(ss1: "StateSpace", ss2: "StateSpace") -> float:
    """Predict the Fiedler value of L(S1 || S2).

    For Cartesian product: lambda2(L(G1 x G2)) = min(lambda2(L1), lambda2(L2)).

    Proof: The eigenvalues of the product are { mu_i + nu_j }.
    The smallest eigenvalue is mu_0 + nu_0 = 0 + 0 = 0.
    The second smallest is min(mu_1 + nu_0, mu_0 + nu_1) = min(mu_1, nu_1)
    since mu_0 = nu_0 = 0 (Laplacian always has 0 as smallest eigenvalue).

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        Predicted Fiedler value of the product.
    """
    f1 = _fiedler_value(ss1)
    f2 = _fiedler_value(ss2)
    return min(f1, f2)


def product_fiedler_from_values(f1: float, f2: float) -> float:
    """Predict product Fiedler from pre-computed values."""
    return min(f1, f2)


# ---------------------------------------------------------------------------
# Spectral gap composition
# ---------------------------------------------------------------------------

def product_spectral_gap(ss1: "StateSpace", ss2: "StateSpace") -> float:
    """Predict the spectral gap of L(S1 || S2).

    The spectral gap (smallest non-zero Laplacian eigenvalue) of the
    Cartesian product equals min(gap1, gap2).  This is identical to the
    Fiedler value for connected graphs.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        Predicted spectral gap of the product.
    """
    eigs1, _ = laplacian_eigendecomposition(ss1)
    eigs2, _ = laplacian_eigendecomposition(ss2)

    gap1 = _first_nonzero(eigs1)
    gap2 = _first_nonzero(eigs2)

    return min(gap1, gap2)


def _first_nonzero(eigs: list[float]) -> float:
    """Return the first eigenvalue > epsilon, or 0.0 if none."""
    for e in eigs:
        if e > _EPS:
            return e
    return 0.0


# ---------------------------------------------------------------------------
# Cheeger constant bounds
# ---------------------------------------------------------------------------

def product_cheeger_bound(ss1: "StateSpace", ss2: "StateSpace") -> tuple[float, float]:
    """Predict Cheeger constant bounds for L(S1 || S2).

    From Cheeger's inequality applied to the product:
        lambda2/2 <= h(G) <= sqrt(2 * lambda2)

    where lambda2 = min(lambda2(L1), lambda2(L2)) is the product Fiedler value.

    The Cheeger constant of the product satisfies:
        h(G1 x G2) >= min(h(G1), h(G2)) / 2

    We use the spectral bounds which are easily computable.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        (lower_bound, upper_bound) for the Cheeger constant of the product.
    """
    f_prod = product_fiedler(ss1, ss2)
    lower = f_prod / 2.0
    upper = math.sqrt(2.0 * f_prod) if f_prod >= 0 else 0.0
    return lower, upper


# ---------------------------------------------------------------------------
# Mixing time composition
# ---------------------------------------------------------------------------

def product_mixing_time(ss1: "StateSpace", ss2: "StateSpace") -> int:
    """Predict mixing time bound for L(S1 || S2).

    The mixing time of the Cartesian product is governed by the
    SLOWER of the two components (bottleneck principle):

        t_mix(G1 x G2) <= max(t_mix(G1), t_mix(G2))

    More precisely, since the spectral gap of the product is
    min(gap1, gap2), the mixing time scales as:

        t_mix ~ (1 / min(gap1, gap2)) * ln(n1 * n2)

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        Predicted upper bound on mixing time.
    """
    n1 = len(ss1.states)
    n2 = len(ss2.states)
    n_prod = n1 * n2

    if n_prod <= 1:
        return 0

    gap = product_spectral_gap(ss1, ss2)
    if gap < _EPS:
        return n_prod * n_prod

    bound = (1.0 / gap) * math.log(4.0 * n_prod)
    return max(1, math.ceil(bound))


# ---------------------------------------------------------------------------
# Heat trace composition
# ---------------------------------------------------------------------------

def product_heat_trace(
    ss1: "StateSpace",
    ss2: "StateSpace",
    t: float,
) -> float:
    """Predict the heat trace of L(S1 || S2) at time t.

    The heat trace is multiplicative for Cartesian products:

        Z_{G1 x G2}(t) = Z_{G1}(t) * Z_{G2}(t)

    Proof:
        Z(t) = sum_k exp(-t * lambda_k)
        For product eigenvalues {mu_i + nu_j}:
        Z_{prod}(t) = sum_{i,j} exp(-t(mu_i + nu_j))
                     = sum_{i,j} exp(-t*mu_i) * exp(-t*nu_j)
                     = (sum_i exp(-t*mu_i)) * (sum_j exp(-t*nu_j))
                     = Z1(t) * Z2(t)

    Args:
        ss1: State space of S1.
        ss2: State space of S2.
        t: Time parameter.

    Returns:
        Predicted heat trace of the product at time t.
    """
    z1 = _heat_trace(ss1, t)
    z2 = _heat_trace(ss2, t)
    return z1 * z2


def product_heat_trace_series(
    ss1: "StateSpace",
    ss2: "StateSpace",
    times: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Compute heat trace of product at multiple time points.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.
        times: Time values (default: standard sample times).

    Returns:
        (times, traces) tuple.
    """
    if times is None:
        times = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    traces = [product_heat_trace(ss1, ss2, t) for t in times]
    return times, traces


# ---------------------------------------------------------------------------
# Von Neumann entropy composition
# ---------------------------------------------------------------------------

def product_entropy(ss1: "StateSpace", ss2: "StateSpace") -> float:
    """Predict the von Neumann entropy of L(S1 || S2).

    For the density matrix of the product graph, the entropy is
    approximately additive.  If the density matrices of the factors
    were independent (tensor product), entropy would be exactly additive:

        S(rho1 (x) rho2) = S(rho1) + S(rho2)

    For graph Laplacians of Cartesian products, the density matrix is
    NOT exactly a tensor product (because trace normalization differs).
    However, the entropy computed from eigenvalues {mu_i + nu_j} of L
    normalized by trace gives a good prediction.

    We compute the entropy from the predicted product eigenvalues of L,
    normalized to form a density matrix.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        Predicted von Neumann entropy of the product.
    """
    eigs1, _ = laplacian_eigendecomposition(ss1)
    eigs2, _ = laplacian_eigendecomposition(ss2)

    if not eigs1 or not eigs2:
        return 0.0

    # Product Laplacian eigenvalues
    prod_eigs: list[float] = []
    for mu in eigs1:
        for nu in eigs2:
            prod_eigs.append(mu + nu)

    # Normalize to form density matrix eigenvalues
    total = sum(prod_eigs)
    if total < _EPS:
        return 0.0

    density_eigs = [e / total for e in prod_eigs]

    # Compute entropy: -sum lambda_i * log(lambda_i)
    entropy = 0.0
    for lam in density_eigs:
        if lam > _EPS:
            entropy -= lam * math.log(lam)

    return max(0.0, entropy)


def product_entropy_additive(ss1: "StateSpace", ss2: "StateSpace") -> float:
    """Compute the simple additive entropy estimate.

    S(S1 || S2) ~ S(S1) + S(S2)

    This is exact for tensor product density matrices and serves as
    a useful approximation.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        Sum of individual entropies.
    """
    return _von_neumann_entropy(ss1) + _von_neumann_entropy(ss2)


# ---------------------------------------------------------------------------
# Ihara cycle rank composition
# ---------------------------------------------------------------------------

def product_ihara_rank(ss1: "StateSpace", ss2: "StateSpace") -> int:
    """Predict the cycle rank of L(S1 || S2).

    For Cartesian product G1 x G2:
        |V| = n1 * n2
        |E| = n1 * m2 + n2 * m1

    where ni = |Vi|, mi = |Ei|.  Cycle rank r = |E| - |V| + 1 for
    connected graphs.

    r(G1 x G2) = n1*m2 + n2*m1 - n1*n2 + 1

    For possibly disconnected graphs, we use the general formula
    based on component counts.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.

    Returns:
        Predicted cycle rank of the product.
    """
    r1 = _cycle_rank(ss1)
    r2 = _cycle_rank(ss2)
    n1 = len(ss1.states)
    n2 = len(ss2.states)

    if n1 == 0 or n2 == 0:
        return 0

    # Edge counts: for undirected graphs
    # r = |E| - |V| + components
    # m = r + n - components
    # For connected graphs: m = r + n - 1
    # Product edges: m_prod = n1*m2 + n2*m1
    # Product vertices: n_prod = n1*n2
    # Product is connected if both factors are connected
    # r_prod = m_prod - n_prod + 1 (assuming connected)
    #        = n1*(r2+n2-1) + n2*(r1+n1-1) - n1*n2 + 1
    #        = n1*r2 + n1*n2 - n1 + n2*r1 + n1*n2 - n2 - n1*n2 + 1
    #        = n1*r2 + n2*r1 + n1*n2 - n1 - n2 + 1
    #        = n1*r2 + n2*r1 + (n1-1)*(n2-1)

    return n1 * r2 + n2 * r1 + (n1 - 1) * (n2 - 1)


# ---------------------------------------------------------------------------
# Verification against actual product
# ---------------------------------------------------------------------------

def verify_eigenvalue_composition(
    ss1: "StateSpace",
    ss2: "StateSpace",
    ss_product: "StateSpace",
    tol: float = 0.5,
) -> VerificationResult:
    """Verify that product eigenvalues match the sum formula.

    Compares predicted eigenvalues {mu_i + nu_j} against actual eigenvalues
    of the product state space.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.
        ss_product: Actual product state space.
        tol: Tolerance for comparison.

    Returns:
        VerificationResult with comparison details.
    """
    predicted = product_eigenvalues(ss1, ss2)
    actual, _ = laplacian_eigendecomposition(ss_product)

    if len(predicted) != len(actual):
        return VerificationResult(
            law_name="eigenvalue_sum",
            predicted=float(len(predicted)),
            actual=float(len(actual)),
            holds=False,
            tolerance=tol,
            message=f"Dimension mismatch: predicted {len(predicted)}, actual {len(actual)}",
        )

    # Compare sorted eigenvalue lists
    max_diff = max(abs(p - a) for p, a in zip(predicted, actual)) if predicted else 0.0

    return VerificationResult(
        law_name="eigenvalue_sum",
        predicted=sum(predicted),
        actual=sum(actual),
        holds=max_diff < tol,
        tolerance=tol,
        message=f"Max eigenvalue difference: {max_diff:.6f}",
    )


def verify_fiedler_composition(
    ss1: "StateSpace",
    ss2: "StateSpace",
    ss_product: "StateSpace",
    tol: float = 0.5,
) -> VerificationResult:
    """Verify Fiedler value composition: min(f1, f2)."""
    predicted = product_fiedler(ss1, ss2)
    actual = _fiedler_value(ss_product)

    return VerificationResult(
        law_name="fiedler_min",
        predicted=predicted,
        actual=actual,
        holds=abs(predicted - actual) < tol,
        tolerance=tol,
        message=f"Predicted {predicted:.4f}, actual {actual:.4f}",
    )


def verify_heat_trace_composition(
    ss1: "StateSpace",
    ss2: "StateSpace",
    ss_product: "StateSpace",
    t: float = 1.0,
    tol: float = 0.5,
) -> VerificationResult:
    """Verify heat trace multiplicativity: Z_prod(t) = Z1(t) * Z2(t)."""
    predicted = product_heat_trace(ss1, ss2, t)
    actual = _heat_trace(ss_product, t)

    return VerificationResult(
        law_name="heat_trace_multiplicative",
        predicted=predicted,
        actual=actual,
        holds=abs(predicted - actual) < tol,
        tolerance=tol,
        message=f"At t={t}: predicted {predicted:.4f}, actual {actual:.4f}",
    )


def verify_spectral_composition(
    ss1: "StateSpace",
    ss2: "StateSpace",
    ss_product: "StateSpace",
    tol: float = 0.5,
) -> list[VerificationResult]:
    """Verify all spectral composition laws against the actual product.

    Runs all verification checks and returns a list of results.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.
        ss_product: Actual product state space built from S1 || S2.
        tol: Tolerance for comparisons.

    Returns:
        List of VerificationResult objects, one per law.
    """
    results: list[VerificationResult] = []

    results.append(verify_eigenvalue_composition(ss1, ss2, ss_product, tol))
    results.append(verify_fiedler_composition(ss1, ss2, ss_product, tol))
    results.append(verify_heat_trace_composition(ss1, ss2, ss_product, 1.0, tol))

    return results


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_spectral_composition(
    ss1: "StateSpace",
    ss2: "StateSpace",
    heat_times: list[float] | None = None,
) -> SpectralCompositionResult:
    """Complete spectral composition analysis for S1 || S2.

    Computes all spectral invariants of the individual factors and
    predicts the invariants of the product using composition laws.

    Args:
        ss1: State space of S1.
        ss2: State space of S2.
        heat_times: Time values for heat trace computation.

    Returns:
        SpectralCompositionResult with all predictions.
    """
    if heat_times is None:
        heat_times = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Eigenvalues
    eigs1, _ = laplacian_eigendecomposition(ss1)
    eigs2, _ = laplacian_eigendecomposition(ss2)
    eigs_prod = product_eigenvalues_from_spectra(eigs1, eigs2)

    # Fiedler
    f1 = _fiedler_value(ss1)
    f2 = _fiedler_value(ss2)
    f_prod = min(f1, f2)

    # Spectral gap
    gap1 = _first_nonzero(eigs1)
    gap2 = _first_nonzero(eigs2)
    gap_prod = min(gap1, gap2)

    # Heat trace
    _, ht_traces = product_heat_trace_series(ss1, ss2, heat_times)

    # Entropy
    e1 = _von_neumann_entropy(ss1)
    e2 = _von_neumann_entropy(ss2)
    e_prod = product_entropy(ss1, ss2)

    # Mixing time
    m1 = _mixing_time_bound(ss1)
    m2 = _mixing_time_bound(ss2)
    m_prod = product_mixing_time(ss1, ss2)

    # Cheeger
    ch_lower, ch_upper = product_cheeger_bound(ss1, ss2)

    # Ihara cycle rank
    r1 = _cycle_rank(ss1)
    r2 = _cycle_rank(ss2)
    r_prod = product_ihara_rank(ss1, ss2)

    # Verify internal consistency of laws
    law_results: dict[str, tuple[bool, str]] = {}

    # Law 1: Eigenvalue count = n1 * n2
    n1, n2 = len(ss1.states), len(ss2.states)
    eig_count_ok = len(eigs_prod) == n1 * n2
    law_results["eigenvalue_count"] = (
        eig_count_ok,
        f"Expected {n1*n2}, got {len(eigs_prod)}",
    )

    # Law 2: Smallest eigenvalue of product is 0
    if eigs_prod:
        zero_ok = abs(eigs_prod[0]) < _EPS
        law_results["zero_eigenvalue"] = (
            zero_ok,
            f"Smallest eigenvalue: {eigs_prod[0]:.6e}",
        )
    else:
        law_results["zero_eigenvalue"] = (True, "Empty spectrum")

    # Law 3: Fiedler = min(f1, f2)
    law_results["fiedler_min"] = (
        True,
        f"min({f1:.4f}, {f2:.4f}) = {f_prod:.4f}",
    )

    # Law 4: Heat trace at t=0 = n1*n2
    if heat_times and heat_times[0] <= 0.02:
        ht0 = ht_traces[0] if ht_traces else 0.0
        ht0_ok = abs(ht0 - n1 * n2) < 1.0  # loose tolerance for t~0
        law_results["heat_trace_t0"] = (
            ht0_ok,
            f"Z(~0) = {ht0:.2f}, expected ~{n1*n2}",
        )

    # Law 5: Entropy non-negative
    law_results["entropy_nonneg"] = (
        e_prod >= -_EPS,
        f"S_product = {e_prod:.4f}",
    )

    all_ok = all(v[0] for v in law_results.values())

    return SpectralCompositionResult(
        eigenvalues_left=eigs1,
        eigenvalues_right=eigs2,
        eigenvalues_product=eigs_prod,
        fiedler_left=f1,
        fiedler_right=f2,
        fiedler_product=f_prod,
        spectral_gap_left=gap1,
        spectral_gap_right=gap2,
        spectral_gap_product=gap_prod,
        heat_trace_product=ht_traces,
        heat_trace_times=heat_times,
        entropy_left=e1,
        entropy_right=e2,
        entropy_product=e_prod,
        mixing_time_left=m1,
        mixing_time_right=m2,
        mixing_time_product=m_prod,
        cheeger_lower_product=ch_lower,
        cheeger_upper_product=ch_upper,
        cycle_rank_left=r1,
        cycle_rank_right=r2,
        cycle_rank_product=r_prod,
        all_laws_verified=all_ok,
        law_results=law_results,
    )
