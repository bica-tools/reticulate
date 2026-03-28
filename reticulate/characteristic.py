"""Rota characteristic polynomial for session type lattices (Step 30c).

The characteristic polynomial χ_P(t) of a finite poset P with rank function
ρ is defined as:

    χ_P(t) = Σ_{x ∈ P} μ(⊤, x) · t^{h - ρ(x)}

where h = ρ(⊤) is the height and μ is the Möbius function.

This module provides deep analysis beyond the basic computation in matrix.py:

- **Characteristic polynomial** with proper rank function from zeta.py
- **Whitney numbers** of the first and second kind
- **Evaluation** at specific points: χ(0) = μ(⊤,⊥), χ(1), χ(-1)
- **Factorization** under parallel: χ(L₁ × L₂) = χ(L₁) · χ(L₂)
- **Log-concavity** test on absolute Whitney numbers
- **Polynomial arithmetic**: multiply, evaluate, derivative
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _compute_sccs,
    compute_rank,
    mobius_function as _mobius_func,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CharPolyResult:
    """Complete characteristic polynomial analysis.

    Attributes:
        coefficients: Polynomial coefficients [a_h, a_{h-1}, ..., a_0] (highest degree first).
        degree: Degree of the polynomial (= height of lattice).
        height: Height of the lattice.
        whitney_first: Whitney numbers of the first kind: w_k = Σ_{ρ(x)=k} μ(⊤,x).
        whitney_second: Whitney numbers of the second kind: W_k = |{x : ρ(x) = k}|.
        eval_at_0: χ(0) = μ(⊤,⊥) (the Möbius value).
        eval_at_1: χ(1).
        eval_at_neg1: χ(-1).
        is_log_concave: True iff |w_0|, |w_1|, ..., |w_h| is log-concave.
        num_states: Number of states.
    """
    coefficients: list[int]
    degree: int
    height: int
    whitney_first: dict[int, int]
    whitney_second: dict[int, int]
    eval_at_0: int
    eval_at_1: int
    eval_at_neg1: int
    is_log_concave: bool
    num_states: int


# ---------------------------------------------------------------------------
# Polynomial arithmetic
# ---------------------------------------------------------------------------

def poly_evaluate(coeffs: list[int], t: int | float) -> int | float:
    """Evaluate polynomial with coefficients [a_n, ..., a_0] at t.

    Uses Horner's method.
    """
    result: int | float = 0
    for c in coeffs:
        result = result * t + c
    return result


def poly_multiply(a: list[int], b: list[int]) -> list[int]:
    """Multiply two polynomials (coefficient lists, highest degree first)."""
    if not a or not b:
        return [0]
    na, nb = len(a), len(b)
    result = [0] * (na + nb - 1)
    for i in range(na):
        for j in range(nb):
            result[i + j] += a[i] * b[j]
    return result


def poly_derivative(coeffs: list[int]) -> list[int]:
    """Formal derivative of polynomial."""
    n = len(coeffs) - 1  # degree
    if n <= 0:
        return [0]
    return [(n - i) * coeffs[i] for i in range(n)]


def poly_to_string(coeffs: list[int], var: str = "t") -> str:
    """Pretty-print polynomial."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        deg = n - i
        if c == 0:
            continue
        if deg == 0:
            terms.append(str(c))
        elif deg == 1:
            if c == 1:
                terms.append(var)
            elif c == -1:
                terms.append(f"-{var}")
            else:
                terms.append(f"{c}{var}")
        else:
            if c == 1:
                terms.append(f"{var}^{deg}")
            elif c == -1:
                terms.append(f"-{var}^{deg}")
            else:
                terms.append(f"{c}{var}^{deg}")
    if not terms:
        return "0"
    result = terms[0]
    for t in terms[1:]:
        if t.startswith("-"):
            result += f" - {t[1:]}"
        else:
            result += f" + {t}"
    return result


# ---------------------------------------------------------------------------
# Characteristic polynomial computation
# ---------------------------------------------------------------------------

def characteristic_polynomial(ss: "StateSpace") -> list[int]:
    """Compute the characteristic polynomial χ_P(t).

    χ_P(t) = Σ_{x} μ(⊤, x) · t^{h - ρ(x)}

    Uses the rank function from zeta.py (longest chain to bottom,
    SCC-aware) and Möbius function from the incidence algebra.

    Returns coefficients [a_h, a_{h-1}, ..., a_0] (highest degree first).
    """
    mu = _mobius_func(ss)
    top = ss.top
    scc_map, scc_members = _compute_sccs(ss)

    # Compute corank: BFS distance from top (shortest path in quotient DAG)
    from reticulate.zeta import _adjacency
    adj = _adjacency(ss)

    # BFS on quotient
    reps = sorted(scc_members.keys())
    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    top_rep = scc_map[top]
    corank: dict[int, int] = {r: -1 for r in reps}
    corank[top_rep] = 0
    queue = [top_rep]
    visited: set[int] = {top_rep}
    while queue:
        u = queue.pop(0)
        for v in q_adj[u]:
            if v not in visited:
                visited.add(v)
                corank[v] = corank[u] + 1
                queue.append(v)
    for r in reps:
        if corank[r] < 0:
            corank[r] = 0

    max_corank = max(corank.values()) if corank else 0
    height = max_corank

    # Build polynomial: χ(t) = Σ_x μ(⊤,x) · t^{max_corank - corank(x)}
    # coeffs[i] = coefficient of t^{max_corank - i}
    coeffs = [0] * (height + 1)

    counted_sccs: set[int] = set()
    for x in sorted(ss.states):
        rep = scc_map[x]
        if rep in counted_sccs:
            continue
        counted_sccs.add(rep)

        mu_val = mu.get((top, x), 0)
        cr = corank[rep]
        degree = max_corank - cr
        if 0 <= degree <= max_corank:
            coeffs[max_corank - degree] += mu_val

    # Remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)

    return coeffs


# ---------------------------------------------------------------------------
# Whitney numbers
# ---------------------------------------------------------------------------

def _compute_corank(ss: "StateSpace") -> dict[int, int]:
    """BFS distance from top on quotient DAG."""
    from reticulate.zeta import _adjacency
    scc_map, scc_members = _compute_sccs(ss)
    adj = _adjacency(ss)
    reps = sorted(scc_members.keys())

    q_adj: dict[int, set[int]] = {r: set() for r in reps}
    for s in ss.states:
        for t in adj[s]:
            sr, tr = scc_map[s], scc_map[t]
            if sr != tr:
                q_adj[sr].add(tr)

    top_rep = scc_map[ss.top]
    corank: dict[int, int] = {r: -1 for r in reps}
    corank[top_rep] = 0
    queue = [top_rep]
    visited: set[int] = {top_rep}
    while queue:
        u = queue.pop(0)
        for v in q_adj[u]:
            if v not in visited:
                visited.add(v)
                corank[v] = corank[u] + 1
                queue.append(v)
    for r in reps:
        if corank[r] < 0:
            corank[r] = 0

    # Map back to states
    return {s: corank[scc_map[s]] for s in ss.states}


def whitney_numbers_first(ss: "StateSpace") -> dict[int, int]:
    """Whitney numbers of the first kind: w_k = Σ_{corank(x)=k} μ(⊤, x).

    These are the coefficients of the characteristic polynomial.
    """
    corank = _compute_corank(ss)
    mu = _mobius_func(ss)
    top = ss.top
    scc_map, _ = _compute_sccs(ss)

    w: dict[int, int] = {}
    counted: set[int] = set()
    for x in sorted(ss.states):
        rep = scc_map[x]
        if rep in counted:
            continue
        counted.add(rep)
        cr = corank[x]
        mu_val = mu.get((top, x), 0)
        w[cr] = w.get(cr, 0) + mu_val

    return w


def whitney_numbers_second(ss: "StateSpace") -> dict[int, int]:
    """Whitney numbers of the second kind: W_k = |{x : corank(x) = k}|.

    These count the number of elements at each corank level.
    """
    corank = _compute_corank(ss)
    scc_map, _ = _compute_sccs(ss)

    W: dict[int, int] = {}
    counted: set[int] = set()
    for x in sorted(ss.states):
        rep = scc_map[x]
        if rep in counted:
            continue
        counted.add(rep)
        cr = corank[x]
        W[cr] = W.get(cr, 0) + 1

    return W


# ---------------------------------------------------------------------------
# Log-concavity
# ---------------------------------------------------------------------------

def check_log_concave(values: list[int]) -> bool:
    """Check if |values[0]|, |values[1]|, ..., |values[n]| is log-concave.

    A sequence a_0, a_1, ..., a_n is log-concave iff a_k² ≥ a_{k-1}·a_{k+1}
    for all 1 ≤ k ≤ n-1.
    """
    abs_vals = [abs(v) for v in values]
    for k in range(1, len(abs_vals) - 1):
        if abs_vals[k] ** 2 < abs_vals[k - 1] * abs_vals[k + 1]:
            return False
    return True


def is_whitney_log_concave(ss: "StateSpace") -> bool:
    """Check if the absolute Whitney numbers of the first kind are log-concave.

    This is related to the Heron-Rota-Welsh conjecture (now proven for
    representable matroids by Adiprasito-Huh-Katz 2018).
    """
    w = whitney_numbers_first(ss)
    if not w:
        return True
    max_rank = max(w.keys())
    values = [w.get(k, 0) for k in range(max_rank + 1)]
    return check_log_concave(values)


# ---------------------------------------------------------------------------
# Factorization under parallel
# ---------------------------------------------------------------------------

def verify_factorization(
    ss_left: "StateSpace",
    ss_right: "StateSpace",
    ss_product: "StateSpace",
) -> bool:
    """Verify χ(L₁ × L₂)(t) = χ(L₁)(t) · χ(L₂)(t).

    The characteristic polynomial factors under product lattices.
    """
    chi_left = characteristic_polynomial(ss_left)
    chi_right = characteristic_polynomial(ss_right)
    chi_product = characteristic_polynomial(ss_product)
    chi_expected = poly_multiply(chi_left, chi_right)

    # Normalize: remove leading zeros from both
    while len(chi_product) > 1 and chi_product[0] == 0:
        chi_product.pop(0)
    while len(chi_expected) > 1 and chi_expected[0] == 0:
        chi_expected.pop(0)

    return chi_product == chi_expected


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_characteristic(ss: "StateSpace") -> CharPolyResult:
    """Complete characteristic polynomial analysis."""
    coeffs = characteristic_polynomial(ss)
    degree = len(coeffs) - 1
    height = degree  # degree equals height by construction

    w_first = whitney_numbers_first(ss)
    w_second = whitney_numbers_second(ss)

    eval_0 = int(poly_evaluate(coeffs, 0))
    eval_1 = int(poly_evaluate(coeffs, 1))
    eval_neg1 = int(poly_evaluate(coeffs, -1))

    # Log-concavity of Whitney numbers
    max_corank = max(w_first.keys()) if w_first else 0
    w_seq = [w_first.get(k, 0) for k in range(max_corank + 1)]
    log_conc = check_log_concave(w_seq)

    scc_map, _ = _compute_sccs(ss)
    n_quotient = len(set(scc_map.values()))

    return CharPolyResult(
        coefficients=coeffs,
        degree=degree,
        height=height,
        whitney_first=w_first,
        whitney_second=w_second,
        eval_at_0=eval_0,
        eval_at_1=eval_1,
        eval_at_neg1=eval_neg1,
        is_log_concave=log_conc,
        num_states=n_quotient,
    )
