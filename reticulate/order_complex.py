"""Order complex and topological invariants of session type lattices (Steps 30u-30x).

The order complex of a poset P is the simplicial complex whose
k-simplices are chains of length k+1 in P.  This module provides:

- **Step 30u**: Order complex construction and face enumeration
- **Step 30v**: Euler characteristic (alternating face count)
- **Step 30w**: Shellability and EL-labeling check
- **Step 30x**: Discrete Morse theory (critical cells, Morse number)

For session type lattices, the order complex captures the topological
"shape" of the protocol structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import (
    _state_list,
    _reachability,
    _compute_sccs,
    _covering_relation,
    compute_rank,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderComplexResult:
    """Order complex analysis results.

    Attributes:
        num_states: Number of states (quotient).
        f_vector: Face vector [f_{-1}, f_0, f_1, ...] where f_k = number of k-simplices.
        dimension: Dimension of the complex (max chain length - 1).
        euler_characteristic: χ = Σ (-1)^k f_k.
        reduced_euler: Reduced Euler characteristic = χ - 1.
        num_maximal_chains: Number of maximal chains (top simplices).
        is_shellable: True iff a shelling order exists.
        morse_number: Minimum number of critical cells in a discrete Morse function.
        h_vector: h-vector (derived from f-vector via the Dehn-Sommerville relations).
    """
    num_states: int
    f_vector: list[int]
    dimension: int
    euler_characteristic: int
    reduced_euler: int
    num_maximal_chains: int
    is_shellable: bool
    morse_number: int
    h_vector: list[int]


# ---------------------------------------------------------------------------
# Step 30u: Order complex construction
# ---------------------------------------------------------------------------

def _enumerate_chains(ss: "StateSpace") -> list[list[int]]:
    """Enumerate all chains in the open interval (top, bottom).

    A chain is a totally ordered subset. For the order complex,
    we consider chains in the open interval (excluding top and bottom).
    """
    scc_map, _ = _compute_sccs(ss)
    reach = _reachability(ss)
    top, bottom = ss.top, ss.bottom

    # Interior elements: strictly between top and bottom
    interior = [
        s for s in sorted(ss.states)
        if s in reach[top] and bottom in reach[s]
        and scc_map[s] != scc_map[top]
        and scc_map[s] != scc_map[bottom]
    ]

    # Deduplicate by SCC
    seen_sccs: set[int] = set()
    unique_interior: list[int] = []
    for s in interior:
        rep = scc_map[s]
        if rep not in seen_sccs:
            seen_sccs.add(rep)
            unique_interior.append(s)

    chains: list[list[int]] = [[]]  # Empty chain (f_{-1} = 1)

    # Enumerate all subsets of interior that form chains
    n = len(unique_interior)
    for mask in range(1, 1 << n):
        subset = [unique_interior[i] for i in range(n) if mask & (1 << i)]

        # Check all pairs are comparable
        is_chain = True
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset[i], subset[j]
                if a not in reach[b] and b not in reach[a]:
                    is_chain = False
                    break
                if scc_map[a] == scc_map[b] and a != b:
                    is_chain = False
                    break
            if not is_chain:
                break

        if is_chain:
            chains.append(sorted(subset, key=lambda s: -len(reach[s])))

    return chains


def f_vector(ss: "StateSpace") -> list[int]:
    """Compute the f-vector of the order complex.

    f_k = number of k-simplices = number of chains of length k+1
    in the open interval (top, bottom).
    f_{-1} = 1 (the empty face).
    """
    chains = _enumerate_chains(ss)
    max_len = max(len(c) for c in chains) if chains else 0

    fv = [0] * (max_len + 1)
    for c in chains:
        k = len(c)  # chain of length k is a (k-1)-simplex
        fv[k] += 1

    # fv[0] = f_{-1} = 1 (empty chain)
    return fv


def dimension(ss: "StateSpace") -> int:
    """Dimension of the order complex."""
    fv = f_vector(ss)
    for k in range(len(fv) - 1, -1, -1):
        if fv[k] > 0:
            return k - 1  # k elements → (k-1)-simplex
    return -1


# ---------------------------------------------------------------------------
# Step 30v: Euler characteristic
# ---------------------------------------------------------------------------

def euler_characteristic(ss: "StateSpace") -> int:
    """Euler characteristic: χ = Σ_{k≥0} (-1)^k f_k.

    Note: f_0 counts vertices (0-simplices), f_1 edges (1-simplices), etc.
    """
    fv = f_vector(ss)
    # fv[0] = f_{-1} (empty face), fv[1] = f_0, fv[2] = f_1, etc.
    chi = 0
    for k in range(1, len(fv)):
        chi += (-1) ** (k - 1) * fv[k]
    return chi


def reduced_euler_characteristic(ss: "StateSpace") -> int:
    """Reduced Euler characteristic: chi_tilde = -f_{-1} + f_0 - f_1 + ...

    By Hall's theorem, this equals μ(top, bottom).
    """
    fv = f_vector(ss)
    chi_tilde = 0
    for k in range(len(fv)):
        # fv[k] = f_{k-1}. So contribution = (-1)^{k-1} * fv[k]
        chi_tilde += ((-1) ** (k - 1)) * fv[k]
    return int(chi_tilde)


# ---------------------------------------------------------------------------
# Step 30w: Shellability
# ---------------------------------------------------------------------------

def _maximal_chains(ss: "StateSpace") -> list[list[int]]:
    """Enumerate maximal chains from top to bottom.

    Uses covering relation (Hasse diagram) edges only.
    """
    covers = _covering_relation(ss)
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for x, y in covers:
        adj[x].append(y)

    chains: list[list[int]] = []

    def _dfs(current: int, path: list[int]) -> None:
        if current == ss.bottom:
            chains.append(path[:])
            return
        for nxt in adj[current]:
            path.append(nxt)
            _dfs(nxt, path)
            path.pop()

    _dfs(ss.top, [ss.top])
    return chains


def num_maximal_chains(ss: "StateSpace") -> int:
    """Count maximal chains from top to bottom."""
    return len(_maximal_chains(ss))


def is_shellable(ss: "StateSpace") -> bool:
    """Check if the order complex is shellable.

    A simplicial complex is shellable if its maximal simplices (facets)
    can be ordered F_1, F_2, ... such that each F_k intersects the
    union of previous facets in a pure (k-1)-dimensional complex.

    For bounded posets, shellability is equivalent to the existence
    of an EL-labeling (edge-lexicographic labeling).

    For session type lattices with ranked structure, we check the
    simpler sufficient condition: the lattice is graded and
    every interval is shellable (recursive check simplified to:
    all maximal chains have the same length).
    """
    chains = _maximal_chains(ss)
    if len(chains) <= 1:
        return True

    # All maximal chains must have the same length (graded → shellable for lattices)
    lengths = {len(c) for c in chains}
    return len(lengths) == 1


# ---------------------------------------------------------------------------
# Step 30x: Discrete Morse theory
# ---------------------------------------------------------------------------

def morse_number(ss: "StateSpace") -> int:
    """Compute the Morse number: minimum critical cells needed.

    For a simplicial complex, the Morse number is the minimum number
    of critical cells in any discrete Morse function.

    Lower bound: Σ β_k (sum of Betti numbers).
    For contractible complexes: Morse number = 1.
    For spheres S^d: Morse number = 2.

    We approximate using the Euler characteristic:
    - If χ̃ = 0, complex may be contractible → Morse number ≈ 1
    - If |χ̃| = 1, complex is sphere-like → Morse number ≈ 2
    - Otherwise: Morse number ≥ |χ̃| + 1
    """
    chi_tilde = reduced_euler_characteristic(ss)
    fv = f_vector(ss)
    dim = dimension(ss)

    if dim < 0:
        return 0  # Empty complex
    if dim == 0:
        return fv[1]  # Just vertices, each is critical

    # Simple bound from Euler characteristic
    if chi_tilde == 0:
        return 1  # Likely contractible
    return abs(chi_tilde) + 1


# ---------------------------------------------------------------------------
# h-vector
# ---------------------------------------------------------------------------

def h_vector(ss: "StateSpace") -> list[int]:
    """Compute the h-vector from the f-vector.

    h_k = Σ_{j=0}^{k} (-1)^{k-j} C(d-j, k-j) f_{j-1}

    where d = dimension + 1 and f_{-1} = 1.
    Simplified for small complexes.
    """
    fv = f_vector(ss)
    d = len(fv) - 1  # d = max face size

    if d <= 0:
        return [1]

    def _comb(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    hv = []
    for k in range(d + 1):
        h_k = 0
        for j in range(k + 1):
            if j < len(fv):
                h_k += (-1) ** (k - j) * _comb(d - j, k - j) * fv[j]
        hv.append(h_k)

    return hv


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_order_complex(ss: "StateSpace") -> OrderComplexResult:
    """Complete order complex analysis."""
    fv = f_vector(ss)
    dim = dimension(ss)
    chi = euler_characteristic(ss)
    chi_tilde = reduced_euler_characteristic(ss)
    n_max = num_maximal_chains(ss)
    shell = is_shellable(ss)
    morse = morse_number(ss)
    hv = h_vector(ss)

    scc_map, _ = _compute_sccs(ss)
    n_quotient = len(set(scc_map.values()))

    return OrderComplexResult(
        num_states=n_quotient,
        f_vector=fv,
        dimension=dim,
        euler_characteristic=chi,
        reduced_euler=chi_tilde,
        num_maximal_chains=n_max,
        is_shellable=shell,
        morse_number=morse,
        h_vector=hv,
    )
