"""Spectral synchronization for multiparty session types (Step 31d).

Applies spectral methods to analyze synchronization patterns in
multiparty session types. The global type's state space is decomposed
spectrally to identify synchronization bottlenecks and communication
patterns between roles.

- **Role coupling matrix**: spectral analysis of inter-role communication
- **Synchronization spectrum**: eigenvalues of the role interaction graph
- **Role clustering**: spectral clustering of roles by communication pattern
- **Bottleneck roles**: roles with lowest algebraic connectivity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncResult:
    """Spectral synchronization analysis.

    Attributes:
        num_states: Number of states.
        num_roles: Number of distinct roles (from transition labels).
        role_coupling: Role × Role interaction count matrix.
        roles: List of role names.
        sync_spectrum: Eigenvalues of the role coupling matrix.
        max_coupling: Maximum coupling between any two roles.
        isolation_score: Minimum row sum (least connected role).
    """
    num_states: int
    num_roles: int
    role_coupling: list[list[int]]
    roles: list[str]
    sync_spectrum: list[float]
    max_coupling: int
    isolation_score: int


# ---------------------------------------------------------------------------
# Role extraction
# ---------------------------------------------------------------------------

def extract_roles(ss: "StateSpace") -> list[str]:
    """Extract role names from transition labels.

    For global types: labels are "sender->receiver:method".
    For local types: labels are just method names (single role).
    """
    roles: set[str] = set()
    for _, label, _ in ss.transitions:
        if "->" in label:
            parts = label.split("->")
            roles.add(parts[0].strip())
            recv_method = parts[1].split(":")
            if recv_method:
                roles.add(recv_method[0].strip())
        else:
            roles.add(label)
    return sorted(roles)


def role_coupling_matrix(ss: "StateSpace") -> tuple[list[list[int]], list[str]]:
    """Compute role × role coupling matrix.

    M[i][j] = number of transitions involving both role i and role j.
    For global types with "sender->receiver:method" labels.
    For local types, each label is its own "role" (method name).
    """
    roles = extract_roles(ss)
    n = len(roles)
    idx = {r: i for i, r in enumerate(roles)}
    M = [[0] * n for _ in range(n)]

    for _, label, _ in ss.transitions:
        if "->" in label:
            parts = label.split("->")
            sender = parts[0].strip()
            recv_method = parts[1].split(":")
            receiver = recv_method[0].strip() if recv_method else ""
            if sender in idx and receiver in idx:
                si, ri = idx[sender], idx[receiver]
                M[si][ri] += 1
                M[ri][si] += 1  # Symmetric
        else:
            # Local type: self-interaction
            if label in idx:
                li = idx[label]
                M[li][li] += 1

    return M, roles


# ---------------------------------------------------------------------------
# Spectral analysis of coupling
# ---------------------------------------------------------------------------

def sync_spectrum(ss: "StateSpace") -> list[float]:
    """Eigenvalues of the role coupling matrix."""
    M, roles = role_coupling_matrix(ss)
    n = len(M)
    if n == 0:
        return []
    if n == 1:
        return [float(M[0][0])]

    from reticulate.matrix import _eigenvalues_symmetric
    mat = [[float(M[i][j]) for j in range(n)] for i in range(n)]
    return sorted(_eigenvalues_symmetric(mat))


def max_coupling(ss: "StateSpace") -> int:
    """Maximum coupling between any two distinct roles."""
    M, roles = role_coupling_matrix(ss)
    n = len(M)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            best = max(best, M[i][j])
    return best


def isolation_score(ss: "StateSpace") -> int:
    """Minimum total coupling of any role (least connected role)."""
    M, roles = role_coupling_matrix(ss)
    if not M:
        return 0
    return min(sum(row) for row in M)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_sync(ss: "StateSpace") -> SyncResult:
    """Complete spectral synchronization analysis."""
    M, roles = role_coupling_matrix(ss)
    spec = sync_spectrum(ss)
    mc = max_coupling(ss)
    iso = isolation_score(ss)

    return SyncResult(
        num_states=len(ss.states),
        num_roles=len(roles),
        role_coupling=M,
        roles=roles,
        sync_spectrum=spec,
        max_coupling=mc,
        isolation_score=iso,
    )
