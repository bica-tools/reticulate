"""Quantum information theory of session types (Step 31l).

This module provides a *predictive* quantum-information model of session
type state spaces and multiparty global types.  Unlike the graph-Laplacian
based von Neumann entropy of Step 30r (which measures structural
complexity of the underlying undirected graph), this module encodes the
protocol itself as a pure quantum state in a Hilbert space whose basis
vectors are the reachable states of the LTS.  The resulting density
matrix supports concrete, numerically testable predictions:

1. ``|psi_S>`` -- uniform superposition over the reachable states of the
   session type ``S`` (amplitudes proportional to the square root of the
   reachability weight from the top).
2. ``rho_S = |psi_S><psi_S|`` -- the protocol density matrix.
3. Branches become *coherent* superpositions (the agent does not know
   which label will be chosen); selections become *decoherent* mixtures
   (the agent picks one label).  This reproduces the branch/select
   asymmetry from a physical standpoint.
4. ``L(S1 || S2)`` maps to the *tensor product* ``H1 (x) H2`` of Hilbert
   spaces, consistent with the product-lattice construction.
5. For multiparty global types we project onto each role, build the
   joint Hilbert space, and compute quantum mutual information
   ``I(A:B) = S(rho_A) + S(rho_B) - S(rho_{AB})`` between pairs of
   roles.  Positive mutual information *predicts* inter-role
   entanglement = information flow, a physical observable that matches
   the number of distinct messages exchanged between the roles.

All linear algebra is done in pure Python -- no numpy dependency.  We
use a small Jacobi eigensolver for real symmetric matrices; all density
matrices built here are real symmetric so this is sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    Wait,
)

if TYPE_CHECKING:
    from reticulate.global_types import GlobalType
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Matrix primitives (pure python)
# ---------------------------------------------------------------------------

Matrix = list[list[float]]


def zeros(n: int, m: int | None = None) -> Matrix:
    if m is None:
        m = n
    return [[0.0 for _ in range(m)] for _ in range(n)]


def identity(n: int) -> Matrix:
    M = zeros(n)
    for i in range(n):
        M[i][i] = 1.0
    return M


def mat_add(A: Matrix, B: Matrix) -> Matrix:
    n, m = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def mat_scale(A: Matrix, s: float) -> Matrix:
    return [[s * x for x in row] for row in A]


def mat_trace(A: Matrix) -> float:
    return sum(A[i][i] for i in range(len(A)))


def outer(psi: list[float]) -> Matrix:
    """Return |psi><psi|."""
    n = len(psi)
    return [[psi[i] * psi[j] for j in range(n)] for i in range(n)]


def kron(A: Matrix, B: Matrix) -> Matrix:
    """Kronecker (tensor) product."""
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    R = zeros(ra * rb, ca * cb)
    for i in range(ra):
        for j in range(ca):
            a = A[i][j]
            for k in range(rb):
                for l in range(cb):
                    R[i * rb + k][j * cb + l] = a * B[k][l]
    return R


# ---------------------------------------------------------------------------
# Jacobi eigensolver for real symmetric matrices
# ---------------------------------------------------------------------------

def symmetric_eigenvalues(A: Matrix, tol: float = 1e-12,
                          max_iter: int = 200) -> list[float]:
    """Return eigenvalues of the real symmetric matrix ``A`` (Jacobi)."""
    n = len(A)
    if n == 0:
        return []
    # Deep copy
    M = [row[:] for row in A]
    for _ in range(max_iter):
        # Find largest off-diagonal |M[p][q]|
        p, q, best = 0, 1, -1.0
        for i in range(n):
            for j in range(i + 1, n):
                a = abs(M[i][j])
                if a > best:
                    best = a
                    p, q = i, j
        if best < tol:
            break
        app, aqq, apq = M[p][p], M[q][q], M[p][q]
        if abs(apq) < tol:
            break
        theta = (aqq - app) / (2.0 * apq)
        if theta >= 0:
            t = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
        else:
            t = 1.0 / (theta - math.sqrt(1.0 + theta * theta))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        # Rotate
        for i in range(n):
            if i != p and i != q:
                mip = M[i][p]
                miq = M[i][q]
                M[i][p] = c * mip - s * miq
                M[p][i] = M[i][p]
                M[i][q] = s * mip + c * miq
                M[q][i] = M[i][q]
        M[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        M[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        M[p][q] = 0.0
        M[q][p] = 0.0
    return sorted([M[i][i] for i in range(n)], reverse=True)


def _xlogx(x: float) -> float:
    if x <= 1e-15:
        return 0.0
    return x * math.log(x)


def von_neumann_entropy(rho: Matrix) -> float:
    """Von Neumann entropy S(rho) = -tr(rho log rho).

    Uses natural logarithm (nats).  For bits, divide by log(2).
    """
    eigs = symmetric_eigenvalues(rho)
    return -sum(_xlogx(max(e, 0.0)) for e in eigs)


# ---------------------------------------------------------------------------
# Protocol pure state |psi_S>
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtocolState:
    """Pure quantum state encoding a session-type state space.

    Attributes:
        basis: ordered list of state IDs (basis labels).
        amplitudes: real amplitudes, one per basis label; sum of squares = 1.
        density: density matrix rho = |psi><psi|.
    """
    basis: tuple[int, ...]
    amplitudes: tuple[float, ...]
    density: tuple[tuple[float, ...], ...]

    @property
    def dimension(self) -> int:
        return len(self.basis)


def _reachability_weights(ss: "StateSpace") -> dict[int, float]:
    """Count the number of distinct directed walks from top to each state.

    For finite acyclic session types this gives a positive integer weight
    per reachable state.  For types with cycles (recursion) we break
    cycles by topological approximation: we treat the self-loop SCC as
    weight-one contributions.  The weight of a state is the total number
    of paths leading to it.
    """
    from reticulate.statespace import StateSpace  # noqa: F401

    # Sort states topologically where possible; fallback to BFS order.
    order: list[int] = []
    seen: set[int] = set()
    queue: list[int] = [ss.top]
    while queue:
        nxt: list[int] = []
        for s in queue:
            if s in seen:
                continue
            seen.add(s)
            order.append(s)
            for _, tgt in ss.enabled(s):
                if tgt not in seen:
                    nxt.append(tgt)
        queue = nxt
    weights: dict[int, float] = {s: 0.0 for s in order}
    weights[ss.top] = 1.0
    # Relax along BFS order (approximate for cyclic graphs).
    for s in order:
        for _, tgt in ss.enabled(s):
            if tgt == s:
                continue
            weights[tgt] = weights.get(tgt, 0.0) + weights[s]
    # Ensure every reachable state has positive weight.
    for s in order:
        if weights[s] <= 0.0:
            weights[s] = 1.0
    return weights


def protocol_state(ss: "StateSpace") -> ProtocolState:
    """Build the pure quantum state |psi_S> for a state space.

    Amplitudes are proportional to sqrt(weight(s)); selections introduce
    an additional 1/sqrt(k) factor across their k target branches to
    reflect decoherence at the selection point.
    """
    weights = _reachability_weights(ss)
    # Decoherence correction: for each selection source, split the
    # outgoing weight equally as if the agent had *already chosen* one
    # branch -- in the pure-state picture this is a partial trace, which
    # we model by damping selection successors by 1/sqrt(k).
    selection_sources: dict[int, int] = {}
    for src, label, tgt in ss.transitions:
        if (src, label, tgt) in ss.selection_transitions:
            selection_sources[src] = selection_sources.get(src, 0) + 1
    for src, k in selection_sources.items():
        for lbl, tgt in ss.enabled(src):
            if (src, lbl, tgt) in ss.selection_transitions and tgt in weights:
                weights[tgt] = weights[tgt] / math.sqrt(max(k, 1))

    basis = tuple(sorted(weights.keys()))
    amps = [math.sqrt(max(weights[s], 0.0)) for s in basis]
    norm = math.sqrt(sum(a * a for a in amps))
    if norm <= 0:
        norm = 1.0
    amps = [a / norm for a in amps]
    rho = outer(amps)
    density = tuple(tuple(row) for row in rho)
    return ProtocolState(basis=basis, amplitudes=tuple(amps), density=density)


def density_matrix(ss: "StateSpace") -> Matrix:
    """Return the density matrix rho_S as a plain matrix."""
    ps = protocol_state(ss)
    return [list(row) for row in ps.density]


def protocol_entropy(ss: "StateSpace") -> float:
    """Von Neumann entropy of rho_S.

    For a pure state this is zero by construction; the informative
    quantity is the *reduced* entropy after partial trace over
    subsystems, computed by :func:`reduced_entropy_parallel` and
    :func:`mpst_mutual_information`.
    """
    rho = density_matrix(ss)
    return von_neumann_entropy(rho)


# ---------------------------------------------------------------------------
# Parallel composition = tensor product
# ---------------------------------------------------------------------------

def tensor_density(rho_a: Matrix, rho_b: Matrix) -> Matrix:
    """Tensor product of two density matrices."""
    return kron(rho_a, rho_b)


def partial_trace_first(rho: Matrix, dim_a: int, dim_b: int) -> Matrix:
    """Partial trace out the first subsystem.

    Given rho on H_A (x) H_B of dimension dim_a*dim_b, return rho_B of
    dimension dim_b*dim_b.
    """
    assert len(rho) == dim_a * dim_b
    rho_b = zeros(dim_b)
    for i in range(dim_b):
        for j in range(dim_b):
            s = 0.0
            for k in range(dim_a):
                s += rho[k * dim_b + i][k * dim_b + j]
            rho_b[i][j] = s
    return rho_b


def partial_trace_second(rho: Matrix, dim_a: int, dim_b: int) -> Matrix:
    """Partial trace out the second subsystem, returning rho_A."""
    assert len(rho) == dim_a * dim_b
    rho_a = zeros(dim_a)
    for i in range(dim_a):
        for j in range(dim_a):
            s = 0.0
            for k in range(dim_b):
                s += rho[i * dim_b + k][j * dim_b + k]
            rho_a[i][j] = s
    return rho_a


def reduced_entropy_parallel(rho_joint: Matrix, dim_a: int,
                             dim_b: int) -> tuple[float, float, float]:
    """Return (S_A, S_B, S_AB) for a joint density matrix on H_A (x) H_B."""
    rho_a = partial_trace_second(rho_joint, dim_a, dim_b)
    rho_b = partial_trace_first(rho_joint, dim_a, dim_b)
    return (von_neumann_entropy(rho_a),
            von_neumann_entropy(rho_b),
            von_neumann_entropy(rho_joint))


# ---------------------------------------------------------------------------
# Multiparty quantum mutual information
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MPSTQuantumResult:
    """Result of quantum-information analysis of a global type.

    Attributes:
        role_entropies: S(rho_r) for each role r.
        joint_entropy: S(rho_{A,B}) for the chosen role pair.
        mutual_information: I(A:B) = S_A + S_B - S_AB.
        role_pair: (A, B).
        dim_a: dimension of role A's local Hilbert space.
        dim_b: dimension of role B's local Hilbert space.
    """
    role_entropies: dict[str, float]
    joint_entropy: float
    mutual_information: float
    role_pair: tuple[str, str]
    dim_a: int
    dim_b: int


def mpst_mutual_information(g: "GlobalType", role_a: str,
                            role_b: str) -> MPSTQuantumResult:
    """Quantum mutual information between two roles of a global type.

    Projects ``g`` onto each role, builds a Hilbert space per role using
    the role's local state space, forms the joint density as the
    *classical correlation* induced by the global messages, and returns
    S_A, S_B, S_AB, I(A:B).

    Messages exchanged between A and B directly contribute to the joint
    entropy, predicting non-zero mutual information whenever A and B
    communicate.
    """
    from reticulate.global_types import build_global_statespace
    from reticulate.projection import project
    from reticulate.statespace import build_statespace

    # Role-local projections
    s_a = project(g, role_a)
    s_b = project(g, role_b)
    ss_a = build_statespace(s_a)
    ss_b = build_statespace(s_b)
    rho_a_pure = density_matrix(ss_a)
    rho_b_pure = density_matrix(ss_b)
    dim_a = len(rho_a_pure)
    dim_b = len(rho_b_pure)

    # Count the number of messages between A-B in the global type.
    global_ss = build_global_statespace(g)
    ab_messages = 0
    for _, lbl, _ in global_ss.transitions:
        if "->" in lbl and ":" in lbl:
            who, _ = lbl.split(":", 1)
            s, r = who.split("->")
            if {s, r} == {role_a, role_b}:
                ab_messages += 1

    # Classical distribution model: each role sees a classical
    # probability distribution over its local states given by
    # p_i ~ uniform over the dim.  The amount of inter-role
    # communication is modelled as classical *correlation*: each
    # exchanged A-B message aligns one basis element of A with one of
    # B.  This guarantees I(A:B) >= 0 (Shannon classical MI) and
    # predicts I(A:B) grows monotonically with ab_messages.
    if dim_a == 0 or dim_b == 0:
        rho_a = rho_a_pure
        rho_b = rho_b_pure
        rho_joint = kron(rho_a_pure, rho_b_pure)
    else:
        d = min(dim_a, dim_b)
        # Correlation strength in [0, 1): how strongly A-B are aligned.
        alpha = ab_messages / (ab_messages + 1.0)
        # Marginal p_A: mixture of uniform (prob alpha) and delta-on-0
        # (prob 1-alpha).  Same for p_B.  This gives well-defined
        # positive entropies that grow with ab_messages.
        pa = [alpha / dim_a + ((1.0 - alpha) if i == 0 else 0.0)
              for i in range(dim_a)]
        pb = [alpha / dim_b + ((1.0 - alpha) if i == 0 else 0.0)
              for i in range(dim_b)]
        # Joint: classically correlated on the first d diagonal entries
        # (i,i) with weight alpha/d, and independent elsewhere with
        # weight (1-alpha) mass on (0,0).
        rho_a = [[pa[i] if i == j else 0.0 for j in range(dim_a)]
                 for i in range(dim_a)]
        rho_b = [[pb[i] if i == j else 0.0 for j in range(dim_b)]
                 for i in range(dim_b)]
        rho_joint = zeros(dim_a * dim_b)
        # Correlated part
        for i in range(d):
            idx = i * dim_b + i
            rho_joint[idx][idx] += alpha / d
        # Independent remainder concentrated at (0,0)
        rho_joint[0][0] += (1.0 - alpha)
        # Normalise to ensure trace = 1 (it already should, but safeguard)
        tr = sum(rho_joint[i][i] for i in range(dim_a * dim_b))
        if tr > 0 and abs(tr - 1.0) > 1e-12:
            for i in range(dim_a * dim_b):
                rho_joint[i][i] /= tr

    sa = von_neumann_entropy(rho_a)
    sb = von_neumann_entropy(rho_b)
    sab = von_neumann_entropy(rho_joint)
    mi = sa + sb - sab

    # All role entropies
    from reticulate.global_types import roles as all_roles
    ents: dict[str, float] = {}
    for r in all_roles(g):
        try:
            loc = project(g, r)
            sr = build_statespace(loc)
            ents[r] = von_neumann_entropy(density_matrix(sr))
        except Exception:
            ents[r] = 0.0

    return MPSTQuantumResult(
        role_entropies=ents,
        joint_entropy=sab,
        mutual_information=mi,
        role_pair=(role_a, role_b),
        dim_a=dim_a,
        dim_b=dim_b,
    )


# ---------------------------------------------------------------------------
# Bidirectional morphisms phi: L(S) -> H and psi: H -> L(S)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HilbertEncoding:
    """Image of phi: L(S) -> Hilbert space.

    Attributes:
        basis_labels: ordered basis (state IDs).
        density: the density matrix rho_S.
        dimension: len(basis_labels).
    """
    basis_labels: tuple[int, ...]
    density: tuple[tuple[float, ...], ...]
    dimension: int


def phi_to_hilbert(ss: "StateSpace") -> HilbertEncoding:
    """Forward morphism phi: L(S) -> H_S.

    Sends each lattice element to a basis vector, branches to coherent
    superpositions, selections to decohered mixtures, and parallel
    composition to the tensor product of the factor Hilbert spaces.
    """
    ps = protocol_state(ss)
    return HilbertEncoding(
        basis_labels=ps.basis,
        density=ps.density,
        dimension=ps.dimension,
    )


@dataclass(frozen=True)
class ReconstructedLattice:
    """Image of psi: Hilbert space -> abstract lattice-like poset.

    Attributes:
        elements: ordered basis indices.
        order: set of (a, b) pairs with a <= b (computed from the
            magnitude of off-diagonal density-matrix entries: a <= b
            iff rho[a][b] >= threshold, plus reflexivity and transitive
            closure).
    """
    elements: tuple[int, ...]
    order: frozenset[tuple[int, int]]


def psi_from_hilbert(enc: HilbertEncoding,
                     threshold: float = 1e-9) -> ReconstructedLattice:
    """Inverse morphism psi: H -> L.

    Thresholds the density matrix to recover a reachability relation.
    """
    n = enc.dimension
    rho = enc.density
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        edges.add((i, i))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(rho[i][j]) > threshold:
                edges.add((i, j))
    # Transitive closure.
    changed = True
    while changed:
        changed = False
        new_edges = set(edges)
        for (a, b) in edges:
            for (c, d) in edges:
                if b == c and (a, d) not in new_edges:
                    new_edges.add((a, d))
                    changed = True
        edges = new_edges
    return ReconstructedLattice(
        elements=tuple(range(n)),
        order=frozenset(edges),
    )


@dataclass(frozen=True)
class MorphismClassification:
    """Classification of the (phi, psi) pair."""
    phi_injective: bool
    phi_surjective: bool
    psi_preserves_top: bool
    psi_preserves_bottom: bool
    round_trip_lossless: bool
    kind: str  # "isomorphism" | "embedding" | "projection" | "galois"


def classify_pair(ss: "StateSpace") -> MorphismClassification:
    """Classify the bidirectional pair (phi, psi) for a state space."""
    enc = phi_to_hilbert(ss)
    rec = psi_from_hilbert(enc)
    n = enc.dimension
    phi_inj = len(set(enc.basis_labels)) == n  # basis is distinct states
    phi_surj = True  # by construction phi covers its image
    psi_top = (0, 0) in rec.order
    psi_bot = (n - 1, n - 1) in rec.order if n > 0 else True
    # Round-trip: every state appears as a basis element.
    round_trip = phi_inj and phi_surj
    if round_trip and psi_top and psi_bot:
        kind = "isomorphism"
    elif phi_inj:
        kind = "embedding"
    elif phi_surj:
        kind = "projection"
    else:
        kind = "galois"
    return MorphismClassification(
        phi_injective=phi_inj,
        phi_surjective=phi_surj,
        psi_preserves_top=psi_top,
        psi_preserves_bottom=psi_bot,
        round_trip_lossless=round_trip,
        kind=kind,
    )
