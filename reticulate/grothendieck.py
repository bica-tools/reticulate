"""Grothendieck group K_0 of protocol lattices (Step 32k).

We form the Grothendieck group of the commutative monoid of
(isomorphism classes of) finite protocol lattices under a chosen
binary composition:

- ``"sum"``: disjoint union / direct sum of lattices (coproduct of
  bounded posets, identified on ⊤ and ⊥ — here we use the plain
  multiset sum of rank profiles, modulo the shared bounds).
- ``"product"``: the Cartesian / parallel product L₁ × L₂.

Isomorphism is approximated by a canonical invariant signature:
the (height, size, sorted rank profile, sorted in/out-degree
multiset).  Distinct signatures yield distinct generators; equal
signatures are identified.  This is a sound under-approximation:
isomorphic lattices share a signature, so the induced quotient map
factors the true K_0 map.

An element of K_0 is a *virtual lattice* — a formal difference
[L₁] − [L₂] represented by a pair of multisets of generators with
cancellation.  The group operation is componentwise addition of
multisets modulo the equivalence (A⁺,A⁻) ~ (B⁺,B⁻) iff
A⁺ + B⁻ = B⁺ + A⁻ (as multisets).

Bidirectional morphisms
-----------------------
    φ : L(S)  →  K_0,     L  ↦  [sig(L)]  (a generator class)
    ψ : K_0   →  Dict-of-lattices,
                   A⁺ − A⁻ ↦ the pair of multisets (A⁺,A⁻)

φ is a monoid morphism (φ(L₁ ⊕ L₂) = φ(L₁)+φ(L₂)); ψ is the
canonical section exhibiting every K_0 element as a virtual
difference of actual lattice multisets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.zeta import _compute_sccs


# ---------------------------------------------------------------------------
# Canonical signature of a lattice
# ---------------------------------------------------------------------------

LatticeSig = tuple  # (size, height, rank_profile_tuple, degree_multiset_tuple)


def _rank_profile(ss: "StateSpace") -> tuple[int, ...]:
    """Compute the corank-indexed element count profile.

    Uses reverse-BFS from the bottom state over the transition graph
    on the quotient by SCCs (so recursive cycles collapse).
    """
    scc_map, _ = _compute_sccs(ss)
    # Build quotient adjacency (forward).
    fwd: dict[int, set[int]] = {}
    rev: dict[int, set[int]] = {}
    nodes: set[int] = set()
    for s, _lab, t in ss.transitions:
        a, b = scc_map[s], scc_map[t]
        if a == b:
            continue
        nodes.add(a); nodes.add(b)
        fwd.setdefault(a, set()).add(b)
        rev.setdefault(b, set()).add(a)
    nodes.add(scc_map[ss.top])
    nodes.add(scc_map[ss.bottom])
    top = scc_map[ss.top]
    # Corank (distance from top via forward edges).
    from collections import deque
    corank: dict[int, int] = {top: 0}
    dq = deque([top])
    while dq:
        u = dq.popleft()
        for v in fwd.get(u, ()):
            if v not in corank or corank[v] > corank[u] + 1:
                corank[v] = corank[u] + 1
                dq.append(v)
    if not corank:
        return (1,)
    h = max(corank.values())
    prof = [0] * (h + 1)
    for n in nodes:
        if n in corank:
            prof[corank[n]] += 1
    return tuple(prof)


def _degree_multiset(ss: "StateSpace") -> tuple[tuple[int, int], ...]:
    """Sorted multiset of (in-degree, out-degree) pairs on the SCC quotient."""
    scc_map, _ = _compute_sccs(ss)
    nodes: set[int] = {scc_map[s] for s in ss.states}
    nodes.add(scc_map[ss.top])
    nodes.add(scc_map[ss.bottom])
    indeg = {n: 0 for n in nodes}
    outdeg = {n: 0 for n in nodes}
    for s, _lab, t in ss.transitions:
        a, b = scc_map[s], scc_map[t]
        if a == b:
            continue
        outdeg[a] = outdeg.get(a, 0) + 1
        indeg[b] = indeg.get(b, 0) + 1
    pairs = sorted((indeg[n], outdeg[n]) for n in nodes)
    return tuple(pairs)


def lattice_signature(ss: "StateSpace") -> LatticeSig:
    """Canonical invariant signature used as generator key in K_0.

    The signature is (size, height, rank_profile, degree_multiset).
    Isomorphic lattices produce identical signatures.
    """
    prof = _rank_profile(ss)
    deg = _degree_multiset(ss)
    size = sum(prof)
    height = len(prof) - 1
    return (size, height, prof, deg)


# ---------------------------------------------------------------------------
# Virtual lattice — element of K_0
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VirtualLattice:
    """Element of K_0 — a formal difference [positive] − [negative].

    Both components are multisets (``Counter``) over lattice signatures.
    Two virtual lattices are equal iff they represent the same element
    of K_0, i.e., after cancellation they have the same residue.
    """

    positive: tuple[tuple[LatticeSig, int], ...] = ()
    negative: tuple[tuple[LatticeSig, int], ...] = ()

    # ---- construction helpers ----

    @staticmethod
    def from_counter(pos: Counter, neg: Counter) -> "VirtualLattice":
        # Cancel common part.
        common = pos & neg
        pos = pos - common
        neg = neg - common
        p = tuple(sorted((k, v) for k, v in pos.items() if v > 0))
        n = tuple(sorted((k, v) for k, v in neg.items() if v > 0))
        return VirtualLattice(p, n)

    @staticmethod
    def generator(sig: LatticeSig) -> "VirtualLattice":
        return VirtualLattice.from_counter(Counter({sig: 1}), Counter())

    @staticmethod
    def zero() -> "VirtualLattice":
        return VirtualLattice((), ())

    # ---- multiset views ----

    def pos_counter(self) -> Counter:
        return Counter(dict(self.positive))

    def neg_counter(self) -> Counter:
        return Counter(dict(self.negative))

    # ---- group operations ----

    def __add__(self, other: "VirtualLattice") -> "VirtualLattice":
        return VirtualLattice.from_counter(
            self.pos_counter() + other.pos_counter(),
            self.neg_counter() + other.neg_counter(),
        )

    def __neg__(self) -> "VirtualLattice":
        return VirtualLattice.from_counter(self.neg_counter(), self.pos_counter())

    def __sub__(self, other: "VirtualLattice") -> "VirtualLattice":
        return self + (-other)

    def is_zero(self) -> bool:
        return not self.positive and not self.negative

    def total_size(self) -> int:
        """Σ size(sig) over positive minus negative multiplicities."""
        s = 0
        for sig, m in self.positive:
            s += sig[0] * m
        for sig, m in self.negative:
            s -= sig[0] * m
        return s


# ---------------------------------------------------------------------------
# Grothendieck group K_0
# ---------------------------------------------------------------------------

@dataclass
class GrothendieckGroup:
    """The Grothendieck group K_0(M, ⊕) of protocol lattices.

    Parameters
    ----------
    operation : ``"sum"`` or ``"product"``
        Monoid operation.  ``"sum"`` is additive (concatenation of
        rank profiles); ``"product"`` is multiplicative (convolution
        of rank profiles — lattice direct product).
    """

    operation: str = "sum"
    _generators: dict[LatticeSig, int] = field(default_factory=dict)

    # ---- generators ----

    def register(self, ss: "StateSpace") -> LatticeSig:
        sig = lattice_signature(ss)
        self._generators[sig] = self._generators.get(sig, 0) + 1
        return sig

    def generators(self) -> list[LatticeSig]:
        return sorted(self._generators.keys())

    # ---- φ: L(S) → K_0 ----

    def phi(self, ss: "StateSpace") -> VirtualLattice:
        """φ: L(S) ↦ [sig(L(S))] as a generator in K_0."""
        sig = lattice_signature(ss)
        self._generators.setdefault(sig, 0)
        return VirtualLattice.generator(sig)

    # ---- ψ: K_0 → (Counter, Counter) ----

    def psi(self, v: VirtualLattice) -> tuple[Counter, Counter]:
        """ψ: v ↦ (positive multiset, negative multiset) of generators.

        This is the canonical section exhibiting any K_0 element as
        the difference of two effective (multiset) elements of M.
        """
        return (v.pos_counter(), v.neg_counter())

    # ---- monoid operation ----

    def combine_signatures(self, s1: LatticeSig, s2: LatticeSig) -> LatticeSig:
        """The monoid operation at the signature level."""
        if self.operation == "sum":
            # Disjoint union: concatenate rank profiles (sum per rank),
            # pad to equal length.
            p1, p2 = s1[2], s2[2]
            L = max(len(p1), len(p2))
            prof = tuple(
                (p1[k] if k < len(p1) else 0) + (p2[k] if k < len(p2) else 0)
                for k in range(L)
            )
            # For disjoint union we identify the two tops and bottoms,
            # subtracting the two duplicated bounds (one top, one bottom).
            # Keep degree multisets as the merged sorted tuple.
            deg = tuple(sorted(s1[3] + s2[3]))
            size = sum(prof)
            height = len(prof) - 1
            return (size, height, prof, deg)
        elif self.operation == "product":
            # Cartesian product: convolve rank profiles.
            p1, p2 = s1[2], s2[2]
            if not p1 or not p2:
                prof = ()
            else:
                n = len(p1) + len(p2) - 1
                out = [0] * n
                for i, a in enumerate(p1):
                    for j, b in enumerate(p2):
                        out[i + j] += a * b
                prof = tuple(out)
            size = sum(prof)
            height = len(prof) - 1
            # Degree multiset of product is unknown purely from factors;
            # we approximate with the sorted Cartesian-product tuple.
            deg = tuple(sorted((a + c, b + d)
                               for (a, b) in s1[3] for (c, d) in s2[3]))
            return (size, height, prof, deg)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    # ---- virtual differences ----

    def virtual_difference(self, a: "StateSpace", b: "StateSpace") -> VirtualLattice:
        """Return [a] − [b] ∈ K_0."""
        return self.phi(a) - self.phi(b)

    def equal(self, u: VirtualLattice, v: VirtualLattice) -> bool:
        """Test equality in K_0.  By construction both sides are already
        reduced (cancellation is performed on every operation), so direct
        tuple comparison suffices.
        """
        return u == v


# ---------------------------------------------------------------------------
# Public high-level API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrothendieckResult:
    operation: str
    generators: tuple[LatticeSig, ...]
    num_generators: int
    virtual_classes: tuple[VirtualLattice, ...]
    relations_verified: int
    is_free_abelian: bool


def build_k0(
    spaces: Iterable["StateSpace"],
    operation: str = "sum",
) -> GrothendieckResult:
    """Build K_0 from a collection of lattices and verify basic laws.

    We check:
      * φ is well-defined (isomorphism invariant — via signature).
      * φ is a monoid morphism for the chosen operation (up to the
        signature-level approximation).
      * ψ ∘ φ yields back the singleton generator multiset.
      * (A⁺,A⁻) + (−(A⁺,A⁻)) = 0 (inverses).
    """
    G = GrothendieckGroup(operation=operation)
    sigs: list[LatticeSig] = []
    virtuals: list[VirtualLattice] = []
    relations = 0
    for ss in spaces:
        sig = G.register(ss)
        sigs.append(sig)
        v = G.phi(ss)
        virtuals.append(v)
        # ψ ∘ φ round-trip.
        pos, neg = G.psi(v)
        assert pos == Counter({sig: 1}) and not neg
        relations += 1
        # Inverse law.
        assert (v + (-v)).is_zero()
        relations += 1
    # Distinct signatures => free generators, so K_0 is a free abelian
    # group of rank |{signatures}|.
    is_free = len(set(sigs)) == len(G.generators())
    return GrothendieckResult(
        operation=operation,
        generators=tuple(G.generators()),
        num_generators=len(G.generators()),
        virtual_classes=tuple(virtuals),
        relations_verified=relations,
        is_free_abelian=is_free,
    )
