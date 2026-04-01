"""Configuration domains for session type event structures (Step 17).

A **configuration domain** D(ES(S)) is the poset of configurations of a
prime event structure, ordered by subset inclusion.  For session types,
the configuration domain is order-isomorphic to the state space L(S)
(with the ordering reversed: more events = further from top).

Key domain-theoretic properties:
- **Scott domain**: algebraic + bounded-complete dcpo
- **Coherence**: pairwise-consistent sets have an upper bound
- **Algebraicity**: every element is a directed join of compact elements
- **Bounded completeness**: every bounded subset has a supremum
- **dI-domain**: Scott domain + meets distribute over directed joins

Since all session type event structures are finite, the configuration
domain is a finite poset.  In a finite poset every element is compact,
directed completeness is automatic, and bounded completeness reduces to
checking that every pair with an upper bound has a join.

The Scott topology consists of upward-closed sets that are
inaccessible by directed joins.  For finite posets this simplifies to
the collection of all upward-closed (upper) sets.

Functions:
  - ``build_config_domain(ss)``         -- full configuration domain
  - ``check_scott_domain(domain)``      -- verify Scott domain properties
  - ``compact_elements(domain)``        -- identify compact elements
  - ``consistent_pairs(domain)``        -- find consistent configuration pairs
  - ``check_coherence(domain)``         -- verify coherence property
  - ``check_algebraicity(domain)``      -- verify algebraicity
  - ``check_bounded_completeness(domain)`` -- bounded-subset sup check
  - ``scott_open_sets(domain)``         -- enumerate Scott-open sets
  - ``check_distributivity(domain)``    -- meets distribute over directed joins
  - ``covering_chains(domain)``         -- maximal chains (linearizations)
  - ``join_irreducibles(domain)``       -- join-irreducible elements
  - ``analyze_config_domain(ss)``       -- full analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.event_structures import (
    Event,
    EventStructure,
    Configuration,
    ConfigDomain,
    build_event_structure,
    configurations,
    config_domain,
)

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScottDomainResult:
    """Result of Scott domain property checking.

    Attributes:
        is_scott_domain: True iff all Scott domain axioms hold.
        is_algebraic: True iff every element is a join of compacts below it.
        is_bounded_complete: True iff every bounded subset has a sup.
        is_dcpo: True iff every directed subset has a sup.
        is_coherent: True iff pairwise-consistent sets are bounded.
        is_distributive: True iff meets distribute over directed joins.
        is_di_domain: True iff it is a Scott domain with distributivity.
        num_compact: Number of compact elements.
        num_configs: Number of configurations.
        num_consistent_pairs: Number of consistent configuration pairs.
        num_scott_open: Number of Scott-open sets.
    """
    is_scott_domain: bool
    is_algebraic: bool
    is_bounded_complete: bool
    is_dcpo: bool
    is_coherent: bool
    is_distributive: bool
    is_di_domain: bool
    num_compact: int
    num_configs: int
    num_consistent_pairs: int
    num_scott_open: int


@dataclass(frozen=True)
class ConfigDomainAnalysis:
    """Complete configuration domain analysis.

    Attributes:
        domain: The configuration domain.
        es: The event structure.
        scott_result: Scott domain check result.
        num_configs: Number of configurations.
        num_covering_chains: Number of maximal chains.
        num_join_irreducibles: Number of join-irreducible elements.
        max_chain_length: Length of the longest chain.
        width: Maximum antichain size.
        is_lattice: True iff the domain is a lattice.
    """
    domain: ConfigDomain
    es: EventStructure
    scott_result: ScottDomainResult
    num_configs: int
    num_covering_chains: int
    num_join_irreducibles: int
    max_chain_length: int
    width: int
    is_lattice: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_index(domain: ConfigDomain) -> dict[frozenset[Event], int]:
    """Map configuration event sets to their index in domain.configs."""
    return {c.events: i for i, c in enumerate(domain.configs)}


def _upper_bounds(domain: ConfigDomain, indices: set[int]) -> set[int]:
    """Find all upper bounds of a set of configuration indices."""
    if not indices:
        return set(range(len(domain.configs)))
    result: set[int] | None = None
    for idx in indices:
        above = {j for (i, j) in domain.ordering if i == idx}
        if result is None:
            result = above
        else:
            result &= above
    return result if result is not None else set()


def _lower_bounds(domain: ConfigDomain, indices: set[int]) -> set[int]:
    """Find all lower bounds of a set of configuration indices."""
    if not indices:
        return set(range(len(domain.configs)))
    result: set[int] | None = None
    for idx in indices:
        below = {i for (i, j) in domain.ordering if j == idx}
        if result is None:
            result = below
        else:
            result &= below
    return result if result is not None else set()


def _sup(domain: ConfigDomain, indices: set[int]) -> int | None:
    """Find the supremum (least upper bound) of a set of indices, or None."""
    ubs = _upper_bounds(domain, indices)
    if not ubs:
        return None
    # Find the least among upper bounds
    for candidate in ubs:
        if all(
            (candidate, other) in domain.ordering
            or candidate == other
            for other in ubs
        ):
            return candidate
    return None


def _inf(domain: ConfigDomain, indices: set[int]) -> int | None:
    """Find the infimum (greatest lower bound) of a set of indices, or None."""
    lbs = _lower_bounds(domain, indices)
    if not lbs:
        return None
    # Find the greatest among lower bounds
    for candidate in lbs:
        if all(
            (other, candidate) in domain.ordering
            or candidate == other
            for other in lbs
        ):
            return candidate
    return None


def _is_directed(domain: ConfigDomain, indices: set[int]) -> bool:
    """Check if a subset is directed (nonempty, every pair has an upper bound in the set)."""
    if not indices:
        return False
    idx_list = sorted(indices)
    for i in range(len(idx_list)):
        for j in range(i + 1, len(idx_list)):
            a, b = idx_list[i], idx_list[j]
            # Check if there's an upper bound of {a, b} within indices
            found = False
            for k in indices:
                if (a, k) in domain.ordering and (b, k) in domain.ordering:
                    found = True
                    break
            if not found:
                return False
    return True


# ---------------------------------------------------------------------------
# Public API: Build configuration domain
# ---------------------------------------------------------------------------

def build_config_domain(ss: StateSpace) -> ConfigDomain:
    """Build the configuration domain D(ES(S)) from a state space.

    Constructs the event structure, enumerates configurations,
    and orders them by subset inclusion.
    """
    es = build_event_structure(ss)
    return config_domain(es)


# ---------------------------------------------------------------------------
# Public API: Compact elements
# ---------------------------------------------------------------------------

def compact_elements(domain: ConfigDomain) -> list[int]:
    """Identify compact (finite) elements of the configuration domain.

    In a finite poset, EVERY element is compact: for any directed set D
    with c <= sup(D), there exists d in D with c <= d.  This is because
    directed sets in finite posets always contain their own supremum.

    Returns a list of indices into domain.configs.
    """
    # In a finite poset, all elements are compact.
    return list(range(len(domain.configs)))


# ---------------------------------------------------------------------------
# Public API: Consistent pairs
# ---------------------------------------------------------------------------

def consistent_pairs(domain: ConfigDomain) -> list[tuple[int, int]]:
    """Find all consistent configuration pairs.

    Two configurations c1, c2 are consistent iff they have an upper bound
    in the domain — i.e., there exists a configuration c3 with c1 ⊆ c3
    and c2 ⊆ c3.

    Returns pairs of indices into domain.configs.
    """
    n = len(domain.configs)
    result: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i, n):
            # Check if there's an upper bound
            ubs = _upper_bounds(domain, {i, j})
            if ubs:
                result.append((i, j))
    return result


# ---------------------------------------------------------------------------
# Public API: Coherence
# ---------------------------------------------------------------------------

def check_coherence(domain: ConfigDomain) -> bool:
    """Check the coherence property of the configuration domain.

    A domain is coherent iff every pairwise-consistent subset has an
    upper bound.  For finite posets: if every pair in a set S has an
    upper bound, then S itself has an upper bound.

    This is equivalent to: for every set of pairwise-consistent elements,
    there exists a common upper bound.
    """
    n = len(domain.configs)
    if n <= 1:
        return True

    # Build pairwise consistency
    consistent: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i, n):
            if _upper_bounds(domain, {i, j}):
                consistent.add((i, j))
                consistent.add((j, i))

    # Check every pairwise-consistent subset has an upper bound.
    # For efficiency on small domains, check all subsets up to a limit.
    # For larger domains, check triples (sufficient for most cases).
    if n <= 12:
        # Check all subsets via BFS on pairwise-consistent sets
        for size in range(3, n + 1):
            for subset in _subsets_of_size(n, size):
                # Check pairwise consistent
                pw_consistent = True
                for a in subset:
                    for b in subset:
                        if a < b and (a, b) not in consistent:
                            pw_consistent = False
                            break
                    if not pw_consistent:
                        break
                if pw_consistent:
                    # Must have common upper bound
                    if not _upper_bounds(domain, subset):
                        return False
    else:
        # For larger domains, check triples only
        idx_list = list(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in consistent:
                    continue
                for k in range(j + 1, n):
                    if (i, k) in consistent and (j, k) in consistent:
                        if not _upper_bounds(domain, {i, j, k}):
                            return False
    return True


def _subsets_of_size(n: int, size: int) -> list[set[int]]:
    """Generate all subsets of {0,...,n-1} of a given size."""
    if size == 0:
        return [set()]
    if size > n:
        return []
    result: list[set[int]] = []

    def _gen(start: int, current: list[int]) -> None:
        if len(current) == size:
            result.append(set(current))
            return
        remaining = size - len(current)
        for i in range(start, n - remaining + 1):
            current.append(i)
            _gen(i + 1, current)
            current.pop()

    _gen(0, [])
    return result


# ---------------------------------------------------------------------------
# Public API: Algebraicity
# ---------------------------------------------------------------------------

def check_algebraicity(domain: ConfigDomain) -> bool:
    """Check algebraicity: every element is a directed join of compacts below it.

    In a finite poset, all elements are compact, and every element is
    the join of the singleton set {x}, which is trivially directed.
    So algebraicity always holds for finite configuration domains.
    """
    # Trivially true for finite posets.
    return True


# ---------------------------------------------------------------------------
# Public API: Bounded completeness
# ---------------------------------------------------------------------------

def check_bounded_completeness(domain: ConfigDomain) -> bool:
    """Check bounded completeness: every subset with an upper bound has a sup.

    For a finite poset this reduces to: for every pair {a, b} with an
    upper bound, the join (least upper bound) exists.
    """
    n = len(domain.configs)
    for i in range(n):
        for j in range(i + 1, n):
            ubs = _upper_bounds(domain, {i, j})
            if ubs:
                # Must have a least upper bound
                sup = _sup(domain, {i, j})
                if sup is None:
                    return False
    return True


# ---------------------------------------------------------------------------
# Public API: Scott domain check
# ---------------------------------------------------------------------------

def check_scott_domain(domain: ConfigDomain) -> ScottDomainResult:
    """Verify all Scott domain properties of a configuration domain.

    For finite posets:
    - dcpo: automatic (all directed sets are finite, supremum exists)
    - algebraicity: automatic (all elements are compact)
    - bounded completeness: check pairwise
    - coherence: pairwise-consistent => common upper bound
    - distributivity: meets distribute over directed joins
    """
    is_algebraic = check_algebraicity(domain)
    is_bounded_complete = check_bounded_completeness(domain)
    is_dcpo = True  # automatic for finite posets
    is_coherent = check_coherence(domain)
    is_distributive = check_distributivity(domain)

    is_scott = is_dcpo and is_algebraic and is_bounded_complete
    is_di = is_scott and is_distributive

    compacts = compact_elements(domain)
    pairs = consistent_pairs(domain)
    scott_opens = scott_open_sets(domain)

    return ScottDomainResult(
        is_scott_domain=is_scott,
        is_algebraic=is_algebraic,
        is_bounded_complete=is_bounded_complete,
        is_dcpo=is_dcpo,
        is_coherent=is_coherent,
        is_distributive=is_distributive,
        is_di_domain=is_di,
        num_compact=len(compacts),
        num_configs=len(domain.configs),
        num_consistent_pairs=len(pairs),
        num_scott_open=len(scott_opens),
    )


# ---------------------------------------------------------------------------
# Public API: Scott topology
# ---------------------------------------------------------------------------

def scott_open_sets(domain: ConfigDomain) -> list[frozenset[int]]:
    """Enumerate Scott-open sets of the configuration domain.

    In a finite poset with the Scott topology, open sets are exactly
    the **upward-closed** (upper) sets: if x is in U and x <= y then
    y is in U.

    (In the finite case, the Scott topology coincides with the
    Alexandrov topology of upper sets, because the directed-join
    inaccessibility condition is vacuous for finite directed sets.)

    Returns a list of frozensets of indices.
    """
    n = len(domain.configs)
    # Build "above" lookup: for each index, which indices are >= it
    above: dict[int, set[int]] = {i: set() for i in range(n)}
    for (i, j) in domain.ordering:
        above[i].add(j)

    result: list[frozenset[int]] = []
    # Enumerate all subsets and check upward-closed
    for mask in range(1 << n):
        subset = {i for i in range(n) if mask & (1 << i)}
        is_upper = True
        for i in subset:
            if not above[i].issubset(subset):
                is_upper = False
                break
        if is_upper:
            result.append(frozenset(subset))
    return result


# ---------------------------------------------------------------------------
# Public API: Distributivity
# ---------------------------------------------------------------------------

def check_distributivity(domain: ConfigDomain) -> bool:
    """Check if meets distribute over directed joins.

    For a finite bounded-complete poset, this reduces to checking
    the standard lattice distributivity: a meet (b join c) = (a meet b) join (a meet c)
    for all triples a, b, c where meets and joins exist.

    If the domain is not a lattice (some joins/meets don't exist),
    we only check the axiom for elements where the operations are defined.
    """
    n = len(domain.configs)
    if n <= 2:
        return True

    for a in range(n):
        for b in range(n):
            for c in range(n):
                # Compute b join c
                bc_join = _sup(domain, {b, c})
                if bc_join is None:
                    continue

                # Compute a meet (b join c)
                lhs = _inf(domain, {a, bc_join})
                if lhs is None:
                    continue

                # Compute a meet b
                ab_meet = _inf(domain, {a, b})
                if ab_meet is None:
                    continue

                # Compute a meet c
                ac_meet = _inf(domain, {a, c})
                if ac_meet is None:
                    continue

                # Compute (a meet b) join (a meet c)
                rhs = _sup(domain, {ab_meet, ac_meet})
                if rhs is None:
                    continue

                if lhs != rhs:
                    return False

    return True


# ---------------------------------------------------------------------------
# Public API: Covering chains (linearizations)
# ---------------------------------------------------------------------------

def covering_chains(domain: ConfigDomain) -> list[list[int]]:
    """Enumerate all maximal covering chains in the configuration domain.

    A covering chain is a sequence c0 < c1 < ... < cn where each ci+1
    covers ci (ci+1 has exactly one more event than ci).
    Maximal chains go from bottom (empty config) to a maximal element.

    Returns lists of config indices.
    """
    n = len(domain.configs)
    if n == 0:
        return []

    # Build covers: i covers j iff i < j and no k with i < k < j
    # In config domain: covers means exactly one event difference
    covers: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) not in domain.ordering:
                continue
            # i <= j; check if it's a cover (no k with i < k < j)
            ci_events = domain.configs[i].events
            cj_events = domain.configs[j].events
            if len(cj_events) == len(ci_events) + 1 and ci_events < cj_events:
                covers[i].append(j)

    # Find bottom index (empty configuration)
    bottom_idx = None
    for i, c in enumerate(domain.configs):
        if len(c.events) == 0:
            bottom_idx = i
            break
    if bottom_idx is None:
        return []

    # Find maximal elements (no element strictly above)
    maximal: set[int] = set()
    for i in range(n):
        has_cover = bool(covers[i])
        if not has_cover:
            maximal.add(i)

    # DFS from bottom to maximal elements
    result: list[list[int]] = []

    def _dfs(current: int, path: list[int]) -> None:
        if current in maximal:
            result.append(list(path))
            return
        for nxt in covers[current]:
            path.append(nxt)
            _dfs(nxt, path)
            path.pop()

    _dfs(bottom_idx, [bottom_idx])
    return result


# ---------------------------------------------------------------------------
# Public API: Join-irreducible elements
# ---------------------------------------------------------------------------

def join_irreducibles(domain: ConfigDomain) -> list[int]:
    """Find join-irreducible elements of the configuration domain.

    An element j is join-irreducible if j != bottom and j covers
    exactly one element (has exactly one lower cover).

    Returns indices into domain.configs.
    """
    n = len(domain.configs)
    if n == 0:
        return []

    # Find bottom
    bottom_idx = None
    for i, c in enumerate(domain.configs):
        if len(c.events) == 0:
            bottom_idx = i
            break

    result: list[int] = []
    for j in range(n):
        if j == bottom_idx:
            continue
        # Count lower covers of j
        lower_covers = 0
        for i in range(n):
            if i == j:
                continue
            if (i, j) not in domain.ordering:
                continue
            ci_events = domain.configs[i].events
            cj_events = domain.configs[j].events
            if len(cj_events) == len(ci_events) + 1 and ci_events < cj_events:
                lower_covers += 1
        if lower_covers == 1:
            result.append(j)

    return result


# ---------------------------------------------------------------------------
# Public API: Domain lattice check
# ---------------------------------------------------------------------------

def is_lattice(domain: ConfigDomain) -> bool:
    """Check if the configuration domain is a lattice.

    A poset is a lattice iff every pair has both a join and a meet.
    """
    n = len(domain.configs)
    if n <= 1:
        return True

    for i in range(n):
        for j in range(i + 1, n):
            if _sup(domain, {i, j}) is None:
                return False
            if _inf(domain, {i, j}) is None:
                return False
    return True


# ---------------------------------------------------------------------------
# Public API: Width (max antichain)
# ---------------------------------------------------------------------------

def width(domain: ConfigDomain) -> int:
    """Compute the width (maximum antichain size) of the domain.

    Uses brute force for small domains. An antichain is a set of
    pairwise incomparable elements.
    """
    n = len(domain.configs)
    if n == 0:
        return 0

    # Build comparability
    comparable: set[tuple[int, int]] = set()
    for (i, j) in domain.ordering:
        if i != j:
            comparable.add((i, j))
            comparable.add((j, i))

    best = 1
    # For small domains, check all subsets
    if n <= 16:
        for mask in range(1, 1 << n):
            subset = [i for i in range(n) if mask & (1 << i)]
            sz = len(subset)
            if sz <= best:
                continue
            is_antichain = True
            for a in range(sz):
                for b in range(a + 1, sz):
                    if (subset[a], subset[b]) in comparable:
                        is_antichain = False
                        break
                if not is_antichain:
                    break
            if is_antichain:
                best = sz
    else:
        # Greedy approximation for larger domains
        remaining = set(range(n))
        antichain: list[int] = []
        for i in sorted(remaining):
            if all((i, j) not in comparable and (j, i) not in comparable
                   for j in antichain):
                antichain.append(i)
        best = max(best, len(antichain))
    return best


# ---------------------------------------------------------------------------
# Public API: Full analysis
# ---------------------------------------------------------------------------

def analyze_config_domain(ss: StateSpace) -> ConfigDomainAnalysis:
    """Full configuration domain analysis of a state space."""
    es = build_event_structure(ss)
    dom = config_domain(es)

    scott_result = check_scott_domain(dom)
    chains = covering_chains(dom)
    ji = join_irreducibles(dom)
    is_lat = is_lattice(dom)
    w = width(dom)

    max_chain_len = max((len(ch) for ch in chains), default=0)

    return ConfigDomainAnalysis(
        domain=dom,
        es=es,
        scott_result=scott_result,
        num_configs=len(dom.configs),
        num_covering_chains=len(chains),
        num_join_irreducibles=len(ji),
        max_chain_length=max_chain_len,
        width=w,
        is_lattice=is_lat,
    )
