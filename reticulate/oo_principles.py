"""Object-Oriented Principles as Lattice Theorems (Step 51b).

Maps the nine foundational OO principles — encapsulation, inheritance,
polymorphism, abstraction, and the five SOLID principles — onto lattice-
theoretic operations on session type state spaces.

The central insight: every OO principle that programmers learned empirically
is a *theorem* about finite bounded lattices.  The lattice L(S) of a
session type S is the algebraic essence of an object's protocol, and OO
principles are structural properties of that lattice.

Principle -> Lattice operation
-------------------------------
Encapsulation     -> enabled_methods() hides internal structure
Inheritance       -> Gay-Hole subtyping + lattice embedding
Polymorphism      -> join (least upper bound) in the subtyping lattice
Abstraction       -> quotient lattice L/theta collapses internal detail
Single Resp. (S)  -> join-irreducibility of the protocol lattice
Open/Closed (O)   -> covariant width subtyping preserves existing branches
Liskov Subst. (L) -> decidable via is_subtype() + constructive embedding
Interface Seg (I) -> Birkhoff decomposition into join-irreducible components
Dep. Inversion(D) -> Galois connection alpha |- gamma between layers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.lattice import check_lattice, compute_join, compute_meet
from reticulate.morphism import (
    find_embedding,
    find_isomorphism,
    is_galois_connection,
    is_order_preserving,
)
from reticulate.parser import Branch, End, Select, SessionType, parse, pretty
from reticulate.statespace import StateSpace, build_statespace
from reticulate.subtyping import is_subtype

if TYPE_CHECKING:
    pass


# ============================================================================
# Result types
# ============================================================================


@dataclass(frozen=True)
class EncapsulationResult:
    """Result of encapsulation (information hiding) analysis.

    Attributes:
        state_count: Number of states in the state space.
        method_sets: Mapping state -> set of enabled method names.
        all_states_have_methods: True iff every non-bottom state has methods.
        is_encapsulated: True iff the state space properly hides internals
            (each state exposes only its own enabled methods).
    """

    state_count: int
    method_sets: dict[int, frozenset[str]]
    all_states_have_methods: bool
    is_encapsulated: bool


@dataclass(frozen=True)
class InheritanceResult:
    """Result of inheritance (subtyping + embedding) check.

    Attributes:
        parent_type: Pretty-printed parent type.
        child_type: Pretty-printed child type.
        is_subtype: True iff child <=_GH parent.
        has_embedding: True iff L(parent) embeds into L(child).
        is_valid_inheritance: True iff both subtype and embedding hold.
        reason: Explanation if inheritance is invalid.
    """

    parent_type: str
    child_type: str
    is_subtype: bool
    has_embedding: bool
    is_valid_inheritance: bool
    reason: str | None = None


@dataclass(frozen=True)
class PolymorphismResult:
    """Result of polymorphic interface computation.

    Attributes:
        type_strings: Pretty-printed input types.
        common_methods: Methods shared by all types.
        has_common_supertype: True iff a join (common supertype) exists.
        common_supertype: The computed join type (if it exists).
    """

    type_strings: tuple[str, ...]
    common_methods: frozenset[str]
    has_common_supertype: bool
    common_supertype: str | None = None


@dataclass(frozen=True)
class AbstractionResult:
    """Result of protocol abstraction via label removal.

    Attributes:
        original_states: Number of states in original.
        abstract_states: Number of states after abstraction.
        removed_labels: Labels that were abstracted away.
        is_valid_abstraction: True iff the abstract state space is a lattice
            and is smaller (or equal) to the original.
        reduction_ratio: abstract_states / original_states.
    """

    original_states: int
    abstract_states: int
    removed_labels: frozenset[str]
    is_valid_abstraction: bool
    reduction_ratio: float


@dataclass(frozen=True)
class SRPResult:
    """Result of Single Responsibility Principle analysis.

    A protocol has single responsibility iff its state-space lattice is
    join-irreducible: it cannot be decomposed as the join of two strictly
    smaller sub-lattices.

    Attributes:
        state_count: Number of states.
        join_irreducibles: Set of join-irreducible state IDs.
        num_responsibilities: Number of distinct responsibility atoms.
        is_single_responsibility: True iff exactly one top-level responsibility.
        responsibilities: Labels at each join-irreducible state.
    """

    state_count: int
    join_irreducibles: frozenset[int]
    num_responsibilities: int
    is_single_responsibility: bool
    responsibilities: dict[int, frozenset[str]]


@dataclass(frozen=True)
class OpenClosedResult:
    """Result of Open/Closed Principle check.

    Open for extension: can add new branches (methods).
    Closed for modification: existing branches are preserved.

    Attributes:
        original_type: Pretty-printed original type.
        extended_type: Pretty-printed extended type.
        original_methods: Methods in the original.
        extended_methods: Methods in the extension.
        new_methods: Methods added by extension.
        preserves_existing: True iff all original methods are preserved.
        is_proper_extension: True iff new methods were added.
        satisfies_ocp: True iff both conditions hold.
    """

    original_type: str
    extended_type: str
    original_methods: frozenset[str]
    extended_methods: frozenset[str]
    new_methods: frozenset[str]
    preserves_existing: bool
    is_proper_extension: bool
    satisfies_ocp: bool


@dataclass(frozen=True)
class LiskovResult:
    """Result of Liskov Substitution Principle check.

    LSP is decidable: S_derived can substitute for S_base iff
    S_derived <=_GH S_base, which is decidable for session types.

    Attributes:
        base_type: Pretty-printed base type.
        derived_type: Pretty-printed derived type.
        is_subtype: True iff derived <=_GH base.
        has_embedding: True iff L(base) embeds into L(derived).
        satisfies_lsp: True iff substitution is safe.
        counterexample: Description of why LSP fails (if it does).
    """

    base_type: str
    derived_type: str
    is_subtype: bool
    has_embedding: bool
    satisfies_lsp: bool
    counterexample: str | None = None


@dataclass(frozen=True)
class Interface:
    """A minimal interface extracted from a protocol lattice.

    Attributes:
        state: The join-irreducible state representing this interface.
        methods: Methods accessible through this interface.
        label: Human-readable label.
    """

    state: int
    methods: frozenset[str]
    label: str


@dataclass(frozen=True)
class DependencyInversionResult:
    """Result of Dependency Inversion Principle check.

    DIP holds iff there is a Galois connection between the abstract
    (interface) layer and the concrete (implementation) layer.

    Attributes:
        abstract_states: Number of states in abstract layer.
        concrete_states: Number of states in concrete layer.
        has_galois_connection: True iff alpha -| gamma exists.
        satisfies_dip: True iff dependency inversion holds.
        reason: Explanation if DIP fails.
    """

    abstract_states: int
    concrete_states: int
    has_galois_connection: bool
    satisfies_dip: bool
    reason: str | None = None


@dataclass(frozen=True)
class SOLIDReport:
    """Traffic-light report for all five SOLID principles.

    Attributes:
        srp: Single Responsibility result.
        ocp: Open/Closed result (None if no extension provided).
        lsp: Liskov Substitution result (None if no base provided).
        isp: Interface Segregation (list of interfaces).
        dip: Dependency Inversion result (None if no abstract provided).
        summary: Dict of principle -> "green"|"yellow"|"red".
    """

    srp: SRPResult
    ocp: OpenClosedResult | None
    lsp: LiskovResult | None
    isp: list[Interface]
    dip: DependencyInversionResult | None
    summary: dict[str, str]


@dataclass(frozen=True)
class OOAnalysis:
    """Complete OO principles analysis for a session type.

    Attributes:
        encapsulation: Encapsulation analysis.
        inheritance: Inheritance analysis (if parent provided).
        polymorphism: Polymorphism analysis (if peers provided).
        abstraction: Abstraction analysis (if detail labels provided).
        solid: SOLID report.
    """

    encapsulation: EncapsulationResult
    inheritance: InheritanceResult | None
    polymorphism: PolymorphismResult | None
    abstraction: AbstractionResult | None
    solid: SOLIDReport


# ============================================================================
# 1. Encapsulation: information hiding via enabled methods
# ============================================================================


def check_encapsulation(ss: StateSpace) -> EncapsulationResult:
    """Verify that the state space properly hides internals.

    Each state exposes only its enabled methods.  The state space IS the
    interface: you cannot see internal implementation, only the set of
    methods available at each protocol state.

    The lattice structure guarantees that state transitions are the ONLY
    way to change which methods are visible — encapsulation is a structural
    property of the lattice, not a convention.
    """
    method_sets: dict[int, frozenset[str]] = {}
    for state in ss.states:
        methods = ss.enabled_methods(state)
        selections = ss.enabled_selections(state)
        all_labels = frozenset(l for l, _ in methods + selections)
        method_sets[state] = all_labels

    # Every non-bottom state should have at least one enabled method
    all_have_methods = all(
        len(method_sets[s]) > 0
        for s in ss.states
        if s != ss.bottom
    )

    # Encapsulation holds: each state's view is determined entirely by
    # the lattice position, not by internal structure
    is_encapsulated = all_have_methods or len(ss.states) == 1

    return EncapsulationResult(
        state_count=len(ss.states),
        method_sets=method_sets,
        all_states_have_methods=all_have_methods,
        is_encapsulated=is_encapsulated,
    )


# ============================================================================
# 2. Inheritance: subtype relationship
# ============================================================================


def check_inheritance(
    s_parent: SessionType,
    s_child: SessionType,
) -> InheritanceResult:
    """Check whether s_child safely inherits from s_parent.

    Inheritance is valid iff:
    1. s_child <=_GH s_parent (Gay-Hole subtyping), AND
    2. L(s_parent) order-embeds into L(s_child).

    The embedding IS the proof that every parent behavior is preserved
    in the child — inheritance is not a convention but a lattice theorem.
    """
    sub = is_subtype(s_child, s_parent)

    try:
        ss_parent = build_statespace(s_parent)
        ss_child = build_statespace(s_child)
        emb = find_embedding(ss_parent, ss_child)
        has_emb = emb is not None
    except (ValueError, RecursionError, KeyError):
        has_emb = False

    reason = None
    valid = sub and has_emb
    if not sub:
        reason = "child is not a subtype of parent (Gay-Hole)"
    elif not has_emb:
        reason = "parent lattice does not embed into child lattice"

    return InheritanceResult(
        parent_type=pretty(s_parent),
        child_type=pretty(s_child),
        is_subtype=sub,
        has_embedding=has_emb,
        is_valid_inheritance=valid,
        reason=reason,
    )


# ============================================================================
# 3. Polymorphism: common supertype as join
# ============================================================================


def find_polymorphic_interface(
    types: list[SessionType],
) -> PolymorphismResult:
    """Compute the polymorphic interface shared by multiple types.

    Polymorphism exists iff the types have a common supertype — i.e., the
    JOIN in the subtyping lattice exists.  The common supertype is
    computed as the intersection of top-level methods (for branch types).

    This is the algebraic content of polymorphism: multiple types that
    can be used interchangeably through a shared interface are types
    whose join exists in the subtype ordering.
    """
    if not types:
        return PolymorphismResult(
            type_strings=(),
            common_methods=frozenset(),
            has_common_supertype=False,
        )

    type_strings = tuple(pretty(t) for t in types)

    # Compute common methods across all types
    method_sets: list[set[str]] = []
    for t in types:
        methods = _top_level_methods(t)
        method_sets.append(methods)

    if method_sets:
        common = frozenset.intersection(*(frozenset(ms) for ms in method_sets))
    else:
        common = frozenset()

    has_common = len(common) > 0

    # Build the common supertype as Branch with shared methods -> end
    if has_common:
        supertype_str = "&{" + ", ".join(
            f"{m}: end" for m in sorted(common)
        ) + "}"
    else:
        supertype_str = None

    return PolymorphismResult(
        type_strings=type_strings,
        common_methods=common,
        has_common_supertype=has_common,
        common_supertype=supertype_str,
    )


def _top_level_methods(t: SessionType) -> set[str]:
    """Extract top-level method names from a session type."""
    from reticulate.parser import Rec

    if isinstance(t, Branch):
        return {m for m, _ in t.choices}
    elif isinstance(t, Select):
        return {l for l, _ in t.choices}
    elif isinstance(t, Rec):
        return _top_level_methods(t.body)
    elif isinstance(t, End):
        return set()
    return set()


# ============================================================================
# 4. Abstraction: quotient lattice via label removal
# ============================================================================


def abstract_protocol(
    ss: StateSpace,
    detail_labels: set[str],
) -> AbstractionResult:
    """Create an abstract view of a protocol by removing detail labels.

    Abstraction works by removing transitions with the specified labels,
    then collapsing unreachable states.  The result is a quotient of the
    original lattice that hides internal detail.

    This corresponds to the OO principle of abstraction: hiding
    implementation details behind a simpler interface.  The quotient
    lattice L/theta IS the abstract class.
    """
    if not detail_labels:
        return AbstractionResult(
            original_states=len(ss.states),
            abstract_states=len(ss.states),
            removed_labels=frozenset(),
            is_valid_abstraction=True,
            reduction_ratio=1.0,
        )

    # Build abstract state space: remove detail transitions
    abstract_transitions = [
        (s, l, t) for s, l, t in ss.transitions
        if l not in detail_labels
    ]
    abstract_selection = {
        (s, l, t) for s, l, t in ss.selection_transitions
        if l not in detail_labels
    }

    # Compute reachable states from top in the abstract graph
    adj: dict[int, set[int]] = {s: set() for s in ss.states}
    for s, _, t in abstract_transitions:
        adj[s].add(t)

    reachable: set[int] = set()
    stack = [ss.top]
    while stack:
        s = stack.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for t in adj.get(s, set()):
            stack.append(t)

    # Always include bottom
    reachable.add(ss.bottom)

    abstract_states = reachable
    abstract_trans_filtered = [
        (s, l, t) for s, l, t in abstract_transitions
        if s in abstract_states and t in abstract_states
    ]

    abstract_ss = StateSpace(
        states=abstract_states,
        transitions=abstract_trans_filtered,
        top=ss.top,
        bottom=ss.bottom,
        labels={s: ss.labels.get(s, "") for s in abstract_states},
        selection_transitions={
            tr for tr in abstract_selection
            if tr[0] in abstract_states and tr[2] in abstract_states
        },
    )

    lattice_result = check_lattice(abstract_ss)

    return AbstractionResult(
        original_states=len(ss.states),
        abstract_states=len(abstract_states),
        removed_labels=frozenset(detail_labels),
        is_valid_abstraction=lattice_result.is_lattice,
        reduction_ratio=len(abstract_states) / len(ss.states) if ss.states else 1.0,
    )


# ============================================================================
# 5. Single Responsibility: join-irreducibility
# ============================================================================


def _compute_join_irreducibles(ss: StateSpace) -> dict[int, frozenset[str]]:
    """Find join-irreducible elements of the state-space lattice.

    A state s is join-irreducible iff it has exactly one immediate
    predecessor in the Hasse diagram (i.e., it covers exactly one element).

    In terms of the lattice, join-irreducibles are the atoms that cannot
    be decomposed as the join of strictly smaller elements.  Each one
    represents a single, atomic responsibility.
    """
    # Build reachability
    reach = {s: ss.reachable_from(s) for s in ss.states}

    # Build Hasse diagram: s covers t iff s -> t in the transitive reduction
    # s covers t iff t in reach(s), t != s, and there is no u with
    # t in reach(u) and u in reach(s) and u != s and u != t
    covers: dict[int, set[int]] = {s: set() for s in ss.states}
    covered_by: dict[int, set[int]] = {s: set() for s in ss.states}

    for s in ss.states:
        for t in reach[s]:
            if t == s:
                continue
            # Check if s covers t (no intermediate u)
            is_cover = True
            for u in reach[s]:
                if u == s or u == t:
                    continue
                if t in reach[u]:
                    is_cover = False
                    break
            if is_cover:
                covers[s].add(t)
                covered_by[t].add(s)

    # Join-irreducible: exactly one element covers it (one immediate predecessor above)
    join_irr: dict[int, frozenset[str]] = {}
    for s in ss.states:
        if s == ss.top:
            continue  # top is never join-irreducible in a non-trivial lattice
        if len(covered_by[s]) == 1:
            methods = frozenset(l for l, _ in ss.enabled_methods(s) + ss.enabled_selections(s))
            join_irr[s] = methods

    return join_irr


def check_single_responsibility(ss: StateSpace) -> SRPResult:
    """Check the Single Responsibility Principle.

    A protocol has single responsibility iff its lattice has a small number
    of join-irreducible elements relative to its size.  Each join-irreducible
    represents one atomic capability — one "responsibility."

    In the ideal case (SRP satisfied), the protocol's lattice is itself
    join-irreducible: it cannot be decomposed into independent sub-protocols.
    """
    join_irr = _compute_join_irreducibles(ss)

    # Count distinct top-level responsibilities (join-irreducibles
    # directly below top)
    reach = {s: ss.reachable_from(s) for s in ss.states}
    top_children = set()
    for s in ss.states:
        if s == ss.top or s == ss.bottom:
            continue
        if s in reach[ss.top] and s != ss.top:
            # Check if directly covered by top
            is_direct = True
            for u in reach[ss.top]:
                if u == ss.top or u == s:
                    continue
                if s in reach[u]:
                    is_direct = False
                    break
            if is_direct:
                top_children.add(s)

    num_resp = max(len(top_children), 1) if len(ss.states) > 1 else 1

    return SRPResult(
        state_count=len(ss.states),
        join_irreducibles=frozenset(join_irr.keys()),
        num_responsibilities=num_resp,
        is_single_responsibility=(num_resp <= 1),
        responsibilities=join_irr,
    )


# ============================================================================
# 6. Open/Closed: extend without modifying
# ============================================================================


def check_open_closed(
    s_original: SessionType,
    s_extended: SessionType,
) -> OpenClosedResult:
    """Check the Open/Closed Principle.

    A type extension satisfies OCP iff:
    1. All methods in the original are preserved in the extension, AND
    2. The extension adds at least one new method.

    This IS covariant width subtyping: the extended type is a subtype of
    the original because it offers MORE methods while preserving all
    existing ones.
    """
    orig_methods = _top_level_methods(s_original)
    ext_methods = _top_level_methods(s_extended)

    preserves = orig_methods <= ext_methods
    new_methods = ext_methods - orig_methods
    is_proper = len(new_methods) > 0
    satisfies = preserves and is_proper

    return OpenClosedResult(
        original_type=pretty(s_original),
        extended_type=pretty(s_extended),
        original_methods=frozenset(orig_methods),
        extended_methods=frozenset(ext_methods),
        new_methods=frozenset(new_methods),
        preserves_existing=preserves,
        is_proper_extension=is_proper,
        satisfies_ocp=satisfies,
    )


# ============================================================================
# 7. Liskov Substitution: decidable via subtyping + embedding
# ============================================================================


def check_liskov(
    s_base: SessionType,
    s_derived: SessionType,
) -> LiskovResult:
    """Check the Liskov Substitution Principle.

    LSP is DECIDABLE for session types: s_derived can safely substitute
    for s_base iff s_derived <=_GH s_base.  When this holds, the lattice
    embedding L(s_base) -> L(s_derived) is the CONSTRUCTIVE PROOF that
    substitution is safe.

    When LSP fails, the failure gives a concrete counterexample showing
    which method is missing or which transition is incompatible.
    """
    sub = is_subtype(s_derived, s_base)

    try:
        ss_base = build_statespace(s_base)
        ss_derived = build_statespace(s_derived)
        emb = find_embedding(ss_base, ss_derived)
        has_emb = emb is not None
    except (ValueError, RecursionError, KeyError):
        has_emb = False

    counterexample = None
    if not sub:
        base_methods = _top_level_methods(s_base)
        derived_methods = _top_level_methods(s_derived)
        missing = base_methods - derived_methods
        if missing:
            counterexample = f"derived type is missing methods: {sorted(missing)}"
        else:
            counterexample = "incompatible continuation types"

    return LiskovResult(
        base_type=pretty(s_base),
        derived_type=pretty(s_derived),
        is_subtype=sub,
        has_embedding=has_emb,
        satisfies_lsp=sub,
        counterexample=counterexample,
    )


# ============================================================================
# 8. Interface Segregation: Birkhoff decomposition
# ============================================================================


def segregate_interfaces(ss: StateSpace) -> list[Interface]:
    """Decompose a protocol into minimal interfaces.

    Uses join-irreducible decomposition: each join-irreducible element
    of the lattice represents a minimal interface that cannot be further
    decomposed.

    "Clients should not depend on methods they don't use" — the Birkhoff
    decomposition gives the EXACT set of minimal interfaces, each
    containing only the methods required for one atomic capability.
    """
    join_irr = _compute_join_irreducibles(ss)

    interfaces: list[Interface] = []
    for state, methods in sorted(join_irr.items()):
        label = ss.labels.get(state, f"state_{state}")
        interfaces.append(Interface(
            state=state,
            methods=methods,
            label=label,
        ))

    # If no join-irreducibles found (trivial lattice), the entire
    # protocol is one interface
    if not interfaces and len(ss.states) > 1:
        all_methods: set[str] = set()
        for s in ss.states:
            for l, _ in ss.enabled_methods(s) + ss.enabled_selections(s):
                all_methods.add(l)
        interfaces.append(Interface(
            state=ss.top,
            methods=frozenset(all_methods),
            label="monolithic",
        ))

    return interfaces


# ============================================================================
# 9. Dependency Inversion: Galois connection
# ============================================================================


def check_dependency_inversion(
    ss_abstract: StateSpace,
    ss_concrete: StateSpace,
) -> DependencyInversionResult:
    """Check the Dependency Inversion Principle.

    DIP holds iff there is a Galois connection (alpha, gamma) between
    the abstract (interface) state space and the concrete (implementation)
    state space:

        alpha: concrete -> abstract  (abstraction map)
        gamma: abstract -> concrete  (concretization map)

    satisfying: alpha(x) <= y  iff  x <= gamma(y).

    This is the algebraic content of DIP: high-level modules depend on
    abstractions (alpha projects down), and low-level modules implement
    abstractions (gamma embeds up).
    """
    # Try to find an embedding of abstract into concrete
    emb = find_embedding(ss_abstract, ss_concrete)
    if emb is None:
        return DependencyInversionResult(
            abstract_states=len(ss_abstract.states),
            concrete_states=len(ss_concrete.states),
            has_galois_connection=False,
            satisfies_dip=False,
            reason="abstract lattice does not embed into concrete lattice",
        )

    # The embedding gives us gamma (concretization)
    gamma = emb.mapping

    # Build alpha (abstraction): for each concrete state, find the
    # closest abstract state (the one whose image is nearest in the
    # concrete lattice)
    concrete_reach = {s: ss_concrete.reachable_from(s) for s in ss_concrete.states}
    gamma_image = set(gamma.values())

    alpha: dict[int, int] = {}
    for c in ss_concrete.states:
        # Find the abstract state whose gamma-image is the greatest
        # lower bound of c among gamma-images
        best_abstract = None
        best_concrete = None
        for a, gc in gamma.items():
            if gc in concrete_reach[c]:  # gc <= c, i.e., c reaches gc
                if best_concrete is None or (
                    best_concrete in concrete_reach[gc]
                    and gc != best_concrete
                ):
                    best_abstract = a
                    best_concrete = gc
        if best_abstract is not None:
            alpha[c] = best_abstract
        else:
            # Map to top of abstract
            alpha[c] = ss_abstract.top

    # Check Galois connection property
    if len(alpha) == len(ss_concrete.states) and len(gamma) == len(ss_abstract.states):
        try:
            is_gc = is_galois_connection(alpha, gamma, ss_concrete, ss_abstract)
        except (KeyError, ValueError):
            is_gc = False
    else:
        is_gc = False

    return DependencyInversionResult(
        abstract_states=len(ss_abstract.states),
        concrete_states=len(ss_concrete.states),
        has_galois_connection=is_gc,
        satisfies_dip=is_gc,
        reason=None if is_gc else "Galois connection conditions not satisfied",
    )


# ============================================================================
# SOLID check: all five principles
# ============================================================================


def solid_check(
    ss: StateSpace,
    s_type: SessionType | None = None,
    s_extended: SessionType | None = None,
    s_base: SessionType | None = None,
    ss_abstract: StateSpace | None = None,
) -> SOLIDReport:
    """Run all five SOLID principle checks.

    Returns a traffic-light report: green (passes), yellow (partial),
    red (fails) for each principle.

    Args:
        ss: The state space to check.
        s_type: The session type (for OCP, LSP checks).
        s_extended: Extended type (for OCP check).
        s_base: Base type (for LSP check).
        ss_abstract: Abstract state space (for DIP check).
    """
    # S - Single Responsibility
    srp = check_single_responsibility(ss)

    # O - Open/Closed
    ocp = None
    if s_type is not None and s_extended is not None:
        ocp = check_open_closed(s_type, s_extended)

    # L - Liskov Substitution
    lsp = None
    if s_type is not None and s_base is not None:
        lsp = check_liskov(s_base, s_type)

    # I - Interface Segregation
    isp = segregate_interfaces(ss)

    # D - Dependency Inversion
    dip = None
    if ss_abstract is not None:
        dip = check_dependency_inversion(ss_abstract, ss)

    # Traffic light summary
    summary: dict[str, str] = {}

    # SRP: green if single responsibility, yellow if 2-3, red if >3
    if srp.is_single_responsibility:
        summary["SRP"] = "green"
    elif srp.num_responsibilities <= 3:
        summary["SRP"] = "yellow"
    else:
        summary["SRP"] = "red"

    # OCP
    if ocp is not None:
        summary["OCP"] = "green" if ocp.satisfies_ocp else "red"
    else:
        summary["OCP"] = "yellow"  # not tested

    # LSP
    if lsp is not None:
        summary["LSP"] = "green" if lsp.satisfies_lsp else "red"
    else:
        summary["LSP"] = "yellow"

    # ISP: green if multiple interfaces, yellow if monolithic, red if empty
    if len(isp) > 1:
        summary["ISP"] = "green"
    elif len(isp) == 1 and isp[0].label != "monolithic":
        summary["ISP"] = "green"
    elif len(isp) == 1:
        summary["ISP"] = "yellow"
    else:
        summary["ISP"] = "red"

    # DIP
    if dip is not None:
        summary["DIP"] = "green" if dip.satisfies_dip else "red"
    else:
        summary["DIP"] = "yellow"

    return SOLIDReport(
        srp=srp,
        ocp=ocp,
        lsp=lsp,
        isp=isp,
        dip=dip,
        summary=summary,
    )


# ============================================================================
# Full OO analysis
# ============================================================================


def analyze_oo_principles(
    ss: StateSpace,
    s_type: SessionType | None = None,
    s_parent: SessionType | None = None,
    s_child: SessionType | None = None,
    peer_types: list[SessionType] | None = None,
    detail_labels: set[str] | None = None,
    s_extended: SessionType | None = None,
    s_base: SessionType | None = None,
    ss_abstract: StateSpace | None = None,
) -> OOAnalysis:
    """Run complete OO principles analysis on a session type state space.

    Args:
        ss: The state space to analyze.
        s_type: The session type AST (for type-level checks).
        s_parent: Parent type (for inheritance check).
        s_child: Child type (for inheritance check).
        peer_types: Peer types (for polymorphism check).
        detail_labels: Labels to abstract away (for abstraction check).
        s_extended: Extended type (for OCP check).
        s_base: Base type (for LSP check).
        ss_abstract: Abstract state space (for DIP check).

    Returns:
        Complete OOAnalysis with all nine principles checked.
    """
    # 1. Encapsulation (always)
    encap = check_encapsulation(ss)

    # 2. Inheritance (if parent and child provided)
    inherit = None
    if s_parent is not None and s_child is not None:
        inherit = check_inheritance(s_parent, s_child)

    # 3. Polymorphism (if peer types provided)
    poly = None
    if peer_types is not None:
        poly = find_polymorphic_interface(peer_types)

    # 4. Abstraction (if detail labels provided)
    abstr = None
    if detail_labels is not None:
        abstr = abstract_protocol(ss, detail_labels)

    # 5-9. SOLID
    solid = solid_check(
        ss,
        s_type=s_type,
        s_extended=s_extended,
        s_base=s_base,
        ss_abstract=ss_abstract,
    )

    return OOAnalysis(
        encapsulation=encap,
        inheritance=inherit,
        polymorphism=poly,
        abstraction=abstr,
        solid=solid,
    )
