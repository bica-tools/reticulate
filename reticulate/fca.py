"""Formal Concept Analysis for session type lattices (Step 30aa).

Formal Concept Analysis (FCA) constructs a concept lattice from a
binary relation (formal context) between objects and attributes.
For session type state spaces:

- **Objects** = states in the state space
- **Attributes** = transition labels (methods) available in the protocol
- **Incidence** = state s has attribute m iff m is enabled at s

The concept lattice of this formal context provides an alternative
lattice representation of the protocol structure.  We compare it to
the original state-space lattice via morphisms.

Key functions:
- ``formal_context(ss)`` — extract (objects, attributes, incidence)
- ``extent(attrs, context)`` — objects sharing all given attributes
- ``intent(objs, context)`` — attributes shared by all given objects
- ``concept_closure(objs, context)`` — close to a formal concept
- ``all_concepts(context)`` — enumerate all formal concepts
- ``concept_lattice(context)`` — build the concept lattice as a poset
- ``analyze_fca(ss)`` — full analysis with comparison to original lattice

All computations are exact and use only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormalContext:
    """A formal context (G, M, I).

    Attributes:
        objects: Sorted list of object identifiers (state IDs).
        attributes: Sorted list of attribute names (transition labels).
        incidence: Mapping object -> set of attributes.
    """
    objects: tuple[int, ...]
    attributes: tuple[str, ...]
    incidence: dict[int, frozenset[str]]


@dataclass(frozen=True)
class FormalConcept:
    """A formal concept (A, B) where A ⊆ G, B ⊆ M.

    Attributes:
        extent: The set of objects (states).
        intent: The set of attributes (labels).
    """
    extent: frozenset[int]
    intent: frozenset[str]


@dataclass(frozen=True)
class ConceptLattice:
    """The concept lattice of a formal context.

    Attributes:
        concepts: List of all formal concepts.
        order: List of (i, j) pairs where concept i ≤ concept j
               (i.e., extent_i ⊆ extent_j).
        top: Index of the top concept (largest extent).
        bottom: Index of the bottom concept (smallest extent).
    """
    concepts: tuple[FormalConcept, ...]
    order: tuple[tuple[int, int], ...]
    top: int
    bottom: int


@dataclass(frozen=True)
class FCAResult:
    """Complete FCA analysis result.

    Attributes:
        context: The formal context.
        num_objects: Number of objects (states).
        num_attributes: Number of attributes (labels).
        num_concepts: Number of formal concepts.
        concept_lattice: The concept lattice.
        density: Incidence density |I| / (|G| × |M|).
        is_clarified: True iff no two objects have same intent
                      and no two attributes have same extent.
        num_original_states: Number of states in the original state space.
        lattice_size_ratio: num_concepts / num_original_states.
    """
    context: FormalContext
    num_objects: int
    num_attributes: int
    num_concepts: int
    concept_lattice: ConceptLattice
    density: float
    is_clarified: bool
    num_original_states: int
    lattice_size_ratio: float


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

def formal_context(ss: StateSpace) -> FormalContext:
    """Extract the formal context from a session type state space.

    Objects are states, attributes are transition labels.
    State s has attribute m iff there is a transition (s, m, _) in ss.
    """
    objects = sorted(ss.states)
    # Collect all labels
    all_labels: set[str] = set()
    for _src, label, _tgt in ss.transitions:
        all_labels.add(label)
    attributes = sorted(all_labels)

    incidence: dict[int, frozenset[str]] = {}
    for s in objects:
        labels = frozenset(l for src, l, _tgt in ss.transitions if src == s)
        incidence[s] = labels

    return FormalContext(
        objects=tuple(objects),
        attributes=tuple(attributes),
        incidence=incidence,
    )


# ---------------------------------------------------------------------------
# Derivation operators
# ---------------------------------------------------------------------------

def extent(attrs: frozenset[str], context: FormalContext) -> frozenset[int]:
    """Compute the extent of a set of attributes.

    Returns the set of objects that have ALL the given attributes.
    """
    if not attrs:
        return frozenset(context.objects)
    result: set[int] = set()
    for obj in context.objects:
        if attrs <= context.incidence[obj]:
            result.add(obj)
    return frozenset(result)


def intent(objs: frozenset[int], context: FormalContext) -> frozenset[str]:
    """Compute the intent of a set of objects.

    Returns the set of attributes shared by ALL the given objects.
    """
    if not objs:
        return frozenset(context.attributes)
    attrs: set[str] | None = None
    for obj in objs:
        obj_attrs = context.incidence.get(obj, frozenset())
        if attrs is None:
            attrs = set(obj_attrs)
        else:
            attrs &= obj_attrs
    return frozenset(attrs if attrs is not None else set())


def concept_closure(objs: frozenset[int], context: FormalContext) -> FormalConcept:
    """Close a set of objects to a formal concept.

    Computes (A'', A') where A'' = extent(intent(A)) and A' = intent(A).
    """
    b = intent(objs, context)
    a = extent(b, context)
    return FormalConcept(extent=a, intent=b)


def attribute_closure(attrs: frozenset[str], context: FormalContext) -> FormalConcept:
    """Close a set of attributes to a formal concept.

    Computes (B', B'') where B' = extent(attrs) and B'' = intent(extent(attrs)).
    """
    a = extent(attrs, context)
    b = intent(a, context)
    return FormalConcept(extent=a, intent=b)


# ---------------------------------------------------------------------------
# Concept enumeration
# ---------------------------------------------------------------------------

def all_concepts(context: FormalContext) -> list[FormalConcept]:
    """Enumerate all formal concepts using the basic closure algorithm.

    Uses the bottom-up approach: for each subset of objects, compute the
    closure and collect unique concepts.  For efficiency, we iterate over
    attribute subsets (typically smaller than object subsets for session types).
    """
    seen: set[tuple[frozenset[int], frozenset[str]]] = set()
    concepts: list[FormalConcept] = []

    # Close each subset of attributes
    n_attrs = len(context.attributes)
    if n_attrs <= 20:  # Enumerate attribute subsets
        for mask in range(1 << n_attrs):
            attrs = frozenset(
                context.attributes[i]
                for i in range(n_attrs)
                if mask & (1 << i)
            )
            concept = attribute_closure(attrs, context)
            key = (concept.extent, concept.intent)
            if key not in seen:
                seen.add(key)
                concepts.append(concept)
    else:
        # Fallback: enumerate object subsets (for contexts with many attributes)
        n_objs = len(context.objects)
        for mask in range(1 << n_objs):
            objs = frozenset(
                context.objects[i]
                for i in range(n_objs)
                if mask & (1 << i)
            )
            concept = concept_closure(objs, context)
            key = (concept.extent, concept.intent)
            if key not in seen:
                seen.add(key)
                concepts.append(concept)

    # Sort by extent size (descending) for consistent ordering
    concepts.sort(key=lambda c: (-len(c.extent), sorted(c.extent)))
    return concepts


# ---------------------------------------------------------------------------
# Concept lattice construction
# ---------------------------------------------------------------------------

def concept_lattice(context: FormalContext) -> ConceptLattice:
    """Build the concept lattice from a formal context.

    The ordering is by extent inclusion: (A1, B1) ≤ (A2, B2) iff A1 ⊆ A2.
    """
    concepts = all_concepts(context)
    n = len(concepts)

    # Build order relation (extent inclusion)
    order: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and concepts[i].extent <= concepts[j].extent:
                order.append((i, j))

    # Find top (largest extent) and bottom (smallest extent)
    top_idx = max(range(n), key=lambda i: len(concepts[i].extent))
    bottom_idx = min(range(n), key=lambda i: len(concepts[i].extent))

    return ConceptLattice(
        concepts=tuple(concepts),
        order=tuple(order),
        top=top_idx,
        bottom=bottom_idx,
    )


# ---------------------------------------------------------------------------
# Context properties
# ---------------------------------------------------------------------------

def is_clarified(context: FormalContext) -> bool:
    """Check if the context is clarified.

    A context is clarified iff no two objects have the same intent
    and no two attributes have the same extent.
    """
    # Check objects: no two should have the same attribute set
    seen_intents: set[frozenset[str]] = set()
    for obj in context.objects:
        attrs = context.incidence[obj]
        if attrs in seen_intents:
            return False
        seen_intents.add(attrs)

    # Check attributes: no two should have the same object set
    seen_extents: set[frozenset[int]] = set()
    for attr in context.attributes:
        objs = frozenset(
            o for o in context.objects
            if attr in context.incidence[o]
        )
        if objs in seen_extents:
            return False
        seen_extents.add(objs)

    return True


def context_density(context: FormalContext) -> float:
    """Compute the density of the incidence relation.

    Returns |I| / (|G| × |M|), the fraction of filled cells.
    """
    n_objs = len(context.objects)
    n_attrs = len(context.attributes)
    if n_objs == 0 or n_attrs == 0:
        return 0.0
    total_ones = sum(len(attrs) for attrs in context.incidence.values())
    return total_ones / (n_objs * n_attrs)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_fca(ss: StateSpace) -> FCAResult:
    """Perform complete FCA analysis on a session type state space.

    Constructs the formal context, enumerates concepts, builds the
    concept lattice, and computes comparison metrics.
    """
    ctx = formal_context(ss)
    lattice = concept_lattice(ctx)
    n_concepts = len(lattice.concepts)
    n_states = len(ss.states)

    return FCAResult(
        context=ctx,
        num_objects=len(ctx.objects),
        num_attributes=len(ctx.attributes),
        num_concepts=n_concepts,
        concept_lattice=lattice,
        density=context_density(ctx),
        is_clarified=is_clarified(ctx),
        num_original_states=n_states,
        lattice_size_ratio=n_concepts / n_states if n_states > 0 else 0.0,
    )
