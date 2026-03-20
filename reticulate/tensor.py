"""Tensor product: external (inter-object) product of session type lattices.

Step 168 of the Reticulate Programme.

Distinguishes two kinds of lattice product:

- **Internal product** (``∥``): one object, concurrent capabilities.
  ``(S₁ ∥ S₂)`` constructs a product *within* a single session type.
  Labels must be disjoint (WF-Par). Already in ``product.py``.

- **External product** (``⊗``): multiple objects, one program.
  A program holding ``o₁: S₁, o₂: S₂, …, oₙ: Sₙ`` has state space
  ``L(S₁) ⊗ L(S₂) ⊗ ⋯ ⊗ L(Sₙ)`` where each transition is
  **object-qualified**: ``"oᵢ.m"`` advances component *i* only.

The two constructions are algebraically identical (both lattice products)
but semantically different:

- Internal: concurrency on one object (threads).
- External: interleaving across objects (program state).

Key properties:

1. **Lattice closure**: ⊗ of lattices is a lattice.
2. **Projections**: πᵢ is a surjective lattice homomorphism.
3. **Flattening**: If Sᵢ uses ``∥``, internal and external products compose:
   ``L(S₁ ∥ S₂) ⊗ L(S₃) ≅ L(S₁) × L(S₂) × L(S₃)``
4. **Coupling**: Constraints between objects carve sublattices of the product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.parser import SessionType
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorResult:
    """Result of external (tensor) product construction.

    Attributes:
        participants: Mapping from object name to session type AST.
        state_spaces: Mapping from object name to state space.
        tensor: The program-level product state space (object-qualified labels).
        is_lattice: Whether the tensor product is a lattice (always True).
        projections: Mapping from object name to projection dict (tensor_id → component_id).
        state_counts: Per-object state count.
    """

    participants: dict[str, SessionType]
    state_spaces: dict[str, StateSpace]
    tensor: StateSpace
    is_lattice: bool
    projections: dict[str, dict[int, int]]
    state_counts: dict[str, int]


@dataclass(frozen=True)
class CouplingConstraint:
    """A constraint between object states.

    Expressed as a predicate on tuples of component states.
    ``required_before[obj_a]`` maps a state in obj_a to the set of
    (obj_b, state) pairs that must have been reached *before* obj_a
    enters that state.

    Example: "Logger must be at state 2+ before File enters state 3"
    → required_before = {"File": {3: {("Logger", 2)}}}
    """

    description: str
    required_before: dict[str, dict[int, set[tuple[str, int]]]]


@dataclass(frozen=True)
class CoupledTensorResult:
    """Result of coupled tensor product (constrained sublattice).

    Attributes:
        base: The unconstrained tensor result.
        constraints: The coupling constraints applied.
        coupled: The constrained state space (reachable sublattice).
        coupled_is_lattice: Whether the coupled sublattice is a lattice.
        base_states: Number of states in unconstrained product.
        coupled_states: Number of states in coupled sublattice.
        reduction_ratio: coupled_states / base_states.
    """

    base: TensorResult
    constraints: list[CouplingConstraint]
    coupled: StateSpace
    coupled_is_lattice: bool
    base_states: int
    coupled_states: int
    reduction_ratio: float


@dataclass(frozen=True)
class FlatteningResult:
    """Result of checking internal/external product flattening.

    If an object's session type uses ``∥``, its state space is already a product.
    The tensor product with other objects creates a nested product.
    Flattening checks that (A × B) ⊗ C ≅ A × B × C.

    Attributes:
        nested: The nested tensor product state space.
        flattened: The flattened (factor-expanded) state space.
        isomorphic: Whether nested ≅ flattened.
        nested_states: Number of states in nested form.
        flattened_states: Number of states in flattened form.
        factor_names: Ordered list of flattened factor names.
    """

    nested: StateSpace
    flattened: StateSpace
    isomorphic: bool
    nested_states: int
    flattened_states: int
    factor_names: list[str]


# ---------------------------------------------------------------------------
# Core: tensor product construction
# ---------------------------------------------------------------------------

def tensor_product(
    *participants: tuple[str, SessionType | StateSpace],
) -> TensorResult:
    """Construct the external (tensor) product of multiple objects.

    Each participant is ``(name, type_or_statespace)``.  If a session type
    AST is given, ``build_statespace`` is called.  Transition labels in the
    product are object-qualified: ``"name.method"``.

    Parameters:
        participants: Pairs of (object_name, session_type_or_statespace).

    Returns:
        TensorResult with the program-level product state space.

    Raises:
        ValueError: If fewer than 2 participants, duplicate names,
                    or a participant's state space is not a lattice.
    """
    from reticulate.statespace import StateSpace as SS, build_statespace
    from reticulate.lattice import check_lattice

    if len(participants) < 2:
        raise ValueError("Tensor product requires at least 2 participants")

    names = [name for name, _ in participants]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate participant names: {names}")

    # Build state spaces
    type_map: dict[str, SessionType] = {}
    ss_map: dict[str, StateSpace] = {}
    for name, obj in participants:
        if isinstance(obj, SS):
            ss_map[name] = obj
            type_map[name] = None  # type: ignore[assignment]
        else:
            type_map[name] = obj
            ss_map[name] = build_statespace(obj)

    # Verify each is a lattice
    for name, ss in ss_map.items():
        lr = check_lattice(ss)
        if not lr.is_lattice:
            raise ValueError(
                f"Participant '{name}' state space is not a lattice "
                f"(counterexample: {lr.counterexample})"
            )

    # Build N-ary tensor product with object-qualified labels
    product_ss = _build_tensor_nary(names, ss_map)

    # Build projections
    projections = _build_projections(names, ss_map, product_ss)

    return TensorResult(
        participants=type_map,
        state_spaces=ss_map,
        tensor=product_ss,
        is_lattice=True,
        projections=projections,
        state_counts={name: len(ss.states) for name, ss in ss_map.items()},
    )


def _build_tensor_nary(
    names: list[str],
    ss_map: dict[str, StateSpace],
) -> StateSpace:
    """Build N-ary tensor product with object-qualified labels."""
    from reticulate.statespace import StateSpace as SS

    ordered = [ss_map[n] for n in names]

    # Pre-compute adjacency lists per component
    adjs: list[dict[int, list[tuple[str, int]]]] = []
    for ss in ordered:
        adj: dict[int, list[tuple[str, int]]] = {s: [] for s in ss.states}
        for src, lbl, tgt in ss.transitions:
            adj[src].append((lbl, tgt))
        adjs.append(adj)

    # Enumerate all tuples of states
    import itertools
    state_lists = [sorted(ss.states) for ss in ordered]
    all_tuples = list(itertools.product(*state_lists))

    next_id = 0
    tuple_to_id: dict[tuple[int, ...], int] = {}
    id_labels: dict[int, str] = {}

    for tup in all_tuples:
        sid = next_id
        next_id += 1
        tuple_to_id[tup] = sid
        parts = []
        for i, s in enumerate(tup):
            parts.append(ordered[i].labels.get(s, str(s)))
        id_labels[sid] = "(" + ", ".join(parts) + ")"

    # Build transitions with object-qualified labels
    transitions: list[tuple[int, str, int]] = []
    selection_transitions: set[tuple[int, str, int]] = set()

    for tup in all_tuples:
        src = tuple_to_id[tup]
        for i, name in enumerate(names):
            for lbl, tgt_s in adjs[i][tup[i]]:
                new_tup = tup[:i] + (tgt_s,) + tup[i + 1:]
                qualified = f"{name}.{lbl}"
                tr = (src, qualified, tuple_to_id[new_tup])
                transitions.append(tr)
                if ordered[i].is_selection(tup[i], lbl, tgt_s):
                    selection_transitions.add(tr)

    top_tup = tuple(ss.top for ss in ordered)
    bot_tup = tuple(ss.bottom for ss in ordered)

    # Coordinate map
    coord_map: dict[int, tuple[int, ...]] = {}
    for tup, pid in tuple_to_id.items():
        coord_map[pid] = tup

    return SS(
        states=set(tuple_to_id.values()),
        transitions=transitions,
        top=tuple_to_id[top_tup],
        bottom=tuple_to_id[bot_tup],
        labels=id_labels,
        selection_transitions=selection_transitions,
        product_coords=coord_map,
        product_factors=list(ordered),
    )


def _build_projections(
    names: list[str],
    ss_map: dict[str, StateSpace],
    tensor_ss: StateSpace,
) -> dict[str, dict[int, int]]:
    """Build projection maps πᵢ: tensor → component_i."""
    projections: dict[str, dict[int, int]] = {}
    for i, name in enumerate(names):
        proj: dict[int, int] = {}
        for tid, coords in tensor_ss.product_coords.items():
            proj[tid] = coords[i]
        projections[name] = proj
    return projections


# ---------------------------------------------------------------------------
# Projections as lattice homomorphisms
# ---------------------------------------------------------------------------

def check_projection_homomorphism(
    result: TensorResult,
    name: str,
) -> bool:
    """Check that πᵢ is a surjective, order-preserving lattice homomorphism.

    Parameters:
        result: A TensorResult from tensor_product().
        name: The participant name whose projection to check.

    Returns:
        True iff the projection preserves order and is surjective.
    """
    from reticulate.morphism import is_order_preserving

    proj = result.projections[name]
    target = result.state_spaces[name]

    # Surjectivity: every target state is hit
    image = set(proj.values())
    if image != target.states:
        return False

    # Order preservation
    return is_order_preserving(result.tensor, target, proj)


# ---------------------------------------------------------------------------
# Coupling constraints
# ---------------------------------------------------------------------------

def coupled_tensor(
    result: TensorResult,
    constraints: list[CouplingConstraint],
) -> CoupledTensorResult:
    """Apply coupling constraints to a tensor product.

    Performs BFS from top, filtering out states that violate constraints.
    A state (s₁, …, sₙ) is reachable iff for every constraint, the
    required_before conditions are satisfied.

    Parameters:
        result: An unconstrained TensorResult.
        constraints: List of CouplingConstraint objects.

    Returns:
        CoupledTensorResult with the constrained sublattice.
    """
    from reticulate.statespace import StateSpace as SS
    from reticulate.lattice import check_lattice

    tensor = result.tensor
    names = list(result.state_spaces.keys())
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Build adjacency for BFS
    adj: dict[int, list[tuple[str, int]]] = {s: [] for s in tensor.states}
    for src, lbl, tgt in tensor.transitions:
        adj[src].append((lbl, tgt))

    def satisfies_constraints(coords: tuple[int, ...]) -> bool:
        """Check if a state tuple satisfies all coupling constraints."""
        for c in constraints:
            for obj_name, state_reqs in c.required_before.items():
                idx = name_to_idx[obj_name]
                obj_state = coords[idx]
                if obj_state in state_reqs:
                    for req_obj, req_state in state_reqs[obj_state]:
                        req_idx = name_to_idx[req_obj]
                        req_ss = result.state_spaces[req_obj]
                        # Check: req_obj must have reached req_state
                        # i.e., req_state is reachable from current state
                        # In our ordering: current ≤ req_state (req_state
                        # was passed through)
                        actual = coords[req_idx]
                        # req_state must be "above" (reachable-from)
                        # the current state, meaning actual ≤ req_state
                        # in the component ordering. Since ordering is
                        # reachability (top=initial, higher=earlier),
                        # "reached req_state" means req_state ≥ actual.
                        reachable = req_ss.reachable_from(req_state)
                        if actual not in reachable and actual != req_state:
                            return False
        return True

    # BFS from top
    reachable_ids: set[int] = set()
    visited: set[int] = set()
    queue = [tensor.top]
    visited.add(tensor.top)

    while queue:
        current = queue.pop(0)
        coords = tensor.product_coords[current]
        if not satisfies_constraints(coords):
            continue
        reachable_ids.add(current)
        for lbl, tgt in adj[current]:
            if tgt not in visited:
                visited.add(tgt)
                queue.append(tgt)

    # Build constrained state space
    coupled_transitions = [
        (src, lbl, tgt) for src, lbl, tgt in tensor.transitions
        if src in reachable_ids and tgt in reachable_ids
    ]
    coupled_selections = {
        tr for tr in tensor.selection_transitions
        if tr[0] in reachable_ids and tr[2] in reachable_ids
    }
    coupled_labels = {s: tensor.labels[s] for s in reachable_ids}
    coupled_coords = {s: tensor.product_coords[s] for s in reachable_ids}

    # Determine top/bottom of coupled space
    coupled_top = tensor.top if tensor.top in reachable_ids else min(reachable_ids)
    coupled_bottom = tensor.bottom if tensor.bottom in reachable_ids else max(reachable_ids)

    coupled_ss = SS(
        states=reachable_ids,
        transitions=coupled_transitions,
        top=coupled_top,
        bottom=coupled_bottom,
        labels=coupled_labels,
        selection_transitions=coupled_selections,
        product_coords=coupled_coords,
        product_factors=tensor.product_factors,
    )

    lr = check_lattice(coupled_ss)
    base_count = len(tensor.states)
    coupled_count = len(reachable_ids)

    return CoupledTensorResult(
        base=result,
        constraints=constraints,
        coupled=coupled_ss,
        coupled_is_lattice=lr.is_lattice,
        base_states=base_count,
        coupled_states=coupled_count,
        reduction_ratio=coupled_count / base_count if base_count > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Flattening: (A × B) ⊗ C ≅ A × B × C
# ---------------------------------------------------------------------------

def check_flattening(
    result: TensorResult,
) -> FlatteningResult:
    """Check that nested internal+external products flatten.

    If any participant's state space is itself a product (from ``∥``),
    the flattened version expands its factors.  The result checks that
    the nested and flattened products are isomorphic.

    Parameters:
        result: A TensorResult, possibly with participants whose
                state spaces have ``product_factors``.

    Returns:
        FlatteningResult with isomorphism check.
    """
    from reticulate.morphism import find_isomorphism

    names = list(result.state_spaces.keys())
    ss_list = [result.state_spaces[n] for n in names]

    # Expand factors
    factor_names: list[str] = []
    flat_factors: list[StateSpace] = []
    for i, (name, ss) in enumerate(zip(names, ss_list)):
        if ss.product_factors and len(ss.product_factors) > 1:
            for j, f in enumerate(ss.product_factors):
                factor_names.append(f"{name}[{j}]")
                flat_factors.append(f)
        else:
            factor_names.append(name)
            flat_factors.append(ss)

    # Build flattened product
    if len(flat_factors) == len(ss_list):
        # No expansion needed: nested == flattened
        return FlatteningResult(
            nested=result.tensor,
            flattened=result.tensor,
            isomorphic=True,
            nested_states=len(result.tensor.states),
            flattened_states=len(result.tensor.states),
            factor_names=factor_names,
        )

    # Build flattened tensor
    from reticulate.product import product_statespace
    from functools import reduce
    flattened = reduce(product_statespace, flat_factors)

    iso = find_isomorphism(result.tensor, flattened)

    return FlatteningResult(
        nested=result.tensor,
        flattened=flattened,
        isomorphic=iso is not None,
        nested_states=len(result.tensor.states),
        flattened_states=len(flattened.states),
        factor_names=factor_names,
    )


# ---------------------------------------------------------------------------
# Comparison: internal vs external
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InternalExternalComparison:
    """Comparison between internal (∥) and external (⊗) product.

    Attributes:
        internal_states: State count from internal product (∥).
        external_states: State count from external tensor (⊗).
        isomorphic: Whether the two are isomorphic (always True for
                    uncoupled tensor with matching factors).
        internal_labels: Set of labels in internal product.
        external_labels: Set of labels in external product (object-qualified).
        label_ratio: |external_labels| / |internal_labels| (≥ 1.0).
    """

    internal_states: int
    external_states: int
    isomorphic: bool
    internal_labels: set[str]
    external_labels: set[str]
    label_ratio: float


def compare_internal_external(
    s1: SessionType,
    s2: SessionType,
    name1: str = "A",
    name2: str = "B",
) -> InternalExternalComparison:
    """Compare internal product (∥) with external tensor (⊗).

    Builds ``L(S₁ ∥ S₂)`` (internal) and ``L(S₁) ⊗ L(S₂)`` (external).
    These have the same number of states but different label sets:
    internal labels are bare, external labels are object-qualified.

    Parameters:
        s1: First session type AST.
        s2: Second session type AST.
        name1: Name for the first object (default "A").
        name2: Name for the second object (default "B").

    Returns:
        InternalExternalComparison with structural comparison.
    """
    from reticulate.statespace import build_statespace
    from reticulate.product import product_statespace
    from reticulate.morphism import find_isomorphism

    ss1 = build_statespace(s1)
    ss2 = build_statespace(s2)

    # Internal: product of state spaces (as in ∥)
    internal = product_statespace(ss1, ss2)

    # External: tensor with qualified labels
    tr = tensor_product((name1, ss1), (name2, ss2))
    external = tr.tensor

    # Structural isomorphism (ignoring labels)
    iso = find_isomorphism(internal, external)

    internal_labels = {lbl for _, lbl, _ in internal.transitions}
    external_labels = {lbl for _, lbl, _ in external.transitions}

    return InternalExternalComparison(
        internal_states=len(internal.states),
        external_states=len(external.states),
        isomorphic=iso is not None,
        internal_labels=internal_labels,
        external_labels=external_labels,
        label_ratio=(
            len(external_labels) / len(internal_labels)
            if internal_labels
            else 1.0
        ),
    )
