"""MPST projection: global type → local session type (Step 12).

Projection maps a global type onto a single role's local view:

    (p -> q : {mᵢ: Gᵢ}) ↓ p  =  +{mᵢ: Gᵢ ↓ p}      (sender selects)
    (p -> q : {mᵢ: Gᵢ}) ↓ q  =  &{mᵢ: Gᵢ ↓ q}      (receiver branches)
    (p -> q : {mᵢ: Gᵢ}) ↓ r  =  merge(Gᵢ ↓ r)        (uninvolved)
    (G₁ || G₂) ↓ r            =  (G₁↓r) || (G₂↓r)     (if r in both)
    (G₁ || G₂) ↓ r            =  G₁↓r or G₂↓r         (if r in one)
    (rec X . G) ↓ r            =  rec X . (G ↓ r)       (if r in G)
    end ↓ r                    =  end

The Step 12 claim: projection is a surjective poset map from the global
state space L(G) to the local state space L(G↓r).
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.global_types import (
    GEnd,
    GMessage,
    GParallel,
    GRec,
    GVar,
    GlobalType,
    roles,
)
from reticulate.parser import (
    Branch,
    End,
    Parallel,
    Rec,
    Select,
    SessionType,
    Var,
    pretty,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProjectionError(Exception):
    """Raised when projection is undefined (e.g., merge failure)."""

    def __init__(self, message: str, role: str | None = None) -> None:
        super().__init__(message)
        self.role = role


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionResult:
    """Result of projecting a global type onto a role.

    Attributes:
        role: The role projected onto.
        local_type: The resulting local session type AST.
        local_type_str: Pretty-printed local type.
        is_well_defined: True iff projection succeeded.
    """
    role: str
    local_type: SessionType
    local_type_str: str
    is_well_defined: bool


@dataclass(frozen=True)
class ProjectionMorphismResult:
    """Result of verifying projection as a state-space morphism.

    Attributes:
        role: The role being projected.
        global_states: Number of global states.
        local_states: Number of local states.
        is_surjective: True iff the natural label-stripping map is surjective.
        is_order_preserving: True iff global reachability implies local reachability.
        global_is_lattice: True iff the global state space is a lattice.
        local_is_lattice: True iff the local state space is a lattice.
    """
    role: str
    global_states: int
    local_states: int
    is_surjective: bool
    is_order_preserving: bool
    global_is_lattice: bool
    local_is_lattice: bool


# ---------------------------------------------------------------------------
# Projection algorithm
# ---------------------------------------------------------------------------

def project(g: GlobalType, role: str) -> SessionType:
    """Project a global type onto a single role's local view.

    Returns a binary session type (Branch, Select, etc.) representing
    the role's local protocol.

    Raises ProjectionError if projection is undefined.
    """
    return _project(g, role, set())


def _project(g: GlobalType, role: str, in_progress: set[str]) -> SessionType:
    """Internal projection with recursion tracking."""
    match g:
        case GEnd():
            return End()

        case GVar(name=name):
            return Var(name)

        case GMessage(sender=s, receiver=r, choices=cs):
            if role == s:
                # Sender: internal choice (Select)
                projected = tuple(
                    (label, _project(body, role, in_progress))
                    for label, body in cs
                )
                return Select(projected)

            elif role == r:
                # Receiver: external choice (Branch)
                projected = tuple(
                    (label, _project(body, role, in_progress))
                    for label, body in cs
                )
                return Branch(projected)

            else:
                # Uninvolved: merge all projections
                projections = [
                    _project(body, role, in_progress) for _, body in cs
                ]
                return _merge(projections, role)

        case GParallel(left=left, right=right):
            left_roles = roles(left)
            right_roles = roles(right)
            in_left = role in left_roles
            in_right = role in right_roles

            if in_left and in_right:
                return Parallel(
                    _project(left, role, in_progress),
                    _project(right, role, in_progress),
                )
            elif in_left:
                return _project(left, role, in_progress)
            elif in_right:
                return _project(right, role, in_progress)
            else:
                return End()

        case GRec(var=var, body=body):
            if role not in roles(body):
                return End()
            if var in in_progress:
                return Var(var)
            projected_body = _project(body, role, in_progress | {var})
            # If the body projects to just End (role not actually involved
            # in the recursive part), don't wrap in Rec
            if projected_body == End():
                return End()
            return Rec(var, projected_body)

        case _:
            raise TypeError(f"unknown global type node: {type(g).__name__}")


def _merge(projections: list[SessionType], role: str) -> SessionType:
    """Merge projections for an uninvolved role.

    All projections must be structurally equal (strict merging).
    """
    if not projections:
        return End()

    first = projections[0]
    for p in projections[1:]:
        if p != first:
            raise ProjectionError(
                f"merge failure for role {role!r}: projections differ "
                f"({pretty(first)} vs {pretty(p)})",
                role=role,
            )
    return first


# ---------------------------------------------------------------------------
# Project onto all roles
# ---------------------------------------------------------------------------

def project_all(g: GlobalType) -> dict[str, SessionType]:
    """Project a global type onto all participating roles.

    Returns a dict mapping role name → local session type.
    Raises ProjectionError if any projection fails.
    """
    all_roles = roles(g)
    return {r: project(g, r) for r in sorted(all_roles)}


def check_projection(g: GlobalType, role: str) -> ProjectionResult:
    """Project with error handling, returning a result object."""
    try:
        local = project(g, role)
        return ProjectionResult(
            role=role,
            local_type=local,
            local_type_str=pretty(local),
            is_well_defined=True,
        )
    except ProjectionError:
        return ProjectionResult(
            role=role,
            local_type=End(),
            local_type_str="end",
            is_well_defined=False,
        )


# ---------------------------------------------------------------------------
# Morphism verification (Step 12 claim)
# ---------------------------------------------------------------------------

def verify_projection_morphism(
    g: GlobalType, role: str
) -> ProjectionMorphismResult:
    """Verify that projection induces a surjective order-preserving map.

    Builds both global and local state spaces, then checks:
    1. The global state space is a lattice.
    2. The local state space is a lattice.
    3. There exists a surjective order-preserving map from global to local.
    """
    from reticulate.global_types import build_global_statespace
    from reticulate.lattice import check_lattice
    from reticulate.statespace import build_statespace

    local_type = project(g, role)

    global_ss = build_global_statespace(g)
    local_ss = build_statespace(local_type)

    global_lattice = check_lattice(global_ss)
    local_lattice = check_lattice(local_ss)

    # Build the natural projection: strip role annotations from labels
    # and find matching transitions in the local state space.
    # For surjectivity, every local state must be reachable via some
    # global state under the projection.
    surjective, order_preserving = _check_morphism_properties(
        global_ss, local_ss, role
    )

    return ProjectionMorphismResult(
        role=role,
        global_states=len(global_ss.states),
        local_states=len(local_ss.states),
        is_surjective=surjective,
        is_order_preserving=order_preserving,
        global_is_lattice=global_lattice.is_lattice,
        local_is_lattice=local_lattice.is_lattice,
    )


def _strip_role_prefix(label: str, role: str) -> str | None:
    """Strip role annotation from a transition label.

    "A->B:method" → "method" if role is A (sender, selection) or B (receiver, method).
    Returns None if the label doesn't involve this role.
    """
    if "->" not in label:
        return label  # Already a plain label
    parts = label.split(":", 1)
    if len(parts) != 2:
        return None
    role_part, method = parts
    arrow_parts = role_part.split("->")
    if len(arrow_parts) != 2:
        return None
    sender, receiver = arrow_parts
    if role == sender or role == receiver:
        return method
    return None


def _check_morphism_properties(
    global_ss: "StateSpace",
    local_ss: "StateSpace",
    role: str,
) -> tuple[bool, bool]:
    """Check surjectivity and order-preservation of the natural projection.

    Uses BFS from top to build a mapping from global states to local states
    by matching transition labels (stripping role annotations).
    """
    # Build mapping by BFS through global state space
    mapping: dict[int, int] = {global_ss.top: local_ss.top}
    visited: set[int] = set()
    queue = [global_ss.top]

    while queue:
        gs = queue.pop(0)
        if gs in visited:
            continue
        visited.add(gs)

        if gs not in mapping:
            continue

        ls = mapping[gs]
        local_enabled = dict(local_ss.enabled(ls))

        for label, gtarget in global_ss.enabled(gs):
            method = _strip_role_prefix(label, role)
            if method is not None and method in local_enabled:
                ltarget = local_enabled[method]
                if gtarget not in mapping:
                    mapping[gtarget] = ltarget
            elif method is None:
                # Transition doesn't involve this role — map to same local state
                if gtarget not in mapping:
                    mapping[gtarget] = ls
            queue.append(gtarget)

    # Ensure bottom maps to bottom
    if global_ss.bottom not in mapping:
        mapping[global_ss.bottom] = local_ss.bottom

    # Surjectivity: every local state must be in the image
    image = set(mapping.values())
    surjective = local_ss.states <= image

    # Order-preservation: if g1 reaches g2, then mapping[g1] reaches mapping[g2]
    order_preserving = True
    for g1 in mapping:
        reach_g1 = global_ss.reachable_from(g1)
        for g2 in mapping:
            if g2 in reach_g1:
                l1, l2 = mapping[g1], mapping[g2]
                reach_l1 = local_ss.reachable_from(l1)
                if l2 not in reach_l1:
                    order_preserving = False
                    break
        if not order_preserving:
            break

    return surjective, order_preserving
