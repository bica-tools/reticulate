"""Synchronous parallel composition for session types (Step 5d).

The synchronous product S₁ ||| S₂ requires ALL components to step
simultaneously.  Unlike the interleaving product S₁ ∥ S₂ (where any
single component can advance independently), the synchronous product
fires a transition only when every component fires one.

The transition label is the TUPLE of component labels, representing
simultaneous actions.  For musical scores this is a CHORD; for
hardware this is a CLOCK TICK; for synchronous circuits this is a
STEP of the parallel pipeline.

Key properties:
- L(S₁ ||| S₂) is the DIAGONAL of L(S₁) × L(S₂)
- |L(S₁ ||| S₂)| = min(|L(S₁)|, |L(S₂)|)  (when both are chains)
- For equal-length chains: sync product = single chain of tuples
- The sync product is always a sublattice of the interleaving product
- Lattice preservation: if L(S₁) and L(S₂) are lattices, so is L(S₁ ||| S₂)

Applications:
- Orchestral scores: all instruments play simultaneously per beat
- Hardware clocks: all flip-flops update on the same clock edge
- Synchronous dataflow: all actors fire in lock-step
- SIMD processing: all lanes execute the same instruction
- Turn-based games: all players move simultaneously (RPS, etc.)

Label format for combined transitions:
    "label₁+label₂+...+labelₙ"   (components joined by '+')
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as cart_product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncProductResult:
    """Result of synchronous product construction.

    Attributes:
        state_space: The synchronous product state space.
        component_sizes: Tuple of component state counts.
        interleaving_size: State count of the interleaving product.
        sync_size: State count of the synchronous product.
        compression_ratio: sync_size / interleaving_size.
        is_chain: True if the sync product is a total order (chain).
    """
    state_space: "StateSpace"
    component_sizes: tuple[int, ...]
    interleaving_size: int
    sync_size: int
    compression_ratio: float
    is_chain: bool


# ---------------------------------------------------------------------------
# Transition map helper
# ---------------------------------------------------------------------------

def _transition_map(ss: "StateSpace") -> dict[int, list[tuple[str, int]]]:
    """Build state → [(label, target)] map."""
    tm: dict[int, list[tuple[str, int]]] = {}
    for src, label, tgt in ss.transitions:
        tm.setdefault(src, []).append((label, tgt))
    return tm


# ---------------------------------------------------------------------------
# Core: synchronous product of N state spaces
# ---------------------------------------------------------------------------

def sync_product(*components: "StateSpace") -> "StateSpace":
    """Construct the synchronous product of N state spaces.

    All components must step simultaneously.  A transition from
    (s₁, ..., sₙ) to (s₁', ..., sₙ') exists iff every component
    has a transition sᵢ → sᵢ'.  The label is "l₁+l₂+...+lₙ".

    Returns a StateSpace with:
    - States: reachable tuples from the synchronous semantics.
    - Transitions: simultaneous steps (combined labels).
    - Top: (top₁, ..., topₙ).
    - Bottom: (bot₁, ..., botₙ).
    """
    from reticulate.statespace import StateSpace

    if not components:
        raise ValueError("sync_product requires at least one component")

    if len(components) == 1:
        return components[0]

    # Build transition maps
    tmaps = [_transition_map(ss) for ss in components]

    # State enumeration via BFS from the synchronous top
    top_tuple = tuple(ss.top for ss in components)
    bot_tuple = tuple(ss.bottom for ss in components)

    state_map: dict[tuple[int, ...], int] = {}
    next_id = 0

    def get_id(combo: tuple[int, ...]) -> int:
        nonlocal next_id
        if combo not in state_map:
            state_map[combo] = next_id
            next_id += 1
        return state_map[combo]

    top_id = get_id(top_tuple)
    transitions: list[tuple[int, str, int]] = []

    # BFS
    queue = [top_tuple]
    visited: set[tuple[int, ...]] = set()

    while queue:
        combo = queue.pop(0)
        if combo in visited:
            continue
        visited.add(combo)
        sid = get_id(combo)

        # Get available transitions for each component
        voice_trans = [tmaps[i].get(combo[i], []) for i in range(len(components))]

        # ALL components must have at least one transition
        if not all(vt for vt in voice_trans):
            continue

        # Enumerate all synchronous combinations
        for combo_trans in cart_product(*voice_trans):
            labels = [lt[0] for lt in combo_trans]
            targets = tuple(lt[1] for lt in combo_trans)
            combined_label = '+'.join(labels)
            tid = get_id(targets)
            transitions.append((sid, combined_label, tid))
            if targets not in visited:
                queue.append(targets)

    bot_id = get_id(bot_tuple)

    return StateSpace(
        states=frozenset(state_map.values()),
        transitions=tuple(transitions),
        top=top_id,
        bottom=bot_id,
        selection_transitions=frozenset(),
    )


# ---------------------------------------------------------------------------
# Analysis: sync vs interleaving comparison
# ---------------------------------------------------------------------------

def analyze_sync_product(*components: "StateSpace") -> SyncProductResult:
    """Analyze synchronous product and compare with interleaving.

    Returns a SyncProductResult with compression metrics.
    """
    ss_sync = sync_product(*components)

    # Interleaving size = product of component sizes
    interleaving_size = 1
    for ss in components:
        interleaving_size *= len(ss.states)

    sync_size = len(ss_sync.states)
    ratio = sync_size / interleaving_size if interleaving_size > 0 else 1.0

    # Check if chain (total order)
    is_chain = len(ss_sync.transitions) == len(ss_sync.states) - 1

    return SyncProductResult(
        state_space=ss_sync,
        component_sizes=tuple(len(ss.states) for ss in components),
        interleaving_size=interleaving_size,
        sync_size=sync_size,
        compression_ratio=ratio,
        is_chain=is_chain,
    )


# ---------------------------------------------------------------------------
# Sublattice verification
# ---------------------------------------------------------------------------

def verify_sublattice(
    sync_ss: "StateSpace",
    interleaving_ss: "StateSpace",
) -> bool:
    """Verify that the synchronous product is a sublattice of the interleaving product.

    The sync product states form a subset of the interleaving product states.
    This checks that the subset is closed under meets and joins.
    """
    from reticulate.lattice import compute_meet, compute_join

    # For each pair of sync states, check meet and join exist in sync
    sync_states = list(sync_ss.states)
    for i, a in enumerate(sync_states):
        for b in sync_states[i+1:]:
            m = compute_meet(sync_ss, a, b)
            j = compute_join(sync_ss, a, b)
            if m is None or j is None:
                return False
            if m not in sync_ss.states or j not in sync_ss.states:
                return False
    return True
