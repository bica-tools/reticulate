"""Impersonation detection via session type duality (Step 89g).

Detects impersonation attacks in session-typed protocols by comparing an
observed (claimed) session type against the expected dual of the legitimate
counterpart.  The key insight from session type theory is that a client
and server should have *dual* types: if the server offers branch, the
client selects, and vice versa.  An impersonator's type will deviate from
the expected dual, and the deviations reveal the impersonation.

Five analyses are provided:

1. **Authenticity check**: Is the claimed type compatible with the
   expected dual? (via subtyping or isomorphism)
2. **Duality distance**: How far is one state space from being the dual
   of another? (Measured by transition mismatches after isomorphism search)
3. **Authentication certificate**: Constructive proof that a client
   matches the dual of a server type.
4. **Impersonation detection**: Identify specific transitions that betray
   an impersonator (present in the claimed type but absent in the dual).
5. **Combined analysis**: All checks in a single result.

Key results:
  - Authentic participants have duality distance 0.
  - Impersonators have duality distance > 0, with betraying transitions.
  - The authentication certificate is a morphism from L(client) to L(dual(server)).
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
class ImpersonationResult:
    """Complete impersonation analysis result.

    Attributes:
        is_authentic: True if the claimed type matches the expected dual.
        duality_distance: Number of mismatched transitions (0 = authentic).
        has_certificate: True if an authentication certificate exists.
        certificate_kind: Kind of morphism ("isomorphism", "embedding", etc.) or None.
        betraying_transitions: Transitions that betray an impersonator.
        missing_transitions: Expected transitions absent from the claimed type.
        extra_transitions: Unexpected transitions present in the claimed type.
        selection_mismatches: Transitions with wrong selection polarity.
        num_states_claimed: Number of states in the claimed type.
        num_states_expected: Number of states in the expected dual.
        structural_similarity: Ratio of matching transitions to total.
    """
    is_authentic: bool
    duality_distance: int
    has_certificate: bool
    certificate_kind: str | None
    betraying_transitions: list[tuple[int, str, int]]
    missing_transitions: list[tuple[int, str, int]]
    extra_transitions: list[tuple[int, str, int]]
    selection_mismatches: list[tuple[int, str, int]]
    num_states_claimed: int
    num_states_expected: int
    structural_similarity: float


@dataclass(frozen=True)
class AuthenticationCertificate:
    """Proof that a client matches the dual of a server.

    Attributes:
        is_valid: True if the certificate validates.
        morphism_kind: Type of morphism found ("isomorphism", "embedding", etc.).
        mapping: State mapping from client to dual(server), or None.
        selection_consistent: True if selection polarity is correctly flipped.
    """
    is_valid: bool
    morphism_kind: str | None
    mapping: dict[int, int] | None
    selection_consistent: bool


# ---------------------------------------------------------------------------
# Authenticity check
# ---------------------------------------------------------------------------

def check_authentic(
    ss_claimed: "StateSpace",
    ss_expected: "StateSpace",
) -> bool:
    """Check if the claimed state space is compatible with the expected dual.

    Two state spaces are compatible if there exists an isomorphism or
    embedding between them.  The expected state space should be the dual
    of the legitimate counterpart (i.e., ss_expected = L(dual(S_server))).

    Args:
        ss_claimed: State space of the party claiming to be authentic.
        ss_expected: State space of the expected dual type.

    Returns:
        True if an isomorphism or embedding exists.
    """
    from reticulate.morphism import find_isomorphism

    # Strict authentication requires isomorphism (bijective correspondence).
    # Embedding is too weak: a smaller state space embeds into a larger one
    # but cannot faithfully represent the full protocol.
    iso = find_isomorphism(ss_claimed, ss_expected)
    return iso is not None


# ---------------------------------------------------------------------------
# Duality distance
# ---------------------------------------------------------------------------

def duality_distance(ss1: "StateSpace", ss2: "StateSpace") -> int:
    """Measure how far ss1 is from being the dual of ss2.

    The duality distance counts the number of transition mismatches
    between ss1 and ss2 when aligned by state count and label matching.
    Distance 0 means they are isomorphic (potentially dual types).

    The distance is computed as:
    |transitions only in ss1| + |transitions only in ss2|
    after normalising by relabeling states to align tops and bottoms.

    Args:
        ss1: First state space.
        ss2: Second state space.

    Returns:
        Non-negative integer distance. 0 iff isomorphic.
    """
    from reticulate.morphism import find_isomorphism

    # If isomorphic, distance is 0
    iso = find_isomorphism(ss1, ss2)
    if iso is not None:
        # Check selection polarity mismatches
        mismatches = _count_selection_mismatches(ss1, ss2, iso.mapping)
        return mismatches

    # Not isomorphic — count structural differences
    # Use label-based alignment
    labels1 = _transition_label_multiset(ss1)
    labels2 = _transition_label_multiset(ss2)

    # Symmetric difference of label multisets
    all_labels = set(labels1.keys()) | set(labels2.keys())
    distance = 0
    for lbl in all_labels:
        c1 = labels1.get(lbl, 0)
        c2 = labels2.get(lbl, 0)
        distance += abs(c1 - c2)

    # Add state count difference
    distance += abs(len(ss1.states) - len(ss2.states))

    return distance


def _transition_label_multiset(ss: "StateSpace") -> dict[str, int]:
    """Count transitions by label."""
    counts: dict[str, int] = {}
    for _, lbl, _ in ss.transitions:
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _count_selection_mismatches(
    ss1: "StateSpace",
    ss2: "StateSpace",
    mapping: dict[int, int],
) -> int:
    """Count transitions where selection polarity differs under a mapping."""
    mismatches = 0
    for src, lbl, tgt in ss1.transitions:
        is_sel1 = ss1.is_selection(src, lbl, tgt)
        mapped_src = mapping.get(src)
        mapped_tgt = mapping.get(tgt)
        if mapped_src is not None and mapped_tgt is not None:
            is_sel2 = ss2.is_selection(mapped_src, lbl, mapped_tgt)
            if is_sel1 != is_sel2:
                mismatches += 1
    return mismatches


# ---------------------------------------------------------------------------
# Authentication certificate
# ---------------------------------------------------------------------------

def authentication_certificate(
    ss_client: "StateSpace",
    ss_server: "StateSpace",
) -> AuthenticationCertificate:
    """Produce an authentication certificate proving client matches dual(server).

    The certificate is a morphism from L(client) to L(dual(server)).
    The server's state space is used to construct the expected dual
    (by building dual(server_type) and its state space).

    Since we work at the state-space level, we check for isomorphism
    between ss_client and ss_server (duality preserves state-space
    structure up to selection flip).

    Args:
        ss_client: Client's state space.
        ss_server: Server's state space (we check client ~ server modulo selection).

    Returns:
        AuthenticationCertificate with morphism details.
    """
    from reticulate.morphism import find_isomorphism

    # Authentication requires isomorphism — a bijective structure-preserving
    # map.  Embedding is insufficient because a subset of the protocol
    # cannot authenticate the full protocol.
    iso = find_isomorphism(ss_client, ss_server)
    if iso is not None:
        sel_ok = _check_selection_consistency(ss_client, ss_server, iso.mapping)
        return AuthenticationCertificate(
            is_valid=True,
            morphism_kind="isomorphism",
            mapping=iso.mapping,
            selection_consistent=sel_ok,
        )

    return AuthenticationCertificate(
        is_valid=False,
        morphism_kind=None,
        mapping=None,
        selection_consistent=False,
    )


def _check_selection_consistency(
    ss1: "StateSpace",
    ss2: "StateSpace",
    mapping: dict[int, int],
) -> bool:
    """Check if selection polarity is consistently flipped under mapping.

    For a valid client-server duality, every selection in ss1 should map
    to a non-selection in ss2 and vice versa (the duality flips choice).
    """
    for src, lbl, tgt in ss1.transitions:
        is_sel1 = ss1.is_selection(src, lbl, tgt)
        mapped_src = mapping.get(src)
        mapped_tgt = mapping.get(tgt)
        if mapped_src is not None and mapped_tgt is not None:
            # Check if corresponding transition exists
            has_match = any(
                s == mapped_src and l == lbl and t == mapped_tgt
                for s, l, t in ss2.transitions
            )
            if has_match:
                is_sel2 = ss2.is_selection(mapped_src, lbl, mapped_tgt)
                # Duality: selections should be flipped
                if is_sel1 == is_sel2:
                    return False
    return True


# ---------------------------------------------------------------------------
# Impersonation detection
# ---------------------------------------------------------------------------

def detect_impersonation(
    ss_observed: "StateSpace",
    ss_expected_dual: "StateSpace",
) -> tuple[list[tuple[int, str, int]], list[tuple[int, str, int]], list[tuple[int, str, int]]]:
    """Detect transitions that betray an impersonator.

    Compares the observed state space against the expected dual.
    Returns three lists:
    - extra_transitions: Present in observed but not expected (impersonator added).
    - missing_transitions: Expected but not observed (impersonator missed).
    - selection_mismatches: Present in both but with wrong polarity.

    Uses label-based matching when no isomorphism exists.
    """
    from reticulate.morphism import find_isomorphism

    iso = find_isomorphism(ss_observed, ss_expected_dual)

    if iso is not None:
        # Isomorphic structure — check selection mismatches
        mismatches: list[tuple[int, str, int]] = []
        for src, lbl, tgt in ss_observed.transitions:
            is_sel_obs = ss_observed.is_selection(src, lbl, tgt)
            m_src = iso.mapping[src]
            m_tgt = iso.mapping[tgt]
            is_sel_exp = ss_expected_dual.is_selection(m_src, lbl, m_tgt)
            if is_sel_obs != is_sel_exp:
                mismatches.append((src, lbl, tgt))
        return ([], [], mismatches)

    # No isomorphism — use label-based comparison
    obs_labels = {(lbl,) for _, lbl, _ in ss_observed.transitions}
    exp_labels = {(lbl,) for _, lbl, _ in ss_expected_dual.transitions}

    obs_label_set = {lbl for _, lbl, _ in ss_observed.transitions}
    exp_label_set = {lbl for _, lbl, _ in ss_expected_dual.transitions}

    extra: list[tuple[int, str, int]] = [
        (src, lbl, tgt) for src, lbl, tgt in ss_observed.transitions
        if lbl not in exp_label_set
    ]
    missing: list[tuple[int, str, int]] = [
        (src, lbl, tgt) for src, lbl, tgt in ss_expected_dual.transitions
        if lbl not in obs_label_set
    ]

    return (extra, missing, [])


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def analyze_impersonation(
    ss_claimed: "StateSpace",
    ss_expected: "StateSpace",
) -> ImpersonationResult:
    """Perform complete impersonation analysis.

    Compares a claimed state space against an expected state space
    (typically the dual of the legitimate counterpart).

    Args:
        ss_claimed: State space of the party claiming to be authentic.
        ss_expected: State space of the expected dual type.

    Returns:
        ImpersonationResult with all analysis details.
    """
    authentic = check_authentic(ss_claimed, ss_expected)
    dist = duality_distance(ss_claimed, ss_expected)
    cert = authentication_certificate(ss_claimed, ss_expected)
    extra, missing, sel_mis = detect_impersonation(ss_claimed, ss_expected)

    # Structural similarity
    total_transitions = len(ss_claimed.transitions) + len(ss_expected.transitions)
    if total_transitions == 0:
        similarity = 1.0
    else:
        matching = total_transitions - len(extra) - len(missing) - len(sel_mis)
        similarity = max(0.0, matching / total_transitions)

    return ImpersonationResult(
        is_authentic=authentic,
        duality_distance=dist,
        has_certificate=cert.is_valid,
        certificate_kind=cert.morphism_kind,
        betraying_transitions=extra + sel_mis,
        missing_transitions=missing,
        extra_transitions=extra,
        selection_mismatches=sel_mis,
        num_states_claimed=len(ss_claimed.states),
        num_states_expected=len(ss_expected.states),
        structural_similarity=round(similarity, 4),
    )
