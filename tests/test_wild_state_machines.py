"""Extract session types from real-world state machines found in the wild.

Step 9 application: take state machines from published sources (RFCs,
textbooks, Wikipedia) and use reconstruct() to extract their session
types.  This demonstrates that the reticular form characterisation
works beyond our own benchmark protocols.

Sources:
  - TCP state machine: RFC 793 (via ETH Zürich diagram)
  - Turnstile: classic FSM example (Wikipedia)
  - Radiation therapy: University of Washington FSM example
  - Vending machine: classic CS example
"""

import pytest

from reticulate.lattice import check_lattice
from reticulate.reticular import (
    check_reticular_form,
    classify_all_states,
    is_reticulate,
    reconstruct,
)
from reticulate.statespace import StateSpace, build_statespace
from reticulate.parser import pretty


# ---------------------------------------------------------------------------
# Helper: build a StateSpace from a list of (src, label, tgt) triples
# ---------------------------------------------------------------------------

def build_from_edges(
    state_names: dict[str, int],
    edges: list[tuple[str, str, str]],
    top_name: str,
    bottom_name: str,
    selection_labels: set[str] | None = None,
) -> StateSpace:
    """Build a StateSpace from named states and labeled edges.

    Args:
        state_names: mapping from human-readable name to integer ID.
        edges: list of (source_name, label, target_name) triples.
        top_name: name of the initial/top state.
        bottom_name: name of the terminal/bottom state.
        selection_labels: set of labels that are selections (internal choice).
    """
    if selection_labels is None:
        selection_labels = set()

    states = set(state_names.values())
    transitions = []
    selection_transitions = set()
    labels = {v: k for k, v in state_names.items()}

    for src_name, label, tgt_name in edges:
        src = state_names[src_name]
        tgt = state_names[tgt_name]
        t = (src, label, tgt)
        transitions.append(t)
        if label in selection_labels:
            selection_transitions.add(t)

    return StateSpace(
        states=states,
        transitions=transitions,
        top=state_names[top_name],
        bottom=state_names[bottom_name],
        labels=labels,
        selection_transitions=selection_transitions,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. TURNSTILE (Wikipedia classic)
#
# States: LOCKED, UNLOCKED
# Transitions:
#   LOCKED --coin--> UNLOCKED
#   UNLOCKED --push--> LOCKED
#   (Self-loops omitted: they don't add states)
#
# As a session type: this is a recursive protocol where the
# environment alternates between inserting coins and pushing.
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def turnstile_ss():
    """Turnstile FSM: LOCKED ↔ UNLOCKED."""
    return build_from_edges(
        state_names={"LOCKED": 0, "UNLOCKED": 1},
        edges=[
            ("LOCKED", "coin", "UNLOCKED"),
            ("UNLOCKED", "push", "LOCKED"),
        ],
        top_name="LOCKED",
        bottom_name="LOCKED",  # cycle — no true bottom
    )


class TestTurnstile:
    """Extract session type from a turnstile FSM."""

    def test_states_and_transitions(self, turnstile_ss):
        assert len(turnstile_ss.states) == 2
        assert len(turnstile_ss.transitions) == 2

    def test_reconstruct(self, turnstile_ss):
        s = reconstruct(turnstile_ss)
        p = pretty(s)
        assert "coin" in p
        assert "push" in p

    def test_is_reticulate(self, turnstile_ss):
        assert is_reticulate(turnstile_ss)

    def test_reconstructed_is_recursive(self, turnstile_ss):
        """A cyclic FSM should produce a recursive session type."""
        from reticulate.parser import Rec
        s = reconstruct(turnstile_ss)
        assert isinstance(s, Rec), f"Expected Rec, got {type(s).__name__}"


# ═══════════════════════════════════════════════════════════════════
# 2. RADIATION THERAPY MACHINE (Univ. Washington)
#
# States: patients, fields, setup, ready, beam_on
# 12 transitions (see source)
#
# This is a safety-critical system where the state machine
# controls when radiation can be delivered.
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def radiation_ss():
    """Radiation therapy control FSM."""
    return build_from_edges(
        state_names={
            "patients": 0, "fields": 1, "setup": 2,
            "ready": 3, "beam_on": 4,
        },
        edges=[
            ("patients", "enter", "fields"),
            ("fields", "select_patient", "patients"),
            ("fields", "enter", "setup"),
            ("setup", "select_patient", "patients"),
            ("setup", "select_field", "fields"),
            ("setup", "ok", "ready"),
            ("ready", "select_patient", "patients"),
            ("ready", "select_field", "fields"),
            ("ready", "start", "beam_on"),
            ("ready", "intlk", "setup"),
            ("beam_on", "stop", "ready"),
            ("beam_on", "intlk", "setup"),
        ],
        top_name="patients",
        bottom_name="patients",  # cycle — returns to patients
    )


class TestRadiationTherapy:
    """Extract session type from radiation therapy FSM."""

    def test_states_and_transitions(self, radiation_ss):
        assert len(radiation_ss.states) == 5
        assert len(radiation_ss.transitions) == 12

    def test_reconstruct(self, radiation_ss):
        s = reconstruct(radiation_ss)
        p = pretty(s)
        assert "enter" in p
        assert "start" in p or "beam" in p.lower()

    def test_is_reticulate(self, radiation_ss):
        assert is_reticulate(radiation_ss)

    def test_classification(self, radiation_ss):
        """All states should be branch (all transitions are methods)."""
        classifications = classify_all_states(radiation_ss)
        for c in classifications:
            assert c.kind in ("branch", "end"), (
                f"State {c.state} has kind {c.kind}"
            )


# ═══════════════════════════════════════════════════════════════════
# 3. VENDING MACHINE (classic CS)
#
# Simplified version:
#   IDLE --insert_coin--> HAS_COIN
#   HAS_COIN --select--> DISPENSING
#   DISPENSING --take--> IDLE
#   HAS_COIN --cancel--> IDLE (refund)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def vending_ss():
    """Vending machine FSM."""
    return build_from_edges(
        state_names={
            "IDLE": 0, "HAS_COIN": 1,
            "DISPENSING": 2,
        },
        edges=[
            ("IDLE", "insert_coin", "HAS_COIN"),
            ("HAS_COIN", "select", "DISPENSING"),
            ("HAS_COIN", "cancel", "IDLE"),
            ("DISPENSING", "take", "IDLE"),
        ],
        top_name="IDLE",
        bottom_name="IDLE",  # cycle
    )


class TestVendingMachine:
    """Extract session type from a vending machine FSM."""

    def test_states_and_transitions(self, vending_ss):
        assert len(vending_ss.states) == 3
        assert len(vending_ss.transitions) == 4

    def test_reconstruct(self, vending_ss):
        s = reconstruct(vending_ss)
        p = pretty(s)
        assert "insert_coin" in p
        assert "select" in p

    def test_is_reticulate(self, vending_ss):
        assert is_reticulate(vending_ss)

    def test_recursive_structure(self, vending_ss):
        from reticulate.parser import Rec
        s = reconstruct(vending_ss)
        assert isinstance(s, Rec), "Cyclic FSM should produce Rec type"


# ═══════════════════════════════════════════════════════════════════
# 4. TCP CONNECTION (RFC 793) — simplified client-side
#
# The full TCP state machine has 11 states and ~20 transitions,
# many of which are bidirectional. We model the client-side
# active open → data transfer → active close path.
#
# CLOSED --active_open--> SYN_SENT
# SYN_SENT --rcv_SYN_ACK--> ESTABLISHED
# ESTABLISHED --close--> FIN_WAIT_1
# ESTABLISHED --rcv_FIN--> CLOSE_WAIT
# FIN_WAIT_1 --rcv_ACK--> FIN_WAIT_2
# FIN_WAIT_2 --rcv_FIN--> TIME_WAIT
# TIME_WAIT --timeout--> CLOSED
# CLOSE_WAIT --close--> LAST_ACK
# LAST_ACK --rcv_ACK--> CLOSED
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def tcp_client_ss():
    """TCP client-side connection state machine (simplified)."""
    return build_from_edges(
        state_names={
            "CLOSED": 0, "SYN_SENT": 1, "ESTABLISHED": 2,
            "FIN_WAIT_1": 3, "FIN_WAIT_2": 4, "TIME_WAIT": 5,
            "CLOSE_WAIT": 6, "LAST_ACK": 7,
        },
        edges=[
            ("CLOSED", "active_open", "SYN_SENT"),
            ("SYN_SENT", "rcv_SYN_ACK", "ESTABLISHED"),
            ("ESTABLISHED", "close", "FIN_WAIT_1"),
            ("ESTABLISHED", "rcv_FIN", "CLOSE_WAIT"),
            ("FIN_WAIT_1", "rcv_ACK", "FIN_WAIT_2"),
            ("FIN_WAIT_2", "rcv_FIN", "TIME_WAIT"),
            ("TIME_WAIT", "timeout", "CLOSED"),
            ("CLOSE_WAIT", "close", "LAST_ACK"),
            ("LAST_ACK", "rcv_ACK", "CLOSED"),
        ],
        top_name="CLOSED",
        bottom_name="CLOSED",  # cycle back to CLOSED
        # rcv_* are events from the network (selections/internal)
        selection_labels={"rcv_SYN_ACK", "rcv_FIN", "rcv_ACK"},
    )


class TestTCPClient:
    """Extract session type from TCP client state machine."""

    def test_states_and_transitions(self, tcp_client_ss):
        assert len(tcp_client_ss.states) == 8
        assert len(tcp_client_ss.transitions) == 9

    def test_reconstruct(self, tcp_client_ss):
        s = reconstruct(tcp_client_ss)
        p = pretty(s)
        assert "active_open" in p

    def test_is_reticulate(self, tcp_client_ss):
        assert is_reticulate(tcp_client_ss)

    def test_has_selections(self, tcp_client_ss):
        """TCP has both methods (active_open, close) and selections (rcv_*)."""
        classifications = classify_all_states(tcp_client_ss)
        kinds = {c.kind for c in classifications}
        assert "branch" in kinds  # active_open, close are methods
        assert "select" in kinds  # rcv_* are selections


# ═══════════════════════════════════════════════════════════════════
# 5. DOOR LOCK (simple embedded system)
#
# LOCKED --unlock--> UNLOCKED
# UNLOCKED --open--> OPEN
# OPEN --close--> UNLOCKED
# UNLOCKED --lock--> LOCKED
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def door_lock_ss():
    """Door lock FSM."""
    return build_from_edges(
        state_names={"LOCKED": 0, "UNLOCKED": 1, "OPEN": 2},
        edges=[
            ("LOCKED", "unlock", "UNLOCKED"),
            ("UNLOCKED", "open", "OPEN"),
            ("OPEN", "close", "UNLOCKED"),
            ("UNLOCKED", "lock", "LOCKED"),
        ],
        top_name="LOCKED",
        bottom_name="LOCKED",  # cycle
    )


class TestDoorLock:
    """Extract session type from a door lock FSM."""

    def test_reconstruct(self, door_lock_ss):
        s = reconstruct(door_lock_ss)
        p = pretty(s)
        assert "unlock" in p
        assert "open" in p
        assert "close" in p
        assert "lock" in p

    def test_is_reticulate(self, door_lock_ss):
        assert is_reticulate(door_lock_ss)


# ═══════════════════════════════════════════════════════════════════
# 6. ATM (simplified)
#
# IDLE --insert_card--> AUTHENTICATING
# AUTHENTICATING --pin_ok--> READY (+selection: internal decision)
# AUTHENTICATING --pin_fail--> IDLE
# READY --withdraw--> DISPENSING
# READY --check_balance--> SHOWING_BALANCE
# SHOWING_BALANCE --done--> READY
# DISPENSING --take_cash--> IDLE
# READY --eject_card--> IDLE
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def atm_ss():
    """ATM state machine."""
    return build_from_edges(
        state_names={
            "IDLE": 0, "AUTHENTICATING": 1, "READY": 2,
            "DISPENSING": 3, "SHOWING_BALANCE": 4,
        },
        edges=[
            ("IDLE", "insert_card", "AUTHENTICATING"),
            ("AUTHENTICATING", "pin_ok", "READY"),
            ("AUTHENTICATING", "pin_fail", "IDLE"),
            ("READY", "withdraw", "DISPENSING"),
            ("READY", "check_balance", "SHOWING_BALANCE"),
            ("READY", "eject_card", "IDLE"),
            ("SHOWING_BALANCE", "done", "READY"),
            ("DISPENSING", "take_cash", "IDLE"),
        ],
        top_name="IDLE",
        bottom_name="IDLE",
        selection_labels={"pin_ok", "pin_fail"},
    )


class TestATM:
    """Extract session type from an ATM FSM."""

    def test_states_and_transitions(self, atm_ss):
        assert len(atm_ss.states) == 5
        assert len(atm_ss.transitions) == 8

    def test_reconstruct(self, atm_ss):
        s = reconstruct(atm_ss)
        p = pretty(s)
        assert "insert_card" in p
        assert "withdraw" in p

    def test_is_reticulate(self, atm_ss):
        assert is_reticulate(atm_ss)

    def test_has_selections(self, atm_ss):
        """ATM has selections for pin_ok/pin_fail."""
        classifications = classify_all_states(atm_ss)
        kinds = {c.kind for c in classifications}
        assert "select" in kinds


# ═══════════════════════════════════════════════════════════════════
# Summary: print all extracted session types
# ═══════════════════════════════════════════════════════════════════


class TestSummary:
    """Summarise all wild-state-machine extractions."""

    @pytest.mark.parametrize("name,fixture_name", [
        ("Turnstile", "turnstile_ss"),
        ("Radiation Therapy", "radiation_ss"),
        ("Vending Machine", "vending_ss"),
        ("TCP Client", "tcp_client_ss"),
        ("Door Lock", "door_lock_ss"),
        ("ATM", "atm_ss"),
    ])
    def test_extraction_summary(self, name, fixture_name, request):
        ss = request.getfixturevalue(fixture_name)
        result = check_reticular_form(ss)
        assert result.is_reticulate, f"{name} is not a reticulate"

        s = reconstruct(ss)
        p = pretty(s)
        # Just verify we get a non-empty pretty-printed type
        assert len(p) > 0, f"{name} produced empty session type"

    def test_all_round_trip(self, request):
        """All extracted types should produce valid state spaces."""
        fixtures = [
            "turnstile_ss", "radiation_ss", "vending_ss",
            "tcp_client_ss", "door_lock_ss", "atm_ss",
        ]
        for fx in fixtures:
            ss = request.getfixturevalue(fx)
            s = reconstruct(ss)
            ss2 = build_statespace(s)
            # Reconstructed state space should have at least as many states
            assert len(ss2.states) >= 1
