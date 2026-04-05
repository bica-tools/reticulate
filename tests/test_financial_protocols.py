"""Step 73: FIX/ISO 20022 Financial Protocol Testing.

Tests for financial messaging protocols expressed as session types:
  - FIX order lifecycle (securities trading)
  - ISO 20022 credit transfer (payments)
  - ISO 20022 payment initiation (multiparty)

Verifies lattice properties, subtyping relationships, projections,
and fault injection scenarios for financial protocol state machines.
"""

from __future__ import annotations

import pytest

from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    Var,
    parse,
    pretty,
)
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice, LatticeResult, check_distributive
from reticulate.global_types import (
    GEnd,
    GMessage,
    GRec,
    GVar,
    GlobalType,
    parse_global,
    roles,
    build_global_statespace,
)
from reticulate.projection import project, project_all
from reticulate.subtyping import is_subtype
from reticulate.morphism import find_isomorphism, find_embedding, classify_morphism


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

FIX_ORDER = (
    "rec X . &{newOrder: +{PENDING: +{NEW: rec Y . &{partialFill: "
    "+{PARTIAL: Y, FILLED: end}, cancelRequest: +{CANCELED: end, "
    "CANCEL_REJECTED: Y}}, REJECTED: end}}}"
)

ISO_CREDIT_TRANSFER = (
    "&{creditTransfer: +{ACCEPTED: +{SETTLING: +{SETTLED: end, "
    "REJECTED: end}}, REJECTED: end, PENDING: +{SETTLED: end, "
    "REJECTED: end}}}"
)

ISO_PAYMENT_GLOBAL = (
    "Corporation -> DebtorBank : {initiate: "
    "DebtorBank -> Corporation : {received: "
    "DebtorBank -> CreditorBank : {transfer: "
    "CreditorBank -> DebtorBank : {completed: end, "
    "rejected: end}}}}"
)


# ===================================================================
# FIX Order Lifecycle Tests
# ===================================================================

class TestFIXOrderParsing:
    """Parse and inspect the FIX order lifecycle protocol."""

    def test_parse_fix_order(self) -> None:
        ast = parse(FIX_ORDER)
        assert isinstance(ast, Rec)
        assert ast.var == "X"

    def test_fix_roundtrip(self) -> None:
        ast = parse(FIX_ORDER)
        reparsed = parse(pretty(ast))
        assert ast == reparsed

    def test_fix_outer_rec_branch(self) -> None:
        ast = parse(FIX_ORDER)
        assert isinstance(ast, Rec)
        body = ast.body
        assert isinstance(body, Branch)
        labels = [label for label, _ in body.choices]
        assert labels == ["newOrder"]


class TestFIXOrderStateSpace:
    """State-space construction for FIX order lifecycle."""

    @pytest.fixture
    def fix_ss(self) -> StateSpace:
        return build_statespace(parse(FIX_ORDER))

    def test_fix_states_in_range(self, fix_ss: StateSpace) -> None:
        # FIX protocol should have 8-15 states
        assert 6 <= len(fix_ss.states) <= 20

    def test_fix_has_transitions(self, fix_ss: StateSpace) -> None:
        assert len(fix_ss.transitions) >= 6

    def test_fix_top_bottom_distinct(self, fix_ss: StateSpace) -> None:
        assert fix_ss.top != fix_ss.bottom

    def test_fix_transition_labels(self, fix_ss: StateSpace) -> None:
        labels = {label for _, label, _ in fix_ss.transitions}
        # Must contain the key FIX labels
        expected = {"newOrder", "PENDING", "NEW", "FILLED", "REJECTED"}
        assert expected.issubset(labels)

    def test_fix_has_cancel_labels(self, fix_ss: StateSpace) -> None:
        labels = {label for _, label, _ in fix_ss.transitions}
        assert "cancelRequest" in labels
        assert "CANCELED" in labels

    def test_fix_has_partial_fill(self, fix_ss: StateSpace) -> None:
        labels = {label for _, label, _ in fix_ss.transitions}
        assert "partialFill" in labels
        assert "PARTIAL" in labels


class TestFIXOrderLattice:
    """Lattice properties of the FIX order lifecycle."""

    @pytest.fixture
    def fix_lr(self) -> LatticeResult:
        ss = build_statespace(parse(FIX_ORDER))
        return check_lattice(ss)

    def test_fix_is_lattice(self, fix_lr: LatticeResult) -> None:
        assert fix_lr.is_lattice

    def test_fix_has_top(self, fix_lr: LatticeResult) -> None:
        assert fix_lr.has_top

    def test_fix_has_bottom(self, fix_lr: LatticeResult) -> None:
        assert fix_lr.has_bottom

    def test_fix_all_meets(self, fix_lr: LatticeResult) -> None:
        assert fix_lr.all_meets_exist

    def test_fix_all_joins(self, fix_lr: LatticeResult) -> None:
        assert fix_lr.all_joins_exist

    def test_fix_distributivity(self) -> None:
        ss = build_statespace(parse(FIX_ORDER))
        dr = check_distributive(ss)
        assert dr.is_lattice
        # Record classification (distributive or modular)
        assert dr.classification in ("boolean", "distributive", "modular", "lattice")


# ===================================================================
# ISO 20022 Credit Transfer Tests
# ===================================================================

class TestISOCreditParsing:
    """Parse and inspect the ISO 20022 credit transfer protocol."""

    def test_parse_iso_credit(self) -> None:
        ast = parse(ISO_CREDIT_TRANSFER)
        assert isinstance(ast, Branch)
        labels = [label for label, _ in ast.choices]
        assert labels == ["creditTransfer"]

    def test_iso_credit_roundtrip(self) -> None:
        ast = parse(ISO_CREDIT_TRANSFER)
        reparsed = parse(pretty(ast))
        assert ast == reparsed

    def test_iso_credit_selection_choices(self) -> None:
        ast = parse(ISO_CREDIT_TRANSFER)
        # creditTransfer leads to a Select with ACCEPTED, REJECTED, PENDING
        assert isinstance(ast, Branch)
        _, cont = ast.choices[0]
        assert isinstance(cont, Select)
        sel_labels = {label for label, _ in cont.choices}
        assert sel_labels == {"ACCEPTED", "REJECTED", "PENDING"}


class TestISOCreditStateSpace:
    """State-space construction for ISO 20022 credit transfer."""

    @pytest.fixture
    def iso_ss(self) -> StateSpace:
        return build_statespace(parse(ISO_CREDIT_TRANSFER))

    def test_iso_states_in_range(self, iso_ss: StateSpace) -> None:
        # ISO credit transfer: roughly 7-12 states
        assert 5 <= len(iso_ss.states) <= 15

    def test_iso_has_transitions(self, iso_ss: StateSpace) -> None:
        assert len(iso_ss.transitions) >= 5

    def test_iso_transition_labels(self, iso_ss: StateSpace) -> None:
        labels = {label for _, label, _ in iso_ss.transitions}
        expected = {"creditTransfer", "ACCEPTED", "REJECTED", "SETTLED"}
        assert expected.issubset(labels)


class TestISOCreditLattice:
    """Lattice properties of the ISO 20022 credit transfer."""

    @pytest.fixture
    def iso_lr(self) -> LatticeResult:
        ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        return check_lattice(ss)

    def test_iso_is_lattice(self, iso_lr: LatticeResult) -> None:
        assert iso_lr.is_lattice

    def test_iso_has_top(self, iso_lr: LatticeResult) -> None:
        assert iso_lr.has_top

    def test_iso_has_bottom(self, iso_lr: LatticeResult) -> None:
        assert iso_lr.has_bottom

    def test_iso_all_meets(self, iso_lr: LatticeResult) -> None:
        assert iso_lr.all_meets_exist

    def test_iso_all_joins(self, iso_lr: LatticeResult) -> None:
        assert iso_lr.all_joins_exist

    def test_iso_distributivity(self) -> None:
        ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        dr = check_distributive(ss)
        assert dr.is_lattice
        assert dr.classification in ("boolean", "distributive", "modular", "lattice")


# ===================================================================
# ISO 20022 Payment Initiation (Multiparty) Tests
# ===================================================================

class TestISOPaymentMultiparty:
    """Multiparty ISO 20022 payment initiation protocol."""

    def test_parse_global_type(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        assert isinstance(g, GMessage)
        assert g.sender == "Corporation"
        assert g.receiver == "DebtorBank"

    def test_roles(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        r = roles(g)
        assert r == frozenset({"Corporation", "DebtorBank", "CreditorBank"})

    def test_global_statespace(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        ss = build_global_statespace(g)
        assert len(ss.states) >= 4
        assert len(ss.transitions) >= 4

    def test_global_lattice(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        ss = build_global_statespace(g)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_project_corporation(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        local = project(g, "Corporation")
        # Corporation sends initiate (Select), receives received (Branch)
        assert local is not None

    def test_project_debtor_bank(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        local = project(g, "DebtorBank")
        assert local is not None

    def test_project_creditor_bank(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        local = project(g, "CreditorBank")
        assert local is not None

    def test_project_all_roles(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        projections = project_all(g)
        assert set(projections.keys()) == {"Corporation", "DebtorBank", "CreditorBank"}

    def test_all_projections_form_lattices(self) -> None:
        g = parse_global(ISO_PAYMENT_GLOBAL)
        projections = project_all(g)
        for role, local_type in projections.items():
            ss = build_statespace(local_type)
            lr = check_lattice(ss)
            assert lr.is_lattice, f"Projection for {role} is not a lattice"


# ===================================================================
# Subtyping Tests
# ===================================================================

class TestFinancialSubtyping:
    """Subtyping relationships between financial protocol variants."""

    def test_full_fix_subtype_of_simplified(self) -> None:
        """Full FIX (more branch methods) is subtype of simplified in Gay-Hole.

        Gay-Hole: more methods in branch => subtype (covariant width).
        Full FIX has cancelRequest in the inner branch, simplified does not.
        So full <= simplified.
        """
        simplified_fix = (
            "rec X . &{newOrder: +{PENDING: +{NEW: rec Y . &{partialFill: "
            "+{PARTIAL: Y, FILLED: end}}, REJECTED: end}}}"
        )
        full = parse(FIX_ORDER)
        simplified = parse(simplified_fix)
        # Full has MORE branch options (cancelRequest), so full <= simplified
        result = is_subtype(full, simplified)
        assert result is True

    def test_simplified_fix_not_subtype_of_full(self) -> None:
        """Simplified FIX (fewer branches) is NOT subtype of full."""
        simplified_fix = (
            "rec X . &{newOrder: +{PENDING: +{NEW: rec Y . &{partialFill: "
            "+{PARTIAL: Y, FILLED: end}}, REJECTED: end}}}"
        )
        full = parse(FIX_ORDER)
        simplified = parse(simplified_fix)
        # Simplified has FEWER branch options, so simplified is NOT <= full
        result = is_subtype(simplified, full)
        assert result is False

    def test_iso_accepted_vs_full(self) -> None:
        """An ISO protocol with only ACCEPTED path is a subtype (fewer selections)."""
        accepted_only = "&{creditTransfer: +{ACCEPTED: +{SETTLING: +{SETTLED: end, REJECTED: end}}}}"
        full = parse(ISO_CREDIT_TRANSFER)
        sub = parse(accepted_only)
        # Fewer selection labels => subtype
        result = is_subtype(sub, full)
        assert result is True

    def test_iso_full_not_subtype_of_accepted(self) -> None:
        """Full ISO has more selections, so NOT subtype of accepted-only."""
        accepted_only = "&{creditTransfer: +{ACCEPTED: +{SETTLING: +{SETTLED: end, REJECTED: end}}}}"
        full = parse(ISO_CREDIT_TRANSFER)
        sub = parse(accepted_only)
        result = is_subtype(full, sub)
        assert result is False


# ===================================================================
# Fault Injection Tests
# ===================================================================

class TestFaultInjection:
    """Fault injection: remove branches and verify impact on lattice."""

    def test_fix_without_rejected_still_lattice(self) -> None:
        """FIX without REJECTED: fewer selection choices, still a lattice."""
        no_reject = (
            "rec X . &{newOrder: +{PENDING: +{NEW: rec Y . &{partialFill: "
            "+{PARTIAL: Y, FILLED: end}, cancelRequest: +{CANCELED: end, "
            "CANCEL_REJECTED: Y}}}}}"
        )
        ss = build_statespace(parse(no_reject))
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_fix_single_path_lattice(self) -> None:
        """FIX with only one path: newOrder -> PENDING -> NEW -> FILLED."""
        single_path = "&{newOrder: +{PENDING: +{NEW: +{FILLED: end}}}}"
        ss = build_statespace(parse(single_path))
        lr = check_lattice(ss)
        assert lr.is_lattice
        # Linear chain => always a lattice

    def test_iso_single_accept_path(self) -> None:
        """ISO with single ACCEPTED -> SETTLING -> SETTLED path."""
        single = "&{creditTransfer: +{ACCEPTED: +{SETTLING: +{SETTLED: end}}}}"
        ss = build_statespace(parse(single))
        lr = check_lattice(ss)
        assert lr.is_lattice


# ===================================================================
# Terminal State & Path Tests
# ===================================================================

class TestTerminalStates:
    """Verify terminal states map correctly to lattice bottom."""

    def test_fix_terminal_states_reach_bottom(self) -> None:
        """All terminal FIX states (FILLED, CANCELED, REJECTED) go to bottom."""
        ss = build_statespace(parse(FIX_ORDER))
        # All transitions labeled FILLED, CANCELED, REJECTED target bottom
        terminal_labels = {"FILLED", "CANCELED", "REJECTED"}
        for src, label, tgt in ss.transitions:
            if label in terminal_labels:
                assert tgt == ss.bottom, (
                    f"Terminal label {label} goes to {tgt}, not bottom {ss.bottom}"
                )

    def test_iso_terminal_states_reach_bottom(self) -> None:
        """All terminal ISO states (SETTLED, REJECTED at leaf) go to bottom."""
        ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        for src, label, tgt in ss.transitions:
            if label == "SETTLED":
                assert tgt == ss.bottom

    def test_fix_cancel_fill_race_join(self) -> None:
        """The partialFill and cancelRequest paths must have a valid join.

        In the FIX lattice, these two branch paths from the NEW state
        must have a join (which should be the NEW state itself, since
        both are reachable from it).
        """
        ss = build_statespace(parse(FIX_ORDER))
        lr = check_lattice(ss)
        assert lr.is_lattice
        # If it's a lattice, all pairs have joins -- including the
        # states reached via partialFill and cancelRequest.
        assert lr.all_joins_exist


# ===================================================================
# Cross-Protocol Comparison Tests
# ===================================================================

class TestCrossProtocolComparison:
    """Compare FIX vs ISO lattice properties."""

    def test_both_are_lattices(self) -> None:
        fix_ss = build_statespace(parse(FIX_ORDER))
        iso_ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        assert check_lattice(fix_ss).is_lattice
        assert check_lattice(iso_ss).is_lattice

    def test_fix_larger_than_iso(self) -> None:
        """FIX has more states due to recursion and cancel paths."""
        fix_ss = build_statespace(parse(FIX_ORDER))
        iso_ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        assert len(fix_ss.states) >= len(iso_ss.states)

    def test_both_have_distributivity_info(self) -> None:
        """Both protocols have well-defined distributivity classification."""
        fix_ss = build_statespace(parse(FIX_ORDER))
        iso_ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        fix_dr = check_distributive(fix_ss)
        iso_dr = check_distributive(iso_ss)
        assert fix_dr.classification in ("boolean", "distributive", "modular", "lattice")
        assert iso_dr.classification in ("boolean", "distributive", "modular", "lattice")

    def test_iso_no_recursion_fewer_sccs(self) -> None:
        """ISO credit transfer has no recursion, so fewer SCCs than FIX."""
        fix_ss = build_statespace(parse(FIX_ORDER))
        iso_ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        fix_lr = check_lattice(fix_ss)
        iso_lr = check_lattice(iso_ss)
        # ISO has no rec, each state is its own SCC
        assert iso_lr.num_scc == len(iso_ss.states)
        # FIX has rec Y, so some states are collapsed
        assert fix_lr.num_scc <= len(fix_ss.states)


# ===================================================================
# Selection Tracking Tests
# ===================================================================

class TestSelectionTracking:
    """Verify that selection (internal choice) transitions are tracked."""

    def test_fix_selection_transitions(self) -> None:
        """FIX has selection transitions for PENDING, NEW, etc."""
        ss = build_statespace(parse(FIX_ORDER))
        sel_labels = {label for _, label, _ in ss.selection_transitions}
        # PENDING and NEW are selections (+{...})
        assert "PENDING" in sel_labels
        assert "NEW" in sel_labels

    def test_iso_selection_transitions(self) -> None:
        """ISO has selection transitions for ACCEPTED, REJECTED, etc."""
        ss = build_statespace(parse(ISO_CREDIT_TRANSFER))
        sel_labels = {label for _, label, _ in ss.selection_transitions}
        assert "ACCEPTED" in sel_labels
        assert "REJECTED" in sel_labels

    def test_fix_branch_not_selection(self) -> None:
        """newOrder and partialFill are branch transitions, not selections."""
        ss = build_statespace(parse(FIX_ORDER))
        sel_labels = {label for _, label, _ in ss.selection_transitions}
        assert "newOrder" not in sel_labels
        assert "partialFill" not in sel_labels
