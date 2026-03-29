"""Tests for synchronous parallel composition (Step 5d).

Tests cover:
- Basic synchronous product construction
- Comparison with interleaving product
- Lattice preservation
- Sublattice property (sync ⊆ interleaving)
- Musical chord encoding
- Chain property for equal-length sequences
- N-ary synchronous products
- Edge cases (single component, mismatched lengths)
- Compression ratio analysis
- Benchmark protocol analysis
"""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.sync_product import (
    sync_product,
    analyze_sync_product,
    SyncProductResult,
)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestSyncProductBasic:
    """Test basic synchronous product construction."""

    def test_single_component(self):
        """Single component returns itself."""
        ss = build_statespace(parse("&{a: end}"))
        result = sync_product(ss)
        assert len(result.states) == len(ss.states)

    def test_two_identical_chains(self):
        """Two identical 3-state chains produce a 3-state sync product."""
        ss1 = build_statespace(parse("&{a: &{b: end}}"))
        ss2 = build_statespace(parse("&{c: &{d: end}}"))
        result = sync_product(ss1, ss2)
        assert len(result.states) == 3
        assert len(result.transitions) == 2

    def test_labels_are_combined(self):
        """Transition labels combine component labels with '+'."""
        ss1 = build_statespace(parse("&{a: end}"))
        ss2 = build_statespace(parse("&{b: end}"))
        result = sync_product(ss1, ss2)
        labels = {l for _, l, _ in result.transitions}
        assert "a+b" in labels

    def test_top_and_bottom(self):
        """Top and bottom are correct."""
        ss1 = build_statespace(parse("&{a: end}"))
        ss2 = build_statespace(parse("&{b: end}"))
        result = sync_product(ss1, ss2)
        assert result.top in result.states
        assert result.bottom in result.states
        assert result.top != result.bottom


# ---------------------------------------------------------------------------
# Lattice preservation
# ---------------------------------------------------------------------------

class TestSyncProductLattice:
    """Test that synchronous products preserve lattice property."""

    def test_two_chains_lattice(self):
        """Sync product of two chains is a lattice."""
        ss1 = build_statespace(parse("&{a: &{b: end}}"))
        ss2 = build_statespace(parse("&{c: &{d: end}}"))
        result = sync_product(ss1, ss2)
        lr = check_lattice(result)
        assert lr.is_lattice

    def test_branch_types_lattice(self):
        """Sync product of branch types is a lattice."""
        ss1 = build_statespace(parse("&{a: end, b: end}"))
        ss2 = build_statespace(parse("&{c: end, d: end}"))
        result = sync_product(ss1, ss2)
        lr = check_lattice(result)
        assert lr.is_lattice

    def test_three_way_lattice(self):
        """3-way sync product is a lattice."""
        ss1 = build_statespace(parse("&{a: end}"))
        ss2 = build_statespace(parse("&{b: end}"))
        ss3 = build_statespace(parse("&{c: end}"))
        result = sync_product(ss1, ss2, ss3)
        lr = check_lattice(result)
        assert lr.is_lattice

    def test_musical_notes_lattice(self):
        """Sync product with musical note labels is a lattice."""
        ss1 = build_statespace(parse("&{G5e: &{G5e: &{G5e: &{Ef5h: end}}}}"))
        ss2 = build_statespace(parse("&{G3e: &{G3e: &{G3e: &{Ef3h: end}}}}"))
        result = sync_product(ss1, ss2)
        lr = check_lattice(result)
        assert lr.is_lattice
        assert len(result.states) == 5  # Same length chains -> chain

    def test_recursive_types_may_not_be_lattice(self):
        """Sync product of recursive types with selection: NOT always a lattice.

        When both components have selection (internal choice), the sync
        product creates diverging paths that may not have meets.
        This is a genuine theoretical finding: synchronous parallel
        does NOT universally preserve lattice structure.
        """
        ss1 = build_statespace(parse("rec X . &{a: X, b: end}"))
        ss2 = build_statespace(parse("rec Y . &{c: Y, d: end}"))
        result = sync_product(ss1, ss2)
        lr = check_lattice(result)
        # NOT a lattice: diverging selection paths lack meets
        assert not lr.is_lattice


# ---------------------------------------------------------------------------
# Comparison with interleaving
# ---------------------------------------------------------------------------

class TestSyncVsInterleaving:
    """Compare synchronous and interleaving products."""

    def test_sync_fewer_states(self):
        """Sync product always has <= states than interleaving."""
        ss1 = build_statespace(parse("&{a: &{b: end}}"))
        ss2 = build_statespace(parse("&{c: &{d: end}}"))
        sync = sync_product(ss1, ss2)

        inter_type = "(&{a: &{b: end}} || &{c: &{d: end}})"
        inter = build_statespace(parse(inter_type))

        assert len(sync.states) <= len(inter.states)

    def test_compression_ratio(self):
        """Compression ratio for equal chains is 1/n (n = chain length)."""
        ss1 = build_statespace(parse("&{a: &{b: &{c: end}}}"))
        ss2 = build_statespace(parse("&{d: &{e: &{f: end}}}"))
        result = analyze_sync_product(ss1, ss2)
        assert result.compression_ratio < 1.0
        assert result.sync_size == 4  # Chain of length 4
        assert result.interleaving_size == 16  # 4 * 4

    def test_beethoven_quartet_compression(self):
        """Beethoven 4-voice fate motif: 625 -> 5 states."""
        vln1 = build_statespace(parse("&{G5e: &{G5e: &{G5e: &{Ef5h: end}}}}"))
        vln2 = build_statespace(parse("&{G4e: &{G4e: &{G4e: &{Ef4h: end}}}}"))
        vla  = build_statespace(parse("&{G4e: &{G4e: &{G4e: &{Ef4h: end}}}}"))
        vc   = build_statespace(parse("&{G3e: &{G3e: &{G3e: &{Ef3h: end}}}}"))

        result = analyze_sync_product(vln1, vln2, vla, vc)
        assert result.sync_size == 5
        assert result.interleaving_size == 625
        assert result.is_chain


# ---------------------------------------------------------------------------
# Chain property
# ---------------------------------------------------------------------------

class TestChainProperty:
    """Equal-length chain components produce a chain sync product."""

    def test_equal_chains_produce_chain(self):
        """Two equal-length chains produce a chain."""
        ss1 = build_statespace(parse("&{a: &{b: end}}"))
        ss2 = build_statespace(parse("&{c: &{d: end}}"))
        result = analyze_sync_product(ss1, ss2)
        assert result.is_chain

    def test_unequal_chains_includes_unreachable_bottom(self):
        """Unequal chains: sync product has reachable + bottom states."""
        ss1 = build_statespace(parse("&{a: &{b: &{c: end}}}"))  # 4 states
        ss2 = build_statespace(parse("&{d: end}"))  # 2 states
        result = sync_product(ss1, ss2)
        # 1 sync step + top + bottom (bottom may be unreachable)
        # Reachable: top -> (a+d) -> state. Then ss2 has no more transitions.
        assert len(result.states) >= 2


# ---------------------------------------------------------------------------
# Musical applications
# ---------------------------------------------------------------------------

class TestMusicalApplications:
    """Test musical score encoding via synchronous parallel."""

    def test_chord_labels(self):
        """Two simultaneous notes produce a chord label."""
        ss1 = build_statespace(parse("&{C4q: end}"))
        ss2 = build_statespace(parse("&{E4q: end}"))
        result = sync_product(ss1, ss2)
        labels = {l for _, l, _ in result.transitions}
        assert "C4q+E4q" in labels  # C major third!

    def test_triad_chord(self):
        """Three notes = triad chord."""
        ss_c = build_statespace(parse("&{C4q: end}"))
        ss_e = build_statespace(parse("&{E4q: end}"))
        ss_g = build_statespace(parse("&{G4q: end}"))
        result = sync_product(ss_c, ss_e, ss_g)
        labels = {l for _, l, _ in result.transitions}
        assert "C4q+E4q+G4q" in labels  # C major triad!

    def test_bach_subject_two_voices(self):
        """Bach BWV 772: two voices playing simultaneously."""
        rh = build_statespace(parse(
            "&{C5x: &{D5x: &{E5x: &{F5x: end}}}}"
        ))
        lh = build_statespace(parse(
            "&{C4x: &{D4x: &{E4x: &{F4x: end}}}}"
        ))
        result = sync_product(rh, lh)
        assert len(result.states) == 5
        labels = sorted(l for _, l, _ in result.transitions)
        assert labels[0] == "C5x+C4x"  # Both start on C

    def test_four_voice_chorale(self):
        """Four-voice chorale (SATB): all move together."""
        s = build_statespace(parse("&{C5q: &{D5q: end}}"))
        a = build_statespace(parse("&{A4q: &{B4q: end}}"))
        t = build_statespace(parse("&{F4q: &{G4q: end}}"))
        b = build_statespace(parse("&{C4q: &{D4q: end}}"))
        result = sync_product(s, a, t, b)
        assert len(result.states) == 3
        labels = {l for _, l, _ in result.transitions}
        assert "C5q+A4q+F4q+C4q" in labels  # F major chord!


# ---------------------------------------------------------------------------
# Branching in synchronous products
# ---------------------------------------------------------------------------

class TestSyncBranching:
    """Sync product with branching types."""

    def test_both_branch_same_arity(self):
        """Two branch types with same arity: all combinations."""
        ss1 = build_statespace(parse("&{a: end, b: end}"))
        ss2 = build_statespace(parse("&{c: end, d: end}"))
        result = sync_product(ss1, ss2)
        labels = {l for _, l, _ in result.transitions}
        # 2x2 = 4 possible simultaneous choices
        assert len(labels) == 4
        assert "a+c" in labels
        assert "a+d" in labels
        assert "b+c" in labels
        assert "b+d" in labels

    def test_branch_sync_mismatched_depth(self):
        """Sync of branch types with mismatched depths: may not be lattice.

        When branches have different continuation depths, synchronous
        stepping creates divergent paths that can lack meets.
        """
        ss1 = build_statespace(parse("&{a: &{x: end}, b: &{y: end}}"))
        ss2 = build_statespace(parse("&{c: end, d: end}"))
        result = sync_product(ss1, ss2)
        # The sync product exists but may not be a lattice
        assert len(result.states) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for synchronous product."""

    def test_single_state_end(self):
        """Sync of two 'end' types: single state."""
        ss1 = build_statespace(parse("end"))
        ss2 = build_statespace(parse("end"))
        result = sync_product(ss1, ss2)
        assert len(result.states) == 1

    def test_one_component_no_transitions(self):
        """If one component is 'end', sync product is just 'end'."""
        ss1 = build_statespace(parse("&{a: &{b: end}}"))
        ss2 = build_statespace(parse("end"))
        result = sync_product(ss1, ss2)
        # ss2 has no transitions, so no sync transitions possible
        assert len(result.transitions) == 0

    def test_empty_raises(self):
        """No components raises ValueError."""
        with pytest.raises(ValueError):
            sync_product()


# ---------------------------------------------------------------------------
# Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Synchronous products of benchmark protocols."""

    def test_iterator_sync(self):
        """Java Iterator synced with itself: selection divergence breaks lattice.

        Two iterators stepping in sync: when both reach hasNext,
        the 4 selection combinations (TT, TF, FT, FF) create
        divergent paths that lack meets. This is correct behavior:
        independent selection choices are fundamentally incompatible
        with lock-step synchronization.
        """
        iter_type = "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        ss = build_statespace(parse(iter_type))
        result = analyze_sync_product(ss, ss)
        # Selection creates non-lattice sync product
        assert not check_lattice(result.state_space).is_lattice
        assert result.sync_size <= result.interleaving_size

    def test_file_sync(self):
        """File protocol synced with itself (copy operation)."""
        file_type = "&{open: &{read: &{close: end}, write: &{close: end}}}"
        ss = build_statespace(parse(file_type))
        result = analyze_sync_product(ss, ss)
        assert check_lattice(result.state_space).is_lattice
