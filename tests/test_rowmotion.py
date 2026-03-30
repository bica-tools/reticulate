"""Tests for rowmotion and toggling analysis (Step 30ad).

Comprehensive tests covering:
  - Order relation computation (6 tests)
  - Covering relation / transitive reduction (5 tests)
  - Order ideal enumeration (7 tests)
  - Antichain enumeration (5 tests)
  - Antichain <-> ideal bijection (5 tests)
  - Toggle operations (6 tests)
  - Toggle sequence / linear extension (3 tests)
  - Rowmotion on ideals (8 tests)
  - Rowmotion on antichains (3 tests)
  - Rowmotion via toggle decomposition (4 tests)
  - Orbit decomposition (5 tests)
  - Rowmotion order (5 tests)
  - Homomesy checking (4 tests)
  - Full analysis (6 tests)
  - Benchmark protocols (5 tests)
  - Mathematical properties (5 tests)

Total: 82 tests.

Poset ordering: s1 >= s2 iff s2 is reachable from s1.
Top = initial state (greatest), bottom = end state (least).
Order ideals are downsets: if x in I and y <= x then y in I.
"""

from __future__ import annotations

import math

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.rowmotion import (
    RowmotionResult,
    all_orbits,
    analyze_rowmotion,
    antichain_to_ideal,
    antichains,
    check_homomesy,
    compute_covers,
    compute_order_relation,
    ideal_to_antichain,
    order_ideals,
    rowmotion_antichain,
    rowmotion_ideal,
    rowmotion_orbit,
    rowmotion_order,
    rowmotion_via_toggles,
    toggle,
    toggle_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(s: str) -> StateSpace:
    """Parse a session type string and build its state space."""
    return build_statespace(parse(s))


def _chain(n: int) -> StateSpace:
    """Build a chain of n elements: 0 > 1 > ... > n-1.

    top = 0, bottom = n-1.
    """
    ss = StateSpace()
    ss.states = set(range(n))
    ss.transitions = [(i, f"t{i}", i + 1) for i in range(n - 1)]
    ss.top = 0
    ss.bottom = n - 1
    ss.labels = {i: f"s{i}" for i in range(n)}
    return ss


def _boolean_lattice_2() -> StateSpace:
    """Boolean lattice 2^2 = diamond: top > a, b > bot.

    States: 0=top, 1=a, 2=b, 3=bot.
    """
    ss = StateSpace()
    ss.states = {0, 1, 2, 3}
    ss.transitions = [
        (0, "a", 1), (0, "b", 2),
        (1, "x", 3), (2, "y", 3),
    ]
    ss.top = 0
    ss.bottom = 3
    ss.labels = {0: "top", 1: "a", 2: "b", 3: "bot"}
    return ss


def _is_downset(ideal: frozenset[int], order: dict[int, set[int]]) -> bool:
    """Check that ideal is a valid order ideal (downset).

    order[x] gives all elements at or below x (including x).
    For I to be a downset: if x in I, then order[x] subset of I.
    """
    for x in ideal:
        if x in order and not order[x].issubset(ideal):
            return False
    return True


def _is_antichain(ac: frozenset[int], order: dict[int, set[int]]) -> bool:
    """Check that no two elements in ac are comparable."""
    elems = list(ac)
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            a, b = elems[i], elems[j]
            if b in order.get(a, set()) or a in order.get(b, set()):
                return False
    return True


# =========================================================================
# TestOrderRelation
# =========================================================================

class TestOrderRelation:
    """Tests for compute_order_relation."""

    def test_end_single_state(self):
        """end: single state, reflexively related to itself."""
        ss = _build("end")
        order = compute_order_relation(ss)
        assert len(order) == 1
        for s, below in order.items():
            assert s in below  # reflexive

    def test_branch_two_states(self):
        """&{a: end}: top > bottom, two states."""
        ss = _build("&{a: end}")
        order = compute_order_relation(ss)
        assert len(order) == 2
        # Find top (the one that reaches more elements)
        top_rep = max(order, key=lambda s: len(order[s]))
        bot_rep = min(order, key=lambda s: len(order[s]))
        assert bot_rep in order[top_rep]  # top >= bottom
        assert top_rep not in order[bot_rep]  # bottom !>= top

    def test_branch_shared_end(self):
        """&{a: end, b: end}: both labels go to same end, so 2-element poset."""
        ss = _build("&{a: end, b: end}")
        order = compute_order_relation(ss)
        # top reaches bottom
        top_rep = max(order, key=lambda s: len(order[s]))
        assert len(order[top_rep]) == len(order)

    def test_chain_three_hand_built(self):
        """Hand-built 3-chain: 0 > 1 > 2."""
        ss = _chain(3)
        order = compute_order_relation(ss)
        assert order[0] == {0, 1, 2}  # top reaches everything
        assert order[1] == {1, 2}     # mid reaches itself and bottom
        assert order[2] == {2}        # bottom only itself

    def test_diamond_incomparability(self):
        """Boolean lattice: a and b are incomparable."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        # a (state 1) and b (state 2) should not be comparable
        assert 2 not in order[1]  # a does not reach b
        assert 1 not in order[2]  # b does not reach a

    def test_order_is_reflexive(self):
        """Every element is in its own downset."""
        for type_str in ["end", "&{a: end}", "&{a: &{b: end}}"]:
            ss = _build(type_str)
            order = compute_order_relation(ss)
            for s in order:
                assert s in order[s], f"reflexivity failed for {s}"


# =========================================================================
# TestCovers
# =========================================================================

class TestCovers:
    """Tests for compute_covers (Hasse diagram / transitive reduction)."""

    def test_chain_covers_successor_only(self):
        """In a 4-chain, each element covers only its immediate successor."""
        ss = _chain(4)
        covers = compute_covers(ss)
        assert covers[0] == {1}
        assert covers[1] == {2}
        assert covers[2] == {3}
        assert covers[3] == set()

    def test_chain_no_transitive_covers(self):
        """In a chain, there are no skip covers (transitive reduction)."""
        ss = _chain(4)
        covers = compute_covers(ss)
        assert 2 not in covers[0]  # 0 does not cover 2
        assert 3 not in covers[0]  # 0 does not cover 3

    def test_diamond_top_covers_middle(self):
        """Diamond: top covers a and b; a,b cover bot."""
        ss = _boolean_lattice_2()
        covers = compute_covers(ss)
        assert covers[0] == {1, 2}  # top covers a and b
        assert covers[1] == {3}     # a covers bot
        assert covers[2] == {3}     # b covers bot
        assert 3 not in covers[0]   # top does NOT cover bot

    def test_end_no_covers(self):
        """Single state: no covering relations."""
        ss = _build("end")
        covers = compute_covers(ss)
        for s in covers:
            assert len(covers[s]) == 0

    def test_total_covers_diamond(self):
        """Diamond has exactly 4 covering pairs."""
        ss = _boolean_lattice_2()
        covers = compute_covers(ss)
        total = sum(len(c) for c in covers.values())
        assert total == 4


# =========================================================================
# TestOrderIdeals
# =========================================================================

class TestOrderIdeals:
    """Tests for order ideal enumeration."""

    def test_single_state_two_ideals(self):
        """Single element poset: 2 ideals (empty and full)."""
        ss = _build("end")
        ideals = order_ideals(ss)
        assert len(ideals) == 2
        assert frozenset() in ideals

    def test_chain_2_three_ideals(self):
        """2-chain has n+1 = 3 ideals: {}, {bot}, {bot, top}."""
        ss = _chain(2)
        ideals = order_ideals(ss)
        assert len(ideals) == 3

    def test_chain_3_four_ideals(self):
        """3-chain has n+1 = 4 ideals."""
        ss = _chain(3)
        ideals = order_ideals(ss)
        assert len(ideals) == 4

    def test_chain_4_five_ideals(self):
        """4-chain has n+1 = 5 ideals."""
        ss = _chain(4)
        ideals = order_ideals(ss)
        assert len(ideals) == 5

    def test_diamond_six_ideals(self):
        """Boolean lattice 2^2 has 6 order ideals."""
        ss = _boolean_lattice_2()
        ideals = order_ideals(ss)
        assert len(ideals) == 6

    def test_every_ideal_is_valid_downset(self):
        """Every enumerated ideal must satisfy the downset property."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        for ideal in order_ideals(ss):
            assert _is_downset(ideal, order), f"{ideal} is not a valid downset"

    def test_empty_and_full_always_present(self):
        """Empty set and full poset are always order ideals."""
        for ss in [_chain(2), _chain(3), _boolean_lattice_2()]:
            ideals = order_ideals(ss)
            order = compute_order_relation(ss)
            assert frozenset() in ideals
            full = frozenset(order.keys())
            assert full in ideals


# =========================================================================
# TestAntichains
# =========================================================================

class TestAntichains:
    """Tests for antichain enumeration."""

    def test_single_state_two_antichains(self):
        """Single state: 2 antichains (empty and singleton)."""
        ss = _build("end")
        acs = antichains(ss)
        assert len(acs) == 2

    def test_chain_antichains_count(self):
        """n-chain: n+1 antichains (empty + n singletons)."""
        ss = _chain(3)
        acs = antichains(ss)
        # Antichains of a chain: empty, {0}, {1}, {2} = 4 = n+1
        assert len(acs) == 4

    def test_diamond_antichains_count(self):
        """Diamond has 6 antichains (same as ideals)."""
        ss = _boolean_lattice_2()
        acs = antichains(ss)
        assert len(acs) == 6

    def test_every_antichain_is_valid(self):
        """No two elements in any antichain are comparable."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        for ac in antichains(ss):
            assert _is_antichain(ac, order), f"{ac} is not a valid antichain"

    def test_count_equals_ideals_count(self):
        """Fundamental bijection: #antichains = #order ideals."""
        for ss in [_build("end"), _chain(2), _chain(3), _boolean_lattice_2()]:
            assert len(antichains(ss)) == len(order_ideals(ss))


# =========================================================================
# TestBijection
# =========================================================================

class TestBijection:
    """Tests for antichain <-> order ideal bijection."""

    def test_roundtrip_ideal_to_antichain_to_ideal(self):
        """For every ideal, ideal -> antichain -> ideal is identity."""
        ss = _boolean_lattice_2()
        for ideal in order_ideals(ss):
            ac = ideal_to_antichain(ss, ideal)
            recovered = antichain_to_ideal(ss, ac)
            assert recovered == ideal, \
                f"Round-trip failed: {ideal} -> {ac} -> {recovered}"

    def test_roundtrip_antichain_to_ideal_to_antichain(self):
        """For every antichain, antichain -> ideal -> antichain is identity."""
        ss = _boolean_lattice_2()
        for ac in antichains(ss):
            ideal = antichain_to_ideal(ss, ac)
            recovered = ideal_to_antichain(ss, ideal)
            assert recovered == ac, \
                f"Round-trip failed: {ac} -> {ideal} -> {recovered}"

    def test_empty_antichain_gives_empty_ideal(self):
        """Empty antichain corresponds to empty ideal."""
        ss = _build("&{a: end}")
        assert antichain_to_ideal(ss, frozenset()) == frozenset()

    def test_empty_ideal_gives_empty_antichain(self):
        """Empty ideal has no maximal elements."""
        ss = _build("&{a: end}")
        assert ideal_to_antichain(ss, frozenset()) == frozenset()

    def test_roundtrip_on_chain(self):
        """Round-trip on a 3-chain."""
        ss = _chain(3)
        for ideal in order_ideals(ss):
            ac = ideal_to_antichain(ss, ideal)
            assert antichain_to_ideal(ss, ac) == ideal


# =========================================================================
# TestToggle
# =========================================================================

class TestToggle:
    """Tests for toggle operations."""

    def test_toggle_add_bottom_to_empty(self):
        """Toggle bottom into empty ideal: adds it (no predecessors)."""
        ss = _chain(2)
        order = compute_order_relation(ss)
        # Bottom is the element with only itself in its downset
        bot = min(order, key=lambda s: len(order[s]))
        result = toggle(ss, frozenset(), bot)
        assert bot in result

    def test_toggle_remove_maximal(self):
        """Toggle removes a maximal element from an ideal."""
        ss = _chain(2)
        seq = toggle_sequence(ss)
        bot = seq[0]
        result = toggle(ss, frozenset({bot}), bot)
        assert bot not in result

    def test_double_toggle_identity(self):
        """Toggling the same element twice returns to original."""
        ss = _boolean_lattice_2()
        seq = toggle_sequence(ss)
        bot = seq[0]
        ideal = frozenset()
        step1 = toggle(ss, ideal, bot)
        step2 = toggle(ss, step1, bot)
        assert step2 == ideal

    def test_toggle_preserves_downset_property(self):
        """After any toggle, the result is still a valid order ideal."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        ideals_set = set(order_ideals(ss))
        for ideal in order_ideals(ss):
            for x in order:
                result = toggle(ss, ideal, x)
                assert result in ideals_set, \
                    f"toggle({ideal}, {x}) = {result} not in ideals"

    def test_toggle_cannot_add_without_predecessors(self):
        """Cannot add element whose predecessors are not in ideal."""
        ss = _boolean_lattice_2()
        # Top (0) has elements below it; cannot add to empty ideal
        result = toggle(ss, frozenset(), 0)
        assert 0 not in result

    def test_toggle_cannot_remove_needed_element(self):
        """Cannot remove element that supports another element in ideal."""
        ss = _boolean_lattice_2()
        # Ideal {3, 1} = {bot, a}: removing bot would leave {a} which
        # is not a downset since a covers bot.
        ideal = frozenset({3, 1})
        result = toggle(ss, ideal, 3)
        assert 3 in result  # cannot remove bot


# =========================================================================
# TestToggleSequence
# =========================================================================

class TestToggleSequence:
    """Tests for toggle_sequence (linear extension)."""

    def test_all_elements_present(self):
        """Toggle sequence contains all poset elements exactly once."""
        ss = _boolean_lattice_2()
        seq = toggle_sequence(ss)
        order = compute_order_relation(ss)
        assert set(seq) == set(order.keys())
        assert len(seq) == len(set(seq))  # no duplicates

    def test_linear_extension_order(self):
        """Toggle sequence is a valid linear extension (top to bottom)."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        seq = toggle_sequence(ss)
        pos = {s: i for i, s in enumerate(seq)}
        # The module returns top-to-bottom order: if x > y, x appears before y
        for y in order:
            for x in order[y]:
                if x != y:
                    # x < y (x reachable from y), so y appears before x
                    assert pos[y] < pos[x], \
                        f"Linear extension violated: {y} (greater) should precede {x} (lesser)"

    def test_top_first_bottom_last(self):
        """Top has highest rank (first), bottom has lowest (last)."""
        ss = _boolean_lattice_2()
        seq = toggle_sequence(ss)
        # Top = 0 (greatest element), bottom = 3 (least element)
        assert seq[0] == 0   # top first
        assert seq[-1] == 3  # bottom last


# =========================================================================
# TestRowmotionIdeal
# =========================================================================

class TestRowmotionIdeal:
    """Tests for rowmotion_ideal."""

    def test_empty_ideal_maps_to_downset_of_min(self):
        """Row({}) = downset(min(P)) = downset({bottom}) = {bottom}."""
        ss = _chain(3)
        result = rowmotion_ideal(ss, frozenset())
        assert ss.bottom in result

    def test_full_ideal_maps_to_empty(self):
        """Row(P) = downset(min(empty)) = empty set."""
        ss = _chain(3)
        order = compute_order_relation(ss)
        full = frozenset(order.keys())
        result = rowmotion_ideal(ss, full)
        assert result == frozenset()

    def test_rowmotion_is_bijection_chain(self):
        """Rowmotion is a bijection on order ideals of a chain."""
        ss = _chain(3)
        ideals_list = order_ideals(ss)
        results = [rowmotion_ideal(ss, i) for i in ideals_list]
        assert len(set(results)) == len(ideals_list)

    def test_rowmotion_is_bijection_diamond(self):
        """Rowmotion is a bijection on order ideals of the diamond."""
        ss = _boolean_lattice_2()
        ideals_list = order_ideals(ss)
        results = [rowmotion_ideal(ss, i) for i in ideals_list]
        # All results are valid ideals
        ideals_set = set(ideals_list)
        for r in results:
            assert r in ideals_set, f"{r} is not a valid ideal"
        # All results are distinct
        assert len(set(results)) == len(results)

    def test_rowmotion_result_is_always_ideal(self):
        """Rowmotion always produces a valid order ideal."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        for ideal in order_ideals(ss):
            result = rowmotion_ideal(ss, ideal)
            assert _is_downset(result, order), \
                f"rowmotion({ideal}) = {result} is not a downset"

    def test_rowmotion_on_session_type(self):
        """Rowmotion is a bijection on a session type state space."""
        ss = _build("&{a: end}")
        ideals_list = order_ideals(ss)
        results = [rowmotion_ideal(ss, i) for i in ideals_list]
        assert len(set(results)) == len(results)

    def test_chain_2_cycle(self):
        """2-chain: rowmotion cycles {} -> {bot} -> {bot,top} -> {}."""
        ss = _chain(2)
        order = compute_order_relation(ss)
        reps = sorted(order.keys())
        bot, top = reps[0] if len(order[reps[0]]) == 1 else reps[1], \
                   reps[0] if len(order[reps[0]]) > 1 else reps[1]
        # Check cycle has length 3
        start = frozenset()
        current = start
        cycle_len = 0
        while True:
            current = rowmotion_ideal(ss, current)
            cycle_len += 1
            if current == start:
                break
            assert cycle_len < 10, "Cycle too long"
        assert cycle_len == 3

    def test_diamond_full_cycle(self):
        """Diamond: verify complete rowmotion cycle structure."""
        ss = _boolean_lattice_2()
        ideals_list = order_ideals(ss)
        # Track each ideal through one full orbit
        for ideal in ideals_list:
            orbit = rowmotion_orbit(ss, ideal)
            # orbit starts with ideal and ends just before returning
            assert orbit[0] == ideal
            # Applying rowmotion to last element returns to start
            assert rowmotion_ideal(ss, orbit[-1]) == ideal


# =========================================================================
# TestRowmotionAntichain
# =========================================================================

class TestRowmotionAntichain:
    """Tests for rowmotion_antichain."""

    def test_empty_antichain(self):
        """Rowmotion on empty antichain."""
        ss = _build("&{a: end}")
        result = rowmotion_antichain(ss, frozenset())
        assert isinstance(result, frozenset)

    def test_consistent_with_ideal_rowmotion(self):
        """Antichain rowmotion agrees with ideal rowmotion via conversion."""
        ss = _boolean_lattice_2()
        for ideal in order_ideals(ss):
            ac = ideal_to_antichain(ss, ideal)
            # Via ideals
            new_ideal = rowmotion_ideal(ss, ideal)
            expected_ac = ideal_to_antichain(ss, new_ideal)
            # Via antichains
            actual_ac = rowmotion_antichain(ss, ac)
            assert actual_ac == expected_ac

    def test_result_is_valid_antichain(self):
        """Result of rowmotion_antichain is a valid antichain."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        for ac in antichains(ss):
            result = rowmotion_antichain(ss, ac)
            assert _is_antichain(result, order), \
                f"rowmotion_antichain({ac}) = {result} is not an antichain"


# =========================================================================
# TestRowmotionViaToggles
# =========================================================================

class TestRowmotionViaToggles:
    """Tests for rowmotion via toggle composition."""

    def test_produces_valid_ideal(self):
        """Toggle-based rowmotion produces a valid order ideal."""
        ss = _boolean_lattice_2()
        order = compute_order_relation(ss)
        for ideal in order_ideals(ss):
            result = rowmotion_via_toggles(ss, ideal)
            assert _is_downset(result, order)

    def test_is_bijection(self):
        """Toggle-based rowmotion is a bijection on order ideals."""
        ss = _boolean_lattice_2()
        ideals_list = order_ideals(ss)
        results = [rowmotion_via_toggles(ss, i) for i in ideals_list]
        assert len(set(results)) == len(results)

    def test_agrees_direct_chain(self):
        """Toggle rowmotion agrees with direct rowmotion on 2-chain."""
        ss = _chain(2)
        for ideal in order_ideals(ss):
            direct = rowmotion_ideal(ss, ideal)
            via_t = rowmotion_via_toggles(ss, ideal)
            assert direct == via_t

    def test_agrees_direct_diamond(self):
        """Toggle rowmotion agrees with direct rowmotion on diamond."""
        ss = _boolean_lattice_2()
        for ideal in order_ideals(ss):
            direct = rowmotion_ideal(ss, ideal)
            via_t = rowmotion_via_toggles(ss, ideal)
            assert direct == via_t


# =========================================================================
# TestOrbits
# =========================================================================

class TestOrbits:
    """Tests for orbit decomposition."""

    def test_orbits_partition_all_ideals(self):
        """All orbits together contain every order ideal exactly once."""
        ss = _boolean_lattice_2()
        ideals_list = order_ideals(ss)
        orbits = all_orbits(ss)
        all_in_orbits = []
        for orbit in orbits:
            all_in_orbits.extend(orbit)
        assert len(all_in_orbits) == len(ideals_list)
        assert set(all_in_orbits) == set(ideals_list)

    def test_orbit_sizes_divide_order(self):
        """Each orbit size divides the rowmotion order (LCM)."""
        ss = _boolean_lattice_2()
        order = rowmotion_order(ss)
        for orbit in all_orbits(ss):
            assert order % len(orbit) == 0

    def test_order_is_lcm_of_sizes(self):
        """Rowmotion order = LCM of orbit sizes."""
        ss = _boolean_lattice_2()
        order = rowmotion_order(ss)
        orbits = all_orbits(ss)
        sizes = [len(o) for o in orbits]
        expected = 1
        for s in sizes:
            expected = math.lcm(expected, s)
        assert order == expected

    def test_each_orbit_is_a_cycle(self):
        """Applying rowmotion len(orbit) times returns to start."""
        ss = _boolean_lattice_2()
        for orbit in all_orbits(ss):
            start = orbit[0]
            current = start
            for _ in range(len(orbit)):
                current = rowmotion_ideal(ss, current)
            assert current == start

    def test_chain_2_single_orbit(self):
        """2-chain: all 3 ideals form one orbit of size 3."""
        ss = _chain(2)
        orbits = all_orbits(ss)
        total = sum(len(o) for o in orbits)
        assert total == 3


# =========================================================================
# TestRowmotionOrder
# =========================================================================

class TestRowmotionOrder:
    """Tests for rowmotion_order."""

    def test_single_state(self):
        """end: rowmotion order is finite."""
        ss = _build("end")
        order = rowmotion_order(ss)
        assert order >= 1

    def test_chain_2_order_3(self):
        """2-chain: rowmotion order = 3."""
        ss = _chain(2)
        assert rowmotion_order(ss) == 3

    def test_chain_3_order_4(self):
        """3-chain: rowmotion order = 4."""
        ss = _chain(3)
        assert rowmotion_order(ss) == 4

    def test_chain_4_order_5(self):
        """4-chain: rowmotion order = 5."""
        ss = _chain(4)
        assert rowmotion_order(ss) == 5

    def test_diamond_order_4(self):
        """Boolean lattice 2^2: rowmotion order = 4 (Brouwer-Schrijver)."""
        ss = _boolean_lattice_2()
        assert rowmotion_order(ss) == 4


# =========================================================================
# TestHomomesy
# =========================================================================

class TestHomomesy:
    """Tests for cardinality homomesy."""

    def test_single_orbit_always_homomesic(self):
        """A single orbit is trivially homomesic."""
        ss = _chain(2)
        orbits = all_orbits(ss)
        # All ideals in one orbit
        is_homo, avg = check_homomesy(ss, orbits)
        assert is_homo is True
        assert avg is not None

    def test_diamond_homomesy(self):
        """Diamond: cardinality should be homomesic."""
        ss = _boolean_lattice_2()
        orbits = all_orbits(ss)
        is_homo, avg = check_homomesy(ss, orbits)
        assert is_homo is True

    def test_chain_2_average(self):
        """2-chain: average cardinality = (0+1+2)/3 = 1.0."""
        ss = _chain(2)
        orbits = all_orbits(ss)
        is_homo, avg = check_homomesy(ss, orbits)
        assert avg is not None
        assert abs(avg - 1.0) < 1e-10

    def test_empty_orbits_trivial(self):
        """Empty orbit list: homomesic by convention."""
        ss = _build("end")
        is_homo, avg = check_homomesy(ss, [])
        assert is_homo is True
        assert avg is None


# =========================================================================
# TestAnalyzeRowmotion
# =========================================================================

class TestAnalyzeRowmotion:
    """Tests for the full analyze_rowmotion function."""

    def test_result_is_frozen(self):
        """RowmotionResult is a frozen dataclass."""
        ss = _build("end")
        result = analyze_rowmotion(ss)
        assert isinstance(result, RowmotionResult)
        with pytest.raises(AttributeError):
            result.rowmotion_order = 99  # type: ignore[misc]

    def test_end_analysis(self):
        """Full analysis on 'end'."""
        ss = _build("end")
        result = analyze_rowmotion(ss)
        assert len(result.order_ideals) == 2
        assert len(result.antichains) == 2
        assert result.num_orbits >= 1
        assert result.rowmotion_order >= 1

    def test_branch_analysis(self):
        """Full analysis on &{a: end}."""
        ss = _build("&{a: end}")
        result = analyze_rowmotion(ss)
        assert len(result.order_ideals) == 3
        assert len(result.antichains) == 3
        assert result.rowmotion_order == 3

    def test_all_fields_populated(self):
        """All fields of RowmotionResult are populated correctly."""
        ss = _boolean_lattice_2()
        result = analyze_rowmotion(ss)
        assert result.order_ideals is not None
        assert result.antichains is not None
        assert result.rowmotion_order == 4
        assert len(result.orbit_sizes) == result.num_orbits
        assert sum(result.orbit_sizes) == len(result.order_ideals)
        assert isinstance(result.is_homomesic_cardinality, bool)
        assert result.toggle_sequence is not None

    def test_parallel_analysis(self):
        """Full analysis on parallel composition."""
        ss = _build("(&{a: end} || &{b: end})")
        result = analyze_rowmotion(ss)
        assert len(result.order_ideals) == 6
        assert result.rowmotion_order == 4
        assert result.is_homomesic_cardinality is True

    def test_orbit_sizes_sorted(self):
        """Orbit sizes are returned sorted."""
        ss = _boolean_lattice_2()
        result = analyze_rowmotion(ss)
        assert result.orbit_sizes == sorted(result.orbit_sizes)


# =========================================================================
# TestBenchmarks
# =========================================================================

class TestBenchmarks:
    """Tests on benchmark protocol session types."""

    def test_iterator(self):
        """Java Iterator protocol."""
        ss = _build("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_rowmotion(ss)
        assert result.rowmotion_order >= 1
        assert result.num_orbits >= 1
        # Verify orbits partition ideals
        assert sum(result.orbit_sizes) == len(result.order_ideals)

    def test_file_object(self):
        """File Object protocol."""
        ss = _build("&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}")
        result = analyze_rowmotion(ss)
        assert result.rowmotion_order >= 1
        assert sum(result.orbit_sizes) == len(result.order_ideals)

    def test_nested_selection(self):
        """Nested selection: &{a: +{x: end, y: end}}."""
        ss = _build("&{a: +{x: end, y: end}}")
        result = analyze_rowmotion(ss)
        assert result.num_orbits >= 1
        assert sum(result.orbit_sizes) == len(result.order_ideals)

    def test_smtp_like(self):
        """SMTP-like protocol."""
        ss = _build(
            "&{connect: &{auth: +{OK: &{send: &{quit: end}}, "
            "FAIL: &{quit: end}}}}"
        )
        result = analyze_rowmotion(ss)
        assert result.rowmotion_order >= 1
        assert sum(result.orbit_sizes) == len(result.order_ideals)

    def test_simple_recursion(self):
        """Simple recursive type: rec X . &{a: X, b: end}."""
        ss = _build("rec X . &{a: X, b: end}")
        result = analyze_rowmotion(ss)
        assert result.rowmotion_order >= 1
        assert result.num_orbits >= 1


# =========================================================================
# TestMathematicalProperties
# =========================================================================

class TestMathematicalProperties:
    """Tests verifying key mathematical properties of rowmotion."""

    def test_rowmotion_finite_order(self):
        """Row^k = identity for some finite k on every finite poset."""
        for type_str in ["end", "&{a: end}", "&{a: end, b: end}"]:
            ss = _build(type_str)
            order = rowmotion_order(ss)
            for ideal in order_ideals(ss):
                current = ideal
                for _ in range(order):
                    current = rowmotion_ideal(ss, current)
                assert current == ideal, \
                    f"Row^{order}({ideal}) != {ideal} for {type_str}"

    def test_orbits_sum_to_total_ideals(self):
        """Sum of orbit sizes = total number of order ideals."""
        for ss in [_chain(2), _chain(3), _boolean_lattice_2()]:
            orbits = all_orbits(ss)
            total = sum(len(o) for o in orbits)
            assert total == len(order_ideals(ss))

    def test_antichain_ideal_bijection_is_perfect(self):
        """#antichains = #order ideals (fundamental bijection)."""
        for ss in [_chain(2), _chain(3), _boolean_lattice_2()]:
            assert len(antichains(ss)) == len(order_ideals(ss))

    def test_brouwer_schrijver_2x2(self):
        """Product of [2] x [2]: rowmotion order = 2+2 = 4."""
        ss = _boolean_lattice_2()
        assert rowmotion_order(ss) == 4

    def test_rowmotion_order_divides_ideal_count_factorial(self):
        """Rowmotion order divides |Ideals|! (permutation group constraint)."""
        ss = _boolean_lattice_2()
        n = len(order_ideals(ss))
        order = rowmotion_order(ss)
        fact = math.factorial(n)
        assert fact % order == 0
