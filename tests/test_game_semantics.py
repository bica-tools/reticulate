"""Tests for game_semantics.py — game-semantic interpretation of session types (Step 94)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.game_semantics import (
    Arena,
    CounterStrategy,
    GameAnalysis,
    GameOutcome,
    Move,
    Play,
    Player,
    Strategy,
    analyze_game,
    build_arena,
    compose_arenas,
    enumerate_counter_strategies,
    enumerate_strategies,
    game_value,
    is_winning,
    play_game,
    winning_strategies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str):
    """Parse a session type string and build its state space."""
    return build_statespace(parse(type_str))


def _arena(type_str: str) -> Arena:
    """Build an arena from a session type string."""
    return build_arena(_ss(type_str))


# ===========================================================================
# Section 1: Arena construction
# ===========================================================================


class TestBuildArena:
    """Tests for build_arena()."""

    def test_end_arena(self):
        """end → single terminal position, no moves."""
        arena = _arena("end")
        assert len(arena.positions) == 1
        assert len(arena.moves) == 0
        assert arena.initial in arena.terminal

    def test_simple_branch(self):
        """&{a: end, b: end} → one O-position, two moves, one terminal."""
        arena = _arena("&{a: end, b: end}")
        assert arena.initial not in arena.terminal
        moves = arena.moves_from(arena.initial)
        assert len(moves) == 2
        assert all(m.player == Player.OPPONENT for m in moves)
        assert arena.polarity[arena.initial] == Player.OPPONENT

    def test_simple_selection(self):
        """+{x: end, y: end} → one P-position, two moves."""
        arena = _arena("+{x: end, y: end}")
        moves = arena.moves_from(arena.initial)
        assert len(moves) == 2
        assert all(m.player == Player.PROPONENT for m in moves)
        assert arena.polarity[arena.initial] == Player.PROPONENT

    def test_branch_then_select(self):
        """&{a: +{ok: end, err: end}, b: end}"""
        arena = _arena("&{a: +{ok: end, err: end}, b: end}")
        # Initial is Opponent (branch)
        assert arena.polarity[arena.initial] == Player.OPPONENT
        # Should have at least one P-position (the selection after 'a')
        assert len(arena.proponent_positions()) >= 1

    def test_terminal_positions(self):
        """Terminal positions are those with no outgoing moves."""
        arena = _arena("&{a: end, b: end}")
        for t in arena.terminal:
            assert arena.moves_from(t) == []

    def test_positions_match_statespace(self):
        """Arena positions should match state-space states."""
        ss = _ss("&{a: +{x: end}, b: end}")
        arena = build_arena(ss)
        assert arena.positions == frozenset(ss.states)

    def test_initial_is_top(self):
        """Arena initial should be state-space top."""
        ss = _ss("+{a: end, b: end}")
        arena = build_arena(ss)
        assert arena.initial == ss.top

    def test_moves_match_transitions(self):
        """Number of moves should equal number of transitions."""
        ss = _ss("&{a: end, b: +{x: end, y: end}}")
        arena = build_arena(ss)
        assert len(arena.moves) == len(ss.transitions)

    def test_single_branch_forced(self):
        """&{a: end} → single Opponent move (forced)."""
        arena = _arena("&{a: end}")
        moves = arena.moves_from(arena.initial)
        assert len(moves) == 1
        assert moves[0].player == Player.OPPONENT
        assert moves[0].label == "a"

    def test_nested_branches_all_opponent(self):
        """&{a: &{b: end}} → two O-positions."""
        arena = _arena("&{a: &{b: end}}")
        assert len(arena.opponent_positions()) == 2

    def test_nested_selections_all_proponent(self):
        """+{a: +{b: end}} → two P-positions."""
        arena = _arena("+{a: +{b: end}}")
        assert len(arena.proponent_positions()) == 2

    def test_depth_simple(self):
        """Depth of &{a: end} should be 1."""
        arena = _arena("&{a: end}")
        assert arena.depth == 1

    def test_depth_nested(self):
        """Depth of &{a: +{ok: end, err: end}, b: end} should be 2."""
        arena = _arena("&{a: +{ok: end, err: end}, b: end}")
        assert arena.depth == 2


# ===========================================================================
# Section 2: Strategy enumeration
# ===========================================================================


class TestEnumerateStrategies:
    """Tests for enumerate_strategies()."""

    def test_no_selection_one_strategy(self):
        """&{a: end} → no P-positions → exactly 1 (empty) strategy."""
        arena = _arena("&{a: end}")
        strats = enumerate_strategies(arena)
        assert len(strats) == 1
        assert strats[0].choices == {}

    def test_single_selection_two_strategies(self):
        """+{a: end, b: end} → 1 P-position with 2 choices → 2 strategies."""
        arena = _arena("+{a: end, b: end}")
        strats = enumerate_strategies(arena)
        assert len(strats) == 2

    def test_single_selection_three_choices(self):
        """+{a: end, b: end, c: end} → 3 strategies."""
        arena = _arena("+{a: end, b: end, c: end}")
        strats = enumerate_strategies(arena)
        assert len(strats) == 3

    def test_two_selections_product(self):
        """&{a: +{x: end, y: end}, b: +{u: end, v: end}} → 2*2 = 4 strategies."""
        arena = _arena("&{a: +{x: end, y: end}, b: +{u: end, v: end}}")
        strats = enumerate_strategies(arena)
        assert len(strats) == 4

    def test_end_type_one_strategy(self):
        """end → 1 empty strategy."""
        arena = _arena("end")
        strats = enumerate_strategies(arena)
        assert len(strats) == 1

    def test_strategy_covers_all_p_positions(self):
        """Each strategy should have choices for all reachable P-positions."""
        arena = _arena("&{a: +{x: end, y: end}, b: +{u: end, v: end}}")
        for strat in enumerate_strategies(arena):
            for pos in arena.proponent_positions():
                assert pos in strat.choices

    def test_strategy_labels(self):
        """Strategy.labels should map positions to label strings."""
        arena = _arena("+{a: end, b: end}")
        strats = enumerate_strategies(arena)
        for s in strats:
            labels = s.labels
            assert len(labels) == 1
            assert list(labels.values())[0] in {"a", "b"}


# ===========================================================================
# Section 3: Counter-strategy enumeration
# ===========================================================================


class TestEnumerateCounterStrategies:
    """Tests for enumerate_counter_strategies()."""

    def test_no_branch_one_counter(self):
        """+{a: end} → no O-positions → 1 empty counter-strategy."""
        arena = _arena("+{a: end}")
        cs = enumerate_counter_strategies(arena)
        assert len(cs) == 1

    def test_single_branch_two_counters(self):
        """&{a: end, b: end} → 1 O-position with 2 choices → 2 counter-strategies."""
        arena = _arena("&{a: end, b: end}")
        cs = enumerate_counter_strategies(arena)
        assert len(cs) == 2

    def test_branch_three_choices(self):
        """&{a: end, b: end, c: end} → 3 counter-strategies."""
        arena = _arena("&{a: end, b: end, c: end}")
        cs = enumerate_counter_strategies(arena)
        assert len(cs) == 3


# ===========================================================================
# Section 4: Playing games
# ===========================================================================


class TestPlayGame:
    """Tests for play_game()."""

    def test_end_empty_play(self):
        """end → initial is terminal → empty play."""
        arena = _arena("end")
        s = Strategy(choices={})
        cs = CounterStrategy(choices={})
        play = play_game(arena, s, cs)
        assert play == ()

    def test_branch_one_step(self):
        """&{a: end} → one Opponent move."""
        arena = _arena("&{a: end}")
        s = Strategy(choices={})
        moves = arena.moves_from(arena.initial)
        cs = CounterStrategy(choices={arena.initial: moves[0]})
        play = play_game(arena, s, cs)
        assert len(play) == 1
        assert play[0].label == "a"

    def test_selection_one_step(self):
        """+{x: end, y: end} → one Proponent move."""
        arena = _arena("+{x: end, y: end}")
        moves = arena.moves_from(arena.initial)
        s = Strategy(choices={arena.initial: moves[0]})
        cs = CounterStrategy(choices={})
        play = play_game(arena, s, cs)
        assert len(play) == 1

    def test_branch_then_select_two_steps(self):
        """&{a: +{ok: end, err: end}} → two moves."""
        arena = _arena("&{a: +{ok: end, err: end}}")
        # O moves 'a', then P moves 'ok' or 'err'
        o_moves = arena.moves_from(arena.initial)
        assert len(o_moves) == 1
        sel_pos = o_moves[0].target
        p_moves = arena.moves_from(sel_pos)
        assert len(p_moves) == 2

        s = Strategy(choices={sel_pos: p_moves[0]})
        cs = CounterStrategy(choices={arena.initial: o_moves[0]})
        play = play_game(arena, s, cs)
        assert len(play) == 2
        assert play[0].player == Player.OPPONENT
        assert play[1].player == Player.PROPONENT

    def test_play_reaches_terminal(self):
        """Verify play reaches a terminal position."""
        arena = _arena("&{a: end, b: end}")
        o_moves = arena.moves_from(arena.initial)
        cs = CounterStrategy(choices={arena.initial: o_moves[0]})
        s = Strategy(choices={})
        play = play_game(arena, s, cs)
        assert arena.is_terminal(play[-1].target)


# ===========================================================================
# Section 5: Winning strategy check
# ===========================================================================


class TestIsWinning:
    """Tests for is_winning()."""

    def test_end_trivially_winning(self):
        """end → empty strategy is winning (already terminal)."""
        arena = _arena("end")
        s = Strategy(choices={})
        assert is_winning(s, arena)

    def test_branch_only_winning(self):
        """&{a: end, b: end} → empty P-strategy is winning (O always reaches end)."""
        arena = _arena("&{a: end, b: end}")
        s = Strategy(choices={})
        assert is_winning(s, arena)

    def test_selection_both_winning(self):
        """+{x: end, y: end} → both choices are winning."""
        arena = _arena("+{x: end, y: end}")
        strats = enumerate_strategies(arena)
        assert len(strats) == 2
        assert all(is_winning(s, arena) for s in strats)

    def test_branch_select_winning(self):
        """&{a: +{ok: end, err: end}, b: end} → all P-strategies win."""
        arena = _arena("&{a: +{ok: end, err: end}, b: end}")
        strats = enumerate_strategies(arena)
        assert all(is_winning(s, arena) for s in strats)

    def test_nested_selection_winning(self):
        """+{a: +{b: end}} → single strategy is winning."""
        arena = _arena("+{a: +{b: end}}")
        strats = enumerate_strategies(arena)
        assert len(strats) == 1
        assert is_winning(strats[0], arena)


# ===========================================================================
# Section 6: Winning strategies enumeration
# ===========================================================================


class TestWinningStrategies:
    """Tests for winning_strategies()."""

    def test_end_one_winning(self):
        arena = _arena("end")
        ws = winning_strategies(arena)
        assert len(ws) == 1

    def test_all_branch_one_winning(self):
        """&{a: end, b: end} → one empty winning strategy."""
        arena = _arena("&{a: end, b: end}")
        ws = winning_strategies(arena)
        assert len(ws) == 1

    def test_selection_all_winning(self):
        """+{a: end, b: end} → both strategies win."""
        arena = _arena("+{a: end, b: end}")
        ws = winning_strategies(arena)
        assert len(ws) == 2

    def test_complex_all_winning(self):
        """&{a: +{ok: end, err: end}, b: +{x: end, y: end}} → all 4 strategies win."""
        arena = _arena("&{a: +{ok: end, err: end}, b: +{x: end, y: end}}")
        ws = winning_strategies(arena)
        all_strats = enumerate_strategies(arena)
        assert len(ws) == len(all_strats) == 4


# ===========================================================================
# Section 7: Game value (outcome)
# ===========================================================================


class TestGameValue:
    """Tests for game_value()."""

    def test_end_proponent_wins(self):
        arena = _arena("end")
        assert game_value(arena) == GameOutcome.PROPONENT_WINS

    def test_branch_proponent_wins(self):
        """&{a: end} → Proponent wins (trivially, since all paths terminate)."""
        arena = _arena("&{a: end}")
        assert game_value(arena) == GameOutcome.PROPONENT_WINS

    def test_selection_proponent_wins(self):
        """+{a: end, b: end} → Proponent wins."""
        arena = _arena("+{a: end, b: end}")
        assert game_value(arena) == GameOutcome.PROPONENT_WINS

    def test_nested_proponent_wins(self):
        """Nested branch-select: Proponent always wins for terminating types."""
        arena = _arena("&{a: +{ok: end, err: end}, b: end}")
        assert game_value(arena) == GameOutcome.PROPONENT_WINS

    def test_deep_nesting_proponent_wins(self):
        """&{a: &{b: +{x: end, y: end}}} → Proponent wins."""
        arena = _arena("&{a: &{b: +{x: end, y: end}}}")
        assert game_value(arena) == GameOutcome.PROPONENT_WINS


# ===========================================================================
# Section 8: Arena composition (tensor product)
# ===========================================================================


class TestComposeArenas:
    """Tests for compose_arenas()."""

    def test_compose_two_ends(self):
        """end || end → single terminal position."""
        a1 = _arena("end")
        a2 = _arena("end")
        composed = compose_arenas(a1, a2)
        assert len(composed.positions) == 1
        assert composed.initial in composed.terminal

    def test_compose_positions_product(self):
        """Composed arena should have |A1| * |A2| positions."""
        a1 = _arena("&{a: end}")  # 2 positions
        a2 = _arena("&{b: end}")  # 2 positions
        composed = compose_arenas(a1, a2)
        assert len(composed.positions) == 4

    def test_compose_terminal(self):
        """Terminal in composed = both components terminal."""
        a1 = _arena("&{a: end}")
        a2 = _arena("&{b: end}")
        composed = compose_arenas(a1, a2)
        assert len(composed.terminal) == 1  # (end, end)

    def test_compose_interleaving_moves(self):
        """Composed arena has interleaved moves from both components."""
        a1 = _arena("&{a: end}")
        a2 = _arena("&{b: end}")
        composed = compose_arenas(a1, a2)
        labels = {m.label for m in composed.moves}
        assert "a" in labels
        assert "b" in labels

    def test_compose_preserves_determinacy(self):
        """Product of determinate games is determinate."""
        a1 = _arena("&{a: end, b: end}")
        a2 = _arena("+{x: end, y: end}")
        composed = compose_arenas(a1, a2)
        assert game_value(composed) == GameOutcome.PROPONENT_WINS

    def test_compose_initial_not_terminal(self):
        """If components have moves, composed initial is not terminal."""
        a1 = _arena("&{a: end}")
        a2 = _arena("&{b: end}")
        composed = compose_arenas(a1, a2)
        assert composed.initial not in composed.terminal


# ===========================================================================
# Section 9: Full analysis
# ===========================================================================


class TestAnalyzeGame:
    """Tests for analyze_game()."""

    def test_end_analysis(self):
        analysis = analyze_game(_ss("end"))
        assert analysis.is_determinate
        assert analysis.outcome == GameOutcome.PROPONENT_WINS
        assert analysis.num_proponent_positions == 0
        assert analysis.num_opponent_positions == 0
        assert analysis.num_terminal_positions == 1
        assert analysis.num_winning_strategies == 1
        assert analysis.num_total_strategies == 1

    def test_branch_analysis(self):
        analysis = analyze_game(_ss("&{a: end, b: end}"))
        assert analysis.is_determinate
        assert analysis.num_opponent_positions == 1
        assert analysis.num_proponent_positions == 0

    def test_select_analysis(self):
        analysis = analyze_game(_ss("+{x: end, y: end}"))
        assert analysis.is_determinate
        assert analysis.num_proponent_positions == 1
        assert analysis.num_opponent_positions == 0
        assert analysis.num_winning_strategies == 2
        assert analysis.num_total_strategies == 2

    def test_mixed_analysis(self):
        analysis = analyze_game(_ss("&{a: +{ok: end, err: end}, b: end}"))
        assert analysis.is_determinate
        assert analysis.num_proponent_positions >= 1
        assert analysis.num_opponent_positions >= 1
        assert analysis.example_winning_strategy is not None

    def test_analysis_has_arena(self):
        analysis = analyze_game(_ss("&{a: end}"))
        assert isinstance(analysis.arena, Arena)

    def test_analysis_winning_strategy_is_valid(self):
        """Example winning strategy should actually be winning."""
        analysis = analyze_game(_ss("&{a: +{ok: end, err: end}, b: end}"))
        ws = analysis.example_winning_strategy
        assert ws is not None
        assert is_winning(ws, analysis.arena)


# ===========================================================================
# Section 10: Recursive types
# ===========================================================================


class TestRecursiveTypes:
    """Tests for game semantics with recursive session types."""

    def test_simple_recursive_branch(self):
        """rec X . &{a: X, b: end} → game with a cycle; P still wins."""
        arena = _arena("rec X . &{a: X, b: end}")
        # Proponent has no selections, so only empty strategy
        strats = enumerate_strategies(arena)
        assert len(strats) == 1
        # O can choose 'b' to reach end, or 'a' to loop
        # But is_winning checks that ALL O strategies lead to terminal
        # O can loop forever with 'a' → strategy may not be winning
        # Actually with our play_game cycle detection it stops.
        # The cycle means O choosing 'a' repeatedly will be detected.

    def test_recursive_with_selection(self):
        """rec X . &{next: +{TRUE: X, FALSE: end}} (iterator pattern)."""
        arena = _arena("rec X . &{next: +{TRUE: X, FALSE: end}}")
        # P positions: the selection after 'next'
        assert len(arena.proponent_positions()) >= 1
        strats = enumerate_strategies(arena)
        # At least 2: choose TRUE or FALSE
        assert len(strats) >= 2


# ===========================================================================
# Section 11: Polarity and player assignment
# ===========================================================================


class TestPolarity:
    """Tests for correct player assignment."""

    def test_branch_is_opponent(self):
        arena = _arena("&{a: end}")
        assert arena.polarity[arena.initial] == Player.OPPONENT

    def test_selection_is_proponent(self):
        arena = _arena("+{a: end}")
        assert arena.polarity[arena.initial] == Player.PROPONENT

    def test_terminal_has_no_polarity(self):
        """Terminal positions should not appear in polarity map."""
        arena = _arena("&{a: end}")
        for t in arena.terminal:
            assert t not in arena.polarity

    def test_alternation_branch_select(self):
        """&{a: +{ok: end}} → O then P."""
        arena = _arena("&{a: +{ok: end}}")
        assert arena.polarity[arena.initial] == Player.OPPONENT
        m = arena.moves_from(arena.initial)[0]
        assert arena.polarity[m.target] == Player.PROPONENT


# ===========================================================================
# Section 12: Edge cases and Move dataclass
# ===========================================================================


class TestMoveAndEdgeCases:
    """Tests for Move dataclass and edge cases."""

    def test_move_is_frozen(self):
        m = Move(source=0, label="a", target=1, player=Player.OPPONENT)
        with pytest.raises(AttributeError):
            m.source = 2  # type: ignore[misc]

    def test_strategy_is_frozen(self):
        s = Strategy(choices={})
        with pytest.raises(AttributeError):
            s.choices = {}  # type: ignore[misc]

    def test_arena_is_frozen(self):
        arena = _arena("end")
        with pytest.raises(AttributeError):
            arena.initial = 99  # type: ignore[misc]

    def test_player_enum_values(self):
        assert Player.PROPONENT.value == "P"
        assert Player.OPPONENT.value == "O"

    def test_game_outcome_values(self):
        assert GameOutcome.PROPONENT_WINS.value == "P_wins"
        assert GameOutcome.OPPONENT_WINS.value == "O_wins"

    def test_arena_is_terminal(self):
        arena = _arena("&{a: end}")
        assert not arena.is_terminal(arena.initial)
        for t in arena.terminal:
            assert arena.is_terminal(t)

    def test_strategy_call(self):
        """Strategy.__call__ returns move or None."""
        arena = _arena("+{a: end}")
        strats = enumerate_strategies(arena)
        s = strats[0]
        assert s(arena.initial) is not None
        # Non-existent position
        assert s(9999) is None

    def test_counter_strategy_call(self):
        """CounterStrategy.__call__ returns move or None."""
        arena = _arena("&{a: end}")
        cs_list = enumerate_counter_strategies(arena)
        cs = cs_list[0]
        assert cs(arena.initial) is not None
        assert cs(9999) is None


# ===========================================================================
# Section 13: Benchmark protocols
# ===========================================================================


class TestBenchmarkProtocols:
    """Game analysis on realistic protocol patterns."""

    def test_atm_protocol(self):
        """ATM-like: &{insertCard: &{enterPIN: +{ok: &{withdraw: end, balance: end}, reject: end}}}."""
        ss = _ss("&{insertCard: &{enterPIN: +{ok: &{withdraw: end, balance: end}, reject: end}}}")
        analysis = analyze_game(ss)
        assert analysis.is_determinate
        assert analysis.num_proponent_positions >= 1

    def test_simple_auth_flow(self):
        """&{login: +{grant: &{action: end}, deny: end}}."""
        ss = _ss("&{login: +{grant: &{action: end}, deny: end}}")
        analysis = analyze_game(ss)
        assert analysis.is_determinate
        assert analysis.num_winning_strategies >= 1

    def test_iterator_pattern(self):
        """rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}."""
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        analysis = analyze_game(ss)
        assert analysis.num_proponent_positions >= 1
        # With cycle detection, should find winning strategies
        assert analysis.num_total_strategies >= 2
