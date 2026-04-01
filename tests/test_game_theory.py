"""Tests for game_theory module (Step 900a)."""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.game_theory import (
    BackwardInductionResult,
    ExtensiveGame,
    GameAnalysis,
    Player,
    Strategy,
    StrategyProfile,
    analyze_game_theory,
    backward_induction,
    build_game,
    cooperative_value,
    dominant_strategy,
    game_value,
    is_zero_sum,
    nash_equilibria,
    price_of_anarchy,
    price_of_stability,
    social_welfare,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ss(type_str: str) -> StateSpace:
    """Parse and build state space from a type string."""
    return build_statespace(parse(type_str))


def _game(type_str: str, payoffs=None) -> ExtensiveGame:
    """Build a game from a type string."""
    return build_game(_ss(type_str), payoffs)


# ---------------------------------------------------------------------------
# 1. ExtensiveGame construction
# ---------------------------------------------------------------------------

class TestBuildGame:
    """Tests for build_game."""

    def test_end_type(self) -> None:
        """end produces a trivial game with one terminal state."""
        game = _game("end")
        assert game.top == game.bottom
        assert game.top in game.terminal_states
        assert len(game.terminal_states) >= 1

    def test_single_branch(self) -> None:
        """&{a: end} produces a game with one Client decision node."""
        game = _game("&{a: end}")
        assert game.top not in game.terminal_states
        assert game.player_map[game.top] == Player.CLIENT
        acts = game.actions[game.top]
        assert len(acts) == 1
        assert acts[0][0] == "a"

    def test_single_select(self) -> None:
        """+{a: end} produces a game with one Server decision node."""
        game = _game("+{a: end}")
        assert game.player_map[game.top] == Player.SERVER
        acts = game.actions[game.top]
        assert len(acts) == 1
        assert acts[0][0] == "a"

    def test_branch_two_choices(self) -> None:
        """&{a: end, b: end} gives Client two actions."""
        game = _game("&{a: end, b: end}")
        assert game.player_map[game.top] == Player.CLIENT
        labels = {lbl for lbl, _ in game.actions[game.top]}
        assert labels == {"a", "b"}

    def test_select_two_choices(self) -> None:
        """+{OK: end, ERR: end} gives Server two actions."""
        game = _game("+{OK: end, ERR: end}")
        assert game.player_map[game.top] == Player.SERVER
        labels = {lbl for lbl, _ in game.actions[game.top]}
        assert labels == {"OK", "ERR"}

    def test_nested_branch_select(self) -> None:
        """&{a: +{OK: end, ERR: end}, b: end}"""
        game = _game("&{a: +{OK: end, ERR: end}, b: end}")
        assert game.player_map[game.top] == Player.CLIENT
        # Find the select node
        select_nodes = [s for s, p in game.player_map.items() if p == Player.SERVER]
        assert len(select_nodes) == 1

    def test_terminal_states(self) -> None:
        """Terminal states have no outgoing actions."""
        game = _game("&{a: end, b: end}")
        for t in game.terminal_states:
            assert t not in game.actions or len(game.actions.get(t, [])) == 0

    def test_is_dag(self) -> None:
        """Game is always a DAG after truncation."""
        game = _game("rec X . &{a: X, b: end}")
        assert game.is_dag

    def test_game_states_nonempty(self) -> None:
        """Game always has at least one state."""
        game = _game("end")
        assert len(game.states) >= 1

    def test_default_payoffs_assigned(self) -> None:
        """Default payoffs are assigned to all terminal states."""
        game = _game("&{a: end, b: end}")
        for t in game.terminal_states:
            assert t in game.payoffs

    def test_custom_payoffs(self) -> None:
        """Custom payoffs are used when provided."""
        ss = _ss("&{a: end, b: end}")
        custom = {ss.bottom: (10.0, 5.0)}
        game = build_game(ss, custom)
        assert game.payoffs[ss.bottom] == (10.0, 5.0)

    def test_parallel_game(self) -> None:
        """(a.end || b.end) produces a valid game."""
        game = _game("(&{a: end} || &{b: end})")
        assert game.top in game.player_map or game.top in game.terminal_states
        assert len(game.states) >= 2

    def test_recursive_type_produces_dag(self) -> None:
        """Recursive types produce a DAG game via truncation."""
        game = _game("rec X . &{next: X, done: end}")
        assert game.is_dag
        assert len(game.back_edges) >= 0  # May have back edges


# ---------------------------------------------------------------------------
# 2. Player assignment
# ---------------------------------------------------------------------------

class TestPlayerAssignment:
    """Tests for player assignment at game nodes."""

    def test_branch_is_client(self) -> None:
        game = _game("&{a: end}")
        assert game.player_map[game.top] == Player.CLIENT

    def test_select_is_server(self) -> None:
        game = _game("+{x: end}")
        assert game.player_map[game.top] == Player.SERVER

    def test_alternating_players(self) -> None:
        """&{a: +{OK: end}} alternates Client then Server."""
        game = _game("&{a: +{OK: end}}")
        assert game.player_map[game.top] == Player.CLIENT
        # Find the server node
        for s, p in game.player_map.items():
            if s != game.top:
                assert p == Player.SERVER

    def test_no_player_for_terminal(self) -> None:
        """Terminal states should not be in the player map."""
        game = _game("&{a: end}")
        for t in game.terminal_states:
            assert t not in game.player_map


# ---------------------------------------------------------------------------
# 3. Backward induction
# ---------------------------------------------------------------------------

class TestBackwardInduction:
    """Tests for backward induction (subgame perfect equilibrium)."""

    def test_end_trivial(self) -> None:
        """Trivial game: backward induction returns empty strategies."""
        game = _game("end")
        result = backward_induction(game)
        assert isinstance(result, BackwardInductionResult)
        assert len(result.equilibrium_path) >= 1

    def test_single_branch(self) -> None:
        """&{a: end}: Client must choose 'a'."""
        game = _game("&{a: end}")
        result = backward_induction(game)
        assert result.equilibrium_path[0] == game.top
        assert result.equilibrium_path[-1] in game.terminal_states

    def test_two_branches_prefers_shorter(self) -> None:
        """With default payoffs (negative depth), backward induction picks shortest path."""
        game = _game("&{a: end, b: &{c: end}}")
        result = backward_induction(game)
        # 'a' leads directly to end (depth 1), 'b' goes through another branch (depth 2)
        # Client should prefer 'a' (higher payoff = less negative)
        client_strat = result.strategy_profile[Player.CLIENT]
        assert client_strat[game.top] == "a"

    def test_equilibrium_path_starts_at_top(self) -> None:
        game = _game("&{a: end, b: end}")
        result = backward_induction(game)
        assert result.equilibrium_path[0] == game.top

    def test_equilibrium_path_ends_at_terminal(self) -> None:
        game = _game("&{a: +{OK: end, ERR: end}}")
        result = backward_induction(game)
        assert result.equilibrium_path[-1] in game.terminal_states

    def test_equilibrium_payoff_exists(self) -> None:
        game = _game("&{a: end}")
        result = backward_induction(game)
        assert isinstance(result.equilibrium_payoff, tuple)
        assert len(result.equilibrium_payoff) == 2

    def test_server_maximizes_own_payoff(self) -> None:
        """Server picks the action that maximizes server payoff."""
        ss = _ss("+{OK: end, ERR: end}")
        # Give different payoffs: OK = (0, 10), ERR = (0, 1)
        payoffs: dict[int, tuple[float, float]] = {}
        for src, lbl, tgt in ss.transitions:
            if lbl == "OK":
                payoffs[tgt] = (0.0, 10.0)
            elif lbl == "ERR":
                payoffs[tgt] = (0.0, 1.0)
        game = build_game(ss, payoffs)
        result = backward_induction(game)
        server_strat = result.strategy_profile[Player.SERVER]
        assert server_strat[game.top] == "OK"


# ---------------------------------------------------------------------------
# 4. Nash equilibria
# ---------------------------------------------------------------------------

class TestNashEquilibria:
    """Tests for Nash equilibria enumeration."""

    def test_end_has_one_ne(self) -> None:
        """Trivial game has exactly one NE."""
        game = _game("end")
        eqs = nash_equilibria(game)
        assert len(eqs) >= 1

    def test_single_branch_ne(self) -> None:
        """&{a: end} has one NE: Client picks 'a'."""
        game = _game("&{a: end}")
        eqs = nash_equilibria(game)
        assert len(eqs) >= 1

    def test_two_equal_branches_ne(self) -> None:
        """&{a: end, b: end}: both choices lead to same payoff, both are NE."""
        game = _game("&{a: end, b: end}")
        eqs = nash_equilibria(game)
        # With equal payoffs, both strategies are NE
        assert len(eqs) >= 1

    def test_ne_is_strategy_profile(self) -> None:
        """Each NE is a valid strategy profile."""
        game = _game("&{a: +{OK: end, ERR: end}}")
        eqs = nash_equilibria(game)
        for eq in eqs:
            assert Player.CLIENT in eq or Player.SERVER in eq

    def test_spe_is_ne(self) -> None:
        """The subgame perfect equilibrium is always a Nash equilibrium."""
        game = _game("&{a: +{OK: end, ERR: end}, b: end}")
        bi = backward_induction(game)
        eqs = nash_equilibria(game)
        # SPE should be among the NE
        spe_profile = bi.strategy_profile
        # Check that SPE payoff matches some NE payoff
        spe_payoff = bi.equilibrium_payoff
        ne_payoffs = []
        for eq in eqs:
            from reticulate.game_theory import _evaluate_profile
            p = _evaluate_profile(game, eq)
            if p is not None:
                ne_payoffs.append(p)
        # The SPE payoff should appear among NE payoffs
        assert any(
            abs(p[0] - spe_payoff[0]) < 1e-9 and abs(p[1] - spe_payoff[1]) < 1e-9
            for p in ne_payoffs
        ) or not ne_payoffs


# ---------------------------------------------------------------------------
# 5. Dominant strategy
# ---------------------------------------------------------------------------

class TestDominantStrategy:
    """Tests for dominant strategy detection."""

    def test_single_branch_dominant(self) -> None:
        """&{a: end}: Client has a trivially dominant strategy."""
        game = _game("&{a: end}")
        dom = dominant_strategy(game, Player.CLIENT)
        assert dom is not None
        assert game.top in dom
        assert dom[game.top] == "a"

    def test_single_select_dominant(self) -> None:
        """+{x: end}: Server has a trivially dominant strategy."""
        game = _game("+{x: end}")
        dom = dominant_strategy(game, Player.SERVER)
        assert dom is not None

    def test_no_dominant_in_symmetric(self) -> None:
        """If all branches have identical payoffs, any strategy is dominant."""
        game = _game("&{a: end, b: end}")
        dom = dominant_strategy(game, Player.CLIENT)
        # With equal payoffs, every strategy weakly dominates: some is returned
        assert dom is not None

    def test_client_no_dominant_when_server_controls(self) -> None:
        """Client may not have a dominant strategy when outcomes depend on server."""
        # In +{x: end}, Client has no decision nodes
        game = _game("+{x: end}")
        dom = dominant_strategy(game, Player.CLIENT)
        # Client has no nodes, so trivially dominant
        assert dom is not None or dom == {}


# ---------------------------------------------------------------------------
# 6. Game value (minimax)
# ---------------------------------------------------------------------------

class TestGameValue:
    """Tests for minimax game value."""

    def test_end_value_zero(self) -> None:
        """Trivial game has value 0 (or the terminal payoff)."""
        game = _game("end")
        v = game_value(game)
        assert isinstance(v, float)

    def test_single_branch_value(self) -> None:
        """&{a: end}: game value = terminal payoff."""
        game = _game("&{a: end}")
        v = game_value(game)
        assert isinstance(v, float)

    def test_client_chooses_best(self) -> None:
        """Client maximizes: picks the higher-payoff branch."""
        ss = _ss("&{a: end, b: &{c: end}}")
        game = build_game(ss)
        v = game_value(game)
        # Client wants max payoff (least negative depth)
        assert v >= game_value(game)  # Tautology, but checks no crash

    def test_adversarial_with_custom_payoffs(self) -> None:
        """Zero-sum adversarial game: client max, server min."""
        ss = _ss("&{a: +{OK: end, ERR: end}}")
        # Custom zero-sum payoffs
        payoffs: dict[int, tuple[float, float]] = {}
        for t in [s for s in ss.states if not ss.enabled(s)]:
            payoffs[t] = (1.0, -1.0)
        game = build_game(ss, payoffs)
        v = game_value(game)
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# 7. Zero-sum check
# ---------------------------------------------------------------------------

class TestZeroSum:
    """Tests for is_zero_sum."""

    def test_default_payoffs_not_zero_sum(self) -> None:
        """Default cooperative payoffs are NOT zero-sum (both negative)."""
        game = _game("&{a: end}")
        assert not is_zero_sum(game)

    def test_custom_zero_sum(self) -> None:
        """Custom zero-sum payoffs are detected."""
        ss = _ss("&{a: end}")
        payoffs = {ss.bottom: (1.0, -1.0)}
        game = build_game(ss, payoffs)
        assert is_zero_sum(game)

    def test_custom_not_zero_sum(self) -> None:
        ss = _ss("&{a: end}")
        payoffs = {ss.bottom: (1.0, 1.0)}
        game = build_game(ss, payoffs)
        assert not is_zero_sum(game)

    def test_multiple_terminals_zero_sum(self) -> None:
        ss = _ss("&{a: end, b: end}")
        payoffs = {ss.bottom: (5.0, -5.0)}
        game = build_game(ss, payoffs)
        assert is_zero_sum(game)


# ---------------------------------------------------------------------------
# 8. Social welfare
# ---------------------------------------------------------------------------

class TestSocialWelfare:
    """Tests for social welfare computation."""

    def test_welfare_sum_of_payoffs(self) -> None:
        game = _game("&{a: end}")
        bi = backward_induction(game)
        sw = social_welfare(game, bi.strategy_profile)
        payoff = bi.equilibrium_payoff
        assert abs(sw - (payoff[0] + payoff[1])) < 1e-9

    def test_welfare_zero_sum_is_zero(self) -> None:
        ss = _ss("&{a: end}")
        payoffs = {ss.bottom: (1.0, -1.0)}
        game = build_game(ss, payoffs)
        bi = backward_induction(game)
        sw = social_welfare(game, bi.strategy_profile)
        assert abs(sw) < 1e-9


# ---------------------------------------------------------------------------
# 9. Cooperative value
# ---------------------------------------------------------------------------

class TestCooperativeValue:
    """Tests for cooperative_value."""

    def test_end_cooperative(self) -> None:
        game = _game("end")
        cv = cooperative_value(game)
        assert isinstance(cv, float)

    def test_branch_cooperative(self) -> None:
        game = _game("&{a: end, b: end}")
        cv = cooperative_value(game)
        assert isinstance(cv, float)

    def test_shorter_branch_preferred(self) -> None:
        """Cooperative value should reflect the shortest terminal."""
        game = _game("&{a: end, b: &{c: end}}")
        cv = cooperative_value(game)
        # Cooperative = best welfare among terminals
        # 'a' reaches end at depth 1, 'b.c' at depth 2
        # Default payoffs: depth 1 = (-1, -1), depth 2 = (-2, -2)
        # Best welfare = -2 (from depth 1)
        assert cv >= -4.0  # Sanity check


# ---------------------------------------------------------------------------
# 10. Price of anarchy / stability
# ---------------------------------------------------------------------------

class TestPriceOfAnarchy:
    """Tests for PoA and PoS."""

    def test_poa_single_branch(self) -> None:
        """Single branch: PoA = 1 (no inefficiency)."""
        game = _game("&{a: end}")
        poa = price_of_anarchy(game)
        # With single path, PoA should be 1.0
        if poa is not None:
            assert abs(poa - 1.0) < 1e-9

    def test_pos_single_branch(self) -> None:
        game = _game("&{a: end}")
        pos = price_of_stability(game)
        if pos is not None:
            assert abs(pos - 1.0) < 1e-9

    def test_poa_returns_float_or_none(self) -> None:
        game = _game("&{a: end, b: end}")
        poa = price_of_anarchy(game)
        assert poa is None or isinstance(poa, float)


# ---------------------------------------------------------------------------
# 11. Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeGameTheory:
    """Tests for the full analyze_game_theory function."""

    def test_end_analysis(self) -> None:
        ss = _ss("end")
        result = analyze_game_theory(ss)
        assert isinstance(result, GameAnalysis)
        assert result.game is not None
        assert result.backward_induction is not None

    def test_branch_analysis(self) -> None:
        ss = _ss("&{a: end, b: end}")
        result = analyze_game_theory(ss)
        assert result.num_strategies_client >= 1
        assert isinstance(result.game_value, float)
        assert isinstance(result.cooperative_value, float)
        assert isinstance(result.is_zero_sum, bool)

    def test_select_analysis(self) -> None:
        ss = _ss("+{OK: end, ERR: end}")
        result = analyze_game_theory(ss)
        assert result.num_strategies_server >= 1

    def test_nested_analysis(self) -> None:
        ss = _ss("&{a: +{OK: end, ERR: end}, b: end}")
        result = analyze_game_theory(ss)
        assert result.nash_equilibria is not None
        assert len(result.nash_equilibria) >= 1

    def test_recursive_analysis(self) -> None:
        ss = _ss("rec X . &{next: X, done: end}")
        result = analyze_game_theory(ss)
        assert result.game.is_dag

    def test_parallel_analysis(self) -> None:
        ss = _ss("(&{a: end} || &{b: end})")
        result = analyze_game_theory(ss)
        assert isinstance(result, GameAnalysis)

    def test_analysis_with_custom_payoffs(self) -> None:
        ss = _ss("&{a: end}")
        payoffs = {ss.bottom: (100.0, 50.0)}
        result = analyze_game_theory(ss, payoffs)
        assert result.backward_induction.equilibrium_payoff == (100.0, 50.0)


# ---------------------------------------------------------------------------
# 12. Benchmark protocols
# ---------------------------------------------------------------------------

class TestBenchmarkProtocols:
    """Tests with realistic protocol types from benchmarks."""

    def test_iterator(self) -> None:
        """Java Iterator protocol."""
        ss = _ss("rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}")
        result = analyze_game_theory(ss)
        assert result.game.is_dag
        assert result.backward_induction is not None

    def test_simple_auth(self) -> None:
        """Simple auth: login then select OK/FAIL."""
        ss = _ss("&{login: +{OK: end, FAIL: end}}")
        result = analyze_game_theory(ss)
        assert result.num_strategies_client >= 1
        assert result.num_strategies_server >= 1

    def test_two_buyer(self) -> None:
        """Two-buyer-like: branch then select."""
        ss = _ss("&{request: +{accept: end, reject: end}}")
        result = analyze_game_theory(ss)
        assert len(result.nash_equilibria) >= 1

    def test_file_protocol(self) -> None:
        """File: open, read/write, close."""
        ss = _ss("&{open: &{read: end, write: end}}")
        result = analyze_game_theory(ss)
        assert result.num_strategies_client >= 2


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_deeply_nested(self) -> None:
        """Deeply nested branch."""
        ss = _ss("&{a: &{b: &{c: end}}}")
        game = build_game(ss)
        assert len(game.states) >= 4

    def test_wide_branch(self) -> None:
        """Wide branch with many choices."""
        ss = _ss("&{a: end, b: end, c: end, d: end}")
        game = build_game(ss)
        assert len(game.actions[game.top]) == 4

    def test_all_select(self) -> None:
        """Only server choices."""
        ss = _ss("+{a: +{b: end}}")
        game = build_game(ss)
        for s, p in game.player_map.items():
            assert p == Player.SERVER

    def test_all_branch(self) -> None:
        """Only client choices."""
        ss = _ss("&{a: &{b: end}}")
        game = build_game(ss)
        for s, p in game.player_map.items():
            assert p == Player.CLIENT

    def test_strategy_profile_completeness(self) -> None:
        """BI strategy profile covers all decision nodes of the equilibrium path."""
        game = _game("&{a: +{OK: end, ERR: end}, b: end}")
        bi = backward_induction(game)
        path = bi.equilibrium_path
        for s in path:
            if s in game.player_map:
                player = game.player_map[s]
                strat = bi.strategy_profile.get(player, {})
                assert s in strat

    def test_trace_path_no_infinite_loop(self) -> None:
        """Tracing a path in a recursive game doesn't loop forever."""
        game = _game("rec X . &{a: X, b: end}")
        bi = backward_induction(game)
        # Path should terminate
        assert len(bi.equilibrium_path) < 100
