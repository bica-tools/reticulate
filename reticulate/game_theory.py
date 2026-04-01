"""Session types as strategic games (Step 900a).

Maps session type state spaces to extensive-form games:
  - Branch states = Player 1 (Client) decision nodes
  - Selection states = Player 2 (Server) decision nodes
  - Terminal states (bottom) = terminal nodes with payoffs
  - Paths top->bottom = complete plays

Key concepts implemented:
  - ExtensiveGame construction from state spaces
  - Backward induction (subgame perfect equilibrium)
  - Nash equilibria enumeration
  - Dominant strategy detection
  - Minimax game value
  - Zero-sum game check
  - Social welfare computation
  - Price of anarchy / price of stability
  - Full game-theoretic analysis
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from itertools import product as itertools_product

from reticulate.statespace import StateSpace, truncate_back_edges


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Player(Enum):
    """Players in the protocol game."""
    CLIENT = 1   # Branch states (environment chooses)
    SERVER = 2   # Selection states (process chooses)
    NATURE = 0   # Single-transition (forced move, no real choice)


@dataclass(frozen=True)
class ExtensiveGame:
    """Extensive-form game with perfect information derived from a state space.

    Attributes:
        states: set of state IDs (game nodes)
        terminal_states: states with no outgoing transitions (leaves)
        top: root of the game tree
        bottom: canonical terminal state
        player_map: maps each non-terminal state to the player who moves
        actions: maps each non-terminal state to list of (label, target)
        payoffs: maps each terminal state to (client_payoff, server_payoff)
        is_dag: True if the game tree is acyclic (required for backward induction)
        back_edges: transitions removed to make the game acyclic
    """
    states: frozenset[int]
    terminal_states: frozenset[int]
    top: int
    bottom: int
    player_map: dict[int, Player]
    actions: dict[int, list[tuple[str, int]]]
    payoffs: dict[int, tuple[float, float]]
    is_dag: bool
    back_edges: list[tuple[int, str, int]] = field(default_factory=list)


# A strategy maps states owned by a player to a chosen action label.
Strategy = dict[int, str]

# A strategy profile is one strategy per player.
StrategyProfile = dict[Player, Strategy]


@dataclass(frozen=True)
class BackwardInductionResult:
    """Result of backward induction.

    Attributes:
        values: maps each state to (client_value, server_value)
        strategy_profile: the subgame perfect equilibrium strategies
        equilibrium_path: the path from top to bottom under the SPE
        equilibrium_payoff: the payoff at the root under the SPE
    """
    values: dict[int, tuple[float, float]]
    strategy_profile: StrategyProfile
    equilibrium_path: list[int]
    equilibrium_payoff: tuple[float, float]


@dataclass(frozen=True)
class GameAnalysis:
    """Full game-theoretic analysis of a session type state space.

    Attributes:
        game: the extensive-form game
        backward_induction: backward induction result
        game_value: minimax value (adversarial)
        cooperative_value: shortest path length (cooperative)
        is_zero_sum: whether payoffs sum to zero at all terminals
        num_strategies_client: number of pure strategies for client
        num_strategies_server: number of pure strategies for server
        price_of_anarchy: ratio worst NE / optimal cooperative
        price_of_stability: ratio best NE / optimal cooperative
        nash_equilibria: all pure-strategy Nash equilibria (if enumerable)
        dominant_client: dominant strategy for client, or None
        dominant_server: dominant strategy for server, or None
    """
    game: ExtensiveGame
    backward_induction: BackwardInductionResult
    game_value: float
    cooperative_value: float
    is_zero_sum: bool
    num_strategies_client: int
    num_strategies_server: int
    price_of_anarchy: float | None
    price_of_stability: float | None
    nash_equilibria: list[StrategyProfile] | None
    dominant_client: Strategy | None
    dominant_server: Strategy | None


# ---------------------------------------------------------------------------
# Game construction
# ---------------------------------------------------------------------------

def build_game(
    ss: StateSpace,
    payoffs: dict[int, tuple[float, float]] | None = None,
) -> ExtensiveGame:
    """Build an extensive-form game from a session type state space.

    If payoffs is None, uses default payoffs: both players receive
    negative path-depth from top (cooperative: both want to finish quickly).

    The state space is truncated to a DAG if it contains cycles
    (from recursive types).
    """
    # Truncate cycles for game tree analysis
    dag_ss, back_edges = truncate_back_edges(ss)

    # Determine terminal states (no outgoing transitions in the DAG)
    has_outgoing: set[int] = set()
    for src, _, _ in dag_ss.transitions:
        has_outgoing.add(src)
    terminal_states = frozenset(dag_ss.states - has_outgoing)

    # Build action map
    actions: dict[int, list[tuple[str, int]]] = {}
    for s in dag_ss.states:
        if s not in terminal_states:
            acts = dag_ss.enabled(s)
            if acts:
                actions[s] = acts

    # Determine player at each non-terminal state
    player_map: dict[int, Player] = {}
    for s in dag_ss.states:
        if s in terminal_states:
            continue
        sel_transitions = dag_ss.enabled_selections(s)
        method_transitions = dag_ss.enabled_methods(s)
        if sel_transitions and not method_transitions:
            player_map[s] = Player.SERVER
        elif method_transitions and not sel_transitions:
            player_map[s] = Player.CLIENT
        elif sel_transitions and method_transitions:
            # Mixed: treat as CLIENT (branch has priority)
            player_map[s] = Player.CLIENT
        else:
            player_map[s] = Player.NATURE

    # Compute default payoffs if not provided
    if payoffs is None:
        payoffs = _default_payoffs(dag_ss, terminal_states)
    else:
        # Ensure all terminals have payoffs
        for t in terminal_states:
            if t not in payoffs:
                payoffs[t] = (0.0, 0.0)

    return ExtensiveGame(
        states=frozenset(dag_ss.states),
        terminal_states=terminal_states,
        top=dag_ss.top,
        bottom=dag_ss.bottom,
        player_map=player_map,
        actions=actions,
        payoffs=payoffs,
        is_dag=True,
        back_edges=back_edges,
    )


def _default_payoffs(
    ss: StateSpace,
    terminal_states: frozenset[int],
) -> dict[int, tuple[float, float]]:
    """Assign default payoffs: negative depth from top for both players.

    This models the cooperative goal of finishing the protocol quickly.
    """
    # BFS from top to compute depths
    depths: dict[int, int] = {ss.top: 0}
    queue: deque[int] = deque([ss.top])
    while queue:
        s = queue.popleft()
        for _, tgt in ss.enabled(s):
            if tgt not in depths:
                depths[tgt] = depths[s] + 1
                queue.append(tgt)

    payoffs: dict[int, tuple[float, float]] = {}
    for t in terminal_states:
        d = depths.get(t, 0)
        # Both players prefer shorter games: payoff = -depth
        payoffs[t] = (-float(d), -float(d))
    return payoffs


# ---------------------------------------------------------------------------
# Backward induction
# ---------------------------------------------------------------------------

def backward_induction(game: ExtensiveGame) -> BackwardInductionResult:
    """Compute the subgame perfect equilibrium via backward induction.

    Works bottom-up from terminal nodes. At each node, the controlling
    player picks the action that maximizes their own payoff component.
    """
    values: dict[int, tuple[float, float]] = {}
    best_action: dict[int, str] = {}

    # Topological order (reverse BFS from terminals)
    topo = _topological_sort(game)

    for s in topo:
        if s in game.terminal_states:
            values[s] = game.payoffs[s]
            continue

        acts = game.actions.get(s, [])
        if not acts:
            values[s] = game.payoffs.get(s, (0.0, 0.0))
            continue

        player = game.player_map.get(s, Player.NATURE)

        # Evaluate each action
        act_values: list[tuple[str, tuple[float, float]]] = []
        for label, tgt in acts:
            if tgt in values:
                act_values.append((label, values[tgt]))

        if not act_values:
            values[s] = (0.0, 0.0)
            continue

        if player == Player.CLIENT:
            # Client maximizes their own payoff (index 0)
            best_label, best_val = max(act_values, key=lambda x: x[1][0])
        elif player == Player.SERVER:
            # Server maximizes their own payoff (index 1)
            best_label, best_val = max(act_values, key=lambda x: x[1][1])
        else:
            # Nature: pick first available
            best_label, best_val = act_values[0]

        values[s] = best_val
        best_action[s] = best_label

    # Build strategy profile
    client_strategy: Strategy = {}
    server_strategy: Strategy = {}
    for s, label in best_action.items():
        player = game.player_map.get(s, Player.NATURE)
        if player == Player.CLIENT:
            client_strategy[s] = label
        elif player == Player.SERVER:
            server_strategy[s] = label

    profile: StrategyProfile = {
        Player.CLIENT: client_strategy,
        Player.SERVER: server_strategy,
    }

    # Trace equilibrium path
    eq_path = _trace_path(game, profile)

    eq_payoff = values.get(game.top, (0.0, 0.0))

    return BackwardInductionResult(
        values=values,
        strategy_profile=profile,
        equilibrium_path=eq_path,
        equilibrium_payoff=eq_payoff,
    )


def _topological_sort(game: ExtensiveGame) -> list[int]:
    """Topological sort of game states (terminals first, root last)."""
    # Kahn's algorithm (reversed)
    in_degree: dict[int, int] = {s: 0 for s in game.states}
    children: dict[int, list[int]] = {s: [] for s in game.states}
    for s in game.states:
        for _, tgt in game.actions.get(s, []):
            if tgt in in_degree:
                in_degree[tgt] += 1
                children[s].append(tgt)

    queue: deque[int] = deque()
    for s in game.states:
        if in_degree[s] == 0:
            queue.append(s)

    order: list[int] = []
    while queue:
        s = queue.popleft()
        order.append(s)
        # "Remove" edges pointing TO s: find parents
        for parent in game.states:
            for _, tgt in game.actions.get(parent, []):
                if tgt == s:
                    in_degree[parent] -= 1  # conceptually wrong

    # Use proper reverse-post-order DFS instead
    visited: set[int] = set()
    order = []

    def dfs(s: int) -> None:
        if s in visited:
            return
        visited.add(s)
        for _, tgt in game.actions.get(s, []):
            dfs(tgt)
        order.append(s)

    # Start from all states to handle disconnected components
    dfs(game.top)
    for s in game.states:
        if s not in visited:
            dfs(s)

    # order is now reverse-post-order: terminals first, root last
    return order


def _trace_path(game: ExtensiveGame, profile: StrategyProfile) -> list[int]:
    """Trace the game path determined by a strategy profile."""
    path = [game.top]
    current = game.top
    visited: set[int] = set()

    while current not in game.terminal_states:
        if current in visited:
            break  # Avoid infinite loops
        visited.add(current)

        player = game.player_map.get(current, Player.NATURE)
        strategy = profile.get(player, {})

        acts = game.actions.get(current, [])
        if not acts:
            break

        chosen_label = strategy.get(current)
        if chosen_label is not None:
            # Find the target for this label
            next_state = None
            for label, tgt in acts:
                if label == chosen_label:
                    next_state = tgt
                    break
            if next_state is not None:
                path.append(next_state)
                current = next_state
                continue

        # Fallback: pick first action
        _, next_state = acts[0]
        path.append(next_state)
        current = next_state

    return path


# ---------------------------------------------------------------------------
# Nash equilibria
# ---------------------------------------------------------------------------

def nash_equilibria(game: ExtensiveGame) -> list[StrategyProfile]:
    """Enumerate all pure-strategy Nash equilibria.

    A strategy profile is a NE if no player can improve their payoff
    by unilaterally changing their strategy.

    For small games only (exponential in number of decision nodes).
    """
    # Collect decision nodes per player
    client_nodes: list[int] = []
    server_nodes: list[int] = []
    for s, p in game.player_map.items():
        if p == Player.CLIENT and s in game.actions:
            client_nodes.append(s)
        elif p == Player.SERVER and s in game.actions:
            server_nodes.append(s)

    # Guard against combinatorial explosion
    max_strategies = 1
    for s in client_nodes + server_nodes:
        n_acts = len(game.actions.get(s, []))
        if n_acts > 0:
            max_strategies *= n_acts
        if max_strategies > 10000:
            return []  # Too many to enumerate

    # Generate all pure strategies for each player
    client_strategies = _enumerate_strategies(game, client_nodes)
    server_strategies = _enumerate_strategies(game, server_nodes)

    if not client_strategies:
        client_strategies = [{}]
    if not server_strategies:
        server_strategies = [{}]

    equilibria: list[StrategyProfile] = []

    for cs in client_strategies:
        for ss in server_strategies:
            profile: StrategyProfile = {
                Player.CLIENT: cs,
                Player.SERVER: ss,
            }
            if _is_nash_equilibrium(game, profile, client_nodes, server_nodes,
                                     client_strategies, server_strategies):
                equilibria.append(profile)

    return equilibria


def _enumerate_strategies(
    game: ExtensiveGame,
    nodes: list[int],
) -> list[Strategy]:
    """Enumerate all pure strategies for a set of decision nodes."""
    if not nodes:
        return [{}]

    choices_per_node: list[list[tuple[int, str]]] = []
    for s in nodes:
        acts = game.actions.get(s, [])
        if acts:
            choices_per_node.append([(s, label) for label, _ in acts])
        else:
            choices_per_node.append([(s, "")])

    strategies: list[Strategy] = []
    for combo in itertools_product(*choices_per_node):
        strat: Strategy = {}
        for state, label in combo:
            strat[state] = label
        strategies.append(strat)

    return strategies


def _is_nash_equilibrium(
    game: ExtensiveGame,
    profile: StrategyProfile,
    client_nodes: list[int],
    server_nodes: list[int],
    client_strategies: list[Strategy],
    server_strategies: list[Strategy],
) -> bool:
    """Check if a strategy profile is a Nash equilibrium."""
    current_payoff = _evaluate_profile(game, profile)
    if current_payoff is None:
        return False

    # Check client deviations
    for alt_cs in client_strategies:
        if alt_cs == profile[Player.CLIENT]:
            continue
        alt_profile: StrategyProfile = {
            Player.CLIENT: alt_cs,
            Player.SERVER: profile[Player.SERVER],
        }
        alt_payoff = _evaluate_profile(game, alt_profile)
        if alt_payoff is not None and alt_payoff[0] > current_payoff[0]:
            return False

    # Check server deviations
    for alt_ss in server_strategies:
        if alt_ss == profile[Player.SERVER]:
            continue
        alt_profile = {
            Player.CLIENT: profile[Player.CLIENT],
            Player.SERVER: alt_ss,
        }
        alt_payoff = _evaluate_profile(game, alt_profile)
        if alt_payoff is not None and alt_payoff[1] > current_payoff[1]:
            return False

    return True


def _evaluate_profile(
    game: ExtensiveGame,
    profile: StrategyProfile,
) -> tuple[float, float] | None:
    """Evaluate a strategy profile by tracing the game path."""
    path = _trace_path(game, profile)
    if not path:
        return None
    terminal = path[-1]
    if terminal in game.payoffs:
        return game.payoffs[terminal]
    return None


# ---------------------------------------------------------------------------
# Dominant strategies
# ---------------------------------------------------------------------------

def dominant_strategy(game: ExtensiveGame, player: Player) -> Strategy | None:
    """Find a dominant strategy for the given player, or None.

    A strategy is dominant if it is at least as good as any alternative
    for every possible opponent strategy.
    """
    if player == Player.CLIENT:
        my_nodes = [s for s, p in game.player_map.items()
                    if p == Player.CLIENT and s in game.actions]
        opp_nodes = [s for s, p in game.player_map.items()
                     if p == Player.SERVER and s in game.actions]
        my_idx = 0
    else:
        my_nodes = [s for s, p in game.player_map.items()
                    if p == Player.SERVER and s in game.actions]
        opp_nodes = [s for s, p in game.player_map.items()
                     if p == Player.CLIENT and s in game.actions]
        my_idx = 1

    my_strategies = _enumerate_strategies(game, my_nodes)
    opp_strategies = _enumerate_strategies(game, opp_nodes)

    if not my_strategies or not opp_strategies:
        return my_strategies[0] if my_strategies else None

    # Guard against explosion
    if len(my_strategies) * len(opp_strategies) > 10000:
        return None

    for candidate in my_strategies:
        is_dominant = True
        for alt in my_strategies:
            if alt == candidate:
                continue
            # Check that candidate >= alt for all opponent strategies
            for opp in opp_strategies:
                if player == Player.CLIENT:
                    prof_cand: StrategyProfile = {Player.CLIENT: candidate, Player.SERVER: opp}
                    prof_alt: StrategyProfile = {Player.CLIENT: alt, Player.SERVER: opp}
                else:
                    prof_cand = {Player.CLIENT: opp, Player.SERVER: candidate}
                    prof_alt = {Player.CLIENT: opp, Player.SERVER: alt}
                pay_cand = _evaluate_profile(game, prof_cand)
                pay_alt = _evaluate_profile(game, prof_alt)
                if pay_cand is None or pay_alt is None:
                    continue
                if pay_cand[my_idx] < pay_alt[my_idx]:
                    is_dominant = False
                    break
            if not is_dominant:
                break
        if is_dominant:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Game value (minimax)
# ---------------------------------------------------------------------------

def game_value(game: ExtensiveGame) -> float:
    """Compute the minimax game value (adversarial mode).

    Client minimizes path length, Server maximizes it.
    Returns the game value from Client's perspective (path length).
    """
    values: dict[int, float] = {}
    topo = _topological_sort(game)

    for s in topo:
        if s in game.terminal_states:
            # Use client payoff as value
            values[s] = game.payoffs.get(s, (0.0, 0.0))[0]
            continue

        acts = game.actions.get(s, [])
        child_vals = [values[tgt] for _, tgt in acts if tgt in values]
        if not child_vals:
            values[s] = 0.0
            continue

        player = game.player_map.get(s, Player.NATURE)
        if player == Player.CLIENT:
            # Client maximizes their payoff (which is negative path length)
            values[s] = max(child_vals)
        elif player == Player.SERVER:
            # Server minimizes client's payoff (adversarial)
            values[s] = min(child_vals)
        else:
            values[s] = child_vals[0]

    return values.get(game.top, 0.0)


# ---------------------------------------------------------------------------
# Zero-sum check
# ---------------------------------------------------------------------------

def is_zero_sum(game: ExtensiveGame) -> bool:
    """Check whether the game is zero-sum (payoffs sum to zero at all terminals)."""
    for t in game.terminal_states:
        p = game.payoffs.get(t, (0.0, 0.0))
        if abs(p[0] + p[1]) > 1e-9:
            return False
    return True


# ---------------------------------------------------------------------------
# Social welfare
# ---------------------------------------------------------------------------

def social_welfare(game: ExtensiveGame, profile: StrategyProfile) -> float:
    """Compute social welfare (sum of all players' payoffs) under a strategy profile."""
    payoff = _evaluate_profile(game, profile)
    if payoff is None:
        return 0.0
    return payoff[0] + payoff[1]


# ---------------------------------------------------------------------------
# Cooperative value
# ---------------------------------------------------------------------------

def cooperative_value(game: ExtensiveGame) -> float:
    """Compute the cooperative game value (shortest path from top to bottom).

    Both players cooperate to finish the protocol as quickly as possible.
    Returns the maximum social welfare achievable.
    """
    # BFS from top to find shortest path to any terminal
    dist: dict[int, int] = {game.top: 0}
    queue: deque[int] = deque([game.top])

    while queue:
        s = queue.popleft()
        for _, tgt in game.actions.get(s, []):
            if tgt not in dist:
                dist[tgt] = dist[s] + 1
                queue.append(tgt)

    # Find the terminal with the best (highest) sum of payoffs
    best_welfare = float('-inf')
    for t in game.terminal_states:
        if t in dist:
            p = game.payoffs.get(t, (0.0, 0.0))
            w = p[0] + p[1]
            if w > best_welfare:
                best_welfare = w

    return best_welfare if best_welfare != float('-inf') else 0.0


# ---------------------------------------------------------------------------
# Price of anarchy / stability
# ---------------------------------------------------------------------------

def price_of_anarchy(game: ExtensiveGame) -> float | None:
    """Compute the price of anarchy: ratio of worst NE welfare to optimal.

    PoA = optimal_welfare / worst_NE_welfare (>= 1 means NE is suboptimal).
    Returns None if no NE found or welfare is zero.
    """
    eqs = nash_equilibria(game)
    if not eqs:
        return None

    opt = cooperative_value(game)
    if abs(opt) < 1e-9:
        return None

    worst_ne_welfare = min(social_welfare(game, eq) for eq in eqs)
    if abs(worst_ne_welfare) < 1e-9:
        return None

    # PoA uses absolute values since payoffs may be negative
    return opt / worst_ne_welfare


def price_of_stability(game: ExtensiveGame) -> float | None:
    """Compute the price of stability: ratio of best NE welfare to optimal.

    PoS = optimal_welfare / best_NE_welfare (>= 1 means even best NE is suboptimal).
    Returns None if no NE found or welfare is zero.
    """
    eqs = nash_equilibria(game)
    if not eqs:
        return None

    opt = cooperative_value(game)
    if abs(opt) < 1e-9:
        return None

    best_ne_welfare = max(social_welfare(game, eq) for eq in eqs)
    if abs(best_ne_welfare) < 1e-9:
        return None

    return opt / best_ne_welfare


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_game_theory(
    ss: StateSpace,
    payoffs: dict[int, tuple[float, float]] | None = None,
) -> GameAnalysis:
    """Perform full game-theoretic analysis of a session type state space."""
    game = build_game(ss, payoffs)
    bi = backward_induction(game)
    gv = game_value(game)
    cv = cooperative_value(game)
    zs = is_zero_sum(game)

    # Count strategies
    client_nodes = [s for s, p in game.player_map.items()
                    if p == Player.CLIENT and s in game.actions]
    server_nodes = [s for s, p in game.player_map.items()
                    if p == Player.SERVER and s in game.actions]

    n_client = 1
    for s in client_nodes:
        n_acts = len(game.actions.get(s, []))
        if n_acts > 0:
            n_client *= n_acts

    n_server = 1
    for s in server_nodes:
        n_acts = len(game.actions.get(s, []))
        if n_acts > 0:
            n_server *= n_acts

    # Nash equilibria (only for small games)
    eqs: list[StrategyProfile] | None = None
    if n_client * n_server <= 10000:
        eqs = nash_equilibria(game)

    poa = price_of_anarchy(game)
    pos = price_of_stability(game)

    # Dominant strategies
    dom_client: Strategy | None = None
    dom_server: Strategy | None = None
    if n_client * n_server <= 10000:
        dom_client = dominant_strategy(game, Player.CLIENT)
        dom_server = dominant_strategy(game, Player.SERVER)

    return GameAnalysis(
        game=game,
        backward_induction=bi,
        game_value=gv,
        cooperative_value=cv,
        is_zero_sum=zs,
        num_strategies_client=n_client,
        num_strategies_server=n_server,
        price_of_anarchy=poa,
        price_of_stability=pos,
        nash_equilibria=eqs,
        dominant_client=dom_client,
        dominant_server=dom_server,
    )
