"""Game-semantic interpretation of session types (Step 94).

Interprets session types as two-player games between Proponent (the process)
and Opponent (the environment/client).

Key correspondences:
  - Branch states → Opponent moves (external choice: environment decides)
  - Selection states → Proponent moves (internal choice: process decides)
  - Terminal state → game over (protocol completed)
  - Winning = reaching the terminal state (deadlock-free implementation)
  - Parallel composition → product (tensor) of games

The arena is derived directly from the session-type state space, with
polarity (P/O) determined by whether transitions are selection or branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product as cartesian_product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class Player(Enum):
    """The two players in a session game."""
    PROPONENT = "P"
    OPPONENT = "O"


class GameOutcome(Enum):
    """Who has a winning strategy."""
    PROPONENT_WINS = "P_wins"
    OPPONENT_WINS = "O_wins"
    DRAW = "draw"  # neither has a winning strategy (infinite games)


@dataclass(frozen=True)
class Move:
    """A move in the game arena.

    Attributes:
        source: state the move originates from.
        label: transition label (method name / selection label).
        target: state the move leads to.
        player: which player makes this move.
    """
    source: int
    label: str
    target: int
    player: Player


@dataclass(frozen=True)
class Arena:
    """Game arena derived from a session-type state space.

    Attributes:
        positions: set of game positions (= state-space states).
        moves: list of all moves.
        initial: starting position (= state-space top).
        terminal: set of terminal positions (at minimum, bottom).
        polarity: mapping from position to the player who moves there.
            Positions with no outgoing moves are terminal (not in polarity).
    """
    positions: frozenset[int]
    moves: tuple[Move, ...]
    initial: int
    terminal: frozenset[int]
    polarity: dict[int, Player]

    # Convenience helpers

    def moves_from(self, position: int) -> list[Move]:
        """Return all moves available from *position*."""
        return [m for m in self.moves if m.source == position]

    def is_terminal(self, position: int) -> bool:
        """True if *position* is terminal (no outgoing moves or is bottom)."""
        return position in self.terminal

    def opponent_positions(self) -> frozenset[int]:
        """Positions where Opponent moves."""
        return frozenset(p for p, pl in self.polarity.items() if pl == Player.OPPONENT)

    def proponent_positions(self) -> frozenset[int]:
        """Positions where Proponent moves."""
        return frozenset(p for p, pl in self.polarity.items() if pl == Player.PROPONENT)

    @property
    def depth(self) -> int:
        """Longest path from initial to any terminal position.

        Uses DFS with memoization. For cyclic arenas (recursive types),
        uses visited-set to avoid infinite loops and returns the longest
        acyclic path.
        """
        memo: dict[int, int] = {}

        def _longest(pos: int, on_path: frozenset[int]) -> int:
            if pos in self.terminal:
                return 0
            if pos in memo and not (on_path & {pos}):
                return memo[pos]
            moves = self.moves_from(pos)
            if not moves:
                return 0
            best = 0
            for m in moves:
                if m.target in on_path:
                    continue  # skip cycles
                val = 1 + _longest(m.target, on_path | {pos})
                if val > best:
                    best = val
            memo[pos] = best
            return best

        return _longest(self.initial, frozenset())


# A Play is a sequence of moves
Play = tuple[Move, ...]


@dataclass(frozen=True)
class Strategy:
    """A deterministic Proponent strategy.

    A strategy maps each reachable Proponent position to the chosen move.
    At Opponent positions the strategy does not prescribe anything (the
    environment decides).

    Attributes:
        choices: mapping from Proponent position to the Move chosen.
    """
    choices: dict[int, Move]

    def __call__(self, position: int) -> Move | None:
        """Return the move prescribed at *position*, or None if not a P-position."""
        return self.choices.get(position)

    @property
    def labels(self) -> dict[int, str]:
        """Position → label chosen."""
        return {pos: m.label for pos, m in self.choices.items()}


@dataclass(frozen=True)
class CounterStrategy:
    """A deterministic Opponent counter-strategy.

    Maps each reachable Opponent position to the chosen move.

    Attributes:
        choices: mapping from Opponent position to the Move chosen.
    """
    choices: dict[int, Move]

    def __call__(self, position: int) -> Move | None:
        return self.choices.get(position)


@dataclass(frozen=True)
class GameAnalysis:
    """Full game-semantic analysis of a session type.

    Attributes:
        arena: the game arena.
        outcome: who has a winning strategy.
        num_proponent_positions: number of P-positions.
        num_opponent_positions: number of O-positions.
        num_terminal_positions: number of terminal positions.
        num_winning_strategies: total number of winning P-strategies.
        num_total_strategies: total number of deterministic P-strategies.
        is_determinate: True iff Proponent has a winning strategy.
        example_winning_strategy: one winning strategy, or None.
    """
    arena: Arena
    outcome: GameOutcome
    num_proponent_positions: int
    num_opponent_positions: int
    num_terminal_positions: int
    num_winning_strategies: int
    num_total_strategies: int
    is_determinate: bool
    example_winning_strategy: Strategy | None


# ---------------------------------------------------------------------------
# Arena construction
# ---------------------------------------------------------------------------

def build_arena(ss: StateSpace) -> Arena:
    """Construct a game arena from a session-type state space.

    Branch transitions → Opponent moves.
    Selection transitions → Proponent moves.
    States with no outgoing transitions are terminal.
    """
    moves: list[Move] = []
    polarity: dict[int, Player] = {}

    for src, lbl, tgt in ss.transitions:
        if ss.is_selection(src, lbl, tgt):
            player = Player.PROPONENT
        else:
            player = Player.OPPONENT
        moves.append(Move(source=src, label=lbl, target=tgt, player=player))

    # Determine polarity for each non-terminal position
    for state in ss.states:
        outgoing = [(src, lbl, tgt) for src, lbl, tgt in ss.transitions if src == state]
        if not outgoing:
            continue  # terminal — no polarity
        # Check if ALL outgoing transitions are selection
        all_sel = all(ss.is_selection(s, l, t) for s, l, t in outgoing)
        any_sel = any(ss.is_selection(s, l, t) for s, l, t in outgoing)
        if all_sel:
            polarity[state] = Player.PROPONENT
        elif not any_sel:
            polarity[state] = Player.OPPONENT
        else:
            # Mixed: product states — treat as Proponent for selections,
            # but in practice this is a concurrent position. We assign
            # based on majority; or more precisely, mixed positions are
            # treated as Opponent (environment can make any available move).
            polarity[state] = Player.OPPONENT

    # Terminal positions: states with no outgoing transitions
    terminal_set: set[int] = set()
    states_with_outgoing = {src for src, _, _ in ss.transitions}
    for state in ss.states:
        if state not in states_with_outgoing:
            terminal_set.add(state)

    return Arena(
        positions=frozenset(ss.states),
        moves=tuple(moves),
        initial=ss.top,
        terminal=frozenset(terminal_set),
        polarity=polarity,
    )


# ---------------------------------------------------------------------------
# Strategy enumeration and checking
# ---------------------------------------------------------------------------

def _reachable_proponent_positions(arena: Arena) -> set[int]:
    """Find all Proponent positions reachable from initial."""
    reachable: set[int] = set()
    visited: set[int] = set()
    stack = [arena.initial]
    while stack:
        pos = stack.pop()
        if pos in visited:
            continue
        visited.add(pos)
        if arena.polarity.get(pos) == Player.PROPONENT:
            reachable.add(pos)
        for m in arena.moves_from(pos):
            stack.append(m.target)
    return reachable


def _reachable_opponent_positions(arena: Arena) -> set[int]:
    """Find all Opponent positions reachable from initial."""
    reachable: set[int] = set()
    visited: set[int] = set()
    stack = [arena.initial]
    while stack:
        pos = stack.pop()
        if pos in visited:
            continue
        visited.add(pos)
        if arena.polarity.get(pos) == Player.OPPONENT:
            reachable.add(pos)
        for m in arena.moves_from(pos):
            stack.append(m.target)
    return reachable


def enumerate_strategies(arena: Arena) -> list[Strategy]:
    """Enumerate all deterministic Proponent strategies.

    A strategy assigns one move at each reachable Proponent position.
    """
    p_positions = _reachable_proponent_positions(arena)
    if not p_positions:
        # No Proponent positions: the single "empty" strategy
        return [Strategy(choices={})]

    # For each P-position, collect available moves
    choices_per_pos: dict[int, list[Move]] = {}
    for pos in sorted(p_positions):
        moves = arena.moves_from(pos)
        if moves:
            choices_per_pos[pos] = moves

    if not choices_per_pos:
        return [Strategy(choices={})]

    # Cartesian product of choices
    positions = sorted(choices_per_pos.keys())
    move_lists = [choices_per_pos[p] for p in positions]

    strategies: list[Strategy] = []
    for combo in cartesian_product(*move_lists):
        choices = {pos: move for pos, move in zip(positions, combo)}
        strategies.append(Strategy(choices=choices))

    return strategies


def enumerate_counter_strategies(arena: Arena) -> list[CounterStrategy]:
    """Enumerate all deterministic Opponent counter-strategies."""
    o_positions = _reachable_opponent_positions(arena)
    if not o_positions:
        return [CounterStrategy(choices={})]

    choices_per_pos: dict[int, list[Move]] = {}
    for pos in sorted(o_positions):
        moves = arena.moves_from(pos)
        if moves:
            choices_per_pos[pos] = moves

    if not choices_per_pos:
        return [CounterStrategy(choices={})]

    positions = sorted(choices_per_pos.keys())
    move_lists = [choices_per_pos[p] for p in positions]

    counter_strategies: list[CounterStrategy] = []
    for combo in cartesian_product(*move_lists):
        choices = {pos: move for pos, move in zip(positions, combo)}
        counter_strategies.append(CounterStrategy(choices=choices))

    return counter_strategies


def play_game(
    arena: Arena,
    strategy: Strategy,
    counter_strategy: CounterStrategy,
    max_steps: int = 1000,
) -> Play:
    """Execute a play given both strategies, returning the sequence of moves.

    Stops when a terminal position is reached or after *max_steps*.
    """
    moves: list[Move] = []
    position = arena.initial
    visited_positions: list[int] = [position]

    for _ in range(max_steps):
        if arena.is_terminal(position):
            break

        player = arena.polarity.get(position)
        if player == Player.PROPONENT:
            move = strategy(position)
        elif player == Player.OPPONENT:
            move = counter_strategy(position)
        else:
            break  # no polarity → stuck

        if move is None:
            break  # strategy doesn't cover this position

        moves.append(move)
        position = move.target

        # Detect infinite loop (cycle)
        if position in visited_positions:
            # We've looped — add the move but stop
            break
        visited_positions.append(position)

    return tuple(moves)


def is_winning(strategy: Strategy, arena: Arena, max_steps: int = 1000) -> bool:
    """Check if a Proponent strategy is winning against ALL Opponent counter-strategies.

    A strategy is winning iff every possible play reaches a terminal position.
    """
    counter_strategies = enumerate_counter_strategies(arena)
    for cs in counter_strategies:
        play = play_game(arena, strategy, cs, max_steps=max_steps)
        # Check that the play ended at a terminal position
        if not play:
            # No moves played — initial position must be terminal
            if not arena.is_terminal(arena.initial):
                return False
            continue
        final_pos = play[-1].target
        if not arena.is_terminal(final_pos):
            return False
    return True


def winning_strategies(arena: Arena) -> list[Strategy]:
    """Return all winning Proponent strategies."""
    all_strats = enumerate_strategies(arena)
    return [s for s in all_strats if is_winning(s, arena)]


def game_value(arena: Arena) -> GameOutcome:
    """Determine who has a winning strategy.

    Returns PROPONENT_WINS if any winning strategy exists for P,
    OPPONENT_WINS if O can prevent P from winning regardless of P's strategy,
    DRAW if neither (possible with infinite games from recursion).
    """
    # Check for Proponent winning
    all_strats = enumerate_strategies(arena)
    p_wins = any(is_winning(s, arena) for s in all_strats)

    if p_wins:
        return GameOutcome.PROPONENT_WINS

    # Check if Opponent can win (prevent termination) against all P strategies
    # Opponent "wins" if for every P strategy, there exists a counter-strategy
    # that does NOT reach terminal.
    o_wins = True
    counter_strats = enumerate_counter_strategies(arena)
    for strat in all_strats:
        # Does some counter-strategy prevent termination?
        all_reach_terminal = True
        for cs in counter_strats:
            play = play_game(arena, strat, cs)
            if play:
                final = play[-1].target
                if not arena.is_terminal(final):
                    all_reach_terminal = False
                    break
            elif not arena.is_terminal(arena.initial):
                all_reach_terminal = False
                break
        if all_reach_terminal:
            # This strategy reaches terminal against ALL counter-strategies
            # so Opponent cannot prevent termination
            o_wins = False
            break

    if o_wins:
        return GameOutcome.OPPONENT_WINS

    return GameOutcome.DRAW


# ---------------------------------------------------------------------------
# Arena composition (tensor product for parallel)
# ---------------------------------------------------------------------------

def compose_arenas(a1: Arena, a2: Arena) -> Arena:
    """Compute the tensor product of two arenas (parallel composition).

    Positions are pairs (p1, p2). Moves interleave: at each position,
    either component can make a move independently.
    """
    # Create composite positions
    positions: set[int] = set()
    pos_map: dict[tuple[int, int], int] = {}
    next_id = 0

    for p1 in sorted(a1.positions):
        for p2 in sorted(a2.positions):
            pos_map[(p1, p2)] = next_id
            positions.add(next_id)
            next_id += 1

    initial = pos_map[(a1.initial, a2.initial)]

    # Terminal: both components terminal
    terminal: set[int] = set()
    for p1 in a1.terminal:
        for p2 in a2.terminal:
            terminal.add(pos_map[(p1, p2)])

    # Moves: interleaving from both components
    moves: list[Move] = []
    polarity: dict[int, Player] = {}

    for (p1, p2), pid in pos_map.items():
        # Moves from component 1 (if p1 is not terminal in a1)
        for m in a1.moves_from(p1):
            target = pos_map[(m.target, p2)]
            moves.append(Move(source=pid, label=m.label, target=target, player=m.player))

        # Moves from component 2 (if p2 is not terminal in a2)
        for m in a2.moves_from(p2):
            target = pos_map[(p1, m.target)]
            moves.append(Move(source=pid, label=m.label, target=target, player=m.player))

        # Polarity: determined by available moves
        outgoing = [mv for mv in moves if mv.source == pid]
        if outgoing:
            all_p = all(mv.player == Player.PROPONENT for mv in outgoing)
            if all_p:
                polarity[pid] = Player.PROPONENT
            else:
                polarity[pid] = Player.OPPONENT

    # Recompute polarity after all moves are built
    polarity_final: dict[int, Player] = {}
    for pid in positions:
        outgoing = [m for m in moves if m.source == pid]
        if not outgoing:
            continue
        all_p = all(m.player == Player.PROPONENT for m in outgoing)
        if all_p:
            polarity_final[pid] = Player.PROPONENT
        else:
            polarity_final[pid] = Player.OPPONENT

    return Arena(
        positions=frozenset(positions),
        moves=tuple(moves),
        initial=initial,
        terminal=frozenset(terminal),
        polarity=polarity_final,
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_game(ss: StateSpace) -> GameAnalysis:
    """Perform a complete game-semantic analysis of a session-type state space."""
    arena = build_arena(ss)
    outcome = game_value(arena)

    all_strats = enumerate_strategies(arena)
    win_strats = [s for s in all_strats if is_winning(s, arena)]

    return GameAnalysis(
        arena=arena,
        outcome=outcome,
        num_proponent_positions=len(arena.proponent_positions()),
        num_opponent_positions=len(arena.opponent_positions()),
        num_terminal_positions=len(arena.terminal),
        num_winning_strategies=len(win_strats),
        num_total_strategies=len(all_strats),
        is_determinate=outcome == GameOutcome.PROPONENT_WINS,
        example_winning_strategy=win_strats[0] if win_strats else None,
    )
