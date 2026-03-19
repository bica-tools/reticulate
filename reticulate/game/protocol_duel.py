"""Protocol Duel — a two-player game on session type state spaces.

The game board is the Hasse diagram of a session type's state space.
Two players — Client (external choice) and Server (internal choice) —
take turns moving a token from ⊤ to ⊥.

At branch nodes (&{...}): the Client chooses.
At selection nodes (+{...}): the Server chooses.
At single-transition nodes: the token moves automatically.

The game ends when the token reaches ⊥ (end).

Modes:
  - cooperative: both players try to reach ⊥ in minimum moves
  - adversarial: one player maximizes moves, the other minimizes
  - solo: one human player, the other is AI (random or strategic)

Usage:
  python -m reticulate.game.protocol_duel "&{a: +{OK: end, ERR: end}, b: end}"
  python -m reticulate.game.protocol_duel --benchmark SMTP
  python -m reticulate.game.protocol_duel --benchmark "Java Iterator" --mode adversarial
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Literal

from reticulate.parser import parse, pretty
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Current state of a Protocol Duel game."""
    ss: StateSpace
    current: int
    moves: list[tuple[str, int]]  # (label, target) history
    turn_count: int
    mode: Literal["cooperative", "adversarial", "solo"]
    human_role: Literal["client", "server", "both"]

    @property
    def is_finished(self) -> bool:
        return self.current == self.ss.bottom

    @property
    def available_moves(self) -> list[tuple[str, int]]:
        """Return (label, target) pairs available from current state."""
        return [
            (label, target)
            for src, label, target in self.ss.transitions
            if src == self.current
        ]

    @property
    def is_branch(self) -> bool:
        """True if the current state is a branch (Client's turn)."""
        moves = self.available_moves
        if not moves:
            return False
        return not any(
            self.ss.is_selection(self.current, label, target)
            for label, target in moves
        )

    @property
    def is_selection(self) -> bool:
        """True if the current state is a selection (Server's turn)."""
        moves = self.available_moves
        if not moves:
            return False
        return any(
            self.ss.is_selection(self.current, label, target)
            for label, target in moves
        )

    @property
    def whose_turn(self) -> str:
        """Return 'client', 'server', or 'auto'."""
        moves = self.available_moves
        if len(moves) == 0:
            return "auto"  # at bottom
        if len(moves) == 1:
            return "auto"  # forced move
        if self.is_selection:
            return "server"
        return "client"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

BLUE = "\033[94m"
ORANGE = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def state_label(ss: StateSpace, state_id: int) -> str:
    """Human-readable label for a state."""
    label = ss.labels.get(state_id, str(state_id))
    if state_id == ss.top:
        return f"⊤ ({label})"
    if state_id == ss.bottom:
        return f"⊥ (end)"
    return label


def display_board(gs: GameState) -> None:
    """Print the current game state."""
    ss = gs.ss
    pos = gs.current

    print()
    print(f"  {BOLD}═══ PROTOCOL DUEL ═══{RESET}")
    print(f"  Turn {gs.turn_count} │ Mode: {gs.mode}")
    print()

    # Show current position
    label = state_label(ss, pos)
    if pos == ss.top:
        print(f"  📍 You are at: {BLUE}{BOLD}{label}{RESET}")
    elif pos == ss.bottom:
        print(f"  🏁 You are at: {GREEN}{BOLD}{label}{RESET}")
    else:
        print(f"  📍 You are at: {BOLD}{label}{RESET}")

    # Show whose turn
    turn = gs.whose_turn
    if turn == "client":
        print(f"  🔵 {BLUE}Client's turn{RESET} (external choice — client picks the method)")
    elif turn == "server":
        print(f"  🟠 {ORANGE}Server's turn{RESET} (internal choice — server picks the outcome)")
    elif not gs.is_finished:
        print(f"  ⚡ Automatic move (only one option)")

    # Show available moves
    moves = gs.available_moves
    if moves:
        print()
        print(f"  Available moves:")
        for i, (lbl, tgt) in enumerate(moves, 1):
            tgt_label = state_label(ss, tgt)
            if gs.is_selection:
                print(f"    {ORANGE}{i}. {lbl}{RESET} → {tgt_label}")
            else:
                print(f"    {BLUE}{i}. {lbl}{RESET} → {tgt_label}")

    # Show move history
    if gs.moves:
        path = " → ".join(lbl for lbl, _ in gs.moves)
        print(f"\n  {DIM}Path so far: {path}{RESET}")


def display_victory(gs: GameState) -> None:
    """Print the victory screen."""
    print()
    print(f"  {GREEN}{BOLD}╔═══════════════════════════════╗{RESET}")
    print(f"  {GREEN}{BOLD}║   🏆 PROTOCOL COMPLETED! 🏆   ║{RESET}")
    print(f"  {GREEN}{BOLD}╚═══════════════════════════════╝{RESET}")
    print()
    print(f"  Reached ⊥ (end) in {gs.turn_count} moves.")
    path = " → ".join(lbl for lbl, _ in gs.moves)
    print(f"  Path: ⊤ → {path} → ⊥")
    print()

    # Score
    n_states = len(gs.ss.states)
    efficiency = max(0, 100 - (gs.turn_count - 1) * 10)
    print(f"  Protocol size: {n_states} states")
    print(f"  Your moves:    {gs.turn_count}")
    print(f"  Efficiency:    {efficiency}%")
    print()


# ---------------------------------------------------------------------------
# AI players
# ---------------------------------------------------------------------------

def ai_random_move(gs: GameState) -> tuple[str, int]:
    """Random AI: pick a random available move."""
    return random.choice(gs.available_moves)


def ai_shortest_path(gs: GameState) -> tuple[str, int]:
    """Greedy AI: pick the move closest to ⊥ (by BFS distance)."""
    ss = gs.ss
    moves = gs.available_moves

    # BFS from each target to bottom
    best = moves[0]
    best_dist = float("inf")

    for label, target in moves:
        # BFS from target
        dist = _bfs_distance(ss, target, ss.bottom)
        if dist < best_dist:
            best_dist = dist
            best = (label, target)

    return best


def ai_longest_path(gs: GameState) -> tuple[str, int]:
    """Adversarial AI: pick the move farthest from ⊥."""
    ss = gs.ss
    moves = gs.available_moves

    best = moves[0]
    best_dist = -1

    for label, target in moves:
        dist = _bfs_distance(ss, target, ss.bottom)
        if dist > best_dist:
            best_dist = dist
            best = (label, target)

    return best


def _bfs_distance(ss: StateSpace, start: int, end: int) -> int:
    """BFS distance from start to end in the state space."""
    if start == end:
        return 0
    visited = {start}
    queue = [(start, 0)]
    adj: dict[int, list[int]] = {}
    for s, _, t in ss.transitions:
        adj.setdefault(s, []).append(t)

    while queue:
        node, dist = queue.pop(0)
        for neighbor in adj.get(node, []):
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return 999  # unreachable


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_game(
    type_string: str,
    mode: str = "cooperative",
    human_role: str = "both",
) -> GameState:
    """Run an interactive Protocol Duel game.

    Parameters:
        type_string: Session type to play on.
        mode: "cooperative", "adversarial", or "solo".
        human_role: "client", "server", or "both".

    Returns:
        The final GameState.
    """
    ast = parse(type_string)
    ss = build_statespace(ast)
    result = check_lattice(ss)

    print(f"\n  {BOLD}Protocol Duel{RESET}")
    print(f"  Type: {pretty(ast)}")
    print(f"  States: {len(ss.states)}, Transitions: {len(ss.transitions)}")
    print(f"  Lattice: {result.is_lattice}")
    print(f"  Mode: {mode} │ You play: {human_role}")

    gs = GameState(
        ss=ss,
        current=ss.top,
        moves=[],
        turn_count=0,
        mode=mode,
        human_role=human_role,
    )

    while not gs.is_finished:
        display_board(gs)
        moves = gs.available_moves

        if not moves:
            print(f"\n  {RED}STUCK! No moves available (this shouldn't happen in a lattice).{RESET}")
            break

        turn = gs.whose_turn

        if turn == "auto":
            # Forced move
            label, target = moves[0]
            print(f"\n  ⚡ Auto: {label}")
            gs.moves.append((label, target))
            gs.current = target
            gs.turn_count += 1
            continue

        # Determine if human or AI moves
        is_human = (
            gs.human_role == "both"
            or (gs.human_role == "client" and turn == "client")
            or (gs.human_role == "server" and turn == "server")
        )

        if is_human:
            # Human input
            while True:
                try:
                    choice = input(f"\n  Your choice (1-{len(moves)}): ").strip()
                    if choice.lower() in ("q", "quit", "exit"):
                        print(f"\n  {DIM}Game abandoned.{RESET}")
                        return gs
                    idx = int(choice) - 1
                    if 0 <= idx < len(moves):
                        break
                    print(f"  {RED}Invalid choice. Try 1-{len(moves)}.{RESET}")
                except (ValueError, EOFError):
                    print(f"  {RED}Enter a number 1-{len(moves)}, or 'q' to quit.{RESET}")

            label, target = moves[idx]
        else:
            # AI move
            if mode == "adversarial":
                if turn == "server":
                    label, target = ai_longest_path(gs)
                else:
                    label, target = ai_shortest_path(gs)
            else:
                label, target = ai_shortest_path(gs)

            role_color = ORANGE if turn == "server" else BLUE
            print(f"\n  🤖 {role_color}AI ({turn}){RESET} chooses: {label}")

        gs.moves.append((label, target))
        gs.current = target
        gs.turn_count += 1

    display_victory(gs)
    return gs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for Protocol Duel."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Protocol Duel: a two-player game on session type lattices",
    )
    parser.add_argument(
        "type_string",
        nargs="?",
        help="Session type to play on (e.g., '&{a: +{OK: end, ERR: end}, b: end}')",
    )
    parser.add_argument(
        "--benchmark", "-b",
        help="Use a named benchmark protocol (e.g., 'SMTP', 'ATM', 'Java Iterator')",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["cooperative", "adversarial", "solo"],
        default="cooperative",
        help="Game mode (default: cooperative)",
    )
    parser.add_argument(
        "--role", "-r",
        choices=["client", "server", "both"],
        default="both",
        help="Which role the human plays (default: both)",
    )
    parser.add_argument(
        "--visual", "-v",
        action="store_true",
        help="Open interactive visual board in the browser",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for visual board HTML (used with --visual)",
    )
    parser.add_argument(
        "--list-benchmarks", "-l",
        action="store_true",
        help="List available benchmark protocols",
    )

    args = parser.parse_args(argv)

    if args.list_benchmarks:
        try:
            from tests.benchmarks.protocols import BENCHMARKS
            print("\nAvailable benchmarks:")
            for b in BENCHMARKS:
                print(f"  {b.name:30s} ({b.expected_states} states)")
        except ImportError:
            print("Benchmarks not available (run from reticulate directory)")
        return

    type_string = args.type_string

    if args.benchmark:
        try:
            from tests.benchmarks.protocols import BENCHMARKS
            for b in BENCHMARKS:
                if b.name.lower() == args.benchmark.lower():
                    type_string = b.type_string
                    break
            else:
                print(f"Benchmark '{args.benchmark}' not found. Use --list-benchmarks.")
                sys.exit(1)
        except ImportError:
            print("Benchmarks not available (run from reticulate directory)")
            sys.exit(1)

    if not type_string:
        # Default: a fun introductory type
        type_string = "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"
        print(f"  {DIM}No type specified — using File Object protocol.{RESET}")

    if args.visual:
        from reticulate.parser import parse as _parse
        from reticulate.statespace import build_statespace as _build
        from reticulate.game.visual_board import open_visual_board

        ast = _parse(type_string)
        ss = _build(ast)
        path = open_visual_board(
            ss, title="Protocol Duel", mode=args.mode,
            human_role=args.role, output_path=args.output,
        )
        print(f"  Visual board opened: {path}")
        return

    play_game(type_string, mode=args.mode, human_role=args.role)


if __name__ == "__main__":
    main()
