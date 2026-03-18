#!/usr/bin/env python3
"""
████████╗██╗  ██╗███████╗
╚══██╔══╝██║  ██║██╔════╝
   ██║   ███████║█████╗
   ██║   ██╔══██║██╔══╝
   ██║   ██║  ██║███████╗
   ╚═╝   ╚═╝  ╚═╝╚══════╝
██████╗ ███████╗████████╗██╗ ██████╗██╗   ██╗██╗      █████╗ ████████╗███████╗
██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██║   ██║██║     ██╔══██╗╚══██╔══╝██╔════╝
██████╔╝█████╗     ██║   ██║██║     ██║   ██║██║     ███████║   ██║   █████╗
██╔══██╗██╔══╝     ██║   ██║██║     ██║   ██║██║     ██╔══██║   ██║   ██╔══╝
██║  ██║███████╗   ██║   ██║╚██████╗╚██████╔╝███████╗██║  ██║   ██║   ███████╗
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝

A dungeon crawler where the dungeon is a lattice.

You descend through a structure you don't fully understand.
Each room offers choices — but not always YOUR choices.
Some doors you open. Some doors open themselves.
The deeper you go, the closer to the exit.
But the path is never straight.

"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field

# ── Theming ──────────────────────────────────────────────────────────────

ROOMS = {
    "entrance": ["The Gate of Beginning", "The First Chamber", "The Hall of Origins"],
    "branch": [
        "A crossroads of stone corridors",
        "A chamber with multiple archways",
        "A room where torchlight splits into paths",
        "A hollow where echoes come from every direction",
        "A junction carved by ancient hands",
        "A vault with doors of different metals",
    ],
    "selection": [
        "A chamber where the walls shift",
        "A room where the floor decides your fate",
        "A hall where shadows choose for you",
        "A place where the wind pushes you forward",
        "A court where invisible hands guide the way",
        "A sanctum where destiny reveals itself",
    ],
    "corridor": [
        "A narrow passage descending",
        "A spiral staircase going down",
        "A bridge over darkness",
        "A tunnel carved through crystal",
    ],
    "loop": [
        "You feel a strange familiarity...",
        "The walls look the same as before.",
        "Déjà vu. You've been here.",
        "The corridor curves back on itself.",
    ],
    "exit": [
        "The Gate of Ending",
        "Light floods in from above",
        "The final chamber — open sky beyond",
    ],
}

DOOR_NAMES = {
    "a": "the iron door",
    "b": "the wooden door",
    "c": "the crystal door",
    "d": "the shadow door",
    "open": "the entrance gate",
    "close": "the exit seal",
    "read": "the library passage",
    "write": "the scriptorium tunnel",
    "send": "the messenger's corridor",
    "receive": "the listener's alcove",
    "connect": "the bridge of connection",
    "disconnect": "the severing gate",
    "login": "the authentication arch",
    "logout": "the farewell passage",
    "quit": "the escape hatch",
    "mail": "the letter chute",
    "data": "the archive door",
    "commit": "the seal of binding",
    "abort": "the door of retreat",
    "retry": "the second chance passage",
    "abandon": "the door of surrender",
    "hasNext": "the oracle's curtain",
    "next": "the forward passage",
    "insertCard": "the slot in the wall",
    "enterPIN": "the combination lock",
    "checkBalance": "the mirror of wealth",
    "withdraw": "the vault door",
    "deposit": "the offering chute",
    "ejectCard": "the return slot",
    "tau": "a hidden door you almost missed",
    "shutdown": "the emergency exit",
}

FATE_MSGS = {
    "OK": "Fortune smiles — the way forward is clear.",
    "ERR": "A grinding sound... something went wrong.",
    "ERROR": "The mechanism jams. An alternative path opens.",
    "TRUE": "The oracle nods. There is more ahead.",
    "FALSE": "The oracle shakes its head. This path ends.",
    "AUTH": "The lock clicks open. You are recognized.",
    "REJECTED": "The lock holds fast. You are turned away.",
    "GRANTED": "Permission granted. The gate swings wide.",
    "DENIED": "Access denied. Darkness follows.",
    "ACK": "A voice echoes: 'Acknowledged.'",
    "NACK": "Silence. Your request was ignored.",
    "TOKEN": "A glowing token materializes in your hand.",
    "EXPIRED": "The token crumbles to dust.",
    "TIMEOUT": "Time runs out. The walls close in.",
    "data": "Scrolls cascade from the ceiling.",
    "eof": "The last page falls. Nothing more to read.",
    "INSUFFICIENT": "The vault is empty.",
    "PASS": "The trial is complete. You passed.",
    "FAIL": "The trial is complete. You did not.",
}

ITEMS = [
    "a rusty key", "a glowing shard", "a torn map", "a silver coin",
    "a vial of liquid light", "a compass that points down",
    "a whisper trapped in glass", "a feather that weighs like stone",
    "an hourglass running upward", "a mirror showing another room",
]

ENCOUNTERS = [
    "A rat scurries past your feet.",
    "You hear distant chanting.",
    "Water drips from the ceiling in a rhythm.",
    "A cold breeze touches your neck.",
    "Symbols glow briefly on the wall, then fade.",
    "You step on something that crunches.",
    "A shadow moves at the edge of your vision.",
    "The air tastes of copper.",
    "Your footsteps echo longer than they should.",
    "A door behind you closes silently.",
    "",  # sometimes nothing happens
    "",
    "",
]

# ── Colors ───────────────────────────────────────────────────────────────

B = "\033[94m"    # blue (your choice)
O = "\033[93m"    # orange (fate's choice)
G = "\033[92m"    # green
R = "\033[91m"    # red
W = "\033[97m"    # white
D = "\033[2m"     # dim
BD = "\033[1m"    # bold
X = "\033[0m"     # reset


# ── Game Engine ──────────────────────────────────────────────────────────

@dataclass
class Player:
    hp: int = 10
    gold: int = 0
    items: list[str] = field(default_factory=list)
    depth: int = 0
    rooms_visited: int = 0
    visited_states: set[int] = field(default_factory=set)


def slow_print(text: str, delay: float = 0.02) -> None:
    """Print text character by character for atmosphere."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if ch in ".!?—":
            time.sleep(delay * 3)
        elif ch == "\n":
            time.sleep(delay * 2)
        else:
            time.sleep(delay)
    print()


def door_name(label: str) -> str:
    """Get a thematic name for a transition label."""
    return DOOR_NAMES.get(label, f"the door marked '{label}'")


def room_description(ss, state_id: int, player: Player, is_selection: bool) -> str:
    """Generate an atmospheric room description."""
    if state_id == ss.top:
        return random.choice(ROOMS["entrance"])
    if state_id == ss.bottom:
        return random.choice(ROOMS["exit"])
    if state_id in player.visited_states:
        return random.choice(ROOMS["loop"])

    transitions = [(l, t) for s, l, t in ss.transitions if s == state_id]
    if len(transitions) == 1:
        return random.choice(ROOMS["corridor"])
    if is_selection:
        return random.choice(ROOMS["selection"])
    return random.choice(ROOMS["branch"])


def play(type_string: str, fast: bool = False) -> None:
    """Run The Reticulate."""
    from reticulate.parser import parse
    from reticulate.statespace import build_statespace
    from reticulate.lattice import check_lattice

    ast = parse(type_string)
    ss = build_statespace(ast)

    pr = slow_print if not fast else print
    player = Player()
    current = ss.top

    # Title
    print()
    pr(f"  {BD}{W}THE RETICULATE{X}")
    pr(f"  {D}A descent through structured darkness{X}")
    print()
    time.sleep(0.5 if not fast else 0)

    pr(f"  You stand before {random.choice(ROOMS['entrance'])}.")
    pr(f"  {D}The only way is down.{X}")
    print()

    turn = 0
    while current != ss.bottom:
        turn += 1
        player.rooms_visited += 1
        player.visited_states.add(current)

        # Get available moves
        moves = [(l, t) for s, l, t in ss.transitions if s == current]

        if not moves:
            pr(f"\n  {R}The dungeon has no exit. You are trapped forever.{X}")
            break

        # Determine if this is a selection (fate decides) or branch (you decide)
        is_sel = any(ss.is_selection(current, l, t) for l, t in moves)

        # Room description
        desc = room_description(ss, current, player, is_sel)
        pr(f"\n  {W}{desc}{X}")

        # Random encounter
        enc = random.choice(ENCOUNTERS)
        if enc:
            pr(f"  {D}{enc}{X}")

        # Random item (10% chance)
        if random.random() < 0.10 and turn > 1:
            item = random.choice(ITEMS)
            player.items.append(item)
            pr(f"  {G}You find {item}.{X}")

        # Random gold
        if random.random() < 0.25:
            gold = random.randint(1, 5)
            player.gold += gold

        # Forced move (single transition)
        if len(moves) == 1:
            label, target = moves[0]
            pr(f"  There is only one way forward: through {door_name(label)}.")
            if not fast:
                time.sleep(0.3)
            pr(f"  {D}You pass through.{X}")
            current = target
            player.depth += 1
            continue

        if is_sel:
            # FATE DECIDES
            pr(f"\n  {O}This is not your choice to make.{X}")
            pr(f"  {D}The dungeon decides...{X}")
            if not fast:
                time.sleep(0.8)

            label, target = random.choice(moves)
            fate_msg = FATE_MSGS.get(label, f"The way of '{label}' opens before you.")
            pr(f"  {O}{fate_msg}{X}")
            pr(f"  {D}You are drawn through {door_name(label)}.{X}")
            current = target
            player.depth += 1

        else:
            # YOU DECIDE
            pr(f"\n  {B}You see {len(moves)} doors:{X}")
            for i, (label, target) in enumerate(moves, 1):
                # Hint: visited rooms get a mark
                visited_mark = f" {D}(familiar){X}" if target in player.visited_states else ""
                pr(f"    {B}{i}.{X} {door_name(label)}{visited_mark}")

            # Input
            while True:
                try:
                    choice = input(f"\n  {B}Which door? {X}").strip()
                    if choice.lower() in ("q", "quit"):
                        pr(f"\n  {D}You sit down and refuse to go further.{X}")
                        pr(f"  {D}The dungeon waits. It has all the time in the world.{X}")
                        return
                    idx = int(choice) - 1
                    if 0 <= idx < len(moves):
                        break
                except (ValueError, EOFError):
                    pass
                pr(f"  {R}Choose 1-{len(moves)}.{X}")

            label, target = moves[idx]
            pr(f"\n  You step through {door_name(label)}.")
            current = target
            player.depth += 1

    # Victory
    print()
    pr(f"  {G}{BD}═══════════════════════════════════{X}")
    pr(f"  {G}{BD}  You emerge into the light.{X}")
    pr(f"  {G}{BD}  The Reticulate releases you.{X}")
    pr(f"  {G}{BD}═══════════════════════════════════{X}")
    print()
    pr(f"  Rooms explored: {player.rooms_visited}")
    pr(f"  Depth reached:  {player.depth}")
    pr(f"  Gold collected: {player.gold}")
    if player.items:
        pr(f"  Items found:    {', '.join(player.items)}")
    print()

    # Secret: reveal the lattice
    pr(f"  {D}[The dungeon was generated from a session type lattice")
    pr(f"   with {len(ss.states)} rooms and {len(ss.transitions)} passages.]{X}")
    print()


# ── Dungeon Generation ───────────────────────────────────────────────────

DUNGEON_CATALOGUE = [
    # Easy (2-4 states)
    ("The Corridor", "&{enter: end}"),
    ("The Fork", "&{a: end, b: end}"),
    ("The Oracle", "&{ask: +{YES: end, NO: end}}"),
    ("The Quick Loop", "rec X . &{again: X, stop: end}"),
    # Medium (5-8 states)
    ("The Vault", "&{open: &{read: +{data: end, eof: end}, close: end}}"),
    ("The Maze", "&{a: &{c: end, d: end}, b: &{e: end, f: end}}"),
    ("The Mill", "rec X . &{grind: +{OK: X, ERR: X}, quit: end}"),
    ("The Archive", "&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}"),
    # Hard (7-15 states)
    ("The Citadel",
     "&{insertCard: &{enterPIN: +{AUTH: &{checkBalance: end, withdraw: +{OK: end, INSUFFICIENT: end}, ejectCard: end}, REJECTED: &{ejectCard: end}}}}"),
    ("The Labyrinth",
     "&{connect: &{login: rec X . &{send: +{OK: X, ERR: X}, receive: +{data: X, eof: &{disconnect: end}}, logout: end}}}"),
    ("The Twin Halls",
     "&{open: +{OK: (&{read: end} || &{write: end}), ERR: end}}"),
    # Boss
    ("The Deep Reticulate",
     "&{connect: rec X . &{mail: &{rcpt: &{data: +{OK: X, ERR: X}}}, quit: end}}"),
]


def random_dungeon() -> tuple[str, str]:
    """Pick a random dungeon from the catalogue."""
    return random.choice(DUNGEON_CATALOGUE)


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="The Reticulate — a dungeon on a lattice")
    parser.add_argument("type_string", nargs="?", help="Session type (dungeon blueprint)")
    parser.add_argument("--random", "-r", action="store_true", help="Random dungeon")
    parser.add_argument("--list", "-l", action="store_true", help="List dungeons")
    parser.add_argument("--fast", "-f", action="store_true", help="Skip slow text")
    parser.add_argument("--dungeon", "-d", type=int, help="Dungeon number (1-based)")
    args = parser.parse_args(argv)

    if args.list:
        print(f"\n  {BD}Available Dungeons:{X}\n")
        for i, (name, ts) in enumerate(DUNGEON_CATALOGUE, 1):
            from reticulate.statespace import build_statespace
            from reticulate.parser import parse
            ss = build_statespace(parse(ts))
            difficulty = "Easy" if len(ss.states) <= 4 else "Medium" if len(ss.states) <= 8 else "Hard"
            print(f"  {i:2d}. {name:25s} [{difficulty}, {len(ss.states)} rooms]")
        print()
        return

    if args.dungeon:
        if 1 <= args.dungeon <= len(DUNGEON_CATALOGUE):
            name, ts = DUNGEON_CATALOGUE[args.dungeon - 1]
            print(f"\n  {BD}Entering: {name}{X}")
            play(ts, fast=args.fast)
        else:
            print(f"Dungeon {args.dungeon} not found. Use --list.")
        return

    if args.random:
        name, ts = random_dungeon()
        print(f"\n  {BD}Entering: {name}{X}")
        play(ts, fast=args.fast)
        return

    if args.type_string:
        play(args.type_string, fast=args.fast)
        return

    # Default: random dungeon
    name, ts = random_dungeon()
    print(f"\n  {BD}Entering: {name}{X}")
    play(ts, fast=args.fast)


if __name__ == "__main__":
    main()
