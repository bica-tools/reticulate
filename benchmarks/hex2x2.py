"""Hex on a 2x2 board — session type benchmark.

Encoding
--------
Hex is a two-player connection game on a rhombus-shaped board.
Player 1 (Select) connects top row to bottom row.
Player 2 (Branch) connects left column to right column.
The game has no draws (the Hex theorem, proved by John Nash).

Board layout (2x2)::

      a --- b          row 0
       \\ / \\ /
        c --- d          row 1

        col0  col1

Cells: a=(0,0), b=(0,1), c=(1,0), d=(1,1).

Hex adjacency: each cell is adjacent to its row neighbours and the cell
diagonally down-left (NW-SE diagonals).  On a 2x2 board:
    - a: adjacent to b, c
    - b: adjacent to a, c, d
    - c: adjacent to a, b, d
    - d: adjacent to b, c

Winning conditions:
    - Player 1 wins: connected path from row 0 to row 1
      Minimal winning sets: {a,c}, {a,d} via c, {b,c}, {b,d}
    - Player 2 wins: connected path from col 0 to col 1
      Minimal winning sets: {a,b}, {c,d}, {c,b} via adjacency, {a,d} via c

The session type encodes the full game tree.  Early termination occurs
when a player achieves a winning connection before all cells are filled.

Convention:
    - Player 1 = Select (+{...}) — places stone, choosing cell
    - Player 2 = Branch (&{...}) — opponent places stone
    - Players alternate; P1 moves first

Self-duality: dual(S) swaps & <-> +, giving Player 2's perspective.
Verified: dual is an involution, state spaces are isomorphic, and
selection annotations are correctly flipped.

State-space properties (verified):
    - States: 30
    - Transitions: 52
    - SCCs: 30
    - Lattice: True
    - Top = empty board (P1 to move), Bottom = game over (end)
"""

HEX2X2_TYPE_STRING = (
    "+{a: &{b: +{c: end, d: &{c: end}}, "
    "c: +{b: &{d: end}, d: &{b: end}}, "
    "d: +{b: &{c: end}, c: end}}, "
    "b: &{a: +{c: end, d: end}, "
    "c: +{a: &{d: end}, d: end}, "
    "d: +{a: &{c: end}, c: end}}, "
    "c: &{a: +{b: end, d: &{b: end}}, "
    "b: +{a: end, d: &{a: end}}, "
    "d: +{a: end, b: end}}, "
    "d: &{a: +{b: end, c: &{b: end}}, "
    "b: +{a: &{c: end}, c: &{a: end}}, "
    "c: +{a: &{b: end}, b: end}}}"
)
