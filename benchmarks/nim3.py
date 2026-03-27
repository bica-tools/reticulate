"""Nim with 3 objects — session type benchmark.

Encoding
--------
Nim is a two-player combinatorial game.  A pile starts with N objects.
Players alternate removing 1..k objects (here k = N for misère-free normal
play: last player to take wins).

With N = 3 the game tree is:

    Position 3 (P1 to move) ─take1→ Position 2 (P2) ─take1→ Position 1 (P1) ─take1→ end
                             ─take2→ Position 1 (P2) ─take1→ end
                             ─take3→ end

Convention:
    - Player 1 = Select (+{...})  — internal choice (we choose our move)
    - Player 2 = Branch (&{...})  — external choice (opponent chooses)
    - Alternation: P1 at positions 3 and 1; P2 at position 2

At each position the available moves are ``take_i`` for i in 1..min(k, remaining).

State-space properties (verified):
    - States: 5
    - Transitions: 7
    - SCCs: 5
    - Lattice: True
    - Top = position 3 (initial), Bottom = position 0 (game over / end)

Sprague-Grundy values:
    - G(0) = 0 (losing for player to move = P-position)
    - G(1) = 1 (N-position)
    - G(2) = 2 (N-position)
    - G(3) = 3 (N-position)
"""

NIM3_TYPE_STRING = (
    "+{take1: &{take1: +{take1: end}, take2: end}, "
    "take2: &{take1: end}, "
    "take3: end}"
)
