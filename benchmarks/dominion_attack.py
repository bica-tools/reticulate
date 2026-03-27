"""Dominion Attack/Moat reaction window — session type benchmark.

Encoding
--------
In the card game Dominion, when a player plays an Attack card, other
players may react with a Moat card *concurrently* — before the attack
resolves.  This is a genuine fork-join: the attacker's action and the
defender's reaction happen in parallel, not sequentially.

The parallel constructor (||) from the BICA Reborn specification (v0.2)
models this directly:

    (attacker_branch || defender_branch) . continuation

Structure:
    - Left branch: +{play_attack: wait}
      The attacker (Select) plays the attack card, then waits for sync.
    - Right branch: &{react: end, pass: end}
      The defender (Branch) either reacts with Moat or passes.
    - Continuation: &{resolve: end}
      After synchronization, the attack resolves.

This is a concrete real-world instance of the parallel constructor
outside the software domain.  No nested parallel (respects spec v0.2).

State-space properties (verified):
    - States: 5
    - Transitions: 7
    - SCCs: 5
    - Lattice: True (distributive — product lattice from parallel)
    - Top = pre-attack state, Bottom = resolved (end)
"""

DOMINION_ATTACK_TYPE_STRING = (
    "(+{play_attack: wait} || &{react: end, pass: end}) "
    ". &{resolve: end}"
)
