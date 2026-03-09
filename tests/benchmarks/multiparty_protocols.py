"""Multiparty benchmark protocol definitions (Step 11).

Classic multiparty protocols expressed as global types with role annotations.
Each protocol includes expected roles and projection expectations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultipartyBenchmark:
    """A multiparty benchmark protocol."""

    name: str
    global_type_string: str
    description: str
    expected_roles: frozenset[str]


MULTIPARTY_BENCHMARKS: list[MultipartyBenchmark] = [
    # 1. Two-Buyer (Honda-Yoshida-Carbone 2008)
    MultipartyBenchmark(
        name="Two-Buyer",
        global_type_string=(
            "Buyer1 -> Seller : {lookup: "
            "Seller -> Buyer1 : {price: "
            "Seller -> Buyer2 : {price: "
            "Buyer1 -> Buyer2 : {share: "
            "Buyer2 -> Seller : {accept: "
            "Seller -> Buyer2 : {deliver: end}, "
            "reject: end}}}}}"
        ),
        description=(
            "Classic two-buyer protocol: Buyer1 looks up item, Seller sends "
            "price to both buyers, Buyer1 shares contribution with Buyer2, "
            "then Buyer2 accepts (Seller delivers) or rejects."
        ),
        expected_roles=frozenset({"Buyer1", "Buyer2", "Seller"}),
    ),
    # 2. Simple Request-Response (binary as global)
    MultipartyBenchmark(
        name="Request-Response",
        global_type_string=(
            "Client -> Server : {request: "
            "Server -> Client : {response: end}}"
        ),
        description="Simple binary request-response expressed as global type.",
        expected_roles=frozenset({"Client", "Server"}),
    ),
    # 3. Two-Phase Commit (two participants, coordinator decides)
    MultipartyBenchmark(
        name="Two-Phase Commit",
        global_type_string=(
            "Coord -> P : {prepare: "
            "P -> Coord : {yes: "
            "Coord -> P : {commit: end}, "
            "no: Coord -> P : {abort: end}}}"
        ),
        description=(
            "Two-phase commit with one participant: coordinator prepares, "
            "participant votes yes (commit) or no (abort)."
        ),
        expected_roles=frozenset({"Coord", "P"}),
    ),
    # 4. Streaming (producer-consumer)
    MultipartyBenchmark(
        name="Streaming",
        global_type_string=(
            "rec X . Producer -> Consumer : {data: X, done: end}"
        ),
        description=(
            "Producer streams data to consumer: repeatedly send data or done."
        ),
        expected_roles=frozenset({"Producer", "Consumer"}),
    ),
    # 5. Ring Protocol (three-node)
    MultipartyBenchmark(
        name="Ring",
        global_type_string=(
            "A -> B : {msg: B -> C : {msg: C -> A : {msg: end}}}"
        ),
        description="Three-node ring: A→B→C→A, each sends one message.",
        expected_roles=frozenset({"A", "B", "C"}),
    ),
    # 6. Delegation
    MultipartyBenchmark(
        name="Delegation",
        global_type_string=(
            "Client -> Master : {task: "
            "Master -> Worker : {delegate: "
            "Worker -> Master : {result: "
            "Master -> Client : {response: end}}}}"
        ),
        description=(
            "Client sends task to master, master delegates to worker, "
            "worker returns result, master responds to client."
        ),
        expected_roles=frozenset({"Client", "Master", "Worker"}),
    ),
    # 7. Recursive Negotiation
    MultipartyBenchmark(
        name="Negotiation",
        global_type_string=(
            "rec X . Buyer -> Seller : {offer: "
            "Seller -> Buyer : {accept: end, "
            "counter: X}}"
        ),
        description=(
            "Buyer and seller negotiate: buyer offers, seller accepts or "
            "counter-offers (loop)."
        ),
        expected_roles=frozenset({"Buyer", "Seller"}),
    ),
    # 8. Auth then Service (two-party: client-server with auth phase)
    MultipartyBenchmark(
        name="Auth-Service",
        global_type_string=(
            "Client -> Server : {login: "
            "Server -> Client : {granted: "
            "rec X . Client -> Server : {request: "
            "Server -> Client : {response: X}, "
            "logout: end}, "
            "denied: end}}"
        ),
        description=(
            "Client authenticates with server, then if granted, "
            "repeatedly makes requests, or logs out."
        ),
        expected_roles=frozenset({"Client", "Server"}),
    ),
]
