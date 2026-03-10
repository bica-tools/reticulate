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
    # 9. OAuth 2.0 (3 roles)
    MultipartyBenchmark(
        name="OAuth2",
        global_type_string=(
            "Client -> AuthServer : {authorize: "
            "AuthServer -> Client : {code: "
            "Client -> AuthServer : {exchange: "
            "AuthServer -> Client : {token: "
            "Client -> ResourceServer : {access: "
            "ResourceServer -> Client : {resource: end}}}}}}"
        ),
        description=(
            "OAuth 2.0 authorization code flow: Client gets auth code from "
            "AuthServer, exchanges for token, then accesses ResourceServer."
        ),
        expected_roles=frozenset({"Client", "AuthServer", "ResourceServer"}),
    ),
    # 10. MCP (3 roles: Host, Client, Server)
    MultipartyBenchmark(
        name="MCP",
        global_type_string=(
            "Host -> Client : {initialize: "
            "Client -> Server : {discover: "
            "Server -> Client : {tools: "
            "Client -> Host : {ready: "
            "rec X . Client -> Server : {call: "
            "Server -> Client : {result: X}, "
            "done: end}}}}}"
        ),
        description=(
            "Model Context Protocol: Host initializes Client, Client discovers "
            "tools on Server, then repeatedly calls tools and reports to Host."
        ),
        expected_roles=frozenset({"Host", "Client", "Server"}),
    ),
    # 11. A2A (3 roles: Client, RemoteAgent, TaskStore)
    MultipartyBenchmark(
        name="A2A",
        global_type_string=(
            "Client -> RemoteAgent : {sendTask: "
            "RemoteAgent -> TaskStore : {store: "
            "TaskStore -> RemoteAgent : {ack: "
            "RemoteAgent -> Client : {status: "
            "Client -> RemoteAgent : {getResult: "
            "RemoteAgent -> TaskStore : {fetch: "
            "TaskStore -> RemoteAgent : {artifact: "
            "RemoteAgent -> Client : {result: end}}}}}}}}"
        ),
        description=(
            "Agent-to-Agent protocol: Client sends task to RemoteAgent, "
            "agent stores artifacts in TaskStore, Client retrieves result."
        ),
        expected_roles=frozenset({"Client", "RemoteAgent", "TaskStore"}),
    ),
    # 12. MQTT (3 roles: Publisher, Broker, Subscriber)
    MultipartyBenchmark(
        name="MQTT",
        global_type_string=(
            "Subscriber -> Broker : {subscribe: "
            "Broker -> Subscriber : {suback: "
            "Publisher -> Broker : {publish: "
            "Broker -> Subscriber : {deliver: "
            "Subscriber -> Broker : {puback: "
            "Publisher -> Broker : {disconnect: end}}}}}}"
        ),
        description=(
            "MQTT pub/sub: Subscriber subscribes to Broker, then Publisher "
            "repeatedly publishes messages that Broker delivers to Subscriber."
        ),
        expected_roles=frozenset({"Publisher", "Broker", "Subscriber"}),
    ),
    # 13. Raft Consensus (3 roles: Candidate, Voter, Leader)
    MultipartyBenchmark(
        name="Raft-Consensus",
        global_type_string=(
            "Candidate -> Voter : {requestVote: "
            "Voter -> Candidate : {granted: "
            "Candidate -> Voter : {appendEntries: "
            "Voter -> Candidate : {ack: end}}, "
            "rejected: end}}"
        ),
        description=(
            "Raft consensus: Candidate requests vote from Voter, if granted "
            "becomes leader and sends AppendEntries, otherwise election fails."
        ),
        expected_roles=frozenset({"Candidate", "Voter"}),
    ),
    # 14. Saga (4 roles: Orchestrator, Service1, Service2, CompLog)
    MultipartyBenchmark(
        name="Saga",
        global_type_string=(
            "Orchestrator -> Service1 : {execute: "
            "Service1 -> Orchestrator : {done1: "
            "Orchestrator -> CompLog : {log1: "
            "Orchestrator -> Service2 : {execute: "
            "Service2 -> Orchestrator : {done2: "
            "Orchestrator -> CompLog : {log2: "
            "Orchestrator -> CompLog : {commit: end}}}}}}}"
        ),
        description=(
            "Saga pattern: Orchestrator coordinates Service1 and Service2 "
            "with compensation logging. On failure, compensates in reverse."
        ),
        expected_roles=frozenset(
            {"Orchestrator", "Service1", "Service2", "CompLog"}
        ),
    ),
    # 15. DNS Resolution (3 roles: Client, Resolver, AuthNS)
    MultipartyBenchmark(
        name="DNS-Resolution",
        global_type_string=(
            "Client -> Resolver : {query: "
            "Resolver -> AuthNS : {lookup: "
            "AuthNS -> Resolver : {answer: "
            "Resolver -> Client : {response: end}}}}"
        ),
        description=(
            "DNS resolution: Client queries Resolver, Resolver queries "
            "AuthoritativeNameServer, response propagates back."
        ),
        expected_roles=frozenset({"Client", "Resolver", "AuthNS"}),
    ),
    # 16. Payment Processing (4 roles: Customer, Merchant, Gateway, Bank)
    MultipartyBenchmark(
        name="Payment-Processing",
        global_type_string=(
            "Customer -> Merchant : {order: "
            "Merchant -> Gateway : {charge: "
            "Gateway -> Bank : {authorize: "
            "Bank -> Gateway : {approved: "
            "Gateway -> Merchant : {success: "
            "Merchant -> Customer : {receipt: end}}}}}}"
        ),
        description=(
            "Payment processing: Customer orders from Merchant, Merchant "
            "charges via Gateway, Gateway authorizes with Bank."
        ),
        expected_roles=frozenset(
            {"Customer", "Merchant", "Gateway", "Bank"}
        ),
    ),
    # 17. Microservice Health (3 roles, parallel)
    MultipartyBenchmark(
        name="Microservice-Health",
        global_type_string=(
            "(Monitor -> ServiceA : {ping: "
            "ServiceA -> Monitor : {pong: end}} || "
            "Monitor -> ServiceB : {ping: "
            "ServiceB -> Monitor : {pong: end}})"
        ),
        description=(
            "Microservice health check: Monitor pings ServiceA and ServiceB "
            "in parallel, both respond with pong."
        ),
        expected_roles=frozenset({"Monitor", "ServiceA", "ServiceB"}),
    ),
    # 18. Map-Reduce (3 roles)
    MultipartyBenchmark(
        name="Map-Reduce",
        global_type_string=(
            "Master -> Mapper : {map: "
            "Mapper -> Reducer : {emit: "
            "Reducer -> Master : {result: end}}}"
        ),
        description=(
            "Map-Reduce: Master distributes work to Mapper, Mapper emits "
            "intermediate results to Reducer, Reducer returns final result."
        ),
        expected_roles=frozenset({"Master", "Mapper", "Reducer"}),
    ),
    # 19. Supply Chain (4 roles)
    MultipartyBenchmark(
        name="Supply-Chain",
        global_type_string=(
            "Customer -> Retailer : {order: "
            "Retailer -> Warehouse : {pick: "
            "Warehouse -> Retailer : {ready: "
            "Retailer -> Shipper : {ship: "
            "Shipper -> Customer : {deliver: "
            "Customer -> Shipper : {sign: "
            "Shipper -> Retailer : {confirm: end}}}}}}}"
        ),
        description=(
            "Supply chain: Customer orders from Retailer, Retailer picks "
            "from Warehouse, Shipper delivers to Customer who signs."
        ),
        expected_roles=frozenset(
            {"Customer", "Retailer", "Warehouse", "Shipper"}
        ),
    ),
    # 20. Parallel Task Distribution (3 roles, parallel)
    MultipartyBenchmark(
        name="Parallel-Tasks",
        global_type_string=(
            "Master -> Worker : {init: "
            "(Master -> Worker : {taskA: "
            "Worker -> Master : {resultA: end}} || "
            "Master -> Worker : {taskB: "
            "Worker -> Master : {resultB: end}})}"
        ),
        description=(
            "Parallel task distribution: Master initializes Worker, then "
            "sends two independent tasks (A and B) in parallel."
        ),
        expected_roles=frozenset({"Master", "Worker"}),
    ),
]
