"""Benchmark protocol definitions for the reticulate test suite.

34 real-world and classic protocols expressed as session types using the
core grammar: branch (&), selection (+), parallel (||), recursion (rec),
continuation (.), wait, and end.  No sequencing sugar.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProtocol:
    """A benchmark protocol with its expected state-space metrics."""

    name: str
    type_string: str
    description: str
    expected_states: int
    expected_transitions: int
    expected_sccs: int
    uses_parallel: bool


BENCHMARKS: list[BenchmarkProtocol] = [
    # 1. Java Iterator
    BenchmarkProtocol(
        name="Java Iterator",
        type_string="rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
        description=(
            "The java.util.Iterator protocol: repeatedly call hasNext, "
            "then next if TRUE or stop if FALSE."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 2. File Object
    BenchmarkProtocol(
        name="File Object",
        type_string="&{open: rec X . &{read: +{data: X, eof: &{close: end}}}}",
        description=(
            "A file handle: open, then repeatedly read until EOF, then close. "
            "Models the classic open/read/close lifecycle."
        ),
        expected_states=5,
        expected_transitions=5,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 3. SMTP
    BenchmarkProtocol(
        name="SMTP",
        type_string=(
            "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: "
            "+{OK: X, ERR: X}}}, quit: end}}}"
        ),
        description=(
            "Simplified SMTP session: connect, EHLO, then loop sending mail "
            "(MAIL/RCPT/DATA with OK or ERR) or QUIT."
        ),
        expected_states=7,
        expected_transitions=8,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 4. HTTP Connection
    BenchmarkProtocol(
        name="HTTP Connection",
        type_string=(
            "&{connect: rec X . &{request: +{OK200: &{readBody: X}, "
            "ERR4xx: X, ERR5xx: X}, close: end}}"
        ),
        description=(
            "A persistent HTTP connection: connect, then repeatedly send "
            "requests (with various response codes) or close."
        ),
        expected_states=5,
        expected_transitions=7,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 5. OAuth 2.0
    BenchmarkProtocol(
        name="OAuth 2.0",
        type_string=(
            "&{requestAuth: +{GRANTED: &{getToken: +{TOKEN: rec X . "
            "&{useToken: X, refreshToken: +{OK: X, EXPIRED: end}, "
            "revoke: end}, ERROR: end}}, DENIED: end}}"
        ),
        description=(
            "OAuth 2.0 authorization code flow: request authorization, "
            "obtain token, then use/refresh/revoke the token."
        ),
        expected_states=7,
        expected_transitions=11,
        expected_sccs=6,
        uses_parallel=False,
    ),
    # 6. Two-Buyer
    BenchmarkProtocol(
        name="Two-Buyer",
        type_string=(
            "&{lookup: &{getPrice: (&{proposeA: end} || "
            "&{proposeB: +{ACCEPT: &{pay: end}, REJECT: end}})}}"
        ),
        description=(
            "The two-buyer protocol: lookup an item, get price, then two "
            "buyers concurrently propose; buyer B may accept and pay or reject."
        ),
        expected_states=10,
        expected_transitions=14,
        expected_sccs=10,
        uses_parallel=True,
    ),
    # 7. MCP (Model Context Protocol)
    BenchmarkProtocol(
        name="MCP",
        type_string=(
            "&{initialize: (rec X . &{callTool: +{RESULT: X, ERROR: X}, "
            "listTools: X, shutdown: end} || "
            "rec Y . +{NOTIFICATION: Y, DONE: end})}"
        ),
        description=(
            "AI agent Model Context Protocol: initialize, then concurrently "
            "handle tool calls/listing/shutdown and notifications."
        ),
        expected_states=7,
        expected_transitions=17,
        expected_sccs=5,
        uses_parallel=True,
    ),
    # 8. A2A (Agent-to-Agent)
    BenchmarkProtocol(
        name="A2A",
        type_string=(
            "&{sendTask: rec X . +{WORKING: &{getStatus: X, cancel: end}, "
            "COMPLETED: &{getArtifact: end}, FAILED: end}}"
        ),
        description=(
            "Google's Agent-to-Agent protocol: send a task, then poll for "
            "status (WORKING/COMPLETED/FAILED) with cancellation support."
        ),
        expected_states=5,
        expected_transitions=7,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 9. File Channel
    BenchmarkProtocol(
        name="File Channel",
        type_string=(
            "&{open: +{OK: (rec X . &{read: X, doneRead: wait} || "
            "rec Y . &{write: Y, doneWrite: wait}) . &{close: end}, ERR: end}}"
        ),
        description=(
            "A file channel with concurrent read/write streams: open, then "
            "parallel read and write loops, then close."
        ),
        expected_states=7,
        expected_transitions=12,
        expected_sccs=7,
        uses_parallel=True,
    ),
    # 10. ATM
    BenchmarkProtocol(
        name="ATM",
        type_string=(
            "&{insertCard: &{enterPIN: +{AUTH: rec X . "
            "&{checkBalance: X, withdraw: +{OK: X, INSUFFICIENT: X}, "
            "deposit: X, ejectCard: end}, REJECTED: &{ejectCard: end}}}}"
        ),
        description=(
            "An ATM session: insert card, enter PIN, authenticate, then "
            "perform banking operations or eject card."
        ),
        expected_states=7,
        expected_transitions=11,
        expected_sccs=6,
        uses_parallel=False,
    ),
    # 11. Reentrant Lock
    BenchmarkProtocol(
        name="Reentrant Lock",
        type_string=(
            "rec X . &{lock: &{unlock: X, newCondition: "
            "&{await: &{signal: &{unlock: X}}}}, close: end}"
        ),
        description=(
            "A reentrant lock protocol: repeatedly lock/unlock with optional "
            "condition variable support (await/signal), or close."
        ),
        expected_states=6,
        expected_transitions=7,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 12. WebSocket
    BenchmarkProtocol(
        name="WebSocket",
        type_string=(
            "&{connect: +{OPEN: (rec X . &{send: X, ping: X, "
            "closeSend: end} || rec Y . +{MESSAGE: Y, PONG: Y, "
            "CLOSE: end}), REFUSED: end}}"
        ),
        description=(
            "A WebSocket connection: connect, then if open, concurrently "
            "send/ping and receive messages/pong/close events."
        ),
        expected_states=6,
        expected_transitions=15,
        expected_sccs=6,
        uses_parallel=True,
    ),
    # 13. DB Transaction
    BenchmarkProtocol(
        name="DB Transaction",
        type_string=(
            "&{begin: rec X . &{query: +{RESULT: X, ERROR: X}, "
            "update: +{ROWS: X, ERROR: X}, commit: end, rollback: end}}"
        ),
        description=(
            "A database transaction: begin, then repeatedly query or update "
            "(with results or errors), then commit or rollback."
        ),
        expected_states=5,
        expected_transitions=9,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 14. Pub/Sub
    BenchmarkProtocol(
        name="Pub/Sub",
        type_string=(
            "&{connect: rec X . &{subscribe: X, unsubscribe: X, "
            "poll: +{MSG: X, EMPTY: X}, disconnect: end}}"
        ),
        description=(
            "A publish/subscribe client: connect, then subscribe, "
            "unsubscribe, poll for messages, or disconnect."
        ),
        expected_states=4,
        expected_transitions=7,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 15. DNS Resolver
    BenchmarkProtocol(
        name="DNS Resolver",
        type_string=(
            "rec X . &{query: +{ANSWER: X, NXDOMAIN: X, SERVFAIL: X, "
            "TIMEOUT: &{retry: X, abandon: end}}, close: end}"
        ),
        description=(
            "A DNS resolver: repeatedly query with various outcomes "
            "(ANSWER, NXDOMAIN, SERVFAIL, TIMEOUT with retry/abandon), "
            "or close."
        ),
        expected_states=4,
        expected_transitions=8,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 16. Reticulate Pipeline (self-reference)
    BenchmarkProtocol(
        name="Reticulate Pipeline",
        type_string=(
            "&{input: &{parse: +{OK: &{buildStateSpace: +{OK: "
            "(&{checkLattice: &{checkTermination: &{checkWFPar: wait}}} "
            "|| &{renderDiagram: wait}) . &{returnResult: end}, "
            "ERROR: end}}, ERROR: end}}}"
        ),
        description=(
            "The reticulate web demo's own analysis pipeline as a session type. "
            "Verification and rendering run concurrently via the parallel "
            "constructor — a self-referential example."
        ),
        expected_states=14,
        expected_transitions=18,
        expected_sccs=14,
        uses_parallel=True,
    ),
    # 17. GitHub CI Workflow
    BenchmarkProtocol(
        name="GitHub CI Workflow",
        type_string=(
            "&{trigger: &{checkout: &{setup: "
            "(&{lint: wait} || &{test: wait}) . "
            "+{PASS: &{deploy: +{OK: end, FAIL: &{rollback: end}}}, FAIL: end}}}}"
        ),
        description=(
            "A GitHub Actions CI/CD pipeline: trigger, checkout, setup, then "
            "parallel lint and test jobs, then deploy on success (with rollback "
            "on failure) or stop on failure."
        ),
        expected_states=11,
        expected_transitions=13,
        expected_sccs=11,
        uses_parallel=True,
    ),
    # 18. TLS Handshake
    BenchmarkProtocol(
        name="TLS Handshake",
        type_string=(
            "&{clientHello: +{HELLO_RETRY: &{clientHello: &{serverHello: "
            "&{certificate: &{verify: &{changeCipher: end}}, "
            "psk: &{changeCipher: end}}}}, "
            "SERVER_HELLO: &{certificate: &{verify: &{changeCipher: end}}, "
            "psk: &{changeCipher: end}}}}"
        ),
        description=(
            "A TLS 1.3 handshake: client hello with possible hello retry, "
            "then server hello with certificate or PSK-based cipher change."
        ),
        expected_states=13,
        expected_transitions=15,
        expected_sccs=13,
        uses_parallel=False,
    ),
    # 19. Raft Leader Election
    BenchmarkProtocol(
        name="Raft Leader Election",
        type_string=(
            "rec X . +{TIMEOUT: &{requestVote: +{ELECTED: rec Y . "
            "&{appendEntries: +{ACK: Y, NACK: Y}, heartbeatTimeout: Y, "
            "stepDown: X}, REJECTED: X}}, HEARTBEAT: X, SHUTDOWN: end}"
        ),
        description=(
            "Raft consensus leader election: followers timeout, request votes, "
            "become leader (append entries, heartbeat) or step down."
        ),
        expected_states=6,
        expected_transitions=11,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 20. MQTT Client
    BenchmarkProtocol(
        name="MQTT Client",
        type_string=(
            "&{connect: +{CONNACK: (rec X . &{publish: +{PUBACK: X, "
            "TIMEOUT: X}, disconnect: wait} || rec Y . +{MESSAGE: Y, "
            "SUBACK: Y, DONE: wait}) . end, REFUSED: end}}"
        ),
        description=(
            "An MQTT client session: connect, then if accepted, concurrently "
            "publish messages (with ack/timeout) and receive messages/subacks."
        ),
        expected_states=8,
        expected_transitions=20,
        expected_sccs=6,
        uses_parallel=True,
    ),
    # 21. Circuit Breaker
    BenchmarkProtocol(
        name="Circuit Breaker",
        type_string=(
            "rec X . &{call: +{SUCCESS: X, FAILURE: +{TRIPPED: rec Y . "
            "&{probe: +{OK: X, FAIL: Y}, timeout: end}, OK: X}}, "
            "reset: end}"
        ),
        description=(
            "A circuit breaker pattern: closed state accepts calls (success "
            "loops, failure may trip), open state probes for recovery."
        ),
        expected_states=6,
        expected_transitions=10,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 22. Connection Pool
    BenchmarkProtocol(
        name="Connection Pool",
        type_string=(
            "&{init: (rec X . &{acquire: &{use: &{release: X}}, drain: end} || "
            "rec Y . &{healthCheck: +{HEALTHY: Y, UNHEALTHY: Y}, "
            "shutdown: end})}"
        ),
        description=(
            "A connection pool: initialize, then concurrently handle "
            "acquire/use/release cycles and health checks with shutdown."
        ),
        expected_states=13,
        expected_transitions=29,
        expected_sccs=5,
        uses_parallel=True,
    ),
    # 23. gRPC BiDi Stream
    BenchmarkProtocol(
        name="gRPC BiDi Stream",
        type_string=(
            "&{open: (rec X . &{send: X, halfClose: end} || "
            "rec Y . +{RESPONSE: Y, TRAILER: end})}"
        ),
        description=(
            "A gRPC bidirectional streaming RPC: open, then concurrently "
            "send requests (with half-close) and receive responses/trailers."
        ),
        expected_states=5,
        expected_transitions=9,
        expected_sccs=5,
        uses_parallel=True,
    ),
    # 24. Blockchain Tx
    BenchmarkProtocol(
        name="Blockchain Tx",
        type_string=(
            "&{createTx: &{sign: &{broadcast: rec X . +{PENDING: X, "
            "CONFIRMED: &{getReceipt: end}, DROPPED: end, FAILED: end}}}}"
        ),
        description=(
            "A blockchain transaction lifecycle: create, sign, broadcast, "
            "then poll for confirmation, drop, or failure."
        ),
        expected_states=6,
        expected_transitions=8,
        expected_sccs=6,
        uses_parallel=False,
    ),
    # 25. Kafka Consumer
    BenchmarkProtocol(
        name="Kafka Consumer",
        type_string=(
            "&{subscribe: (rec X . &{poll: +{RECORDS: &{process: &{commit: X}}, "
            "EMPTY: X}, pause: end} || rec Y . +{REBALANCE: Y, "
            "REVOKED: end})}"
        ),
        description=(
            "A Kafka consumer: subscribe, then concurrently poll/process/commit "
            "records and handle rebalance/revocation events."
        ),
        expected_states=11,
        expected_transitions=23,
        expected_sccs=5,
        uses_parallel=True,
    ),
    # 26. Rate Limiter
    BenchmarkProtocol(
        name="Rate Limiter",
        type_string=(
            "rec X . &{tryAcquire: +{ALLOWED: X, THROTTLED: "
            "&{wait_retry: X, abort: end}}, close: end}"
        ),
        description=(
            "A rate limiter: repeatedly try to acquire a permit (allowed loops, "
            "throttled can wait or abort), or close."
        ),
        expected_states=4,
        expected_transitions=6,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 27. Saga Orchestrator
    BenchmarkProtocol(
        name="Saga Orchestrator",
        type_string=(
            "&{begin: (&{step1: +{OK: wait, FAIL: &{compensate1: wait}}} || "
            "&{step2: +{OK: wait, FAIL: &{compensate2: wait}}}) . "
            "+{ALL_OK: &{commit: end}, PARTIAL: &{rollback: end}}}"
        ),
        description=(
            "A saga orchestrator: begin, then run two steps concurrently "
            "(each may fail and compensate), then commit or rollback."
        ),
        expected_states=20,
        expected_transitions=37,
        expected_sccs=20,
        uses_parallel=True,
    ),
    # 28. Two-Phase Commit
    BenchmarkProtocol(
        name="Two-Phase Commit",
        type_string=(
            "&{prepare: &{allYes: &{commit: +{ACK: end, TIMEOUT: &{abort: end}}}, "
            "anyNo: &{abort: end}}}"
        ),
        description=(
            "Two-phase commit protocol: prepare, then either all vote yes "
            "(commit with ack or timeout-abort) or any votes no (abort)."
        ),
        expected_states=7,
        expected_transitions=8,
        expected_sccs=7,
        uses_parallel=False,
    ),
    # 29. Leader Replication
    BenchmarkProtocol(
        name="Leader Replication",
        type_string=(
            "&{electLeader: rec X . &{write: (&{replicate1: +{ACK: wait, "
            "NACK: wait}} || &{replicate2: +{ACK: wait, NACK: wait}}) . "
            "+{QUORUM: &{apply: X}, NO_QUORUM: X}, stepDown: end}}"
        ),
        description=(
            "Leader-based replication: elect leader, then repeatedly write "
            "with parallel replication to two nodes, check quorum, or step down."
        ),
        expected_states=13,
        expected_transitions=24,
        expected_sccs=3,
        uses_parallel=True,
    ),
    # 30. Failover
    BenchmarkProtocol(
        name="Failover",
        type_string=(
            "&{connect: rec X . &{request: +{OK: X, FAIL: &{reconnect: "
            "+{UP: X, DOWN: end}}}, close: end}}"
        ),
        description=(
            "A failover protocol: connect, then repeatedly request (OK loops, "
            "FAIL triggers reconnect with UP/DOWN), or close."
        ),
        expected_states=6,
        expected_transitions=8,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 31. Enzyme (Michaelis-Menten)
    BenchmarkProtocol(
        name="Enzyme (Michaelis-Menten)",
        type_string=(
            "rec X . &{bind_substrate: +{CATALYZE: &{release_product: X}, "
            "DISSOCIATE: X}, shutdown: end}"
        ),
        description=(
            "Michaelis-Menten enzyme kinetics: enzyme binds substrate, then "
            "either catalyzes (release product, loop) or substrate dissociates "
            "(loop), or the enzyme is shut down."
        ),
        expected_states=4,
        expected_transitions=5,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 32. Enzyme (Competitive Inhibition)
    BenchmarkProtocol(
        name="Enzyme (Competitive Inhibition)",
        type_string=(
            "rec X . &{bind_substrate: +{CATALYZE: &{release_product: X}, "
            "DISSOCIATE: X}, bind_inhibitor: &{release_inhibitor: X}, "
            "shutdown: end}"
        ),
        description=(
            "Enzyme with competitive inhibition: enzyme can bind substrate "
            "(catalyze or dissociate) or bind inhibitor (must release before "
            "resuming), or shut down."
        ),
        expected_states=5,
        expected_transitions=7,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 33. Ion Channel (Single)
    BenchmarkProtocol(
        name="Ion Channel (Single)",
        type_string=(
            "rec X . &{depolarize: +{OPEN: &{conduct_ions: +{INACTIVATE: "
            "&{repolarize: X, permanent_inactivation: end}, CLOSE_DIRECT: X}}, "
            "SUBTHRESHOLD: X}, shutdown: end}"
        ),
        description=(
            "Voltage-gated ion channel: depolarize may open (conduct ions, "
            "then inactivate or close directly) or be subthreshold (loop). "
            "Inactivated channels can repolarize (recover) or permanently "
            "inactivate."
        ),
        expected_states=6,
        expected_transitions=9,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 34. Ion Channel (Na+/K+ Parallel)
    BenchmarkProtocol(
        name="Ion Channel (Na+/K+ Parallel)",
        type_string=(
            "rec X . &{depolarize: (&{conduct_Na: +{INACTIVATE_Na: wait, "
            "CLOSE_Na: wait}} || &{conduct_K: &{delayed_close_K: wait}}) . "
            "&{repolarize: X, permanent_inactivation: end}, shutdown: end}"
        ),
        description=(
            "Parallel Na+/K+ ion channels during action potential: "
            "depolarization opens both channels concurrently — Na+ fast "
            "activation with inactivation, K+ delayed rectifier — then "
            "repolarize (recover) or permanently inactivate."
        ),
        expected_states=11,
        expected_transitions=19,
        expected_sccs=2,
        uses_parallel=True,
    ),
]
