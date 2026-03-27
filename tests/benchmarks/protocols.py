"""Benchmark protocol definitions for the reticulate test suite.

79 real-world and classic protocols expressed as session types using the
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
    # 35. Ki3 Onboarding (Tenant Provisioning)
    BenchmarkProtocol(
        name="Ki3 Onboarding",
        type_string=(
            "&{validateContract: +{APPROVED: "
            "(&{createVPS: +{PROVISIONED: wait, FAILED: wait}} || "
            "&{configureDNS: +{PROPAGATED: wait, FAILED: wait}}) . "
            "&{createKeycloakRealm: +{CREATED: "
            "(&{createSchema: &{seedData: wait}} || "
            "&{configureProxy: &{requestSSL: wait}}) . "
            "(&{setupMonitoring: &{createDashboards: wait}} || "
            "&{configureBackup: wait}) . "
            "&{runHealthChecks: +{HEALTHY: &{notifyTenant: "
            "&{activateSubscription: end}}, "
            "UNHEALTHY: &{rollback: &{notifyOps: end}}}}, "
            "FAILED: end}}, "
            "REJECTED: end}}"
        ),
        description=(
            "Ki3 SaaS tenant provisioning (Berlin daycare platform). "
            "Two-phase parallel: Phase 1 provisions VPS and DNS concurrently, "
            "then Phase 2 deploys database, proxy, monitoring, and backups "
            "concurrently. Health checks branch to activation or rollback. "
            "Real production system at ki3.tech."
        ),
        expected_states=32,
        expected_transitions=50,
        expected_sccs=32,
        uses_parallel=True,
    ),
    # 36. Ki3 Offboarding (Tenant Teardown)
    BenchmarkProtocol(
        name="Ki3 Offboarding",
        type_string=(
            "&{confirmCancellation: &{exportData: &{notifyTenant: "
            "(&{removeMonitoring: wait} || &{removeBackup: wait}) . "
            "(&{revokeSSL: wait} || &{deleteSchema: wait}) . "
            "(&{deleteKeycloakRealm: wait} || &{removeDNS: wait}) . "
            "&{deleteVPS: &{archiveAuditLog: &{closeSubscription: end}}}}}}"
        ),
        description=(
            "Ki3 SaaS tenant offboarding (reverse of onboarding). "
            "Three parallel teardown phases: services first (monitoring, backups), "
            "then middleware (SSL, database), then infrastructure (Keycloak, DNS), "
            "finally VPS deletion and audit archival. Order ensures dependencies "
            "are removed before their providers."
        ),
        expected_states=16,
        expected_transitions=18,
        expected_sccs=16,
        uses_parallel=True,
    ),
    # 37. Carnot Cycle (formerly #35)
    BenchmarkProtocol(
        name="Carnot Cycle",
        type_string=(
            "rec X . &{isothermal_expand: &{adiabatic_expand: "
            "&{isothermal_compress: &{adiabatic_compress: X, stop: end}}}}"
        ),
        description=(
            "A Carnot thermodynamic engine as a session type: four-stroke "
            "cycle (isothermal expand, adiabatic expand, isothermal compress, "
            "adiabatic compress) with optional stop. Entropy is the "
            "order-embedding of the state-space reachability into R."
        ),
        expected_states=5,
        expected_transitions=5,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 36. Quantum Measurement
    BenchmarkProtocol(
        name="Quantum Measurement",
        type_string=(
            "&{prepare: &{measure_x: +{up_x: &{measure_z: end}, "
            "down_x: &{measure_z: end}}, "
            "measure_z: +{up_z: &{measure_x: end}, "
            "down_z: &{measure_x: end}}}}"
        ),
        description=(
            "Sequential spin measurements on a qubit: prepare, then choose "
            "measurement basis (X or Z). Non-commutativity of observables "
            "means path order matters — swapping bases gives isomorphic but "
            "distinct state spaces. Inspired by Birkhoff-von Neumann (1936)."
        ),
        expected_states=9,
        expected_transitions=11,
        expected_sccs=9,
        uses_parallel=False,
    ),
    # 37. Billiard Map
    BenchmarkProtocol(
        name="Billiard Map",
        type_string=(
            "rec X . &{reflect_north: X, reflect_south: X, "
            "reflect_east: X, reflect_west: X, exit: end}"
        ),
        description=(
            "A 2D billiard table as a session type: periodic orbits as "
            "recursive reflections off four walls. Ergodic decomposition "
            "of the state space corresponds to bisimulation quotient."
        ),
        expected_states=2,
        expected_transitions=5,
        expected_sccs=2,
        uses_parallel=False,
    ),

    # 38. Ki3 CI/CD Pipeline (Dev → Staging → Production)
    BenchmarkProtocol(
        name="Ki3 CI/CD Pipeline",
        type_string=(
            "(&{lintBackend: +{PASS: wait, FAIL: wait}} || "
            "&{lintFrontend: +{PASS: wait, FAIL: wait}}) . "
            "+{LINT_OK: "
            "(&{testBackend: +{PASS: wait, FAIL: wait}} || "
            "&{testFrontend: +{PASS: wait, FAIL: wait}}) . "
            "+{TESTS_OK: "
            "(&{buildBackend: +{PUSHED: wait, FAIL: wait}} || "
            "&{buildFrontend: +{PUSHED: wait, FAIL: wait}}) . "
            "+{IMAGES_OK: "
            "&{securityScan: +{CLEAN: "
            "&{deployStaging: +{HEALTHY: "
            "&{e2eSmoke: +{PASS: "
            "&{approveProduction: +{APPROVED: "
            "&{deployProduction: +{HEALTHY: end, "
            "UNHEALTHY: &{rollbackProduction: end}}}, "
            "REJECTED: end}}, "
            "FAIL: &{rollbackStaging: end}}}, "
            "UNHEALTHY: &{rollbackStaging: end}}}, "
            "CRITICAL: end}}, "
            "BUILD_FAIL: end}, "
            "TEST_FAIL: end}, "
            "LINT_FAIL: end}"
        ),
        description=(
            "Ki3 multi-tenant SaaS CI/CD pipeline: parallel lint, parallel "
            "test, parallel Docker build, security scan, staging deploy with "
            "health check, E2E smoke tests, manual production approval, "
            "production deploy with rollback. Three parallel phases model "
            "concurrent backend/frontend pipelines."
        ),
        expected_states=41,
        expected_transitions=78,
        expected_sccs=41,
        uses_parallel=True,
    ),
    # ── Molecular Biology Benchmarks (41–48) ──────────────────
    # 41. Ribosome Translation
    BenchmarkProtocol(
        name="Ribosome Translation",
        type_string=(
            "rec X . &{sense_codon: &{match_tRNA: &{peptide_bond: X}}, "
            "stop_codon: &{release: end}}"
        ),
        description=(
            "Ribosome translation protocol: initiation, then elongation loop "
            "(sense codon, match tRNA, form peptide bond) until a stop codon "
            "triggers release factor binding and termination."
        ),
        expected_states=5,
        expected_transitions=5,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 42. Polysome (Concurrent Translation)
    BenchmarkProtocol(
        name="Polysome",
        type_string=(
            "&{initiate: (rec X . &{sense_codon: &{match_tRNA: "
            "&{peptide_bond: X}}, stop_codon: &{release: wait}} || "
            "rec Y . &{sense_codon: &{match_tRNA: &{peptide_bond: Y}}, "
            "stop_codon: &{release: wait}}) . end}"
        ),
        description=(
            "Polysome: two ribosomes translating the same mRNA concurrently. "
            "Each ribosome independently performs elongation cycles. Models "
            "concurrent access to a shared protocol (the mRNA template)."
        ),
        expected_states=26,
        expected_transitions=51,
        expected_sccs=10,
        uses_parallel=True,
    ),
    # 43. Alternative Splicing
    BenchmarkProtocol(
        name="Alternative Splicing",
        type_string=(
            "&{transcribe: &{splice: +{ISOFORM_A: &{translate: end}, "
            "ISOFORM_B: &{translate: end}, ISOFORM_C: &{translate: end}}}}"
        ),
        description=(
            "Alternative splicing: transcription produces pre-mRNA, then the "
            "spliceosome selects among three exon combinations (isoforms). "
            "Models external choice by the splicing machinery."
        ),
        expected_states=7,
        expected_transitions=8,
        expected_sccs=7,
        uses_parallel=False,
    ),
    # 44. Lac Operon Regulation
    BenchmarkProtocol(
        name="Lac Operon",
        type_string=(
            "rec X . &{sense_lactose: +{PRESENT: &{transcribe: "
            "&{translate: X}}, ABSENT: &{repress: X}}, cell_death: end}"
        ),
        description=(
            "Lac operon gene regulation: sense lactose presence, then either "
            "transcribe and translate (lactose present) or repress (absent). "
            "Recursive: the operon continuously monitors its environment. "
            "Cell death provides the termination exit."
        ),
        expected_states=6,
        expected_transitions=7,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 45. DNA Replication Fork
    BenchmarkProtocol(
        name="DNA Replication Fork",
        type_string=(
            "&{unwind: (rec X . &{synthesize_leading: X, "
            "complete_leading: wait} || rec Y . &{synthesize_okazaki: "
            "&{ligate: Y}, complete_lagging: wait}) . end}"
        ),
        description=(
            "DNA replication fork: helicase unwinds, then leading strand "
            "synthesis (continuous) runs in parallel with lagging strand "
            "synthesis (Okazaki fragments with ligation). Models the "
            "asymmetric concurrent replication mechanism."
        ),
        expected_states=7,
        expected_transitions=13,
        expected_sccs=5,
        uses_parallel=True,
    ),
    # 46. tRNA Charging (Aminoacyl-tRNA Synthetase)
    BenchmarkProtocol(
        name="tRNA Charging",
        type_string=(
            "rec X . &{bind_amino_acid: &{bind_tRNA: &{transfer: "
            "&{release: X}}}, shutdown: end}"
        ),
        description=(
            "Aminoacyl-tRNA synthetase charging cycle: bind amino acid, "
            "bind tRNA, transfer aminoacyl group, release charged tRNA, "
            "then loop. Essential for translation fidelity."
        ),
        expected_states=5,
        expected_transitions=5,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 47. mRNA Lifecycle
    BenchmarkProtocol(
        name="mRNA Lifecycle",
        type_string=(
            "&{transcribe: &{cap: &{splice: &{export: &{translate: "
            "&{decay: end}}}}}}"
        ),
        description=(
            "Complete mRNA lifecycle: transcription, 5' capping, splicing, "
            "nuclear export, translation, and finally mRNA decay. A linear "
            "protocol with no branching — each step must complete before "
            "the next begins."
        ),
        expected_states=7,
        expected_transitions=6,
        expected_sccs=7,
        uses_parallel=False,
    ),
    # 48. Protein Folding (Chaperone-Assisted)
    BenchmarkProtocol(
        name="Protein Folding",
        type_string=(
            "rec X . &{fold_attempt: +{NATIVE: end, MISFOLDED: "
            "&{chaperone_bind: &{unfold: X}}, AGGREGATE: &{degrade: end}}}"
        ),
        description=(
            "Chaperone-assisted protein folding: attempt folding, then "
            "either reach native state (end), misfold (chaperone rescues "
            "and retries), or aggregate irreversibly (targeted for "
            "degradation). Models the cellular quality control system."
        ),
        expected_states=6,
        expected_transitions=7,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # ── Cell Biology Benchmarks (49–56) ───────────────────────
    # 49. Cell Cycle (Eukaryotic)
    BenchmarkProtocol(
        name="Cell Cycle",
        type_string=(
            "rec X . &{G1_checkpoint: +{PASS: &{S_phase: &{G2_checkpoint: "
            "+{PASS: &{mitosis: X}, FAIL: &{repair: X}}}}, FAIL: "
            "&{repair: X}}, differentiate: end, apoptosis: end}"
        ),
        description=(
            "Eukaryotic cell cycle with checkpoints: G1 checkpoint "
            "(pass → S phase → G2 checkpoint → mitosis, or fail → repair), "
            "with exits to differentiation or apoptosis. Guarded recursion "
            "ensures checkpoints precede division."
        ),
        expected_states=9,
        expected_transitions=12,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 50. Signal Transduction (PKA Pathway)
    BenchmarkProtocol(
        name="Signal Transduction PKA",
        type_string=(
            "&{hormone_bind: &{activate_G_protein: &{produce_cAMP: "
            "&{activate_PKA: (&{phosphorylate: wait} || "
            "&{open_ion_channel: wait}) . end}}}}"
        ),
        description=(
            "PKA signal transduction cascade: hormone binds receptor, "
            "activates G-protein, produces cAMP, activates PKA, then "
            "PKA concurrently phosphorylates targets and opens ion "
            "channels. The parallel fork models PKA's multiple "
            "simultaneous downstream effects."
        ),
        expected_states=8,
        expected_transitions=8,
        expected_sccs=8,
        uses_parallel=True,
    ),
    # 51. ER-Golgi Secretory Pathway
    BenchmarkProtocol(
        name="ER-Golgi Secretory",
        type_string=(
            "&{synthesize: &{fold: +{NATIVE: &{ER_exit: &{golgi_sort: "
            "+{MEMBRANE: end, SECRETED: end, LYSOSOME: end}}}, MISFOLDED: "
            "&{ER_retain: &{degrade: end}}}}}"
        ),
        description=(
            "ER-Golgi secretory pathway: synthesize protein in ER, fold, "
            "then quality control branch — native proteins exit ER and are "
            "sorted by Golgi to membrane, secretion, or lysosome; misfolded "
            "proteins are retained and degraded (ERAD)."
        ),
        expected_states=9,
        expected_transitions=11,
        expected_sccs=9,
        uses_parallel=False,
    ),
    # 52. Homeostatic Glucose Regulation
    BenchmarkProtocol(
        name="Glucose Regulation",
        type_string=(
            "rec X . &{measure_glucose: +{LOW: &{release_glucagon: X}, "
            "NORMAL: X, HIGH: &{release_insulin: X}}, apoptosis: end}"
        ),
        description=(
            "Homeostatic glucose regulation: continuously measure blood "
            "glucose, respond with glucagon (low), nothing (normal), or "
            "insulin (high), then loop. Apoptosis provides the termination "
            "exit. Models pancreatic islet cell behavior."
        ),
        expected_states=5,
        expected_transitions=7,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 53. Action Potential (Full)
    BenchmarkProtocol(
        name="Action Potential Full",
        type_string=(
            "rec X . &{stimulus: +{THRESHOLD: (&{Na_activate: "
            "+{INACTIVATE: wait, CLOSE: wait}} || &{K_activate: "
            "&{K_delayed_close: wait}}) . &{repolarize: &{refractory: X}}, "
            "SUBTHRESHOLD: X}, shutdown: end}"
        ),
        description=(
            "Full action potential with parallel Na+/K+ channels: stimulus "
            "above threshold triggers concurrent Na+ activation "
            "(fast inactivation or direct close) and K+ delayed "
            "rectification, followed by repolarization and refractory "
            "period. Extends benchmark 34 with the full cycle."
        ),
        expected_states=13,
        expected_transitions=21,
        expected_sccs=2,
        uses_parallel=True,
    ),
    # 54. Apoptosis (Programmed Cell Death)
    BenchmarkProtocol(
        name="Apoptosis",
        type_string=(
            "&{receive_signal: +{INTRINSIC: &{mitochondrial_release: "
            "&{activate_caspase9: &{activate_caspase3: "
            "&{DNA_fragmentation: end}}}}, EXTRINSIC: "
            "&{activate_caspase8: &{activate_caspase3: "
            "&{DNA_fragmentation: end}}}}}"
        ),
        description=(
            "Programmed cell death (apoptosis): two initiation pathways — "
            "intrinsic (mitochondrial cytochrome c release → caspase-9) "
            "and extrinsic (death receptor → caspase-8) — both converge "
            "on caspase-3 activation and DNA fragmentation."
        ),
        expected_states=10,
        expected_transitions=10,
        expected_sccs=10,
        uses_parallel=False,
    ),
    # 55. Immune Response (T-Cell Activation)
    BenchmarkProtocol(
        name="T-Cell Activation",
        type_string=(
            "&{antigen_present: &{activate: (&{release_cytokine: wait} || "
            "&{proliferate: wait}) . +{MEMORY: end, APOPTOSIS: end}}}"
        ),
        description=(
            "T-cell immune response: antigen-presenting cell activates "
            "T-cell, then T-cell concurrently releases cytokines and "
            "proliferates. After the parallel phase, the T-cell either "
            "becomes a memory cell or undergoes apoptosis."
        ),
        expected_states=7,
        expected_transitions=8,
        expected_sccs=7,
        uses_parallel=True,
    ),
    # 56. Photosynthesis-Respiration Coupling
    BenchmarkProtocol(
        name="Photosynthesis-Respiration",
        type_string=(
            "(&{light_reactions: &{produce_ATP: wait}} || "
            "&{calvin_cycle: &{fix_carbon: wait}}) . "
            "(&{glycolysis: &{pyruvate: wait}} || "
            "&{krebs_cycle: &{electron_transport: wait}}) . end"
        ),
        description=(
            "Plant cell energy metabolism: photosynthesis (light reactions "
            "∥ Calvin cycle) followed by respiration (glycolysis ∥ Krebs "
            "cycle). Two sequential parallel phases model the concurrent "
            "sub-pathways within each metabolic process."
        ),
        expected_states=17,
        expected_transitions=24,
        expected_sccs=17,
        uses_parallel=True,
    ),
    # ── Physics Benchmarks (57–69) — Step 157i ────────────────
    # 57. QED Vertex (Quantum Electrodynamics)
    BenchmarkProtocol(
        name="QED Vertex",
        type_string=(
            "&{emit_photon: &{absorb: end}, absorb_photon: &{emit: end}}"
        ),
        description=(
            "QED vertex interaction: an electron can emit or absorb a "
            "virtual photon. The two branches model the two orientations "
            "of the fundamental QED vertex in a Feynman diagram. "
            "Crossing symmetry corresponds to session type duality."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 58. QCD Gluon Exchange (Strong Force)
    BenchmarkProtocol(
        name="QCD Gluon Exchange",
        type_string=(
            "&{emit_gluon: +{RR: &{absorb_gluon: end}, "
            "RG: &{absorb_gluon: end}, RB: &{absorb_gluon: end}}}"
        ),
        description=(
            "QCD gluon exchange between quarks: emit a gluon carrying "
            "a colour charge (red-red, red-green, red-blue), then the "
            "target quark absorbs. The selection models the colour "
            "charge assignment from the SU(3) gauge group."
        ),
        expected_states=6,
        expected_transitions=7,
        expected_sccs=6,
        uses_parallel=False,
    ),
    # 59. Weak Decay (Beta Decay)
    BenchmarkProtocol(
        name="Weak Decay (Beta)",
        type_string=(
            "&{W_boson_emit: +{BETA_MINUS: &{electron: "
            "&{antineutrino: end}}, BETA_PLUS: &{positron: "
            "&{neutrino: end}}}}"
        ),
        description=(
            "Weak-force beta decay: a quark emits a W boson, then "
            "either beta-minus (W⁻ → electron + antineutrino) or "
            "beta-plus (W⁺ → positron + neutrino). Models the "
            "electroweak vertex with lepton-number conservation."
        ),
        expected_states=7,
        expected_transitions=7,
        expected_sccs=7,
        uses_parallel=False,
    ),
    # 60. Double Slit (Unobserved — Interference)
    BenchmarkProtocol(
        name="Double Slit Unobserved",
        type_string=(
            "&{emit: (&{propagate_A: wait} || &{propagate_B: wait}) "
            ". &{interfere: end}}"
        ),
        description=(
            "Double-slit experiment without observation: particle is "
            "emitted, propagates through both slits in parallel "
            "(superposition as ∥), then the paths recombine to "
            "produce an interference pattern. The parallel constructor "
            "models quantum superposition."
        ),
        expected_states=6,
        expected_transitions=6,
        expected_sccs=6,
        uses_parallel=True,
    ),
    # 61. Double Slit (Observed — Collapse)
    BenchmarkProtocol(
        name="Double Slit Observed",
        type_string=(
            "&{emit: +{SLIT_A: &{detect_A: end}, "
            "SLIT_B: &{detect_B: end}}}"
        ),
        description=(
            "Double-slit experiment with observation: measurement "
            "collapses superposition (∥) to selection (⊕). The "
            "particle chooses one slit. No interference pattern. "
            "Demonstrates measurement as the ∥→⊕ collapse."
        ),
        expected_states=5,
        expected_transitions=5,
        expected_sccs=5,
        uses_parallel=False,
    ),
    # 62. Bell Pair (EPR Entanglement)
    BenchmarkProtocol(
        name="Bell Pair",
        type_string=(
            "(&{measure_A: +{UP_A: wait, DOWN_A: wait}} || "
            "&{measure_B: +{UP_B: wait, DOWN_B: wait}}) . end"
        ),
        description=(
            "Bell pair measurement: two entangled particles are "
            "measured concurrently. Each measurement selects spin up "
            "or down. The product lattice L(A)×L(B) has 4 outcome "
            "states, but Bell correlations constrain the reachable "
            "subset. Models EPR as parallel session composition."
        ),
        expected_states=9,
        expected_transitions=18,
        expected_sccs=9,
        uses_parallel=True,
    ),
    # 63. Quantum Teleportation
    BenchmarkProtocol(
        name="Quantum Teleportation",
        type_string=(
            "&{prepare_bell: (&{alice_measure: +{PHI_PLUS: wait, "
            "PHI_MINUS: wait, PSI_PLUS: wait, PSI_MINUS: wait}} || "
            "&{bob_receive: &{classical_bits: +{CORRECT_00: wait, "
            "CORRECT_01: wait, CORRECT_10: wait, CORRECT_11: wait}}}) "
            ". &{reconstruct: end}}"
        ),
        description=(
            "Quantum teleportation protocol: prepare a Bell pair, then "
            "Alice measures (4 Bell states) in parallel with Bob "
            "receiving classical correction bits (4 options). After "
            "synchronisation, Bob reconstructs the teleported state. "
            "Models the full Bennett et al. (1993) protocol."
        ),
        expected_states=14,
        expected_transitions=40,
        expected_sccs=14,
        uses_parallel=True,
    ),
    # 64. Hydrogen Atom (Energy Levels)
    BenchmarkProtocol(
        name="Hydrogen Atom",
        type_string=(
            "rec X . &{absorb_photon: +{EXCITE: X, IONIZE: end}, "
            "emit_photon: X, ground_state: end}"
        ),
        description=(
            "Hydrogen atom energy transitions: repeatedly absorb "
            "photons (excite to higher level or ionize) or emit "
            "photons (relax). The recursion models discrete energy "
            "levels; ionization and ground-state decay are exits."
        ),
        expected_states=3,
        expected_transitions=5,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 65. Cascade Decay (Nuclear/Atomic)
    BenchmarkProtocol(
        name="Cascade Decay",
        type_string=(
            "&{excited_3: +{EMIT_GAMMA1: &{excited_2: +{EMIT_GAMMA2: "
            "&{excited_1: +{EMIT_GAMMA3: end}}}}, DIRECT: end}}"
        ),
        description=(
            "Three-level cascade decay: an excited nucleus or atom "
            "in state |3> can emit gamma-1 to reach |2>, then "
            "gamma-2 to |1>, then gamma-3 to ground state, or "
            "directly decay. Models sequential de-excitation with "
            "branching at each level."
        ),
        expected_states=7,
        expected_transitions=7,
        expected_sccs=7,
        uses_parallel=False,
    ),
    # 66. Stellar Nucleosynthesis
    BenchmarkProtocol(
        name="Stellar Nucleosynthesis",
        type_string=(
            "rec X . &{fuse: +{HELIUM: &{contract: &{heat: X}}, "
            "CARBON: &{contract: &{heat: X}}, "
            "OXYGEN: &{contract: &{heat: X}}, "
            "IRON: &{collapse: +{SUPERNOVA: end, WHITE_DWARF: end}}}}"
        ),
        description=(
            "Stellar nucleosynthesis: a star repeatedly fuses lighter "
            "elements into heavier ones (H->He->C->O), each time "
            "contracting and heating. When iron is reached, fusion "
            "becomes endothermic: the star either goes supernova "
            "or collapses to a white dwarf. Models the onion-shell "
            "burning sequence."
        ),
        expected_states=11,
        expected_transitions=14,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 67. Big Bang Nucleosynthesis (BBN)
    BenchmarkProtocol(
        name="Big Bang Nucleosynthesis",
        type_string=(
            "&{quark_gluon_plasma: &{hadronize: "
            "(&{proton_neutron_ratio: &{freeze_out: wait}} || "
            "&{deuterium_form: &{helium_fuse: wait}}) "
            ". &{primordial_abundances: end}}}"
        ),
        description=(
            "Big Bang nucleosynthesis (BBN): quark-gluon plasma "
            "hadronizes, then two parallel channels — proton-neutron "
            "ratio freeze-out and light element fusion (D->He) — "
            "run concurrently. After synchronisation, primordial "
            "abundances are fixed. Models the first 3 minutes."
        ),
        expected_states=12,
        expected_transitions=15,
        expected_sccs=12,
        uses_parallel=True,
    ),
    # 68. Gravitational Collapse
    BenchmarkProtocol(
        name="Gravitational Collapse",
        type_string=(
            "&{accrete: rec X . &{compress: +{STABLE: &{radiate: X}, "
            "CHANDRASEKHAR: +{WHITE_DWARF: end, NEUTRON_STAR: end, "
            "BLACK_HOLE: end}}}}"
        ),
        description=(
            "Gravitational collapse: matter accretes, then a "
            "compression loop — if stable, radiate and continue; "
            "if the Chandrasekhar limit is reached, collapse to "
            "a white dwarf, neutron star, or black hole. Models "
            "the Tolman-Oppenheimer-Volkoff stability criterion."
        ),
        expected_states=6,
        expected_transitions=8,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 69. Black Hole (Hawking Radiation)
    BenchmarkProtocol(
        name="Black Hole Hawking",
        type_string=(
            "rec X . &{accrete_matter: X, hawking_emit: +{PHOTON: X, "
            "PARTICLE: X, EVAPORATE: end}, merge: end}"
        ),
        description=(
            "Black hole lifecycle with Hawking radiation: the black "
            "hole can accrete matter (grow), emit Hawking radiation "
            "(photon or particle, then loop), or eventually "
            "evaporate completely. Merger with another black hole "
            "is an alternative exit. Models the information paradox "
            "as irreversible terminal states."
        ),
        expected_states=3,
        expected_transitions=6,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # ── Security Protocol Benchmarks (70–79) ─────────────────
    # 70. Needham-Schroeder Public Key
    BenchmarkProtocol(
        name="Needham-Schroeder",
        type_string=(
            "&{alice_nonce: &{bob_nonce_reply: &{alice_confirm: "
            "+{AUTHENTICATED: end, REPLAY_DETECTED: end}}}}"
        ),
        description=(
            "Needham-Schroeder public-key authentication protocol: Alice "
            "sends a nonce, Bob replies with both nonces encrypted, Alice "
            "confirms. The selection models the verifier's decision — "
            "authenticated or replay attack detected. Lowe (1995) found "
            "the man-in-the-middle attack on the original 1978 protocol."
        ),
        expected_states=5,
        expected_transitions=5,
        expected_sccs=5,
        uses_parallel=False,
    ),
    # 71. Kerberos Authentication
    BenchmarkProtocol(
        name="Kerberos",
        type_string=(
            "&{request_TGT: +{TGT_GRANTED: &{request_service_ticket: "
            "+{TICKET_GRANTED: &{authenticate_to_service: "
            "+{ACCESS_GRANTED: rec X . &{use_service: X, renew_ticket: "
            "+{RENEWED: X, EXPIRED: end}, logout: end}, "
            "ACCESS_DENIED: end}}, TICKET_DENIED: end}}, TGT_DENIED: end}}"
        ),
        description=(
            "Kerberos authentication: request TGT from KDC, then request "
            "service ticket from TGS, then authenticate to service. Each "
            "step is gated by a selection (granted/denied). After access, "
            "the service session supports use, ticket renewal, and logout. "
            "Models the MIT Kerberos V5 three-party authentication flow."
        ),
        expected_states=9,
        expected_transitions=14,
        expected_sccs=8,
        uses_parallel=False,
    ),
    # 72. SSH Handshake
    BenchmarkProtocol(
        name="SSH Handshake",
        type_string=(
            "&{protocol_version: &{key_exchange: &{server_auth: "
            "+{HOST_VERIFIED: &{user_auth: +{AUTH_SUCCESS: "
            "rec X . &{open_channel: &{exec_command: +{OUTPUT: X, "
            "EXIT: &{close_channel: X}}}, disconnect: end}, "
            "AUTH_FAILED: end}}, HOST_UNKNOWN: +{ACCEPT: &{user_auth: "
            "+{AUTH_SUCCESS: rec X . &{open_channel: &{exec_command: "
            "+{OUTPUT: X, EXIT: &{close_channel: X}}}, disconnect: end}, "
            "AUTH_FAILED: end}}, REJECT: end}}}}}"
        ),
        description=(
            "SSH handshake (RFC 4253): protocol version exchange, key "
            "exchange (DH), server host key verification (verified or "
            "unknown with accept/reject), user authentication, then a "
            "multiplexed channel session with command execution. The "
            "HOST_UNKNOWN branch models the trust-on-first-use pattern."
        ),
        expected_states=18,
        expected_transitions=25,
        expected_sccs=12,
        uses_parallel=False,
    ),
    # 73. Diffie-Hellman Key Exchange
    BenchmarkProtocol(
        name="Diffie-Hellman",
        type_string=(
            "&{send_params: &{send_public_A: &{receive_public_B: "
            "+{VALID: &{derive_shared_secret: &{verify_key: "
            "+{CONFIRMED: end, MISMATCH: end}}}, INVALID_PARAMS: end}}}}"
        ),
        description=(
            "Diffie-Hellman key exchange: send group parameters, exchange "
            "public values, validate parameters, derive shared secret, "
            "verify key agreement. The INVALID_PARAMS and MISMATCH exits "
            "model parameter downgrade attacks and key confirmation failure."
        ),
        expected_states=8,
        expected_transitions=9,
        expected_sccs=8,
        uses_parallel=False,
    ),
    # 74. Mutual TLS (mTLS)
    BenchmarkProtocol(
        name="Mutual TLS",
        type_string=(
            "&{client_hello: +{SERVER_HELLO: "
            "(&{server_cert_verify: +{TRUSTED: wait, UNTRUSTED: wait}} || "
            "&{client_cert_send: +{ACCEPTED: wait, REJECTED: wait}}) "
            ". +{BOTH_OK: &{establish_channel: end}, AUTH_FAIL: end}, "
            "INCOMPATIBLE: end}}"
        ),
        description=(
            "Mutual TLS: after client/server hello, both parties verify "
            "each other's certificates concurrently (server cert + client "
            "cert in parallel). After synchronisation, a joint decision "
            "determines whether both verifications passed. The parallel "
            "constructor models simultaneous certificate validation."
        ),
        expected_states=13,
        expected_transitions=24,
        expected_sccs=13,
        uses_parallel=True,
    ),
    # 75. Signal Protocol (Double Ratchet)
    BenchmarkProtocol(
        name="Signal Protocol",
        type_string=(
            "&{establish_session: &{x3dh_handshake: "
            "+{KEYS_DERIVED: rec X . &{send_message: &{ratchet_step: X}, "
            "receive_message: &{ratchet_step: X}, rekey: X, "
            "close_session: end}, HANDSHAKE_FAILED: end}}}"
        ),
        description=(
            "Signal Protocol with X3DH and Double Ratchet: establish a "
            "session via Extended Triple Diffie-Hellman, then enter the "
            "messaging loop with forward secrecy (each message ratchets "
            "the key). Supports send, receive, explicit rekey, and session "
            "close. Models the core of Signal/WhatsApp/Matrix encryption."
        ),
        expected_states=7,
        expected_transitions=10,
        expected_sccs=5,
        uses_parallel=False,
    ),
    # 76. X.509 Certificate Chain Validation
    BenchmarkProtocol(
        name="Certificate Chain",
        type_string=(
            "&{receive_cert: &{check_signature: +{VALID_SIG: "
            "&{check_issuer: +{ROOT_CA: &{check_expiry: "
            "+{NOT_EXPIRED: &{check_revocation: +{NOT_REVOKED: end, "
            "REVOKED: end}}, EXPIRED: end}}, INTERMEDIATE: "
            "&{validate_parent: +{CHAIN_VALID: &{check_expiry: "
            "+{NOT_EXPIRED: &{check_revocation: +{NOT_REVOKED: end, "
            "REVOKED: end}}, EXPIRED: end}}, CHAIN_BROKEN: end}}}}, "
            "INVALID_SIG: end}}}"
        ),
        description=(
            "X.509 certificate chain validation: receive certificate, "
            "check signature, then branch on issuer type (root CA or "
            "intermediate). For intermediates, recursively validate the "
            "parent certificate. Each path checks expiry and revocation "
            "status (CRL/OCSP). Models the PKI trust chain traversal."
        ),
        expected_states=16,
        expected_transitions=22,
        expected_sccs=16,
        uses_parallel=False,
    ),
    # 77. SAML Single Sign-On
    BenchmarkProtocol(
        name="SAML SSO",
        type_string=(
            "&{sp_request: &{redirect_to_idp: &{user_authenticate: "
            "+{AUTH_OK: &{saml_assertion: &{send_assertion_to_sp: "
            "+{VALID: &{create_session: end}, "
            "INVALID_ASSERTION: end}}}, AUTH_FAIL: end}}}}"
        ),
        description=(
            "SAML 2.0 Single Sign-On: service provider redirects user to "
            "identity provider, user authenticates, IdP issues SAML "
            "assertion, SP validates assertion and creates session. "
            "Models the browser redirect SSO profile with assertion "
            "validation as a security gate."
        ),
        expected_states=9,
        expected_transitions=10,
        expected_sccs=9,
        uses_parallel=False,
    ),
    # 78. Zero-Knowledge Proof (Interactive)
    BenchmarkProtocol(
        name="Zero-Knowledge Proof",
        type_string=(
            "rec X . &{prover_commit: &{verifier_challenge: "
            "&{prover_response: +{ACCEPT: X, REJECT: end}}}, "
            "complete_proof: end}"
        ),
        description=(
            "Interactive zero-knowledge proof: the prover commits, the "
            "verifier sends a random challenge, the prover responds. "
            "The verifier accepts (loop for more rounds) or rejects. "
            "After sufficient rounds, the prover declares proof complete. "
            "Models Schnorr identification and similar sigma protocols."
        ),
        expected_states=5,
        expected_transitions=6,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 79. DNSSEC Validation
    BenchmarkProtocol(
        name="DNSSEC",
        type_string=(
            "&{query: &{receive_response: &{check_RRSIG: "
            "+{SIG_VALID: &{follow_chain: +{DS_MATCH: "
            "&{verify_DNSKEY: +{TRUSTED: end, UNTRUSTED: end}}, "
            "DS_MISSING: end}}, SIG_INVALID: end, "
            "NO_DNSSEC: +{INSECURE_OK: end, REQUIRE_SECURE: end}}}}}"
        ),
        description=(
            "DNSSEC validation chain: query DNS, receive response with "
            "RRSIG, validate signature, follow DS record chain to trust "
            "anchor, verify DNSKEY. Handles unsigned zones (NO_DNSSEC) "
            "with policy decision. Models RFC 4033-4035 validation."
        ),
        expected_states=10,
        expected_transitions=14,
        expected_sccs=10,
        uses_parallel=False,
    ),
    # ─── Dialogue Types (Step 18) ───────────────────────────────
    # 80. Socratic Questioning
    BenchmarkProtocol(
        name="Socratic Questioning",
        type_string="rec X . &{question: +{answer: X, aporia: end}}",
        description=(
            "Lorenzen-style Socratic dialogue: Opponent repeatedly questions, "
            "Proponent answers (continuing) or reaches aporia (terminating). "
            "Step 18 dialogue benchmark."
        ),
        expected_states=3,
        expected_transitions=3,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 81. Mathematical Proof Dialogue
    BenchmarkProtocol(
        name="Mathematical Proof Dialogue",
        type_string=(
            "&{conjecture: +{prove: rec X . &{challenge: "
            "+{justify: X, qed: end}}, abandon: end}}"
        ),
        description=(
            "Dialogue for mathematical proof: Opponent poses conjecture, "
            "Proponent proves (iterative challenge-justify loop) or abandons. "
            "Step 18 dialogue benchmark."
        ),
        expected_states=5,
        expected_transitions=6,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 82. Legal Cross-Examination
    BenchmarkProtocol(
        name="Legal Cross-Examination",
        type_string=(
            "rec X . &{examine: +{answer: &{followup: X, accept: end}, "
            "refuse: end, object: end}}"
        ),
        description=(
            "Legal cross-examination: examiner questions, witness answers "
            "(with follow-up or acceptance), refuses, or objects. "
            "Step 18 dialogue benchmark."
        ),
        expected_states=4,
        expected_transitions=6,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 83. AI Agent Negotiation
    BenchmarkProtocol(
        name="AI Agent Negotiation",
        type_string="rec X . &{propose: +{accept: end, counter: X, reject: end}}",
        description=(
            "MCP/A2A-style agent negotiation: proposer offers, responder "
            "accepts, rejects, or counter-proposes (looping). "
            "Step 18 dialogue benchmark."
        ),
        expected_states=3,
        expected_transitions=4,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # ─── GoF Design Pattern Benchmarks (84–94) — Step 51 ──────
    # 84. GoF Observer
    BenchmarkProtocol(
        name="GoF Observer",
        type_string="rec X . &{subscribe: X, notify: +{update: X, unsubscribe: end}}",
        description=(
            "Observer pattern: subject maintains a subscription loop; on notify, "
            "the server selects update (continue observing) or unsubscribe (end). "
            "Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=4,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 85. GoF State
    BenchmarkProtocol(
        name="GoF State",
        type_string=(
            "&{request: +{idle: rec X . &{request: +{idle: X, active: end}}, "
            "active: end}}"
        ),
        description=(
            "State pattern: client sends request, server transitions between "
            "idle (recursive) and active (terminal) states. The selection models "
            "internal state change. Step 51 GoF benchmark."
        ),
        expected_states=5,
        expected_transitions=6,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 86. GoF Strategy
    BenchmarkProtocol(
        name="GoF Strategy",
        type_string="+{strategyA: &{execute: end}, strategyB: &{execute: end}}",
        description=(
            "Strategy pattern: server selects an algorithm (strategyA or "
            "strategyB), then client calls execute. The selection models "
            "the runtime algorithm choice. Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 87. GoF Command
    BenchmarkProtocol(
        name="GoF Command",
        type_string="&{execute: +{OK: end, UNDO: &{undo: end}}}",
        description=(
            "Command pattern: client executes a command, server selects OK "
            "(done) or UNDO (client must call undo). Models the execute/undo "
            "protocol of the Command pattern. Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 88. GoF Chain of Responsibility
    BenchmarkProtocol(
        name="GoF Chain of Responsibility",
        type_string="rec X . &{handle: +{handled: end, pass: X}}",
        description=(
            "Chain of Responsibility pattern: client sends handle request, "
            "server either handles it (end) or passes to the next handler "
            "(recurse). Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=3,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 89. GoF Template Method
    BenchmarkProtocol(
        name="GoF Template Method",
        type_string=(
            "&{step1: &{step2: +{hookA: &{step3: end}, hookB: &{step3: end}}}}"
        ),
        description=(
            "Template Method pattern: fixed algorithm skeleton (step1, step2, "
            "step3) with a hook point where the server selects the variant "
            "(hookA or hookB). Step 51 GoF benchmark."
        ),
        expected_states=6,
        expected_transitions=6,
        expected_sccs=6,
        uses_parallel=False,
    ),
    # 90. GoF Visitor
    BenchmarkProtocol(
        name="GoF Visitor",
        type_string="&{accept: +{visitA: end, visitB: end, visitC: end}}",
        description=(
            "Visitor pattern: client calls accept, then the element dispatches "
            "to the appropriate visit method (visitA, visitB, or visitC) via "
            "selection. Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=4,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 91. GoF Composite
    BenchmarkProtocol(
        name="GoF Composite",
        type_string="rec X . &{add: X, remove: X, operation: end}",
        description=(
            "Composite pattern: client can add children, remove children "
            "(both recurse), or call operation (leaf behavior, end). "
            "Step 51 GoF benchmark."
        ),
        expected_states=2,
        expected_transitions=3,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 92. GoF Decorator
    BenchmarkProtocol(
        name="GoF Decorator",
        type_string="&{extra: &{base: end}}",
        description=(
            "Decorator pattern: client calls the extra (decorated) method, "
            "then the base method. The decorator wraps the base protocol "
            "with additional behavior. Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=2,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 93. GoF Factory Method
    BenchmarkProtocol(
        name="GoF Factory Method",
        type_string="+{createA: &{use: end}, createB: &{use: end}}",
        description=(
            "Factory Method pattern: server selects which product to create "
            "(createA or createB), then client uses the product. The selection "
            "models the factory's product decision. Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 94. GoF Builder
    BenchmarkProtocol(
        name="GoF Builder",
        type_string="&{setA: &{setB: &{build: end}}}",
        description=(
            "Builder pattern: client sequentially sets configuration (setA, "
            "setB) then calls build. The linear protocol models the builder's "
            "step-by-step construction. Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=3,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 95. GoF Adapter
    BenchmarkProtocol(
        name="GoF Adapter",
        type_string="&{request: &{specificRequest: end}}",
        description=(
            "Adapter pattern: client calls request, adapter delegates to "
            "specificRequest on the adaptee. The wrapper is a sequential "
            "session type. Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=2,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 96. GoF Bridge
    BenchmarkProtocol(
        name="GoF Bridge",
        type_string="+{implA: &{operation: end}, implB: &{operation: end}}",
        description=(
            "Bridge pattern: abstraction selects implementation (implA or "
            "implB) via internal choice, then client calls operation. "
            "Decouples abstraction from implementation. Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 97. GoF Facade
    BenchmarkProtocol(
        name="GoF Facade",
        type_string="&{simplifiedOp: end}",
        description=(
            "Facade pattern: single simplified entry point hiding a complex "
            "subsystem. The simplest possible session type (one method). "
            "Step 51 GoF benchmark."
        ),
        expected_states=2,
        expected_transitions=1,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 98. GoF Flyweight
    BenchmarkProtocol(
        name="GoF Flyweight",
        type_string="&{getFlyweight: +{cached: &{draw: end}, created: &{draw: end}}}",
        description=(
            "Flyweight pattern: client requests flyweight, server returns "
            "either a cached or newly created instance, then client draws. "
            "The selection models the caching decision. Step 51 GoF benchmark."
        ),
        expected_states=5,
        expected_transitions=5,
        expected_sccs=5,
        uses_parallel=False,
    ),
    # 99. GoF Proxy
    BenchmarkProtocol(
        name="GoF Proxy",
        type_string="&{request: +{allowed: &{realRequest: end}, denied: end}}",
        description=(
            "Proxy pattern: client requests, proxy checks access and either "
            "forwards to real subject (allowed) or denies. The selection "
            "models the access control decision. Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 100. GoF Abstract Factory
    BenchmarkProtocol(
        name="GoF Abstract Factory",
        type_string="+{familyA: &{createProduct: end}, familyB: &{createProduct: end}}",
        description=(
            "Abstract Factory pattern: server selects product family (familyA "
            "or familyB), then client creates a product. The selection models "
            "the factory family decision. Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=4,
        uses_parallel=False,
    ),
    # 101. GoF Prototype
    BenchmarkProtocol(
        name="GoF Prototype",
        type_string="&{clone: +{shallow: end, deep: end}}",
        description=(
            "Prototype pattern: client clones the prototype, server selects "
            "shallow or deep copy. Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=3,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 102. GoF Singleton
    BenchmarkProtocol(
        name="GoF Singleton",
        type_string="&{getInstance: &{use: end}}",
        description=(
            "Singleton pattern: client gets the single instance, then uses "
            "it. Sequential two-step protocol. Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=2,
        expected_sccs=3,
        uses_parallel=False,
    ),
    # 103. GoF Interpreter
    BenchmarkProtocol(
        name="GoF Interpreter",
        type_string="rec X . &{interpret: +{terminal: end, nonterminal: X}}",
        description=(
            "Interpreter pattern: client interprets an expression; server "
            "selects terminal (end) or nonterminal (recurse into sub-expression). "
            "Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=3,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 104. GoF Mediator
    BenchmarkProtocol(
        name="GoF Mediator",
        type_string="rec X . &{notify: +{forward: X, resolved: end}}",
        description=(
            "Mediator pattern: colleague notifies mediator, mediator either "
            "forwards to another colleague (recurse) or resolves the "
            "interaction (end). Step 51 GoF benchmark."
        ),
        expected_states=3,
        expected_transitions=3,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # 105. GoF Memento
    BenchmarkProtocol(
        name="GoF Memento",
        type_string="rec X . &{save: &{doWork: +{restore: X, commit: end}}}",
        description=(
            "Memento pattern: save state, do work, then either restore "
            "(rollback to saved state, recurse) or commit (finalize). "
            "Step 51 GoF benchmark."
        ),
        expected_states=4,
        expected_transitions=4,
        expected_sccs=2,
        uses_parallel=False,
    ),
    # ------------------------------------------------------------------
    # Game theory benchmarks (106–108)
    # ------------------------------------------------------------------
    # 106. Nim (3 objects, 2 players)
    BenchmarkProtocol(
        name="Nim (3 objects)",
        type_string=(
            "+{take1: &{take1: +{take1: end}, take2: end}, "
            "take2: &{take1: end}, "
            "take3: end}"
        ),
        description=(
            "Nim with 3 objects and 2 players (normal play convention). "
            "Player 1 (Select) removes 1–3 objects, Player 2 (Branch) responds. "
            "Last player to take wins. Full game tree as session type."
        ),
        expected_states=5,
        expected_transitions=7,
        expected_sccs=5,
        uses_parallel=False,
    ),
    # 107. Hex (2x2 board)
    BenchmarkProtocol(
        name="Hex (2x2)",
        type_string=(
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
        ),
        description=(
            "Hex on a 2x2 board with full game tree. Cells a=(0,0), b=(0,1), "
            "c=(1,0), d=(1,1). Player 1 (Select) connects top–bottom, "
            "Player 2 (Branch) connects left–right. Self-dual session type — "
            "duality = player swap. No draws possible (Hex theorem)."
        ),
        expected_states=30,
        expected_transitions=52,
        expected_sccs=30,
        uses_parallel=False,
    ),
    # 108. Dominion Attack/Moat Reaction
    BenchmarkProtocol(
        name="Dominion Attack/Moat",
        type_string=(
            "(+{play_attack: wait} || &{react: end, pass: end}) "
            ". &{resolve: end}"
        ),
        description=(
            "Dominion card game: Attack/Moat reaction window. Attacker plays "
            "concurrently with defender's reaction choice (Moat or pass). "
            "Uses parallel constructor for genuine fork-join concurrency. "
            "After synchronization, attack resolves."
        ),
        expected_states=5,
        expected_transitions=7,
        expected_sccs=5,
        uses_parallel=True,
    ),
]
