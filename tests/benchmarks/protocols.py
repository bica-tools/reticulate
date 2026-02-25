"""Benchmark protocol definitions for the reticulate test suite.

15 real-world and classic protocols expressed as session types, covering
all constructors: branch (&), selection (+), parallel (||), recursion (rec),
and sequencing (.).
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
        type_string="open . rec X . &{read: +{data: X, eof: close . end}}",
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
            "connect . ehlo . rec X . &{mail: rcpt . data . "
            "+{OK: X, ERR: X}, quit: end}"
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
            "connect . rec X . &{request: +{OK200: readBody . X, "
            "ERR4xx: X, ERR5xx: X}, close: end}"
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
            "requestAuth . +{GRANTED: getToken . +{TOKEN: rec X . "
            "&{useToken: X, refreshToken: +{OK: X, EXPIRED: end}, "
            "revoke: end}, ERROR: end}, DENIED: end}"
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
            "lookup . getPrice . (proposeA . end || "
            "proposeB . +{ACCEPT: pay . end, REJECT: end})"
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
            "initialize . (rec X . &{callTool: +{RESULT: X, ERROR: X}, "
            "listTools: X, shutdown: end} || "
            "rec Y . +{NOTIFICATION: Y, DONE: end})"
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
            "sendTask . rec X . +{WORKING: &{getStatus: X, cancel: end}, "
            "COMPLETED: getArtifact . end, FAILED: end}"
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
            "open . +{OK: (rec X . &{read: X, doneRead: end} || "
            "rec Y . &{write: Y, doneWrite: end}) . close . end, ERR: end}"
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
            "insertCard . enterPIN . +{AUTH: rec X . "
            "&{checkBalance: X, withdraw: +{OK: X, INSUFFICIENT: X}, "
            "deposit: X, ejectCard: end}, REJECTED: ejectCard . end}"
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
            "connect . +{OPEN: (rec X . &{send: X, ping: X, "
            "closeSend: end} || rec Y . +{MESSAGE: Y, PONG: Y, "
            "CLOSE: end}), REFUSED: end}"
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
            "begin . rec X . &{query: +{RESULT: X, ERROR: X}, "
            "update: +{ROWS: X, ERROR: X}, commit: end, rollback: end}"
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
            "connect . rec X . &{subscribe: X, unsubscribe: X, "
            "poll: +{MSG: X, EMPTY: X}, disconnect: end}"
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
            "input . parse . +{OK: buildStateSpace . +{OK: "
            "(checkLattice . checkTermination . checkWFPar . end "
            "|| renderDiagram . end) . returnResult . end, "
            "ERROR: end}, ERROR: end}"
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
            "trigger . checkout . setup . "
            "(lint . end || test . end) . "
            "+{PASS: deploy . +{OK: end, FAIL: rollback . end}, FAIL: end}"
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
]
