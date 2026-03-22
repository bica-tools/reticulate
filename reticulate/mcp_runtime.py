"""Runtime conformance tester for MCP/A2A servers (Step 70b).

Connects to a running MCP or A2A server and verifies that it follows
the protocol state machine. Sends actual JSON-RPC messages and checks:

1. Valid transitions are accepted (server responds correctly).
2. Invalid transitions are rejected (server returns an error).
3. The server's state machine matches the session type.

Usage:
    from reticulate.mcp_runtime import McpConformanceTester
    tester = McpConformanceTester("http://localhost:3000")
    report = tester.run_all()
    print(report.summary())
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JsonRpcRequest:
    """A JSON-RPC 2.0 request."""
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int | str = 1

    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class JsonRpcResponse:
    """A parsed JSON-RPC 2.0 response."""
    id: int | str | None
    result: Any = None
    error: dict[str, Any] | None = None
    raw: str = ""

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def error_code(self) -> int | None:
        if self.error and "code" in self.error:
            return self.error["code"]
        return None


def parse_jsonrpc_response(raw: str) -> JsonRpcResponse:
    """Parse a JSON-RPC 2.0 response string."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return JsonRpcResponse(id=None, error={"code": -1, "message": "Invalid JSON"}, raw=raw)

    return JsonRpcResponse(
        id=data.get("id"),
        result=data.get("result"),
        error=data.get("error"),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# MCP message templates
# ---------------------------------------------------------------------------

MCP_MESSAGES: dict[str, dict[str, Any]] = {
    "initialize": {
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "reticulate-conformance-tester",
                "version": "0.1.0",
            },
        },
    },
    "listTools": {
        "method": "tools/list",
        "params": {},
    },
    "callTool": {
        "method": "tools/call",
        "params": {
            "name": "__conformance_test_tool__",
            "arguments": {},
        },
    },
    "shutdown": {
        "method": "shutdown",
        "params": {},
    },
}

A2A_MESSAGES: dict[str, dict[str, Any]] = {
    "sendTask": {
        "method": "tasks/send",
        "params": {
            "id": "conformance-test-task-001",
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "conformance test"}],
            },
        },
    },
    "getStatus": {
        "method": "tasks/get",
        "params": {"id": "conformance-test-task-001"},
    },
    "cancel": {
        "method": "tasks/cancel",
        "params": {"id": "conformance-test-task-001"},
    },
    "getArtifact": {
        "method": "tasks/get",
        "params": {"id": "conformance-test-task-001"},
    },
}


# ---------------------------------------------------------------------------
# Test result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestStep:
    """A single step in a conformance test."""
    label: str
    request: JsonRpcRequest
    response: JsonRpcResponse | None
    expected_success: bool
    actual_success: bool | None
    passed: bool
    error_message: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class TestCase:
    """A complete conformance test case (sequence of steps)."""
    name: str
    kind: str  # "valid_path", "violation", "state_check"
    steps: tuple[TestStep, ...]
    passed: bool

    @property
    def num_steps(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class RuntimeConformanceReport:
    """Complete runtime conformance report."""
    protocol: str
    target: str
    test_cases: tuple[TestCase, ...]
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    transition_coverage: float
    error_message: str = ""

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  RUNTIME CONFORMANCE: {self.protocol}")
        lines.append(f"  Target: {self.target}")
        lines.append("=" * 60)
        lines.append("")

        if self.error_message:
            lines.append(f"  CONNECTION ERROR: {self.error_message}")
            lines.append("")
            return "\n".join(lines)

        for tc in self.test_cases:
            status = "PASS" if tc.passed else "FAIL"
            lines.append(f"  [{status}] {tc.name} ({tc.kind})")
            for step in tc.steps:
                s = "ok" if step.passed else "FAIL"
                lines.append(f"       {step.label}: {s}")
                if not step.passed and step.error_message:
                    lines.append(f"         -> {step.error_message}")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"  Total:   {self.total_tests}")
        lines.append(f"  Passed:  {self.passed_tests}")
        lines.append(f"  Failed:  {self.failed_tests}")
        lines.append(f"  Skipped: {self.skipped_tests}")
        lines.append(f"  Rate:    {self.pass_rate:.1%}")
        lines.append(f"  Coverage:{self.transition_coverage:.1%}")
        lines.append("=" * 60)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------

class HttpTransport:
    """Send JSON-RPC requests over HTTP POST."""

    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.timeout = timeout

    def send(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Send a JSON-RPC request and return the response."""
        import time
        data = request.to_json().encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return parse_jsonrpc_response(raw)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8") if e.fp else ""
            return parse_jsonrpc_response(raw) if raw else JsonRpcResponse(
                id=request.id,
                error={"code": e.code, "message": str(e)},
                raw=str(e),
            )
        except urllib.error.URLError as e:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -1, "message": f"Connection failed: {e.reason}"},
                raw=str(e),
            )
        except Exception as e:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -1, "message": str(e)},
                raw=str(e),
            )


class MockTransport:
    """Mock transport for testing without a real server."""

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        reject_methods: set[str] | None = None,
    ):
        self.responses = responses or {}
        self.reject_methods = reject_methods or set()
        self.call_log: list[JsonRpcRequest] = []

    def send(self, request: JsonRpcRequest) -> JsonRpcResponse:
        self.call_log.append(request)
        method = request.method
        if method in self.reject_methods:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not allowed: {method}"},
            )
        result = self.responses.get(method, {"status": "ok"})
        return JsonRpcResponse(id=request.id, result=result)


# ---------------------------------------------------------------------------
# Protocol state tracker
# ---------------------------------------------------------------------------

class ProtocolStateTracker:
    """Track the current state in the protocol state machine."""

    def __init__(self, protocol: str = "mcp"):
        self.protocol = protocol
        self._transitions_taken: set[str] = set()

        if protocol == "mcp":
            self.state = "pre_init"
        elif protocol == "a2a":
            self.state = "pre_send"
        else:
            self.state = "pre_init"

        if protocol == "mcp":
            self._valid_transitions = {
                "pre_init": {"initialize"},
                "initialized": {"callTool", "listTools", "shutdown"},
                "awaiting_result": {"RESULT", "ERROR"},
                "shutdown": set(),
            }
            self._next_state = {
                ("pre_init", "initialize"): "initialized",
                ("initialized", "callTool"): "awaiting_result",
                ("initialized", "listTools"): "initialized",
                ("initialized", "shutdown"): "shutdown",
                ("awaiting_result", "RESULT"): "initialized",
                ("awaiting_result", "ERROR"): "initialized",
            }
        elif protocol == "a2a":
            self._valid_transitions = {
                "pre_send": {"sendTask"},
                "awaiting_status": {"WORKING", "COMPLETED", "FAILED"},
                "working": {"getStatus", "cancel"},
                "completed": {"getArtifact"},
                "terminal": set(),
            }
            self._next_state = {
                ("pre_send", "sendTask"): "awaiting_status",
                ("awaiting_status", "WORKING"): "working",
                ("awaiting_status", "COMPLETED"): "completed",
                ("awaiting_status", "FAILED"): "terminal",
                ("working", "getStatus"): "awaiting_status",
                ("working", "cancel"): "terminal",
                ("completed", "getArtifact"): "terminal",
            }

    def is_valid(self, label: str) -> bool:
        """Check if a transition is valid from the current state."""
        valid = self._valid_transitions.get(self.state, set())
        return label in valid

    def advance(self, label: str) -> bool:
        """Advance the state machine. Returns True if transition was valid."""
        key = (self.state, label)
        if key in self._next_state:
            self.state = self._next_state[key]
            self._transitions_taken.add(label)
            return True
        return False

    def reset(self) -> None:
        """Reset to initial state."""
        if self.protocol == "mcp":
            self.state = "pre_init"
        elif self.protocol == "a2a":
            self.state = "pre_send"
        else:
            self.state = "pre_init"

    @property
    def transitions_covered(self) -> set[str]:
        return self._transitions_taken.copy()


# ---------------------------------------------------------------------------
# Conformance tester
# ---------------------------------------------------------------------------

class ConformanceTester:
    """Runtime conformance tester for MCP or A2A servers.

    Usage:
        tester = ConformanceTester("http://localhost:3000", protocol="mcp")
        report = tester.run_all()
        print(report.summary())
    """

    def __init__(
        self,
        target: str | None = None,
        protocol: str = "mcp",
        transport: HttpTransport | MockTransport | None = None,
    ):
        self.target = target or "mock://localhost"
        self.protocol = protocol
        self.transport = transport or (
            HttpTransport(target) if target else MockTransport()
        )
        self.messages = MCP_MESSAGES if protocol == "mcp" else A2A_MESSAGES

    def _make_request(self, label: str, req_id: int = 1) -> JsonRpcRequest:
        """Build a JSON-RPC request for a protocol transition label."""
        if label in self.messages:
            tmpl = self.messages[label]
            return JsonRpcRequest(
                method=tmpl["method"],
                params=tmpl.get("params", {}),
                id=req_id,
            )
        return JsonRpcRequest(method=label, id=req_id)

    def _run_path(
        self,
        name: str,
        kind: str,
        labels: list[str],
        expect_last_fails: bool = False,
    ) -> TestCase:
        """Execute a sequence of transitions and record results."""
        tracker = ProtocolStateTracker(self.protocol)
        steps: list[TestStep] = []
        all_passed = True

        for i, label in enumerate(labels):
            is_last = (i == len(labels) - 1)
            expect_success = not (is_last and expect_last_fails)

            req = self._make_request(label, req_id=i + 1)
            resp = self.transport.send(req)

            actual_success = resp.is_success
            passed = (actual_success == expect_success)

            error_msg = ""
            if not passed:
                if expect_success and not actual_success:
                    error_msg = f"Expected success but got error: {resp.error}"
                elif not expect_success and actual_success:
                    error_msg = f"Expected rejection but server accepted {label}"

            if not passed:
                all_passed = False

            steps.append(TestStep(
                label=label,
                request=req,
                response=resp,
                expected_success=expect_success,
                actual_success=actual_success,
                passed=passed,
                error_message=error_msg,
            ))

            # Advance state tracker (for valid transitions)
            if expect_success:
                tracker.advance(label)

        return TestCase(
            name=name,
            kind=kind,
            steps=tuple(steps),
            passed=all_passed,
        )

    def test_valid_path(self, path_name: str, labels: list[str]) -> TestCase:
        """Test a valid protocol execution path."""
        return self._run_path(path_name, "valid_path", labels)

    def test_violation(self, name: str, prefix: list[str], illegal: str) -> TestCase:
        """Test that an illegal transition is rejected."""
        return self._run_path(name, "violation", prefix + [illegal], expect_last_fails=True)

    def run_mcp_suite(self) -> list[TestCase]:
        """Run the standard MCP conformance test suite."""
        cases: list[TestCase] = []

        # Valid paths
        cases.append(self.test_valid_path(
            "basic_tool_call",
            ["initialize", "callTool"],
        ))
        cases.append(self.test_valid_path(
            "list_then_call",
            ["initialize", "listTools", "callTool"],
        ))
        cases.append(self.test_valid_path(
            "list_and_shutdown",
            ["initialize", "listTools", "shutdown"],
        ))
        cases.append(self.test_valid_path(
            "immediate_shutdown",
            ["initialize", "shutdown"],
        ))

        # Violations
        cases.append(self.test_violation(
            "callTool_before_init", [], "callTool",
        ))
        cases.append(self.test_violation(
            "listTools_before_init", [], "listTools",
        ))
        cases.append(self.test_violation(
            "shutdown_before_init", [], "shutdown",
        ))

        return cases

    def run_a2a_suite(self) -> list[TestCase]:
        """Run the standard A2A conformance test suite."""
        cases: list[TestCase] = []

        # Valid paths
        cases.append(self.test_valid_path(
            "send_task",
            ["sendTask"],
        ))

        # Violations
        cases.append(self.test_violation(
            "getStatus_before_send", [], "getStatus",
        ))
        cases.append(self.test_violation(
            "cancel_before_send", [], "cancel",
        ))
        cases.append(self.test_violation(
            "getArtifact_before_send", [], "getArtifact",
        ))

        return cases

    def run_all(self) -> RuntimeConformanceReport:
        """Run the full conformance suite and return a report."""
        try:
            if self.protocol == "mcp":
                cases = self.run_mcp_suite()
            elif self.protocol == "a2a":
                cases = self.run_a2a_suite()
            else:
                cases = []
        except Exception as e:
            return RuntimeConformanceReport(
                protocol=self.protocol.upper(),
                target=self.target,
                test_cases=(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                transition_coverage=0.0,
                error_message=str(e),
            )

        passed = sum(1 for c in cases if c.passed)
        failed = sum(1 for c in cases if not c.passed)
        all_labels = set()
        covered_labels = set()
        for c in cases:
            for s in c.steps:
                all_labels.add(s.label)
                if s.passed:
                    covered_labels.add(s.label)

        total_protocol_labels = len(self.messages)
        coverage = len(covered_labels) / total_protocol_labels if total_protocol_labels > 0 else 0.0

        return RuntimeConformanceReport(
            protocol=self.protocol.upper(),
            target=self.target,
            test_cases=tuple(cases),
            total_tests=len(cases),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=0,
            transition_coverage=coverage,
        )
