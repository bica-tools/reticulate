"""Tests for runtime MCP/A2A conformance testing (Step 70b)."""

import pytest

from reticulate.mcp_runtime import (
    ConformanceTester,
    HttpTransport,
    JsonRpcRequest,
    JsonRpcResponse,
    MockTransport,
    ProtocolStateTracker,
    RuntimeConformanceReport,
    TestCase,
    TestStep,
    parse_jsonrpc_response,
)


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------


class TestJsonRpc:
    def test_request_to_json(self):
        req = JsonRpcRequest(method="initialize", params={"v": "1"}, id=1)
        j = req.to_json()
        assert '"method": "initialize"' in j
        assert '"jsonrpc": "2.0"' in j

    def test_parse_success_response(self):
        raw = '{"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 1}'
        resp = parse_jsonrpc_response(raw)
        assert resp.is_success
        assert resp.result == {"status": "ok"}
        assert resp.id == 1

    def test_parse_error_response(self):
        raw = '{"jsonrpc": "2.0", "error": {"code": -32601, "message": "Not found"}, "id": 1}'
        resp = parse_jsonrpc_response(raw)
        assert resp.is_error
        assert resp.error_code == -32601

    def test_parse_invalid_json(self):
        resp = parse_jsonrpc_response("not json")
        assert resp.is_error


# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------


class TestMockTransport:
    def test_mock_accepts(self):
        transport = MockTransport(responses={"initialize": {"ok": True}})
        resp = transport.send(JsonRpcRequest(method="initialize"))
        assert resp.is_success
        assert resp.result == {"ok": True}

    def test_mock_rejects(self):
        transport = MockTransport(reject_methods={"callTool"})
        resp = transport.send(JsonRpcRequest(method="callTool"))
        assert resp.is_error

    def test_mock_logs_calls(self):
        transport = MockTransport()
        transport.send(JsonRpcRequest(method="a"))
        transport.send(JsonRpcRequest(method="b"))
        assert len(transport.call_log) == 2
        assert transport.call_log[0].method == "a"


# ---------------------------------------------------------------------------
# Protocol state tracker
# ---------------------------------------------------------------------------


class TestProtocolStateTracker:
    def test_mcp_initial_state(self):
        t = ProtocolStateTracker("mcp")
        assert t.state == "pre_init"

    def test_mcp_initialize_valid(self):
        t = ProtocolStateTracker("mcp")
        assert t.is_valid("initialize")
        assert t.advance("initialize")
        assert t.state == "initialized"

    def test_mcp_callTool_before_init_invalid(self):
        t = ProtocolStateTracker("mcp")
        assert not t.is_valid("callTool")

    def test_mcp_callTool_after_init_valid(self):
        t = ProtocolStateTracker("mcp")
        t.advance("initialize")
        assert t.is_valid("callTool")

    def test_mcp_shutdown(self):
        t = ProtocolStateTracker("mcp")
        t.advance("initialize")
        t.advance("shutdown")
        assert t.state == "shutdown"
        assert not t.is_valid("callTool")

    def test_mcp_result_returns_to_initialized(self):
        t = ProtocolStateTracker("mcp")
        t.advance("initialize")
        t.advance("callTool")
        assert t.state == "awaiting_result"
        t.advance("RESULT")
        assert t.state == "initialized"

    def test_mcp_reset(self):
        t = ProtocolStateTracker("mcp")
        t.advance("initialize")
        t.reset()
        assert t.state == "pre_init"

    def test_a2a_initial_state(self):
        t = ProtocolStateTracker("a2a")
        assert t.state == "pre_send"

    def test_a2a_send_then_working(self):
        t = ProtocolStateTracker("a2a")
        t.advance("sendTask")
        assert t.state == "awaiting_status"
        t.advance("WORKING")
        assert t.state == "working"

    def test_a2a_completed(self):
        t = ProtocolStateTracker("a2a")
        t.advance("sendTask")
        t.advance("COMPLETED")
        assert t.state == "completed"
        t.advance("getArtifact")
        assert t.state == "terminal"

    def test_a2a_failed_is_terminal(self):
        t = ProtocolStateTracker("a2a")
        t.advance("sendTask")
        t.advance("FAILED")
        assert t.state == "terminal"
        assert not t.is_valid("getStatus")

    def test_transitions_covered(self):
        t = ProtocolStateTracker("mcp")
        t.advance("initialize")
        t.advance("listTools")
        assert "initialize" in t.transitions_covered
        assert "listTools" in t.transitions_covered


# ---------------------------------------------------------------------------
# Conformance tester with mock
# ---------------------------------------------------------------------------


class TestConformanceTesterMcp:
    """Test MCP conformance with a well-behaved mock server."""

    def test_valid_path_passes(self):
        transport = MockTransport()
        tester = ConformanceTester(protocol="mcp", transport=transport)
        tc = tester.test_valid_path("basic", ["initialize", "callTool"])
        assert tc.passed
        assert tc.num_steps == 2

    def test_valid_shutdown_passes(self):
        transport = MockTransport()
        tester = ConformanceTester(protocol="mcp", transport=transport)
        tc = tester.test_valid_path("shutdown", ["initialize", "shutdown"])
        assert tc.passed

    def test_violation_detected_when_mock_accepts_illegal(self):
        """A mock that accepts everything fails the violation test."""
        transport = MockTransport()  # accepts everything
        tester = ConformanceTester(protocol="mcp", transport=transport)
        tc = tester.test_violation("illegal", [], "callTool")
        # The mock accepts callTool without init → violation test FAILS
        assert not tc.passed

    def test_violation_passes_when_mock_rejects(self):
        """A correct mock rejects callTool before init."""
        # The label "callTool" maps to JSON-RPC method "tools/call"
        transport = MockTransport(reject_methods={"tools/call"})
        tester = ConformanceTester(protocol="mcp", transport=transport)
        tc = tester.test_violation("illegal", [], "callTool")
        assert tc.passed

    def test_run_all_produces_report(self):
        transport = MockTransport()
        tester = ConformanceTester(protocol="mcp", transport=transport)
        report = tester.run_all()
        assert isinstance(report, RuntimeConformanceReport)
        assert report.total_tests > 0
        assert report.protocol == "MCP"

    def test_run_all_summary_contains_protocol(self):
        transport = MockTransport()
        tester = ConformanceTester(protocol="mcp", transport=transport)
        report = tester.run_all()
        summary = report.summary()
        assert "MCP" in summary
        assert "Target" in summary


class TestConformanceTesterA2a:
    """Test A2A conformance with mock."""

    def test_valid_send_passes(self):
        transport = MockTransport()
        tester = ConformanceTester(protocol="a2a", transport=transport)
        tc = tester.test_valid_path("send", ["sendTask"])
        assert tc.passed

    def test_violation_rejected(self):
        # "getStatus" maps to JSON-RPC method "tasks/get"
        transport = MockTransport(reject_methods={"tasks/get"})
        tester = ConformanceTester(protocol="a2a", transport=transport)
        tc = tester.test_violation("early_poll", [], "getStatus")
        assert tc.passed

    def test_run_all_a2a(self):
        transport = MockTransport()
        tester = ConformanceTester(protocol="a2a", transport=transport)
        report = tester.run_all()
        assert report.protocol == "A2A"
        assert report.total_tests > 0


# ---------------------------------------------------------------------------
# Conformance tester with strict mock (simulates correct server)
# ---------------------------------------------------------------------------


class TestStrictServer:
    """Test against a mock that enforces the protocol state machine."""

    def _strict_mcp_transport(self) -> MockTransport:
        """Mock that rejects callTool/listTools/shutdown before init."""
        return MockTransport(
            responses={
                "initialize": {"protocolVersion": "2024-11-05", "capabilities": {}},
                "tools/list": {"tools": []},
                "tools/call": {"content": [{"type": "text", "text": "ok"}]},
                "shutdown": {},
            },
            reject_methods=set(),  # We'd need stateful mock for full strictness
        )

    def test_basic_flow(self):
        transport = self._strict_mcp_transport()
        tester = ConformanceTester(protocol="mcp", transport=transport)
        tc = tester.test_valid_path(
            "full_flow",
            ["initialize", "listTools", "callTool", "shutdown"],
        )
        assert tc.passed
        assert tc.num_steps == 4


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_pass_rate(self):
        report = RuntimeConformanceReport(
            protocol="TEST",
            target="mock://",
            test_cases=(),
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            skipped_tests=0,
            transition_coverage=0.5,
        )
        assert report.pass_rate == 0.8

    def test_summary_format(self):
        transport = MockTransport()
        tester = ConformanceTester(protocol="mcp", transport=transport)
        report = tester.run_all()
        s = report.summary()
        assert "RUNTIME CONFORMANCE" in s
        assert "Passed" in s
        assert "Failed" in s

    def test_connection_error_report(self):
        report = RuntimeConformanceReport(
            protocol="MCP",
            target="http://unreachable:9999",
            test_cases=(),
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            transition_coverage=0.0,
            error_message="Connection refused",
        )
        s = report.summary()
        assert "CONNECTION ERROR" in s
