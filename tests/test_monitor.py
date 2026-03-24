"""Tests for runtime monitor generation (Step 80)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.monitor import (
    MonitorConfig,
    MonitorTemplate,
    MonitoredProxy,
    ProtocolViolationError,
    SessionMonitor,
    build_transition_table,
    generate_express_middleware,
    generate_java_monitor,
    generate_python_monitor,
)


# ---------------------------------------------------------------------------
# SessionMonitor: basic construction and transitions
# ---------------------------------------------------------------------------

class TestSessionMonitorBasic:
    def test_from_session_type(self):
        ast = parse("&{a: end}")
        monitor = SessionMonitor.from_session_type(ast)
        assert monitor.current_state == monitor.initial_state
        assert not monitor.is_terminal

    def test_from_statespace(self):
        ast = parse("&{a: end}")
        ss = build_statespace(ast)
        monitor = SessionMonitor.from_statespace(ss)
        assert monitor.current_state == ss.top

    def test_simple_transition(self):
        ast = parse("&{a: end}")
        monitor = SessionMonitor.from_session_type(ast)
        new_state = monitor.transition("a")
        assert new_state == monitor.terminal_state
        assert monitor.is_terminal

    def test_two_step_protocol(self):
        ast = parse("&{open: &{close: end}}")
        monitor = SessionMonitor.from_session_type(ast)
        monitor.transition("open")
        assert not monitor.is_terminal
        monitor.transition("close")
        assert monitor.is_terminal

    def test_branching_protocol(self):
        ast = parse("&{a: end, b: end}")
        monitor = SessionMonitor.from_session_type(ast)
        assert set(monitor.allowed_methods) == {"a", "b"}
        monitor.transition("a")
        assert monitor.is_terminal

    def test_branch_then_select(self):
        ast = parse("&{init: +{ok: end, err: end}}")
        monitor = SessionMonitor.from_session_type(ast)
        monitor.transition("init")
        assert set(monitor.allowed_methods) == {"ok", "err"}
        monitor.transition("ok")
        assert monitor.is_terminal


# ---------------------------------------------------------------------------
# Protocol violations
# ---------------------------------------------------------------------------

class TestProtocolViolations:
    def test_violation_raises(self):
        ast = parse("&{a: end}")
        monitor = SessionMonitor.from_session_type(ast)
        with pytest.raises(ProtocolViolationError) as exc_info:
            monitor.transition("b")
        assert exc_info.value.method == "b"
        assert "a" in exc_info.value.allowed

    def test_violation_after_terminal(self):
        ast = parse("&{a: end}")
        monitor = SessionMonitor.from_session_type(ast)
        monitor.transition("a")
        with pytest.raises(ProtocolViolationError):
            monitor.transition("a")

    def test_violation_wrong_order(self):
        ast = parse("&{open: &{close: end}}")
        monitor = SessionMonitor.from_session_type(ast)
        with pytest.raises(ProtocolViolationError):
            monitor.transition("close")

    def test_violation_log_mode(self, capsys):
        ast = parse("&{a: end}")
        config = MonitorConfig(strict_mode=False, on_violation="log")
        monitor = SessionMonitor.from_session_type(ast, config=config)
        result = monitor.transition("b")
        # In log mode, state should not change
        assert result == monitor.initial_state
        assert not monitor.is_terminal

    def test_violation_ignore_mode(self):
        ast = parse("&{a: end}")
        config = MonitorConfig(strict_mode=False, on_violation="ignore")
        monitor = SessionMonitor.from_session_type(ast, config=config)
        result = monitor.transition("b")
        assert result == monitor.initial_state


# ---------------------------------------------------------------------------
# History and reset
# ---------------------------------------------------------------------------

class TestHistoryAndReset:
    def test_history_recorded(self):
        ast = parse("&{a: &{b: end}}")
        monitor = SessionMonitor.from_session_type(ast)
        monitor.transition("a")
        monitor.transition("b")
        assert len(monitor.history) == 2
        assert monitor.history[0][0] == "a"
        assert monitor.history[1][0] == "b"

    def test_reset(self):
        ast = parse("&{a: end}")
        monitor = SessionMonitor.from_session_type(ast)
        monitor.transition("a")
        assert monitor.is_terminal
        monitor.reset()
        assert not monitor.is_terminal
        assert monitor.current_state == monitor.initial_state
        assert len(monitor.history) == 0

    def test_is_allowed(self):
        ast = parse("&{a: end, b: end}")
        monitor = SessionMonitor.from_session_type(ast)
        assert monitor.is_allowed("a")
        assert monitor.is_allowed("b")
        assert not monitor.is_allowed("c")

    def test_check_complete(self):
        ast = parse("&{a: end}")
        monitor = SessionMonitor.from_session_type(ast)
        assert not monitor.check_complete()
        monitor.transition("a")
        assert monitor.check_complete()


# ---------------------------------------------------------------------------
# Recursive types
# ---------------------------------------------------------------------------

class TestRecursiveProtocol:
    def test_recursive_protocol(self):
        ast = parse("rec X . &{next: X, done: end}")
        monitor = SessionMonitor.from_session_type(ast)
        monitor.transition("next")
        assert not monitor.is_terminal
        monitor.transition("next")
        assert not monitor.is_terminal
        monitor.transition("done")
        assert monitor.is_terminal

    def test_recursive_selections(self):
        ast = parse("rec X . &{call: +{OK: X, ERR: end}}")
        monitor = SessionMonitor.from_session_type(ast)
        monitor.transition("call")
        monitor.transition("OK")
        assert not monitor.is_terminal
        monitor.transition("call")
        monitor.transition("ERR")
        assert monitor.is_terminal


# ---------------------------------------------------------------------------
# MonitoredProxy
# ---------------------------------------------------------------------------

class TestMonitoredProxy:
    def test_proxy_delegates(self):
        class FileHandle:
            def __init__(self):
                self.data = []
            def open(self):
                self.data.append("opened")
            def close(self):
                self.data.append("closed")

        ast = parse("&{open: &{close: end}}")
        monitor = SessionMonitor.from_session_type(ast)
        proxy = monitor.wrap(FileHandle())
        proxy.open()
        proxy.close()
        target = object.__getattribute__(proxy, "_target")
        assert target.data == ["opened", "closed"]

    def test_proxy_violation(self):
        class FileHandle:
            def open(self): pass
            def close(self): pass

        ast = parse("&{open: &{close: end}}")
        monitor = SessionMonitor.from_session_type(ast)
        proxy = monitor.wrap(FileHandle())
        with pytest.raises(ProtocolViolationError):
            proxy.close()  # not allowed before open


# ---------------------------------------------------------------------------
# Python monitor generation
# ---------------------------------------------------------------------------

class TestPythonMonitorGeneration:
    def test_generates_source(self):
        ast = parse("&{a: end}")
        tmpl = generate_python_monitor(ast, "TestMonitor")
        assert isinstance(tmpl, MonitorTemplate)
        assert tmpl.language == "python"
        assert "class TestMonitor" in tmpl.source_code
        assert "ProtocolViolationError" in tmpl.source_code

    def test_generated_source_is_executable(self):
        ast = parse("&{a: &{b: end}}")
        tmpl = generate_python_monitor(ast, "Proto")
        # Execute the generated source
        ns: dict = {}
        exec(tmpl.source_code, ns)
        Monitor = ns["Proto"]
        m = Monitor()
        m.call("a")
        assert not m.is_terminal
        m.call("b")
        assert m.is_terminal

    def test_generated_source_raises_on_violation(self):
        ast = parse("&{a: end}")
        tmpl = generate_python_monitor(ast, "StrictMon")
        ns: dict = {}
        exec(tmpl.source_code, ns)
        Monitor = ns["StrictMon"]
        m = Monitor()
        with pytest.raises(ns["ProtocolViolationError"]):
            m.call("b")

    def test_generated_monitor_has_allowed(self):
        ast = parse("&{x: end, y: end}")
        tmpl = generate_python_monitor(ast, "AllowedMon")
        ns: dict = {}
        exec(tmpl.source_code, ns)
        m = ns["AllowedMon"]()
        assert set(m.allowed) == {"x", "y"}

    def test_log_mode_generation(self):
        ast = parse("&{a: end}")
        config = MonitorConfig(on_violation="log")
        tmpl = generate_python_monitor(ast, "LogMon", config=config)
        assert "WARNING" in tmpl.source_code

    def test_ignore_mode_generation(self):
        ast = parse("&{a: end}")
        config = MonitorConfig(on_violation="ignore")
        tmpl = generate_python_monitor(ast, "IgnoreMon", config=config)
        assert "violation ignored" in tmpl.source_code

    def test_template_metadata(self):
        ast = parse("&{a: end, b: end}")
        tmpl = generate_python_monitor(ast, "MetaMon")
        assert tmpl.class_name == "MetaMon"
        assert tmpl.initial_state != tmpl.terminal_state
        assert len(tmpl.transition_table) == 2
        assert len(tmpl.state_machine) >= 2


# ---------------------------------------------------------------------------
# Java monitor generation
# ---------------------------------------------------------------------------

class TestJavaMonitorGeneration:
    def test_generates_java_source(self):
        ast = parse("&{init: &{run: end}}")
        tmpl = generate_java_monitor(ast, "ServerMonitor")
        assert tmpl.language == "java"
        assert "public class ServerMonitor" in tmpl.source_code
        assert "ProtocolViolationException" in tmpl.source_code

    def test_java_has_transitions(self):
        ast = parse("&{a: end, b: end}")
        tmpl = generate_java_monitor(ast, "JavaMon")
        assert "computeIfAbsent" in tmpl.source_code
        assert '"a"' in tmpl.source_code
        assert '"b"' in tmpl.source_code

    def test_java_package(self):
        ast = parse("&{a: end}")
        tmpl = generate_java_monitor(ast, "PkgMon", package_name="org.example")
        assert "package org.example;" in tmpl.source_code


# ---------------------------------------------------------------------------
# Express middleware generation
# ---------------------------------------------------------------------------

class TestExpressMiddlewareGeneration:
    def test_generates_express_source(self):
        ast = parse("&{create: &{read: end, delete: end}}")
        tmpl = generate_express_middleware(ast)
        assert tmpl.language == "express"
        assert "protocolMiddleware" in tmpl.source_code
        assert "module.exports" in tmpl.source_code
        assert "409" in tmpl.source_code

    def test_route_map(self):
        ast = parse("&{create: end}")
        tmpl = generate_express_middleware(
            ast, route_map={"create": "POST /items"}
        )
        assert "POST /items" in tmpl.source_code

    def test_express_session_tracking(self):
        ast = parse("&{a: end}")
        tmpl = generate_express_middleware(ast)
        assert "x-session-id" in tmpl.source_code
        assert "resetSession" in tmpl.source_code


# ---------------------------------------------------------------------------
# build_transition_table convenience function
# ---------------------------------------------------------------------------

class TestBuildTransitionTable:
    def test_returns_table(self):
        ast = parse("&{a: end, b: end}")
        sm, top, bottom = build_transition_table(ast)
        assert top != bottom
        assert len(sm[top]) == 2
        labels = {lbl for lbl, _ in sm[top]}
        assert labels == {"a", "b"}


# ---------------------------------------------------------------------------
# Selection-aware monitoring
# ---------------------------------------------------------------------------

class TestSelectionAwareness:
    def test_allowed_selections(self):
        ast = parse("+{ok: end, err: end}")
        monitor = SessionMonitor.from_session_type(ast)
        # Selection labels should be in allowed_selections
        selections = monitor.allowed_selections
        assert set(selections) == {"ok", "err"}

    def test_branch_has_no_selections(self):
        ast = parse("&{a: end, b: end}")
        monitor = SessionMonitor.from_session_type(ast)
        assert monitor.allowed_selections == []
        assert set(monitor.allowed_methods) == {"a", "b"}
