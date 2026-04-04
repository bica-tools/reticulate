"""Tests for conditional selection detection (Step 80f enhancement).

Validates that if/try/switch conditionals on return values are
correctly detected as Select nodes in directed traces.
"""

import os
import pytest

from reticulate.selection_detector import (
    extract_directed_traces,
    compute_selection_stats,
    DirectedStep,
    DirectedTrace,
    SelectionStats,
    HAS_JAVALANG,
)
from reticulate.universe import (
    _scan_java_files,
    _discover_classes,
    _preload_jdk_types,
)

JEDIS_SRC = "/tmp/jedis-test/src/main/java"

skip_no_javalang = pytest.mark.skipif(not HAS_JAVALANG, reason="javalang not installed")
skip_no_jedis = pytest.mark.skipif(not os.path.isdir(JEDIS_SRC), reason="Jedis not cloned")


# ---------------------------------------------------------------------------
# Unit tests on synthetic Java
# ---------------------------------------------------------------------------

@skip_no_javalang
class TestSyntheticDetection:
    """Test selection detection on synthetic Java snippets."""

    def _parse_snippet(self, java_source: str):
        import javalang
        tree = javalang.parse.parse(java_source)
        parsed = [("<synthetic>", tree)]
        classes = _discover_classes(parsed)
        jdk = _preload_jdk_types()
        return extract_directed_traces(parsed, classes, jdk)

    def test_if_boolean_method(self):
        """if(it.hasNext()) should produce hasNext(r) → TRUE(s)."""
        src = '''
class Client {
    void test(Iterator it) {
        if (it.hasNext()) {
            it.next();
        }
    }
}
'''
        directed = self._parse_snippet(src)
        assert ("Client", "Iterator") in directed
        traces = directed[("Client", "Iterator")]
        assert len(traces) >= 1
        steps = traces[0].steps
        # Should have hasNext(r) → TRUE(s) → next(r)
        labels = [(s.label, s.direction) for s in steps]
        assert ("hasNext", "r") in labels
        assert ("TRUE", "s") in labels

    def test_try_catch_detection(self):
        """try { conn.connect() } catch should detect the call."""
        src = '''
class Client {
    void test(Connection conn) {
        try {
            conn.connect();
            conn.sendCommand();
        } catch (Exception e) {
            conn.close();
        }
    }
}
class Connection {
    public void connect() {}
    public void sendCommand() {}
    public void close() {}
}
'''
        directed = self._parse_snippet(src)
        assert ("Client", "Connection") in directed
        traces = directed[("Client", "Connection")]
        assert len(traces) >= 1
        methods = [s.label for s in traces[0].steps if s.direction == "r"]
        assert "connect" in methods
        assert "sendCommand" in methods

    def test_assignment_captures_value(self):
        """Object x = obj.get() should produce get(r) → VALUE(s)."""
        src = '''
class Client {
    void test(Builder b) {
        Object x = b.build();
    }
}
class Builder {
    public Object build() { return null; }
}
'''
        directed = self._parse_snippet(src)
        assert ("Client", "Builder") in directed
        traces = directed[("Client", "Builder")]
        steps = traces[0].steps
        labels = [(s.label, s.direction) for s in steps]
        assert ("build", "r") in labels
        assert ("VALUE", "s") in labels

    def test_plain_call_no_selection(self):
        """obj.doSomething() with no conditional → Branch only, no Select."""
        src = '''
class Client {
    void test(Service svc) {
        svc.process();
        svc.finish();
    }
}
class Service {
    public void process() {}
    public void finish() {}
}
'''
        directed = self._parse_snippet(src)
        assert ("Client", "Service") in directed
        traces = directed[("Client", "Service")]
        steps = traces[0].steps
        # All should be "r" (Branch), no "s" (Select)
        assert all(s.direction == "r" for s in steps)
        assert len(steps) == 2

    def test_while_loop_with_condition(self):
        """while(it.hasNext()) should detect selection."""
        src = '''
class Client {
    void test(Iterator it) {
        while (it.hasNext()) {
            it.next();
        }
    }
}
'''
        directed = self._parse_snippet(src)
        assert ("Client", "Iterator") in directed
        traces = directed[("Client", "Iterator")]
        steps = traces[0].steps
        labels = [(s.label, s.direction) for s in steps]
        assert ("hasNext", "r") in labels
        assert ("TRUE", "s") in labels


# ---------------------------------------------------------------------------
# Jedis integration tests
# ---------------------------------------------------------------------------

@skip_no_javalang
@skip_no_jedis
class TestJedisSelections:
    @pytest.fixture(scope="class")
    def directed(self):
        parsed = _scan_java_files(JEDIS_SRC)
        classes = _discover_classes(parsed)
        jdk = _preload_jdk_types()
        return extract_directed_traces(parsed, classes, jdk)

    @pytest.fixture(scope="class")
    def stats(self, directed):
        return compute_selection_stats(directed)

    def test_finds_selections(self, stats):
        assert stats.traces_with_selections >= 50

    def test_selection_rate_positive(self, stats):
        assert stats.selection_rate > 0.0

    def test_has_branch_and_select_steps(self, stats):
        assert stats.branch_steps > 0
        assert stats.select_steps > 0

    def test_iterator_hasNext_detected(self, directed):
        """Iterator.hasNext() in conditionals should produce TRUE selection."""
        iter_traces = []
        for (caller, callee), traces in directed.items():
            if callee == "Iterator":
                for dt in traces:
                    if any(s.label == "TRUE" and s.direction == "s" for s in dt.steps):
                        iter_traces.append(dt)
        assert len(iter_traces) >= 1, "No Iterator hasNext→TRUE selections found"

    def test_many_callee_pairs(self, directed):
        assert len(directed) >= 50

    def test_total_traces(self, stats):
        assert stats.total_traces >= 500


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

class TestSelectionStats:
    def test_empty_stats(self):
        stats = compute_selection_stats({})
        assert stats.total_traces == 0
        assert stats.selection_rate == 0.0

    def test_stats_from_manual(self):
        traces = {
            ("A", "B"): [
                DirectedTrace("A", "test", "B", [
                    DirectedStep("method", "r"),
                    DirectedStep("TRUE", "s"),
                ]),
                DirectedTrace("A", "other", "B", [
                    DirectedStep("call", "r"),
                ]),
            ]
        }
        stats = compute_selection_stats(traces)
        assert stats.total_traces == 2
        assert stats.traces_with_selections == 1
        assert stats.branch_steps == 2
        assert stats.select_steps == 1
        assert stats.selection_rate == 0.5
