"""Tests for truncate_back_edges (Step 13a: Drawing Recursive Lattices Without Collapse)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import build_statespace, truncate_back_edges
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_and_truncate(type_str: str):
    """Parse, build state space, truncate, return (original_ss, truncated_ss, back_edges)."""
    ast = parse(type_str)
    ss = build_statespace(ast)
    truncated, back_edges = truncate_back_edges(ss)
    return ss, truncated, back_edges


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestBasicProperties:
    """Truncated state spaces preserve fundamental invariants."""

    def test_end_has_no_back_edges(self):
        """Non-recursive type has no back-edges."""
        _, truncated, back_edges = _build_and_truncate("end")
        assert back_edges == []

    def test_branch_has_no_back_edges(self):
        """Pure branch (no recursion) has no back-edges."""
        _, truncated, back_edges = _build_and_truncate("&{a: end, b: end}")
        assert back_edges == []

    def test_select_has_no_back_edges(self):
        """Pure selection has no back-edges."""
        _, truncated, back_edges = _build_and_truncate("+{OK: end, ERR: end}")
        assert back_edges == []

    def test_parallel_no_recursion_has_no_back_edges(self):
        """Parallel without recursion has no back-edges."""
        _, truncated, back_edges = _build_and_truncate(
            "(&{a: end} || &{b: end})"
        )
        assert back_edges == []

    def test_simple_recursion_has_back_edge(self):
        """Simple rec produces at least one back-edge."""
        _, truncated, back_edges = _build_and_truncate(
            "rec X . &{a: X, b: end}"
        )
        assert len(back_edges) >= 1

    def test_truncated_preserves_top(self):
        """Truncated state space has the same top."""
        ss, truncated, _ = _build_and_truncate("rec X . &{a: X, b: end}")
        assert truncated.top == ss.top

    def test_truncated_preserves_bottom(self):
        """Truncated state space has the same bottom."""
        ss, truncated, _ = _build_and_truncate("rec X . &{a: X, b: end}")
        assert truncated.bottom == ss.bottom

    def test_truncated_is_acyclic(self):
        """Truncated state space has no cycles (back-edges removed)."""
        _, truncated, _ = _build_and_truncate("rec X . &{a: X, b: end}")
        # DFS to check for cycles
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {s: WHITE for s in truncated.states}
        adj = {s: [] for s in truncated.states}
        for src, _, tgt in truncated.transitions:
            adj[src].append(tgt)

        def has_cycle(start):
            stack = [(start, False)]
            while stack:
                node, done = stack.pop()
                if done:
                    color[node] = BLACK
                    continue
                if color[node] == GRAY:
                    return True
                if color[node] == BLACK:
                    continue
                color[node] = GRAY
                stack.append((node, True))
                for tgt in adj[node]:
                    if color[tgt] == GRAY:
                        return True
                    if color[tgt] == WHITE:
                        stack.append((tgt, False))
            return False

        assert not has_cycle(truncated.top)

    def test_truncated_no_more_states_than_original(self):
        """Truncated has at most as many states as original."""
        ss, truncated, _ = _build_and_truncate(
            "rec X . &{a: X, b: end}"
        )
        assert len(truncated.states) <= len(ss.states)


# ---------------------------------------------------------------------------
# Lattice property of truncated state space
# ---------------------------------------------------------------------------

class TestTruncatedIsLattice:
    """The truncated single unfolding forms a lattice (Theorem 2 of Step 13a)."""

    def test_simple_rec_is_lattice(self):
        _, truncated, _ = _build_and_truncate("rec X . &{a: X, b: end}")
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_iterator_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_smtp_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "&{connect: &{ehlo: rec X . &{mail: &{rcpt: &{data: "
            "+{OK: X, ERR: X}}}, quit: end}}}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_oauth_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "&{requestAuth: +{GRANTED: &{getToken: +{TOKEN: "
            "rec X . &{useToken: X, refreshToken: +{OK: X, EXPIRED: end}, "
            "revoke: end}, ERROR: end}}, DENIED: end}}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_raft_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "rec X . +{TIMEOUT: &{requestVote: +{ELECTED: "
            "rec Y . &{appendEntries: +{ACK: Y, NACK: Y}, "
            "heartbeatTimeout: Y, stepDown: X}, REJECTED: X}}, "
            "HEARTBEAT: X, SHUTDOWN: end}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_circuit_breaker_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "rec X . &{call: +{SUCCESS: X, FAILURE: +{TRIPPED: "
            "rec Y . &{probe: +{OK: X, FAIL: Y}, timeout: end}, "
            "OK: X}}, reset: end}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_dns_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "rec X . &{query: +{ANSWER: X, NXDOMAIN: X, SERVFAIL: X, "
            "TIMEOUT: &{retry: X, abandon: end}}, close: end}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_rate_limiter_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "rec X . &{tryAcquire: +{ALLOWED: X, "
            "THROTTLED: &{wait_retry: X, abort: end}}, close: end}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_enzyme_mm_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "rec X . &{bind_substrate: +{CATALYZE: &{release_product: X}, "
            "DISSOCIATE: X}, shutdown: end}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice

    def test_failover_is_lattice(self):
        _, truncated, _ = _build_and_truncate(
            "&{connect: rec X . &{request: +{OK: X, FAIL: "
            "&{reconnect: +{UP: X, DOWN: end}}}, close: end}}"
        )
        result = check_lattice(truncated)
        assert result.is_lattice


# ---------------------------------------------------------------------------
# Back-edge identification
# ---------------------------------------------------------------------------

class TestBackEdgeIdentification:
    """Back-edges are correctly identified."""

    def test_simple_rec_back_edge_label(self):
        """The back-edge in rec X . &{a: X, b: end} is on label 'a'."""
        _, _, back_edges = _build_and_truncate("rec X . &{a: X, b: end}")
        labels = [lbl for _, lbl, _ in back_edges]
        assert "a" in labels

    def test_nested_rec_multiple_back_edges(self):
        """Nested recursion produces back-edges for each cycle."""
        _, _, back_edges = _build_and_truncate(
            "rec X . +{TIMEOUT: &{vote: +{ELECTED: "
            "rec Y . &{append: +{ACK: Y, NACK: Y}, step: X}, "
            "REJECTED: X}}, HB: X, SHUT: end}"
        )
        # Should have multiple back-edges (for X and Y references)
        assert len(back_edges) >= 2

    def test_non_recursive_no_back_edges(self):
        """Complex non-recursive type has zero back-edges."""
        _, _, back_edges = _build_and_truncate(
            "&{a: +{OK: &{c: end, d: end}, ERR: end}, b: end}"
        )
        assert len(back_edges) == 0


# ---------------------------------------------------------------------------
# Size properties
# ---------------------------------------------------------------------------

class TestSizeProperties:
    """Truncated state space size relationships."""

    def test_same_states_simple_rec(self):
        """Simple recursion: truncated has same number of states."""
        ss, truncated, _ = _build_and_truncate("rec X . &{a: X, b: end}")
        assert len(truncated.states) == len(ss.states)

    def test_same_transition_count(self):
        """Truncation redirects edges, doesn't add or remove them."""
        ss, truncated, _ = _build_and_truncate("rec X . &{a: X, b: end}")
        assert len(truncated.transitions) == len(ss.transitions)

    def test_iterator_same_size(self):
        ss, truncated, _ = _build_and_truncate(
            "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}"
        )
        assert len(truncated.states) == len(ss.states)

    def test_truncated_more_nodes_than_quotient(self):
        """Truncated has at least as many nodes as the SCC quotient."""
        ss, truncated, _ = _build_and_truncate(
            "rec X . &{a: X, b: end}"
        )
        lattice_result = check_lattice(ss)
        quotient_size = lattice_result.num_scc
        assert len(truncated.states) >= quotient_size


# ---------------------------------------------------------------------------
# Selection transitions preserved
# ---------------------------------------------------------------------------

class TestSelectionPreservation:
    """Selection transitions are preserved in truncation."""

    def test_select_back_edge_stays_selection(self):
        """If a back-edge was a selection, the redirected edge is also selection."""
        ss, truncated, back_edges = _build_and_truncate(
            "rec X . +{OK: X, ERR: end}"
        )
        # The OK transition is a selection and a back-edge
        ok_back = [b for b in back_edges if b[1] == "OK"]
        assert len(ok_back) == 1
        # In truncated, the redirected (src, "OK", bottom) should be selection
        redirected = [(s, l, t) for s, l, t in truncated.transitions
                      if l == "OK" and t == truncated.bottom]
        assert len(redirected) >= 1
        for r in redirected:
            assert r in truncated.selection_transitions

    def test_branch_back_edge_not_selection(self):
        """Branch back-edges remain non-selection after redirect."""
        _, truncated, back_edges = _build_and_truncate(
            "rec X . &{a: X, b: end}"
        )
        a_redirected = [(s, l, t) for s, l, t in truncated.transitions
                        if l == "a" and t == truncated.bottom]
        for r in a_redirected:
            assert r not in truncated.selection_transitions


# ---------------------------------------------------------------------------
# Benchmark integration: all recursive benchmarks truncate to lattices
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    """All recursive benchmarks from the catalogue truncate to lattices."""

    @pytest.fixture
    def recursive_benchmarks(self):
        """Load benchmark protocols that use recursion."""
        try:
            from tests.benchmarks.protocols import BENCHMARKS
        except ImportError:
            pytest.skip("benchmarks not available")
        return [
            b for b in BENCHMARKS
            if "rec" in b.type_string.lower() or "rec" in b.type_string
        ]

    def test_all_recursive_benchmarks_truncate_to_lattice(self, recursive_benchmarks):
        """Every recursive benchmark produces a lattice after truncation."""
        failures = []
        for b in recursive_benchmarks:
            ast = parse(b.type_string)
            ss = build_statespace(ast)
            truncated, back_edges = truncate_back_edges(ss)
            result = check_lattice(truncated)
            if not result.is_lattice:
                failures.append(f"{b.name}: not a lattice after truncation")
        assert not failures, f"Failures:\n" + "\n".join(failures)

    def test_truncated_size_bounded(self, recursive_benchmarks):
        """Truncated state space never exceeds original size."""
        for b in recursive_benchmarks:
            ast = parse(b.type_string)
            ss = build_statespace(ast)
            truncated, _ = truncate_back_edges(ss)
            assert len(truncated.states) <= len(ss.states), (
                f"{b.name}: truncated ({len(truncated.states)}) > "
                f"original ({len(ss.states)})"
            )
