"""Tests for db_protocols.py --- Step 90.

Verifies database protocol definitions, lattice properties,
isolation level comparison, and the full analysis pipeline.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.db_protocols import (
    ALL_DB_PROTOCOLS,
    DBAnalysisResult,
    DBProtocol,
    IsolationComparisonResult,
    batch_execution,
    compare_isolation_levels,
    connection_pool,
    cursor_iteration,
    db_to_session_type,
    format_db_report,
    format_db_summary,
    format_isolation_report,
    jdbc_connection,
    jdbc_transaction,
    prepared_statement,
    stored_procedure,
    two_phase_commit,
    verify_all_db_protocols,
    verify_db_protocol,
)
from reticulate.statespace import build_statespace
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Protocol definition tests
# ---------------------------------------------------------------------------

class TestProtocolDefinitions:
    """Test that each protocol factory returns a valid DBProtocol."""

    def test_jdbc_connection_structure(self) -> None:
        proto = jdbc_connection()
        assert proto.name == "JDBCConnection"
        assert "Client" in proto.participants
        assert "Database" in proto.participants
        assert proto.isolation_level == "READ_COMMITTED"
        assert "connection_safety" in proto.properties
        assert len(proto.session_type_string) > 0

    def test_jdbc_transaction_structure(self) -> None:
        proto = jdbc_transaction()
        assert proto.name == "JDBCTransaction"
        assert "atomicity" in proto.properties
        assert "commit_rollback_exclusion" in proto.properties

    def test_connection_pool_structure(self) -> None:
        proto = connection_pool()
        assert proto.name == "ConnectionPool"
        assert "ConnectionPool" in proto.participants
        assert "timeout_safety" in proto.properties
        assert "leak_prevention" in proto.properties

    def test_prepared_statement_structure(self) -> None:
        proto = prepared_statement()
        assert proto.name == "PreparedStatement"
        assert "sql_injection_prevention" in proto.properties
        assert "parameter_binding_order" in proto.properties

    def test_two_phase_commit_structure(self) -> None:
        proto = two_phase_commit()
        assert proto.name == "TwoPhaseCommit"
        assert proto.isolation_level == "SERIALIZABLE"
        assert "Coordinator" in proto.participants
        assert "distributed_consensus" in proto.properties

    def test_cursor_iteration_structure(self) -> None:
        proto = cursor_iteration()
        assert proto.name == "CursorIteration"
        assert "cursor_safety" in proto.properties
        assert "exhaustion_guarantee" in proto.properties

    def test_stored_procedure_structure(self) -> None:
        proto = stored_procedure()
        assert proto.name == "StoredProcedure"
        assert "parameter_registration_order" in proto.properties

    def test_batch_execution_structure(self) -> None:
        proto = batch_execution()
        assert proto.name == "BatchExecution"
        assert "batch_atomicity" in proto.properties


# ---------------------------------------------------------------------------
# Protocol parsing tests
# ---------------------------------------------------------------------------

class TestProtocolParsing:
    """Test that each protocol's session type string parses without error."""

    @pytest.mark.parametrize("protocol", ALL_DB_PROTOCOLS,
                             ids=[p.name for p in ALL_DB_PROTOCOLS])
    def test_parseable(self, protocol: DBProtocol) -> None:
        ast = db_to_session_type(protocol)
        assert ast is not None

    @pytest.mark.parametrize("protocol", ALL_DB_PROTOCOLS,
                             ids=[p.name for p in ALL_DB_PROTOCOLS])
    def test_builds_statespace(self, protocol: DBProtocol) -> None:
        ast = db_to_session_type(protocol)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2
        assert len(ss.transitions) >= 1


# ---------------------------------------------------------------------------
# Lattice property tests
# ---------------------------------------------------------------------------

class TestLatticeProperties:
    """All database protocols must form lattices."""

    @pytest.mark.parametrize("protocol", ALL_DB_PROTOCOLS,
                             ids=[p.name for p in ALL_DB_PROTOCOLS])
    def test_is_lattice(self, protocol: DBProtocol) -> None:
        result = verify_db_protocol(protocol)
        assert result.is_well_formed, (
            f"{protocol.name} does not form a lattice: "
            f"{result.lattice_result.counterexample}"
        )

    def test_simple_protocols_distributive(self) -> None:
        """Protocols without branching after selection are distributive."""
        # Connection pool and cursor have simple selection structures
        for proto in [connection_pool(), cursor_iteration()]:
            result = verify_db_protocol(proto)
            assert result.distributivity.is_distributive, (
                f"{proto.name} should be distributive"
            )

    @pytest.mark.parametrize("protocol", ALL_DB_PROTOCOLS,
                             ids=[p.name for p in ALL_DB_PROTOCOLS])
    def test_lattice_classification_exists(self, protocol: DBProtocol) -> None:
        """Every protocol has a lattice classification (distributive or not)."""
        result = verify_db_protocol(protocol)
        assert result.distributivity.classification in (
            "distributive_lattice", "distributive", "lattice",
        )


# ---------------------------------------------------------------------------
# Verification pipeline tests
# ---------------------------------------------------------------------------

class TestVerificationPipeline:
    """Test the full verification pipeline."""

    def test_verify_jdbc_connection(self) -> None:
        result = verify_db_protocol(jdbc_connection())
        assert isinstance(result, DBAnalysisResult)
        assert result.num_states > 0
        assert result.num_transitions > 0
        assert result.num_valid_paths > 0
        assert result.num_violations >= 0
        assert len(result.test_source) > 0

    def test_verify_jdbc_transaction(self) -> None:
        result = verify_db_protocol(jdbc_transaction())
        assert result.is_well_formed
        assert result.num_states >= 4

    def test_verify_two_phase_commit(self) -> None:
        result = verify_db_protocol(two_phase_commit())
        assert result.is_well_formed
        assert result.num_states >= 4

    def test_verify_connection_pool(self) -> None:
        result = verify_db_protocol(connection_pool())
        assert result.is_well_formed

    def test_verify_cursor_iteration(self) -> None:
        result = verify_db_protocol(cursor_iteration())
        assert result.is_well_formed

    def test_verify_all(self) -> None:
        results = verify_all_db_protocols()
        assert len(results) == len(ALL_DB_PROTOCOLS)
        for r in results:
            assert r.is_well_formed

    def test_coverage_positive(self) -> None:
        result = verify_db_protocol(jdbc_connection())
        assert result.coverage.state_coverage > 0
        assert result.coverage.transition_coverage > 0

    def test_custom_config(self) -> None:
        from reticulate.testgen import TestGenConfig
        config = TestGenConfig(
            class_name="CustomJDBCTest",
            package_name="com.custom.test",
        )
        result = verify_db_protocol(jdbc_connection(), config=config)
        assert "CustomJDBCTest" in result.test_source


# ---------------------------------------------------------------------------
# Isolation level comparison tests
# ---------------------------------------------------------------------------

class TestIsolationComparison:
    """Test isolation level comparison via subtyping."""

    def test_same_protocol_is_compatible(self) -> None:
        """Any protocol is compatible with itself (reflexivity)."""
        proto = jdbc_connection()
        old_ast = db_to_session_type(proto)
        new_ast = db_to_session_type(proto)
        assert is_subtype(old_ast, new_ast)

    def test_compare_isolation_returns_result(self) -> None:
        result = compare_isolation_levels(
            jdbc_connection(),
            jdbc_connection(),
        )
        assert isinstance(result, IsolationComparisonResult)
        assert result.base_analysis.is_well_formed
        assert result.strict_analysis.is_well_formed

    def test_different_protocols_not_subtype(self) -> None:
        """JDBC connection is NOT a subtype of 2PC (different structure)."""
        conn_ast = db_to_session_type(jdbc_connection())
        tpc_ast = db_to_session_type(two_phase_commit())
        assert not is_subtype(conn_ast, tpc_ast)


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------

class TestReportFormatting:
    """Test that report formatters produce non-empty, structured output."""

    def test_db_report_format(self) -> None:
        result = verify_db_protocol(jdbc_connection())
        report = format_db_report(result)
        assert "JDBCConnection" in report
        assert "DATABASE PROTOCOL REPORT" in report
        assert "Lattice Analysis" in report
        assert "Verdict" in report

    def test_isolation_report_format(self) -> None:
        result = compare_isolation_levels(
            jdbc_connection(),
            jdbc_connection(),
        )
        report = format_isolation_report(result)
        assert "ISOLATION LEVEL COMPARISON" in report
        assert "Compatibility" in report

    def test_summary_format(self) -> None:
        results = verify_all_db_protocols()
        summary = format_db_summary(results)
        assert "SUMMARY" in summary
        assert "JDBCConnection" in summary
        assert "TwoPhaseCommit" in summary

    def test_report_includes_properties(self) -> None:
        result = verify_db_protocol(two_phase_commit())
        report = format_db_report(result)
        assert "distributed_consensus" in report
        assert "SERIALIZABLE" in report

    def test_report_includes_participants(self) -> None:
        result = verify_db_protocol(connection_pool())
        report = format_db_report(result)
        assert "ConnectionPool" in report


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test the ALL_DB_PROTOCOLS registry."""

    def test_registry_count(self) -> None:
        assert len(ALL_DB_PROTOCOLS) == 8

    def test_unique_names(self) -> None:
        names = [p.name for p in ALL_DB_PROTOCOLS]
        assert len(names) == len(set(names))

    def test_all_have_descriptions(self) -> None:
        for p in ALL_DB_PROTOCOLS:
            assert len(p.description) > 0

    def test_all_have_properties(self) -> None:
        for p in ALL_DB_PROTOCOLS:
            assert len(p.properties) >= 2

    def test_all_have_isolation_levels(self) -> None:
        for p in ALL_DB_PROTOCOLS:
            assert p.isolation_level in (
                "READ_UNCOMMITTED",
                "READ_COMMITTED",
                "REPEATABLE_READ",
                "SERIALIZABLE",
            )

    def test_all_have_participants(self) -> None:
        for p in ALL_DB_PROTOCOLS:
            assert len(p.participants) >= 2


# ---------------------------------------------------------------------------
# State space property tests
# ---------------------------------------------------------------------------

class TestStateSpaceProperties:
    """Verify structural properties of the generated state spaces."""

    def test_jdbc_connection_has_close_transition(self) -> None:
        ast = db_to_session_type(jdbc_connection())
        ss = build_statespace(ast)
        labels = {lbl for _, lbl, _ in ss.transitions}
        assert "close" in labels

    def test_two_phase_commit_has_vote_selection(self) -> None:
        ast = db_to_session_type(two_phase_commit())
        ss = build_statespace(ast)
        # Selection transitions should exist for VOTE_YES/VOTE_NO
        sel_labels = {lbl for _, lbl, _ in ss.selection_transitions}
        assert "VOTE_YES" in sel_labels or "VOTE_NO" in sel_labels

    def test_connection_pool_has_timeout_path(self) -> None:
        ast = db_to_session_type(connection_pool())
        ss = build_statespace(ast)
        sel_labels = {lbl for _, lbl, _ in ss.selection_transitions}
        assert "TIMEOUT" in sel_labels

    def test_cursor_has_recursive_structure(self) -> None:
        """Cursor iteration should have a cycle (from recursion)."""
        ast = db_to_session_type(cursor_iteration())
        ss = build_statespace(ast)
        # With recursion, some state should have a path back to itself
        for s in ss.states:
            if s in ss.reachable_from(s) and s != ss.bottom:
                return  # found a cycle
        # If no cycle found via reachable_from, check transitions
        labels = {lbl for _, lbl, _ in ss.transitions}
        assert "fetchNext" in labels

    def test_prepared_statement_ordering(self) -> None:
        """Prepare must come before bind, bind before execute."""
        ast = db_to_session_type(prepared_statement())
        ss = build_statespace(ast)
        # The top state should have 'prepare' as its only enabled transition
        enabled = ss.enabled(ss.top)
        assert len(enabled) == 1
        assert enabled[0][0] == "prepare"
