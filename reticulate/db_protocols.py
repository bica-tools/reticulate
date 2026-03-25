"""Database transaction protocol verification via session types (Step 90).

Models database transaction protocols as session types, enabling formal
verification of protocol correctness through lattice analysis, coverage
computation, and conformance test generation.

Database protocols --- JDBC connection lifecycle, transactions with
savepoints, connection pooling, prepared statements, and two-phase
commit --- follow strict state-machine protocols mandated by the JDBC
specification, XA transaction standard, and connection pool contracts.

This module encodes each protocol as a session type, builds the
corresponding state space (reticulate), checks lattice properties,
and generates conformance test suites.  The key insight is that
well-designed database protocols naturally form lattices: every pair
of protocol states has a well-defined join (least common continuation)
and meet (greatest common predecessor), ensuring unambiguous recovery
from any reachable state --- a property directly relevant to database
error handling and connection cleanup.

Isolation levels are tracked as metadata on the protocol definition
but do not alter the session type structure: the same state machine
governs connection lifecycle regardless of isolation level.  The
isolation level determines *runtime semantics* (visibility of
concurrent writes) rather than *protocol structure*.

Usage:
    from reticulate.db_protocols import (
        jdbc_connection,
        jdbc_transaction,
        verify_db_protocol,
        format_db_report,
    )
    proto = jdbc_connection()
    result = verify_db_protocol(proto)
    print(format_db_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.coverage import CoverageResult, compute_coverage
from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.subtyping import SubtypingResult, check_subtype, is_subtype
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DBProtocol:
    """A named database protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "JDBCConnection").
        session_type_string: Session type encoding of the protocol.
        isolation_level: Database isolation level (e.g., "READ_COMMITTED").
        description: Free-text description of the protocol purpose.
        participants: Roles involved (e.g., ("Client", "Database")).
        properties: Protocol properties guaranteed by the type
            (e.g., ("connection_safety", "resource_cleanup")).
    """
    name: str
    session_type_string: str
    isolation_level: str
    description: str
    participants: tuple[str, ...] = ("Client", "Database")
    properties: tuple[str, ...] = ()


@dataclass(frozen=True)
class DBAnalysisResult:
    """Complete analysis result for a database protocol.

    Attributes:
        protocol: The analysed protocol definition.
        ast: Parsed session type AST.
        state_space: Constructed state space (reticulate).
        lattice_result: Lattice property check.
        distributivity: Distributivity check result.
        enumeration: Test path enumeration.
        test_source: Generated JUnit 5 test source.
        coverage: Coverage analysis.
        num_states: Number of states in the state space.
        num_transitions: Number of transitions.
        num_valid_paths: Count of valid execution paths.
        num_violations: Count of protocol violation points.
        is_well_formed: True iff state space is a lattice.
    """
    protocol: DBProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    enumeration: EnumerationResult
    test_source: str
    coverage: CoverageResult
    num_states: int
    num_transitions: int
    num_valid_paths: int
    num_violations: int
    is_well_formed: bool


@dataclass(frozen=True)
class IsolationComparisonResult:
    """Result of comparing protocol variants at different isolation levels.

    Attributes:
        base_protocol: The base protocol (e.g., READ_COMMITTED).
        strict_protocol: The stricter protocol (e.g., SERIALIZABLE).
        subtyping_result: Gay-Hole subtyping check.
        is_compatible: True iff base is subtype of strict.
        base_analysis: Full analysis of base protocol.
        strict_analysis: Full analysis of strict protocol.
    """
    base_protocol: DBProtocol
    strict_protocol: DBProtocol
    subtyping_result: SubtypingResult
    is_compatible: bool
    base_analysis: DBAnalysisResult
    strict_analysis: DBAnalysisResult


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

def jdbc_connection() -> DBProtocol:
    """JDBC Connection lifecycle protocol.

    Models the standard JDBC connection lifecycle:
    1. Connect to the database
    2. Create a statement
    3. Execute the statement (query or update)
    4. Process results or handle errors
    5. Close the connection

    Based on JDBC 4.3 specification (JSR 221), Section 9: Connections.
    The session type captures the mandatory ordering: connect before
    creating statements, execute before fetching results, close after
    all operations complete.
    """
    return DBProtocol(
        name="JDBCConnection",
        session_type_string=(
            "&{connect: &{createStatement: &{execute: "
            "+{SUCCESS: &{processResults: &{close: end}}, "
            "ERROR: &{close: end}}}}}"
        ),
        isolation_level="READ_COMMITTED",
        description=(
            "JDBC Connection lifecycle: connect, create statement, "
            "execute, process results or handle error, then close. "
            "Models the mandatory ordering specified by JDBC 4.3."
        ),
        participants=("Client", "Database"),
        properties=(
            "connection_safety",
            "resource_cleanup",
            "error_handling",
            "statement_ordering",
        ),
    )


def jdbc_transaction() -> DBProtocol:
    """JDBC transaction protocol with savepoints and commit/rollback.

    Models the transaction lifecycle:
    1. Begin transaction (setAutoCommit(false))
    2. Execute operations (one or more)
    3. Optionally set savepoints
    4. Commit or rollback
    5. End transaction (setAutoCommit(true))

    Based on JDBC 4.3, Section 10: Transactions.  The key protocol
    constraint is that commit and rollback are mutually exclusive
    terminal operations within a transaction scope.
    """
    return DBProtocol(
        name="JDBCTransaction",
        session_type_string=(
            "&{beginTx: &{executeOp: &{checkpoint: "
            "+{SAVEPOINT: &{executeMore: +{COMMIT: &{commitTx: end}, "
            "ROLLBACK: &{rollbackTx: end}}}, "
            "NO_SAVEPOINT: +{COMMIT: &{commitTx: end}, "
            "ROLLBACK: &{rollbackTx: end}}}}}}"
        ),
        isolation_level="READ_COMMITTED",
        description=(
            "JDBC transaction with savepoints: begin, execute, "
            "optionally set savepoint, execute more, then commit "
            "or rollback.  Models the transactional contract of "
            "JDBC 4.3 Section 10."
        ),
        participants=("Client", "TransactionManager"),
        properties=(
            "atomicity",
            "isolation",
            "commit_rollback_exclusion",
            "savepoint_ordering",
        ),
    )


def connection_pool() -> DBProtocol:
    """Connection pool acquire/use/release protocol.

    Models the connection pool lifecycle:
    1. Request a connection from the pool
    2. Pool validates or times out
    3. If acquired, use the connection (execute operations)
    4. Return connection to pool
    5. Pool validates returned connection

    Based on common pool implementations (HikariCP, Apache DBCP, C3P0).
    The timeout path models the case where no connections are available
    within the configured wait period.
    """
    return DBProtocol(
        name="ConnectionPool",
        session_type_string=(
            "&{requestConn: +{ACQUIRED: &{useConnection: "
            "&{executeQuery: &{releaseConn: &{validateReturn: end}}}}, "
            "TIMEOUT: end}}"
        ),
        isolation_level="READ_COMMITTED",
        description=(
            "Connection pool lifecycle: request connection, acquire "
            "or timeout, use for queries, release back to pool, "
            "validate on return.  Models HikariCP/DBCP contracts."
        ),
        participants=("Client", "ConnectionPool", "Database"),
        properties=(
            "resource_bounding",
            "timeout_safety",
            "connection_reuse",
            "leak_prevention",
        ),
    )


def prepared_statement() -> DBProtocol:
    """Prepared statement lifecycle protocol.

    Models the prepared statement pattern:
    1. Prepare the SQL statement (server-side compilation)
    2. Bind parameters to the prepared statement
    3. Execute the prepared statement
    4. Fetch results (success) or handle error
    5. Close the prepared statement

    Based on JDBC 4.3, Section 11: PreparedStatement.  The session
    type enforces the prepare-before-bind-before-execute ordering
    that prevents SQL injection through proper parameterization.
    """
    return DBProtocol(
        name="PreparedStatement",
        session_type_string=(
            "&{prepare: &{bindParams: &{executePrepared: "
            "+{RESULTS: &{fetchRows: &{closePrepared: end}}, "
            "EXEC_ERROR: &{closePrepared: end}}}}}"
        ),
        isolation_level="READ_COMMITTED",
        description=(
            "Prepared statement lifecycle: prepare SQL, bind parameters, "
            "execute, fetch results or handle error, close.  Enforces "
            "prepare-before-bind-before-execute ordering (JDBC 4.3 S11)."
        ),
        participants=("Client", "Database"),
        properties=(
            "sql_injection_prevention",
            "parameter_binding_order",
            "statement_cleanup",
            "execution_safety",
        ),
    )


def two_phase_commit() -> DBProtocol:
    """Two-phase commit (2PC) protocol for distributed transactions.

    Models the XA two-phase commit protocol:
    1. Coordinator sends prepare to all participants
    2. Participants vote (YES/NO)
    3. If all vote YES: coordinator sends commit
    4. If any vote NO: coordinator sends abort
    5. Participants acknowledge

    Based on the XA specification (X/Open CAE) and JTA (JSR 907).
    The session type captures the fundamental constraint that the
    commit/abort decision depends on the collective vote.
    """
    return DBProtocol(
        name="TwoPhaseCommit",
        session_type_string=(
            "&{prepare: +{VOTE_YES: &{doCommit: &{ackCommit: end}}, "
            "VOTE_NO: &{doAbort: &{ackAbort: end}}}}"
        ),
        isolation_level="SERIALIZABLE",
        description=(
            "XA two-phase commit: prepare, vote yes/no, "
            "commit or abort, acknowledge.  Models the fundamental "
            "distributed consensus protocol from X/Open CAE / JTA."
        ),
        participants=("Coordinator", "Participant"),
        properties=(
            "atomicity",
            "distributed_consensus",
            "failure_recovery",
            "global_serializability",
        ),
    )


def cursor_iteration() -> DBProtocol:
    """Database cursor iteration protocol.

    Models the server-side cursor pattern:
    1. Open cursor (declare + open)
    2. Fetch rows iteratively (hasNext/next pattern)
    3. Close cursor when done or on error

    Based on SQL:2003 cursor operations.  The recursive structure
    models the unbounded iteration over result sets.
    """
    return DBProtocol(
        name="CursorIteration",
        session_type_string=(
            "&{openCursor: rec X . &{fetchNext: "
            "+{HAS_ROW: &{processRow: X}, "
            "NO_MORE_ROWS: &{closeCursor: end}}}}"
        ),
        isolation_level="READ_COMMITTED",
        description=(
            "Database cursor: open, iteratively fetch/process rows, "
            "close when exhausted.  Models SQL:2003 cursor operations "
            "with the standard hasNext/next iteration pattern."
        ),
        participants=("Client", "Database"),
        properties=(
            "cursor_safety",
            "resource_cleanup",
            "iteration_ordering",
            "exhaustion_guarantee",
        ),
    )


def stored_procedure() -> DBProtocol:
    """Stored procedure call protocol.

    Models the stored procedure invocation pattern:
    1. Prepare the callable statement
    2. Register OUT parameters
    3. Set IN parameters
    4. Execute the procedure
    5. Retrieve OUT parameters or handle error
    6. Close the callable statement

    Based on JDBC 4.3, Section 13: CallableStatement.
    """
    return DBProtocol(
        name="StoredProcedure",
        session_type_string=(
            "&{prepareCall: &{registerOutParams: &{setInParams: "
            "&{executeProc: +{PROC_OK: &{getOutParams: "
            "&{closeCallable: end}}, "
            "PROC_ERROR: &{closeCallable: end}}}}}}"
        ),
        isolation_level="READ_COMMITTED",
        description=(
            "Stored procedure call: prepare callable statement, "
            "register OUT parameters, set IN parameters, execute, "
            "retrieve OUT params or handle error, close.  "
            "Models JDBC 4.3 Section 13 CallableStatement."
        ),
        participants=("Client", "Database"),
        properties=(
            "parameter_registration_order",
            "callable_cleanup",
            "error_handling",
            "out_parameter_access",
        ),
    )


def batch_execution() -> DBProtocol:
    """Batch statement execution protocol.

    Models JDBC batch execution:
    1. Create statement or prepared statement
    2. Add statements to batch (accumulate)
    3. Execute batch
    4. Process batch results (success count or errors)
    5. Clear and close

    Based on JDBC 4.3, Section 14: Batch Updates.
    """
    return DBProtocol(
        name="BatchExecution",
        session_type_string=(
            "&{createBatch: &{addStatement: &{executeBatch: "
            "+{BATCH_OK: &{processCounts: &{clearBatch: end}}, "
            "BATCH_ERROR: &{clearBatch: end}}}}}"
        ),
        isolation_level="READ_COMMITTED",
        description=(
            "Batch statement execution: create batch, add statements, "
            "execute batch, process results or handle errors, clear. "
            "Models JDBC 4.3 Section 14 batch update protocol."
        ),
        participants=("Client", "Database"),
        properties=(
            "batch_atomicity",
            "batch_ordering",
            "resource_cleanup",
            "error_reporting",
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_DB_PROTOCOLS: tuple[DBProtocol, ...] = (
    jdbc_connection(),
    jdbc_transaction(),
    connection_pool(),
    prepared_statement(),
    two_phase_commit(),
    cursor_iteration(),
    stored_procedure(),
    batch_execution(),
)
"""All pre-defined database protocols."""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def db_to_session_type(protocol: DBProtocol) -> SessionType:
    """Parse the protocol's session type string into an AST."""
    return parse(protocol.session_type_string)


def verify_db_protocol(
    protocol: DBProtocol,
    config: TestGenConfig | None = None,
) -> DBAnalysisResult:
    """Run the full verification pipeline on a database protocol.

    Parses the protocol's session type, builds the state space,
    checks lattice properties, generates conformance tests, and
    computes coverage.

    Args:
        protocol: The database protocol to verify.
        config: Optional test generation configuration.

    Returns:
        A complete DBAnalysisResult.
    """
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}ProtocolTest",
            package_name="com.db.conformance",
        )

    # 1. Parse and build state space
    ast = db_to_session_type(protocol)
    ss = build_statespace(ast)

    # 2. Lattice analysis
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    # 3. Test generation
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.session_type_string)

    # 4. Coverage
    cov = compute_coverage(ss, result=enum)

    return DBAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        enumeration=enum,
        test_source=test_src,
        coverage=cov,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_valid_paths=len(enum.valid_paths),
        num_violations=len(enum.violations),
        is_well_formed=lr.is_lattice,
    )


def verify_all_db_protocols() -> list[DBAnalysisResult]:
    """Verify all pre-defined database protocols and return results."""
    return [verify_db_protocol(p) for p in ALL_DB_PROTOCOLS]


def compare_isolation_levels(
    base: DBProtocol,
    strict: DBProtocol,
) -> IsolationComparisonResult:
    """Compare two protocol variants at different isolation levels.

    Uses Gay-Hole subtyping to check whether the base protocol
    (lower isolation) is compatible with the strict protocol
    (higher isolation).  If the base is a subtype of the strict,
    then code written for the base isolation level will work
    correctly at the stricter level.

    Args:
        base: Protocol at lower isolation level.
        strict: Protocol at higher isolation level.

    Returns:
        IsolationComparisonResult with compatibility verdict.
    """
    base_ast = db_to_session_type(base)
    strict_ast = db_to_session_type(strict)

    sub_result = check_subtype(base_ast, strict_ast)

    base_analysis = verify_db_protocol(base)
    strict_analysis = verify_db_protocol(strict)

    return IsolationComparisonResult(
        base_protocol=base,
        strict_protocol=strict,
        subtyping_result=sub_result,
        is_compatible=sub_result.is_subtype,
        base_analysis=base_analysis,
        strict_analysis=strict_analysis,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_db_report(result: DBAnalysisResult) -> str:
    """Format a DBAnalysisResult as structured text for terminal output."""
    lines: list[str] = []
    proto = result.protocol

    lines.append("=" * 70)
    lines.append(f"  DATABASE PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")

    # Participants
    lines.append("--- Participants ---")
    for p in proto.participants:
        lines.append(f"  - {p}")
    lines.append("")

    # Isolation level
    lines.append("--- Isolation Level ---")
    lines.append(f"  {proto.isolation_level}")
    lines.append("")

    # Properties
    lines.append("--- Protocol Properties ---")
    for p in proto.properties:
        lines.append(f"  - {p}")
    lines.append("")

    # Session type
    lines.append("--- Session Type ---")
    lines.append(f"  {proto.session_type_string}")
    lines.append("")

    # State space
    lines.append("--- State Space ---")
    lines.append(f"  States:      {result.num_states}")
    lines.append(f"  Transitions: {result.num_transitions}")
    lines.append(f"  Top (init):  {result.state_space.top}")
    lines.append(f"  Bottom (end):{result.state_space.bottom}")
    lines.append("")

    # Lattice
    lines.append("--- Lattice Analysis ---")
    lines.append(f"  Is lattice:      {result.lattice_result.is_lattice}")
    lines.append(f"  Distributive:    {result.distributivity.is_distributive}")
    if not result.lattice_result.is_lattice and result.lattice_result.counterexample:
        lines.append(f"  Counterexample:  {result.lattice_result.counterexample}")
    lines.append("")

    # Test generation
    lines.append("--- Test Generation ---")
    lines.append(f"  Valid paths:     {result.num_valid_paths}")
    lines.append(f"  Violation points:{result.num_violations}")
    lines.append("")

    # Coverage
    lines.append("--- Coverage ---")
    lines.append(f"  State coverage:      {result.coverage.state_coverage:.1%}")
    lines.append(f"  Transition coverage: {result.coverage.transition_coverage:.1%}")
    lines.append("")

    # Verdict
    lines.append("--- Verdict ---")
    if result.is_well_formed:
        lines.append("  PASS: Protocol forms a valid lattice.")
        lines.append("  Every pair of protocol states has a well-defined")
        lines.append("  join and meet, ensuring unambiguous error recovery.")
    else:
        lines.append("  FAIL: Protocol does NOT form a valid lattice.")
        lines.append("  Some protocol states lack a join or meet,")
        lines.append("  which may lead to ambiguous recovery scenarios.")
    lines.append("")

    return "\n".join(lines)


def format_isolation_report(result: IsolationComparisonResult) -> str:
    """Format an IsolationComparisonResult as structured text."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append(f"  ISOLATION LEVEL COMPARISON: {result.base_protocol.name} "
                 f"({result.base_protocol.isolation_level}) -> "
                 f"{result.strict_protocol.name} "
                 f"({result.strict_protocol.isolation_level})")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"  Base protocol: {result.base_protocol.name} "
                 f"({result.base_protocol.isolation_level})")
    lines.append(f"    States: {result.base_analysis.num_states}, "
                 f"Transitions: {result.base_analysis.num_transitions}")
    lines.append(f"  Strict protocol: {result.strict_protocol.name} "
                 f"({result.strict_protocol.isolation_level})")
    lines.append(f"    States: {result.strict_analysis.num_states}, "
                 f"Transitions: {result.strict_analysis.num_transitions}")
    lines.append("")

    lines.append("--- Subtyping Analysis ---")
    lines.append(f"  {result.base_protocol.name} <= {result.strict_protocol.name}: "
                 f"{result.subtyping_result.is_subtype}")
    if result.subtyping_result.reason:
        lines.append(f"  Reason: {result.subtyping_result.reason}")
    lines.append("")

    lines.append("--- Compatibility ---")
    if result.is_compatible:
        lines.append("  COMPATIBLE: Base protocol is compatible with strict.")
        lines.append("  Code written for the base isolation level will work")
        lines.append("  correctly at the stricter isolation level.")
    else:
        lines.append("  INCOMPATIBLE: Base protocol is NOT compatible with strict.")
        lines.append("  Code written for the base isolation level may behave")
        lines.append("  differently at the stricter isolation level.")
    lines.append("")

    return "\n".join(lines)


def format_db_summary(results: list[DBAnalysisResult]) -> str:
    """Format a summary table of all verified database protocols."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("  DATABASE PROTOCOL VERIFICATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    header = (
        f"  {'Protocol':<22} {'Iso.Level':<16} "
        f"{'States':>6} {'Trans':>6} {'Lattice':>8} {'Dist':>6} {'Paths':>6}"
    )
    lines.append(header)
    lines.append("  " + "-" * 68)

    for r in results:
        lattice_str = "YES" if r.is_well_formed else "NO"
        dist_str = "YES" if r.distributivity.is_distributive else "NO"
        row = (
            f"  {r.protocol.name:<22} "
            f"{r.protocol.isolation_level:<16} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{lattice_str:>8} "
            f"{dist_str:>6} "
            f"{r.num_valid_paths:>6}"
        )
        lines.append(row)

    lines.append("")
    all_lattice = all(r.is_well_formed for r in results)
    lines.append(f"  All protocols form lattices: {'YES' if all_lattice else 'NO'}")
    lines.append("")

    return "\n".join(lines)
