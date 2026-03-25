"""Smart contract lifecycle verification via session types (Step 81).

Models blockchain smart contract state machines as session types, enabling
formal verification of contract correctness through lattice analysis,
conformance testing, and runtime monitor generation.

Smart contracts enforce strict state-machine protocols: an ERC-20 token
must be deployed before transfers, an auction must start before bids are
accepted, a DeFi lending pool must receive deposits before issuing loans.
These lifecycle constraints map naturally to session types.

This module encodes five canonical smart contract patterns as session
types, builds their state spaces (reticulates), checks lattice properties,
and generates conformance monitors.  The key insight is that well-designed
smart contract lifecycles form lattices: every pair of contract states
has a well-defined join (least common continuation) and meet (greatest
common predecessor), ensuring unambiguous protocol recovery from any
reachable state.

Usage::

    from reticulate.smart_contract import (
        erc20_lifecycle,
        verify_contract,
        format_contract_report,
    )
    wf = erc20_lifecycle()
    result = verify_contract(wf)
    print(format_contract_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reticulate.coverage import CoverageResult, compute_coverage
from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)
from reticulate.monitor import (
    MonitorTemplate,
    build_transition_table,
    generate_python_monitor,
    generate_java_monitor,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmartContractState:
    """A named state in a smart contract lifecycle.

    Attributes:
        name: Human-readable state name (e.g. "Deployed", "Active").
        allowed_transitions: Method names available from this state.
        requires_auth: Whether this state transition requires authorisation
            (e.g. only the owner can call ``pause``).
    """
    name: str
    allowed_transitions: tuple[str, ...]
    requires_auth: bool = False


@dataclass(frozen=True)
class SmartContractWorkflow:
    """A smart contract lifecycle modelled as a session type.

    Attributes:
        name: Contract name (e.g. "ERC20Token").
        standard: The standard or pattern (e.g. "ERC-20", "ERC-721").
        states: Named contract states with their allowed transitions.
        session_type_string: Session type encoding of the lifecycle.
        description: Free-text description of the contract purpose.
        solidity_events: Solidity events emitted during transitions.
    """
    name: str
    standard: str
    states: tuple[SmartContractState, ...]
    session_type_string: str
    description: str
    solidity_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractAnalysisResult:
    """Complete analysis result for a smart contract lifecycle.

    Attributes:
        workflow: The analysed contract workflow definition.
        ast: Parsed session type AST.
        state_space: Constructed state space (reticulate).
        lattice_result: Lattice property check.
        distributivity: Distributivity check result.
        enumeration: Test path enumeration.
        test_source: Generated JUnit 5 test source.
        coverage: Coverage analysis.
        monitor_source: Generated Python runtime monitor source.
        num_states: Number of states in the state space.
        num_transitions: Number of transitions.
        num_valid_paths: Count of valid execution paths.
        num_violations: Count of protocol violation points.
        is_well_formed: True iff state space is a lattice.
    """
    workflow: SmartContractWorkflow
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    enumeration: EnumerationResult
    test_source: str
    coverage: CoverageResult
    monitor_source: str
    num_states: int
    num_transitions: int
    num_valid_paths: int
    num_violations: int
    is_well_formed: bool


# ---------------------------------------------------------------------------
# State machine → session type conversion
# ---------------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    """Convert a transition label to a valid session type method name.

    Replaces spaces and special characters with underscores; lowercases.
    """
    out = label.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower()


def solidity_to_session_type(
    states: Sequence[SmartContractState],
    transitions: Sequence[tuple[str, str, str]],
) -> str:
    """Convert a contract state machine to a session type string.

    Args:
        states: The contract states.
        transitions: Triples of (source_state_name, method, target_state_name).

    Returns:
        A session type string encoding the lifecycle.
    """
    if not states or not transitions:
        return "end"

    # Build adjacency: state_name -> [(method, target_state_name)]
    sm: dict[str, list[tuple[str, str]]] = {}
    all_state_names = {s.name for s in states}

    for src, method, tgt in transitions:
        sm.setdefault(src, []).append((_sanitize_label(method), tgt))

    # Find initial state: appears only as source, never as target,
    # or fall back to first state.
    all_targets = {tgt for _, _, tgt in transitions}
    initial_candidates = all_state_names - all_targets
    if initial_candidates:
        initial = sorted(initial_candidates)[0]
    else:
        initial = states[0].name

    # Terminal states: no outgoing transitions.
    terminal = {s.name for s in states if s.name not in sm}

    # Build session type string from the state machine.
    return _build_type_string(sm, initial, terminal)


def _build_type_string(
    sm: dict[str, list[tuple[str, str]]],
    initial: str,
    terminal: set[str],
) -> str:
    """Convert a state machine to a session type string."""

    def build(state: str, in_progress: frozenset[str]) -> str:
        if state in terminal or state not in sm or not sm[state]:
            return "end"
        if state in in_progress:
            return f"X{_state_index(state, sm)}"

        in_progress_next = in_progress | {state}
        transitions = sm[state]

        choices = []
        for method, next_state in transitions:
            cont = build(next_state, in_progress_next)
            choices.append(f"{method}: {cont}")

        if len(choices) == 1:
            body = f"&{{{choices[0]}}}"
        else:
            body = "&{" + ", ".join(choices) + "}"

        # Check if this state is reachable from itself (cycle).
        if _state_reachable(sm, state, state):
            var = f"X{_state_index(state, sm)}"
            return f"rec {var} . {body}"
        return body

    return build(initial, frozenset())


def _state_index(state: str, sm: dict[str, list[tuple[str, str]]]) -> str:
    """Map state name to a variable index."""
    keys = sorted(sm.keys())
    if state in keys:
        idx = keys.index(state)
        return "" if idx == 0 else str(idx)
    return state


def _state_reachable(
    sm: dict[str, list[tuple[str, str]]],
    start: str,
    target: str,
) -> bool:
    """Check if target is reachable from any successor of start."""
    visited: set[str] = set()
    stack = [tgt for _, tgt in sm.get(start, [])]
    while stack:
        s = stack.pop()
        if s == target:
            return True
        if s in visited:
            continue
        visited.add(s)
        for _, ns in sm.get(s, []):
            stack.append(ns)
    return False


# ---------------------------------------------------------------------------
# Predefined contract lifecycles
# ---------------------------------------------------------------------------

def erc20_lifecycle() -> SmartContractWorkflow:
    """ERC-20 fungible token lifecycle: deploy, approve, transfer, burn.

    Models the standard ERC-20 token contract with deployment,
    approval management, token transfers, and burning.  The approve-
    then-transferFrom pattern prevents the well-known double-spend
    race condition.

    Solidity events: Transfer, Approval.
    """
    return SmartContractWorkflow(
        name="ERC20Token",
        standard="ERC-20",
        states=(
            SmartContractState("Undeployed", ("deploy",), requires_auth=True),
            SmartContractState("Active", ("approve", "transfer", "burn")),
            SmartContractState("Approved", ("transferFrom", "transfer", "burn")),
            SmartContractState("Burned", ()),
        ),
        session_type_string=(
            "&{deploy: rec X . &{approve: &{transferFrom: X, "
            "transfer: X, burn: end}, "
            "transfer: X, burn: end}}"
        ),
        description=(
            "ERC-20 fungible token lifecycle: deploy the contract, "
            "then repeatedly approve allowances, transfer tokens, or "
            "burn tokens to terminate."
        ),
        solidity_events=("Transfer", "Approval"),
    )


def erc721_lifecycle() -> SmartContractWorkflow:
    """ERC-721 non-fungible token lifecycle: mint, transfer, approve, burn.

    Models the NFT lifecycle including minting, ownership transfer,
    approval management, and burning.  Each token has a unique ID
    and at most one owner.

    Solidity events: Transfer, Approval, ApprovalForAll.
    """
    return SmartContractWorkflow(
        name="ERC721Token",
        standard="ERC-721",
        states=(
            SmartContractState("Undeployed", ("deploy",), requires_auth=True),
            SmartContractState("Deployed", ("mint",), requires_auth=True),
            SmartContractState("Minted", ("transfer", "approve", "burn")),
            SmartContractState("Approved", ("transferFrom", "transfer", "burn")),
            SmartContractState("Burned", ()),
        ),
        session_type_string=(
            "&{deploy: &{mint: rec X . &{transfer: X, "
            "approve: &{transferFrom: X, transfer: X, burn: end}, "
            "burn: end}}}"
        ),
        description=(
            "ERC-721 non-fungible token lifecycle: deploy the contract, "
            "mint a unique token, then repeatedly transfer, approve "
            "operators, or burn to destroy the token."
        ),
        solidity_events=("Transfer", "Approval", "ApprovalForAll"),
    )


def defi_lending() -> SmartContractWorkflow:
    """DeFi lending pool: deposit, borrow, repay, withdraw, liquidate.

    Models a decentralised lending protocol where users deposit
    collateral, borrow against it, repay loans, and withdraw.
    If the collateral ratio drops below the threshold, the position
    is liquidated.

    Solidity events: Deposit, Borrow, Repay, Withdraw, Liquidation.
    """
    return SmartContractWorkflow(
        name="DeFiLending",
        standard="DeFi-Lending",
        states=(
            SmartContractState("Empty", ("deposit",)),
            SmartContractState("Funded", ("borrow", "withdraw")),
            SmartContractState("Borrowed", ("repay", "liquidate")),
            SmartContractState("Repaid", ("withdraw", "borrow")),
            SmartContractState("Liquidated", ()),
            SmartContractState("Withdrawn", ()),
        ),
        session_type_string=(
            "&{deposit: rec X . &{borrow: +{HEALTHY: &{repay: X}, "
            "UNDERCOLLATERALIZED: &{liquidate: end}}, "
            "withdraw: end}}"
        ),
        description=(
            "DeFi lending pool lifecycle: deposit collateral, borrow "
            "against it (health check: healthy or undercollateralised), "
            "repay loans to release collateral, withdraw, or face "
            "liquidation if the position becomes undercollateralised."
        ),
        solidity_events=("Deposit", "Borrow", "Repay", "Withdraw", "Liquidation"),
    )


def multisig_wallet() -> SmartContractWorkflow:
    """Multi-signature wallet: propose, approve, execute, reject.

    Models a multi-sig wallet requiring N-of-M approvals before
    executing a transaction.  Any signer can propose; once enough
    approvals are collected, the transaction is executed.  A proposal
    can also be rejected if a quorum rejects.

    Solidity events: Submission, Confirmation, Execution, Revocation.
    """
    return SmartContractWorkflow(
        name="MultisigWallet",
        standard="Multisig",
        states=(
            SmartContractState("Idle", ("propose",)),
            SmartContractState("Proposed", ("approve", "reject")),
            SmartContractState("QuorumReached", ("execute",)),
            SmartContractState("Executed", ()),
            SmartContractState("Rejected", ()),
        ),
        session_type_string=(
            "rec X . &{propose: +{QUORUM_REACHED: &{execute: X}, "
            "QUORUM_NOT_REACHED: &{approve: +{QUORUM_REACHED: "
            "&{execute: X}, QUORUM_NOT_REACHED: end}}, "
            "REJECTED: end}}"
        ),
        description=(
            "Multi-signature wallet lifecycle: propose a transaction, "
            "collect approvals (quorum reached or not), execute if "
            "quorum is met, or reject.  The cycle repeats for each "
            "new proposal."
        ),
        solidity_events=("Submission", "Confirmation", "Execution", "Revocation"),
    )


def auction_contract() -> SmartContractWorkflow:
    """Auction contract: start, bid, end, withdraw.

    Models an English auction where the owner starts the auction,
    bidders place bids during the bidding period, the auction ends,
    and participants withdraw their funds (losers get refunds, the
    winner pays, the seller receives payment).

    Solidity events: AuctionStarted, BidPlaced, AuctionEnded,
    FundsWithdrawn.
    """
    return SmartContractWorkflow(
        name="Auction",
        standard="Auction",
        states=(
            SmartContractState("Created", ("start",), requires_auth=True),
            SmartContractState("Active", ("bid", "endAuction")),
            SmartContractState("Ended", ("withdraw",)),
            SmartContractState("Settled", ()),
        ),
        session_type_string=(
            "&{start: rec X . &{bid: X, endAuction: &{withdraw: end}}}"
        ),
        description=(
            "English auction lifecycle: start the auction, accept bids "
            "(repeated), end the auction, and withdraw funds.  Bidding "
            "is a recursive loop that terminates when the auctioneer "
            "ends the auction."
        ),
        solidity_events=(
            "AuctionStarted", "BidPlaced", "AuctionEnded", "FundsWithdrawn",
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_CONTRACT_WORKFLOWS: tuple[SmartContractWorkflow, ...] = (
    erc20_lifecycle(),
    erc721_lifecycle(),
    defi_lending(),
    multisig_wallet(),
    auction_contract(),
)
"""All pre-defined smart contract lifecycle workflows."""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def contract_to_session_type(workflow: SmartContractWorkflow) -> SessionType:
    """Parse the workflow's session type string into an AST."""
    return parse(workflow.session_type_string)


def verify_contract(
    workflow: SmartContractWorkflow,
    config: TestGenConfig | None = None,
) -> ContractAnalysisResult:
    """Run the full verification pipeline on a smart contract lifecycle.

    Parses the contract's session type, builds the state space,
    checks lattice properties, generates conformance tests, computes
    coverage, and generates a runtime monitor.

    Args:
        workflow: The contract workflow to verify.
        config: Optional test generation configuration.

    Returns:
        A complete ContractAnalysisResult.
    """
    if config is None:
        config = TestGenConfig(
            class_name=f"{workflow.name}LifecycleTest",
            package_name="com.contract.conformance",
        )

    # 1. Parse and build state space
    ast = contract_to_session_type(workflow)
    ss = build_statespace(ast)

    # 2. Lattice analysis
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    # 3. Test generation
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, workflow.session_type_string)

    # 4. Coverage
    cov = compute_coverage(ss, result=enum)

    # 5. Runtime monitor
    monitor_template = generate_python_monitor(ast, workflow.name)
    monitor_src = monitor_template.source_code

    return ContractAnalysisResult(
        workflow=workflow,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        enumeration=enum,
        test_source=test_src,
        coverage=cov,
        monitor_source=monitor_src,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_valid_paths=len(enum.valid_paths),
        num_violations=len(enum.violations),
        is_well_formed=lr.is_lattice,
    )


def verify_all_contracts() -> list[ContractAnalysisResult]:
    """Verify all pre-defined smart contract workflows and return results."""
    return [verify_contract(wf) for wf in ALL_CONTRACT_WORKFLOWS]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_contract_report(result: ContractAnalysisResult) -> str:
    """Format a ContractAnalysisResult as structured text for terminal output."""
    lines: list[str] = []
    wf = result.workflow

    lines.append("=" * 70)
    lines.append(f"  SMART CONTRACT REPORT: {wf.name} ({wf.standard})")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {wf.description}")
    lines.append("")

    # Contract states
    lines.append("--- Contract States ---")
    for s in wf.states:
        auth = " [AUTH]" if s.requires_auth else ""
        trans = ", ".join(s.allowed_transitions) if s.allowed_transitions else "(terminal)"
        lines.append(f"  - {s.name}{auth}: {trans}")
    lines.append("")

    # Session type
    lines.append("--- Session Type ---")
    lines.append(f"  {wf.session_type_string}")
    lines.append("")

    # State space
    lines.append("--- State Space ---")
    lines.append(f"  States:      {result.num_states}")
    lines.append(f"  Transitions: {result.num_transitions}")
    lines.append(f"  Top (init):  {result.state_space.top}")
    lines.append(f"  Bottom (end):{result.state_space.bottom}")
    lines.append("")

    # Lattice
    lattice_str = "YES" if result.lattice_result.is_lattice else "NO"
    dist_str = "yes" if result.distributivity.is_distributive else "no"
    lines.append("--- Lattice Properties ---")
    lines.append(f"  Is lattice:     {lattice_str}")
    lines.append(f"  Has top:        {result.lattice_result.has_top}")
    lines.append(f"  Has bottom:     {result.lattice_result.has_bottom}")
    lines.append(f"  Distributive:   {dist_str}")
    lines.append("")

    # Tests
    lines.append("--- Conformance Tests ---")
    lines.append(f"  Valid paths:          {result.num_valid_paths}")
    lines.append(f"  Violation points:     {result.num_violations}")
    lines.append(f"  Transition coverage:  {result.coverage.transition_coverage:.1%}")
    lines.append(f"  State coverage:       {result.coverage.state_coverage:.1%}")
    lines.append("")

    # Solidity events
    if wf.solidity_events:
        lines.append("--- Solidity Events ---")
        for ev in wf.solidity_events:
            lines.append(f"  - {ev}")
        lines.append("")

    # Verdict
    lines.append("=" * 70)
    if result.is_well_formed:
        lines.append(f"  VERDICT: {wf.name} lifecycle is WELL-FORMED")
        lines.append(f"  The state space forms a bounded lattice.")
        lines.append(f"  {result.num_valid_paths} conformance tests generated.")
    else:
        lines.append(f"  VERDICT: {wf.name} lifecycle has ISSUES")
        lines.append(f"  WARNING: State space does NOT form a lattice!")
        if result.lattice_result.counterexample:
            lines.append(
                f"  Counterexample: {result.lattice_result.counterexample}"
            )
    lines.append("=" * 70)

    return "\n".join(lines)


def format_contract_summary(results: list[ContractAnalysisResult]) -> str:
    """Format a summary table of multiple contract lifecycle analyses."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  SMART CONTRACT LIFECYCLE VERIFICATION SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"  {'Contract':<20s} {'Standard':<12s} {'States':>6s} {'Trans':>6s} "
        f"{'Lattice':>8s} {'Dist':>5s} {'Paths':>6s}"
    )
    lines.append("  " + "-" * 66)

    for r in results:
        latt = "YES" if r.is_well_formed else "NO"
        dist = "yes" if r.distributivity.is_distributive else "no"
        lines.append(
            f"  {r.workflow.name:<20s} {r.workflow.standard:<12s} "
            f"{r.num_states:>6d} {r.num_transitions:>6d} "
            f"{latt:>8s} {dist:>5s} {r.num_valid_paths:>6d}"
        )

    lines.append("")
    total_wf = len(results)
    total_lattice = sum(1 for r in results if r.is_well_formed)
    lines.append(f"  {total_lattice}/{total_wf} contract lifecycles form lattices.")
    lines.append("=" * 78)

    return "\n".join(lines)
