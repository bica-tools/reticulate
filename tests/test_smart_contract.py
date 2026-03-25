"""Tests for smart contract lifecycle verification (Step 81).

Verifies that all smart contract lifecycle workflows parse correctly,
produce valid state spaces, satisfy lattice properties, and generate
conformance test suites and runtime monitors.
"""

import pytest

from reticulate.parser import parse, End
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.smart_contract import (
    ALL_CONTRACT_WORKFLOWS,
    ContractAnalysisResult,
    SmartContractState,
    SmartContractWorkflow,
    auction_contract,
    contract_to_session_type,
    defi_lending,
    erc20_lifecycle,
    erc721_lifecycle,
    format_contract_report,
    format_contract_summary,
    multisig_wallet,
    solidity_to_session_type,
    verify_all_contracts,
    verify_contract,
)


# ---------------------------------------------------------------------------
# Workflow construction
# ---------------------------------------------------------------------------


class TestWorkflowConstruction:
    """Test that workflow constructors return valid SmartContractWorkflow objects."""

    def test_erc20_is_workflow(self):
        wf = erc20_lifecycle()
        assert isinstance(wf, SmartContractWorkflow)
        assert wf.name == "ERC20Token"
        assert wf.standard == "ERC-20"

    def test_erc721_is_workflow(self):
        wf = erc721_lifecycle()
        assert isinstance(wf, SmartContractWorkflow)
        assert wf.name == "ERC721Token"
        assert wf.standard == "ERC-721"

    def test_defi_lending_is_workflow(self):
        wf = defi_lending()
        assert isinstance(wf, SmartContractWorkflow)
        assert wf.name == "DeFiLending"
        assert wf.standard == "DeFi-Lending"

    def test_multisig_is_workflow(self):
        wf = multisig_wallet()
        assert isinstance(wf, SmartContractWorkflow)
        assert wf.name == "MultisigWallet"
        assert wf.standard == "Multisig"

    def test_auction_is_workflow(self):
        wf = auction_contract()
        assert isinstance(wf, SmartContractWorkflow)
        assert wf.name == "Auction"
        assert wf.standard == "Auction"

    def test_all_workflows_registry(self):
        assert len(ALL_CONTRACT_WORKFLOWS) == 5
        names = {wf.name for wf in ALL_CONTRACT_WORKFLOWS}
        assert "ERC20Token" in names
        assert "ERC721Token" in names
        assert "DeFiLending" in names
        assert "MultisigWallet" in names
        assert "Auction" in names


# ---------------------------------------------------------------------------
# SmartContractState
# ---------------------------------------------------------------------------


class TestSmartContractState:
    """Test SmartContractState dataclass."""

    def test_state_creation(self):
        s = SmartContractState("Active", ("transfer", "approve"))
        assert s.name == "Active"
        assert s.allowed_transitions == ("transfer", "approve")
        assert s.requires_auth is False

    def test_state_with_auth(self):
        s = SmartContractState("Deployed", ("mint",), requires_auth=True)
        assert s.requires_auth is True

    def test_terminal_state(self):
        s = SmartContractState("Burned", ())
        assert s.allowed_transitions == ()

    def test_state_is_frozen(self):
        s = SmartContractState("Active", ("transfer",))
        with pytest.raises(AttributeError):
            s.name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Session type parsing
# ---------------------------------------------------------------------------


class TestSessionTypeParsing:
    """Verify all contract session types parse correctly."""

    @pytest.mark.parametrize("wf", ALL_CONTRACT_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_parses(self, wf: SmartContractWorkflow):
        ast = parse(wf.session_type_string)
        assert ast is not None

    @pytest.mark.parametrize("wf", ALL_CONTRACT_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_builds_statespace(self, wf: SmartContractWorkflow):
        ast = parse(wf.session_type_string)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2  # at least top and bottom

    @pytest.mark.parametrize("wf", ALL_CONTRACT_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_has_description(self, wf: SmartContractWorkflow):
        assert len(wf.description) > 20

    @pytest.mark.parametrize("wf", ALL_CONTRACT_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_has_states(self, wf: SmartContractWorkflow):
        assert len(wf.states) >= 2


# ---------------------------------------------------------------------------
# Lattice properties
# ---------------------------------------------------------------------------


class TestLatticeProperties:
    """Verify all contract lifecycles form lattices."""

    @pytest.mark.parametrize("wf", ALL_CONTRACT_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_is_lattice(self, wf: SmartContractWorkflow):
        ast = parse(wf.session_type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, f"{wf.name} does not form a lattice: {lr.counterexample}"

    @pytest.mark.parametrize("wf", ALL_CONTRACT_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_has_top_and_bottom(self, wf: SmartContractWorkflow):
        ast = parse(wf.session_type_string)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.has_top
        assert lr.has_bottom


# ---------------------------------------------------------------------------
# Verification pipeline
# ---------------------------------------------------------------------------


class TestVerificationPipeline:
    """Test the full verify_contract pipeline."""

    def test_verify_erc20(self):
        result = verify_contract(erc20_lifecycle())
        assert isinstance(result, ContractAnalysisResult)
        assert result.is_well_formed
        assert result.num_states >= 2
        assert result.num_transitions >= 1
        assert result.num_valid_paths >= 1

    def test_verify_erc721(self):
        result = verify_contract(erc721_lifecycle())
        assert result.is_well_formed
        assert result.num_valid_paths >= 1

    def test_verify_defi_lending(self):
        result = verify_contract(defi_lending())
        assert result.is_well_formed
        assert result.num_valid_paths >= 1

    def test_verify_multisig(self):
        result = verify_contract(multisig_wallet())
        assert result.is_well_formed

    def test_verify_auction(self):
        result = verify_contract(auction_contract())
        assert result.is_well_formed
        assert result.num_valid_paths >= 1

    def test_verify_all_contracts(self):
        results = verify_all_contracts()
        assert len(results) == 5
        for r in results:
            assert r.is_well_formed, f"{r.workflow.name} failed lattice check"

    def test_coverage_computed(self):
        result = verify_contract(auction_contract())
        assert result.coverage.transition_coverage > 0
        assert result.coverage.state_coverage > 0

    def test_monitor_generated(self):
        result = verify_contract(erc20_lifecycle())
        assert "class" in result.monitor_source.lower() or "def " in result.monitor_source

    def test_test_source_generated(self):
        result = verify_contract(erc20_lifecycle())
        assert "Test" in result.test_source or "test" in result.test_source


# ---------------------------------------------------------------------------
# solidity_to_session_type conversion
# ---------------------------------------------------------------------------


class TestSolidityConversion:
    """Test the solidity_to_session_type converter."""

    def test_empty_inputs(self):
        result = solidity_to_session_type([], [])
        assert result == "end"

    def test_simple_linear(self):
        states = [
            SmartContractState("Init", ("deploy",)),
            SmartContractState("Active", ("transfer",)),
            SmartContractState("Done", ()),
        ]
        transitions = [
            ("Init", "deploy", "Active"),
            ("Active", "transfer", "Done"),
        ]
        result = solidity_to_session_type(states, transitions)
        # Should produce a parseable session type
        ast = parse(result)
        assert ast is not None

    def test_branching(self):
        states = [
            SmartContractState("Init", ("approve", "reject")),
            SmartContractState("Approved", ()),
            SmartContractState("Rejected", ()),
        ]
        transitions = [
            ("Init", "approve", "Approved"),
            ("Init", "reject", "Rejected"),
        ]
        result = solidity_to_session_type(states, transitions)
        ast = parse(result)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestReportFormatting:
    """Test report and summary formatting."""

    def test_format_report_contains_name(self):
        result = verify_contract(erc20_lifecycle())
        report = format_contract_report(result)
        assert "ERC20Token" in report
        assert "ERC-20" in report

    def test_format_report_contains_verdict(self):
        result = verify_contract(auction_contract())
        report = format_contract_report(result)
        assert "VERDICT" in report
        assert "WELL-FORMED" in report

    def test_format_report_contains_events(self):
        result = verify_contract(erc20_lifecycle())
        report = format_contract_report(result)
        assert "Transfer" in report
        assert "Approval" in report

    def test_format_summary(self):
        results = verify_all_contracts()
        summary = format_contract_summary(results)
        assert "VERIFICATION SUMMARY" in summary
        assert "5/5" in summary

    def test_format_report_contains_lattice_info(self):
        result = verify_contract(defi_lending())
        report = format_contract_report(result)
        assert "Lattice" in report
        assert "States" in report


# ---------------------------------------------------------------------------
# contract_to_session_type
# ---------------------------------------------------------------------------


class TestContractToSessionType:
    """Test AST conversion from workflows."""

    def test_contract_to_ast(self):
        wf = erc20_lifecycle()
        ast = contract_to_session_type(wf)
        assert ast is not None
        assert not isinstance(ast, End)

    def test_all_contracts_convert(self):
        for wf in ALL_CONTRACT_WORKFLOWS:
            ast = contract_to_session_type(wf)
            assert ast is not None
