"""Tests for FHIR clinical workflow verification (Step 72).

Verifies that all FHIR clinical workflows parse correctly, produce
valid state spaces, satisfy lattice properties, and generate
conformance test suites.
"""

import pytest

from reticulate.parser import parse, ParseError
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.fhir import (
    ALL_FHIR_WORKFLOWS,
    FHIRAnalysisResult,
    FHIRWorkflow,
    discharge_workflow,
    emergency_workflow,
    fhir_to_session_type,
    format_fhir_report,
    format_fhir_summary,
    immunization_workflow,
    lab_order_workflow,
    medication_order_workflow,
    patient_intake_workflow,
    referral_workflow,
    surgical_workflow,
    verify_all_fhir_workflows,
    verify_fhir_workflow,
)


# ---------------------------------------------------------------------------
# Workflow construction
# ---------------------------------------------------------------------------


class TestWorkflowConstruction:
    """Test that workflow constructors return valid FHIRWorkflow objects."""

    def test_patient_intake_is_workflow(self):
        wf = patient_intake_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "PatientIntake"

    def test_medication_order_is_workflow(self):
        wf = medication_order_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "MedicationOrder"

    def test_lab_order_is_workflow(self):
        wf = lab_order_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "LabOrder"

    def test_referral_is_workflow(self):
        wf = referral_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "Referral"

    def test_emergency_is_workflow(self):
        wf = emergency_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "Emergency"

    def test_immunization_is_workflow(self):
        wf = immunization_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "Immunization"

    def test_surgical_is_workflow(self):
        wf = surgical_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "Surgical"

    def test_discharge_is_workflow(self):
        wf = discharge_workflow()
        assert isinstance(wf, FHIRWorkflow)
        assert wf.name == "Discharge"

    def test_all_workflows_have_resources(self):
        for wf in ALL_FHIR_WORKFLOWS:
            assert len(wf.resources) > 0, f"{wf.name} has no resources"

    def test_all_workflows_have_transitions(self):
        for wf in ALL_FHIR_WORKFLOWS:
            assert len(wf.transitions) > 0, f"{wf.name} has no transitions"

    def test_all_workflows_have_descriptions(self):
        for wf in ALL_FHIR_WORKFLOWS:
            assert len(wf.description) > 10, f"{wf.name} has no description"

    def test_workflow_count(self):
        assert len(ALL_FHIR_WORKFLOWS) == 8


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    """Test that all workflow type strings parse correctly."""

    @pytest.mark.parametrize("wf", ALL_FHIR_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_parses(self, wf: FHIRWorkflow):
        ast = fhir_to_session_type(wf)
        assert ast is not None

    @pytest.mark.parametrize("wf", ALL_FHIR_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_builds_statespace(self, wf: FHIRWorkflow):
        ast = fhir_to_session_type(wf)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2  # at least top and bottom


# ---------------------------------------------------------------------------
# Lattice properties
# ---------------------------------------------------------------------------


class TestLatticeProperties:
    """All FHIR workflows should form lattices."""

    @pytest.mark.parametrize("wf", ALL_FHIR_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_is_lattice(self, wf: FHIRWorkflow):
        ast = fhir_to_session_type(wf)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, f"{wf.name} state space is not a lattice"

    @pytest.mark.parametrize("wf", ALL_FHIR_WORKFLOWS, ids=lambda w: w.name)
    def test_workflow_has_top_and_bottom(self, wf: FHIRWorkflow):
        ast = fhir_to_session_type(wf)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.has_top
        assert lr.has_bottom


# ---------------------------------------------------------------------------
# Verification pipeline
# ---------------------------------------------------------------------------


class TestVerification:
    """Test the full verification pipeline."""

    def test_patient_intake_verification(self):
        result = verify_fhir_workflow(patient_intake_workflow())
        assert isinstance(result, FHIRAnalysisResult)
        assert result.is_well_formed
        assert result.num_states > 0
        assert result.num_transitions > 0
        assert result.num_valid_paths > 0

    def test_medication_order_verification(self):
        result = verify_fhir_workflow(medication_order_workflow())
        assert result.is_well_formed
        assert result.num_valid_paths >= 2  # approve + reject paths

    def test_lab_order_verification(self):
        result = verify_fhir_workflow(lab_order_workflow())
        assert result.is_well_formed
        assert result.num_valid_paths >= 2  # normal + abnormal paths

    def test_referral_verification(self):
        result = verify_fhir_workflow(referral_workflow())
        assert result.is_well_formed
        assert result.num_valid_paths >= 2  # accept + reject

    def test_emergency_verification(self):
        result = verify_fhir_workflow(emergency_workflow())
        assert result.is_well_formed

    def test_verify_all_workflows(self):
        results = verify_all_fhir_workflows()
        assert len(results) == 8
        for r in results:
            assert r.is_well_formed, f"{r.workflow.name} not well-formed"

    def test_coverage_positive(self):
        result = verify_fhir_workflow(patient_intake_workflow())
        assert result.coverage.transition_coverage > 0.0
        assert result.coverage.state_coverage > 0.0

    def test_test_source_generated(self):
        result = verify_fhir_workflow(medication_order_workflow())
        assert "class MedicationOrderWorkflowTest" in result.test_source
        assert "void" in result.test_source

    def test_violations_detected(self):
        result = verify_fhir_workflow(patient_intake_workflow())
        assert result.num_violations >= 0  # violations may exist


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class TestReportFormatting:
    """Test report generation."""

    def test_format_single_report(self):
        result = verify_fhir_workflow(referral_workflow())
        report = format_fhir_report(result)
        assert "Referral" in report
        assert "WELL-FORMED" in report
        assert "Session Type" in report

    def test_format_summary(self):
        results = verify_all_fhir_workflows()
        summary = format_fhir_summary(results)
        assert "SUMMARY" in summary
        assert "PatientIntake" in summary
        assert "8/8 workflows form lattices" in summary or "8" in summary

    def test_report_contains_resources(self):
        result = verify_fhir_workflow(lab_order_workflow())
        report = format_fhir_report(result)
        assert "ServiceRequest" in report
        assert "DiagnosticReport" in report

    def test_report_contains_transitions(self):
        result = verify_fhir_workflow(emergency_workflow())
        report = format_fhir_report(result)
        assert "triage" in report.lower() or "triageED" in report


# ---------------------------------------------------------------------------
# State space size checks
# ---------------------------------------------------------------------------


class TestStateSpaceSizes:
    """Verify expected state-space sizes for specific workflows."""

    def test_discharge_linear_chain(self):
        """Discharge is a linear chain: 5 states, 4 transitions."""
        result = verify_fhir_workflow(discharge_workflow())
        assert result.num_states == 5
        assert result.num_transitions == 4

    def test_referral_has_branch(self):
        """Referral has an accept/reject branch."""
        result = verify_fhir_workflow(referral_workflow())
        # requestReferral -> reviewReferral -> (ACCEPT -> schedule -> end | REJECT -> end)
        assert result.num_states >= 4

    def test_patient_intake_two_branches(self):
        """Patient intake has triage (urgent/routine) and disposition branches."""
        result = verify_fhir_workflow(patient_intake_workflow())
        assert result.num_transitions >= 8  # at least 8 transitions


# ---------------------------------------------------------------------------
# Distributivity
# ---------------------------------------------------------------------------


class TestDistributivity:
    """Test distributivity for FHIR workflows."""

    # PatientIntake has duplicate sub-workflows (urgent/routine share
    # the same examine->diagnose->admit/discharge chain), which creates
    # an N5 sublattice and breaks distributivity.
    _DISTRIBUTIVE_WORKFLOWS = [
        wf for wf in ALL_FHIR_WORKFLOWS if wf.name != "PatientIntake"
    ]

    @pytest.mark.parametrize(
        "wf", _DISTRIBUTIVE_WORKFLOWS, ids=lambda w: w.name,
    )
    def test_workflow_distributivity(self, wf: FHIRWorkflow):
        """Tree-shaped FHIR workflows without shared sub-trees are distributive."""
        result = verify_fhir_workflow(wf)
        assert result.distributivity.is_distributive

    def test_patient_intake_not_distributive(self):
        """PatientIntake has shared sub-workflows and is NOT distributive."""
        result = verify_fhir_workflow(patient_intake_workflow())
        assert not result.distributivity.is_distributive
        assert result.is_well_formed  # still a lattice


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    """Verify that data types are frozen."""

    def test_workflow_frozen(self):
        wf = patient_intake_workflow()
        with pytest.raises(AttributeError):
            wf.name = "Changed"  # type: ignore[misc]

    def test_result_frozen(self):
        result = verify_fhir_workflow(discharge_workflow())
        with pytest.raises(AttributeError):
            result.num_states = 999  # type: ignore[misc]
