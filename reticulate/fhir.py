"""FHIR clinical workflow verification via session types (Step 72).

Models HL7 FHIR clinical workflows as session types, enabling formal
verification of healthcare protocol correctness through lattice analysis,
conformance testing, and coverage computation.

FHIR (Fast Healthcare Interoperability Resources) defines RESTful APIs
for clinical data exchange.  Clinical workflows — patient intake,
medication ordering, lab orders, referrals, emergency care — follow
strict state-machine protocols mandated by healthcare regulations.

This module encodes each workflow as a session type, builds the
corresponding state space (reticulate), checks lattice properties,
and generates conformance test suites.  The key insight is that
well-designed clinical workflows naturally form lattices: every pair
of workflow states has a well-defined join (least common continuation)
and meet (greatest common predecessor), ensuring unambiguous protocol
recovery from any reachable state.

Usage:
    from reticulate.fhir import (
        patient_intake_workflow,
        verify_fhir_workflow,
        format_fhir_report,
    )
    wf = patient_intake_workflow()
    result = verify_fhir_workflow(wf)
    print(format_fhir_report(result))
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
class FHIRWorkflow:
    """A named FHIR clinical workflow modelled as a session type.

    Attributes:
        name: Workflow name (e.g., "PatientIntake").
        resources: FHIR resources involved (e.g., ["Patient", "Encounter"]).
        transitions: Human-readable transition descriptions.
        session_type_string: Session type encoding of the workflow.
        description: Free-text description of the workflow purpose.
    """
    name: str
    resources: tuple[str, ...]
    transitions: tuple[str, ...]
    session_type_string: str
    description: str


@dataclass(frozen=True)
class FHIRAnalysisResult:
    """Complete analysis result for a FHIR workflow.

    Attributes:
        workflow: The analysed workflow definition.
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
    workflow: FHIRWorkflow
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


# ---------------------------------------------------------------------------
# Workflow definitions
# ---------------------------------------------------------------------------

def patient_intake_workflow() -> FHIRWorkflow:
    """Patient registration, triage, examination, and diagnosis.

    Models the clinical intake workflow where a patient is registered,
    triaged (urgent vs routine), examined, diagnosed, and either
    admitted or discharged.

    FHIR resources: Patient, Encounter, Condition, Observation.
    """
    return FHIRWorkflow(
        name="PatientIntake",
        resources=("Patient", "Encounter", "Condition", "Observation"),
        transitions=(
            "register: Create Patient resource",
            "triage: Assess urgency (urgent/routine)",
            "examine: Record Observations",
            "diagnose: Create Condition resource",
            "admit: Set Encounter status to in-progress",
            "discharge: Set Encounter status to finished",
        ),
        session_type_string=(
            "&{register: &{triage: +{URGENT: &{examine: &{diagnose: "
            "+{ADMIT: end, DISCHARGE: end}}}, "
            "ROUTINE: &{examine: &{diagnose: "
            "+{ADMIT: end, DISCHARGE: end}}}}}}"
        ),
        description=(
            "Patient intake workflow: registration, triage assessment "
            "(urgent or routine), clinical examination, diagnosis, and "
            "disposition (admit or discharge)."
        ),
    )


def medication_order_workflow() -> FHIRWorkflow:
    """Medication prescribing, review, dispensing, and administration.

    Models the closed-loop medication management process from
    prescribing through pharmacy review to dispensing and
    administration.

    FHIR resources: MedicationRequest, MedicationDispense,
    MedicationAdministration.
    """
    return FHIRWorkflow(
        name="MedicationOrder",
        resources=(
            "MedicationRequest", "MedicationDispense",
            "MedicationAdministration",
        ),
        transitions=(
            "prescribe: Create MedicationRequest",
            "reviewRx: Pharmacy reviews prescription",
            "APPROVED: Prescription approved",
            "REJECTED: Prescription rejected",
            "dispense: Create MedicationDispense",
            "administer: Create MedicationAdministration",
        ),
        session_type_string=(
            "&{prescribe: &{reviewRx: +{APPROVED: &{dispense: "
            "&{administer: end}}, REJECTED: end}}}"
        ),
        description=(
            "Medication order workflow: prescribe, pharmacy review "
            "(approve or reject), dispense, and administer."
        ),
    )


def lab_order_workflow() -> FHIRWorkflow:
    """Lab order, specimen collection, processing, reporting, review.

    Models the laboratory workflow from order entry through
    specimen collection, processing, result reporting, and
    clinician review.

    FHIR resources: ServiceRequest, Specimen, DiagnosticReport,
    Observation.
    """
    return FHIRWorkflow(
        name="LabOrder",
        resources=(
            "ServiceRequest", "Specimen", "DiagnosticReport",
            "Observation",
        ),
        transitions=(
            "order: Create ServiceRequest",
            "collect: Record Specimen collection",
            "process: Lab processes specimen",
            "report: Create DiagnosticReport",
            "reviewResult: Clinician reviews report",
            "NORMAL: Results within normal range",
            "ABNORMAL: Results outside normal range",
        ),
        session_type_string=(
            "&{order: &{collect: &{process: &{report: "
            "&{reviewResult: +{NORMAL: end, ABNORMAL: end}}}}}}"
        ),
        description=(
            "Laboratory order workflow: order, specimen collection, "
            "processing, result reporting, and clinician review "
            "with normal/abnormal disposition."
        ),
    )


def referral_workflow() -> FHIRWorkflow:
    """Referral request, review, accept/reject, and scheduling.

    Models the clinical referral process where a referring provider
    submits a referral, the specialist reviews it, accepts or
    rejects, and if accepted, schedules the appointment.

    FHIR resources: ServiceRequest, Appointment, Task.
    """
    return FHIRWorkflow(
        name="Referral",
        resources=("ServiceRequest", "Appointment", "Task"),
        transitions=(
            "requestReferral: Create ServiceRequest (referral)",
            "reviewReferral: Specialist reviews referral",
            "ACCEPT: Referral accepted",
            "REJECT: Referral rejected",
            "schedule: Create Appointment",
        ),
        session_type_string=(
            "&{requestReferral: &{reviewReferral: "
            "+{ACCEPT: &{schedule: end}, REJECT: end}}}"
        ),
        description=(
            "Clinical referral workflow: request, specialist review, "
            "accept or reject, and appointment scheduling."
        ),
    )


def emergency_workflow() -> FHIRWorkflow:
    """Emergency triage, assessment, treatment, admit/discharge.

    Models the emergency department workflow with triage,
    clinical assessment, treatment, and disposition.

    FHIR resources: Encounter, Condition, Procedure, Observation.
    """
    return FHIRWorkflow(
        name="Emergency",
        resources=("Encounter", "Condition", "Procedure", "Observation"),
        transitions=(
            "triageED: Emergency triage assessment",
            "assess: Clinical assessment",
            "treat: Perform treatment (Procedure)",
            "ADMIT: Admit to inpatient",
            "DISCHARGE_ED: Discharge from ED",
        ),
        session_type_string=(
            "&{triageED: &{assess: &{treat: "
            "+{ADMIT: end, DISCHARGE_ED: end}}}}"
        ),
        description=(
            "Emergency department workflow: triage, assessment, "
            "treatment, and disposition (admit or discharge)."
        ),
    )


def immunization_workflow() -> FHIRWorkflow:
    """Immunization screening, administration, and observation.

    Models the vaccination workflow including eligibility screening,
    consent, administration, and post-vaccination observation.

    FHIR resources: Immunization, ImmunizationRecommendation,
    Consent, Observation.
    """
    return FHIRWorkflow(
        name="Immunization",
        resources=(
            "Immunization", "ImmunizationRecommendation",
            "Consent", "Observation",
        ),
        transitions=(
            "screen: Check immunization eligibility",
            "ELIGIBLE: Patient is eligible",
            "INELIGIBLE: Patient is not eligible",
            "consent: Obtain patient consent",
            "vaccinate: Administer vaccine (Immunization)",
            "observe: Post-vaccination observation period",
        ),
        session_type_string=(
            "&{screen: +{ELIGIBLE: &{consent: &{vaccinate: "
            "&{observe: end}}}, INELIGIBLE: end}}"
        ),
        description=(
            "Immunization workflow: eligibility screening, consent, "
            "vaccine administration, and post-vaccination observation."
        ),
    )


def surgical_workflow() -> FHIRWorkflow:
    """Surgical workflow: consent, prep, operate, recovery.

    Models the perioperative workflow from surgical consent through
    pre-operative preparation, the procedure itself, and
    post-operative recovery.

    FHIR resources: Consent, Procedure, Encounter, Observation.
    """
    return FHIRWorkflow(
        name="Surgical",
        resources=("Consent", "Procedure", "Encounter", "Observation"),
        transitions=(
            "surgicalConsent: Obtain surgical consent",
            "prep: Pre-operative preparation",
            "operate: Perform surgical procedure",
            "recovery: Post-operative recovery",
            "COMPLICATIONS: Complications detected",
            "STABLE: Patient is stable",
        ),
        session_type_string=(
            "&{surgicalConsent: &{prep: &{operate: "
            "&{recovery: +{STABLE: end, COMPLICATIONS: end}}}}}"
        ),
        description=(
            "Surgical workflow: informed consent, pre-operative "
            "preparation, procedure, and recovery with complication "
            "monitoring."
        ),
    )


def discharge_workflow() -> FHIRWorkflow:
    """Discharge planning: summary, instructions, follow-up.

    Models the hospital discharge process including discharge
    summary creation, patient instructions, medication
    reconciliation, and follow-up scheduling.

    FHIR resources: Encounter, CarePlan, MedicationRequest,
    Appointment.
    """
    return FHIRWorkflow(
        name="Discharge",
        resources=("Encounter", "CarePlan", "MedicationRequest", "Appointment"),
        transitions=(
            "createSummary: Create discharge summary",
            "reconcileMeds: Medication reconciliation",
            "instructions: Provide patient instructions",
            "scheduleFU: Schedule follow-up appointment",
        ),
        session_type_string=(
            "&{createSummary: &{reconcileMeds: "
            "&{instructions: &{scheduleFU: end}}}}"
        ),
        description=(
            "Discharge workflow: summary creation, medication "
            "reconciliation, patient instructions, and follow-up "
            "scheduling."
        ),
    )


# ---------------------------------------------------------------------------
# Registry of all workflows
# ---------------------------------------------------------------------------

ALL_FHIR_WORKFLOWS: tuple[FHIRWorkflow, ...] = (
    patient_intake_workflow(),
    medication_order_workflow(),
    lab_order_workflow(),
    referral_workflow(),
    emergency_workflow(),
    immunization_workflow(),
    surgical_workflow(),
    discharge_workflow(),
)
"""All pre-defined FHIR clinical workflows."""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def fhir_to_session_type(workflow: FHIRWorkflow) -> SessionType:
    """Parse the workflow's session type string into an AST."""
    return parse(workflow.session_type_string)


def verify_fhir_workflow(
    workflow: FHIRWorkflow,
    config: TestGenConfig | None = None,
) -> FHIRAnalysisResult:
    """Run the full verification pipeline on a FHIR workflow.

    Parses the workflow's session type, builds the state space,
    checks lattice properties, generates conformance tests, and
    computes coverage.

    Args:
        workflow: The FHIR workflow to verify.
        config: Optional test generation configuration.

    Returns:
        A complete FHIRAnalysisResult.
    """
    if config is None:
        config = TestGenConfig(
            class_name=f"{workflow.name}WorkflowTest",
            package_name="com.fhir.conformance",
        )

    # 1. Parse and build state space
    ast = fhir_to_session_type(workflow)
    ss = build_statespace(ast)

    # 2. Lattice analysis
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    # 3. Test generation
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, workflow.session_type_string)

    # 4. Coverage
    cov = compute_coverage(ss, result=enum)

    return FHIRAnalysisResult(
        workflow=workflow,
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


def verify_all_fhir_workflows() -> list[FHIRAnalysisResult]:
    """Verify all pre-defined FHIR workflows and return results."""
    return [verify_fhir_workflow(wf) for wf in ALL_FHIR_WORKFLOWS]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_fhir_report(result: FHIRAnalysisResult) -> str:
    """Format a FHIRAnalysisResult as structured text for terminal output."""
    lines: list[str] = []
    wf = result.workflow

    lines.append("=" * 70)
    lines.append(f"  FHIR WORKFLOW REPORT: {wf.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {wf.description}")
    lines.append("")

    # Resources
    lines.append("--- FHIR Resources ---")
    for r in wf.resources:
        lines.append(f"  - {r}")
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

    # Transitions
    lines.append("--- Workflow Transitions ---")
    for t in wf.transitions:
        lines.append(f"  - {t}")
    lines.append("")

    # Verdict
    lines.append("=" * 70)
    if result.is_well_formed:
        lines.append(f"  VERDICT: {wf.name} workflow is WELL-FORMED")
        lines.append(f"  The state space forms a bounded lattice.")
        lines.append(f"  {result.num_valid_paths} conformance tests generated.")
    else:
        lines.append(f"  VERDICT: {wf.name} workflow has ISSUES")
        lines.append(f"  WARNING: State space does NOT form a lattice!")
        if result.lattice_result.counterexample:
            lines.append(
                f"  Counterexample: {result.lattice_result.counterexample}"
            )
    lines.append("=" * 70)

    return "\n".join(lines)


def format_fhir_summary(results: list[FHIRAnalysisResult]) -> str:
    """Format a summary table of multiple FHIR workflow analyses."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  FHIR WORKFLOW VERIFICATION SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"  {'Workflow':<20s} {'States':>6s} {'Trans':>6s} "
        f"{'Lattice':>8s} {'Dist':>5s} {'Paths':>6s} {'Viols':>6s}"
    )
    lines.append("  " + "-" * 58)

    for r in results:
        latt = "YES" if r.is_well_formed else "NO"
        dist = "yes" if r.distributivity.is_distributive else "no"
        lines.append(
            f"  {r.workflow.name:<20s} {r.num_states:>6d} "
            f"{r.num_transitions:>6d} {latt:>8s} {dist:>5s} "
            f"{r.num_valid_paths:>6d} {r.num_violations:>6d}"
        )

    lines.append("")
    total_wf = len(results)
    total_lattice = sum(1 for r in results if r.is_well_formed)
    lines.append(f"  {total_lattice}/{total_wf} workflows form lattices.")
    lines.append("=" * 78)

    return "\n".join(lines)
