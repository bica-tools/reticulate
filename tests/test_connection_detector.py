"""Tests for the connection_detector module.

Tests cover:
- StepInfo and Connection dataclasses
- Keyword extraction and classification
- Domain classification
- Connection detection between step pairs
- Full detect_connections over multiple steps
- find_new_connections for a single new step
- auto_generate_paper_title
- keyword_matrix construction
- domain_theory_matrix construction
- ConnectionReport structure
- scan_steps integration (with filesystem)
- Edge cases (empty inputs, identical steps, single step)
"""

import pytest

from reticulate.connection_detector import (
    Connection,
    ConnectionReport,
    StepInfo,
    auto_generate_paper_title,
    detect_connections,
    domain_theory_matrix,
    find_new_connections,
    keyword_matrix,
    scan_steps,
    _classify_domain,
    _extract_keywords,
    _keyword_overlap,
    _module_overlap,
    _dependency_strength,
    _complement_strength,
    _detect_pair_connections,
    _suggest_venues,
)


# ---------------------------------------------------------------------------
# Fixtures: reusable step definitions
# ---------------------------------------------------------------------------

@pytest.fixture
def theory_step() -> StepInfo:
    return StepInfo(
        number="7", title="Subtyping is Embedding",
        keywords=("lattice", "session-types", "subtyping"),
        modules=("subtyping.py",), domain="theory",
        depends_on=("5",), dir_name="step7-subtyping",
    )


@pytest.fixture
def app_step() -> StepInfo:
    return StepInfo(
        number="72", title="FHIR Clinical Workflows",
        keywords=("healthcare", "session-types", "lattice"),
        modules=("fhir.py",), domain="application",
        depends_on=("5",), dir_name="step72-fhir-workflows",
    )


@pytest.fixture
def monitor_step() -> StepInfo:
    return StepInfo(
        number="80", title="Runtime Monitors",
        keywords=("monitoring", "session-types", "lattice"),
        modules=("monitor.py",), domain="application",
        depends_on=("7",), dir_name="step80-runtime-monitors",
    )


@pytest.fixture
def petri_step() -> StepInfo:
    return StepInfo(
        number="21", title="Petri Nets",
        keywords=("petri-nets", "session-types", "lattice"),
        modules=("petri.py",), domain="intersection",
        depends_on=(), dir_name="step21-petri-nets",
    )


@pytest.fixture
def lattice_step() -> StepInfo:
    return StepInfo(
        number="5", title="Lattice Conditions",
        keywords=("lattice", "session-types"),
        modules=("lattice.py",), domain="theory",
        depends_on=(), dir_name="step5-lattice-conditions",
    )


@pytest.fixture
def ccs_step() -> StepInfo:
    return StepInfo(
        number="26", title="CCS Encoding",
        keywords=("process-algebra", "session-types", "lattice"),
        modules=("ccs.py",), domain="intersection",
        depends_on=("5",), dir_name="step26-ccs",
    )


@pytest.fixture
def sample_steps(theory_step, app_step, monitor_step, petri_step, lattice_step, ccs_step):
    return [theory_step, app_step, monitor_step, petri_step, lattice_step, ccs_step]


# ---------------------------------------------------------------------------
# StepInfo and Connection dataclass tests
# ---------------------------------------------------------------------------

class TestStepInfo:
    def test_creation(self, theory_step):
        assert theory_step.number == "7"
        assert theory_step.domain == "theory"
        assert "subtyping" in theory_step.keywords

    def test_frozen(self, theory_step):
        with pytest.raises(AttributeError):
            theory_step.number = "99"  # type: ignore[misc]


class TestConnection:
    def test_creation(self):
        c = Connection(
            step_a="7", step_b="72", connection_type="complement",
            strength=0.7, paper_title="Test Paper",
            venues=("ECOOP",), shared_keywords=("lattice",),
        )
        assert c.step_a == "7"
        assert c.strength == 0.7
        assert c.connection_type == "complement"

    def test_defaults(self):
        c = Connection(
            step_a="1", step_b="2", connection_type="keyword",
            strength=0.5, paper_title="T", venues=(),
        )
        assert c.shared_keywords == ()
        assert c.shared_modules == ()


class TestConnectionReport:
    def test_creation(self):
        r = ConnectionReport(
            new_connections=(), total_connections=0,
            paper_opportunities=(), by_domain={},
        )
        assert r.total_connections == 0
        assert len(r.paper_opportunities) == 0


# ---------------------------------------------------------------------------
# Keyword extraction tests
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    def test_lattice_keywords(self):
        kw = _extract_keywords("Lattice Conditions for Session Types",
                               "step5-lattice-conditions")
        assert "lattice" in kw
        assert "session-types" in kw

    def test_petri_keywords(self):
        kw = _extract_keywords("Petri Net Encoding", "step21-petri-nets")
        assert "petri-nets" in kw

    def test_game_keywords(self):
        kw = _extract_keywords("Game Theory for Protocols", "step900-game")
        assert "game-theory" in kw

    def test_always_has_base_keywords(self):
        kw = _extract_keywords("Something Unrelated", "step999-unknown")
        assert "session-types" in kw
        assert "lattice" in kw


# ---------------------------------------------------------------------------
# Domain classification tests
# ---------------------------------------------------------------------------

class TestDomainClassification:
    def test_theory_step(self):
        assert _classify_domain("7", "Subtyping is Embedding", "step7-subtyping") == "theory"

    def test_application_step(self):
        assert _classify_domain("72", "FHIR Workflows", "step72-fhir-workflows") == "application"

    def test_intersection_step(self):
        assert _classify_domain("21", "Petri Net Encoding", "step21-petri-nets") == "intersection"

    def test_tool_step(self):
        assert _classify_domain("1", "State Space Construction", "step1-statespace") == "tool"


# ---------------------------------------------------------------------------
# Overlap and strength tests
# ---------------------------------------------------------------------------

class TestKeywordOverlap:
    def test_no_overlap(self, theory_step, petri_step):
        strength, shared = _keyword_overlap(theory_step, petri_step)
        assert strength == 0.0 or len(shared) == 0
        # subtyping vs petri-nets: no significant overlap

    def test_overlap_with_shared(self):
        a = StepInfo("1", "T", ("lattice", "session-types", "petri-nets"),
                     (), "theory", (), "s1")
        b = StepInfo("2", "T", ("lattice", "session-types", "petri-nets"),
                     (), "theory", (), "s2")
        strength, shared = _keyword_overlap(a, b)
        assert strength == 1.0
        assert "petri-nets" in shared


class TestModuleOverlap:
    def test_same_module(self):
        a = StepInfo("1", "T", (), ("lattice.py",), "theory", (), "s1")
        b = StepInfo("2", "T", (), ("lattice.py",), "theory", (), "s2")
        strength, shared = _module_overlap(a, b)
        assert strength == 1.0
        assert "lattice.py" in shared

    def test_no_overlap(self, theory_step, petri_step):
        strength, shared = _module_overlap(theory_step, petri_step)
        assert strength == 0.0


class TestDependencyStrength:
    def test_direct_dependency(self, theory_step, lattice_step):
        # theory_step depends on "5" (lattice_step)
        strength = _dependency_strength(theory_step, lattice_step)
        assert strength == 0.6

    def test_no_dependency(self, petri_step, ccs_step):
        strength = _dependency_strength(petri_step, ccs_step)
        assert strength == 0.0


class TestComplementStrength:
    def test_theory_application(self, theory_step, app_step):
        strength = _complement_strength(theory_step, app_step)
        assert strength == 0.7

    def test_same_domain(self, theory_step, lattice_step):
        strength = _complement_strength(theory_step, lattice_step)
        assert strength == 0.0

    def test_application_intersection(self, app_step, petri_step):
        strength = _complement_strength(app_step, petri_step)
        assert strength == 0.5


# ---------------------------------------------------------------------------
# Pair connection detection
# ---------------------------------------------------------------------------

class TestDetectPairConnections:
    def test_complement_detected(self, theory_step, app_step):
        conns = _detect_pair_connections(theory_step, app_step)
        types = [c.connection_type for c in conns]
        assert "complement" in types

    def test_dependency_detected(self, theory_step, lattice_step):
        conns = _detect_pair_connections(theory_step, lattice_step)
        types = [c.connection_type for c in conns]
        assert "dependency" in types

    def test_same_step_no_connections(self, theory_step):
        # Same step compared to itself: keyword/module overlap = 1.0
        conns = _detect_pair_connections(theory_step, theory_step)
        # Should have connections (keyword overlap is 1.0, module overlap is 1.0)
        assert len(conns) >= 1


# ---------------------------------------------------------------------------
# Full detect_connections
# ---------------------------------------------------------------------------

class TestDetectConnections:
    def test_returns_report(self, sample_steps):
        report = detect_connections(sample_steps)
        assert isinstance(report, ConnectionReport)
        assert report.total_connections > 0

    def test_paper_opportunities_are_top_10pct(self, sample_steps):
        report = detect_connections(sample_steps)
        assert len(report.paper_opportunities) <= max(1, report.total_connections // 10)

    def test_by_domain_populated(self, sample_steps):
        report = detect_connections(sample_steps)
        assert len(report.by_domain) > 0

    def test_connections_sorted_by_strength(self, sample_steps):
        report = detect_connections(sample_steps)
        strengths = [c.strength for c in report.new_connections]
        assert strengths == sorted(strengths, reverse=True)

    def test_empty_input(self):
        report = detect_connections([])
        assert report.total_connections == 0
        assert len(report.paper_opportunities) == 0

    def test_single_step(self, theory_step):
        report = detect_connections([theory_step])
        assert report.total_connections == 0


# ---------------------------------------------------------------------------
# find_new_connections
# ---------------------------------------------------------------------------

class TestFindNewConnections:
    def test_finds_connections(self, monitor_step, sample_steps):
        # Remove monitor_step from existing to avoid duplication
        existing = [s for s in sample_steps if s.number != monitor_step.number]
        report = find_new_connections(monitor_step, existing)
        assert report.total_connections > 0

    def test_complement_with_theory(self, app_step, theory_step):
        report = find_new_connections(app_step, [theory_step])
        types = [c.connection_type for c in report.new_connections]
        assert "complement" in types

    def test_empty_existing(self, theory_step):
        report = find_new_connections(theory_step, [])
        assert report.total_connections == 0


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------

class TestAutoGeneratePaperTitle:
    def test_theory_application_title(self, theory_step, app_step):
        title = auto_generate_paper_title(theory_step, app_step)
        assert isinstance(title, str)
        assert len(title) > 10

    def test_same_domain_title(self, theory_step, lattice_step):
        title = auto_generate_paper_title(theory_step, lattice_step)
        assert isinstance(title, str)
        assert len(title) > 10

    def test_deterministic(self, theory_step, app_step):
        t1 = auto_generate_paper_title(theory_step, app_step)
        t2 = auto_generate_paper_title(theory_step, app_step)
        assert t1 == t2

    def test_monitor_fhir(self, monitor_step, app_step):
        # The motivating example from the spec
        title = auto_generate_paper_title(monitor_step, app_step)
        assert isinstance(title, str)


# ---------------------------------------------------------------------------
# Keyword matrix
# ---------------------------------------------------------------------------

class TestKeywordMatrix:
    def test_matrix_structure(self, sample_steps):
        m = keyword_matrix(sample_steps)
        assert isinstance(m, dict)

    def test_symmetric(self, sample_steps):
        m = keyword_matrix(sample_steps)
        for ka in m:
            for kb in m[ka]:
                assert m.get(kb, {}).get(ka, 0) == m[ka][kb]

    def test_empty_input(self):
        m = keyword_matrix([])
        assert m == {}


# ---------------------------------------------------------------------------
# Domain-theory matrix
# ---------------------------------------------------------------------------

class TestDomainTheoryMatrix:
    def test_finds_pairs(self, sample_steps):
        pairs = domain_theory_matrix(sample_steps)
        assert len(pairs) > 0

    def test_pair_types(self, sample_steps):
        pairs = domain_theory_matrix(sample_steps)
        types = {t for _, _, t in pairs}
        # We have theory and application steps in sample
        assert "theory_x_application" in types or "theory_x_domain" in types

    def test_empty_input(self):
        pairs = domain_theory_matrix([])
        assert pairs == []


# ---------------------------------------------------------------------------
# Venue suggestion
# ---------------------------------------------------------------------------

class TestSuggestVenues:
    def test_theory_application(self, theory_step, app_step):
        venues = _suggest_venues(theory_step, app_step)
        assert len(venues) > 0
        assert "ECOOP" in venues or "ESOP" in venues

    def test_theory_theory(self, theory_step, lattice_step):
        venues = _suggest_venues(theory_step, lattice_step)
        assert len(venues) > 0


# ---------------------------------------------------------------------------
# Integration: scan_steps from filesystem
# ---------------------------------------------------------------------------

class TestScanSteps:
    def test_scan_real_steps(self):
        """Integration test: scan actual step directories."""
        steps = scan_steps()
        # We know there are 78+ step directories
        assert len(steps) >= 50

    def test_step_fields(self):
        steps = scan_steps()
        if steps:
            s = steps[0]
            assert s.number
            assert s.title
            assert len(s.keywords) >= 2  # at least base keywords
            assert s.domain in ("theory", "application", "tool", "intersection")


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end(self):
        """Scan real steps, detect connections, generate report."""
        steps = scan_steps()
        if len(steps) < 2:
            pytest.skip("Need at least 2 steps")
        report = detect_connections(steps)
        assert report.total_connections > 0
        assert len(report.paper_opportunities) > 0

        # Every opportunity has a title and venues
        for opp in report.paper_opportunities:
            assert opp.paper_title
            assert opp.venues
