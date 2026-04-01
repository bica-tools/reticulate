"""Tests for real-world API session type extraction (Step 71b).

Scientific Method validation:
  Hypothesis: Real-world API specs produce lattice-structured session types
  Prediction: ≥80% of extracted session types form lattices
  Experiment: Extract from 10 real API specs, check lattice properties
"""

import pytest
from reticulate.real_world_extraction import (
    ExtractionResult,
    AnalysisReport,
    extract_and_analyze,
    analyze_all_specs,
    print_report,
    REAL_WORLD_SPECS,
    petstore_spec,
    github_repos_spec,
    stripe_payments_spec,
    twilio_messaging_spec,
    kubernetes_pods_spec,
    docker_containers_spec,
    elasticsearch_index_spec,
    redis_commands_spec,
    auth0_spec,
    s3_objects_spec,
)
from reticulate.importers import from_openapi
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Individual API extraction tests
# ---------------------------------------------------------------------------

class TestPetstore:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("Petstore", petstore_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("Petstore", petstore_spec())
        for r in results:
            assert r.is_lattice, f"Petstore/{r.tag} is not a lattice"

    def test_has_multiple_endpoints(self):
        results = extract_and_analyze("Petstore", petstore_spec())
        for r in results:
            assert r.num_endpoints >= 2, f"Petstore/{r.tag} has too few endpoints"

    def test_session_type_parseable(self):
        types = from_openapi(petstore_spec())
        for tag, st_str in types.items():
            ast = parse(st_str)
            assert ast is not None


class TestGitHub:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("GitHub", github_repos_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("GitHub", github_repos_spec())
        for r in results:
            assert r.is_lattice

    def test_has_states(self):
        results = extract_and_analyze("GitHub", github_repos_spec())
        for r in results:
            assert r.num_states >= 2


class TestStripe:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("Stripe", stripe_payments_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("Stripe", stripe_payments_spec())
        for r in results:
            assert r.is_lattice

    def test_has_payment_endpoints(self):
        results = extract_and_analyze("Stripe", stripe_payments_spec())
        for r in results:
            assert r.num_endpoints >= 3


class TestTwilio:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("Twilio", twilio_messaging_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("Twilio", twilio_messaging_spec())
        for r in results:
            assert r.is_lattice


class TestKubernetes:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("K8s", kubernetes_pods_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("K8s", kubernetes_pods_spec())
        for r in results:
            assert r.is_lattice

    def test_has_many_endpoints(self):
        results = extract_and_analyze("K8s", kubernetes_pods_spec())
        for r in results:
            assert r.num_endpoints >= 4


class TestDocker:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("Docker", docker_containers_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("Docker", docker_containers_spec())
        for r in results:
            assert r.is_lattice


class TestElasticsearch:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("ES", elasticsearch_index_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("ES", elasticsearch_index_spec())
        for r in results:
            assert r.is_lattice


class TestRedis:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("Redis", redis_commands_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("Redis", redis_commands_spec())
        for r in results:
            assert r.is_lattice


class TestAuth0:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("Auth0", auth0_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("Auth0", auth0_spec())
        for r in results:
            assert r.is_lattice


class TestS3:
    def test_extraction_succeeds(self):
        results = extract_and_analyze("S3", s3_objects_spec())
        assert len(results) >= 1

    def test_forms_lattice(self):
        results = extract_and_analyze("S3", s3_objects_spec())
        for r in results:
            assert r.is_lattice


# ---------------------------------------------------------------------------
# Aggregate analysis tests
# ---------------------------------------------------------------------------

class TestAggregateAnalysis:
    def test_all_specs_registered(self):
        assert len(REAL_WORLD_SPECS) >= 10

    def test_analyze_all_runs(self):
        report = analyze_all_specs()
        assert isinstance(report, AnalysisReport)
        assert report.total_apis >= 10

    def test_lattice_rate_above_threshold(self):
        """Core hypothesis: ≥80% of real API session types form lattices."""
        report = analyze_all_specs()
        assert report.lattice_rate >= 0.80, (
            f"Lattice rate {report.lattice_rate:.0%} below 80% threshold"
        )

    def test_all_results_have_states(self):
        report = analyze_all_specs()
        for r in report.results:
            assert r.num_states >= 2, f"{r.api_name}/{r.tag} has < 2 states"

    def test_all_results_parseable(self):
        report = analyze_all_specs()
        for r in report.results:
            assert r.session_type, f"{r.api_name}/{r.tag} has empty session type"

    def test_report_formatting(self):
        report = analyze_all_specs()
        text = print_report(report)
        assert "REAL-WORLD API" in text
        assert "Lattice rate" in text


# ---------------------------------------------------------------------------
# Structural property tests
# ---------------------------------------------------------------------------

class TestStructuralProperties:
    """Verify structural properties of extracted session types."""

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_bounded_lattice(self, name, spec_fn):
        """Every extracted type should have top and bottom states."""
        results = extract_and_analyze(name, spec_fn())
        for r in results:
            if r.is_lattice:
                ast = parse(r.session_type)
                ss = build_statespace(ast)
                assert ss.top is not None
                assert ss.bottom is not None

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_transitions_exist(self, name, spec_fn):
        """Every extracted type should have transitions."""
        results = extract_and_analyze(name, spec_fn())
        for r in results:
            assert r.num_transitions > 0, f"{name}/{r.tag} has no transitions"


# ---------------------------------------------------------------------------
# Round-trip tests: import → parse → build → check
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Verify the full extraction pipeline for each spec."""

    @pytest.mark.parametrize("name,spec_fn", list(REAL_WORLD_SPECS.items()))
    def test_full_pipeline(self, name, spec_fn):
        """import → parse → build_statespace → check_lattice succeeds."""
        spec = spec_fn()
        tag_types = from_openapi(spec)
        assert len(tag_types) >= 1, f"{name} produced no session types"

        for tag, st_str in tag_types.items():
            ast = parse(st_str)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            assert lr.is_lattice, f"{name}/{tag}: not a lattice"
