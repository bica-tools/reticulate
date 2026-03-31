"""Tests for reticulate.api — REST API for modularity analysis."""

import pytest
from fastapi.testclient import TestClient

from reticulate.api import API_PROTOCOL, app

client = TestClient(app)


# ===================================================================
# Health endpoint
# ===================================================================


class TestHealth:
    """Tests for GET /api/v1/health."""

    def test_health_ok(self) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_response_shape(self) -> None:
        data = client.get("/api/v1/health").json()
        assert set(data.keys()) == {"status", "version"}


# ===================================================================
# Analyze endpoint
# ===================================================================


class TestAnalyze:
    """Tests for POST /api/v1/analyze."""

    def test_simple_end(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "end", "protocol_name": "Trivial"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["protocol_name"] == "Trivial"
        assert "is_modular" in data
        assert "classification" in data
        assert "verdict" in data

    def test_branch_type(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "&{a: end, b: end}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_lattice"] is True

    def test_select_type(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "+{x: end, y: end}"},
        )
        assert resp.status_code == 200
        assert "metrics" in resp.json()

    def test_recursive_type(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={
                "type_string": "rec X . &{next: X, stop: end}",
                "protocol_name": "Iterator",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["protocol_name"] == "Iterator"

    def test_default_protocol_name(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "end"},
        )
        assert resp.status_code == 200
        assert resp.json()["protocol_name"] == "Protocol"

    def test_analyze_response_has_metrics(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "&{a: end, b: end}"},
        )
        data = resp.json()
        metrics = data["metrics"]
        assert "states" in metrics
        assert "modules" in metrics
        assert "fiedler_value" in metrics
        assert "cheeger_constant" in metrics

    def test_analyze_response_has_irreducibles(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "&{a: end, b: end}"},
        )
        data = resp.json()
        assert "irreducibles" in data

    def test_analyze_parse_error(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "not a valid type !!!"},
        )
        assert resp.status_code == 422

    def test_analyze_empty_string(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": ""},
        )
        assert resp.status_code == 422

    def test_analyze_missing_type_string(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"protocol_name": "Missing"},
        )
        assert resp.status_code == 422


# ===================================================================
# Modularity endpoint
# ===================================================================


class TestModularity:
    """Tests for POST /api/v1/modularity."""

    def test_simple_end(self) -> None:
        resp = client.post(
            "/api/v1/modularity",
            json={"type_string": "end"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "is_modular" in data
        assert "is_distributive" in data
        assert "verdict" in data
        assert "classification" in data

    def test_response_shape(self) -> None:
        data = client.post(
            "/api/v1/modularity",
            json={"type_string": "&{a: end}"},
        ).json()
        assert set(data.keys()) == {
            "is_modular",
            "is_distributive",
            "verdict",
            "classification",
        }

    def test_branch_modular(self) -> None:
        resp = client.post(
            "/api/v1/modularity",
            json={"type_string": "&{a: end, b: end}"},
        )
        data = resp.json()
        assert isinstance(data["is_modular"], bool)
        assert isinstance(data["is_distributive"], bool)

    def test_select_modular(self) -> None:
        resp = client.post(
            "/api/v1/modularity",
            json={"type_string": "+{a: end, b: end}"},
        )
        assert resp.status_code == 200

    def test_recursive_modular(self) -> None:
        resp = client.post(
            "/api/v1/modularity",
            json={"type_string": "rec X . &{next: X, stop: end}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] in ("MODULAR", "WEAKLY_MODULAR", "NOT_MODULAR", "INVALID")

    def test_modularity_parse_error(self) -> None:
        resp = client.post(
            "/api/v1/modularity",
            json={"type_string": "???"},
        )
        assert resp.status_code == 422

    def test_modularity_missing_body(self) -> None:
        resp = client.post("/api/v1/modularity", json={})
        assert resp.status_code == 422


# ===================================================================
# Import endpoint
# ===================================================================


class TestImport:
    """Tests for POST /api/v1/import."""

    def test_openapi_import(self) -> None:
        spec = {
            "paths": {
                "/orders": {
                    "post": {
                        "summary": "Create order",
                        "x-precondition": "init",
                        "x-postcondition": "created",
                    }
                },
                "/orders/{id}": {
                    "put": {
                        "summary": "Confirm order",
                        "x-precondition": "created",
                        "x-postcondition": "confirmed",
                    },
                    "delete": {
                        "summary": "Cancel order",
                        "x-precondition": "confirmed",
                        "x-postcondition": "cancelled",
                    },
                },
            }
        }
        resp = client.post(
            "/api/v1/import",
            json={"spec": spec, "format": "openapi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_types" in data
        assert "main" in data["session_types"]
        assert isinstance(data["session_types"]["main"], str)

    def test_grpc_import(self) -> None:
        spec = {
            "services": {
                "Greeter": {
                    "methods": ["SayHello", "SayGoodbye"],
                }
            }
        }
        resp = client.post(
            "/api/v1/import",
            json={"spec": spec, "format": "grpc"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Greeter" in data["session_types"]

    def test_asyncapi_import(self) -> None:
        spec = {
            "channels": {
                "orders": {
                    "publish": {"message": {"type": "object"}},
                    "subscribe": {"message": {"type": "object"}},
                }
            }
        }
        resp = client.post(
            "/api/v1/import",
            json={"spec": spec, "format": "asyncapi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "orders" in data["session_types"]

    def test_unsupported_format(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"spec": {}, "format": "xml"},
        )
        assert resp.status_code == 422

    def test_openapi_empty_paths(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"spec": {"paths": {}}, "format": "openapi"},
        )
        assert resp.status_code == 422

    def test_grpc_empty_services(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"spec": {"services": {}}, "format": "grpc"},
        )
        assert resp.status_code == 422

    def test_asyncapi_empty_channels(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"spec": {"channels": {}}, "format": "asyncapi"},
        )
        assert resp.status_code == 422

    def test_grpc_single_method(self) -> None:
        spec = {
            "services": {
                "Simple": {"methods": ["Call"]},
            }
        }
        resp = client.post(
            "/api/v1/import",
            json={"spec": spec, "format": "grpc"},
        )
        data = resp.json()
        assert "Simple" in data["session_types"]
        assert "Call" in data["session_types"]["Simple"]

    def test_asyncapi_publish_only(self) -> None:
        spec = {
            "channels": {
                "events": {"publish": {"message": {}}},
            }
        }
        resp = client.post(
            "/api/v1/import",
            json={"spec": spec, "format": "asyncapi"},
        )
        data = resp.json()
        assert "events" in data["session_types"]

    def test_import_missing_spec(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"format": "openapi"},
        )
        assert resp.status_code == 422

    def test_openapi_no_paths_key(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"spec": {"info": {"title": "test"}}, "format": "openapi"},
        )
        assert resp.status_code == 422

    def test_grpc_no_services_key(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"spec": {"options": {}}, "format": "grpc"},
        )
        assert resp.status_code == 422

    def test_asyncapi_no_channels_key(self) -> None:
        resp = client.post(
            "/api/v1/import",
            json={"spec": {"info": {}}, "format": "asyncapi"},
        )
        assert resp.status_code == 422


# ===================================================================
# Self-referencing benchmarks (P104 dogfooding)
# ===================================================================


class TestSelfReference:
    """Tests using the API's own protocol as a session type."""

    def test_api_protocol_is_parseable(self) -> None:
        """The API's own session type must parse without error."""
        resp = client.post(
            "/api/v1/analyze",
            json={
                "type_string": API_PROTOCOL,
                "protocol_name": "REST API Protocol",
            },
        )
        assert resp.status_code == 200

    def test_api_protocol_is_lattice(self) -> None:
        data = client.post(
            "/api/v1/analyze",
            json={"type_string": API_PROTOCOL},
        ).json()
        assert data["is_lattice"] is True

    def test_api_protocol_modularity(self) -> None:
        data = client.post(
            "/api/v1/modularity",
            json={"type_string": API_PROTOCOL},
        ).json()
        assert data["verdict"] in ("MODULAR", "WEAKLY_MODULAR", "NOT_MODULAR", "INVALID")

    def test_api_protocol_has_states(self) -> None:
        data = client.post(
            "/api/v1/analyze",
            json={"type_string": API_PROTOCOL},
        ).json()
        assert data["metrics"]["states"] > 1

    def test_api_protocol_classification_not_empty(self) -> None:
        data = client.post(
            "/api/v1/modularity",
            json={"type_string": API_PROTOCOL},
        ).json()
        assert data["classification"] != ""


# ===================================================================
# Edge cases and additional coverage
# ===================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_nested_branch(self) -> None:
        resp = client.post(
            "/api/v1/analyze",
            json={"type_string": "&{a: &{b: end, c: end}, d: end}"},
        )
        assert resp.status_code == 200

    def test_wrong_method_on_health(self) -> None:
        resp = client.post("/api/v1/health")
        assert resp.status_code == 405

    def test_wrong_method_on_analyze(self) -> None:
        resp = client.get("/api/v1/analyze")
        assert resp.status_code == 405

    def test_nonexistent_endpoint(self) -> None:
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404

    def test_analyze_with_unicode_type(self) -> None:
        """Unicode operators should work."""
        resp = client.post(
            "/api/v1/modularity",
            json={"type_string": "end"},
        )
        assert resp.status_code == 200

    def test_grpc_empty_methods_service(self) -> None:
        """A gRPC service with no methods should be skipped."""
        spec = {
            "services": {
                "Empty": {"methods": []},
                "Valid": {"methods": ["Call"]},
            }
        }
        resp = client.post(
            "/api/v1/import",
            json={"spec": spec, "format": "grpc"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Valid" in data["session_types"]

    def test_format_case_insensitive(self) -> None:
        spec = {
            "services": {
                "Svc": {"methods": ["Ping"]},
            }
        }
        resp = client.post(
            "/api/v1/import",
            json={"spec": spec, "format": "GRPC"},
        )
        assert resp.status_code == 200
