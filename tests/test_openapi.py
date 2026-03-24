"""Tests for OpenAPI stateful API contracts (Step 71)."""

import pytest

from reticulate.parser import Branch, End, Rec, Var, parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice
from reticulate.openapi import (
    APIContract,
    OpenAPIEndpoint,
    TraceValidationResult,
    api_to_contract,
    api_to_session_type,
    auth_flow,
    crud_lifecycle,
    order_fulfillment_flow,
    pagination_flow,
    payment_flow,
    session_type_to_openapi_extension,
    validate_api_trace,
    webhook_subscription_flow,
)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestOpenAPIEndpoint:
    """Test endpoint dataclass basics."""

    def test_endpoint_creation(self):
        ep = OpenAPIEndpoint("POST", "/items", "init", "created")
        assert ep.method == "POST"
        assert ep.path == "/items"
        assert ep.precondition_state == "init"
        assert ep.postcondition_state == "created"

    def test_endpoint_with_description(self):
        ep = OpenAPIEndpoint("GET", "/items/{id}", "created", "created",
                             description="Fetch item details")
        assert ep.description == "Fetch item details"

    def test_endpoint_frozen(self):
        ep = OpenAPIEndpoint("GET", "/x", "a", "b")
        with pytest.raises(AttributeError):
            ep.method = "POST"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Session type construction
# ---------------------------------------------------------------------------


class TestAPIToSessionType:
    """Test conversion from endpoints to session types."""

    def test_single_endpoint(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        ast = api_to_session_type(eps)
        assert isinstance(ast, Branch)
        assert len(ast.choices) == 1

    def test_empty_endpoints(self):
        ast = api_to_session_type([])
        assert isinstance(ast, End)

    def test_linear_chain(self):
        eps = [
            OpenAPIEndpoint("POST", "/items", "init", "created"),
            OpenAPIEndpoint("DELETE", "/items/{id}", "created", "deleted"),
        ]
        ast = api_to_session_type(eps)
        assert isinstance(ast, Branch)

    def test_branching_from_same_state(self):
        eps = [
            OpenAPIEndpoint("GET", "/items", "init", "listed"),
            OpenAPIEndpoint("POST", "/items", "init", "created"),
        ]
        ast = api_to_session_type(eps)
        assert isinstance(ast, Branch)
        assert len(ast.choices) == 2

    def test_cycle_produces_rec(self):
        eps = [
            OpenAPIEndpoint("GET", "/next", "listing", "listing"),
            OpenAPIEndpoint("GET", "/done", "listing", "exhausted"),
        ]
        ast = api_to_session_type(eps)
        # Should produce a recursive type since listing -> listing.
        assert isinstance(ast, Rec)


# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------


class TestAPIContract:
    """Test full contract construction."""

    def test_contract_fields(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        assert isinstance(contract, APIContract)
        assert contract.initial_state == "init"
        assert "done" in contract.terminal_states
        assert len(contract.states) >= 2

    def test_contract_has_session_type(self):
        eps = crud_lifecycle("items")
        contract = api_to_contract(eps)
        assert contract.session_type is not None
        assert contract.session_type_string != ""

    def test_contract_has_state_space(self):
        eps = [OpenAPIEndpoint("POST", "/x", "a", "b")]
        contract = api_to_contract(eps)
        assert len(contract.state_space.states) >= 2
        assert len(contract.state_space.transitions) >= 1

    def test_contract_lattice_checked(self):
        eps = [OpenAPIEndpoint("POST", "/x", "a", "b")]
        contract = api_to_contract(eps)
        assert contract.lattice_result is not None

    def test_empty_contract(self):
        contract = api_to_contract([])
        assert isinstance(contract.session_type, End)
        assert contract.lattice_result.is_lattice


# ---------------------------------------------------------------------------
# Trace validation
# ---------------------------------------------------------------------------


class TestValidateAPITrace:
    """Test API trace validation."""

    def test_valid_single_step(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, ["POST /items"])
        assert result.valid
        assert result.steps_completed == 1

    def test_empty_trace_is_valid(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, [])
        assert result.valid
        assert result.steps_completed == 0

    def test_invalid_call(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, ["DELETE /items"])
        assert not result.valid
        assert result.violation_index == 0

    def test_valid_two_step_chain(self):
        eps = [
            OpenAPIEndpoint("POST", "/items", "init", "created"),
            OpenAPIEndpoint("DELETE", "/items/{id}", "created", "deleted"),
        ]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, [
            "POST /items",
            "DELETE /items/{id}",
        ])
        assert result.valid
        assert result.steps_completed == 2

    def test_out_of_order_call(self):
        eps = [
            OpenAPIEndpoint("POST", "/items", "init", "created"),
            OpenAPIEndpoint("DELETE", "/items/{id}", "created", "deleted"),
        ]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, [
            "DELETE /items/{id}",  # Wrong! Need POST first.
        ])
        assert not result.valid
        assert result.violation_index == 0

    def test_violation_message_informative(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, ["GET /items"])
        assert "GET /items" in result.violation_message
        assert "init" in result.violation_message

    def test_trace_final_state(self):
        eps = [
            OpenAPIEndpoint("POST", "/items", "init", "created"),
            OpenAPIEndpoint("PUT", "/items/{id}", "created", "updated"),
        ]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, ["POST /items"])
        assert result.final_state == "created"


# ---------------------------------------------------------------------------
# OpenAPI extension generation
# ---------------------------------------------------------------------------


class TestOpenAPIExtension:
    """Test YAML extension generation."""

    def test_extension_contains_type(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        yaml = session_type_to_openapi_extension(contract)
        assert "x-session-type:" in yaml
        assert "type:" in yaml

    def test_extension_contains_paths(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        yaml = session_type_to_openapi_extension(contract)
        assert "paths:" in yaml
        assert "/items:" in yaml

    def test_extension_contains_pre_post(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        yaml = session_type_to_openapi_extension(contract)
        assert 'precondition: "init"' in yaml
        assert 'postcondition: "done"' in yaml

    def test_extension_lattice_flag(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        yaml = session_type_to_openapi_extension(contract)
        assert "is-lattice:" in yaml

    def test_extension_terminal_states(self):
        eps = [OpenAPIEndpoint("POST", "/items", "init", "done")]
        contract = api_to_contract(eps)
        yaml = session_type_to_openapi_extension(contract)
        assert "terminal-states:" in yaml


# ---------------------------------------------------------------------------
# Common patterns: CRUD lifecycle
# ---------------------------------------------------------------------------


class TestCRUDLifecycle:
    """Test CRUD lifecycle pattern."""

    def test_crud_endpoints_count(self):
        eps = crud_lifecycle("orders")
        assert len(eps) == 6

    def test_crud_contract_builds(self):
        contract = api_to_contract(crud_lifecycle("orders"))
        assert contract.initial_state == "init"
        assert "deleted" in contract.terminal_states

    def test_crud_valid_create_read_delete(self):
        contract = api_to_contract(crud_lifecycle("orders"))
        result = validate_api_trace(contract, [
            "POST /orders",
            "GET /orders/{id}",
            "DELETE /orders/{id}",
        ])
        assert result.valid

    def test_crud_valid_create_update_delete(self):
        contract = api_to_contract(crud_lifecycle("orders"))
        result = validate_api_trace(contract, [
            "POST /orders",
            "PUT /orders/{id}",
            "DELETE /orders/{id}",
        ])
        assert result.valid

    def test_crud_invalid_delete_before_create(self):
        contract = api_to_contract(crud_lifecycle("orders"))
        result = validate_api_trace(contract, [
            "DELETE /orders/{id}",
        ])
        assert not result.valid

    def test_crud_lattice(self):
        contract = api_to_contract(crud_lifecycle("orders"))
        assert contract.lattice_result.is_lattice


# ---------------------------------------------------------------------------
# Common patterns: Auth flow
# ---------------------------------------------------------------------------


class TestAuthFlow:
    """Test authentication flow pattern."""

    def test_auth_endpoints_count(self):
        eps = auth_flow()
        assert len(eps) == 4

    def test_auth_valid_login_access_revoke(self):
        contract = api_to_contract(auth_flow())
        result = validate_api_trace(contract, [
            "POST /auth/token",
            "GET /api/resource",
            "POST /auth/revoke",
        ])
        assert result.valid

    def test_auth_invalid_access_without_login(self):
        contract = api_to_contract(auth_flow())
        result = validate_api_trace(contract, [
            "GET /api/resource",
        ])
        assert not result.valid

    def test_auth_multiple_refreshes(self):
        contract = api_to_contract(auth_flow())
        result = validate_api_trace(contract, [
            "POST /auth/token",
            "POST /auth/refresh",
            "POST /auth/refresh",
            "GET /api/resource",
        ])
        assert result.valid

    def test_auth_lattice(self):
        contract = api_to_contract(auth_flow())
        assert contract.lattice_result.is_lattice


# ---------------------------------------------------------------------------
# Common patterns: Payment flow
# ---------------------------------------------------------------------------


class TestPaymentFlow:
    """Test payment processing flow pattern."""

    def test_payment_endpoints_count(self):
        eps = payment_flow()
        assert len(eps) == 6

    def test_payment_happy_path(self):
        contract = api_to_contract(payment_flow())
        result = validate_api_trace(contract, [
            "POST /payments",
            "POST /payments/{id}/authorize",
            "POST /payments/{id}/capture",
            "POST /payments/{id}/settle",
        ])
        assert result.valid

    def test_payment_void_path(self):
        contract = api_to_contract(payment_flow())
        result = validate_api_trace(contract, [
            "POST /payments",
            "POST /payments/{id}/authorize",
            "POST /payments/{id}/void",
        ])
        assert result.valid

    def test_payment_cancel_path(self):
        contract = api_to_contract(payment_flow())
        result = validate_api_trace(contract, [
            "POST /payments",
            "POST /payments/{id}/cancel",
        ])
        assert result.valid

    def test_payment_invalid_capture_before_auth(self):
        contract = api_to_contract(payment_flow())
        result = validate_api_trace(contract, [
            "POST /payments",
            "POST /payments/{id}/capture",
        ])
        assert not result.valid

    def test_payment_lattice(self):
        contract = api_to_contract(payment_flow())
        assert contract.lattice_result.is_lattice


# ---------------------------------------------------------------------------
# Common patterns: Pagination
# ---------------------------------------------------------------------------


class TestPaginationFlow:
    """Test pagination pattern."""

    def test_pagination_endpoints(self):
        eps = pagination_flow()
        assert len(eps) == 3

    def test_pagination_multi_page(self):
        contract = api_to_contract(pagination_flow())
        result = validate_api_trace(contract, [
            "GET /items?page=1",
            "GET /items?page=next",
            "GET /items?page=next",
            "GET /items?page=last",
        ])
        assert result.valid

    def test_pagination_lattice(self):
        contract = api_to_contract(pagination_flow())
        assert contract.lattice_result.is_lattice


# ---------------------------------------------------------------------------
# Common patterns: Webhooks
# ---------------------------------------------------------------------------


class TestWebhookFlow:
    """Test webhook subscription lifecycle."""

    def test_webhook_subscribe_unsubscribe(self):
        contract = api_to_contract(webhook_subscription_flow())
        result = validate_api_trace(contract, [
            "POST /webhooks",
            "GET /webhooks/{id}",
            "DELETE /webhooks/{id}",
        ])
        assert result.valid

    def test_webhook_pause_resume(self):
        contract = api_to_contract(webhook_subscription_flow())
        result = validate_api_trace(contract, [
            "POST /webhooks",
            "POST /webhooks/{id}/pause",
            "POST /webhooks/{id}/resume",
            "DELETE /webhooks/{id}",
        ])
        assert result.valid

    def test_webhook_lattice(self):
        contract = api_to_contract(webhook_subscription_flow())
        assert contract.lattice_result.is_lattice


# ---------------------------------------------------------------------------
# Common patterns: Order fulfillment
# ---------------------------------------------------------------------------


class TestOrderFulfillment:
    """Test order fulfillment lifecycle."""

    def test_order_happy_path(self):
        contract = api_to_contract(order_fulfillment_flow())
        result = validate_api_trace(contract, [
            "POST /orders",
            "POST /orders/{id}/pay",
            "POST /orders/{id}/ship",
            "POST /orders/{id}/deliver",
        ])
        assert result.valid

    def test_order_return_path(self):
        contract = api_to_contract(order_fulfillment_flow())
        result = validate_api_trace(contract, [
            "POST /orders",
            "POST /orders/{id}/pay",
            "POST /orders/{id}/ship",
            "POST /orders/{id}/deliver",
            "POST /orders/{id}/return",
        ])
        assert result.valid

    def test_order_cancel_path(self):
        contract = api_to_contract(order_fulfillment_flow())
        result = validate_api_trace(contract, [
            "POST /orders",
            "POST /orders/{id}/cancel",
        ])
        assert result.valid

    def test_order_invalid_ship_before_pay(self):
        contract = api_to_contract(order_fulfillment_flow())
        result = validate_api_trace(contract, [
            "POST /orders",
            "POST /orders/{id}/ship",
        ])
        assert not result.valid

    def test_order_lattice(self):
        contract = api_to_contract(order_fulfillment_flow())
        assert contract.lattice_result.is_lattice


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_self_loop_state(self):
        """A state that transitions to itself (e.g., repeated reads)."""
        eps = [
            OpenAPIEndpoint("GET", "/items", "ready", "ready"),
            OpenAPIEndpoint("POST", "/done", "ready", "finished"),
        ]
        contract = api_to_contract(eps)
        result = validate_api_trace(contract, [
            "GET /items",
            "GET /items",
            "POST /done",
        ])
        assert result.valid

    def test_multiple_terminal_states(self):
        eps = [
            OpenAPIEndpoint("POST", "/a", "init", "done_a"),
            OpenAPIEndpoint("POST", "/b", "init", "done_b"),
        ]
        contract = api_to_contract(eps)
        assert len(contract.terminal_states) == 2

    def test_contract_state_space_has_bottom(self):
        eps = [OpenAPIEndpoint("POST", "/x", "a", "b")]
        contract = api_to_contract(eps)
        assert contract.state_space.bottom in contract.state_space.states

    def test_yaml_extension_is_string(self):
        eps = [OpenAPIEndpoint("POST", "/x", "a", "b")]
        contract = api_to_contract(eps)
        yaml = session_type_to_openapi_extension(contract)
        assert isinstance(yaml, str)
        assert len(yaml) > 0
