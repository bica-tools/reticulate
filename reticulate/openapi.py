"""Stateful API contracts via OpenAPI extensions (Step 71).

Models REST API lifecycles as session types and generates OpenAPI
``x-session-type`` extensions.  Validates that sequences of API calls
conform to the protocol defined by the session type.

Usage::

    endpoints = [
        OpenAPIEndpoint("POST", "/orders", "init", "created"),
        OpenAPIEndpoint("PUT",  "/orders/{id}", "created", "confirmed"),
        OpenAPIEndpoint("DELETE", "/orders/{id}", "confirmed", "cancelled"),
    ]
    contract = api_to_contract(endpoints)
    result   = validate_api_trace(contract.session_type, ["POST /orders", "PUT /orders/{id}"])
    ext_yaml = session_type_to_openapi_extension(contract)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    SessionType,
    Var,
    parse,
)
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import LatticeResult, check_lattice


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenAPIEndpoint:
    """A single REST API endpoint with state constraints.

    Attributes:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE).
        path: URL path pattern (e.g. ``/orders/{id}``).
        precondition_state: State required before this call.
        postcondition_state: State after successful call.
        description: Human-readable description of the endpoint.
        response_codes: Expected HTTP status codes.
    """
    method: str
    path: str
    precondition_state: str
    postcondition_state: str
    description: str = ""
    response_codes: tuple[int, ...] = (200,)


@dataclass(frozen=True)
class TraceValidationResult:
    """Result of validating an API call trace against a session type.

    Attributes:
        valid: Whether the trace conforms to the protocol.
        steps_completed: Number of valid steps before failure (or total).
        violation_index: Index of the first violating call, or -1 if valid.
        violation_message: Human-readable description of the violation.
        final_state: Name of the state reached after the last valid step.
    """
    valid: bool
    steps_completed: int
    violation_index: int = -1
    violation_message: str = ""
    final_state: str = ""


@dataclass(frozen=True)
class APIContract:
    """Complete stateful API contract.

    Attributes:
        endpoints: The endpoint definitions.
        session_type: The derived session type AST.
        session_type_string: The session type in string form.
        state_machine: Mapping from state names to available transitions.
        states: All state names in the contract.
        initial_state: Starting state.
        terminal_states: States with no outgoing transitions.
        state_space: Constructed state space (LTS).
        lattice_result: Lattice check result.
    """
    endpoints: tuple[OpenAPIEndpoint, ...]
    session_type: SessionType
    session_type_string: str
    state_machine: dict[str, list[tuple[str, str]]]  # state -> [(label, next_state)]
    states: tuple[str, ...]
    initial_state: str
    terminal_states: tuple[str, ...]
    state_space: StateSpace
    lattice_result: LatticeResult


# ---------------------------------------------------------------------------
# State machine construction
# ---------------------------------------------------------------------------

def _build_state_machine(
    endpoints: Sequence[OpenAPIEndpoint],
) -> tuple[dict[str, list[tuple[str, str]]], str, tuple[str, ...]]:
    """Build a state machine from endpoint definitions.

    Returns (state_machine, initial_state, terminal_states).
    """
    sm: dict[str, list[tuple[str, str]]] = {}
    all_pre: set[str] = set()
    all_post: set[str] = set()

    for ep in endpoints:
        label = f"{ep.method} {ep.path}"
        sm.setdefault(ep.precondition_state, []).append((label, ep.postcondition_state))
        all_pre.add(ep.precondition_state)
        all_post.add(ep.postcondition_state)

    all_states = all_pre | all_post
    # Initial state: appears only as precondition, never as postcondition,
    # or the first endpoint's precondition as fallback.
    initial_candidates = all_pre - all_post
    if initial_candidates:
        initial = sorted(initial_candidates)[0]
    else:
        initial = endpoints[0].precondition_state if endpoints else "init"

    # Terminal states: appear as postcondition but have no outgoing transitions.
    terminal = tuple(sorted(s for s in all_states if s not in sm))

    # Ensure all states are in the state machine (terminal ones with empty lists).
    for s in all_states:
        sm.setdefault(s, [])

    return sm, initial, terminal


def _sanitize_label(label: str) -> str:
    """Convert an API label to a valid session type method name.

    Replaces spaces, slashes, braces with underscores; lowercases.
    """
    out = label.replace(" ", "_").replace("/", "_").replace("{", "").replace("}", "")
    # Collapse multiple underscores and strip leading/trailing.
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower()


def _state_machine_to_ast(
    sm: dict[str, list[tuple[str, str]]],
    initial: str,
    terminal_states: tuple[str, ...],
) -> SessionType:
    """Convert a state machine to a session type AST.

    Uses recursive types for cycles, Branch for multiple transitions.
    """
    # Map states to variable names for recursion.
    state_vars: dict[str, str] = {}
    for i, state in enumerate(sorted(sm.keys())):
        state_vars[state] = f"X{i}" if i > 0 else "X"

    def build(state: str, in_progress: set[str]) -> SessionType:
        if state in terminal_states or not sm.get(state):
            return End()
        if state in in_progress:
            return Var(state_vars[state])

        in_progress_next = in_progress | {state}
        transitions = sm[state]

        choices: list[tuple[str, SessionType]] = []
        for label, next_state in transitions:
            method = _sanitize_label(label)
            cont = build(next_state, in_progress_next)
            choices.append((method, cont))

        if len(choices) == 1:
            body = Branch(choices=(choices[0],))
        else:
            body = Branch(choices=tuple(choices))

        # Wrap in rec if this state can be reached again (cycle detection).
        needs_rec = _state_reachable_from(sm, state, state)
        if needs_rec:
            return Rec(var=state_vars[state], body=body)
        return body

    return build(initial, set())


def _state_reachable_from(
    sm: dict[str, list[tuple[str, str]]],
    start: str,
    target: str,
) -> bool:
    """Check if *target* is reachable from any successor of *start*."""
    visited: set[str] = set()
    stack = [next_s for _, next_s in sm.get(start, [])]
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
# Public API
# ---------------------------------------------------------------------------

def api_to_session_type(endpoints: Sequence[OpenAPIEndpoint]) -> SessionType:
    """Convert a list of API endpoints with state constraints to a session type.

    Each endpoint is treated as a branch choice in the state where its
    precondition holds.  The resulting AST uses Branch (external choice)
    for states with multiple available operations.
    """
    if not endpoints:
        return End()
    sm, initial, terminal = _build_state_machine(endpoints)
    return _state_machine_to_ast(sm, initial, terminal)


def api_to_contract(endpoints: Sequence[OpenAPIEndpoint]) -> APIContract:
    """Build a full APIContract from endpoint definitions.

    Constructs the session type, builds the state space, and checks
    lattice properties.
    """
    if not endpoints:
        ast: SessionType = End()
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        return APIContract(
            endpoints=tuple(endpoints),
            session_type=ast,
            session_type_string="end",
            state_machine={},
            states=(),
            initial_state="",
            terminal_states=(),
            state_space=ss,
            lattice_result=lr,
        )

    sm, initial, terminal = _build_state_machine(endpoints)
    ast = _state_machine_to_ast(sm, initial, terminal)

    # Build string representation via pretty-print.
    from reticulate.parser import pretty
    type_string = pretty(ast)

    ss = build_statespace(ast)
    lr = check_lattice(ss)

    all_states = tuple(sorted(sm.keys()))

    return APIContract(
        endpoints=tuple(endpoints),
        session_type=ast,
        session_type_string=type_string,
        state_machine=sm,
        states=all_states,
        initial_state=initial,
        terminal_states=terminal,
        state_space=ss,
        lattice_result=lr,
    )


def validate_api_trace(
    contract: APIContract,
    trace: Sequence[str],
) -> TraceValidationResult:
    """Validate a sequence of API calls against a contract.

    Each element of *trace* should be a string like ``"POST /orders"``
    matching the method+path of an endpoint.

    Returns a TraceValidationResult indicating whether the trace is valid.
    """
    if not trace:
        return TraceValidationResult(
            valid=True,
            steps_completed=0,
            final_state=contract.initial_state,
        )

    current_state = contract.initial_state
    sm = contract.state_machine

    for i, call in enumerate(trace):
        transitions = sm.get(current_state, [])
        matched = False
        for label, next_state in transitions:
            if label == call:
                current_state = next_state
                matched = True
                break

        if not matched:
            available = [lbl for lbl, _ in transitions]
            return TraceValidationResult(
                valid=False,
                steps_completed=i,
                violation_index=i,
                violation_message=(
                    f"Call '{call}' not allowed in state '{current_state}'. "
                    f"Available: {available}"
                ),
                final_state=current_state,
            )

    return TraceValidationResult(
        valid=True,
        steps_completed=len(trace),
        final_state=current_state,
    )


def session_type_to_openapi_extension(contract: APIContract) -> str:
    """Generate an OpenAPI extension YAML snippet for the contract.

    Produces ``x-session-type`` annotations that can be merged into an
    existing OpenAPI 3.x specification.
    """
    lines: list[str] = []
    lines.append("# OpenAPI Session Type Extension")
    lines.append("# Generated by reticulate (Step 71)")
    lines.append("")
    lines.append("x-session-type:")
    lines.append(f"  type: \"{contract.session_type_string}\"")
    lines.append(f"  initial-state: \"{contract.initial_state}\"")

    if contract.terminal_states:
        terms = ", ".join(f'"{s}"' for s in contract.terminal_states)
        lines.append(f"  terminal-states: [{terms}]")

    lines.append(f"  is-lattice: {str(contract.lattice_result.is_lattice).lower()}")
    lines.append(f"  states: {len(contract.state_space.states)}")
    lines.append(f"  transitions: {len(contract.state_space.transitions)}")
    lines.append("")

    # Per-path annotations.
    lines.append("paths:")
    for ep in contract.endpoints:
        lines.append(f"  {ep.path}:")
        lines.append(f"    {ep.method.lower()}:")
        lines.append(f"      x-session-type:")
        lines.append(f"        precondition: \"{ep.precondition_state}\"")
        lines.append(f"        postcondition: \"{ep.postcondition_state}\"")
        if ep.description:
            lines.append(f"        description: \"{ep.description}\"")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Common API patterns
# ---------------------------------------------------------------------------

def crud_lifecycle(resource: str) -> list[OpenAPIEndpoint]:
    """Generate CRUD lifecycle endpoints for a resource.

    States: init -> created -> updated -> deleted (terminal).
    Also: created -> deleted, init -> listed (with list as a loop).
    """
    base = f"/{resource}"
    item = f"/{resource}/{{id}}"
    return [
        OpenAPIEndpoint("POST", base, "init", "created",
                        f"Create a new {resource}"),
        OpenAPIEndpoint("GET", item, "created", "created",
                        f"Read {resource} details"),
        OpenAPIEndpoint("PUT", item, "created", "updated",
                        f"Update {resource}"),
        OpenAPIEndpoint("GET", item, "updated", "updated",
                        f"Read updated {resource}"),
        OpenAPIEndpoint("DELETE", item, "updated", "deleted",
                        f"Delete {resource}"),
        OpenAPIEndpoint("DELETE", item, "created", "deleted",
                        f"Delete {resource} without update"),
    ]


def auth_flow() -> list[OpenAPIEndpoint]:
    """Generate OAuth2-like authentication flow endpoints.

    States: unauthenticated -> token_requested -> authenticated -> refreshed -> revoked.
    """
    return [
        OpenAPIEndpoint("POST", "/auth/token", "unauthenticated", "authenticated",
                        "Request access token"),
        OpenAPIEndpoint("POST", "/auth/refresh", "authenticated", "authenticated",
                        "Refresh access token"),
        OpenAPIEndpoint("GET", "/api/resource", "authenticated", "authenticated",
                        "Access protected resource"),
        OpenAPIEndpoint("POST", "/auth/revoke", "authenticated", "revoked",
                        "Revoke token"),
    ]


def payment_flow() -> list[OpenAPIEndpoint]:
    """Generate payment processing flow endpoints.

    States: init -> created -> authorized -> captured -> settled.
    Also: authorized -> voided, created -> cancelled.
    """
    return [
        OpenAPIEndpoint("POST", "/payments", "init", "created",
                        "Create payment intent"),
        OpenAPIEndpoint("POST", "/payments/{id}/authorize", "created", "authorized",
                        "Authorize payment"),
        OpenAPIEndpoint("POST", "/payments/{id}/capture", "authorized", "captured",
                        "Capture authorized payment"),
        OpenAPIEndpoint("POST", "/payments/{id}/settle", "captured", "settled",
                        "Settle captured payment"),
        OpenAPIEndpoint("POST", "/payments/{id}/void", "authorized", "voided",
                        "Void authorized payment"),
        OpenAPIEndpoint("POST", "/payments/{id}/cancel", "created", "cancelled",
                        "Cancel payment before authorization"),
    ]


def pagination_flow() -> list[OpenAPIEndpoint]:
    """Generate paginated list API pattern.

    States: init -> listing -> exhausted.
    """
    return [
        OpenAPIEndpoint("GET", "/items?page=1", "init", "listing",
                        "Fetch first page"),
        OpenAPIEndpoint("GET", "/items?page=next", "listing", "listing",
                        "Fetch next page"),
        OpenAPIEndpoint("GET", "/items?page=last", "listing", "exhausted",
                        "Fetch last page"),
    ]


def webhook_subscription_flow() -> list[OpenAPIEndpoint]:
    """Generate webhook subscription lifecycle.

    States: init -> subscribed -> paused -> cancelled (terminal).
    """
    return [
        OpenAPIEndpoint("POST", "/webhooks", "init", "subscribed",
                        "Create webhook subscription"),
        OpenAPIEndpoint("GET", "/webhooks/{id}", "subscribed", "subscribed",
                        "Check subscription status"),
        OpenAPIEndpoint("POST", "/webhooks/{id}/pause", "subscribed", "paused",
                        "Pause webhook delivery"),
        OpenAPIEndpoint("POST", "/webhooks/{id}/resume", "paused", "subscribed",
                        "Resume webhook delivery"),
        OpenAPIEndpoint("DELETE", "/webhooks/{id}", "subscribed", "cancelled",
                        "Delete subscription"),
        OpenAPIEndpoint("DELETE", "/webhooks/{id}", "paused", "cancelled",
                        "Delete paused subscription"),
    ]


def order_fulfillment_flow() -> list[OpenAPIEndpoint]:
    """Generate e-commerce order fulfillment lifecycle.

    States: cart -> placed -> paid -> shipped -> delivered -> returned.
    """
    return [
        OpenAPIEndpoint("POST", "/orders", "cart", "placed",
                        "Place order"),
        OpenAPIEndpoint("POST", "/orders/{id}/pay", "placed", "paid",
                        "Process payment"),
        OpenAPIEndpoint("POST", "/orders/{id}/ship", "paid", "shipped",
                        "Ship order"),
        OpenAPIEndpoint("POST", "/orders/{id}/deliver", "shipped", "delivered",
                        "Confirm delivery"),
        OpenAPIEndpoint("POST", "/orders/{id}/return", "delivered", "returned",
                        "Initiate return"),
        OpenAPIEndpoint("POST", "/orders/{id}/cancel", "placed", "cancelled",
                        "Cancel unpaid order"),
    ]
