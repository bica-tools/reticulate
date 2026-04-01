"""L3 session type extractor from structured API descriptions.

Generates session types at L3 fidelity (with return-value selections
and preconditions) from a structured API description format.

Input format: a dict describing an API's methods, their return types,
preconditions, and state transitions. The extractor generates a session
type string that can be parsed by ``reticulate.parser.parse``.

Key insight (Step 97d): methods returning boolean/enum are SELECTION
points. The environment (the API) decides, not the client. A method
``available() -> bool`` becomes ``&{available: +{TRUE: ..., FALSE: ...}}``.

Three method kinds:
  - CALL: client calls, always succeeds → branch label
  - CHECK: client calls, environment selects result → branch + selection
  - EVENT: environment sends → selection label (in recursive context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from reticulate.parser import parse, pretty


# ---------------------------------------------------------------------------
# API description types
# ---------------------------------------------------------------------------

@dataclass
class MethodDesc:
    """Description of a single API method.

    Attributes:
        name: Method name (becomes a label in the session type).
        returns: Return type. Special values:
            - "void" or None → no selection (just a branch)
            - "bool" → +{TRUE: ..., FALSE: ...}
            - list of strings → +{label1: ..., label2: ...}
        next_state: State name after successful call (or after each return value).
            Can be a string (same state for all returns) or dict mapping
            return values to state names.
        precondition: State(s) in which this method is available.
            None means available in the method's owning state.
    """
    name: str
    returns: Optional[str | list[str]] = None
    next_state: Optional[str | dict[str, str]] = None
    terminates: bool = False  # True if this method ends the session


@dataclass
class StateDesc:
    """Description of a protocol state.

    Attributes:
        name: State name (used for cross-referencing).
        methods: Methods available in this state.
        is_recursive: If True, this state can loop back to itself.
    """
    name: str
    methods: list[MethodDesc] = field(default_factory=list)


@dataclass
class APIDesc:
    """Complete API description for L3 extraction.

    Attributes:
        name: API name.
        initial_state: Name of the starting state.
        states: List of state descriptions.
    """
    name: str
    initial_state: str
    states: list[StateDesc] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction engine
# ---------------------------------------------------------------------------

def _method_to_st(method: MethodDesc, state_map: dict[str, str],
                  current_state: str) -> str:
    """Convert a single method description to a session type fragment."""
    if method.terminates:
        if method.returns is None or method.returns == "void":
            return f"{method.name}: end"
        elif method.returns == "bool":
            return f"{method.name}: +{{TRUE: end, FALSE: end}}"
        elif isinstance(method.returns, list):
            labels = ", ".join(f"{r}: end" for r in method.returns)
            return f"{method.name}: +{{{labels}}}"

    if method.returns is None or method.returns == "void":
        # Simple call — goes to next state
        next_st = _resolve_next(method.next_state, state_map, current_state)
        return f"{method.name}: {next_st}"

    elif method.returns == "bool":
        if isinstance(method.next_state, dict):
            true_st = _resolve_next(method.next_state.get("TRUE"), state_map, current_state)
            false_st = _resolve_next(method.next_state.get("FALSE"), state_map, current_state)
        else:
            next_st = _resolve_next(method.next_state, state_map, current_state)
            true_st = next_st
            false_st = next_st
        return f"{method.name}: +{{TRUE: {true_st}, FALSE: {false_st}}}"

    elif isinstance(method.returns, list):
        if isinstance(method.next_state, dict):
            labels = []
            for r in method.returns:
                r_st = _resolve_next(method.next_state.get(r), state_map, current_state)
                labels.append(f"{r}: {r_st}")
        else:
            next_st = _resolve_next(method.next_state, state_map, current_state)
            labels = [f"{r}: {next_st}" for r in method.returns]
        return f"{method.name}: +{{{', '.join(labels)}}}"

    return f"{method.name}: end"


def _resolve_next(next_state: Optional[str], state_map: dict[str, str],
                  current: str) -> str:
    """Resolve a next-state reference to a session type fragment."""
    if next_state is None:
        return state_map.get(current, "end")
    if next_state == "end":
        return "end"
    if next_state == "self":
        return state_map.get(current, "end")
    return state_map.get(next_state, "end")


def extract_session_type(api: APIDesc) -> str:
    """Extract an L3 session type from an API description.

    Two-pass algorithm:
    1. Build state map: state_name → session type variable or fragment
    2. Assemble: construct the session type by inlining state references

    For recursive states, uses ``rec X . ...`` with variable ``X``.
    """
    # Collect state names
    state_names = {s.name for s in api.states}

    # Detect recursive states: a state is recursive if any method
    # transitions back to it (directly or via "self")
    recursive_states: set[str] = set()
    for state in api.states:
        for method in state.methods:
            if isinstance(method.next_state, str):
                if method.next_state == "self" or method.next_state == state.name:
                    recursive_states.add(state.name)
            elif isinstance(method.next_state, dict):
                for target in method.next_state.values():
                    if target == "self" or target == state.name:
                        recursive_states.add(state.name)

    # Assign variable names to recursive states
    var_names: dict[str, str] = {}
    var_counter = 0
    for sname in recursive_states:
        var_names[sname] = chr(ord('X') + var_counter) if var_counter < 3 else f"X{var_counter}"
        var_counter += 1

    # Build state fragments bottom-up
    # Simple approach: inline everything (works for acyclic + single recursion)
    state_map: dict[str, str] = {}

    # First pass: assign recursive variables
    for sname in var_names:
        state_map[sname] = var_names[sname]

    # Second pass: build each state's session type
    def build_state(state: StateDesc) -> str:
        methods_st = []
        for method in state.methods:
            methods_st.append(_method_to_st(method, state_map, state.name))

        if not methods_st:
            return "end"

        body = "&{" + ", ".join(methods_st) + "}"

        if state.name in var_names:
            var = var_names[state.name]
            return f"rec {var} . {body}"
        return body

    # Build all states
    built: dict[str, str] = {}
    for state in api.states:
        built[state.name] = build_state(state)

    # Update state_map with built fragments (for non-recursive states)
    for sname, fragment in built.items():
        if sname not in var_names:
            state_map[sname] = fragment

    # Rebuild with resolved references
    for state in api.states:
        built[state.name] = build_state(state)

    return built.get(api.initial_state, "end")


# ---------------------------------------------------------------------------
# Pre-built API descriptions for common protocols
# ---------------------------------------------------------------------------

def java_iterator_api() -> APIDesc:
    return APIDesc(
        name="java.util.Iterator",
        initial_state="ready",
        states=[
            StateDesc("ready", [
                MethodDesc("hasNext", returns="bool",
                           next_state={"TRUE": "has_element", "FALSE": "end"}),
            ]),
            StateDesc("has_element", [
                MethodDesc("next", next_state="ready"),
            ]),
        ],
    )


def java_inputstream_api() -> APIDesc:
    return APIDesc(
        name="java.io.InputStream",
        initial_state="open",
        states=[
            StateDesc("open", [
                MethodDesc("read", returns=["data", "EOF"],
                           next_state={"data": "self", "EOF": "eof"}),
                MethodDesc("skip", next_state="self"),
                MethodDesc("close", terminates=True),
            ]),
            StateDesc("eof", [
                MethodDesc("close", terminates=True),
            ]),
        ],
    )


def jdbc_connection_api() -> APIDesc:
    """JDBC: single recursive state with query/commit/close."""
    return APIDesc(
        name="java.sql.Connection",
        initial_state="active",
        states=[
            StateDesc("active", [
                MethodDesc("executeQuery", returns=["RESULT", "ERR"],
                           next_state={"RESULT": "self", "ERR": "self"}),
                MethodDesc("executeUpdate", returns=["OK", "ERR"],
                           next_state={"OK": "self", "ERR": "self"}),
                MethodDesc("commit", next_state="self"),
                MethodDesc("rollback", next_state="self"),
                MethodDesc("close", terminates=True),
            ]),
        ],
    )


def python_file_api() -> APIDesc:
    return APIDesc(
        name="Python file object",
        initial_state="open",
        states=[
            StateDesc("open", [
                MethodDesc("read", returns=["data", "EOF"],
                           next_state={"data": "self", "EOF": "eof"}),
                MethodDesc("write", next_state="self"),
                MethodDesc("close", terminates=True),
            ]),
            StateDesc("eof", [
                MethodDesc("close", terminates=True),
            ]),
        ],
    )


def kafka_producer_api() -> APIDesc:
    return APIDesc(
        name="Apache Kafka Producer",
        initial_state="ready",
        states=[
            StateDesc("ready", [
                MethodDesc("send", returns=["ACK", "ERR"],
                           next_state={"ACK": "self", "ERR": "self"}),
                MethodDesc("flush", next_state="self"),
                MethodDesc("close", terminates=True),
            ]),
        ],
    )


def mongodb_api() -> APIDesc:
    """Simplified MongoDB: insert/find cycle with cursor iteration."""
    return APIDesc(
        name="MongoDB Client",
        initial_state="connected",
        states=[
            StateDesc("connected", [
                MethodDesc("insertOne", returns=["OK", "ERR"],
                           next_state={"OK": "self", "ERR": "self"}),
                MethodDesc("find", returns=["FOUND", "EMPTY"],
                           next_state={"FOUND": "self", "EMPTY": "self"}),
                MethodDesc("close", terminates=True),
            ]),
        ],
    )


# Registry of all API descriptions
API_REGISTRY: dict[str, callable] = {
    "java_iterator": java_iterator_api,
    "java_inputstream": java_inputstream_api,
    "jdbc_connection": jdbc_connection_api,
    "python_file": python_file_api,
    "kafka_producer": kafka_producer_api,
    "mongodb": mongodb_api,
}


def extract_all() -> dict[str, str]:
    """Extract session types for all registered APIs."""
    results: dict[str, str] = {}
    for name, api_fn in API_REGISTRY.items():
        api = api_fn()
        st = extract_session_type(api)
        results[name] = st
    return results
