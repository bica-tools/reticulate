"""Runtime monitor generation from session types (Step 80).

Generates executable middleware that enforces session type protocols at
runtime. Given a session type AST, produces monitors for multiple target
languages that track protocol state and reject invalid transitions.

Supported targets:
- **Python**: Decorator/wrapper class that intercepts method calls
- **Java**: Aspect-style interceptor source code
- **Express.js**: Middleware function for REST API route guarding

Usage:
    from reticulate.parser import parse
    from reticulate.monitor import generate_python_monitor, SessionMonitor

    ast = parse("&{open: &{read: end, close: end}}")
    template = generate_python_monitor(ast, "FileProtocol")
    print(template.source_code)

    # Or use SessionMonitor directly at runtime:
    monitor = SessionMonitor.from_session_type(ast)
    monitor.transition("open")   # ok
    monitor.transition("read")   # ok
    monitor.transition("write")  # raises ProtocolViolationError
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

from reticulate.parser import SessionType, pretty
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProtocolViolationError(Exception):
    """Raised when a transition violates the session type protocol."""

    def __init__(
        self,
        method: str,
        current_state: int,
        allowed: list[str],
        *,
        message: str = "",
    ) -> None:
        self.method = method
        self.current_state = current_state
        self.allowed = allowed
        if not message:
            message = (
                f"Protocol violation: '{method}' not allowed in state "
                f"{current_state}. Allowed: {allowed}"
            )
        super().__init__(message)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitorConfig:
    """Configuration for monitor generation and runtime behavior.

    Attributes:
        language: Target language ("python", "java", "express").
        strict_mode: If True, violations raise/throw immediately.
                     If False, violations are handled per on_violation.
        log_transitions: If True, log every transition to stderr/console.
        on_violation: Action on protocol violation:
                      "raise" — raise ProtocolViolationError (Python) or throw (Java/JS)
                      "log" — log warning but allow the call
                      "ignore" — silently allow
    """
    language: str = "python"
    strict_mode: bool = True
    log_transitions: bool = False
    on_violation: str = "raise"  # "raise" | "log" | "ignore"


# ---------------------------------------------------------------------------
# Monitor template (generated source code + metadata)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitorTemplate:
    """Generated monitor source code with metadata.

    Attributes:
        language: Target language.
        source_code: Complete, self-contained source file.
        state_machine: Mapping from state ID to list of (label, target) pairs.
        transition_table: Flat list of (source, label, target) triples.
        initial_state: The top/initial state ID.
        terminal_state: The bottom/terminal state ID.
        class_name: Name used for the generated class/function.
    """
    language: str
    source_code: str
    state_machine: dict[int, list[tuple[str, int]]]
    transition_table: list[tuple[int, str, int]]
    initial_state: int
    terminal_state: int
    class_name: str


# ---------------------------------------------------------------------------
# SessionMonitor: live Python runtime monitor
# ---------------------------------------------------------------------------

class SessionMonitor:
    """Runtime monitor that tracks protocol state and validates transitions.

    This is a live monitor object (not generated source code). Use it
    directly in Python applications to enforce session type protocols.

    Usage:
        monitor = SessionMonitor.from_session_type(parse("&{a: &{b: end}}"))
        monitor.transition("a")  # advances state
        monitor.transition("b")  # advances to terminal
        monitor.transition("c")  # raises ProtocolViolationError
    """

    def __init__(
        self,
        state_machine: dict[int, list[tuple[str, int]]],
        initial_state: int,
        terminal_state: int,
        *,
        config: MonitorConfig | None = None,
        selection_transitions: set[tuple[int, str, int]] | None = None,
    ) -> None:
        self._state_machine = state_machine
        self._initial_state = initial_state
        self._terminal_state = terminal_state
        self._config = config or MonitorConfig()
        self._current_state = initial_state
        self._history: list[tuple[str, int, int]] = []
        self._selection_transitions = selection_transitions or set()

    @classmethod
    def from_session_type(
        cls,
        ast: SessionType,
        *,
        config: MonitorConfig | None = None,
    ) -> SessionMonitor:
        """Create a monitor from a parsed session type AST."""
        ss = build_statespace(ast)
        return cls.from_statespace(ss, config=config)

    @classmethod
    def from_statespace(
        cls,
        ss: StateSpace,
        *,
        config: MonitorConfig | None = None,
    ) -> SessionMonitor:
        """Create a monitor from a pre-built state space."""
        sm = _build_state_machine(ss)
        return cls(
            sm,
            ss.top,
            ss.bottom,
            config=config,
            selection_transitions=ss.selection_transitions,
        )

    @property
    def current_state(self) -> int:
        """Current protocol state."""
        return self._current_state

    @property
    def initial_state(self) -> int:
        """Initial protocol state."""
        return self._initial_state

    @property
    def terminal_state(self) -> int:
        """Terminal protocol state."""
        return self._terminal_state

    @property
    def is_terminal(self) -> bool:
        """True if the monitor is in a terminal state."""
        return self._current_state == self._terminal_state

    @property
    def allowed_methods(self) -> list[str]:
        """Methods allowed in the current state."""
        transitions = self._state_machine.get(self._current_state, [])
        return [label for label, _ in transitions]

    @property
    def allowed_selections(self) -> list[str]:
        """Selection labels allowed in the current state."""
        transitions = self._state_machine.get(self._current_state, [])
        return [
            label for label, tgt in transitions
            if (self._current_state, label, tgt) in self._selection_transitions
        ]

    @property
    def history(self) -> list[tuple[str, int, int]]:
        """List of (method, from_state, to_state) tuples for all transitions taken."""
        return list(self._history)

    def is_allowed(self, method: str) -> bool:
        """Check if a method call is allowed in the current state."""
        return method in self.allowed_methods

    def transition(self, method: str) -> int:
        """Execute a transition. Returns the new state.

        Raises ProtocolViolationError if the method is not allowed
        (when config.on_violation == "raise" or config.strict_mode is True).
        """
        transitions = self._state_machine.get(self._current_state, [])
        targets = [tgt for lbl, tgt in transitions if lbl == method]

        if not targets:
            allowed = [lbl for lbl, _ in transitions]
            if self._config.on_violation == "raise" or self._config.strict_mode:
                raise ProtocolViolationError(
                    method, self._current_state, allowed
                )
            elif self._config.on_violation == "log":
                import sys
                print(
                    f"WARNING: Protocol violation: '{method}' not allowed "
                    f"in state {self._current_state}. Allowed: {allowed}",
                    file=sys.stderr,
                )
                return self._current_state
            else:
                # ignore
                return self._current_state

        old_state = self._current_state
        self._current_state = targets[0]
        self._history.append((method, old_state, self._current_state))

        if self._config.log_transitions:
            import sys
            print(
                f"MONITOR: {method}: {old_state} -> {self._current_state}",
                file=sys.stderr,
            )

        return self._current_state

    def reset(self) -> None:
        """Reset the monitor to the initial state."""
        self._current_state = self._initial_state
        self._history.clear()

    def check_complete(self) -> bool:
        """Check if the session has completed (reached terminal state)."""
        return self.is_terminal

    def wrap(self, obj: Any, *, method_map: dict[str, str] | None = None) -> MonitoredProxy:
        """Wrap an object with this monitor, intercepting method calls.

        Args:
            obj: The object to wrap.
            method_map: Optional mapping from session type labels to
                        actual method names on the object.
        Returns:
            A MonitoredProxy that intercepts method calls.
        """
        return MonitoredProxy(obj, self, method_map=method_map)


# ---------------------------------------------------------------------------
# MonitoredProxy: wraps an object with session type checking
# ---------------------------------------------------------------------------

class MonitoredProxy:
    """Proxy that intercepts method calls and validates against a monitor.

    Usage:
        monitor = SessionMonitor.from_session_type(ast)
        proxy = monitor.wrap(my_file_handle)
        proxy.open()   # validated by monitor, then delegated
        proxy.read()   # validated, then delegated
        proxy.write()  # raises ProtocolViolationError if not allowed
    """

    def __init__(
        self,
        target: Any,
        monitor: SessionMonitor,
        *,
        method_map: dict[str, str] | None = None,
    ) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_monitor", monitor)
        object.__setattr__(self, "_method_map", method_map or {})

    def __getattr__(self, name: str) -> Any:
        monitor = object.__getattribute__(self, "_monitor")
        method_map = object.__getattribute__(self, "_method_map")
        target = object.__getattribute__(self, "_target")

        # Reverse lookup: find protocol label for this method name
        label = name
        for proto_label, actual_name in method_map.items():
            if actual_name == name:
                label = proto_label
                break

        # If the label is in the protocol, validate it
        if label in monitor.allowed_methods:
            monitor.transition(label)
        elif monitor.allowed_methods:
            # The method is in the protocol vocabulary but not currently allowed
            all_labels = set()
            for transitions in monitor._state_machine.values():
                for lbl, _ in transitions:
                    all_labels.add(lbl)
            if label in all_labels:
                monitor.transition(label)  # will raise ProtocolViolationError

        return getattr(target, name)


# ---------------------------------------------------------------------------
# Internal: build state machine from StateSpace
# ---------------------------------------------------------------------------

def _build_state_machine(ss: StateSpace) -> dict[int, list[tuple[str, int]]]:
    """Convert a StateSpace into a state machine lookup table."""
    sm: dict[int, list[tuple[str, int]]] = {}
    for state in ss.states:
        sm[state] = []
    for src, label, tgt in ss.transitions:
        sm.setdefault(src, []).append((label, tgt))
    return sm


def _build_transition_table(ss: StateSpace) -> list[tuple[int, str, int]]:
    """Extract a flat transition table from a StateSpace."""
    return list(ss.transitions)


# ---------------------------------------------------------------------------
# Python monitor generation
# ---------------------------------------------------------------------------

def generate_python_monitor(
    ast: SessionType,
    class_name: str = "ProtocolMonitor",
    *,
    config: MonitorConfig | None = None,
) -> MonitorTemplate:
    """Generate a Python monitor class as source code.

    The generated class is a self-contained decorator/wrapper that can be
    pasted into any Python project. It tracks protocol state and raises
    ProtocolViolationError on invalid transitions.

    Args:
        ast: Parsed session type.
        class_name: Name for the generated monitor class.
        config: Monitor configuration.

    Returns:
        MonitorTemplate with the generated source code.
    """
    if config is None:
        config = MonitorConfig(language="python")

    ss = build_statespace(ast)
    sm = _build_state_machine(ss)
    tt = _build_transition_table(ss)

    # Build transition dict as Python source literal
    trans_dict_lines = []
    for state in sorted(sm.keys()):
        entries = sm[state]
        if entries:
            pairs = ", ".join(f"({lbl!r}, {tgt})" for lbl, tgt in entries)
            trans_dict_lines.append(f"        {state}: [{pairs}],")
        else:
            trans_dict_lines.append(f"        {state}: [],")
    trans_dict_src = "\n".join(trans_dict_lines)

    # Build violation handler
    if config.on_violation == "raise":
        violation_code = textwrap.dedent("""\
            raise ProtocolViolationError(
                f"Protocol violation: '{method}' not allowed in state "
                f"{self._state}. Allowed: {allowed}"
            )""")
    elif config.on_violation == "log":
        violation_code = textwrap.dedent("""\
            import sys
            print(
                f"WARNING: Protocol violation: '{method}' in state {self._state}",
                file=sys.stderr,
            )
            return None""")
    else:
        violation_code = "return None  # violation ignored"

    violation_indented = textwrap.indent(violation_code, "            ")

    # Log transitions code
    log_code = ""
    if config.log_transitions:
        log_code = textwrap.dedent("""\
        import sys
        print(f"MONITOR: {method}: {old} -> {self._state}", file=sys.stderr)
        """)
    log_indented = textwrap.indent(log_code, "        ")

    source = textwrap.dedent(f"""\
    \"\"\"Generated runtime monitor for session type protocol.

    Session type: {pretty(ast)}
    Generated by reticulate.monitor
    \"\"\"

    from __future__ import annotations


    class ProtocolViolationError(Exception):
        \"\"\"Raised when a method call violates the session type protocol.\"\"\"
        pass


    class {class_name}:
        \"\"\"Runtime monitor that enforces the session type protocol.

        Tracks current protocol state and validates each method call
        against the allowed transitions.

        Usage:
            monitor = {class_name}()
            monitor.call("open")   # valid
            monitor.call("read")   # valid if allowed
            monitor.call("write")  # raises ProtocolViolationError if not allowed
        \"\"\"

        INITIAL_STATE = {ss.top}
        TERMINAL_STATE = {ss.bottom}
        TRANSITIONS = {{
    {trans_dict_src}
        }}

        def __init__(self) -> None:
            self._state: int = self.INITIAL_STATE
            self._history: list[tuple[str, int, int]] = []

        @property
        def current_state(self) -> int:
            return self._state

        @property
        def is_terminal(self) -> bool:
            return self._state == self.TERMINAL_STATE

        @property
        def allowed(self) -> list[str]:
            return [label for label, _ in self.TRANSITIONS.get(self._state, [])]

        def call(self, method: str) -> int:
            \"\"\"Validate and execute a protocol transition.\"\"\"
            transitions = self.TRANSITIONS.get(self._state, [])
            allowed = [label for label, _ in transitions]
            targets = [tgt for lbl, tgt in transitions if lbl == method]

            if not targets:
    {violation_indented}

            old = self._state
            self._state = targets[0]
            self._history.append((method, old, self._state))
    {log_indented}
            return self._state

        def reset(self) -> None:
            \"\"\"Reset to initial state.\"\"\"
            self._state = self.INITIAL_STATE
            self._history.clear()

        @property
        def history(self) -> list[tuple[str, int, int]]:
            return list(self._history)


    def monitor_decorator(cls):
        \"\"\"Class decorator that wraps methods with protocol checking.

        Usage:
            @monitor_decorator
            class MyService:
                def open(self): ...
                def read(self): ...
                def close(self): ...
        \"\"\"
        monitor = {class_name}()
        original_getattribute = cls.__getattribute__ if hasattr(cls, '__getattribute__') else object.__getattribute__

        def checked_getattr(self, name):
            attr = original_getattribute(self, name)
            if callable(attr) and not name.startswith('_'):
                if name in monitor.allowed:
                    def wrapper(*args, **kwargs):
                        monitor.call(name)
                        return attr(*args, **kwargs)
                    return wrapper
            return attr

        cls.__getattribute__ = checked_getattr
        cls._monitor = monitor
        return cls
    """)

    return MonitorTemplate(
        language="python",
        source_code=source,
        state_machine=sm,
        transition_table=tt,
        initial_state=ss.top,
        terminal_state=ss.bottom,
        class_name=class_name,
    )


# ---------------------------------------------------------------------------
# Java monitor generation
# ---------------------------------------------------------------------------

def generate_java_monitor(
    ast: SessionType,
    class_name: str = "ProtocolMonitor",
    *,
    config: MonitorConfig | None = None,
    package_name: str = "com.reticulate.monitor",
) -> MonitorTemplate:
    """Generate a Java interceptor/aspect class as source code.

    The generated class uses a state machine to validate method calls.
    It can be used as a JUnit rule, an AOP aspect, or a manual interceptor.

    Args:
        ast: Parsed session type.
        class_name: Name for the generated class.
        config: Monitor configuration.
        package_name: Java package name.

    Returns:
        MonitorTemplate with the generated Java source.
    """
    if config is None:
        config = MonitorConfig(language="java")

    ss = build_statespace(ast)
    sm = _build_state_machine(ss)
    tt = _build_transition_table(ss)

    # Build transition map initialization
    init_lines = []
    for state in sorted(sm.keys()):
        for label, target in sm[state]:
            init_lines.append(
                f'        transitions.computeIfAbsent({state}, '
                f'k -> new java.util.HashMap<>()).put("{label}", {target});'
            )
    init_src = "\n".join(init_lines)

    # Violation handler
    if config.on_violation == "raise":
        violation = (
            f'throw new ProtocolViolationException('
            f'"Method \'" + method + "\' not allowed in state " + currentState '
            f'+ ". Allowed: " + getAllowed());'
        )
    elif config.on_violation == "log":
        violation = (
            f'System.err.println("WARNING: Protocol violation: \'" + method + '
            f'"\' in state " + currentState);'
        )
    else:
        violation = "// violation ignored"

    log_line = ""
    if config.log_transitions:
        log_line = (
            '        System.err.println("MONITOR: " + method + ": " + '
            'oldState + " -> " + currentState);'
        )

    source = textwrap.dedent(f"""\
    package {package_name};

    import java.util.*;

    /**
     * Generated runtime monitor for session type protocol.
     *
     * <p>Session type: {pretty(ast)}
     * <p>Generated by reticulate.monitor
     */
    public class {class_name} {{

        public static class ProtocolViolationException extends RuntimeException {{
            public ProtocolViolationException(String message) {{
                super(message);
            }}
        }}

        private static final int INITIAL_STATE = {ss.top};
        private static final int TERMINAL_STATE = {ss.bottom};

        private int currentState = INITIAL_STATE;
        private final Map<Integer, Map<String, Integer>> transitions = new HashMap<>();
        private final List<String> history = new ArrayList<>();

        public {class_name}() {{
    {init_src}
        }}

        public int getCurrentState() {{
            return currentState;
        }}

        public boolean isTerminal() {{
            return currentState == TERMINAL_STATE;
        }}

        public List<String> getAllowed() {{
            Map<String, Integer> trans = transitions.getOrDefault(currentState, Collections.emptyMap());
            return new ArrayList<>(trans.keySet());
        }}

        public int call(String method) {{
            Map<String, Integer> trans = transitions.getOrDefault(currentState, Collections.emptyMap());

            if (!trans.containsKey(method)) {{
                {violation}
            }}

            int oldState = currentState;
            currentState = trans.get(method);
            history.add(method + ": " + oldState + " -> " + currentState);
    {log_line}
            return currentState;
        }}

        public void reset() {{
            currentState = INITIAL_STATE;
            history.clear();
        }}

        public List<String> getHistory() {{
            return Collections.unmodifiableList(history);
        }}
    }}
    """)

    return MonitorTemplate(
        language="java",
        source_code=source,
        state_machine=sm,
        transition_table=tt,
        initial_state=ss.top,
        terminal_state=ss.bottom,
        class_name=class_name,
    )


# ---------------------------------------------------------------------------
# Express.js middleware generation
# ---------------------------------------------------------------------------

def generate_express_middleware(
    ast: SessionType,
    *,
    config: MonitorConfig | None = None,
    route_map: dict[str, str] | None = None,
) -> MonitorTemplate:
    """Generate Express.js middleware that enforces a session type protocol.

    The generated middleware tracks protocol state per session (using a
    session ID header or cookie) and returns 409 Conflict on violations.

    Args:
        ast: Parsed session type.
        config: Monitor configuration.
        route_map: Optional mapping from session type labels to Express
                   route patterns (e.g., {"open": "POST /files",
                   "read": "GET /files/:id"}).

    Returns:
        MonitorTemplate with the generated JavaScript source.
    """
    if config is None:
        config = MonitorConfig(language="express")

    ss = build_statespace(ast)
    sm = _build_state_machine(ss)
    tt = _build_transition_table(ss)

    # Build transition map as JS object literal
    trans_entries = []
    for state in sorted(sm.keys()):
        entries = sm[state]
        if entries:
            pairs = ", ".join(f'"{lbl}": {tgt}' for lbl, tgt in entries)
            trans_entries.append(f"  {state}: {{ {pairs} }},")
        else:
            trans_entries.append(f"  {state}: {{}},")
    trans_src = "\n".join(trans_entries)

    # Route mapping
    route_map_entries = []
    if route_map:
        for label, route in route_map.items():
            route_map_entries.append(f'  "{route}": "{label}",')
    route_map_src = "\n".join(route_map_entries) if route_map_entries else '  // Add route-to-label mappings here'

    # Violation handler
    if config.on_violation == "raise":
        violation_code = textwrap.dedent("""\
            return res.status(409).json({
              error: 'ProtocolViolation',
              message: `Method '${label}' not allowed in state ${state}`,
              allowed: Object.keys(transitions[state] || {}),
              state: state,
            });""")
    elif config.on_violation == "log":
        violation_code = textwrap.dedent("""\
            console.warn(`WARNING: Protocol violation: '${label}' in state ${state}`);
            return next();""")
    else:
        violation_code = "return next(); // violation ignored"

    violation_indented = textwrap.indent(violation_code, "    ")

    log_code = ""
    if config.log_transitions:
        log_code = "    console.log(`MONITOR: ${label}: ${oldState} -> ${sessions[sessionId]}`);\n"

    source = textwrap.dedent(f"""\
    /**
     * Generated Express.js middleware for session type protocol enforcement.
     *
     * Session type: {pretty(ast)}
     * Generated by reticulate.monitor
     */

    'use strict';

    const INITIAL_STATE = {ss.top};
    const TERMINAL_STATE = {ss.bottom};

    const transitions = {{
    {trans_src}
    }};

    // Map route patterns to protocol labels
    const routeMap = {{
    {route_map_src}
    }};

    // Per-session state tracking
    const sessions = {{}};

    /**
     * Extract the protocol label from the request.
     * Override this function to customize label extraction.
     */
    function extractLabel(req) {{
      // Try route map first
      const key = `${{req.method}} ${{req.path}}`;
      if (routeMap[key]) return routeMap[key];

      // Fall back to X-Protocol-Method header
      if (req.headers['x-protocol-method']) {{
        return req.headers['x-protocol-method'];
      }}

      // Fall back to last path segment
      const segments = req.path.split('/').filter(Boolean);
      return segments[segments.length - 1] || null;
    }}

    /**
     * Session type protocol middleware.
     *
     * Tracks per-session protocol state and rejects invalid transitions
     * with HTTP 409 Conflict.
     */
    function protocolMiddleware(req, res, next) {{
      const sessionId = req.headers['x-session-id'] || req.sessionID || 'default';
      const label = extractLabel(req);

      if (!label) {{
        return next(); // No protocol label — pass through
      }}

      // Initialize session state if needed
      if (!(sessionId in sessions)) {{
        sessions[sessionId] = INITIAL_STATE;
      }}

      const state = sessions[sessionId];
      const trans = transitions[state] || {{}};

      if (!(label in trans)) {{
    {violation_indented}
      }}

      const oldState = state;
      sessions[sessionId] = trans[label];
    {log_code}
      // Attach monitor info to request for downstream use
      req.protocolState = sessions[sessionId];
      req.protocolLabel = label;
      req.protocolTerminal = sessions[sessionId] === TERMINAL_STATE;

      next();
    }}

    /**
     * Reset a session's protocol state.
     */
    function resetSession(sessionId) {{
      sessions[sessionId || 'default'] = INITIAL_STATE;
    }}

    /**
     * Get current state for a session.
     */
    function getSessionState(sessionId) {{
      return sessions[sessionId || 'default'] || INITIAL_STATE;
    }}

    module.exports = {{
      protocolMiddleware,
      resetSession,
      getSessionState,
      INITIAL_STATE,
      TERMINAL_STATE,
      transitions,
    }};
    """)

    return MonitorTemplate(
        language="express",
        source_code=source,
        state_machine=sm,
        transition_table=tt,
        initial_state=ss.top,
        terminal_state=ss.bottom,
        class_name="protocolMiddleware",
    )


# ---------------------------------------------------------------------------
# Convenience: build transition table from AST
# ---------------------------------------------------------------------------

def build_transition_table(
    ast: SessionType,
) -> tuple[dict[int, list[tuple[str, int]]], int, int]:
    """Build a transition table from a session type AST.

    Returns:
        (state_machine, initial_state, terminal_state)
    """
    ss = build_statespace(ast)
    return _build_state_machine(ss), ss.top, ss.bottom
