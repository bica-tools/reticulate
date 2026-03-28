"""Protocol engineering toolkit using Lagrangian mechanics (Step 60r).

Unified framework for DESCRIBE → BUILD → VERIFY → MONITOR:

1. DESCRIBE: Energy landscape visualization and profiling
2. BUILD: Design rules from Hamilton's principle
3. VERIFY: Conservation law checking (compile-time)
4. MONITOR: Runtime energy tracking and anomaly detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace

from reticulate.lagrangian import (
    kinetic_energy,
    potential_energy,
    lagrangian_field,
    hamiltonian_field,
    momentum,
    least_action_path,
    gravitational_field,
    analyze_circular,
    total_energy,
)
from reticulate.zeta import compute_rank, _reachability


# ---------------------------------------------------------------------------
# 1. DESCRIBE: Energy landscape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateProfile:
    """Energy profile of a single protocol state."""
    state_id: int
    label: str
    kinetic: float
    potential: float
    lagrangian: float
    hamiltonian: float
    momentum_val: int
    gravity: float
    is_hotspot: bool     # High kinetic energy (many choices)
    is_bottleneck: bool  # Only 1 outgoing transition in high-V region
    is_deadlock: bool    # T=0 and not bottom


def describe_protocol(ss: "StateSpace") -> list[StateProfile]:
    """Generate energy profile for every state in the protocol."""
    T = kinetic_energy(ss)
    V = potential_energy(ss)
    L = lagrangian_field(ss)
    H = hamiltonian_field(ss)
    p = momentum(ss)
    g = gravitational_field(ss)

    # Thresholds
    mean_T = sum(T.values()) / max(len(T), 1)

    profiles = []
    for s in sorted(ss.states):
        label = ss.labels.get(s, str(s)) if hasattr(ss, 'labels') else str(s)
        is_hot = T[s] > mean_T * 1.5
        is_bottle = (p[s] == 1 and V[s] > 1.0)
        is_dead = (T[s] == 0.0 and s != ss.bottom)

        profiles.append(StateProfile(
            state_id=s,
            label=label,
            kinetic=T[s],
            potential=V[s],
            lagrangian=L[s],
            hamiltonian=H[s],
            momentum_val=p[s],
            gravity=g[s],
            is_hotspot=is_hot,
            is_bottleneck=is_bottle,
            is_deadlock=is_dead,
        ))

    return profiles


# ---------------------------------------------------------------------------
# 2. BUILD: Design rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignScore:
    """Design quality score based on mechanical principles.

    Attributes:
        efficiency: Lower total action = more efficient.
        smoothness: Lower max |Δ²f| = smoother.
        robustness: Higher Fiedler value = more robust.
        termination: All paths reach bottom = guaranteed termination.
        balance: How constant H is along the least-action path.
        overall: Weighted combination.
        suggestions: List of improvement suggestions.
    """
    efficiency: float
    smoothness: float
    robustness: float
    termination: bool
    balance: float
    overall: float
    suggestions: list[str]


def evaluate_design(ss: "StateSpace") -> DesignScore:
    """Evaluate protocol design quality using mechanical principles."""
    from reticulate.protocol_mechanics import smoothness as smooth_fn
    from reticulate.eigenvalues import algebraic_connectivity

    # Efficiency: inverse of total action
    _, action = least_action_path(ss)
    n = len(ss.states)
    efficiency = 1.0 / (1.0 + abs(action) / max(n, 1))

    # Smoothness
    smooth = smooth_fn(ss)

    # Robustness (Fiedler)
    fiedler = algebraic_connectivity(ss)
    robustness = min(1.0, fiedler / 2.0)  # Normalize to [0, 1]

    # Termination: bottom reachable from all states
    reach = _reachability(ss)
    terminates = all(ss.bottom in reach[s] for s in ss.states)

    # Energy balance along least-action path
    path, _ = least_action_path(ss)
    H = hamiltonian_field(ss)
    if len(path) >= 2:
        h_values = [H[s] for s in path]
        h_range = max(h_values) - min(h_values)
        balance = 1.0 / (1.0 + h_range)
    else:
        balance = 1.0

    # Overall score
    overall = (efficiency + smooth + robustness + balance) / 4.0
    if not terminates:
        overall *= 0.5  # Heavy penalty for non-termination

    # Suggestions
    suggestions: list[str] = []
    if efficiency < 0.3:
        suggestions.append("High action: consider removing unnecessary branching")
    if smooth < 0.3:
        suggestions.append("Low smoothness: add intermediate states to reduce sharp transitions")
    if robustness < 0.2:
        suggestions.append("Low robustness: add parallel paths for redundancy")
    if not terminates:
        suggestions.append("CRITICAL: not all states can reach termination")
    if balance < 0.3:
        suggestions.append("Energy imbalance: some path segments are much more complex than others")

    # Check for deadlocks
    T = kinetic_energy(ss)
    deadlocks = [s for s in ss.states if T[s] == 0.0 and s != ss.bottom]
    if deadlocks:
        suggestions.append(f"DEADLOCK: states {deadlocks} have no outgoing transitions")

    return DesignScore(
        efficiency=efficiency,
        smoothness=smooth,
        robustness=robustness,
        termination=terminates,
        balance=balance,
        overall=overall,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# 3. VERIFY: Conservation law checking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationResult:
    """Result of mechanical verification."""
    all_passed: bool
    checks: list[tuple[str, bool, str]]  # (name, passed, message)


def verify_protocol(ss: "StateSpace") -> VerificationResult:
    """Verify protocol correctness using conservation laws."""
    checks: list[tuple[str, bool, str]] = []

    T = kinetic_energy(ss)
    V = potential_energy(ss)
    p = momentum(ss)
    reach = _reachability(ss)

    # Check 1: No deadlocks (T > 0 for non-bottom states)
    deadlocks = [s for s in ss.states if T[s] == 0 and s != ss.bottom]
    checks.append((
        "no_deadlock",
        len(deadlocks) == 0,
        f"Deadlock states: {deadlocks}" if deadlocks else "No deadlocks",
    ))

    # Check 2: Termination (all states reach bottom)
    unreachable = [s for s in ss.states if ss.bottom not in reach[s]]
    checks.append((
        "termination",
        len(unreachable) == 0,
        f"Cannot reach end from: {unreachable}" if unreachable else "All paths terminate",
    ))

    # Check 3: Potential monotonically decreasing along transitions
    V_violations = []
    for src, label, tgt in ss.transitions:
        if V[tgt] > V[src]:
            V_violations.append((src, label, tgt, V[src], V[tgt]))
    checks.append((
        "potential_decreasing",
        len(V_violations) == 0,
        f"Potential increases in {len(V_violations)} transitions" if V_violations
        else "Potential decreases along all transitions",
    ))

    # Check 4: No isolated states (every state is reachable from top)
    from_top = reach[ss.top]
    isolated = [s for s in ss.states if s not in from_top]
    checks.append((
        "no_isolated",
        len(isolated) == 0,
        f"Isolated states: {isolated}" if isolated else "All states reachable from top",
    ))

    # Check 5: Recursion termination (recursive loops have escape)
    circs = analyze_circular(ss)
    stuck_loops = [c for c in circs if c.escape_transitions == 0]
    checks.append((
        "recursive_escape",
        len(stuck_loops) == 0,
        f"{len(stuck_loops)} loops with no escape" if stuck_loops
        else "All recursive loops have escape transitions",
    ))

    all_passed = all(passed for _, passed, _ in checks)
    return VerificationResult(all_passed=all_passed, checks=checks)


# ---------------------------------------------------------------------------
# 4. MONITOR: Runtime energy tracking
# ---------------------------------------------------------------------------

@dataclass
class RuntimeMonitor:
    """Runtime monitor that tracks protocol execution using energy.

    Instantiate with a state space, then call step() for each
    protocol transition to check for anomalies.
    """
    ss: "StateSpace"
    _T: dict[int, float] = field(default_factory=dict, init=False)
    _V: dict[int, float] = field(default_factory=dict, init=False)
    _H: dict[int, float] = field(default_factory=dict, init=False)
    _p: dict[int, int] = field(default_factory=dict, init=False)
    _current: int = field(default=-1, init=False)
    _history: list[tuple[int, str]] = field(default_factory=list, init=False)
    _anomalies: list[str] = field(default_factory=list, init=False)
    _loop_counter: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._T = kinetic_energy(self.ss)
        self._V = potential_energy(self.ss)
        self._H = hamiltonian_field(self.ss)
        self._p = momentum(self.ss)
        self._current = self.ss.top

    def step(self, label: str) -> list[str]:
        """Execute one protocol step and return any anomalies detected."""
        anomalies: list[str] = []

        # Find transition matching label from current state
        targets = [
            tgt for src, lbl, tgt in self.ss.transitions
            if src == self._current and lbl == label
        ]

        if not targets:
            anomalies.append(
                f"INVALID_TRANSITION: '{label}' not available at state {self._current}")
            self._anomalies.extend(anomalies)
            return anomalies

        next_state = targets[0]

        # Check potential decrease
        if self._V[next_state] > self._V[self._current]:
            anomalies.append(
                f"REGRESSION: V increased from {self._V[self._current]:.1f} "
                f"to {self._V[next_state]:.1f}")

        # Check for approaching deadlock
        if self._T.get(next_state, 0) == 0 and next_state != self.ss.bottom:
            anomalies.append(f"DEADLOCK_RISK: state {next_state} has no outgoing transitions")

        # Track loop iterations
        if next_state in self._loop_counter:
            self._loop_counter[next_state] += 1
            if self._loop_counter[next_state] > 100:
                anomalies.append(
                    f"INFINITE_LOOP_RISK: state {next_state} visited "
                    f"{self._loop_counter[next_state]} times")
        else:
            self._loop_counter[next_state] = 1

        # Update state
        self._history.append((self._current, label))
        self._current = next_state
        self._anomalies.extend(anomalies)

        return anomalies

    @property
    def current_state(self) -> int:
        return self._current

    @property
    def current_energy(self) -> float:
        return self._H.get(self._current, 0.0)

    @property
    def current_potential(self) -> float:
        return self._V.get(self._current, 0.0)

    @property
    def is_terminated(self) -> bool:
        return self._current == self.ss.bottom

    @property
    def all_anomalies(self) -> list[str]:
        return list(self._anomalies)

    @property
    def history(self) -> list[tuple[int, str]]:
        return list(self._history)

    def energy_trace(self) -> list[float]:
        """Return the Hamiltonian value at each step of the execution."""
        trace = [self._H.get(self.ss.top, 0.0)]
        current = self.ss.top
        for state, label in self._history:
            # Find next state
            for src, lbl, tgt in self.ss.transitions:
                if src == state and lbl == label:
                    current = tgt
                    break
            trace.append(self._H.get(current, 0.0))
        return trace
