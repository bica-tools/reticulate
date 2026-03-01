"""Test generation from session type state spaces.

Produces JUnit 5 protocol conformance tests in three categories:
1. Valid paths — bounded DFS from top to bottom
2. Violations — BFS prefix + disabled methods at each reachable state
3. Incomplete prefixes — proper prefixes of valid paths not at bottom
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import NamedTuple

from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Step(NamedTuple):
    """A single step in a path: transition label, target state, and kind."""
    label: str
    target: int
    kind: str = "method"  # "method" or "selection"


@dataclass(frozen=True)
class ValidPath:
    """A complete top-to-bottom valid path."""
    steps: tuple[Step, ...]

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.steps]


@dataclass(frozen=True)
class ViolationPoint:
    """A reachable state where a method is disabled."""
    state: int
    disabled_method: str
    enabled_methods: frozenset[str]
    prefix_path: tuple[Step, ...]

    @property
    def prefix_labels(self) -> list[str]:
        return [s.label for s in self.prefix_path]


@dataclass(frozen=True)
class IncompletePrefix:
    """An incomplete protocol execution (not at bottom)."""
    steps: tuple[Step, ...]
    remaining_methods: frozenset[str]

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.steps]


@dataclass(frozen=True)
class EnumerationResult:
    """Result of path enumeration."""
    valid_paths: list[ValidPath]
    violations: list[ViolationPoint]
    incomplete_prefixes: list[IncompletePrefix]
    truncated: bool


@dataclass(frozen=True)
class TestGenConfig:
    """Configuration for test generation."""
    class_name: str
    package_name: str | None = None
    var_name: str = "obj"
    max_revisits: int = 2
    max_paths: int = 100


# ---------------------------------------------------------------------------
# Path enumeration
# ---------------------------------------------------------------------------


def enumerate_valid_paths(
    ss: StateSpace, max_revisits: int = 2, max_paths: int = 100,
) -> tuple[list[ValidPath], bool]:
    """Bounded DFS from top to bottom. Returns (paths, truncated)."""
    paths: list[ValidPath] = []
    visit_counts: dict[int, int] = {}
    current_path: list[Step] = []
    truncated = _dfs(ss, ss.top, current_path, visit_counts, paths,
                     max_revisits, max_paths)
    return paths, truncated


def _dfs(
    ss: StateSpace, state: int, path: list[Step],
    visit_counts: dict[int, int], paths: list[ValidPath],
    max_revisits: int, max_paths: int,
) -> bool:
    if state == ss.bottom:
        paths.append(ValidPath(tuple(path)))
        return len(paths) >= max_paths

    count = visit_counts.get(state, 0)
    if count > max_revisits:
        return False

    visit_counts[state] = count + 1
    for label, target in ss.enabled(state):
        kind = "selection" if ss.is_selection(state, label, target) else "method"
        path.append(Step(label, target, kind))
        done = _dfs(ss, target, path, visit_counts, paths,
                     max_revisits, max_paths)
        path.pop()
        if done:
            visit_counts[state] = count
            return True
    visit_counts[state] = count
    return False


def enumerate_violations(ss: StateSpace) -> list[ViolationPoint]:
    """BFS from top to find shortest prefix, then identify disabled methods.

    Only METHOD labels are considered for violations. Pure selection states
    (no methods enabled, only selections) are skipped entirely.
    """
    # Only METHOD labels — selection labels are not callable by the client
    all_method_labels = {
        label for src, label, tgt in ss.transitions
        if not ss.is_selection(src, label, tgt)
    }
    shortest = _bfs_shortest_prefixes(ss)
    violations: list[ViolationPoint] = []
    for state, prefix in shortest.items():
        if state == ss.bottom:
            continue
        enabled_methods = frozenset(
            label for label, _ in ss.enabled_methods(state)
        )
        enabled_selections = ss.enabled_selections(state)
        # Skip pure selection states: the client has no agency here
        if not enabled_methods and enabled_selections:
            continue
        disabled = sorted(all_method_labels - enabled_methods)
        for method in disabled:
            violations.append(ViolationPoint(
                state=state,
                disabled_method=method,
                enabled_methods=enabled_methods,
                prefix_path=prefix,
            ))
    return violations


def _bfs_shortest_prefixes(ss: StateSpace) -> OrderedDict[int, tuple[Step, ...]]:
    prefixes: OrderedDict[int, tuple[Step, ...]] = OrderedDict()
    prefixes[ss.top] = ()
    queue = [ss.top]
    while queue:
        state = queue.pop(0)
        current = prefixes[state]
        for label, target in ss.enabled(state):
            if target not in prefixes:
                kind = "selection" if ss.is_selection(state, label, target) else "method"
                prefixes[target] = current + (Step(label, target, kind),)
                queue.append(target)
    return prefixes


def enumerate_incomplete_prefixes(
    ss: StateSpace, valid_paths: list[ValidPath],
) -> list[IncompletePrefix]:
    """Proper prefixes of valid paths where final state != bottom."""
    seen: set[tuple[str, ...]] = set()
    result: list[IncompletePrefix] = []
    for path in valid_paths:
        steps = path.steps
        for length in range(1, len(steps)):
            prefix = steps[:length]
            label_seq = tuple(s.label for s in prefix)
            if label_seq in seen:
                continue
            seen.add(label_seq)
            final_state = prefix[-1].target
            if final_state != ss.bottom:
                remaining = frozenset(label for label, _ in ss.enabled(final_state))
                result.append(IncompletePrefix(prefix, remaining))
    return result


def enumerate(ss: StateSpace, config: TestGenConfig) -> EnumerationResult:
    """Enumerate all three path categories."""
    paths, truncated = enumerate_valid_paths(ss, config.max_revisits, config.max_paths)
    violations = enumerate_violations(ss)
    incomplete = enumerate_incomplete_prefixes(ss, paths)
    return EnumerationResult(paths, violations, incomplete, truncated)


# ---------------------------------------------------------------------------
# Test source generation
# ---------------------------------------------------------------------------


def generate_test_source(
    ss: StateSpace, config: TestGenConfig, session_type: str,
) -> str:
    """Generate JUnit 5 test class source from a state space."""
    result = enumerate(ss, config)
    lines: list[str] = []

    # Package
    if config.package_name:
        lines.append(f"package {config.package_name};\n")

    # Imports
    lines.append("import org.junit.jupiter.api.Test;")
    lines.append("import org.junit.jupiter.api.Disabled;")
    lines.append("")

    # Javadoc
    lines.append("/**")
    lines.append(f" * Protocol conformance tests for {config.class_name}.")
    lines.append(f" * Session type: {session_type}")
    lines.append(" * Generated by BICA Reborn test generator.")
    lines.append(" */")
    lines.append(f"class {config.class_name}ProtocolTest {{")

    # Valid paths
    lines.append("")
    lines.append(f"    // ===== Valid paths ({len(result.valid_paths)}) =====")
    if result.truncated:
        lines.append(f"")
        lines.append(f"    // WARNING: path enumeration truncated at {config.max_paths} paths")
    for path in result.valid_paths:
        labels = path.labels
        suffix = "empty" if not labels else "_".join(labels)
        lines.append("")
        lines.append("    @Test")
        lines.append(f"    void validPath_{suffix}() {{")
        lines.append(f"        {config.class_name} {config.var_name} = new {config.class_name}();")
        for step in path.steps:
            if step.kind == "selection":
                lines.append(f"        // -> {step.label} (selected by object)")
            else:
                lines.append(f"        {config.var_name}.{step.label}();")
        lines.append("    }")

    # Violations
    lines.append("")
    lines.append(f"    // ===== Violations ({len(result.violations)}) =====")
    for v in result.violations:
        prefix_labels = v.prefix_labels
        enabled_sorted = sorted(v.enabled_methods)
        state_desc = ("initial state" if not prefix_labels
                      else "state after [" + ", ".join(prefix_labels) + "]")
        enabled_str = "[" + ", ".join(enabled_sorted) + "]"
        suffix = (("initial_" + v.disabled_method) if not prefix_labels
                  else "_".join(prefix_labels) + "_" + v.disabled_method)
        lines.append("")
        lines.append(
            f'    @Disabled("Protocol violation: \'{v.disabled_method}\''
            f' not enabled in {state_desc}'
            f'; enabled: {enabled_str}")'
        )
        lines.append("    @Test")
        lines.append(f"    void violation_{suffix}() {{")
        lines.append(f"        {config.class_name} {config.var_name} = new {config.class_name}();")
        for step in v.prefix_path:
            if step.kind == "selection":
                lines.append(f"        // -> {step.label} (selected by object)")
            else:
                lines.append(f"        {config.var_name}.{step.label}();")
        lines.append(f"        {config.var_name}.{v.disabled_method}(); // VIOLATION: not enabled here")
        lines.append("    }")

    # Incomplete
    lines.append("")
    lines.append(f"    // ===== Incomplete protocols ({len(result.incomplete_prefixes)}) =====")
    for p in result.incomplete_prefixes:
        labels = p.labels
        remaining_sorted = sorted(p.remaining_methods)
        labels_str = "[" + ", ".join(labels) + "]"
        remaining_str = "[" + ", ".join(remaining_sorted) + "]"
        suffix = "_".join(labels)
        lines.append("")
        lines.append(
            f'    @Disabled("Incomplete: object not in terminal state after {labels_str}")'
        )
        lines.append("    @Test")
        lines.append(f"    void incomplete_{suffix}() {{")
        lines.append(f"        {config.class_name} {config.var_name} = new {config.class_name}();")
        for step in p.steps:
            if step.kind == "selection":
                lines.append(f"        // -> {step.label} (selected by object)")
            else:
                lines.append(f"        {config.var_name}.{step.label}();")
        lines.append(f"        // NOT COMPLETE: enabled methods remain: {remaining_str}")
        lines.append("    }")

    lines.append("}")
    lines.append("")
    return "\n".join(lines)
