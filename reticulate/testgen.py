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
# Client program tree — selection-aware test structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodCallNode:
    """A method call followed by the next program step."""
    label: str
    next: 'ClientProgram'


@dataclass(frozen=True)
class SelectionSwitchNode:
    """A method call whose return value determines the branch via switch."""
    method_label: str
    branches: dict[str, 'ClientProgram']  # label → sub-program


@dataclass(frozen=True)
class TerminalNode:
    """Protocol complete."""
    pass


ClientProgram = MethodCallNode | SelectionSwitchNode | TerminalNode


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
# Client program enumeration — tree-shaped DFS with selection switches
# ---------------------------------------------------------------------------


def enumerate_client_programs(
    ss: StateSpace, max_revisits: int = 2, max_paths: int = 100,
) -> tuple[list[ClientProgram], bool]:
    """Build tree-shaped client programs with selection switches.

    At branch (&) states the client decides — separate programs.
    At selection (+) states after a method — single SelectionSwitch node.
    Uses zip matching across selection arms (not cartesian product).

    Returns (programs, truncated).
    """
    visit_counts: dict[int, int] = {}
    programs = _build_from_state(ss, ss.top, visit_counts, max_revisits)
    truncated = len(programs) > max_paths
    if truncated:
        programs = programs[:max_paths]
    return programs, truncated


def _is_pure_selection_state(ss: StateSpace, state: int) -> bool:
    """True if state has only SELECTION transitions."""
    methods = list(ss.enabled_methods(state))
    selections = ss.enabled_selections(state)
    return not methods and bool(selections)


def _build_from_state(
    ss: StateSpace, state: int,
    visit_counts: dict[int, int], max_revisits: int,
) -> list[ClientProgram]:
    """Recursively build all client programs from a state."""
    if state == ss.bottom:
        return [TerminalNode()]

    count = visit_counts.get(state, 0)
    if count > max_revisits:
        return []

    visit_counts[state] = count + 1

    method_transitions = [
        (label, target) for label, target in ss.enabled(state)
        if not ss.is_selection(state, label, target)
    ]
    selection_transitions = [
        (label, target) for label, target in ss.enabled(state)
        if ss.is_selection(state, label, target)
    ]

    result: list[ClientProgram] = []

    if method_transitions:
        for label, target in method_transitions:
            if _is_pure_selection_state(ss, target):
                result.extend(_build_selection_switch(
                    ss, label, target, visit_counts, max_revisits))
            else:
                subs = _build_from_state(ss, target, visit_counts, max_revisits)
                for sub in subs:
                    result.append(MethodCallNode(label, sub))
    elif selection_transitions:
        # Top-level selection — each branch is a separate program
        for _label, target in selection_transitions:
            result.extend(_build_from_state(ss, target, visit_counts, max_revisits))

    visit_counts[state] = count  # restore
    return result


def _build_selection_switch(
    ss: StateSpace, method: str, sel_state: int,
    visit_counts: dict[int, int], max_revisits: int,
) -> list[ClientProgram]:
    """Build SelectionSwitch programs using zip matching."""
    sel_transitions = [
        (label, target) for label, target in ss.enabled(sel_state)
        if ss.is_selection(sel_state, label, target)
    ]

    branch_programs: dict[str, list[ClientProgram]] = {}
    for label, target in sel_transitions:
        subs = _build_from_state(ss, target, visit_counts, max_revisits)
        if not subs:
            subs = [TerminalNode()]  # dead-end bounded by max_revisits
        branch_programs[label] = subs

    return _zip_selection_branches(method, branch_programs)


def _zip_selection_branches(
    method: str, branch_programs: dict[str, list[ClientProgram]],
) -> list[ClientProgram]:
    """Zip selection branch programs: max(|branch_i|) switches, cycling shorter lists."""
    max_len = max((len(v) for v in branch_programs.values()), default=0)
    if max_len == 0:
        return []

    keys = list(branch_programs.keys())
    result: list[ClientProgram] = []
    for i in range(max_len):
        branches = {}
        for key in keys:
            lst = branch_programs[key]
            branches[key] = lst[i % len(lst)]
        result.append(SelectionSwitchNode(method, branches))
    return result


def client_program_name_suffix(cp: ClientProgram) -> str:
    """Collect METHOD labels depth-first from a client program tree."""
    labels: list[str] = []
    _collect_method_labels(cp, labels)
    return "_".join(labels)


def _collect_method_labels(cp: ClientProgram, labels: list[str]) -> None:
    if isinstance(cp, TerminalNode):
        pass
    elif isinstance(cp, MethodCallNode):
        labels.append(cp.label)
        _collect_method_labels(cp.next, labels)
    elif isinstance(cp, SelectionSwitchNode):
        labels.append(cp.method_label)
        for branch in cp.branches.values():
            _collect_method_labels(branch, labels)


def _result_var_name(method_label: str, var_counts: dict[str, int]) -> str:
    """Generate unique result variable name. First: mResult, then mResult2, etc."""
    count = var_counts.get(method_label, 0) + 1
    var_counts[method_label] = count
    return f"{method_label}Result" if count == 1 else f"{method_label}Result{count}"


def _append_client_program(
    lines: list[str], cp: ClientProgram,
    var_name: str, indent: str, var_counts: dict[str, int],
) -> None:
    """Recursively emit Java code for a client program tree."""
    if isinstance(cp, TerminalNode):
        pass
    elif isinstance(cp, MethodCallNode):
        lines.append(f"{indent}{var_name}.{cp.label}();")
        _append_client_program(lines, cp.next, var_name, indent, var_counts)
    elif isinstance(cp, SelectionSwitchNode):
        result_var = _result_var_name(cp.method_label, var_counts)
        lines.append(f"{indent}var {result_var} = {var_name}.{cp.method_label}();")
        lines.append(f"{indent}switch ({result_var}) {{")
        for label, branch in cp.branches.items():
            lines.append(f"{indent}    case {label} -> {{")
            _append_client_program(lines, branch, var_name, indent + "        ", var_counts)
            lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")


# ---------------------------------------------------------------------------
# Test source generation
# ---------------------------------------------------------------------------


def generate_test_source(
    ss: StateSpace, config: TestGenConfig, session_type: str,
) -> str:
    """Generate JUnit 5 test class source from a state space."""
    result = enumerate(ss, config)
    programs, cp_truncated = enumerate_client_programs(
        ss, config.max_revisits, config.max_paths)
    lines: list[str] = []

    # Package
    if config.package_name:
        lines.append(f"package {config.package_name};\n")

    # Imports
    lines.append("import org.junit.jupiter.api.Test;")
    lines.append("import org.junit.jupiter.api.Disabled;")
    lines.append("import static org.junit.jupiter.api.Assertions.*;")
    lines.append("")

    # Javadoc
    lines.append("/**")
    lines.append(f" * Protocol conformance tests for {config.class_name}.")
    lines.append(f" * Session type: {session_type}")
    lines.append(" * Generated by BICA Reborn test generator.")
    lines.append(" */")
    lines.append(f"class {config.class_name}ProtocolTest {{")

    # Valid paths (client programs with switch statements)
    lines.append("")
    lines.append(f"    // ===== Valid paths ({len(programs)}) =====")
    if cp_truncated:
        lines.append(f"")
        lines.append(f"    // WARNING: path enumeration truncated at {config.max_paths} paths")
    for program in programs:
        suffix = client_program_name_suffix(program)
        suffix = suffix or "empty"
        lines.append("")
        lines.append("    @Test")
        lines.append(f"    void validPath_{suffix}() {{")
        lines.append(f"        {config.class_name} {config.var_name} = new {config.class_name}();")
        var_counts: dict[str, int] = {}
        _append_client_program(lines, program, config.var_name, "        ", var_counts)
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
        has_selection = any(s.kind == "selection" for s in v.prefix_path)
        lines.append("")
        if has_selection:
            lines.append(
                '    @Disabled("Selection-dependent: object may choose different branch")'
            )
        lines.append("    @Test")
        lines.append(f"    void violation_{suffix}() {{")
        lines.append(f"        {config.class_name} {config.var_name} = new {config.class_name}();")
        for step in v.prefix_path:
            if step.kind == "selection":
                lines.append(f"        // -> {step.label} (selected by object)")
            else:
                lines.append(f"        {config.var_name}.{step.label}();")
        if has_selection:
            lines.append(f"        {config.var_name}.{v.disabled_method}(); // VIOLATION: not enabled here")
        else:
            lines.append(
                f"        assertThrows(IllegalStateException.class, "
                f"() -> {config.var_name}.{v.disabled_method}());"
            )
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
