"""Tests for reticulate.testgen — test generation from state spaces."""

import pytest
from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.testgen import (
    ClientProgram,
    EnumerationResult,
    IncompletePrefix,
    MethodCallNode,
    SelectionSwitchNode,
    TerminalNode,
    TestGenConfig,
    ValidPath,
    ViolationPoint,
    enumerate,
    enumerate_client_programs,
    enumerate_incomplete_prefixes,
    enumerate_valid_paths,
    enumerate_violations,
    generate_test_source,
)


def ss(source: str):
    return build_statespace(parse(source))


def config(max_revisits: int = 2, max_paths: int = 100) -> TestGenConfig:
    return TestGenConfig("Obj", max_revisits=max_revisits, max_paths=max_paths)


def gen(source: str, cfg: TestGenConfig | None = None) -> str:
    ast = parse(source)
    c = cfg or TestGenConfig("Obj")
    return generate_test_source(build_statespace(ast), c, pretty(ast))


# =========================================================================
# Valid paths
# =========================================================================


class TestValidPaths:
    def test_end_type_single_empty_path(self):
        paths, _ = enumerate_valid_paths(ss("end"), 2)
        assert len(paths) == 1
        assert paths[0].labels == []

    def test_simple_chain_one_path(self):
        paths, _ = enumerate_valid_paths(ss("&{a: &{b: end}}"), 2)
        assert len(paths) == 1
        assert paths[0].labels == ["a", "b"]

    def test_branch_two_paths(self):
        paths, _ = enumerate_valid_paths(ss("&{m: end, n: end}"), 2)
        assert len(paths) == 2
        label_sets = {tuple(p.labels) for p in paths}
        assert ("m",) in label_sets
        assert ("n",) in label_sets

    def test_recursion_max_revisits_zero(self):
        paths, _ = enumerate_valid_paths(ss("rec X . &{next: X, done: end}"), 0)
        assert len(paths) == 1
        assert paths[0].labels == ["done"]

    def test_recursion_max_revisits_one(self):
        paths, _ = enumerate_valid_paths(ss("rec X . &{next: X, done: end}"), 1)
        assert len(paths) == 2

    def test_recursion_max_revisits_two(self):
        paths, _ = enumerate_valid_paths(ss("rec X . &{next: X, done: end}"), 2)
        assert len(paths) == 3

    def test_max_paths_truncation(self):
        result = enumerate(ss("&{a: end, b: end, c: end, d: end}"),
                           TestGenConfig("Obj", max_paths=2))
        assert len(result.valid_paths) == 2
        assert result.truncated

    def test_no_truncation(self):
        result = enumerate(ss("&{a: end}"), TestGenConfig("Obj"))
        assert not result.truncated

    def test_chain_with_branch(self):
        paths, _ = enumerate_valid_paths(
            ss("&{open: &{read: &{close: end}, write: &{close: end}}}"), 2)
        assert len(paths) == 2
        for p in paths:
            assert p.labels[0] == "open"
            assert p.labels[-1] == "close"


# =========================================================================
# Violations
# =========================================================================


class TestViolations:
    def test_end_type_no_violations(self):
        violations = enumerate_violations(ss("end"))
        assert violations == []

    def test_simple_chain_violations(self):
        violations = enumerate_violations(ss("&{a: &{b: end}}"))
        assert len(violations) == 2

    def test_single_method_no_violations(self):
        violations = enumerate_violations(ss("&{a: end}"))
        assert violations == []

    def test_violation_has_correct_prefix(self):
        violations = enumerate_violations(ss("&{a: &{b: end}}"))
        v_for_a = [v for v in violations if v.disabled_method == "a"]
        assert len(v_for_a) == 1
        assert v_for_a[0].prefix_labels == ["a"]

    def test_violation_enabled_methods(self):
        violations = enumerate_violations(ss("&{a: &{b: end}}"))
        v_at_top = [v for v in violations if not v.prefix_path]
        assert len(v_at_top) == 1
        assert v_at_top[0].enabled_methods == frozenset({"a"})

    def test_recursive_all_enabled_no_violations(self):
        violations = enumerate_violations(ss("rec X . &{next: X, done: end}"))
        assert violations == []


# =========================================================================
# Incomplete prefixes
# =========================================================================


class TestIncompletePrefixes:
    def test_end_type_no_incomplete(self):
        paths, _ = enumerate_valid_paths(ss("end"), 2)
        incomplete = enumerate_incomplete_prefixes(ss("end"), paths)
        assert incomplete == []

    def test_simple_chain_one_incomplete(self):
        space = ss("&{a: &{b: end}}")
        paths, _ = enumerate_valid_paths(space, 2)
        incomplete = enumerate_incomplete_prefixes(space, paths)
        assert len(incomplete) == 1
        assert incomplete[0].labels == ["a"]

    def test_longer_chain_multiple(self):
        space = ss("&{a: &{b: &{c: end}}}")
        paths, _ = enumerate_valid_paths(space, 2)
        incomplete = enumerate_incomplete_prefixes(space, paths)
        assert len(incomplete) == 2
        label_sets = {tuple(p.labels) for p in incomplete}
        assert ("a",) in label_sets
        assert ("a", "b") in label_sets

    def test_deduplicates(self):
        space = ss("&{m: &{a: end}, n: &{a: end}}")
        paths, _ = enumerate_valid_paths(space, 2)
        incomplete = enumerate_incomplete_prefixes(space, paths)
        label_seqs = [tuple(p.labels) for p in incomplete]
        assert len(label_seqs) == len(set(label_seqs))

    def test_remaining_methods(self):
        space = ss("&{open: &{read: &{close: end}, write: &{close: end}}}")
        paths, _ = enumerate_valid_paths(space, 2)
        incomplete = enumerate_incomplete_prefixes(space, paths)
        after_open = [p for p in incomplete if p.labels == ["open"]]
        assert len(after_open) == 1
        assert after_open[0].remaining_methods == frozenset({"read", "write"})


# =========================================================================
# Test source generation
# =========================================================================


class TestGenerate:
    def test_contains_class(self):
        out = gen("end")
        assert "class ObjProtocolTest {" in out

    def test_contains_imports(self):
        out = gen("end")
        assert "import org.junit.jupiter.api.Test;" in out

    def test_contains_javadoc(self):
        out = gen("end")
        assert "Protocol conformance tests for Obj." in out

    def test_with_package(self):
        out = gen("end", TestGenConfig("Obj", package_name="com.example"))
        assert "package com.example;" in out

    def test_valid_path_chain(self):
        out = gen("&{a: &{b: end}}")
        assert "validPath_a_b" in out
        assert "obj.a();" in out
        assert "obj.b();" in out

    def test_custom_class_name(self):
        out = gen("&{a: end}", TestGenConfig("FileHandle"))
        assert "FileHandle obj = new FileHandle()" in out
        assert "class FileHandleProtocolTest" in out

    def test_custom_var_name(self):
        out = gen("&{a: end}", TestGenConfig("Obj", var_name="sut"))
        assert "Obj sut = new Obj()" in out

    def test_violations_section(self):
        out = gen("&{a: &{b: end}}")
        assert "Violations" in out
        assert "assertThrows(IllegalStateException.class" in out

    def test_violation_uses_assert_throws(self):
        """Clean violations (no selections in prefix) use assertThrows."""
        out = gen("&{a: &{b: end}}")
        assert "assertThrows(IllegalStateException.class, () -> obj." in out
        # No @Disabled on clean violations
        assert "@Disabled" not in out.split("// ===== Violations")[1].split("// ===== Incomplete")[0]

    def test_selection_dependent_violation_uses_assert_throws(self):
        """Violations with selection steps in prefix also use assertThrows."""
        out = gen("&{a: +{OK: &{b: end}, ERR: end}}")
        # No @Disabled on any violation
        violations_section = out.split("// ===== Violations")[1].split("// ===== Incomplete")[0]
        assert "@Disabled" not in violations_section
        # Selection steps are comments, violation uses assertThrows
        if "violation_" in violations_section:
            assert "assertThrows(IllegalStateException.class" in violations_section

    def test_assertions_import(self):
        """Generated source includes static Assertions import."""
        out = gen("&{a: end}")
        assert "import static org.junit.jupiter.api.Assertions.*;" in out

    def test_incomplete_section(self):
        out = gen("&{a: &{b: end}}")
        assert "Incomplete protocols" in out

    def test_file_handle_e2e(self):
        out = gen("&{open: &{read: &{close: end}, write: &{close: end}}}",
                  TestGenConfig("FileHandle"))
        assert "validPath_open_read_close" in out
        assert "validPath_open_write_close" in out
        assert "Valid paths (2)" in out

    def test_recursive_iterator(self):
        out = gen("rec X . &{next: X, done: end}",
                  TestGenConfig("Iter", package_name="com.example", max_revisits=1))
        assert "package com.example;" in out
        assert "validPath_done" in out
        assert "validPath_next_done" in out

    def test_balanced_braces(self):
        out = gen("&{open: &{read: &{close: end}, write: &{close: end}}}")
        assert out.count("{") == out.count("}")

    def test_truncation_warning(self):
        out = gen("&{a: end, b: end}", TestGenConfig("Obj", max_paths=1))
        assert "WARNING" in out

    def test_select_steps_emitted_as_switch(self):
        out = gen("&{m: +{OK: end, ERR: end}}")
        assert "var mResult = obj.m();" in out
        assert "switch (mResult)" in out
        assert "case OK ->" in out
        assert "case ERR ->" in out

    def test_select_only_no_violations(self):
        out = gen("+{OK: end, ERR: end}")
        assert "Violations (0)" in out

    def test_mixed_protocol_no_violations_at_selection_state(self):
        out = gen("&{m: +{OK: end, ERR: end}}")
        assert "Violations (0)" in out


# =========================================================================
# Step kind
# =========================================================================


class TestStepKind:
    def test_select_steps_have_selection_kind(self):
        paths, _ = enumerate_valid_paths(ss("+{OK: end, ERR: end}"), 2)
        for path in paths:
            for step in path.steps:
                assert step.kind == "selection"

    def test_branch_steps_have_method_kind(self):
        paths, _ = enumerate_valid_paths(ss("&{m: end, n: end}"), 2)
        for path in paths:
            for step in path.steps:
                assert step.kind == "method"

    def test_mixed_protocol_step_kinds(self):
        paths, _ = enumerate_valid_paths(ss("&{m: +{OK: end, ERR: end}}"), 2)
        for path in paths:
            assert path.steps[0].kind == "method"
            assert path.steps[1].kind == "selection"

    def test_select_only_no_violations(self):
        violations = enumerate_violations(ss("+{OK: end, ERR: end}"))
        assert violations == []

    def test_mixed_protocol_skips_pure_selection_state(self):
        violations = enumerate_violations(ss("&{m: +{OK: end, ERR: end}}"))
        assert violations == []


# =========================================================================
# Client programs — tree-shaped with selection switches
# =========================================================================


class TestClientPrograms:
    def test_end_type_single_terminal(self):
        programs, truncated = enumerate_client_programs(ss("end"), 2)
        assert len(programs) == 1
        assert isinstance(programs[0], TerminalNode)
        assert not truncated

    def test_simple_chain(self):
        programs, _ = enumerate_client_programs(ss("&{a: &{b: end}}"), 2)
        assert len(programs) == 1
        cp = programs[0]
        assert isinstance(cp, MethodCallNode)
        assert cp.label == "a"
        assert isinstance(cp.next, MethodCallNode)
        assert cp.next.label == "b"
        assert isinstance(cp.next.next, TerminalNode)

    def test_branch_two_programs(self):
        programs, _ = enumerate_client_programs(ss("&{m: end, n: end}"), 2)
        assert len(programs) == 2
        labels = {p.label for p in programs if isinstance(p, MethodCallNode)}
        assert labels == {"m", "n"}

    def test_simple_selection_switch(self):
        programs, _ = enumerate_client_programs(ss("&{m: +{OK: end, ERR: end}}"), 2)
        assert len(programs) == 1
        cp = programs[0]
        assert isinstance(cp, SelectionSwitchNode)
        assert cp.method_label == "m"
        assert set(cp.branches.keys()) == {"OK", "ERR"}
        for branch in cp.branches.values():
            assert isinstance(branch, TerminalNode)

    def test_selection_with_sub_branches_zip(self):
        programs, _ = enumerate_client_programs(
            ss("&{m: +{A: &{x: end, y: end}, B: end}}"), 2)
        assert len(programs) == 2
        for cp in programs:
            assert isinstance(cp, SelectionSwitchNode)
            assert cp.method_label == "m"
            assert "A" in cp.branches
            assert "B" in cp.branches

    def test_top_level_selection(self):
        programs, _ = enumerate_client_programs(ss("+{OK: end, ERR: end}"), 2)
        assert len(programs) == 2
        for cp in programs:
            assert isinstance(cp, TerminalNode)

    def test_recursive_selection(self):
        programs, _ = enumerate_client_programs(
            ss("rec X . &{m: +{A: X, B: end}}"), 2)
        assert len(programs) >= 1
        for cp in programs:
            assert isinstance(cp, SelectionSwitchNode)

    def test_max_paths_truncation(self):
        programs, truncated = enumerate_client_programs(
            ss("&{a: end, b: end, c: end, d: end}"), 2, max_paths=2)
        assert len(programs) == 2
        assert truncated

    def test_file_handle_no_selection(self):
        programs, _ = enumerate_client_programs(
            ss("&{open: &{read: &{close: end}, write: &{close: end}}}"), 2)
        assert len(programs) == 2
        for cp in programs:
            assert isinstance(cp, MethodCallNode)
            assert cp.label == "open"

    def test_nested_selection(self):
        programs, _ = enumerate_client_programs(
            ss("&{m: +{A: &{n: +{X: end, Y: end}}, B: end}}"), 2)
        assert len(programs) == 1
        cp = programs[0]
        assert isinstance(cp, SelectionSwitchNode)
        assert cp.method_label == "m"
        a_branch = cp.branches["A"]
        assert isinstance(a_branch, SelectionSwitchNode)
        assert a_branch.method_label == "n"

    def test_recursive_max_revisits_zero(self):
        programs, _ = enumerate_client_programs(
            ss("rec X . &{m: +{A: X, B: end}}"), 0)
        assert len(programs) == 1
        cp = programs[0]
        assert isinstance(cp, SelectionSwitchNode)
        # A branch should be Terminal (dead-end at max_revisits=0)
        assert isinstance(cp.branches["A"], TerminalNode)
        assert isinstance(cp.branches["B"], TerminalNode)

    def test_name_suffix_method_call(self):
        from reticulate.testgen import client_program_name_suffix
        cp = MethodCallNode("a", MethodCallNode("b", TerminalNode()))
        assert client_program_name_suffix(cp) == "a_b"

    def test_name_suffix_selection_switch(self):
        from reticulate.testgen import client_program_name_suffix
        cp = SelectionSwitchNode("m", {"OK": TerminalNode(), "ERR": TerminalNode()})
        assert client_program_name_suffix(cp) == "m"


# =========================================================================
# Generated source — switch statements
# =========================================================================


class TestGeneratedSwitchStatements:
    def test_switch_variable_declaration(self):
        out = gen("&{m: +{OK: end, ERR: end}}")
        assert "var mResult = obj.m();" in out

    def test_switch_keyword(self):
        out = gen("&{m: +{OK: end, ERR: end}}")
        assert "switch (mResult) {" in out

    def test_switch_case_labels(self):
        out = gen("&{m: +{OK: end, ERR: end}}")
        assert "case OK -> {" in out
        assert "case ERR -> {" in out

    def test_nested_switch_generation(self):
        out = gen("&{m: +{A: &{n: +{X: end, Y: end}}, B: end}}")
        assert "var mResult = obj.m();" in out
        assert "var nResult = obj.n();" in out
        assert "case A -> {" in out
        assert "case X -> {" in out

    def test_recursive_switch_variable_names(self):
        out = gen("rec X . &{m: +{A: X, B: end}}", TestGenConfig("Obj", max_revisits=1))
        assert "var mResult = obj.m();" in out
        assert "mResult2" in out

    def test_switch_balanced_braces(self):
        out = gen("&{m: +{A: &{n: +{X: end, Y: end}}, B: end}}")
        assert out.count("{") == out.count("}")

    def test_file_handle_no_switch(self):
        out = gen("&{open: &{read: &{close: end}, write: &{close: end}}}")
        assert "switch" not in out
        assert "obj.open();" in out

    def test_valid_path_count_with_selection(self):
        out = gen("&{m: +{OK: end, ERR: end}}")
        assert "Valid paths (1)" in out
