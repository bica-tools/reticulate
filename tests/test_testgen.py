"""Tests for reticulate.testgen — test generation from state spaces."""

import pytest
from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace
from reticulate.testgen import (
    EnumerationResult,
    IncompletePrefix,
    TestGenConfig,
    ValidPath,
    ViolationPoint,
    enumerate,
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
        paths, _ = enumerate_valid_paths(ss("a . b . end"), 2)
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
        result = enumerate(ss("a . end"), TestGenConfig("Obj"))
        assert not result.truncated

    def test_chain_with_branch(self):
        paths, _ = enumerate_valid_paths(
            ss("open . &{read: close . end, write: close . end}"), 2)
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
        violations = enumerate_violations(ss("a . b . end"))
        assert len(violations) == 2

    def test_single_method_no_violations(self):
        violations = enumerate_violations(ss("a . end"))
        assert violations == []

    def test_violation_has_correct_prefix(self):
        violations = enumerate_violations(ss("a . b . end"))
        v_for_a = [v for v in violations if v.disabled_method == "a"]
        assert len(v_for_a) == 1
        assert v_for_a[0].prefix_labels == ["a"]

    def test_violation_enabled_methods(self):
        violations = enumerate_violations(ss("a . b . end"))
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
        space = ss("a . b . end")
        paths, _ = enumerate_valid_paths(space, 2)
        incomplete = enumerate_incomplete_prefixes(space, paths)
        assert len(incomplete) == 1
        assert incomplete[0].labels == ["a"]

    def test_longer_chain_multiple(self):
        space = ss("a . b . c . end")
        paths, _ = enumerate_valid_paths(space, 2)
        incomplete = enumerate_incomplete_prefixes(space, paths)
        assert len(incomplete) == 2
        label_sets = {tuple(p.labels) for p in incomplete}
        assert ("a",) in label_sets
        assert ("a", "b") in label_sets

    def test_deduplicates(self):
        space = ss("&{m: a.end, n: a.end}")
        paths, _ = enumerate_valid_paths(space, 2)
        incomplete = enumerate_incomplete_prefixes(space, paths)
        label_seqs = [tuple(p.labels) for p in incomplete]
        assert len(label_seqs) == len(set(label_seqs))

    def test_remaining_methods(self):
        space = ss("open . &{read: close . end, write: close . end}")
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
        out = gen("a . b . end")
        assert "validPath_a_b" in out
        assert "obj.a();" in out
        assert "obj.b();" in out

    def test_custom_class_name(self):
        out = gen("a . end", TestGenConfig("FileHandle"))
        assert "FileHandle obj = new FileHandle()" in out
        assert "class FileHandleProtocolTest" in out

    def test_custom_var_name(self):
        out = gen("a . end", TestGenConfig("Obj", var_name="sut"))
        assert "Obj sut = new Obj()" in out

    def test_violations_section(self):
        out = gen("a . b . end")
        assert "Violations" in out
        assert "@Disabled(" in out

    def test_incomplete_section(self):
        out = gen("a . b . end")
        assert "Incomplete protocols" in out

    def test_file_handle_e2e(self):
        out = gen("open . &{read: close . end, write: close . end}",
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
        out = gen("open . &{read: close . end, write: close . end}")
        assert out.count("{") == out.count("}")

    def test_truncation_warning(self):
        out = gen("&{a: end, b: end}", TestGenConfig("Obj", max_paths=1))
        assert "WARNING" in out
