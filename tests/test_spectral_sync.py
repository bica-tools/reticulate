"""Tests for spectral synchronization (Step 31d)."""

import pytest
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.spectral_sync import (
    extract_roles,
    role_coupling_matrix,
    sync_spectrum,
    max_coupling,
    isolation_score,
    analyze_sync,
)


def _build(s: str):
    return build_statespace(parse(s))


class TestRoleExtraction:
    def test_single_method(self):
        roles = extract_roles(_build("&{a: end}"))
        assert "a" in roles

    def test_multiple_methods(self):
        roles = extract_roles(_build("&{a: end, b: end}"))
        assert "a" in roles and "b" in roles

    def test_nested(self):
        roles = extract_roles(_build("&{a: &{b: end}}"))
        assert "a" in roles and "b" in roles


class TestCouplingMatrix:
    def test_single_method(self):
        M, roles = role_coupling_matrix(_build("&{a: end}"))
        assert len(M) == 1  # One role "a"
        assert M[0][0] >= 1

    def test_two_methods(self):
        M, roles = role_coupling_matrix(_build("&{a: end, b: end}"))
        assert len(M) == 2

    def test_symmetric(self):
        M, _ = role_coupling_matrix(_build("&{a: end, b: end}"))
        n = len(M)
        for i in range(n):
            for j in range(n):
                assert M[i][j] == M[j][i]


class TestSyncSpectrum:
    def test_single_role(self):
        spec = sync_spectrum(_build("&{a: end}"))
        assert len(spec) == 1

    def test_two_roles(self):
        spec = sync_spectrum(_build("&{a: end, b: end}"))
        assert len(spec) == 2


class TestMetrics:
    def test_max_coupling(self):
        mc = max_coupling(_build("&{a: end}"))
        assert mc >= 0

    def test_isolation(self):
        iso = isolation_score(_build("&{a: end}"))
        assert iso >= 0


class TestAnalyze:
    def test_end(self):
        r = analyze_sync(_build("end"))
        assert r.num_states == 1

    def test_branch(self):
        r = analyze_sync(_build("&{a: end, b: end}"))
        assert r.num_roles >= 2

    def test_parallel(self):
        r = analyze_sync(_build("(&{a: end} || &{b: end})"))
        assert r.num_roles >= 2


class TestBenchmarks:
    BENCHMARKS = [
        ("File Object", "&{open: &{read: end, write: end}}"),
        ("SMTP", "&{EHLO: &{MAIL: &{RCPT: &{DATA: end}}}}"),
        ("Two-choice", "&{a: end, b: end}"),
        ("Parallel", "(&{a: end} || &{b: end})"),
    ]

    @pytest.mark.parametrize("name,typ", BENCHMARKS)
    def test_sync_runs(self, name, typ):
        ss = _build(typ)
        r = analyze_sync(ss)
        assert r.num_states == len(ss.states)
        assert r.num_roles >= 1
        assert len(r.sync_spectrum) == r.num_roles
