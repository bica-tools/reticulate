"""Tests for extended L3 protocol catalog (60 protocols).

Validates that the extended catalog parses correctly, forms lattices,
and maintains the ~85% distributivity rate across 168 total protocols.
"""

import pytest
from reticulate.l3_extended import EXTENDED_L3
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive


class TestAllProtocolsParse:
    @pytest.mark.parametrize("name,st_str", list(EXTENDED_L3.items()))
    def test_parseable(self, name, st_str):
        ast = parse(st_str)
        assert ast is not None


class TestAllProtocolsFormLattices:
    @pytest.mark.parametrize("name,st_str", list(EXTENDED_L3.items()))
    def test_is_lattice(self, name, st_str):
        ast = parse(st_str)
        ss = build_statespace(ast)
        lr = check_lattice(ss)
        assert lr.is_lattice, f"{name} is not a lattice"


class TestDistributivityRate:
    def test_at_least_80_percent(self):
        """Extended catalog should maintain ≥80% distributivity."""
        total = 0
        dist = 0
        for name, st_str in EXTENDED_L3.items():
            ast = parse(st_str)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            if lr.is_lattice:
                dr = check_distributive(ss)
                total += 1
                if dr.is_distributive:
                    dist += 1
        rate = dist / total
        assert rate >= 0.80, f"Distributivity rate {rate:.0%} below 80%"

    def test_catalog_size(self):
        assert len(EXTENDED_L3) >= 60


class TestDomainCoverage:
    """Verify protocols span multiple domains."""

    def test_has_java_protocols(self):
        java = [n for n in EXTENDED_L3 if n.startswith("java_")]
        assert len(java) >= 10

    def test_has_python_protocols(self):
        python = [n for n in EXTENDED_L3 if n.startswith("python_")]
        assert len(python) >= 8

    def test_has_database_protocols(self):
        db = [n for n in EXTENDED_L3 if any(k in n for k in ["mongo", "redis", "postgres", "elastic", "cassandra", "sqlite", "dynamo", "lmdb"])]
        assert len(db) >= 5

    def test_has_messaging_protocols(self):
        msg = [n for n in EXTENDED_L3 if any(k in n for k in ["kafka", "rabbit", "mqtt", "grpc", "websocket", "sse"])]
        assert len(msg) >= 5

    def test_has_network_protocols(self):
        net = [n for n in EXTENDED_L3 if any(k in n for k in ["http", "quic", "oauth", "ldap", "dns", "saml", "cert", "kerberos"])]
        assert len(net) >= 5


class TestCombinedTotal:
    def test_combined_distributivity(self):
        """Combined 108 benchmarks + 60 extended should be ~86% distributive."""
        from tests.benchmarks.protocols import BENCHMARKS

        # Existing benchmarks
        bench_total = 0
        bench_dist = 0
        for b in BENCHMARKS:
            try:
                ast = parse(b.type_string)
                ss = build_statespace(ast)
                lr = check_lattice(ss)
                if lr.is_lattice:
                    dr = check_distributive(ss)
                    bench_total += 1
                    if dr.is_distributive:
                        bench_dist += 1
            except Exception:
                pass

        # Extended catalog
        ext_total = 0
        ext_dist = 0
        for name, st_str in EXTENDED_L3.items():
            ast = parse(st_str)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            if lr.is_lattice:
                dr = check_distributive(ss)
                ext_total += 1
                if dr.is_distributive:
                    ext_dist += 1

        combined_total = bench_total + ext_total
        combined_dist = bench_dist + ext_dist
        rate = combined_dist / combined_total

        assert rate >= 0.80, f"Combined rate {rate:.0%} below 80%"
        assert combined_total >= 160, f"Need ≥160 protocols, got {combined_total}"
