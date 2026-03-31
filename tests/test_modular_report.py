"""Tests for the modular_report module and CLI modular subcommand."""

from __future__ import annotations

import json

import pytest

from reticulate.cli import main
from reticulate.modular_report import ModularityReport, generate_report
from reticulate.parser import parse
from reticulate.statespace import build_statespace

from tests.benchmarks.p104_self import P104_BENCHMARKS, P104Benchmark


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Test generate_report on P104 self-referencing benchmarks."""

    @pytest.mark.parametrize(
        "bench", P104_BENCHMARKS, ids=[b.name for b in P104_BENCHMARKS]
    )
    def test_report_created(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        assert isinstance(report, ModularityReport)
        assert report.protocol_name == bench.name

    @pytest.mark.parametrize(
        "bench", P104_BENCHMARKS, ids=[b.name for b in P104_BENCHMARKS]
    )
    def test_text_output(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        text = report.to_text()
        assert bench.name in text
        assert "CERTIFICATE" in text
        assert "Verdict:" in text
        assert "Metrics:" in text

    @pytest.mark.parametrize(
        "bench", P104_BENCHMARKS, ids=[b.name for b in P104_BENCHMARKS]
    )
    def test_json_output(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        j = report.to_json()
        data = json.loads(j)
        assert data["protocol_name"] == bench.name
        assert data["is_lattice"] == bench.expected_lattice
        assert data["is_distributive"] == bench.expected_distributive
        assert "metrics" in data

    @pytest.mark.parametrize(
        "bench", P104_BENCHMARKS, ids=[b.name for b in P104_BENCHMARKS]
    )
    def test_dict_output(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        d = report.to_dict()
        assert d["metrics"]["states"] == bench.expected_states

    @pytest.mark.parametrize(
        "bench", P104_BENCHMARKS, ids=[b.name for b in P104_BENCHMARKS]
    )
    def test_dot_output(self, bench: P104Benchmark) -> None:
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        dot = report.to_dot()
        assert "digraph" in dot


class TestReportVerdict:
    """Verify that verdicts match expected modularity."""

    def test_cli_modular(self) -> None:
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["CLI"]
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        d = report.to_dict()
        assert d["verdict"] == "MODULAR"

    def test_importer_not_modular(self) -> None:
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["Importer"]
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        d = report.to_dict()
        # algebraically modular but not distributive
        assert d["verdict"] in ("WEAKLY_MODULAR", "NOT_MODULAR")
        assert "diagnosis" in d

    def test_importer_text_has_diagnosis(self) -> None:
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["Importer"]
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        text = report.to_text()
        assert "Non-modularity diagnosis:" in text
        assert "Refactoring suggestions:" in text

    def test_importer_json_has_refactorings(self) -> None:
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["Importer"]
        ast = parse(bench.type_string)
        ss = build_statespace(ast)
        report = generate_report(bench.type_string, ss, protocol_name=bench.name)
        d = report.to_dict()
        assert "refactorings" in d
        assert len(d["refactorings"]) > 0


# ---------------------------------------------------------------------------
# CLI modular subcommand
# ---------------------------------------------------------------------------


class TestCLIModular:
    """Test the `reticulate modular` CLI subcommand."""

    def test_text_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["modular", "&{a: end, b: end}"])
        out = capsys.readouterr().out
        assert "CERTIFICATE" in out
        assert "Verdict:" in out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["modular", "--format", "json", "&{a: end, b: end}"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["is_lattice"] is True
        assert "metrics" in data

    def test_dot_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["modular", "--format", "dot", "&{a: end, b: end}"])
        out = capsys.readouterr().out
        assert "digraph" in out

    def test_named_protocol(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["modular", "--name", "SMTP", "rec X . &{mail: &{send: X}, quit: end}"])
        out = capsys.readouterr().out
        assert "SMTP" in out

    def test_modular_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["modular", "&{a: end}"])
        out = capsys.readouterr().out
        assert "MODULAR" in out

    def test_non_modular_verdict(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([
            "modular",
            "&{a: +{OK: end, ERR: end}, b: +{OK: end, ERR: end}, c: +{OK: end, ERR: end}}",
        ])
        out = capsys.readouterr().out
        # Should not say MODULAR (distributive) — it's a diamond
        assert "NOT MODULAR" in out or "WEAKLY MODULAR" in out

    def test_parse_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["modular", "{{invalid}}"])
        assert exc_info.value.code == 1

    def test_no_input_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["modular"])
        assert exc_info.value.code == 1

    def test_file_input(self, capsys: pytest.CaptureFixture[str], tmp_path) -> None:
        f = tmp_path / "proto.st"
        f.write_text("&{open: &{close: end}}")
        main(["modular", "-f", str(f)])
        out = capsys.readouterr().out
        assert "CERTIFICATE" in out

    def test_p104_cli_self(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The CLI analyzes its own protocol session type."""
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["CLI"]
        main(["modular", "--name", "P104-CLI", bench.type_string])
        out = capsys.readouterr().out
        assert "P104-CLI" in out
        assert "MODULAR" in out

    def test_p104_importer_self(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The CLI analyzes its own Importer protocol — non-modular!"""
        from tests.benchmarks.p104_self import P104_BY_COMPONENT
        bench = P104_BY_COMPONENT["Importer"]
        main(["modular", "--name", "P104-Importer", "--format", "json", bench.type_string])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["is_distributive"] is False
        assert "diagnosis" in data


# ---------------------------------------------------------------------------
# Legacy CLI still works
# ---------------------------------------------------------------------------


class TestLegacyCLI:
    """Ensure the original CLI interface is unbroken."""

    def test_positional_type_string(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["&{a: end, b: end}"])
        out = capsys.readouterr().out
        assert "Lattice check:" in out

    def test_dot_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--dot", "&{a: end}"])
        out = capsys.readouterr().out
        assert "digraph" in out

    def test_distributive_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--distributive", "&{a: end}"])
        out = capsys.readouterr().out
        assert "Distributivity:" in out
