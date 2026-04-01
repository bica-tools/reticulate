"""Tests for CLI check and extract subcommands (Step 15c)."""

import pytest
import json
from reticulate.cli import main


class TestCheckSubcommand:
    def test_valid_trace_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            main(["check",
                  "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
                  "hasNext,TRUE,next,hasNext,FALSE"])
        assert exc.value.code == 0

    def test_violation_exits_one(self):
        with pytest.raises(SystemExit) as exc:
            main(["check",
                  "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
                  "next"])
        assert exc.value.code == 1

    def test_protocol_flag(self):
        with pytest.raises(SystemExit) as exc:
            main(["check", "--protocol", "java_iterator", "hasNext,FALSE"])
        assert exc.value.code == 0

    def test_protocol_violation(self):
        with pytest.raises(SystemExit) as exc:
            main(["check", "--protocol", "java_iterator", "next"])
        assert exc.value.code == 1

    def test_json_output(self, capsys):
        with pytest.raises(SystemExit):
            main(["check", "--json",
                  "rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}",
                  "hasNext,FALSE"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["conforms"] is True

    def test_unknown_protocol(self):
        with pytest.raises(SystemExit) as exc:
            main(["check", "--protocol", "nonexistent", "foo"])
        assert exc.value.code == 1


class TestExtractSubcommand:
    def test_list_protocols(self, capsys):
        main(["extract", "--list"])
        out = capsys.readouterr().out
        assert "java_iterator" in out
        assert "python_file" in out

    def test_extract_protocol(self, capsys):
        main(["extract", "--protocol", "java_iterator"])
        out = capsys.readouterr().out
        assert "hasNext" in out

    def test_extract_with_analyze(self, capsys):
        main(["extract", "--protocol", "python_file", "--analyze"])
        out = capsys.readouterr().out
        assert "distributive" in out.lower() or "Distributive" in out

    def test_unknown_protocol(self):
        with pytest.raises(SystemExit) as exc:
            main(["extract", "--protocol", "nonexistent"])
        assert exc.value.code == 1


class TestExistingCLI:
    """Ensure existing CLI functionality still works."""

    def test_legacy_analysis(self, capsys):
        main(["&{a: end, b: end}"])
        out = capsys.readouterr().out
        assert "lattice" in out.lower() or "Lattice" in out or "states" in out.lower()

    def test_version(self):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
