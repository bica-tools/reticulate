"""Tests for the CLI module."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

from reticulate.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Run main(argv) and return captured stdout."""
    main(argv)
    return capsys.readouterr().out


def _run_err(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Run main(argv) expecting SystemExit; return (exit_code, stderr)."""
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    return exc_info.value.code, capsys.readouterr().err


# ---------------------------------------------------------------------------
# Default text output
# ---------------------------------------------------------------------------

class TestDefaultOutput:
    def test_basic_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["end"], capsys)
        assert "Session type: end" in out
        assert "1" in out  # 1 state
        assert "IS a lattice" in out

    def test_basic_chain(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["a . b . end"], capsys)
        assert "States: 3" in out
        assert "IS a lattice" in out

    def test_basic_diamond(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["&{m: a.end, n: b.end}"], capsys)
        # Branch with 2 choices + 2 continuations + end = several states
        assert "IS a lattice" in out

    def test_parallel(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["(a.end || b.end)"], capsys)
        assert "IS a lattice" in out

    def test_recursive(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["rec X . &{a: X, b: end}"], capsys)
        assert "IS a lattice" in out

    def test_top_reachable_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["end"], capsys)
        assert "Top reachable:" in out

    def test_bottom_reachable_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["end"], capsys)
        assert "Bottom reachable:" in out

    def test_meets_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["end"], capsys)
        assert "All meets exist:" in out

    def test_joins_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["end"], capsys)
        assert "All joins exist:" in out

    def test_scc_count_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["a . b . end"], capsys)
        assert "SCCs:" in out

    def test_transition_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["a . end"], capsys)
        assert "Transitions:" in out

    def test_pretty_prints_type(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The output should show the pretty-printed session type."""
        out = _run(["&{m: end, n: end}"], capsys)
        assert "Session type:" in out
        assert "&{m: end, n: end}" in out


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_parse_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, err = _run_err(["&{"], capsys)
        assert code == 1
        assert "Parse error" in err

    def test_unbound_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, err = _run_err(["X"], capsys)
        assert code == 1
        assert "unbound" in err.lower()

    def test_empty_string(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, err = _run_err([""], capsys)
        assert code == 1

    def test_invalid_syntax(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, err = _run_err(["&{m:}"], capsys)
        assert code == 1


# ---------------------------------------------------------------------------
# --dot output
# ---------------------------------------------------------------------------

class TestDotOutput:
    def test_dot_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["--dot", "a.end"], capsys)
        assert "digraph" in out
        assert "rankdir=TB" in out

    def test_dot_no_labels(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["--dot", "--no-labels", "a.end"], capsys)
        assert "digraph" in out
        # The output should not contain the label text "a" as a node label

    def test_dot_no_edge_labels(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["--dot", "--no-edge-labels", "a.end"], capsys)
        # Edge lines should not have label= attributes
        edge_lines = [l for l in out.splitlines() if "->" in l]
        for line in edge_lines:
            assert "label=" not in line

    def test_dot_with_title(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["--dot", "--title", "Test Title", "a.end"], capsys)
        assert "Test Title" in out

    def test_dot_diamond(self, capsys: pytest.CaptureFixture[str]) -> None:
        out = _run(["--dot", "&{m: a.end, n: b.end}"], capsys)
        assert "digraph" in out
        assert "->" in out


# ---------------------------------------------------------------------------
# --hasse output
# ---------------------------------------------------------------------------

class TestHasseOutput:
    @pytest.fixture()
    def tmpdir(self) -> str:
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_hasse_render(self, tmpdir: str, capsys: pytest.CaptureFixture[str]) -> None:
        """Render a Hasse diagram to a file (skipped if graphviz not available)."""
        try:
            import graphviz  # noqa: F401
        except ImportError:
            pytest.skip("graphviz Python package not installed")

        path = os.path.join(tmpdir, "out")
        out = _run(["--hasse", path, "a.end"], capsys)
        assert "Rendered to" in out

    def test_hasse_svg(self, tmpdir: str, capsys: pytest.CaptureFixture[str]) -> None:
        """Render a Hasse diagram as SVG."""
        try:
            import graphviz  # noqa: F401
        except ImportError:
            pytest.skip("graphviz Python package not installed")

        path = os.path.join(tmpdir, "out")
        out = _run(["--hasse", path, "--fmt", "svg", "a.end"], capsys)
        assert "Rendered to" in out

    def test_hasse_default_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--hasse with no path argument uses default."""
        try:
            import graphviz  # noqa: F401
        except ImportError:
            pytest.skip("graphviz Python package not installed")

        # Use tmpdir to avoid polluting the working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Put positional arg first so --hasse uses its const default
                out = _run(["a.end", "--hasse"], capsys)
                assert "Rendered to" in out
            finally:
                os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Module invocation (python -m reticulate)
# ---------------------------------------------------------------------------

class TestModuleInvocation:
    def test_reticulate_cli_module(self) -> None:
        """python -m reticulate.cli end → exit code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "reticulate.cli", "end"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0
        assert "IS a lattice" in result.stdout

    def test_reticulate_module(self) -> None:
        """python -m reticulate end → exit code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "reticulate", "end"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0
        assert "IS a lattice" in result.stdout

    def test_parse_error_exit_code(self) -> None:
        """Parse errors yield exit code 1."""
        result = subprocess.run(
            [sys.executable, "-m", "reticulate.cli", "&{"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 1
        assert "Parse error" in result.stderr
