"""Tests for product_viz — grid, 3D, and factored visualization modes."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.product_viz import (
    _topo_order,
    _validate_product,
    factored_dashboard,
    factored_dot_source,
    grid_dot_source,
    render_3d_product,
    render_grid_hasse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def par2() -> StateSpace:
    """Binary parallel: (&{a: end} || &{b: end})."""
    return build_statespace(parse("(&{a: end} || &{b: end})"))


@pytest.fixture
def par3() -> StateSpace:
    """Ternary parallel: (&{a: end} || &{b: end} || &{c: end})."""
    return build_statespace(parse("(&{a: end} || &{b: end} || &{c: end})"))


@pytest.fixture
def non_parallel() -> StateSpace:
    """Non-parallel type: &{a: end, b: end}."""
    return build_statespace(parse("&{a: end, b: end}"))


@pytest.fixture
def par4() -> StateSpace:
    """4-factor parallel: (&{a: end} || &{b: end} || &{c: end} || &{d: end})."""
    return build_statespace(parse("(&{a: end} || &{b: end} || &{c: end} || &{d: end})"))


@pytest.fixture
def par2_deep() -> StateSpace:
    """Binary parallel with deeper factors."""
    return build_statespace(parse("(&{a: &{c: end}} || &{b: end})"))


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestProductMetadata:
    """Test that product metadata is correctly populated and propagated."""

    def test_binary_has_coords(self, par2: StateSpace) -> None:
        assert par2.product_coords is not None
        assert len(par2.product_coords) == len(par2.states)

    def test_binary_coords_are_2tuples(self, par2: StateSpace) -> None:
        assert par2.product_coords is not None
        for coord in par2.product_coords.values():
            assert len(coord) == 2

    def test_binary_has_2_factors(self, par2: StateSpace) -> None:
        assert par2.product_factors is not None
        assert len(par2.product_factors) == 2

    def test_ternary_has_coords(self, par3: StateSpace) -> None:
        assert par3.product_coords is not None
        assert len(par3.product_coords) == len(par3.states)

    def test_ternary_coords_are_3tuples(self, par3: StateSpace) -> None:
        assert par3.product_coords is not None
        for coord in par3.product_coords.values():
            assert len(coord) == 3

    def test_ternary_has_3_factors(self, par3: StateSpace) -> None:
        assert par3.product_factors is not None
        assert len(par3.product_factors) == 3

    def test_non_parallel_has_no_coords(self, non_parallel: StateSpace) -> None:
        assert non_parallel.product_coords is None
        assert non_parallel.product_factors is None

    def test_factor_statespaces_are_valid(self, par2: StateSpace) -> None:
        assert par2.product_factors is not None
        for f in par2.product_factors:
            assert isinstance(f, StateSpace)
            assert f.top in f.states
            assert f.bottom in f.states

    def test_coords_unique(self, par2: StateSpace) -> None:
        """Each state has a unique coordinate."""
        assert par2.product_coords is not None
        coords = list(par2.product_coords.values())
        assert len(coords) == len(set(coords))

    def test_top_bottom_coords(self, par2: StateSpace) -> None:
        """Top has factor tops, bottom has factor bottoms."""
        assert par2.product_coords is not None
        assert par2.product_factors is not None
        top_coord = par2.product_coords[par2.top]
        bot_coord = par2.product_coords[par2.bottom]
        for i, f in enumerate(par2.product_factors):
            assert top_coord[i] == f.top
            assert bot_coord[i] == f.bottom


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:

    def test_validate_non_parallel_raises(self, non_parallel: StateSpace) -> None:
        with pytest.raises(ValueError, match="not a parallel type"):
            _validate_product(non_parallel)

    def test_validate_wrong_ndim_raises(self, par2: StateSpace) -> None:
        with pytest.raises(ValueError, match="Expected 3 factors, got 2"):
            _validate_product(par2, ndim=3)

    def test_validate_correct_ndim(self, par2: StateSpace) -> None:
        _validate_product(par2, ndim=2)  # should not raise

    def test_validate_any_ndim(self, par2: StateSpace) -> None:
        _validate_product(par2)  # no ndim check, should not raise


# ---------------------------------------------------------------------------
# Grid tests (2D)
# ---------------------------------------------------------------------------

class TestGridDot:

    def test_valid_dot(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2)
        assert "digraph grid" in dot
        assert "pos=" in dot

    def test_neato_positions(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2)
        # Every node should have pos="x,y!" pinned
        for sid in par2.states:
            assert f'{sid} [label=' in dot

    def test_has_edges(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2)
        assert "->" in dot

    def test_title(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2, title="My Grid")
        assert "My Grid" in dot

    def test_no_labels(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2, labels=False)
        # Node labels should just be IDs
        for sid in par2.states:
            assert f'label="{sid}"' in dot

    def test_no_edge_labels(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2, edge_labels=False)
        # Edges should have no label attribute
        lines = [l for l in dot.split('\n') if '->' in l]
        for line in lines:
            assert 'label=' not in line

    def test_error_on_non_parallel(self, non_parallel: StateSpace) -> None:
        with pytest.raises(ValueError, match="not a parallel type"):
            grid_dot_source(non_parallel)

    def test_error_on_3d(self, par3: StateSpace) -> None:
        with pytest.raises(ValueError, match="Expected 2 factors, got 3"):
            grid_dot_source(par3)

    def test_top_bottom_colors(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2)
        assert "#dbeafe" in dot  # top fill
        assert "#dcfce7" in dot  # bottom fill

    def test_deep_factors(self, par2_deep: StateSpace) -> None:
        dot = grid_dot_source(par2_deep)
        assert "digraph grid" in dot

    @pytest.mark.skipif(
        subprocess.run(["which", "neato"], capture_output=True).returncode != 0,
        reason="neato not installed",
    )
    def test_render_grid(self, par2: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "grid_test")
            out = render_grid_hasse(par2, path, fmt="svg")
            assert os.path.exists(out)

    def test_all_states_present(self, par2: StateSpace) -> None:
        dot = grid_dot_source(par2)
        for sid in par2.states:
            assert f'{sid} [label=' in dot


# ---------------------------------------------------------------------------
# 3D tests
# ---------------------------------------------------------------------------

class TestThreeD:

    def test_valid_html(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            out = render_3d_product(par3, path)
            assert out.endswith(".html")
            with open(out) as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "three" in html.lower() or "THREE" in html

    def test_has_nodes_and_edges(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            out = render_3d_product(par3, path)
            with open(out) as f:
                html = f.read()
            assert "const nodes" in html
            assert "const edges" in html

    def test_title_in_html(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            render_3d_product(par3, path, title="My 3D View")
            with open(path + ".html") as f:
                html = f.read()
            assert "My 3D View" in html

    def test_axis_labels(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            render_3d_product(par3, path)
            with open(path + ".html") as f:
                html = f.read()
            assert "Factor 1" in html
            assert "Factor 2" in html
            assert "Factor 3" in html

    def test_error_on_non_parallel(self, non_parallel: StateSpace) -> None:
        with pytest.raises(ValueError, match="not a parallel type"):
            render_3d_product(non_parallel, "/tmp/nope")

    def test_error_on_2d(self, par2: StateSpace) -> None:
        with pytest.raises(ValueError, match="Expected at least 3 factors, got 2"):
            render_3d_product(par2, "/tmp/nope")

    def test_node_count(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            render_3d_product(par3, path)
            with open(path + ".html") as f:
                html = f.read()
            assert f"{len(par3.states)} states" in html

    def test_top_color(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            render_3d_product(par3, path)
            with open(path + ".html") as f:
                html = f.read()
            assert "#2563eb" in html  # top node color

    def test_bottom_color(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            render_3d_product(par3, path)
            with open(path + ".html") as f:
                html = f.read()
            assert "#16a34a" in html  # bottom node color

    def test_html_extension_appended(self, par3: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            out = render_3d_product(par3, path)
            assert out == path + ".html"

    def test_4d_product(self, par4: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test4d")
            out = render_3d_product(par4, path)
            with open(out) as f:
                html = f.read()
            assert "Color:" in html
            assert "Factor 4" in html
            assert f"{len(par4.states)} states" in html

    def test_4d_has_gradient_colors(self, par4: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test4d")
            render_3d_product(par4, path)
            with open(path + ".html") as f:
                html = f.read()
            # Should have orange-purple gradient colors (not just gray)
            assert "#f97316" in html or "#7c3aed" in html or "color:\"#" in html


# ---------------------------------------------------------------------------
# Factored view tests
# ---------------------------------------------------------------------------

class TestFactoredDot:

    def test_valid_dot(self, par2: StateSpace) -> None:
        dot = factored_dot_source(par2)
        assert "digraph factored" in dot

    def test_correct_cluster_count(self, par2: StateSpace) -> None:
        dot = factored_dot_source(par2)
        assert dot.count("subgraph cluster_dim") == 2

    def test_correct_cluster_count_3(self, par3: StateSpace) -> None:
        dot = factored_dot_source(par3)
        assert dot.count("subgraph cluster_dim") == 3

    def test_factor_state_counts(self, par2: StateSpace) -> None:
        assert par2.product_factors is not None
        dot = factored_dot_source(par2)
        for i, f in enumerate(par2.product_factors):
            # Each factor's states should be prefixed
            for sid in f.states:
                assert f"f{i}s{sid}" in dot

    def test_title(self, par2: StateSpace) -> None:
        dot = factored_dot_source(par2, title="My Factors")
        assert "My Factors" in dot

    def test_error_on_non_parallel(self, non_parallel: StateSpace) -> None:
        with pytest.raises(ValueError, match="not a parallel type"):
            factored_dot_source(non_parallel)

    def test_no_labels(self, par2: StateSpace) -> None:
        dot = factored_dot_source(par2, labels=False)
        assert par2.product_factors is not None
        for i, f in enumerate(par2.product_factors):
            for sid in f.states:
                assert f'f{i}s{sid} [label="{sid}"' in dot

    def test_no_edge_labels(self, par2: StateSpace) -> None:
        dot = factored_dot_source(par2, edge_labels=False)
        edge_lines = [l for l in dot.split('\n') if '->' in l]
        for line in edge_lines:
            assert 'label=' not in line

    def test_top_bottom_colors(self, par2: StateSpace) -> None:
        dot = factored_dot_source(par2)
        assert "#dbeafe" in dot
        assert "#dcfce7" in dot

    def test_deep_factors(self, par2_deep: StateSpace) -> None:
        dot = factored_dot_source(par2_deep)
        assert par2_deep.product_factors is not None
        # First factor should have 3 states (init, a, end)
        f0 = par2_deep.product_factors[0]
        assert len(f0.states) >= 2


class TestFactoredDashboard:

    @pytest.mark.skipif(
        subprocess.run(["which", "dot"], capture_output=True).returncode != 0,
        reason="graphviz dot not installed",
    )
    def test_dashboard_html(self, par2: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "factored")
            out = factored_dashboard(par2, path)
            assert out.endswith(".html")
            with open(out) as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "Factor 1" in html
            assert "Factor 2" in html

    @pytest.mark.skipif(
        subprocess.run(["which", "dot"], capture_output=True).returncode != 0,
        reason="graphviz dot not installed",
    )
    def test_dashboard_title(self, par2: StateSpace) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "factored")
            factored_dashboard(par2, path, title="Test Title")
            with open(path + ".html") as f:
                html = f.read()
            assert "Test Title" in html

    def test_error_on_non_parallel(self, non_parallel: StateSpace) -> None:
        with pytest.raises(ValueError, match="not a parallel type"):
            factored_dashboard(non_parallel, "/tmp/nope")


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_grid_non_parallel_error(self) -> None:
        from reticulate.cli import main
        with pytest.raises(SystemExit):
            main(["--grid", "&{a: end, b: end}"])

    def test_3d_non_parallel_error(self) -> None:
        from reticulate.cli import main
        with pytest.raises(SystemExit):
            main(["--3d", "&{a: end}"])

    def test_factored_non_parallel_error(self) -> None:
        from reticulate.cli import main
        with pytest.raises(SystemExit):
            main(["--factored", "&{a: end}"])

    def test_3d_on_2d_error(self) -> None:
        from reticulate.cli import main
        with pytest.raises(SystemExit):
            main(["--3d", "(&{a: end} || &{b: end})"])

    def test_3d_on_4d(self) -> None:
        from reticulate.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test4d")
            main(["--3d", path, "(&{a: end} || &{b: end} || &{c: end} || &{d: end})"])
            assert os.path.exists(path + ".html")

    def test_grid_on_3d_error(self) -> None:
        from reticulate.cli import main
        with pytest.raises(SystemExit):
            main(["--grid", "(&{a: end} || &{b: end} || &{c: end})"])

    def test_3d_on_parallel(self) -> None:
        from reticulate.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test3d")
            main(["--3d", path, "(&{a: end} || &{b: end} || &{c: end})"])
            assert os.path.exists(path + ".html")

    @pytest.mark.skipif(
        subprocess.run(["which", "dot"], capture_output=True).returncode != 0,
        reason="graphviz dot not installed",
    )
    def test_factored_on_parallel(self) -> None:
        from reticulate.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "factored")
            main(["--factored", path, "(&{a: end} || &{b: end})"])
            assert os.path.exists(path + ".html")

    def test_factored_dot_mode(self, capsys) -> None:
        from reticulate.cli import main
        main(["--factored", "--dot", "(&{a: end} || &{b: end})"])
        captured = capsys.readouterr()
        assert "digraph factored" in captured.out


# ---------------------------------------------------------------------------
# Topo order helper test
# ---------------------------------------------------------------------------

class TestTopoOrder:

    def test_simple(self) -> None:
        ss = build_statespace(parse("&{a: &{b: end}}"))
        order = _topo_order(ss)
        assert order[0] == ss.top
        assert order[-1] == ss.bottom
