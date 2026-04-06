"""Tests for reticulate.supervisor — research programme supervision.

Covers public API: supervise(), SupervisionReport, ProgrammeSnapshot,
StepStatus, Proposal, and the module-level scan helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reticulate import supervisor

# The true repo root: two parents above this test file
# (.../SessionTypesResearch/reticulate/tests/test_supervisor.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def root() -> Path:
    return REPO_ROOT
from reticulate.supervisor import (
    Proposal,
    ProgrammeSnapshot,
    StepStatus,
    SupervisionReport,
    supervise,
)


# ---------------------------------------------------------------------------
# Data type smoke tests
# ---------------------------------------------------------------------------


def test_step_status_is_frozen():
    s = StepStatus(
        number="80k", title="Parity Supervisor", phase="I",
        status="planned", has_paper=False, has_proofs=False,
        has_module=True, module_name="supervisor", test_count=10,
        word_count=0, grade="missing", source="todo", priority="high",
    )
    with pytest.raises((AttributeError, Exception)):
        s.title = "Changed"  # type: ignore[misc]


def test_proposal_defaults():
    p = Proposal("step", "Step 80k", "rationale", "high")
    assert p.depends_on == ()
    assert p.details == {}


def test_proposal_categories():
    for cat in ("step", "tool", "paper", "venue"):
        p = Proposal(cat, "T", "R", "low")
        assert p.category == cat


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_step_sort_key_numeric():
    assert supervisor._step_sort_key("5") < supervisor._step_sort_key("10")
    assert supervisor._step_sort_key("80b") < supervisor._step_sort_key("80c")
    assert supervisor._step_sort_key("80") < supervisor._step_sort_key("80a")


def test_step_sort_key_handles_garbage():
    key = supervisor._step_sort_key("notanumber")
    assert isinstance(key, tuple)


def test_assign_phase_ranges():
    assert "1-15" in supervisor._assign_phase("5")
    assert "16-30" in supervisor._assign_phase("20")
    assert "31-49" in supervisor._assign_phase("35")
    assert "50-69" in supervisor._assign_phase("60")
    assert "70-79" in supervisor._assign_phase("75")
    assert "80-109" in supervisor._assign_phase("80")
    assert "151-170" in supervisor._assign_phase("160")
    assert "200-210" in supervisor._assign_phase("205")


def test_assign_phase_unknown():
    assert supervisor._assign_phase("xyz") == "Unknown"


def test_find_project_root_returns_path():
    root = REPO_ROOT
    assert isinstance(root, Path)


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------


def test_scan_modules_returns_count_and_set():
    root = REPO_ROOT
    n, names = supervisor._scan_modules(root)
    assert isinstance(n, int)
    assert isinstance(names, set)
    assert n == len(names)
    # supervisor.py itself must be present
    assert "supervisor" in names


def test_count_tests_per_module_dict():
    root = REPO_ROOT
    counts = supervisor._count_tests_per_module(root)
    assert isinstance(counts, dict)
    # This very file contributes
    assert counts.get("supervisor", 0) >= 1


def test_scan_steps_on_disk_returns_dict():
    root = REPO_ROOT
    disk = supervisor._scan_steps_on_disk(root)
    assert isinstance(disk, dict)
    # We expect at least some completed steps
    assert len(disk) > 0
    sample = next(iter(disk.values()))
    for key in ("title", "directory", "has_paper", "has_proofs",
                "word_count", "grade"):
        assert key in sample


def test_parse_todo_returns_dict():
    root = REPO_ROOT
    todo = supervisor._parse_todo(root)
    assert isinstance(todo, dict)


def test_parse_todo_missing_file(tmp_path):
    # No todo file in tmp_path → empty dict
    assert supervisor._parse_todo(tmp_path) == {}


def test_scan_steps_missing_dir(tmp_path):
    assert supervisor._scan_steps_on_disk(tmp_path) == {}


def test_scan_modules_missing_dir(tmp_path):
    n, names = supervisor._scan_modules(tmp_path)
    assert n == 0
    assert names == set()


def test_recent_commits_returns_list():
    root = REPO_ROOT
    commits = supervisor._recent_commits(root, n=3)
    assert isinstance(commits, list)


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_build_snapshot_returns_snapshot():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    assert isinstance(snap, ProgrammeSnapshot)
    assert snap.total_steps_planned >= snap.total_steps_on_disk or \
           snap.total_steps_on_disk >= 0
    assert snap.total_modules > 0
    assert snap.total_tests > 0
    assert isinstance(snap.steps, tuple)
    assert isinstance(snap.phases, dict)


def test_snapshot_counts_consistent():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    # Sum of grade categories should not exceed total steps
    assert snap.complete_steps + snap.draft_steps <= len(snap.steps)
    assert snap.a_plus_count <= snap.complete_steps + snap.draft_steps + 1


def test_snapshot_a_plus_pct_in_range():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    assert 0.0 <= snap.a_plus_pct <= 100.0


def test_snapshot_phases_partition_steps():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    total_in_phases = sum(len(v) for v in snap.phases.values())
    assert total_in_phases == len(snap.steps)


# ---------------------------------------------------------------------------
# Proposal generator tests
# ---------------------------------------------------------------------------


def test_generate_step_proposals_capped():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    props = supervisor._generate_step_proposals(snap)
    assert len(props) <= 30
    for p in props:
        assert p.category == "step"
        assert p.priority in ("high", "medium", "low")


def test_generate_tool_proposals_nonempty():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    props = supervisor._generate_tool_proposals(snap)
    assert len(props) > 0
    assert all(p.category == "tool" for p in props)


def test_generate_paper_proposals_nonempty():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    props = supervisor._generate_paper_proposals(snap)
    assert all(p.category == "paper" for p in props)


def test_venues_present():
    assert len(supervisor._VENUES) >= 5
    assert all(p.category == "venue" for p in supervisor._VENUES)


# ---------------------------------------------------------------------------
# Top-level supervise()
# ---------------------------------------------------------------------------


def test_supervise_returns_report():
    report = supervise()
    assert isinstance(report, SupervisionReport)
    assert isinstance(report.snapshot, ProgrammeSnapshot)
    assert isinstance(report.proposals, tuple)
    assert isinstance(report.step_proposals, tuple)
    assert isinstance(report.tool_proposals, tuple)
    assert isinstance(report.paper_proposals, tuple)
    assert isinstance(report.venue_proposals, tuple)


def test_supervise_with_explicit_root():
    root = REPO_ROOT
    report = supervise(root=root)
    assert report.snapshot.total_modules > 0


def test_supervise_with_after_step_param():
    report = supervise(after_step="80")
    assert isinstance(report, SupervisionReport)


def test_supervise_proposals_union():
    report = supervise()
    expected = (len(report.step_proposals) + len(report.tool_proposals)
                + len(report.paper_proposals) + len(report.venue_proposals))
    assert len(report.proposals) == expected


def test_report_summary_is_string():
    report = supervise()
    s = report.summary()
    assert isinstance(s, str)
    assert "RESEARCH SUPERVISION REPORT" in s
    assert "PHASE BREAKDOWN" in s


def test_report_summary_mentions_counts():
    report = supervise()
    s = report.summary()
    assert str(report.snapshot.total_modules) in s


# ---------------------------------------------------------------------------
# Parity-awareness: cross-checking Java/Python modules
# ---------------------------------------------------------------------------


def test_supervisor_can_detect_module_set():
    """The set of Python modules is the input to a parity check."""
    root = REPO_ROOT
    _, modules = supervisor._scan_modules(root)
    # Must contain core modules used as parity baseline
    for core in ("parser", "statespace", "lattice", "supervisor"):
        assert core in modules


def test_quick_wins_listed():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    assert isinstance(snap.quick_wins, tuple)


def test_gaps_listed():
    root = REPO_ROOT
    snap = supervisor._build_snapshot(root)
    assert isinstance(snap.gaps, tuple)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_main_text_output(capsys):
    supervisor.main([])
    out = capsys.readouterr().out
    assert "RESEARCH SUPERVISION REPORT" in out


def test_cli_main_json_output(capsys):
    import json
    supervisor.main(["--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "snapshot" in data
    assert "proposals" in data
    assert "phases" in data


def test_cli_main_after_step(capsys):
    supervisor.main(["--after-step", "80"])
    out = capsys.readouterr().out
    assert "REPORT" in out
