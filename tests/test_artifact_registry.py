"""Tests for artifact_registry module."""

import time
import pytest
from reticulate.artifact_registry import (
    ArtifactKind,
    Artifact,
    ArtifactRegistry,
)


# ---------------------------------------------------------------------------
# ArtifactKind
# ---------------------------------------------------------------------------

class TestArtifactKind:
    def test_all_kinds_defined(self):
        assert len(ArtifactKind) == 7

    def test_kind_values(self):
        assert ArtifactKind.MODULE.value == "module"
        assert ArtifactKind.TEST.value == "test"
        assert ArtifactKind.PAPER.value == "paper"
        assert ArtifactKind.PROOFS.value == "proofs"
        assert ArtifactKind.REPORT.value == "report"
        assert ArtifactKind.CONFIG.value == "config"
        assert ArtifactKind.DIAGRAM.value == "diagram"


# ---------------------------------------------------------------------------
# Artifact dataclass
# ---------------------------------------------------------------------------

class TestArtifact:
    def test_create_artifact(self):
        a = Artifact(
            kind=ArtifactKind.MODULE,
            path="reticulate/reticulate/foo.py",
            agent_id="impl-23-1",
            agent_type="Implementer",
            step_number="23",
        )
        assert a.kind == ArtifactKind.MODULE
        assert a.agent_type == "Implementer"
        assert a.step_number == "23"

    def test_artifact_is_frozen(self):
        a = Artifact(
            kind=ArtifactKind.MODULE,
            path="foo.py",
            agent_id="x",
            agent_type="Implementer",
            step_number="1",
        )
        with pytest.raises(AttributeError):
            a.kind = ArtifactKind.TEST  # type: ignore

    def test_exists_false_for_nonexistent(self):
        a = Artifact(
            kind=ArtifactKind.MODULE,
            path="/nonexistent/file.py",
            agent_id="x",
            agent_type="Implementer",
            step_number="1",
        )
        assert a.exists is False

    def test_exists_true_for_this_file(self):
        a = Artifact(
            kind=ArtifactKind.TEST,
            path=__file__,
            agent_id="x",
            agent_type="Tester",
            step_number="1",
        )
        assert a.exists is True

    def test_metadata_default_empty(self):
        a = Artifact(
            kind=ArtifactKind.REPORT,
            path="x",
            agent_id="x",
            agent_type="Evaluator",
            step_number="1",
        )
        assert a.metadata == {}

    def test_metadata_custom(self):
        a = Artifact(
            kind=ArtifactKind.PAPER,
            path="x",
            agent_id="x",
            agent_type="Writer",
            step_number="1",
            metadata={"word_count": 5000},
        )
        assert a.metadata["word_count"] == 5000

    def test_created_at_default(self):
        before = time.time()
        a = Artifact(
            kind=ArtifactKind.MODULE,
            path="x",
            agent_id="x",
            agent_type="Implementer",
            step_number="1",
        )
        after = time.time()
        assert before <= a.created_at <= after


# ---------------------------------------------------------------------------
# ArtifactRegistry
# ---------------------------------------------------------------------------

class TestArtifactRegistry:
    def test_empty_registry(self):
        reg = ArtifactRegistry()
        assert reg.count == 0
        assert reg.all_artifacts == []

    def test_register_artifact(self):
        reg = ArtifactRegistry()
        a = reg.register(
            kind=ArtifactKind.MODULE,
            path="reticulate/reticulate/foo.py",
            agent_id="impl-23-1",
            agent_type="Implementer",
            step_number="23",
        )
        assert reg.count == 1
        assert a.kind == ArtifactKind.MODULE
        assert a.step_number == "23"

    def test_register_multiple(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, "a.py", "a1", "Implementer", "23")
        reg.register(ArtifactKind.TEST, "b.py", "a2", "Tester", "23")
        reg.register(ArtifactKind.PAPER, "c.tex", "a3", "Writer", "24")
        assert reg.count == 3

    def test_get_for_step(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, "a.py", "a1", "Implementer", "23")
        reg.register(ArtifactKind.TEST, "b.py", "a2", "Tester", "23")
        reg.register(ArtifactKind.PAPER, "c.tex", "a3", "Writer", "24")

        step23 = reg.get_for_step("23")
        assert len(step23) == 2
        step24 = reg.get_for_step("24")
        assert len(step24) == 1

    def test_get_for_step_with_kind_filter(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, "a.py", "a1", "Implementer", "23")
        reg.register(ArtifactKind.TEST, "b.py", "a2", "Tester", "23")

        modules = reg.get_for_step("23", ArtifactKind.MODULE)
        assert len(modules) == 1
        assert modules[0].kind == ArtifactKind.MODULE

    def test_get_by_agent(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, "a.py", "impl-23", "Implementer", "23")
        reg.register(ArtifactKind.MODULE, "b.py", "impl-23", "Implementer", "24")
        reg.register(ArtifactKind.PAPER, "c.tex", "writer-23", "Writer", "23")

        impl_artifacts = reg.get_by_agent("impl-23")
        assert len(impl_artifacts) == 2

    def test_get_by_kind(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, "a.py", "a1", "Implementer", "23")
        reg.register(ArtifactKind.MODULE, "b.py", "a2", "Implementer", "24")
        reg.register(ArtifactKind.PAPER, "c.tex", "a3", "Writer", "23")

        modules = reg.get_by_kind(ArtifactKind.MODULE)
        assert len(modules) == 2

    def test_get_for_nonexistent_step(self):
        reg = ArtifactRegistry()
        assert reg.get_for_step("999") == []

    def test_summary(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, __file__, "a1", "Implementer", "23")
        reg.register(ArtifactKind.TEST, "/nonexistent", "a2", "Tester", "23")
        reg.register(ArtifactKind.PAPER, "/nonexistent2", "a3", "Writer", "24")

        s = reg.summary()
        assert s["total"] == 3
        assert s["by_kind"]["module"] == 1
        assert s["by_kind"]["test"] == 1
        assert s["by_step"]["23"] == 2
        assert s["by_step"]["24"] == 1
        assert s["existing"] == 1  # only __file__ exists

    def test_clear(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, "a.py", "a1", "Implementer", "23")
        assert reg.count == 1
        reg.clear()
        assert reg.count == 0

    def test_register_with_metadata(self):
        reg = ArtifactRegistry()
        a = reg.register(
            kind=ArtifactKind.PAPER,
            path="main.tex",
            agent_id="w1",
            agent_type="Writer",
            step_number="23",
            metadata={"word_count": 6000, "sections": 8},
        )
        assert a.metadata["word_count"] == 6000
        assert a.metadata["sections"] == 8

    def test_all_artifacts_returns_copy(self):
        reg = ArtifactRegistry()
        reg.register(ArtifactKind.MODULE, "a.py", "a1", "Implementer", "23")
        arts = reg.all_artifacts
        arts.clear()
        assert reg.count == 1  # original not affected
