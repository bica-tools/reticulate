"""Artifact registry: tracks outputs between agent phases.

When agents produce files (modules, papers, tests, proofs), the registry
records what was created, by which agent, for which step. Downstream agents
query the registry to find their inputs.

The git repo is the source of truth; the registry is an in-memory index
that maps agent work to filesystem locations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactKind(Enum):
    """Categories of agent-produced artifacts."""
    MODULE = "module"
    TEST = "test"
    PAPER = "paper"
    PROOFS = "proofs"
    REPORT = "report"
    CONFIG = "config"
    DIAGRAM = "diagram"


@dataclass(frozen=True)
class Artifact:
    """A single artifact produced by an agent."""
    kind: ArtifactKind
    path: str                    # relative to project root
    agent_id: str                # who created it
    agent_type: str
    step_number: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()


class ArtifactRegistry:
    """In-memory registry of artifacts produced during a sprint.

    Usage:
        registry = ArtifactRegistry(root="/path/to/project")
        registry.register(ArtifactKind.MODULE, "reticulate/reticulate/foo.py",
                         agent_id="impl-23-1", agent_type="Implementer", step="23")
        # Downstream agent queries:
        modules = registry.get_for_step("23", ArtifactKind.MODULE)
    """

    def __init__(self, root: str | Path = "") -> None:
        self.root = Path(root) if root else Path.cwd()
        self._artifacts: list[Artifact] = []

    def register(
        self,
        kind: ArtifactKind,
        path: str,
        agent_id: str,
        agent_type: str,
        step_number: str,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Register a new artifact."""
        artifact = Artifact(
            kind=kind,
            path=path,
            agent_id=agent_id,
            agent_type=agent_type,
            step_number=step_number,
            metadata=metadata or {},
        )
        self._artifacts.append(artifact)
        return artifact

    def get_for_step(
        self, step_number: str, kind: ArtifactKind | None = None
    ) -> list[Artifact]:
        """Get all artifacts for a step, optionally filtered by kind."""
        results = [a for a in self._artifacts if a.step_number == step_number]
        if kind is not None:
            results = [a for a in results if a.kind == kind]
        return results

    def get_by_agent(self, agent_id: str) -> list[Artifact]:
        """Get all artifacts produced by a specific agent."""
        return [a for a in self._artifacts if a.agent_id == agent_id]

    def get_by_kind(self, kind: ArtifactKind) -> list[Artifact]:
        """Get all artifacts of a specific kind."""
        return [a for a in self._artifacts if a.kind == kind]

    @property
    def all_artifacts(self) -> list[Artifact]:
        return list(self._artifacts)

    @property
    def count(self) -> int:
        return len(self._artifacts)

    def summary(self) -> dict[str, Any]:
        """Summary statistics of registered artifacts."""
        by_kind: dict[str, int] = {}
        by_step: dict[str, int] = {}
        for a in self._artifacts:
            by_kind[a.kind.value] = by_kind.get(a.kind.value, 0) + 1
            by_step[a.step_number] = by_step.get(a.step_number, 0) + 1
        return {
            "total": self.count,
            "by_kind": by_kind,
            "by_step": by_step,
            "existing": sum(1 for a in self._artifacts if a.exists),
        }

    def clear(self) -> None:
        """Clear all artifacts (between sprints)."""
        self._artifacts.clear()
