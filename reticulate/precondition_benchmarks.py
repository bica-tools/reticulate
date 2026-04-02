"""Benchmark session types extracted from real Java preconditions (Step 97e).

174 classes from 7 real Java projects, automatically extracted by
scanning for checkState/checkArgument patterns.

Each entry: (project, class_name, field, phases, session_type)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from reticulate.precondition_extractor import analyze_java_file, SelectionProposal
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive


@dataclass(frozen=True)
class PreconditionBenchmark:
    """A session type benchmark extracted from a real Java precondition."""
    project: str
    class_name: str
    field_name: str
    num_phases: int
    phase_names: list[str]
    guarded_methods: int
    always_available: int
    session_type: str


def extract_all_benchmarks(project_dirs: dict[str, str]) -> list[PreconditionBenchmark]:
    """Extract benchmarks from all Java projects."""
    benchmarks: list[PreconditionBenchmark] = []

    for proj_name, proj_dir in project_dirs.items():
        if not os.path.isdir(proj_dir):
            continue
        for root, dirs, files in os.walk(proj_dir):
            dirs[:] = [d for d in dirs if d not in
                       ('target', 'build', '.git', 'test', 'tests', 'testing', 'testlib')]
            for fname in files:
                if not fname.endswith('.java'):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    result = analyze_java_file(fpath)
                    if result and len(result.phases) >= 1:
                        benchmarks.append(PreconditionBenchmark(
                            project=proj_name,
                            class_name=result.class_name,
                            field_name=result.field_name,
                            num_phases=len(result.phases) + 1,  # +1 for INITIAL
                            phase_names=result.enum_values,
                            guarded_methods=sum(len(p.methods) for p in result.phases),
                            always_available=len(result.always_available),
                            session_type=result.session_type,
                        ))
                except Exception:
                    continue

    return benchmarks


def analyze_benchmarks(benchmarks: list[PreconditionBenchmark]) -> str:
    """Analyze all benchmarks and produce a report."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  PRECONDITION BENCHMARK ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"  Total benchmarks: {len(benchmarks)}")

    # Per project
    from collections import Counter
    by_project = Counter(b.project for b in benchmarks)
    lines.append("")
    lines.append("  By project:")
    for proj, count in by_project.most_common():
        lines.append(f"    {proj}: {count}")

    # Analyze session types
    lattice_count = 0
    dist_count = 0
    errors = 0
    classifications = Counter()

    for b in benchmarks:
        try:
            ast = parse(b.session_type)
            ss = build_statespace(ast)
            lr = check_lattice(ss)
            if lr.is_lattice:
                lattice_count += 1
                dr = check_distributive(ss)
                if dr.is_distributive:
                    dist_count += 1
                classifications[dr.classification] += 1
            else:
                classifications['non-lattice'] += 1
        except Exception:
            errors += 1

    analyzed = lattice_count + errors
    lines.append("")
    lines.append(f"  Analyzed: {analyzed - errors}/{len(benchmarks)} (errors: {errors})")
    lines.append(f"  Lattice: {lattice_count}/{analyzed - errors} ({lattice_count / (analyzed - errors):.0%})" if analyzed > errors else "")
    lines.append(f"  Distributive: {dist_count}/{analyzed - errors} ({dist_count / (analyzed - errors):.0%})" if analyzed > errors else "")
    lines.append("")
    lines.append("  Classification:")
    for cls, count in classifications.most_common():
        lines.append(f"    {cls}: {count}")

    # Phase statistics
    phase_counts = Counter(b.num_phases for b in benchmarks)
    lines.append("")
    lines.append("  Phase distribution:")
    for np, count in sorted(phase_counts.items()):
        lines.append(f"    {np} phases: {count} classes")

    lines.append("=" * 80)
    return "\n".join(lines)


# Default project directories
DEFAULT_PROJECTS = {
    'Guava': '/tmp/guava/guava/src/com/google/common',
    'Commons IO': '/tmp/commons-io/src/main/java',
    'Commons Lang': '/tmp/commons-lang/src/main/java',
    'Commons Collections': '/tmp/commons-collections/src/main/java',
    'Commons DBCP': '/tmp/commons-dbcp/src/main/java',
    'Commons Pool': '/tmp/commons-pool/src/main/java',
    'BICA Reborn': os.path.join(os.path.dirname(__file__), '..', '..', 'bica', 'src', 'main', 'java'),
}
