"""Self-referencing benchmarks for P104 Modularity Analysis.

Each P104 component's protocol is specified as a session type, analyzed by
the tool itself, and used as a real-world benchmark.  This makes P104 its
own test subject -- the modularity analysis tool analyzes its own protocols.

Session type legend:
  CLI workflow:  parse -> analyze -> report (branching on modular/non-modular/error)
  REST API:      submit -> poll loop -> fetch/fail (with retry on rejection)
  Importer:      three import formats, each branching OK/ERROR
  Report Gen:    generate -> select output format
  MCP Server:    recursive analysis/modularity/quit loop

Verified 2026-03-31:
  - All 5 form bounded lattices
  - 4/5 are distributive; Importer is non-distributive (shared OK/ERROR outcomes
    across three branches create an M3 forbidden sublattice — diamond)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class P104Benchmark:
    """A P104 self-referencing benchmark protocol."""

    name: str
    component: str
    type_string: str
    description: str
    session_type_for: str  # which P104 component this specifies
    expected_states: int
    expected_transitions: int
    expected_sccs: int
    expected_lattice: bool
    expected_distributive: bool
    uses_parallel: bool


P104_BENCHMARKS: list[P104Benchmark] = [
    # --- 1. CLI workflow ---
    P104Benchmark(
        name="P104-CLI",
        component="CLI",
        type_string=(
            "&{parse: +{OK: &{analyze: +{modular: &{report: end}, "
            "non_modular: &{report: end}}}, ERROR: end}}"
        ),
        description=(
            "The reticulate modular CLI workflow: parse the input session type, "
            "then either analyze (branching on modular/non-modular result and "
            "generating a report) or report a parse error."
        ),
        session_type_for="reticulate modular CLI subcommand (M1.1)",
        expected_states=7,
        expected_transitions=8,
        expected_sccs=7,
        expected_lattice=True,
        expected_distributive=True,
        uses_parallel=False,
    ),

    # --- 2. REST API ---
    P104Benchmark(
        name="P104-REST-API",
        component="REST API",
        type_string=(
            "rec X . &{submit: +{accepted: &{poll: rec Y . +{pending: Y, "
            "complete: &{fetch: end}, failed: end}}, rejected: X}}"
        ),
        description=(
            "The REST API protocol: submit an analysis request, then either "
            "get accepted (enter poll loop until complete or failed) or get "
            "rejected (retry submission).  Models async job-based API pattern."
        ),
        session_type_for="FastAPI REST endpoint (M1.2)",
        expected_states=6,
        expected_transitions=8,
        expected_sccs=5,
        expected_lattice=True,
        expected_distributive=True,
        uses_parallel=False,
    ),

    # --- 3. Protocol Importer ---
    P104Benchmark(
        name="P104-Importer",
        component="Importer",
        type_string=(
            "&{import_openapi: +{OK: end, ERROR: end}, "
            "import_grpc: +{OK: end, ERROR: end}, "
            "import_asyncapi: +{OK: end, ERROR: end}}"
        ),
        description=(
            "The protocol importer: choose one of three formats (OpenAPI, gRPC, "
            "AsyncAPI) and get OK or ERROR.  Non-distributive because the three "
            "branches share outcome states, creating an M3 forbidden sublattice "
            "(diamond: three independent paths reconverge at shared outcome). "
            "This is a genuine modularity finding: the importer's protocol has "
            "non-trivial coupling between import paths."
        ),
        session_type_for="Protocol importers (M1.3)",
        expected_states=5,
        expected_transitions=9,
        expected_sccs=5,
        expected_lattice=True,
        expected_distributive=False,  # M3 (diamond) present!
        uses_parallel=False,
    ),

    # --- 4. Report Generator ---
    P104Benchmark(
        name="P104-Report-Gen",
        component="Report Generator",
        type_string="&{generate: +{json: end, pdf: end, text: end, dot: end}}",
        description=(
            "The report generator: generate a modularity certificate and select "
            "the output format (JSON, PDF, text, or DOT).  Simple two-stage "
            "protocol with a 4-way selection."
        ),
        session_type_for="Report generation (M1.4)",
        expected_states=3,
        expected_transitions=5,
        expected_sccs=3,
        expected_lattice=True,
        expected_distributive=True,
        uses_parallel=False,
    ),

    # --- 5. MCP Server ---
    P104Benchmark(
        name="P104-MCP-Server",
        component="MCP Server",
        type_string=(
            "rec X . &{analyze: +{result: X}, "
            "check_modularity: +{result: X}, quit: end}"
        ),
        description=(
            "The MCP server protocol: repeatedly accept analysis or modularity "
            "requests (returning results and looping) or quit.  Models the AI "
            "agent tool-calling loop."
        ),
        session_type_for="MCP server modularity tools (M1.5)",
        expected_states=4,
        expected_transitions=5,
        expected_sccs=2,
        expected_lattice=True,
        expected_distributive=True,
        uses_parallel=False,
    ),
]


# Convenience: name -> benchmark
P104_BY_NAME: dict[str, P104Benchmark] = {b.name: b for b in P104_BENCHMARKS}
P104_BY_COMPONENT: dict[str, P104Benchmark] = {b.component: b for b in P104_BENCHMARKS}
