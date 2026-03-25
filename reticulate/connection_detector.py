"""Connection detector: finds publication opportunities from step intersections.

Scans all research steps, builds keyword and domain matrices, detects
connections between steps, and proposes new papers from high-strength
intersections. Runs after every sprint to surface new opportunities.

Usage:
    from reticulate.connection_detector import detect_connections, find_new_connections
    report = detect_connections(steps)
    new = find_new_connections(new_step, existing_steps)

    python -m reticulate.connection_detector
    python -m reticulate.connection_detector --top 20
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepInfo:
    """Parsed information about a research step."""
    number: str              # e.g. "70b"
    title: str               # human-readable title
    keywords: tuple[str, ...]
    modules: tuple[str, ...]  # reticulate modules used
    domain: str              # "theory" | "application" | "tool" | "intersection"
    depends_on: tuple[str, ...]  # steps this one cites/depends on
    dir_name: str            # e.g. "step70b-mcp-server-tool"


@dataclass(frozen=True)
class Connection:
    """A detected connection between two steps."""
    step_a: str
    step_b: str
    connection_type: str     # "keyword" | "module" | "domain" | "complement" | "dependency"
    strength: float          # 0.0 to 1.0
    paper_title: str         # auto-generated from the connection
    venues: tuple[str, ...]  # suggested venues
    shared_keywords: tuple[str, ...] = ()
    shared_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectionReport:
    """Summary report of all detected connections."""
    new_connections: tuple[Connection, ...]
    total_connections: int
    paper_opportunities: tuple[Connection, ...]  # top 10% by strength
    by_domain: dict[str, list[Connection]]       # grouped by domain pair


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

# Canonical keywords grouped by domain
_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "lattice": ("lattice", "meet", "join", "bounded", "distributiv", "modular", "reticulate"),
    "session-types": ("session type", "protocol", "branch", "select", "parallel", "end"),
    "process-algebra": ("ccs", "csp", "bisimulation", "failure", "trace", "process"),
    "petri-nets": ("petri", "marking", "place", "invariant", "coverab", "net"),
    "event-structures": ("event struct", "configuration", "domain", "causality"),
    "category-theory": ("category", "functor", "morphism", "product", "coproduct",
                         "equalizer", "coequalizer", "monoidal", "tensor", "galois"),
    "game-theory": ("game", "strategy", "nash", "equilibri", "parity", "supermodular",
                    "power", "rational", "pendular"),
    "lambda-calculus": ("lambda", "calculus", "preservation", "progress", "typing"),
    "monitoring": ("monitor", "runtime", "enforce", "middleware", "conformance"),
    "ai-agents": ("mcp", "a2a", "agent", "llm", "ai"),
    "api-verification": ("openapi", "rest", "api", "contract", "swagger"),
    "healthcare": ("fhir", "clinical", "hl7", "workflow", "triage"),
    "recursion": ("recursion", "recursive", "unfold", "guarded", "contractive", "mu"),
    "subtyping": ("subtyp", "refine", "embed", "gay-hole", "width"),
    "duality": ("dual", "involution", "complement", "opposite"),
    "algebra": ("algebra", "spectral", "mobius", "polynomial", "entropy", "compress"),
    "multiparty": ("multiparty", "global", "projection", "role", "mpst"),
    "termination": ("terminat", "fair", "liveness", "progress"),
    "software-engineering": ("uml", "design pattern", "gof", "state diagram"),
    "interdisciplinary": ("narrative", "biology", "law", "music", "cooking", "big brother"),
    "polarity": ("polarity", "fca", "formal concept"),
    "channels": ("channel", "buffer", "async", "synchron"),
    "context-free": ("chomsky", "regular", "context-free", "grammar", "pushdown"),
}

# Module-to-domain mapping
_MODULE_DOMAINS: dict[str, str] = {
    "parser.py": "tool",
    "statespace.py": "tool",
    "product.py": "tool",
    "lattice.py": "theory",
    "morphism.py": "theory",
    "subtyping.py": "theory",
    "duality.py": "theory",
    "termination.py": "theory",
    "reticular.py": "theory",
    "endomorphism.py": "theory",
    "recursion.py": "theory",
    "enumerate_types.py": "theory",
    "context_free.py": "intersection",
    "global_types.py": "tool",
    "projection.py": "theory",
    "petri.py": "intersection",
    "marking_lattice.py": "intersection",
    "place_invariants.py": "intersection",
    "coverability.py": "intersection",
    "ccs.py": "intersection",
    "csp.py": "intersection",
    "failures.py": "intersection",
    "category.py": "intersection",
    "coproduct.py": "intersection",
    "equalizer.py": "intersection",
    "coequalizer.py": "intersection",
    "monoidal.py": "intersection",
    "tensor.py": "intersection",
    "polarity.py": "intersection",
    "channel.py": "intersection",
    "async_channel.py": "intersection",
    "buffered_channel.py": "intersection",
    "spectral.py": "intersection",
    "compress.py": "intersection",
    "monitor.py": "application",
    "openapi.py": "application",
    "fhir.py": "application",
    "mcp_server.py": "application",
    "mcp_conformance.py": "application",
    "mcp_runtime.py": "application",
    "testgen.py": "tool",
    "coverage.py": "tool",
    "visualize.py": "tool",
    "cli.py": "tool",
    "game": "intersection",
    "evaluator.py": "intersection",
}

# Step number to module mapping (known associations)
_STEP_MODULES: dict[str, tuple[str, ...]] = {
    "1": ("parser.py", "statespace.py"),
    "2": ("statespace.py",),
    "3": ("lattice.py",),
    "4": ("lattice.py",),
    "5": ("lattice.py",),
    "5b": ("product.py",),
    "5c": ("nary_parallel.py",),
    "6": ("enumerate_types.py",),
    "6b": ("termination.py",),
    "7": ("subtyping.py",),
    "8": ("duality.py",),
    "9": ("reticular.py",),
    "10": ("endomorphism.py",),
    "11": ("global_types.py",),
    "12": ("projection.py",),
    "13a": ("recursion.py",),
    "14": ("context_free.py",),
    "155b": ("polarity.py",),
    "156": ("realizability.py",),
    "157a": ("channel.py",),
    "157b": ("async_channel.py",),
    "158": ("buffered_channel.py",),
    "163": ("category.py",),
    "164": ("coproduct.py",),
    "165": ("equalizer.py",),
    "166": ("coequalizer.py",),
    "167": ("monoidal.py",),
    "168": ("tensor.py",),
    "16": ("statespace.py",),
    "17": ("statespace.py",),
    "18": ("statespace.py",),
    "19": ("morphism.py",),
    "20": ("statespace.py",),
    "21": ("petri.py",),
    "22": ("marking_lattice.py",),
    "23": ("place_invariants.py",),
    "23b": ("st_invariants.py",),
    "24": ("coverability.py",),
    "25": ("petri.py",),
    "26": ("ccs.py",),
    "27": ("csp.py",),
    "28": ("failures.py",),
    "29": ("lattice.py",),
    "29a": ("lattice.py",),
    "29b": ("lattice.py",),
    "29c": ("process_equivalence.py",),
    "30": ("spectral.py",),
    "31a": ("spectral.py",),
    "50": ("statespace.py",),
    "51": ("statespace.py",),
    "52": ("statespace.py",),
    "53": ("statespace.py",),
    "70": ("mcp_conformance.py",),
    "70b": ("mcp_server.py", "mcp_runtime.py"),
    "70c": ("statespace.py",),
    "70d": ("compress.py",),
    "70e": ("compress.py",),
    "71": ("openapi.py",),
    "72": ("fhir.py",),
    "80": ("monitor.py",),
    "104": ("statespace.py",),
    "200": ("evaluator.py",),
    "200a": ("evaluator.py",),
    "200d": ("evaluator.py",),
    "200e": ("evaluator.py",),
    "200f": ("evaluator.py",),
    "200g": ("evaluator.py",),
    "900": ("game",),
    "900a": ("game",),
    "900b": ("game",),
    "900c": ("game",),
    "900d": ("game",),
    "900e": ("game",),
    "900f": ("game",),
    "900g": ("game",),
    "900h": ("game",),
    "900i": ("game",),
}

# Venue suggestions by domain pair
_VENUE_MAP: dict[tuple[str, str], tuple[str, ...]] = {
    ("theory", "application"): ("ECOOP", "ESOP", "OOPSLA", "ICSE-SEIP"),
    ("theory", "intersection"): ("CONCUR", "FoSSaCS", "LICS", "CALCO"),
    ("theory", "tool"): ("TACAS", "SCP", "CAV"),
    ("application", "intersection"): ("COORDINATION", "FORTE", "MIDDLEWARE"),
    ("application", "tool"): ("ISSTA", "ASE", "ICSE-Demo"),
    ("intersection", "tool"): ("TACAS", "ATVA", "SCP"),
    ("theory", "theory"): ("LICS", "CONCUR", "FoSSaCS", "ICALP"),
    ("application", "application"): ("ICSE-SEIP", "ESEC/FSE", "ASE"),
    ("intersection", "intersection"): ("CONCUR", "FORTE", "CALCO"),
    ("tool", "tool"): ("TACAS", "SCP", "ASE"),
}


def _extract_keywords(title: str, dir_name: str) -> tuple[str, ...]:
    """Extract keywords from a step title and directory name."""
    text = (title + " " + dir_name.replace("-", " ")).lower()
    keywords: list[str] = []
    for group_name, terms in _KEYWORD_GROUPS.items():
        for term in terms:
            if term in text:
                keywords.append(group_name)
                break
    # Always include base keywords
    keywords.extend(["session-types", "lattice"])
    return tuple(sorted(set(keywords)))


def _classify_domain(step_num: str, title: str, dir_name: str) -> str:
    """Classify a step as theory, application, tool, or intersection."""
    text = (title + " " + dir_name.replace("-", " ")).lower()

    theory_signals = {"theorem", "proof", "lattice condition", "universality",
                      "subtyping", "duality", "endomorphism", "reticular form",
                      "termination", "recursion", "chomsky", "absorption"}
    app_signals = {"monitor", "openapi", "fhir", "mcp", "a2a", "agent", "uml",
                   "gof", "narrative", "big brother", "interdisciplin", "benchmark",
                   "runtime", "conformance", "compress"}
    tool_signals = {"state space", "construction", "parser", "visualization",
                    "tool", "cli", "test gen", "coverage"}
    intersection_signals = {"petri", "ccs", "csp", "event struct", "category",
                           "coproduct", "equalizer", "tensor", "monoidal",
                           "polarity", "channel", "spectral", "algebra",
                           "grammar", "marking", "game", "lambda", "dialogue",
                           "distributiv", "galois", "gratzer", "failure"}

    scores = {
        "theory": sum(1 for w in theory_signals if w in text),
        "application": sum(1 for w in app_signals if w in text),
        "tool": sum(1 for w in tool_signals if w in text),
        "intersection": sum(1 for w in intersection_signals if w in text),
    }

    # Use module mapping as tiebreaker
    modules = _STEP_MODULES.get(step_num, ())
    for mod in modules:
        mod_domain = _MODULE_DOMAINS.get(mod, "")
        if mod_domain:
            scores[mod_domain] = scores.get(mod_domain, 0) + 0.5

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "theory"


def _extract_dependencies(step_dir: Path) -> tuple[str, ...]:
    """Extract step dependencies from LaTeX citations in a step directory."""
    deps: set[str] = set()
    main_tex = step_dir / "main.tex"
    if main_tex.exists():
        try:
            text = main_tex.read_text(errors="replace")
        except OSError:
            return ()
        # Look for references to other steps: "Step N", "step N"
        for m in re.finditer(r"[Ss]tep\s+(\d+\w*)", text):
            deps.add(m.group(1))
        # Look for \cite references to step papers
        for m in re.finditer(r"\\cite\{step(\d+\w*)", text):
            deps.add(m.group(1))
    return tuple(sorted(deps))


# ---------------------------------------------------------------------------
# Step scanning
# ---------------------------------------------------------------------------

def scan_steps(root: Path | str | None = None) -> list[StepInfo]:
    """Scan all step directories and extract StepInfo."""
    if root is None:
        for candidate in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
            if (candidate / "papers" / "steps").is_dir():
                root = candidate
                break
        else:
            root = Path.cwd()
    else:
        root = Path(root)

    steps_dir = root / "papers" / "steps"
    if not steps_dir.is_dir():
        return []

    results: list[StepInfo] = []
    for d in sorted(steps_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("step"):
            continue
        match = re.match(r"step(\d+\w*)-(.+)", d.name)
        if not match:
            continue
        num = match.group(1)
        raw_title = match.group(2).replace("-", " ").title()

        # Try to extract real title from LaTeX
        main = d / "main.tex"
        title = raw_title
        if main.exists():
            try:
                tex = main.read_text(errors="replace")
                tmatch = re.search(r"\\title\{[^}]*?([A-Z].*?)(?:\\\\|\})", tex)
                if tmatch:
                    title = tmatch.group(1).strip()[:80]
            except OSError:
                pass

        keywords = _extract_keywords(title, d.name)
        modules = _STEP_MODULES.get(num, ())
        domain = _classify_domain(num, title, d.name)
        deps = _extract_dependencies(d)

        results.append(StepInfo(
            number=num,
            title=title,
            keywords=keywords,
            modules=modules,
            domain=domain,
            depends_on=deps,
            dir_name=d.name,
        ))

    return results


def scan_steps_from_list(steps: Sequence[StepInfo]) -> list[StepInfo]:
    """Pass-through for programmatic use with pre-built StepInfo objects."""
    return list(steps)


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------

# Domain-specific title templates
_TITLE_TEMPLATES: dict[str, list[str]] = {
    "theory+application": [
        "{app_concept} via {theory_concept}: A Lattice-Theoretic Approach",
        "Applying {theory_concept} to {app_concept} Verification",
        "{theory_concept} for {app_concept}: From Theory to Practice",
    ],
    "theory+intersection": [
        "{concept_a} Meets {concept_b}: A Unified View Through Session Types",
        "Connecting {concept_a} and {concept_b} via Lattice Structure",
        "{concept_a} as {concept_b}: A Session Type Perspective",
    ],
    "application+application": [
        "Unified Protocol Verification for {concept_a} and {concept_b}",
        "Cross-Domain Protocol Analysis: {concept_a} and {concept_b}",
    ],
    "default": [
        "{concept_a} and {concept_b} in Session Type Lattices",
        "Bridging {concept_a} and {concept_b} via Reticulate Theory",
    ],
}

# Concept names for steps (short human-readable names)
_STEP_CONCEPTS: dict[str, str] = {
    "1": "State Space Construction",
    "3": "Branching Meet",
    "4": "Selection Join",
    "5": "Lattice Conditions",
    "5b": "Parallel Product",
    "6": "Universality",
    "7": "Subtyping",
    "8": "Duality",
    "9": "Reticular Form",
    "10": "Endomorphisms",
    "11": "Multiparty Types",
    "12": "Projection",
    "13a": "Recursion Analysis",
    "14": "Chomsky Classification",
    "155b": "Polarity",
    "156": "Realizability",
    "157a": "Channel Duality",
    "157b": "Async Channels",
    "158": "Buffered Channels",
    "163": "Category Theory",
    "164": "Coproducts",
    "165": "Equalizers",
    "166": "Coequalizers",
    "167": "Monoidal Structure",
    "168": "Tensor Products",
    "16": "Event Structures",
    "17": "Configuration Domains",
    "18": "Dialogue Types",
    "19": "ES Morphisms",
    "21": "Petri Nets",
    "22": "Marking Lattice",
    "23": "Place Invariants",
    "24": "Coverability",
    "26": "CCS",
    "27": "CSP",
    "28": "Failures",
    "29": "Distributivity",
    "30": "Algebraic Toolkit",
    "50": "UML State Diagrams",
    "51": "GoF Patterns",
    "52": "Narrative Types",
    "53": "Interdisciplinary Frontiers",
    "70": "Agent Conformance",
    "70b": "MCP Server",
    "70c": "NL-to-Session Types",
    "70d": "Engineering Distributivity",
    "70e": "Compression",
    "71": "OpenAPI Contracts",
    "72": "FHIR Workflows",
    "80": "Runtime Monitors",
    "104": "Big Brother",
    "200": "Lambda-S Calculus",
    "900": "Game Theory",
    "900a": "Game Semantics",
}


def auto_generate_paper_title(step_a: StepInfo, step_b: StepInfo) -> str:
    """Generate a paper title from combining two steps.

    Uses domain classification and step concepts to produce
    meaningful titles like:
        Step 80 (monitors) + Step 72 (FHIR)
        -> "Runtime Monitoring of FHIR Clinical Workflows via Session Types"
    """
    concept_a = _STEP_CONCEPTS.get(step_a.number, step_a.title)
    concept_b = _STEP_CONCEPTS.get(step_b.number, step_b.title)

    # Pick template based on domain pair
    pair_key = "+".join(sorted([step_a.domain, step_b.domain]))

    # For theory+application, assign roles
    if {step_a.domain, step_b.domain} == {"theory", "application"}:
        theory_step = step_a if step_a.domain == "theory" else step_b
        app_step = step_b if step_b.domain == "application" else step_a
        theory_concept = _STEP_CONCEPTS.get(theory_step.number, theory_step.title)
        app_concept = _STEP_CONCEPTS.get(app_step.number, app_step.title)
        templates = _TITLE_TEMPLATES.get("theory+application", _TITLE_TEMPLATES["default"])
        template = templates[hash((step_a.number, step_b.number)) % len(templates)]
        return template.format(
            theory_concept=theory_concept,
            app_concept=app_concept,
            concept_a=concept_a,
            concept_b=concept_b,
        )

    templates = _TITLE_TEMPLATES.get(pair_key, _TITLE_TEMPLATES["default"])
    template = templates[hash((step_a.number, step_b.number)) % len(templates)]
    return template.format(concept_a=concept_a, concept_b=concept_b)


# ---------------------------------------------------------------------------
# Connection detection
# ---------------------------------------------------------------------------

def _keyword_overlap(a: StepInfo, b: StepInfo) -> tuple[float, tuple[str, ...]]:
    """Compute keyword overlap strength between two steps."""
    ka = set(a.keywords)
    kb = set(b.keywords)
    # Remove universal keywords for overlap computation
    universal = {"session-types", "lattice"}
    ka_sig = ka - universal
    kb_sig = kb - universal

    shared = ka_sig & kb_sig
    if not ka_sig or not kb_sig:
        return 0.0, ()

    # Jaccard coefficient on significant keywords
    jaccard = len(shared) / len(ka_sig | kb_sig) if (ka_sig | kb_sig) else 0.0
    return jaccard, tuple(sorted(shared))


def _module_overlap(a: StepInfo, b: StepInfo) -> tuple[float, tuple[str, ...]]:
    """Compute module overlap strength between two steps."""
    ma = set(a.modules)
    mb = set(b.modules)
    shared = ma & mb
    if not ma or not mb:
        return 0.0, ()
    jaccard = len(shared) / len(ma | mb) if (ma | mb) else 0.0
    return jaccard, tuple(sorted(shared))


def _dependency_strength(a: StepInfo, b: StepInfo) -> float:
    """Check if there is a dependency between two steps."""
    if b.number in a.depends_on:
        return 0.6
    if a.number in b.depends_on:
        return 0.6
    return 0.0


def _complement_strength(a: StepInfo, b: StepInfo) -> float:
    """Check if steps are complementary (theory+application)."""
    if {a.domain, b.domain} == {"theory", "application"}:
        return 0.7
    if {a.domain, b.domain} == {"theory", "tool"}:
        return 0.4
    if {a.domain, b.domain} == {"application", "intersection"}:
        return 0.5
    return 0.0


def _suggest_venues(a: StepInfo, b: StepInfo) -> tuple[str, ...]:
    """Suggest venues based on domain pairing."""
    pair = (a.domain, b.domain)
    result = _VENUE_MAP.get(pair)
    if result is None:
        # Try reversed order
        result = _VENUE_MAP.get((b.domain, a.domain))
    if result is None:
        result = ("CONCUR", "ICE", "EPTCS")
    return result


def _detect_pair_connections(a: StepInfo, b: StepInfo) -> list[Connection]:
    """Detect all connections between a pair of steps."""
    connections: list[Connection] = []

    # 1. Keyword overlap
    kw_strength, shared_kw = _keyword_overlap(a, b)
    if kw_strength > 0.0:
        connections.append(Connection(
            step_a=a.number,
            step_b=b.number,
            connection_type="keyword",
            strength=kw_strength,
            paper_title=auto_generate_paper_title(a, b),
            venues=_suggest_venues(a, b),
            shared_keywords=shared_kw,
        ))

    # 2. Module overlap
    mod_strength, shared_mod = _module_overlap(a, b)
    if mod_strength > 0.0:
        connections.append(Connection(
            step_a=a.number,
            step_b=b.number,
            connection_type="module",
            strength=mod_strength,
            paper_title=auto_generate_paper_title(a, b),
            venues=_suggest_venues(a, b),
            shared_modules=shared_mod,
        ))

    # 3. Dependency chain
    dep_strength = _dependency_strength(a, b)
    if dep_strength > 0.0:
        connections.append(Connection(
            step_a=a.number,
            step_b=b.number,
            connection_type="dependency",
            strength=dep_strength,
            paper_title=auto_generate_paper_title(a, b),
            venues=_suggest_venues(a, b),
        ))

    # 4. Complementary results
    comp_strength = _complement_strength(a, b)
    if comp_strength > 0.0:
        connections.append(Connection(
            step_a=a.number,
            step_b=b.number,
            connection_type="complement",
            strength=comp_strength,
            paper_title=auto_generate_paper_title(a, b),
            venues=_suggest_venues(a, b),
        ))

    return connections


def detect_connections(steps: Sequence[StepInfo]) -> ConnectionReport:
    """For every pair of steps, detect connections and build a report.

    Returns only the top 10% of connections by strength as paper opportunities.
    """
    all_connections: list[Connection] = []
    by_domain: dict[str, list[Connection]] = {}

    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            conns = _detect_pair_connections(a, b)
            all_connections.extend(conns)
            for c in conns:
                domain_key = "+".join(sorted([a.domain, b.domain]))
                by_domain.setdefault(domain_key, []).append(c)

    # Sort by strength descending
    all_connections.sort(key=lambda c: c.strength, reverse=True)

    # Top 10% are paper opportunities
    cutoff = max(1, len(all_connections) // 10) if all_connections else 0
    opportunities = all_connections[:cutoff]

    return ConnectionReport(
        new_connections=tuple(all_connections),
        total_connections=len(all_connections),
        paper_opportunities=tuple(opportunities),
        by_domain=by_domain,
    )


def find_new_connections(
    new_step: StepInfo,
    existing_steps: Sequence[StepInfo],
) -> ConnectionReport:
    """Given a newly completed step, find all connections to existing steps.

    Runs after every sprint. Returns connections sorted by strength.
    """
    all_connections: list[Connection] = []
    by_domain: dict[str, list[Connection]] = {}

    for existing in existing_steps:
        conns = _detect_pair_connections(new_step, existing)
        all_connections.extend(conns)
        for c in conns:
            domain_key = "+".join(sorted([new_step.domain, existing.domain]))
            by_domain.setdefault(domain_key, []).append(c)

    all_connections.sort(key=lambda c: c.strength, reverse=True)
    cutoff = max(1, len(all_connections) // 10) if all_connections else 0
    opportunities = all_connections[:cutoff]

    return ConnectionReport(
        new_connections=tuple(all_connections),
        total_connections=len(all_connections),
        paper_opportunities=tuple(opportunities),
        by_domain=by_domain,
    )


# ---------------------------------------------------------------------------
# Keyword matrix
# ---------------------------------------------------------------------------

def keyword_matrix(steps: Sequence[StepInfo]) -> dict[str, dict[str, int]]:
    """Build a keyword co-occurrence matrix across all steps.

    Returns a dict-of-dicts where matrix[kw_a][kw_b] = number of steps
    that share both keywords. High co-occurrence but no existing paper
    = opportunity.
    """
    # Collect all significant keywords
    universal = {"session-types", "lattice"}
    all_keywords: set[str] = set()
    for s in steps:
        all_keywords.update(set(s.keywords) - universal)

    matrix: dict[str, dict[str, int]] = {kw: {} for kw in sorted(all_keywords)}

    for s in steps:
        sig_kw = sorted(set(s.keywords) - universal)
        for i, ka in enumerate(sig_kw):
            for kb in sig_kw[i + 1:]:
                matrix.setdefault(ka, {})[kb] = matrix.get(ka, {}).get(kb, 0) + 1
                matrix.setdefault(kb, {})[ka] = matrix.get(kb, {}).get(ka, 0) + 1

    return matrix


def domain_theory_matrix(
    steps: Sequence[StepInfo],
) -> list[tuple[StepInfo, StepInfo, str]]:
    """Classify each step and find theory x application/domain paper opportunities.

    Returns a list of (theory_step, other_step, opportunity_type) triples.
    Every theory x application pair is a potential paper.
    Every theory x domain (intersection) pair is a potential paper.
    """
    theory_steps = [s for s in steps if s.domain == "theory"]
    app_steps = [s for s in steps if s.domain == "application"]
    inter_steps = [s for s in steps if s.domain == "intersection"]

    opportunities: list[tuple[StepInfo, StepInfo, str]] = []

    for t in theory_steps:
        for a in app_steps:
            opportunities.append((t, a, "theory_x_application"))
        for x in inter_steps:
            opportunities.append((t, x, "theory_x_domain"))

    return opportunities


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="reticulate.connection_detector")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N connections (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    steps = scan_steps()
    report = detect_connections(steps)

    if args.json:
        import json
        output = {
            "total_connections": report.total_connections,
            "paper_opportunities": len(report.paper_opportunities),
            "top_connections": [
                {
                    "step_a": c.step_a,
                    "step_b": c.step_b,
                    "type": c.connection_type,
                    "strength": round(c.strength, 3),
                    "title": c.paper_title,
                    "venues": list(c.venues),
                }
                for c in report.new_connections[:args.top]
            ],
            "domains": {k: len(v) for k, v in report.by_domain.items()},
        }
        print(json.dumps(output, indent=2))
    else:
        print("=" * 75)
        print("  CONNECTION DETECTOR")
        print("=" * 75)
        print(f"  Steps scanned: {len(steps)}")
        print(f"  Total connections: {report.total_connections}")
        print(f"  Paper opportunities (top 10%): {len(report.paper_opportunities)}")
        print()

        print(f"  -- TOP {args.top} CONNECTIONS --")
        for i, c in enumerate(report.new_connections[:args.top], 1):
            print(f"  {i:2d}. [{c.connection_type:10s}] "
                  f"Step {c.step_a} <-> Step {c.step_b}  "
                  f"(strength: {c.strength:.2f})")
            print(f"      Title: {c.paper_title[:65]}")
            if c.shared_keywords:
                print(f"      Keywords: {', '.join(c.shared_keywords)}")
            if c.shared_modules:
                print(f"      Modules: {', '.join(c.shared_modules)}")
            print(f"      Venues: {', '.join(c.venues[:3])}")
            print()

        print("  -- DOMAIN SUMMARY --")
        for domain_pair, conns in sorted(report.by_domain.items()):
            print(f"    {domain_pair}: {len(conns)} connections")

        print("=" * 75)


if __name__ == "__main__":
    main()
