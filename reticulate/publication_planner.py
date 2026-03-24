"""Publication planner: maintains a living list of publishable results.

Scans the research programme, identifies publishable contributions,
groups them into papers, matches to venues, and maintains a prioritized
publication plan. Runs after every sprint to detect new opportunities.

Principles:
1. Every paper must be a REAL contribution (novel theorem, tool, empirical result)
2. Papers must be ADEQUATE for the venue (scope, format, community)
3. Cross-disciplinary venues welcome IF they welcome formal methods
4. The list must be ALWAYS COMPLETE — no result left unmapped
5. Results can combine into one paper, or stand alone
6. Earlier papers establish priority; later ones extend

Usage:
    python -m reticulate.publication_planner
    python -m reticulate.publication_planner --save
    python -m reticulate.publication_planner --json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PublishableResult:
    """A single publishable result from the research programme."""
    step: str
    title: str
    contribution_type: str   # "theorem" | "algorithm" | "tool" | "empirical" | "survey" | "application"
    novelty: str             # one-sentence: what's new?
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class PaperProposal:
    """A proposed paper combining one or more results."""
    id: str                  # e.g. "F1", "M2", "AI1"
    title: str
    layer: str               # "foundation" | "consequence" | "intersection" | "application" | "extension" | "synthesis"
    steps: tuple[str, ...]   # contributing steps
    contribution: str        # what the paper contributes (1-2 sentences)
    venues: tuple[str, ...]  # ranked venue list
    format: str              # "workshop 4-8pp" | "conference 15pp" | "journal 25-40pp"
    priority: str            # "immediate" | "short-term" | "medium-term" | "long-term"
    deadline: str            # nearest deadline or "rolling"
    cites: tuple[str, ...]   # paper IDs this paper should cite
    status: str              # "planned" | "drafting" | "submitted" | "accepted" | "published"


@dataclass(frozen=True)
class PublicationPlan:
    """Complete publication plan."""
    results: tuple[PublishableResult, ...]
    papers: tuple[PaperProposal, ...]
    unmapped_results: tuple[str, ...]  # steps not yet in any paper

    def summary(self) -> str:
        lines = []
        lines.append("=" * 75)
        lines.append("  PUBLICATION PLAN")
        lines.append("=" * 75)
        lines.append(f"  Results: {len(self.results)}")
        lines.append(f"  Papers: {len(self.papers)}")
        lines.append(f"  Unmapped: {len(self.unmapped_results)}")
        lines.append("")

        by_layer = {}
        for p in self.papers:
            by_layer.setdefault(p.layer, []).append(p)

        for layer in ["foundation", "consequence", "intersection", "application", "extension", "synthesis"]:
            papers = by_layer.get(layer, [])
            if not papers:
                continue
            lines.append(f"  ── {layer.upper()} ({len(papers)} papers) ──")
            for p in papers:
                venues = ", ".join(p.venues[:3])
                lines.append(f"    [{p.id:>4}] {p.title[:55]}")
                lines.append(f"           Steps: {', '.join(p.steps[:5])}{'...' if len(p.steps) > 5 else ''}")
                lines.append(f"           Venues: {venues}")
                lines.append(f"           {p.priority} | {p.format} | {p.status}")
            lines.append("")

        if self.unmapped_results:
            lines.append(f"  ── UNMAPPED ({len(self.unmapped_results)}) ──")
            for s in self.unmapped_results[:10]:
                lines.append(f"    Step {s}")
            if len(self.unmapped_results) > 10:
                lines.append(f"    ... and {len(self.unmapped_results) - 10} more")

        lines.append("=" * 75)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result scanner
# ---------------------------------------------------------------------------

def _scan_results(root: Path) -> list[PublishableResult]:
    """Scan all steps and classify their publishable contributions."""
    steps_dir = root / "papers" / "steps"
    results = []

    # Classification rules
    theorem_words = {"lattice", "theorem", "meet", "join", "embedding", "involution",
                     "universality", "endomorphism", "reticular", "termination",
                     "subtyping", "duality", "chomsky", "coincide", "equivalence",
                     "supermodular", "coverability", "invariant", "realizab"}
    intersection_words = {"petri", "ccs", "csp", "failure", "event struct", "algebra",
                         "grammar", "gratzer", "distributiv", "tensor", "category",
                         "coproduct", "equalizer", "coequalizer", "polarity", "channel",
                         "monoidal", "galois", "marking"}
    application_words = {"mcp", "a2a", "openapi", "fhir", "monitor", "compress",
                        "uml", "gof", "narrative", "big brother", "interdiscipl",
                        "benchmark", "ci gate", "runtime"}
    extension_words = {"lambda", "game", "power", "supermod", "crdt", "parity",
                      "quantit", "rational", "pendular", "categorical game"}
    tool_words = {"tool", "construction", "state space", "multiparty", "projection",
                 "recursion", "parallel", "product", "api contract", "workflow"}

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
            tex = main.read_text()
            tmatch = re.search(r"\\title\{[^}]*?([A-Z].*?)(?:\\\\|\})", tex)
            if tmatch:
                title = tmatch.group(1).strip()[:80]

        tl = title.lower() + " " + raw_title.lower()

        # Classify
        if any(w in tl for w in theorem_words):
            ctype = "theorem"
        elif any(w in tl for w in intersection_words):
            ctype = "empirical"
        elif any(w in tl for w in application_words):
            ctype = "application"
        elif any(w in tl for w in extension_words):
            ctype = "theorem"
        elif any(w in tl for w in tool_words):
            ctype = "tool"
        else:
            ctype = "empirical"

        # Generate novelty description
        novelty = f"Step {num}: {title}"

        # Keywords
        keywords = tuple(w for w in ["session types", "lattice", "reticulate"]
                        if True)  # base keywords always

        results.append(PublishableResult(
            step=num, title=title, contribution_type=ctype,
            novelty=novelty, keywords=keywords,
        ))

    return results


# ---------------------------------------------------------------------------
# Paper proposals
# ---------------------------------------------------------------------------

def _generate_papers(results: list[PublishableResult]) -> list[PaperProposal]:
    """Generate paper proposals from publishable results."""
    step_set = {r.step for r in results}
    papers = []

    # ── FOUNDATION ──
    papers.append(PaperProposal(
        id="F1", title="The Reticulate Theorem: Session Type State Spaces Form Lattices",
        layer="foundation", steps=("3", "4", "5", "5b", "6"),
        contribution="Every well-formed session type state space, quotiented by SCCs, forms a bounded lattice. Branch gives meet, selection gives join, parallel gives product, recursion is absorbed.",
        venues=("CONCUR 2026", "JLAMP", "LICS 2027"), format="conference 15pp",
        priority="immediate", deadline="Apr 27, 2026", cites=(), status="drafting",
    ))
    papers.append(PaperProposal(
        id="F2", title="State Space Construction and Reticular Form",
        layer="foundation", steps=("1", "9", "156"),
        contribution="Algorithm for building session type state spaces + characterization of which state machines are reticular (expressible as session types).",
        venues=("TACAS 2027", "SCP"), format="conference 15pp",
        priority="medium-term", deadline="~Oct 2026", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="F3", title="Recursion, Termination, and Trace Languages",
        layer="foundation", steps=("6b", "13a", "14"),
        contribution="Fair termination via tau-completion, unfolded lattice visualization, and Chomsky classification of session type trace languages (all regular for finite state spaces).",
        venues=("FoSSaCS 2027", "ICALP 2027"), format="conference 15pp",
        priority="medium-term", deadline="~Oct 2026", cites=("F1",), status="planned",
    ))

    # ── CONSEQUENCE ──
    papers.append(PaperProposal(
        id="M1", title="Subtyping is Embedding: A Lattice-Theoretic View of Session Type Refinement",
        layer="consequence", steps=("7",),
        contribution="Gay-Hole width subtyping corresponds to lattice embedding between reticulates.",
        venues=("ICE 2026", "ICTAC 2026", "APLAS 2026"), format="workshop 8-15pp",
        priority="immediate", deadline="Apr 2, 2026", cites=("F1",), status="drafting",
    ))
    papers.append(PaperProposal(
        id="M2", title="Duality as Lattice Involution",
        layer="consequence", steps=("8",),
        contribution="Session type duality (Branch↔Select) is a lattice anti-isomorphism. Dual of dual is identity. Duality reverses subtyping.",
        venues=("PLACES 2027", "ICE 2027"), format="workshop 8pp",
        priority="short-term", deadline="~Feb 2027", cites=("F1", "M1"), status="planned",
    ))
    papers.append(PaperProposal(
        id="M3", title="97% of Transition Labels Are Lattice Endomorphisms",
        layer="consequence", steps=("10",),
        contribution="Transition maps are order/meet/join-preserving in 97.6% of cases across 105 benchmarks. Analysis of the 2.4% that fail.",
        venues=("ICE 2026", "CONCUR 2026 workshop"), format="workshop 8pp",
        priority="short-term", deadline="~Jun 2026", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="M4", title="Multiparty Projection as Surjective Lattice Morphism",
        layer="consequence", steps=("11", "12"),
        contribution="MPST projection from global to local types is a surjective, order-preserving lattice morphism.",
        venues=("FORTE 2027", "COORDINATION 2027"), format="conference 15pp",
        priority="medium-term", deadline="~Feb 2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="M5", title="Realizability: Which Bounded Lattices Are Session Types?",
        layer="consequence", steps=("9", "156"),
        contribution="Characterization of which state machines can be expressed as session types (reticular form) + reconstruction algorithm.",
        venues=("ESOP 2027", "FoSSaCS 2027"), format="conference 15pp",
        priority="medium-term", deadline="~Oct 2026", cites=("F1",), status="planned",
    ))

    # ── INTERSECTION: EVENT STRUCTURES ──
    papers.append(PaperProposal(
        id="E1", title="Session Types as Event Structures",
        layer="intersection", steps=("16", "17", "19", "20"),
        contribution="Translation from session types to prime event structures. Configuration domains correspond to reticulates. ES morphisms correspond to session type morphisms.",
        venues=("CONCUR 2027", "LICS 2027"), format="conference 15pp",
        priority="medium-term", deadline="~2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="E2", title="Dialogue Types: Session Types Meet Lorenzen Games",
        layer="intersection", steps=("18",),
        contribution="Session types as dialogue games in the Lorenzen tradition. Proponent/Opponent correspond to Branch/Select.",
        venues=("FoSSaCS 2027", "CSL 2027"), format="conference 15pp",
        priority="medium-term", deadline="~2027", cites=("F1",), status="planned",
    ))

    # ── INTERSECTION: PETRI NETS ──
    papers.append(PaperProposal(
        id="N1", title="Session Types as Petri Nets: Encoding, Invariants, and Benchmarks",
        layer="intersection", steps=("21", "22", "23", "24", "25"),
        contribution="State-machine encoding of session types as 1-safe free-choice Petri nets. Place invariants, coverability, marking lattice. All 105 benchmarks verified.",
        venues=("Petri Nets 2027", "CONCUR 2027", "ATVA 2027"), format="conference 15pp",
        priority="medium-term", deadline="~2027", cites=("F1",), status="planned",
    ))

    # ── INTERSECTION: PROCESS ALGEBRA ──
    papers.append(PaperProposal(
        id="A1", title="Three Views Coincide: CCS, CSP, and Failures on Session Type Lattices",
        layer="intersection", steps=("26", "27", "28", "29c"),
        contribution="CCS bisimulation, CSP trace refinement, and failure refinement coincide on session type state spaces (all deterministic LTS). Verified on 105 benchmarks.",
        venues=("CONCUR 2026", "LICS 2027", "EXPRESS/SOS 2026"), format="conference 15pp",
        priority="short-term", deadline="Jun 2026", cites=("F1", "M1"), status="planned",
    ))

    # ── INTERSECTION: ALGEBRA ──
    papers.append(PaperProposal(
        id="G1", title="Distributivity, Compression, and Protocol Quality",
        layer="intersection", steps=("29", "29a", "29b", "70d", "70e"),
        contribution="Distributivity census of 105 benchmarks. Compression ratio perfectly predicts non-distributivity. Reconvergence degree as engineering quality metric.",
        venues=("FORTE 2027", "COORDINATION 2027", "ICE 2027"), format="conference 15pp",
        priority="short-term", deadline="~Feb 2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="G2", title="Algebraic Invariants for Session Type Lattices",
        layer="intersection", steps=("30",),
        contribution="Möbius function, Rota polynomial, spectral radius, von Neumann entropy as protocol complexity measures.",
        venues=("JLAMP", "Order (journal)"), format="journal 25pp",
        priority="long-term", deadline="rolling", cites=("F1",), status="planned",
    ))

    # ── INTERSECTION: CATEGORY THEORY ──
    papers.append(PaperProposal(
        id="C1", title="The Category SessLat: Products, Equalizers, and Limits",
        layer="intersection", steps=("163", "164", "165", "166", "168"),
        contribution="Session type lattices form a category with products (parallel), equalizers, but no general coproducts or coequalizers. Tensor product as monoidal structure.",
        venues=("CALCO 2027", "MSCS", "FoSSaCS 2027"), format="conference 15pp",
        priority="medium-term", deadline="~2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="C2", title="Polarity and Galois Connections in Session Types",
        layer="intersection", steps=("155b",),
        contribution="FCA-based polarity analysis. Branch/select polarity forms a Galois connection between state and label lattices.",
        venues=("ICFCA 2027", "CLA 2027"), format="conference 12pp",
        priority="long-term", deadline="~2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="C3", title="Channel Duality and Asynchronous Session Types",
        layer="intersection", steps=("157a", "157b"),
        contribution="Channel duality as role-restricted parallel. Buffered channels as lattice extensions preserving 1-safety.",
        venues=("COORDINATION 2027", "FORTE 2027"), format="conference 15pp",
        priority="medium-term", deadline="~Feb 2027", cites=("F1",), status="planned",
    ))

    # ── APPLICATION: AI AGENTS ──
    papers.append(PaperProposal(
        id="AI1", title="Session Types for AI Agent Protocol Conformance",
        layer="application", steps=("70", "70b"),
        contribution="First formal conformance testing tool for MCP and A2A protocols. Session type models, runtime tester, self-referential MCP server.",
        venues=("AAMAS 2027", "AGENT@ICSE 2027", "WMAC@AAAI 2027"), format="conference 10pp",
        priority="short-term", deadline="~Oct 2026", cites=("F1", "M1"), status="planned",
    ))
    papers.append(PaperProposal(
        id="AI2", title="Natural Language to Session Types via AI Agent Mediation",
        layer="application", steps=("70c",),
        contribution="AI agents translate natural language protocol descriptions to session types. 30 protocols, distributivity dichotomy.",
        venues=("ICE 2027", "PLACES 2027", "NLP4SE"), format="workshop 8pp",
        priority="medium-term", deadline="~2027", cites=("F1", "AI1"), status="planned",
    ))

    # ── APPLICATION: INDUSTRY ──
    papers.append(PaperProposal(
        id="I1", title="Stateful API Contracts: Session Types as OpenAPI Extensions",
        layer="application", steps=("71",),
        contribution="REST API lifecycle verification via session types. OpenAPI x-session-type extension. CRUD, auth, payment patterns.",
        venues=("ICSE-SEIP 2027", "ESEC/FSE 2027", "API World"), format="conference 10pp",
        priority="short-term", deadline="~Oct 2026", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="I2", title="Session Types for FHIR Clinical Workflow Verification",
        layer="application", steps=("72",),
        contribution="Model HL7 FHIR workflows as session types. 8 clinical workflows verified. Non-distributivity from shared triage paths.",
        venues=("MedInfo 2027", "FHIR DevDays", "AMIA"), format="conference 10pp",
        priority="medium-term", deadline="~2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="I3", title="Runtime Monitor Generation from Session Types",
        layer="application", steps=("80",),
        contribution="Generate Python/Java/Express.js middleware that enforces session type protocols at runtime. Soundness and transparency proofs.",
        venues=("ECOOP 2027", "ISSTA 2027", "RV 2027"), format="conference 15pp",
        priority="medium-term", deadline="~Oct 2026", cites=("F1", "M1"), status="planned",
    ))

    # ── APPLICATION: CROSS-DOMAIN ──
    papers.append(PaperProposal(
        id="X1", title="Session Types for UML and Design Patterns",
        layer="application", steps=("50", "51"),
        contribution="UML state diagrams as reticulates. GoF behavioral patterns (Iterator, State, Observer) as session types.",
        venues=("MODELS 2027", "ECOOP 2027"), format="conference 15pp",
        priority="long-term", deadline="~2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="X2", title="Session Types for Narrative Structure",
        layer="application", steps=("52",),
        contribution="Propp's narrative morphology as session types. Story branching forms lattices.",
        venues=("ICIDS 2027", "CHI 2027 alt.chi"), format="conference 8pp",
        priority="long-term", deadline="~2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="X3", title="Session Types Beyond Computer Science",
        layer="application", steps=("53",),
        contribution="Cross-disciplinary applications: biology, law, music, cooking, sports.",
        venues=("Dagstuhl seminar proposal", "survey paper"), format="survey 20pp",
        priority="long-term", deadline="rolling", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="X4", title="Big Brother as a Session Type Game",
        layer="application", steps=("104",),
        contribution="Game show protocol as session type. Game-theoretic analysis of elimination.",
        venues=("FUN 2027", "AAMAS poster"), format="short paper 6pp",
        priority="long-term", deadline="~2027", cites=("F1", "GT1"), status="planned",
    ))

    # ── EXTENSION: LAMBDA CALCULUS ──
    papers.append(PaperProposal(
        id="L1", title="The λ_S Hierarchy: A Typed Lambda Calculus for Session Types",
        layer="extension", steps=("200", "200a", "200d", "200e", "200f", "200g"),
        contribution="Five-level lambda calculus hierarchy (λ_S⁰–λ_S⁴) for session types. Preservation, progress, session completion.",
        venues=("ESOP 2027", "POPL 2027", "ICFP 2027"), format="conference 15pp",
        priority="medium-term", deadline="~Oct 2026", cites=("F1",), status="planned",
    ))

    # ── EXTENSION: GAME THEORY ──
    papers.append(PaperProposal(
        id="GT1", title="Session Types as Games: A Lattice-Theoretic Game Semantics",
        layer="extension", steps=("900", "900a"),
        contribution="Session types as two-player games. Lattice structure gives equilibria. Nash existence for finite session type games.",
        venues=("LICS 2027", "FoSSaCS 2027", "CSL 2027"), format="conference 15pp",
        priority="medium-term", deadline="~2027", cites=("F1",), status="planned",
    ))
    papers.append(PaperProposal(
        id="GT2", title="Supermodularity and CRDTs from Session Type Lattices",
        layer="extension", steps=("900c", "900e"),
        contribution="Cooperative session type games are supermodular. CRDT-like convergence from lattice structure.",
        venues=("CONCUR 2027", "DISC 2027"), format="conference 15pp",
        priority="long-term", deadline="~2027", cites=("F1", "GT1"), status="planned",
    ))
    papers.append(PaperProposal(
        id="GT3", title="Parity Games and Quantitative Verification for Recursive Session Types",
        layer="extension", steps=("900f", "900g"),
        contribution="Recursive session types as parity games. Quantitative protocol languages with quality lattices.",
        venues=("CSL 2027", "LICS 2027"), format="conference 15pp",
        priority="long-term", deadline="~2027", cites=("F1", "GT1"), status="planned",
    ))
    papers.append(PaperProposal(
        id="GT4", title="Rational Verification and Pendular Types",
        layer="extension", steps=("900h", "900i"),
        contribution="Rational verification for session type protocols. Pendular types for oscillating protocols.",
        venues=("AAMAS 2027", "AAAI 2027"), format="conference 10pp",
        priority="long-term", deadline="~2027", cites=("F1", "GT1"), status="planned",
    ))
    papers.append(PaperProposal(
        id="GT5", title="Protocol Power and Categorical Games",
        layer="extension", steps=("900b", "900d"),
        contribution="Protocol power decomposition. Category of lattice games (LGam) with symmetric monoidal structure.",
        venues=("CALCO 2027", "MSCS"), format="conference/journal",
        priority="long-term", deadline="~2027", cites=("F1", "GT1", "C1"), status="planned",
    ))

    # ── TOOL PAPERS ──
    papers.append(PaperProposal(
        id="T1", title="Reticulate: A Session Type Lattice Checker",
        layer="application", steps=("1", "2"),
        contribution="Python tool: parser, state-space builder, lattice checker, MCP server, CI gate, 4000+ tests, 105 benchmarks.",
        venues=("TACAS 2027", "SCP"), format="tool paper 15pp",
        priority="short-term", deadline="~Oct 2026", cites=("F1",), status="planned",
    ))

    # ── SYNTHESIS ──
    papers.append(PaperProposal(
        id="S1", title="Session Types as Algebraic Reticulates: A Complete Treatment",
        layer="synthesis", steps=tuple(r.step for r in results),
        contribution="Journal paper unifying all results: lattice theorem, subtyping, duality, process algebra, Petri nets, category theory, applications.",
        venues=("TOPLAS", "JLAMP", "LMCS"), format="journal 40-60pp",
        priority="long-term", deadline="Q4 2026", cites=tuple(p.id for p in papers), status="planned",
    ))

    return papers


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def plan_publications(root: Path | str | None = None) -> PublicationPlan:
    """Generate the complete publication plan."""
    if root is None:
        for candidate in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent.parent]:
            if (candidate / "papers" / "steps").is_dir():
                root = candidate
                break
        else:
            root = Path.cwd()
    else:
        root = Path(root)

    results = _scan_results(root)
    papers = _generate_papers(results)

    # Find unmapped results
    mapped_steps = set()
    for p in papers:
        mapped_steps.update(p.steps)
    unmapped = tuple(r.step for r in results if r.step not in mapped_steps)

    return PublicationPlan(
        results=tuple(results),
        papers=tuple(papers),
        unmapped_results=unmapped,
    )


def save_plan(plan: PublicationPlan, root: Path) -> Path:
    """Save publication plan to docs/planning/publication-plan.md."""
    lines = ["# Publication Plan", "",
             f"**Generated**: auto-updated by publication_planner.py", "",
             f"**Results**: {len(plan.results)} publishable",
             f"**Papers**: {len(plan.papers)} proposed",
             f"**Unmapped**: {len(plan.unmapped_results)}", ""]

    by_priority = {"immediate": [], "short-term": [], "medium-term": [], "long-term": []}
    for p in plan.papers:
        by_priority.get(p.priority, by_priority["long-term"]).append(p)

    for priority in ["immediate", "short-term", "medium-term", "long-term"]:
        papers = by_priority[priority]
        if not papers:
            continue
        lines.append(f"## {priority.upper()} ({len(papers)} papers)")
        lines.append("")
        lines.append("| ID | Title | Steps | Venues | Format | Status |")
        lines.append("|---|---|---|---|---|---|")
        for p in papers:
            venues = ", ".join(p.venues[:2])
            steps = ", ".join(p.steps[:4]) + ("..." if len(p.steps) > 4 else "")
            lines.append(f"| {p.id} | {p.title[:50]} | {steps} | {venues} | {p.format} | {p.status} |")
        lines.append("")

    path = root / "docs" / "planning" / "publication-plan.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="reticulate.publication_planner")
    parser.add_argument("--save", action="store_true", help="Save plan to docs/planning/")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args(argv)

    plan = plan_publications()

    if args.json:
        output = {
            "results": len(plan.results),
            "papers": [{"id": p.id, "title": p.title, "layer": p.layer,
                        "venues": list(p.venues), "priority": p.priority,
                        "deadline": p.deadline, "status": p.status} for p in plan.papers],
            "unmapped": list(plan.unmapped_results),
        }
        print(json.dumps(output, indent=2))
    else:
        print(plan.summary())

    if args.save:
        root = Path(__file__).parent.parent.parent
        path = save_plan(plan, root)
        print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()
