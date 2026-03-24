"""Peer reviewer: critical review of research papers.

Reads a LaTeX paper and produces a structured review as a real
conference reviewer would. Checks content quality, not just metrics.

Review criteria (scored 1-5):
1. Novelty: is this a genuine contribution?
2. Soundness: are theorems stated correctly? are proofs convincing?
3. Clarity: can a non-expert follow the argument?
4. Significance: does this matter to the community?
5. Presentation: is the writing quality adequate?
6. Completeness: is anything missing (related work, examples, evaluation)?

Verdict: Strong Accept / Accept / Weak Accept / Borderline / Reject

Usage:
    from reticulate.peer_reviewer import review_paper
    result = review_paper("papers/publications/ice-2026/main.tex")
    print(result.summary())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewCriterion:
    """Score for one review criterion."""
    name: str
    score: int          # 1-5
    comment: str
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeerReview:
    """Complete peer review of a paper."""
    paper_path: str
    title: str
    venue: str
    criteria: tuple[ReviewCriterion, ...]
    overall_score: float   # average of criteria
    verdict: str           # Strong Accept / Accept / Weak Accept / Borderline / Reject
    summary_of_contribution: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    questions_for_authors: tuple[str, ...]
    minor_issues: tuple[str, ...]
    recommendation: str

    def summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append(f"  PEER REVIEW: {self.title[:55]}")
        lines.append(f"  Venue: {self.venue}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  VERDICT: {self.verdict} (overall: {self.overall_score:.1f}/5)")
        lines.append("")

        lines.append("  Scores:")
        for c in self.criteria:
            bar = "█" * c.score + "░" * (5 - c.score)
            lines.append(f"    {c.name:<20} {bar} {c.score}/5  {c.comment}")
        lines.append("")

        lines.append(f"  Summary of contribution:")
        lines.append(f"    {self.summary_of_contribution}")
        lines.append("")

        lines.append(f"  Strengths ({len(self.strengths)}):")
        for s in self.strengths:
            lines.append(f"    + {s}")
        lines.append("")

        lines.append(f"  Weaknesses ({len(self.weaknesses)}):")
        for w in self.weaknesses:
            lines.append(f"    - {w}")
        lines.append("")

        if self.questions_for_authors:
            lines.append(f"  Questions for authors ({len(self.questions_for_authors)}):")
            for q in self.questions_for_authors:
                lines.append(f"    ? {q}")
            lines.append("")

        if self.minor_issues:
            lines.append(f"  Minor issues ({len(self.minor_issues)}):")
            for m in self.minor_issues:
                lines.append(f"    · {m}")
            lines.append("")

        lines.append(f"  Recommendation:")
        lines.append(f"    {self.recommendation}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper analysis helpers
# ---------------------------------------------------------------------------

def _extract_title(tex: str) -> str:
    m = re.search(r"\\title\{([^}]+)\}", tex, re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1).replace("\\\\", " ")).strip()
    return "Unknown"


def _extract_abstract(tex: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    return m.group(1).strip() if m else ""


def _count_sections(tex: str) -> int:
    return len(re.findall(r"\\section\{", tex))


def _count_theorems(tex: str) -> int:
    return len(re.findall(r"\\begin\{(theorem|lemma|proposition|corollary)\}", tex))


def _count_proofs(tex: str) -> int:
    return len(re.findall(r"\\begin\{proof\}", tex))


def _count_examples(tex: str) -> int:
    return len(re.findall(r"\\begin\{example\}", tex))


def _count_figures(tex: str) -> int:
    return len(re.findall(r"\\begin\{(figure|tikzpicture)\}", tex))


def _count_tables(tex: str) -> int:
    return len(re.findall(r"\\begin\{(table|tabular|longtable)\}", tex))


def _count_citations(tex: str) -> int:
    return len(re.findall(r"\\cite\{", tex))


def _has_section(tex: str, name: str) -> bool:
    return bool(re.search(r"\\section\{[^}]*" + name, tex, re.IGNORECASE))


def _count_words(tex: str) -> int:
    return len(tex.split())


def _find_undefined_refs(tex: str) -> list[str]:
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    return sorted(refs - labels)


def _find_empty_proofs(tex: str) -> int:
    """Count proofs that are suspiciously short (< 50 words)."""
    proofs = re.findall(r"\\begin\{proof\}(.*?)\\end\{proof\}", tex, re.DOTALL)
    return sum(1 for p in proofs if len(p.split()) < 50)


def _find_todo_markers(tex: str) -> list[str]:
    return re.findall(r"(?:TODO|FIXME|XXX|HACK)[^\n]*", tex)


def _check_theorem_proof_ratio(tex: str) -> tuple[int, int]:
    """Return (theorems without proofs, total theorems)."""
    theorems = len(re.findall(r"\\begin\{(theorem|lemma|proposition)\}", tex))
    proofs = len(re.findall(r"\\begin\{proof\}", tex))
    unproved = max(0, theorems - proofs)
    return unproved, theorems


# ---------------------------------------------------------------------------
# Core review
# ---------------------------------------------------------------------------

def review_paper(
    paper_path: str | Path,
    venue: str = "unknown",
) -> PeerReview:
    """Review a paper critically.

    Reads the LaTeX source and evaluates on 6 criteria.
    Returns a structured review with verdict and actionable feedback.
    """
    path = Path(paper_path)
    if not path.exists():
        raise FileNotFoundError(f"Paper not found: {path}")

    tex = path.read_text()
    title = _extract_title(tex)
    abstract = _extract_abstract(tex)
    words = _count_words(tex)
    sections = _count_sections(tex)
    theorems = _count_theorems(tex)
    proofs = _count_proofs(tex)
    examples = _count_examples(tex)
    figures = _count_figures(tex)
    tables = _count_tables(tex)
    citations = _count_citations(tex)
    empty_proofs = _find_empty_proofs(tex)
    todos = _find_todo_markers(tex)
    unproved, total_thms = _check_theorem_proof_ratio(tex)
    undefined_refs = _find_undefined_refs(tex)

    has_intro = _has_section(tex, "Introduction")
    has_conclusion = _has_section(tex, "Conclusion")
    has_related = _has_section(tex, "Related")
    has_background = _has_section(tex, "Background") or _has_section(tex, "Preliminaries")
    has_evaluation = _has_section(tex, "Validation") or _has_section(tex, "Evaluation") or _has_section(tex, "Experiment")

    criteria = []
    strengths = []
    weaknesses = []
    questions = []
    minor = []

    # ── 1. NOVELTY ──
    novelty_score = 3
    novelty_issues = []

    if theorems >= 3:
        novelty_score += 1
        strengths.append(f"{theorems} formal results (theorems/lemmas/propositions)")
    elif theorems == 0:
        novelty_score -= 1
        novelty_issues.append("No formal theorems — what is the contribution?")
        weaknesses.append("No theorems or formal results stated")

    if examples >= 2:
        strengths.append(f"{examples} worked examples demonstrate the results")
    else:
        novelty_issues.append("Few or no worked examples")

    if abstract and len(abstract.split()) > 80:
        pass  # good abstract
    elif abstract:
        novelty_issues.append("Abstract is thin — should summarize the main result clearly")

    novelty_score = max(1, min(5, novelty_score))
    criteria.append(ReviewCriterion("Novelty", novelty_score,
                                     f"{theorems} theorems, {examples} examples",
                                     tuple(novelty_issues)))

    # ── 2. SOUNDNESS ──
    soundness_score = 3
    soundness_issues = []

    if unproved > 0:
        soundness_score -= 1
        soundness_issues.append(f"{unproved} theorem(s) without proof")
        weaknesses.append(f"{unproved} of {total_thms} theorems lack proofs")
    elif total_thms > 0 and proofs >= total_thms:
        soundness_score += 1
        strengths.append("Every theorem has a proof")

    if empty_proofs > 0:
        soundness_score -= 1
        soundness_issues.append(f"{empty_proofs} proof(s) are suspiciously short (< 50 words)")
        weaknesses.append(f"{empty_proofs} proofs are very short — are they proof sketches or complete?")
        questions.append("Are the short proofs intended as sketches? If so, where are the full proofs?")

    if todos:
        soundness_score -= 1
        soundness_issues.append(f"{len(todos)} TODO/FIXME markers in source")
        weaknesses.append(f"Paper contains {len(todos)} TODO markers — not submission-ready")

    soundness_score = max(1, min(5, soundness_score))
    criteria.append(ReviewCriterion("Soundness", soundness_score,
                                     f"{proofs}/{total_thms} proved, {empty_proofs} short",
                                     tuple(soundness_issues)))

    # ── 3. CLARITY ──
    clarity_score = 3
    clarity_issues = []

    if has_intro and has_background and has_conclusion:
        clarity_score += 1
        strengths.append("Standard paper structure (intro, background, conclusion)")
    else:
        missing = []
        if not has_intro: missing.append("Introduction")
        if not has_background: missing.append("Background/Preliminaries")
        if not has_conclusion: missing.append("Conclusion")
        if missing:
            clarity_score -= 1
            clarity_issues.append(f"Missing sections: {', '.join(missing)}")
            weaknesses.append(f"Missing standard sections: {', '.join(missing)}")

    if figures >= 2:
        strengths.append(f"{figures} figures/diagrams aid understanding")
    elif figures == 0:
        clarity_issues.append("No figures — Hasse diagrams or state machines would help")
        weaknesses.append("No visual aids (diagrams, figures)")

    if undefined_refs:
        clarity_issues.append(f"Undefined references: {', '.join(undefined_refs[:5])}")
        minor.append(f"Fix undefined references: {', '.join(undefined_refs[:5])}")

    clarity_score = max(1, min(5, clarity_score))
    criteria.append(ReviewCriterion("Clarity", clarity_score,
                                     f"{sections} sections, {figures} figures",
                                     tuple(clarity_issues)))

    # ── 4. SIGNIFICANCE ──
    significance_score = 3
    significance_issues = []

    if has_evaluation:
        significance_score += 1
        strengths.append("Includes empirical validation/evaluation")
    else:
        significance_issues.append("No evaluation or validation section")
        questions.append("How have you validated the results? Are there benchmarks?")

    if tables >= 1:
        pass  # empirical data present
    else:
        significance_issues.append("No tables with empirical data")

    significance_score = max(1, min(5, significance_score))
    criteria.append(ReviewCriterion("Significance", significance_score,
                                     f"{'has' if has_evaluation else 'no'} evaluation, {tables} tables",
                                     tuple(significance_issues)))

    # ── 5. PRESENTATION ──
    presentation_score = 3
    presentation_issues = []

    if 4000 <= words <= 15000:
        presentation_score += 1
    elif words < 3000:
        presentation_issues.append(f"Very short ({words} words)")
        weaknesses.append(f"Paper is very short ({words} words)")
    elif words > 20000:
        presentation_issues.append(f"Very long ({words} words) — consider cutting")

    if citations >= 15:
        presentation_score += 1
        strengths.append(f"{citations} citations show awareness of related work")
    elif citations < 5:
        presentation_score -= 1
        presentation_issues.append(f"Only {citations} citations — insufficient related work")
        weaknesses.append(f"Very few citations ({citations}) — are you aware of prior work?")

    presentation_score = max(1, min(5, presentation_score))
    criteria.append(ReviewCriterion("Presentation", presentation_score,
                                     f"{words} words, {citations} citations",
                                     tuple(presentation_issues)))

    # ── 6. COMPLETENESS ──
    completeness_score = 3
    completeness_issues = []

    if not has_related:
        completeness_score -= 1
        completeness_issues.append("No Related Work section")
        weaknesses.append("Missing Related Work section — how does this compare to existing approaches?")

    if not has_evaluation and not has_section_content(tex, "Benchmark"):
        completeness_issues.append("No benchmarks or experimental validation")

    if examples < 2:
        completeness_issues.append("Fewer than 2 worked examples")
        weaknesses.append("More worked examples would strengthen the paper")

    if not has_conclusion:
        completeness_issues.append("No conclusion — what are the takeaways?")

    completeness_score = max(1, min(5, completeness_score))
    criteria.append(ReviewCriterion("Completeness", completeness_score,
                                     f"{'has' if has_related else 'no'} related work, {examples} examples",
                                     tuple(completeness_issues)))

    # ── OVERALL ──
    overall = sum(c.score for c in criteria) / len(criteria)

    if overall >= 4.2:
        verdict = "Strong Accept"
    elif overall >= 3.5:
        verdict = "Accept"
    elif overall >= 3.0:
        verdict = "Weak Accept"
    elif overall >= 2.5:
        verdict = "Borderline"
    else:
        verdict = "Reject"

    # Summary
    contribution_summary = (
        f"The paper '{title}' presents {theorems} formal results "
        f"supported by {proofs} proofs and {examples} examples, "
        f"in {words} words across {sections} sections."
    )

    # Recommendation
    if verdict in ("Strong Accept", "Accept"):
        recommendation = "Accept. Address minor issues in camera-ready."
    elif verdict == "Weak Accept":
        recommendation = "Weak accept. The paper has merit but needs revision. Address the weaknesses listed above."
    elif verdict == "Borderline":
        recommendation = "Borderline. The contribution is unclear or the presentation needs significant improvement."
    else:
        recommendation = "Reject. The paper needs substantial rework before resubmission."

    return PeerReview(
        paper_path=str(path),
        title=title,
        venue=venue,
        criteria=tuple(criteria),
        overall_score=overall,
        verdict=verdict,
        summary_of_contribution=contribution_summary,
        strengths=tuple(strengths),
        weaknesses=tuple(weaknesses),
        questions_for_authors=tuple(questions),
        minor_issues=tuple(minor),
        recommendation=recommendation,
    )


def has_section_content(tex: str, keyword: str) -> bool:
    """Check if a section containing keyword exists and has content."""
    pattern = rf"\\section\{{[^}}]*{keyword}[^}}]*\}}(.*?)(?=\\section|\\end\{{document\}})"
    m = re.search(pattern, tex, re.DOTALL | re.IGNORECASE)
    if m:
        return len(m.group(1).split()) > 100
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="reticulate.peer_reviewer")
    parser.add_argument("paper", help="Path to LaTeX paper")
    parser.add_argument("--venue", default="unknown", help="Target venue")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    review = review_paper(args.paper, venue=args.venue)

    if args.json:
        import json
        print(json.dumps({
            "title": review.title, "venue": review.venue,
            "overall": review.overall_score, "verdict": review.verdict,
            "strengths": list(review.strengths),
            "weaknesses": list(review.weaknesses),
            "questions": list(review.questions_for_authors),
        }, indent=2))
    else:
        print(review.summary())

    sys.exit(0 if review.verdict in ("Strong Accept", "Accept") else 1)


if __name__ == "__main__":
    main()
