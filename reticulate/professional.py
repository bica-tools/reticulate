"""Professional session types: mapping human professions to protocols (Step 60i).

Models human professions as session types, capturing the interaction patterns
inherent in professional activities:

- **Healthcare**: Doctor-patient consultations with examination, diagnosis,
  treatment, and follow-up cycles.
- **Education**: Teacher-student interactions with lectures, assignments,
  and grading.
- **Technology**: Developer workflows with tickets, code review, and deployment.
- **Service**: Customer-facing transactions (barista, waiter, hairdresser).
- **Creative**: Iterative rehearsal-performance-feedback loops.
- **Legal**: Consultation, representation, and adjudication protocols.
- **Finance**: Application, review, and approval workflows.
- **Emergency**: Dispatch-response-assessment protocols.

Each profession's session type captures its characteristic interaction
pattern — whether it is a consultation, transaction, instruction,
collaboration, negotiation, or performance.  Analysis reveals structural
properties: branching entropy (decision complexity), pendularity
(turn-taking regularity), and binding energy (cyclic commitment).

This module provides:
    ``get_profession(name)``         — look up a profession profile.
    ``analyze_profession(name)``     — full structural analysis.
    ``compare_professions(a, b)``    — pairwise comparison and compatibility.
    ``professions_by_category(cat)`` — list professions in a category.
    ``most_complex_profession()``    — profession with most states.
    ``most_compatible_pair()``       — highest-compatibility pair.
    ``profession_lattice()``         — subsumption map across professions.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import parse, Branch, End, Rec, Select, SessionType
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice
from reticulate.pendular import is_pendular
from reticulate.information import branching_entropy
from reticulate.negotiation import compatibility_score
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfessionProfile:
    """Profile describing a profession's session-type characteristics.

    Attributes:
        name: Profession name (e.g. "doctor", "teacher").
        category: Domain category (healthcare, education, technology, etc.).
        session_type_str: The session type string encoding the profession.
        interaction_pattern: High-level interaction style.
        is_client_facing: Whether the profession directly serves clients.
        recursion_type: Kind of recursion in the session type.
    """

    name: str
    category: str
    session_type_str: str
    interaction_pattern: str
    is_client_facing: bool
    recursion_type: str


@dataclass(frozen=True)
class ProfessionAnalysis:
    """Full structural analysis of a profession's session type.

    Attributes:
        profile: The profession profile.
        state_count: Number of states in the state space.
        transition_count: Number of transitions.
        is_lattice: Whether the state space forms a lattice.
        is_pendular: Whether the state space has strict turn-taking.
        branching_entropy: Average Shannon entropy per state.
        binding_energy: Fraction of states in non-trivial SCCs.
        complexity_class: Qualitative complexity classification.
    """

    profile: ProfessionProfile
    state_count: int
    transition_count: int
    is_lattice: bool
    is_pendular: bool
    branching_entropy: float
    binding_energy: float
    complexity_class: str


@dataclass(frozen=True)
class ProfessionComparison:
    """Comparison between two professions.

    Attributes:
        profession_a: Name of the first profession.
        profession_b: Name of the second profession.
        compatibility_score: Jaccard similarity of top-level methods.
        shared_interactions: Methods shared by both professions.
        unique_to_a: Methods unique to the first profession.
        unique_to_b: Methods unique to the second profession.
        subsumption: Subtyping relationship between the two.
    """

    profession_a: str
    profession_b: str
    compatibility_score: float
    shared_interactions: tuple[str, ...]
    unique_to_a: tuple[str, ...]
    unique_to_b: tuple[str, ...]
    subsumption: str


# ---------------------------------------------------------------------------
# Profession library
# ---------------------------------------------------------------------------

PROFESSION_LIBRARY: dict[str, str] = {
    # Healthcare
    "doctor": (
        "rec X . &{examine: +{diagnose: &{treat: +{followup: X, discharge: end},"
        " refer: end}}, emergency: +{stabilize: &{admit: X, transfer: end}}}"
    ),
    "nurse": (
        "rec X . &{assess: +{medicate: X, alert_doctor: end},"
        " monitor: +{stable: X, critical: end}}"
    ),
    "pharmacist": (
        "&{prescribe: +{dispense: &{counsel: end}, reject: +{notify: end}}}"
    ),
    # Education
    "teacher": (
        "rec X . &{present: +{question: &{answer: +{correct: X,"
        " incorrect: +{explain: X}}}}, assign: &{submit: +{grade: end}}}"
    ),
    "professor": (
        "rec X . &{lecture: +{discuss: X},"
        " supervise: &{draft: +{review: &{revise: X, accept: end}}},"
        " examine: +{pass: end, fail: end}}"
    ),
    # Technology
    "developer": (
        "rec X . &{ticket: +{implement: &{review: +{approve: X,"
        " reject: +{fix: X}, close: end}}}}"
    ),
    "sysadmin": (
        "rec X . &{alert: +{diagnose: &{fix: +{verify: X, escalate: end}}},"
        " deploy: +{monitor: X}}"
    ),
    # Service
    "barista": "&{order: +{prepare: &{serve: +{payment: end}}}}",
    "waiter": (
        "rec X . &{seat: +{menu: &{order: +{serve:"
        " &{check: +{bill: end, dessert: X}}}}}}"
    ),
    "hairdresser": (
        "&{consult: +{wash: &{cut: +{style:"
        " &{approve: end, redo: +{style: end}}}}}}"
    ),
    # Creative
    "musician": (
        "rec X . &{rehearse: +{perform: &{feedback:"
        " +{improve: X, ready: end}}}}"
    ),
    "writer": (
        "rec X . &{research: +{draft: &{edit: +{revise: X, publish: end}}}}"
    ),
    # Legal
    "lawyer": (
        "rec X . &{consult: +{advise: &{proceed:"
        " +{represent: &{verdict: end}}, settle: end}}}"
    ),
    "judge": (
        "&{hear: +{deliberate: &{rule:"
        " +{sentence: end, acquit: end, appeal: end}}}}"
    ),
    # Finance
    "banker": (
        "&{apply: +{review: &{approve: +{disburse: end},"
        " deny: +{appeal: end}}}}"
    ),
    "trader": "rec X . &{analyze: +{buy: X, sell: X, hold: X, close: end}}",
    # Emergency
    "firefighter": (
        "rec X . &{dispatch: +{respond: &{assess:"
        " +{extinguish: X, rescue: +{evacuate: end}}}}}"
    ),
    "paramedic": (
        "&{respond: +{triage: &{treat: +{transport: end, release: end}}}}"
    ),
}

_PROFESSION_METADATA: dict[str, dict[str, object]] = {
    # Healthcare
    "doctor": {
        "category": "healthcare",
        "interaction_pattern": "consultation",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    "nurse": {
        "category": "healthcare",
        "interaction_pattern": "consultation",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    "pharmacist": {
        "category": "healthcare",
        "interaction_pattern": "transaction",
        "is_client_facing": True,
        "recursion_type": "none",
    },
    # Education
    "teacher": {
        "category": "education",
        "interaction_pattern": "instruction",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    "professor": {
        "category": "education",
        "interaction_pattern": "instruction",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    # Technology
    "developer": {
        "category": "technology",
        "interaction_pattern": "collaboration",
        "is_client_facing": False,
        "recursion_type": "iterative",
    },
    "sysadmin": {
        "category": "technology",
        "interaction_pattern": "collaboration",
        "is_client_facing": False,
        "recursion_type": "iterative",
    },
    # Service
    "barista": {
        "category": "service",
        "interaction_pattern": "transaction",
        "is_client_facing": True,
        "recursion_type": "none",
    },
    "waiter": {
        "category": "service",
        "interaction_pattern": "transaction",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    "hairdresser": {
        "category": "service",
        "interaction_pattern": "consultation",
        "is_client_facing": True,
        "recursion_type": "none",
    },
    # Creative
    "musician": {
        "category": "creative",
        "interaction_pattern": "performance",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    "writer": {
        "category": "creative",
        "interaction_pattern": "collaboration",
        "is_client_facing": False,
        "recursion_type": "iterative",
    },
    # Legal
    "lawyer": {
        "category": "legal",
        "interaction_pattern": "consultation",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    "judge": {
        "category": "legal",
        "interaction_pattern": "negotiation",
        "is_client_facing": True,
        "recursion_type": "none",
    },
    # Finance
    "banker": {
        "category": "finance",
        "interaction_pattern": "transaction",
        "is_client_facing": True,
        "recursion_type": "none",
    },
    "trader": {
        "category": "finance",
        "interaction_pattern": "negotiation",
        "is_client_facing": False,
        "recursion_type": "iterative",
    },
    # Emergency
    "firefighter": {
        "category": "emergency",
        "interaction_pattern": "collaboration",
        "is_client_facing": True,
        "recursion_type": "iterative",
    },
    "paramedic": {
        "category": "emergency",
        "interaction_pattern": "consultation",
        "is_client_facing": True,
        "recursion_type": "none",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_top_level_methods(s: SessionType) -> set[str]:
    """Extract top-level method names from a session type."""
    if isinstance(s, (Branch, Select)):
        return {name for name, _ in s.choices}
    if isinstance(s, Rec):
        return _get_top_level_methods(s.body)
    return set()


def _compute_binding_energy(ss: StateSpace) -> float:
    """Compute binding energy: fraction of states in non-trivial SCCs.

    Non-trivial SCCs (size >= 2) represent cyclic commitment — states
    that form loops in the protocol (from recursive types).  A higher
    binding energy means the profession's protocol has more cyclic,
    committed interaction patterns.
    """
    if not ss.states:
        return 0.0

    # Build adjacency
    adj: dict[int, list[int]] = {s: [] for s in ss.states}
    for src, _, tgt in ss.transitions:
        adj[src].append(tgt)

    # Iterative Tarjan's SCC
    index_counter = [0]
    stack: list[int] = []
    on_stack: set[int] = set()
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    sccs: list[set[int]] = []

    def strongconnect(v: int) -> None:
        work_stack: list[tuple[int, int, bool]] = [(v, 0, False)]
        while work_stack:
            node, child_idx, returning = work_stack.pop()
            if not returning:
                index[node] = index_counter[0]
                lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)
            children = adj.get(node, [])
            pushed = False
            while child_idx < len(children):
                w = children[child_idx]
                child_idx += 1
                if w not in index:
                    work_stack.append((node, child_idx, True))
                    work_stack.append((w, 0, False))
                    pushed = True
                    break
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], index[w])
            if pushed:
                continue
            if returning or not pushed:
                if lowlink[node] == index[node]:
                    scc: set[int] = set()
                    while True:
                        w2 = stack.pop()
                        on_stack.discard(w2)
                        scc.add(w2)
                        if w2 == node:
                            break
                    sccs.append(scc)
                # Update parent lowlink
                if work_stack:
                    parent_node = work_stack[-1][0]
                    lowlink[parent_node] = min(
                        lowlink[parent_node], lowlink[node]
                    )

    for s in ss.states:
        if s not in index:
            strongconnect(s)

    nontrivial = sum(len(scc) for scc in sccs if len(scc) >= 2)
    return round(nontrivial / len(ss.states), 4)


def _classify_complexity(
    state_count: int, transition_count: int, has_recursion: bool
) -> str:
    """Classify a profession's protocol complexity."""
    if state_count <= 5 and not has_recursion:
        return "simple"
    elif state_count <= 10:
        return "moderate"
    elif state_count <= 20:
        return "complex"
    else:
        return "highly_complex"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def get_profession(name: str) -> ProfessionProfile:
    """Look up a profession from the library and create its profile.

    Args:
        name: Profession name (case-insensitive, lowered internally).

    Returns:
        ProfessionProfile for the named profession.

    Raises:
        KeyError: If the profession is not in the library.
    """
    key = name.lower()
    if key not in PROFESSION_LIBRARY:
        raise KeyError(f"Unknown profession: {name!r}")

    meta = _PROFESSION_METADATA[key]
    return ProfessionProfile(
        name=key,
        category=str(meta["category"]),
        session_type_str=PROFESSION_LIBRARY[key],
        interaction_pattern=str(meta["interaction_pattern"]),
        is_client_facing=bool(meta["is_client_facing"]),
        recursion_type=str(meta["recursion_type"]),
    )


def analyze_profession(name: str) -> ProfessionAnalysis:
    """Parse a profession's session type and perform full structural analysis.

    Builds the state space, checks lattice properties, pendularity,
    branching entropy, and binding energy (cyclic commitment).

    Args:
        name: Profession name.

    Returns:
        ProfessionAnalysis with all computed metrics.

    Raises:
        KeyError: If the profession is not in the library.
    """
    profile = get_profession(name)
    ast = parse(profile.session_type_str)
    ss = build_statespace(ast)

    lattice_result = check_lattice(ss)
    pendular = is_pendular(ss)
    entropy = branching_entropy(ss)
    binding = _compute_binding_energy(ss)
    has_recursion = profile.recursion_type != "none"
    complexity = _classify_complexity(
        len(ss.states), len(ss.transitions), has_recursion
    )

    return ProfessionAnalysis(
        profile=profile,
        state_count=len(ss.states),
        transition_count=len(ss.transitions),
        is_lattice=lattice_result.is_lattice,
        is_pendular=pendular,
        branching_entropy=round(entropy, 4),
        binding_energy=binding,
        complexity_class=complexity,
    )


def compare_professions(name_a: str, name_b: str) -> ProfessionComparison:
    """Compare two professions using their session types.

    Computes compatibility (Jaccard similarity of top-level methods),
    extracts shared and unique methods, and checks subtyping.

    Args:
        name_a: First profession name.
        name_b: Second profession name.

    Returns:
        ProfessionComparison with compatibility and subsumption data.

    Raises:
        KeyError: If either profession is not in the library.
    """
    profile_a = get_profession(name_a)
    profile_b = get_profession(name_b)

    ast_a = parse(profile_a.session_type_str)
    ast_b = parse(profile_b.session_type_str)

    score = compatibility_score(ast_a, ast_b)

    methods_a = _get_top_level_methods(ast_a)
    methods_b = _get_top_level_methods(ast_b)

    shared = methods_a & methods_b
    unique_a = methods_a - methods_b
    unique_b = methods_b - methods_a

    a_sub_b = is_subtype(ast_a, ast_b)
    b_sub_a = is_subtype(ast_b, ast_a)

    if a_sub_b and b_sub_a:
        subsumption = "equivalent"
    elif a_sub_b:
        subsumption = "a_subsumes_b"
    elif b_sub_a:
        subsumption = "b_subsumes_a"
    else:
        subsumption = "incomparable"

    return ProfessionComparison(
        profession_a=name_a.lower(),
        profession_b=name_b.lower(),
        compatibility_score=round(score, 4),
        shared_interactions=tuple(sorted(shared)),
        unique_to_a=tuple(sorted(unique_a)),
        unique_to_b=tuple(sorted(unique_b)),
        subsumption=subsumption,
    )


def professions_by_category(category: str) -> list[str]:
    """List all professions in a given category.

    Args:
        category: Category name (e.g. "healthcare", "education").

    Returns:
        Sorted list of profession names in that category.
    """
    cat = category.lower()
    return sorted(
        name
        for name, meta in _PROFESSION_METADATA.items()
        if meta["category"] == cat
    )


def most_complex_profession() -> str:
    """Return the profession with the most states in its state space.

    Returns:
        Name of the most complex profession.
    """
    best_name = ""
    best_count = -1
    for name in PROFESSION_LIBRARY:
        ast = parse(PROFESSION_LIBRARY[name])
        ss = build_statespace(ast)
        if len(ss.states) > best_count:
            best_count = len(ss.states)
            best_name = name
    return best_name


def most_compatible_pair() -> tuple[str, str, float]:
    """Find the two professions with the highest compatibility score.

    Compares all pairs of distinct professions and returns the pair
    with the highest Jaccard similarity of top-level methods.

    Returns:
        (profession_a, profession_b, score) tuple.
    """
    names = sorted(PROFESSION_LIBRARY.keys())
    best: tuple[str, str, float] = ("", "", 0.0)

    # Pre-parse all types
    parsed: dict[str, SessionType] = {}
    for name in names:
        parsed[name] = parse(PROFESSION_LIBRARY[name])

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            score = compatibility_score(parsed[a], parsed[b])
            if score > best[2]:
                best = (a, b, round(score, 4))

    return best


def profession_lattice() -> dict[str, list[str]]:
    """Build a subsumption map: for each profession, list those it subsumes.

    Profession A subsumes profession B if A's session type is a subtype
    of B's (A offers at least all methods B offers, Gay-Hole subtyping).

    Returns:
        Dict mapping profession name to list of professions it subsumes.
    """
    names = sorted(PROFESSION_LIBRARY.keys())

    # Pre-parse all types
    parsed: dict[str, SessionType] = {}
    for name in names:
        parsed[name] = parse(PROFESSION_LIBRARY[name])

    result: dict[str, list[str]] = {}
    for a in names:
        subsumes: list[str] = []
        for b in names:
            if a != b and is_subtype(parsed[a], parsed[b]):
                subsumes.append(b)
        result[a] = sorted(subsumes)

    return result
