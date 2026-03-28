"""Mathematics as Self-Referential Session Types (Step 208).

Mathematics is what happens when you apply the recursion mechanism to the
mechanisms themselves.  Proofs, constructions, abstractions — all are
session types.  Goedel's incompleteness IS a session type that cannot be
typed by itself.

Every mathematical process is a protocol: a proof proceeds through
premises to conclusion, an algebraic computation composes elements
according to axioms, and a Turing machine reads/writes/moves along a
tape.  This module encodes these as session types and provides tools to
compose proofs, detect self-reference, and verify lattice structure.

This module provides:
    ``get_math(name)``               -- look up a math entry by name.
    ``math_by_domain(domain)``       -- find entries in a domain.
    ``proof_type(steps)``            -- compose proof steps sequentially.
    ``is_self_referential(name)``    -- check if entry uses recursion.
    ``all_math_form_lattices()``     -- verify every entry forms a lattice.
    ``godel_check(type_str)``        -- simplified self-reference detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.parser import Branch, End, Rec, Select, SessionType, Var, parse, pretty
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MathEntry:
    """A single entry in the mathematics session type library.

    Attributes:
        name: Unique identifier for this mathematical concept.
        domain: Mathematics domain (logic, algebra, analysis, topology,
                computation, foundations).
        session_type_str: Session type in textual syntax.
        description: Short human-readable description.
    """

    name: str
    domain: str
    session_type_str: str
    description: str


# ---------------------------------------------------------------------------
# The Mathematics Library
# ---------------------------------------------------------------------------

MATH_LIBRARY: dict[str, MathEntry] = {
    # -- Logic --
    "modus_ponens": MathEntry(
        "modus_ponens", "logic",
        "&{premise_p: &{premise_p_implies_q: +{conclude_q: end}}}",
        "Modus ponens: from P and P->Q derive Q",
    ),
    "proof_by_contradiction": MathEntry(
        "proof_by_contradiction", "logic",
        "&{assume_not_p: +{derive_contradiction: &{conclude_p: end}}}",
        "Proof by contradiction: assume negation, derive contradiction",
    ),
    "induction": MathEntry(
        "induction", "logic",
        "rec X . &{base_case: +{inductive_step: X, qed: end}}",
        "Mathematical induction: base case then inductive steps",
    ),
    "reductio": MathEntry(
        "reductio", "logic",
        "&{assume: +{derive: &{contradict: +{negate: end}}}}",
        "Reductio ad absurdum: assume, derive, contradict, negate",
    ),

    # -- Algebra --
    "group_operation": MathEntry(
        "group_operation", "algebra",
        "rec X . &{element_a: &{element_b: +{compose: X, identity: end}}}",
        "Group operation: compose elements or reach identity",
    ),
    "ring_arithmetic": MathEntry(
        "ring_arithmetic", "algebra",
        "&{add: +{result: &{multiply: +{result: end}}}}",
        "Ring arithmetic: addition followed by multiplication",
    ),
    "field_division": MathEntry(
        "field_division", "algebra",
        "&{numerator: &{denominator: +{nonzero: &{divide: +{quotient: end}}, zero: end}}}",
        "Field division: division guarded by nonzero denominator check",
    ),
    "galois_symmetry": MathEntry(
        "galois_symmetry", "algebra",
        "rec X . &{permute: +{fixed_field: X, resolve: end}}",
        "Galois symmetry: permute until field is resolved",
    ),

    # -- Analysis --
    "limit": MathEntry(
        "limit", "analysis",
        "rec X . &{epsilon: +{delta: &{approach: +{converge: end, refine: X}}}}",
        "Limit: epsilon-delta refinement until convergence",
    ),
    "derivative": MathEntry(
        "derivative", "analysis",
        "&{function: +{perturb: &{ratio: +{limit: end}}}}",
        "Derivative: perturb function, take ratio, pass to limit",
    ),
    "integral": MathEntry(
        "integral", "analysis",
        "rec X . &{partition: +{sum: &{refine: +{converge: end, subdivide: X}}}}",
        "Integral: partition, sum, refine until convergence",
    ),

    # -- Topology --
    "continuity": MathEntry(
        "continuity", "topology",
        "&{open_set: +{preimage: &{still_open: end}}}",
        "Continuity: preimage of open set is open",
    ),
    "compactness": MathEntry(
        "compactness", "topology",
        "&{open_cover: +{finite_subcover: end}}",
        "Compactness: every open cover has a finite subcover",
    ),
    "connectedness": MathEntry(
        "connectedness", "topology",
        "&{partition: +{impossible: end}}",
        "Connectedness: no nontrivial partition into open sets",
    ),

    # -- Computation --
    "turing_machine": MathEntry(
        "turing_machine", "computation",
        "rec X . &{read: +{write: &{move: +{halt: end, continue: X}}}}",
        "Turing machine: read, write, move, halt or continue",
    ),
    "lambda_calculus": MathEntry(
        "lambda_calculus", "computation",
        "rec X . &{abstract: +{apply: X, reduce: end}}",
        "Lambda calculus: abstract then apply or reduce",
    ),
    "halting_problem": MathEntry(
        "halting_problem", "computation",
        "&{simulate: +{halts: end, undecidable: end}}",
        "Halting problem: simulation yields halts or undecidable",
    ),

    # -- Foundations --
    "axiom_of_choice": MathEntry(
        "axiom_of_choice", "foundations",
        "&{family: +{choose: &{function: end}}}",
        "Axiom of choice: given a family of sets, choose one element each",
    ),
    "godel_incompleteness": MathEntry(
        "godel_incompleteness", "foundations",
        "&{system: +{consistent: &{incomplete: +{true_unprovable: end}}, inconsistent: end}}",
        "Goedel incompleteness: consistent systems are incomplete",
    ),
    "zfc_set": MathEntry(
        "zfc_set", "foundations",
        "rec X . &{element: +{member: X, empty: end}}",
        "ZFC set: elements are members of sets, recursively",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_math(name: str) -> MathEntry:
    """Look up a mathematics entry by name.

    Raises:
        KeyError: If *name* is not in the library.
    """
    if name not in MATH_LIBRARY:
        raise KeyError(f"Unknown math entry: {name!r}")
    return MATH_LIBRARY[name]


def math_by_domain(domain: str) -> list[MathEntry]:
    """Return all entries belonging to *domain*."""
    return [e for e in MATH_LIBRARY.values() if e.domain == domain]


def proof_type(steps: list[str]) -> str:
    """Compose a proof as sequential branch composition of named math entries.

    Given a list of entry names, builds a session type where each step's
    top-level methods are nested sequentially: the first step's methods
    lead into the second step's type, and so on.

    Returns the pretty-printed composed session type string.

    Raises:
        KeyError: If any step name is not in the library.
        ValueError: If the step list is empty.
    """
    if not steps:
        raise ValueError("Cannot compose an empty proof")

    asts: list[SessionType] = []
    for name in steps:
        entry = get_math(name)
        asts.append(parse(entry.session_type_str))

    # Build right-to-left nesting: last step is innermost
    result: SessionType = asts[-1]
    for ast in reversed(asts[:-1]):
        result = _nest_into(ast, result)

    return pretty(result)


def _nest_into(outer: SessionType, inner: SessionType) -> SessionType:
    """Replace every ``End`` leaf in *outer* with *inner*."""
    if isinstance(outer, End):
        return inner
    if isinstance(outer, Var):
        return outer
    if isinstance(outer, Branch):
        return Branch(tuple(
            (label, _nest_into(cont, inner)) for label, cont in outer.choices
        ))
    if isinstance(outer, Select):
        return Select(tuple(
            (label, _nest_into(cont, inner)) for label, cont in outer.choices
        ))
    if isinstance(outer, Rec):
        return Rec(outer.var, _nest_into(outer.body, inner))
    return outer


def is_self_referential(name: str) -> bool:
    """Return True if the named entry's session type uses recursion.

    A self-referential type is one that contains a ``Rec`` node,
    meaning it can refer to its own structure.
    """
    entry = get_math(name)
    ast = parse(entry.session_type_str)
    return _contains_rec(ast)


def _contains_rec(node: SessionType) -> bool:
    """Check whether *node* contains any ``Rec`` constructor."""
    if isinstance(node, Rec):
        return True
    if isinstance(node, (Branch, Select)):
        return any(_contains_rec(cont) for _, cont in node.choices)
    if isinstance(node, End | Var):
        return False
    return False


def all_math_form_lattices() -> bool:
    """Verify that every entry in the math library forms a lattice.

    Returns True if and only if every entry parses and its state space
    is a lattice.
    """
    for entry in MATH_LIBRARY.values():
        ast = parse(entry.session_type_str)
        ss = build_statespace(ast)
        result = check_lattice(ss)
        if not result.is_lattice:
            return False
    return True


def godel_check(type_str: str) -> bool:
    """Simplified self-reference detection for a session type string.

    Returns True if the type contains recursion where the recursive
    variable appears inside a branch that structurally resembles the
    full body — i.e. the type 'talks about itself'.

    Specifically, returns True when:
    1. The type is recursive (contains ``Rec``), AND
    2. The recursive variable appears within a ``Branch`` or ``Select``
       node that has at least as many choices as the outermost branch/select
       of the rec body (structural self-similarity).
    """
    ast = parse(type_str)
    return _check_godel(ast)


def _check_godel(node: SessionType) -> bool:
    """Recursive Goedel self-reference check."""
    if not isinstance(node, Rec):
        # Check sub-nodes for nested Rec
        if isinstance(node, (Branch, Select)):
            return any(_check_godel(cont) for _, cont in node.choices)
        return False

    var_name = node.var
    body = node.body
    body_width = _choice_width(body)

    if body_width == 0:
        return False

    return _var_in_similar_context(body, var_name, body_width)


def _choice_width(node: SessionType) -> int:
    """Return the number of choices at the top level, or 0 if not a choice."""
    if isinstance(node, (Branch, Select)):
        return len(node.choices)
    return 0


def _var_in_similar_context(
    node: SessionType, var: str, target_width: int
) -> bool:
    """Check if *var* appears in a branch/select with >= *target_width* choices."""
    if isinstance(node, Var):
        return False
    if isinstance(node, End):
        return False
    if isinstance(node, Rec):
        return _var_in_similar_context(node.body, var, target_width)
    if isinstance(node, (Branch, Select)):
        # Check if this node contains the var and has similar width
        width = len(node.choices)
        has_var = any(
            _directly_contains_var(cont, var) for _, cont in node.choices
        )
        if has_var and width >= target_width:
            return True
        # Recurse into children
        return any(
            _var_in_similar_context(cont, var, target_width)
            for _, cont in node.choices
        )
    return False


def _directly_contains_var(node: SessionType, var: str) -> bool:
    """Check if *node* contains a ``Var`` with the given name."""
    if isinstance(node, Var):
        return node.name == var
    if isinstance(node, End):
        return False
    if isinstance(node, Rec):
        if node.var == var:
            return False  # shadowed
        return _directly_contains_var(node.body, var)
    if isinstance(node, (Branch, Select)):
        return any(_directly_contains_var(cont, var) for _, cont in node.choices)
    return False
