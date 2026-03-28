"""Populational session types: mutation, evolution, and extinction (Step 59m).

Models population dynamics through session type operations:

**Mutation operators** — local AST perturbations that model biological mutations:
    - ``add_branch``: gain a new capability (gene duplication → new function)
    - ``remove_branch``: lose a capability (gene deletion)
    - ``swap_polarity``: role reversal (predator becomes prey)
    - ``deepen_branch``: add intermediate step (regulatory complexity)
    - ``widen_branch``: duplicate an existing branch with a new label

**Fitness** — gravitational potential of the resulting state space.
    Higher potential = more paths to survival = fitter organism.

**Evolution** — iterated mutation + selection:
    1. Start with a population of session types
    2. Mutate each type (random perturbation)
    3. Compute fitness (gravitational potential)
    4. Select fittest types (tournament selection)
    5. Repeat

**Extinction** — type reaches bottom or becomes degenerate:
    - ``is_extinct(s)`` — type has no non-trivial paths
    - ``extinction_risk(s)`` — 1 / gravitational_potential (inverse fitness)

**Speciation** — mutation breaks subtyping relationship:
    - ``is_speciation(s_original, s_mutant)`` — neither is subtype of the other

**Ecological interactions** — predator-prey, SIR, mutualism as session types.

This module provides:
    Mutation: ``mutate(ast, operator)`` — apply a mutation operator.
    Fitness: ``fitness(ast)`` — compute fitness score.
    Evolution: ``evolve(population, generations)`` — simulate evolution.
    Extinction: ``is_extinct(ast)``, ``extinction_risk(ast)``.
    Speciation: ``is_speciation(s1, s2)``.
    Ecology: ``predator_prey(n)``, ``sir_model()``, ``lotka_volterra(n)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto

from reticulate.parser import (
    Branch,
    End,
    Rec,
    Select,
    SessionType,
    Var,
    pretty,
)
from reticulate.gravitational import analyze_gravity, gravitational_potential
from reticulate.statespace import StateSpace, build_statespace
from reticulate.lattice import check_lattice
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------


class MutationKind(Enum):
    """Types of mutation that can be applied to a session type AST."""
    ADD_BRANCH = auto()
    REMOVE_BRANCH = auto()
    SWAP_POLARITY = auto()
    DEEPEN_BRANCH = auto()
    WIDEN_BRANCH = auto()


@dataclass(frozen=True)
class Mutation:
    """A record of a mutation applied to a session type.

    Attributes:
        kind: The type of mutation.
        original: The original session type string.
        mutant: The resulting session type string.
        site: Human-readable description of where the mutation occurred.
    """
    kind: MutationKind
    original: str
    mutant: str
    site: str


def add_branch(ast: SessionType, label: str = "new") -> SessionType:
    """Add a new branch leading to end at the outermost choice point.

    Models gene duplication producing a new capability.
    """
    if isinstance(ast, Branch):
        existing = dict(ast.choices)
        if label not in existing:
            new_choices = ast.choices + ((label, End()),)
            return Branch(new_choices)
        return ast
    if isinstance(ast, Select):
        existing = dict(ast.choices)
        if label not in existing:
            new_choices = ast.choices + ((label, End()),)
            return Select(new_choices)
        return ast
    if isinstance(ast, Rec):
        return Rec(ast.var, add_branch(ast.body, label))
    return ast


def remove_branch(ast: SessionType, label: str | None = None) -> SessionType:
    """Remove a branch from the outermost choice point.

    Models gene deletion / loss of capability. If label is None,
    removes the last branch. Returns End if only one branch remains.
    """
    if isinstance(ast, (Branch, Select)):
        choices = list(ast.choices)
        if len(choices) <= 1:
            return End()
        if label is not None:
            choices = [(l, s) for l, s in choices if l != label]
        else:
            choices = choices[:-1]
        if not choices:
            return End()
        cls = Branch if isinstance(ast, Branch) else Select
        return cls(tuple(choices))
    if isinstance(ast, Rec):
        return Rec(ast.var, remove_branch(ast.body, label))
    return ast


def swap_polarity(ast: SessionType) -> SessionType:
    """Swap Branch <-> Select at the outermost level.

    Models ecological role reversal (predator becomes prey).
    This is NOT full duality — it only swaps the top-level construct.
    """
    if isinstance(ast, Branch):
        return Select(ast.choices)
    if isinstance(ast, Select):
        return Branch(ast.choices)
    if isinstance(ast, Rec):
        return Rec(ast.var, swap_polarity(ast.body))
    return ast


def deepen_branch(ast: SessionType, label: str, wrapper_label: str = "step") -> SessionType:
    """Add an intermediate step before an existing branch.

    Models regulatory complexity increase: what was a direct action
    now requires a preliminary step.
    """
    if isinstance(ast, (Branch, Select)):
        new_choices = []
        for l, s in ast.choices:
            if l == label:
                # Wrap: label -> wrapper_label -> original_continuation
                intermediate = Branch(((wrapper_label, s),))
                new_choices.append((l, intermediate))
            else:
                new_choices.append((l, s))
        cls = Branch if isinstance(ast, Branch) else Select
        return cls(tuple(new_choices))
    if isinstance(ast, Rec):
        return Rec(ast.var, deepen_branch(ast.body, label, wrapper_label))
    return ast


def widen_branch(ast: SessionType, source_label: str, new_label: str) -> SessionType:
    """Duplicate a branch with a new label (same continuation).

    Models gene duplication where a duplicated gene develops a new name
    but initially has the same function.
    """
    if isinstance(ast, (Branch, Select)):
        existing = dict(ast.choices)
        if source_label in existing and new_label not in existing:
            new_choices = ast.choices + ((new_label, existing[source_label]),)
            cls = Branch if isinstance(ast, Branch) else Select
            return cls(tuple(new_choices))
        return ast
    if isinstance(ast, Rec):
        return Rec(ast.var, widen_branch(ast.body, source_label, new_label))
    return ast


def mutate(ast: SessionType, kind: MutationKind, **kwargs: str) -> tuple[SessionType, Mutation]:
    """Apply a named mutation to a session type.

    Returns (mutant_ast, mutation_record).
    """
    original_str = pretty(ast)

    if kind == MutationKind.ADD_BRANCH:
        label = kwargs.get("label", "new")
        result = add_branch(ast, label)
        site = f"added branch '{label}' at top level"
    elif kind == MutationKind.REMOVE_BRANCH:
        label = kwargs.get("label")
        result = remove_branch(ast, label)
        site = f"removed branch '{label or 'last'}' at top level"
    elif kind == MutationKind.SWAP_POLARITY:
        result = swap_polarity(ast)
        site = "swapped polarity at top level"
    elif kind == MutationKind.DEEPEN_BRANCH:
        label = kwargs.get("label", "")
        wrapper = kwargs.get("wrapper_label", "step")
        result = deepen_branch(ast, label, wrapper)
        site = f"deepened branch '{label}' with '{wrapper}'"
    elif kind == MutationKind.WIDEN_BRANCH:
        source = kwargs.get("source_label", "")
        new = kwargs.get("new_label", "copy")
        result = widen_branch(ast, source, new)
        site = f"widened '{source}' to '{new}'"
    else:
        raise ValueError(f"Unknown mutation kind: {kind}")

    return result, Mutation(
        kind=kind,
        original=original_str,
        mutant=pretty(result),
        site=site,
    )


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


def fitness(ast: SessionType) -> int:
    """Compute fitness of a session type.

    Fitness = gravitational potential at top (longest path to termination).
    Higher = more paths to survival = fitter.
    Returns 0 for degenerate types (end only).
    """
    try:
        ss = build_statespace(ast)
        pot = gravitational_potential(ss)
        return max(0, pot.get(ss.top, 0))
    except Exception:
        return 0


def fitness_detailed(ast: SessionType) -> dict[str, int | float | bool]:
    """Compute detailed fitness metrics.

    Returns dict with: potential, escape_velocity, force, is_lattice, states, transitions.
    """
    try:
        ss = build_statespace(ast)
        gf = analyze_gravity(ss)
        lr = check_lattice(ss)
        return {
            "potential": gf.max_potential,
            "escape_velocity": gf.escape_velocity.get(ss.top, -1),
            "force": gf.force.get(ss.top, 0),
            "is_lattice": lr.is_lattice,
            "states": len(ss.states),
            "transitions": len(ss.transitions),
            "total_energy": gf.total_energy,
            "orbits": len(gf.orbits),
            "is_lyapunov": gf.is_lyapunov,
        }
    except Exception:
        return {
            "potential": 0, "escape_velocity": 0, "force": 0,
            "is_lattice": False, "states": 0, "transitions": 0,
            "total_energy": 0, "orbits": 0, "is_lyapunov": True,
        }


# ---------------------------------------------------------------------------
# Extinction
# ---------------------------------------------------------------------------


def is_extinct(ast: SessionType) -> bool:
    """Check if a session type is "extinct" — degenerate with no real behavior.

    A type is extinct if it is End, or its state space has only 1 state.
    """
    if isinstance(ast, End):
        return True
    try:
        ss = build_statespace(ast)
        return len(ss.states) <= 1
    except Exception:
        return True


def extinction_risk(ast: SessionType) -> float:
    """Compute extinction risk: inverse of fitness.

    Returns a value in [0, 1]. 1.0 = already extinct. 0.0 = maximally fit.
    """
    f = fitness(ast)
    if f <= 0:
        return 1.0
    return round(1.0 / (1.0 + f), 3)


# ---------------------------------------------------------------------------
# Speciation
# ---------------------------------------------------------------------------


def is_speciation(s1: SessionType, s2: SessionType) -> bool:
    """Check if two types represent a speciation event.

    Speciation occurs when neither type is a subtype of the other —
    they have diverged to the point of incompatibility.
    """
    return not is_subtype(s1, s2) and not is_subtype(s2, s1)


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolutionResult:
    """Result of an evolutionary simulation.

    Attributes:
        generations: Number of generations simulated.
        initial_population: Starting session type strings.
        final_population: Surviving session type strings.
        fitness_history: List of (generation, max_fitness, avg_fitness).
        extinctions: Number of types that went extinct.
        speciations: Number of speciation events detected.
        mutations_applied: Total mutations applied.
    """
    generations: int
    initial_population: tuple[str, ...]
    final_population: tuple[str, ...]
    fitness_history: tuple[tuple[int, int, float], ...]
    extinctions: int
    speciations: int
    mutations_applied: int


def evolve(
    population: list[SessionType],
    generations: int = 10,
    mutation_rate: float = 0.5,
    labels: tuple[str, ...] = ("a", "b", "c", "d", "e"),
    seed: int | None = None,
) -> EvolutionResult:
    """Simulate evolution of a population of session types.

    Each generation:
    1. Compute fitness of each type.
    2. Remove extinct types.
    3. Mutate surviving types (with probability mutation_rate).
    4. Record fitness statistics.

    Args:
        population: Initial population of session type ASTs.
        generations: Number of generations to simulate.
        mutation_rate: Probability of mutation per type per generation.
        labels: Pool of labels for new branches.
        seed: Random seed for reproducibility.

    Returns:
        EvolutionResult with full history.
    """
    rng = random.Random(seed)
    initial_strs = tuple(pretty(t) for t in population)

    current = list(population)
    history: list[tuple[int, int, float]] = []
    total_extinctions = 0
    total_speciations = 0
    total_mutations = 0

    mutation_kinds = [
        MutationKind.ADD_BRANCH,
        MutationKind.REMOVE_BRANCH,
        MutationKind.SWAP_POLARITY,
        MutationKind.WIDEN_BRANCH,
    ]

    for gen in range(generations):
        # Compute fitness
        fitnesses = [fitness(t) for t in current]

        # Record stats
        max_f = max(fitnesses) if fitnesses else 0
        avg_f = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        history.append((gen, max_f, round(avg_f, 2)))

        # Remove extinct
        survivors = []
        for t, f in zip(current, fitnesses):
            if f > 0:
                survivors.append(t)
            else:
                total_extinctions += 1

        if not survivors:
            break

        # Mutate
        next_gen: list[SessionType] = []
        for t in survivors:
            if rng.random() < mutation_rate:
                kind = rng.choice(mutation_kinds)
                kwargs: dict[str, str] = {}
                if kind == MutationKind.ADD_BRANCH:
                    kwargs["label"] = rng.choice(labels)
                elif kind == MutationKind.REMOVE_BRANCH:
                    if isinstance(t, (Branch, Select)) and t.choices:
                        kwargs["label"] = rng.choice(t.choices)[0]
                elif kind == MutationKind.WIDEN_BRANCH:
                    if isinstance(t, (Branch, Select)) and t.choices:
                        kwargs["source_label"] = rng.choice(t.choices)[0]
                        kwargs["new_label"] = rng.choice(labels)

                mutant, _record = mutate(t, kind, **kwargs)
                total_mutations += 1

                # Check speciation
                if is_speciation(t, mutant):
                    total_speciations += 1

                next_gen.append(mutant)
            else:
                next_gen.append(t)

        current = next_gen

    final_strs = tuple(pretty(t) for t in current)

    return EvolutionResult(
        generations=len(history),
        initial_population=initial_strs,
        final_population=final_strs,
        fitness_history=tuple(history),
        extinctions=total_extinctions,
        speciations=total_speciations,
        mutations_applied=total_mutations,
    )


# ---------------------------------------------------------------------------
# Ecological type constructors
# ---------------------------------------------------------------------------


def predator_prey(n_interactions: int = 3) -> tuple[SessionType, SessionType]:
    """Construct predator and prey session types.

    Models n_interactions rounds of hunt/flee before the encounter ends.
    Predator uses Select (chooses action), Prey uses Branch (receives action).

    Returns (predator_type, prey_type).
    """
    # Build from inside out
    predator: SessionType = End()
    prey: SessionType = End()

    for _ in range(n_interactions):
        # Predator: +{hunt: &{SUCCESS: ..., FAIL: ...}}
        predator = Select((
            ("hunt", Branch((
                ("SUCCESS", predator),
                ("FAIL", End()),
            ))),
            ("rest", End()),
        ))
        # Prey: &{hunt: +{ESCAPE: ..., CAUGHT: end}}
        prey = Branch((
            ("hunt", Select((
                ("ESCAPE", prey),
                ("CAUGHT", End()),
            ))),
            ("rest", End()),
        ))

    return predator, prey


def sir_model() -> SessionType:
    """Construct the SIR epidemiological model as a session type.

    States: Susceptible -> Infected -> Recovered (-> end)
    At each state, the environment (Branch) determines the transition.
    """
    recovered: SessionType = End()
    infected: SessionType = Branch((
        ("recover", recovered),
        ("worsen", End()),  # death
    ))
    susceptible: SessionType = Branch((
        ("contact_infected", Select((
            ("INFECTED", infected),
            ("HEALTHY", End()),
        ))),
        ("no_contact", End()),
    ))
    return susceptible


def lotka_volterra(prey_pop: int = 3, pred_pop: int = 2) -> SessionType:
    """Construct a discrete Lotka-Volterra interaction as a session type.

    Models prey_pop prey individuals and pred_pop predator individuals
    as a sequence of encounter rounds. Each round: one predator hunts,
    one prey responds.

    Returns a session type where:
    - Select = predator action (hunt/rest)
    - Branch = prey response (escape/caught)
    """
    encounter: SessionType = End()

    total_rounds = min(prey_pop, pred_pop)
    for _ in range(total_rounds):
        encounter = Select((
            ("hunt", Branch((
                ("escape", encounter),
                ("caught", encounter),
            ))),
            ("rest", encounter),
        ))

    return encounter
