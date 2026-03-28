"""Session Type Game of Life: an evolutionary simulation (Step 59n).

Organisms are session types living in a world.  Each generation:

1. **Interact** — neighbours check compatibility:
   - Dual types → predator/prey encounter (one gains energy, other loses)
   - Subtype relation → imitation (resources shared)
   - Incompatible → competition (both lose energy)

2. **Metabolize** — each organism spends energy proportional to its complexity
   (number of states).  High-fitness types are more efficient.

3. **Reproduce** — organisms with enough energy spawn mutant offspring into
   empty adjacent cells.

4. **Die** — organisms with zero energy are removed (extinction).

The world is a toroidal grid where each cell is empty or contains one
organism.  The simulation runs until equilibrium, total extinction, or
a generation limit.

Key novelty: the organisms' *behaviour* (session type) determines their
fitness and interactions — not arbitrary rules.  Predation, mutualism,
parasitism, and speciation all emerge from the type-theoretic structure.

This module provides:
    ``Organism`` — a session type with energy, age, and lineage.
    ``World`` — a toroidal grid of organisms.
    ``tick(world)`` — advance one generation.
    ``run(world, generations)`` — simulate multiple generations.
    ``render(world)`` — ASCII visualization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

from reticulate.parser import (
    Branch,
    End,
    Select,
    SessionType,
    pretty,
)
from reticulate.populational import (
    MutationKind,
    fitness,
    is_extinct,
    is_speciation,
    mutate,
)
from reticulate.subtyping import is_subtype
from reticulate.duality import dual


# ---------------------------------------------------------------------------
# Organism
# ---------------------------------------------------------------------------


@dataclass
class Organism:
    """A living session type.

    Attributes:
        session_type: The organism's behavioral protocol.
        energy: Current energy level.  Dies at 0.
        age: Number of generations survived.
        species_id: Identifier for lineage tracking.
        generation_born: Which generation this organism was created.
    """
    session_type: SessionType
    energy: int
    age: int = 0
    species_id: int = 0
    generation_born: int = 0

    @property
    def is_alive(self) -> bool:
        return self.energy > 0 and not is_extinct(self.session_type)

    @property
    def type_str(self) -> str:
        return pretty(self.session_type)

    def __repr__(self) -> str:
        return f"Organism(energy={self.energy}, age={self.age}, type={self.type_str[:30]})"


# ---------------------------------------------------------------------------
# Interaction outcomes
# ---------------------------------------------------------------------------


class Interaction:
    """Types of interaction between two organisms."""
    PREDATION = "predation"       # Dual types: predator gains, prey loses
    MUTUALISM = "mutualism"       # Subtype both ways (isomorphic): both gain
    PARASITISM = "parasitism"     # One-way subtype: parasite gains, host loses
    COMPETITION = "competition"   # Incompatible: both lose


def classify_interaction(a: SessionType, b: SessionType) -> str:
    """Classify the interaction between two session types.

    - If a and b are mutual subtypes → mutualism (cooperation)
    - If a <: b (but not b <: a) → parasitism (a exploits b's niche)
    - If b <: a (but not a <: b) → parasitism (b exploits a's niche)
    - If dual(a) has same top-level polarity swap as b → predation
    - Otherwise → competition
    """
    a_sub_b = is_subtype(a, b)
    b_sub_a = is_subtype(b, a)

    if a_sub_b and b_sub_a:
        return Interaction.MUTUALISM
    if a_sub_b or b_sub_a:
        return Interaction.PARASITISM

    # Check for predator/prey (dual polarity)
    if _is_dual_polarity(a, b):
        return Interaction.PREDATION

    return Interaction.COMPETITION


def _is_dual_polarity(a: SessionType, b: SessionType) -> bool:
    """Check if a and b have opposite top-level polarity (one Select, one Branch)."""
    a_is_select = isinstance(a, Select)
    a_is_branch = isinstance(a, Branch)
    b_is_select = isinstance(b, Select)
    b_is_branch = isinstance(b, Branch)
    return (a_is_select and b_is_branch) or (a_is_branch and b_is_select)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------


@dataclass
class WorldConfig:
    """Configuration for the simulation.

    Attributes:
        width: Grid width.
        height: Grid height.
        metabolism_cost: Energy cost per generation per state in the organism.
        interaction_gain: Energy gained from favorable interaction.
        interaction_loss: Energy lost from unfavorable interaction.
        reproduction_threshold: Minimum energy to reproduce.
        reproduction_cost: Energy spent to reproduce.
        offspring_energy: Initial energy of offspring.
        mutation_rate: Probability of mutation during reproduction.
        labels: Pool of labels for mutations.
        seed: Random seed for reproducibility.
    """
    width: int = 10
    height: int = 10
    metabolism_cost: float = 0.5
    interaction_gain: int = 3
    interaction_loss: int = 2
    reproduction_threshold: int = 8
    reproduction_cost: int = 4
    offspring_energy: int = 5
    mutation_rate: float = 0.3
    labels: tuple[str, ...] = ("a", "b", "c", "d", "e")
    seed: int | None = None


@dataclass
class WorldStats:
    """Statistics for one generation.

    Attributes:
        generation: The generation number.
        population: Number of living organisms.
        max_energy: Maximum energy in the population.
        avg_energy: Average energy in the population.
        avg_fitness: Average fitness (gravitational potential).
        species_count: Number of distinct species.
        births: Number of births this generation.
        deaths: Number of deaths this generation.
        predations: Number of predation events.
        mutualisms: Number of mutualism events.
        competitions: Number of competition events.
    """
    generation: int
    population: int
    max_energy: int
    avg_energy: float
    avg_fitness: float
    species_count: int
    births: int
    deaths: int
    predations: int
    mutualisms: int
    competitions: int


@dataclass
class World:
    """A toroidal grid of organisms.

    The grid wraps around: (x + width) % width, (y + height) % height.
    """
    config: WorldConfig
    grid: dict[tuple[int, int], Organism] = field(default_factory=dict)
    generation: int = 0
    history: list[WorldStats] = field(default_factory=list)
    _next_species_id: int = 1
    _rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            self._rng = random.Random(self.config.seed)

    @property
    def population(self) -> int:
        return sum(1 for o in self.grid.values() if o.is_alive)

    @property
    def organisms(self) -> list[Organism]:
        return [o for o in self.grid.values() if o.is_alive]

    def place(self, x: int, y: int, session_type: SessionType, energy: int = 5) -> Organism:
        """Place a new organism at (x, y)."""
        org = Organism(
            session_type=session_type,
            energy=energy,
            species_id=self._next_species_id,
            generation_born=self.generation,
        )
        self._next_species_id += 1
        self.grid[(x % self.config.width, y % self.config.height)] = org
        return org

    def get(self, x: int, y: int) -> Organism | None:
        """Get organism at (x, y), or None if empty."""
        pos = (x % self.config.width, y % self.config.height)
        org = self.grid.get(pos)
        if org is not None and org.is_alive:
            return org
        return None

    def neighbours(self, x: int, y: int) -> list[tuple[int, int, Organism]]:
        """Get living neighbours (Von Neumann neighbourhood: 4 adjacent cells)."""
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (x + dx) % self.config.width
            ny = (y + dy) % self.config.height
            org = self.get(nx, ny)
            if org is not None:
                result.append((nx, ny, org))
        return result

    def empty_neighbours(self, x: int, y: int) -> list[tuple[int, int]]:
        """Get empty adjacent cells."""
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (x + dx) % self.config.width
            ny = (y + dy) % self.config.height
            if self.get(nx, ny) is None:
                result.append((nx, ny))
        return result


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def tick(world: World) -> WorldStats:
    """Advance the world by one generation.

    Phase 1: Interactions (neighbours affect each other's energy).
    Phase 2: Metabolism (energy cost proportional to complexity).
    Phase 3: Reproduction (fit organisms spawn mutant offspring).
    Phase 4: Death (zero-energy organisms removed).
    """
    cfg = world.config
    births = 0
    deaths = 0
    predations = 0
    mutualisms = 0
    competitions = 0

    # Phase 1: Interactions
    processed: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for (x, y), org in list(world.grid.items()):
        if not org.is_alive:
            continue
        for nx, ny, neighbour in world.neighbours(x, y):
            pair = (min((x, y), (nx, ny)), max((x, y), (nx, ny)))
            if pair in processed:
                continue
            processed.add(pair)

            interaction = classify_interaction(org.session_type, neighbour.session_type)

            if interaction == Interaction.PREDATION:
                predations += 1
                # Predator = higher fitness; prey = lower
                f_org = fitness(org.session_type)
                f_nei = fitness(neighbour.session_type)
                if f_org >= f_nei:
                    org.energy += cfg.interaction_gain
                    neighbour.energy -= cfg.interaction_loss
                else:
                    neighbour.energy += cfg.interaction_gain
                    org.energy -= cfg.interaction_loss

            elif interaction == Interaction.MUTUALISM:
                mutualisms += 1
                org.energy += cfg.interaction_gain // 2
                neighbour.energy += cfg.interaction_gain // 2

            elif interaction == Interaction.PARASITISM:
                # Parasite = the subtype (more capable)
                if is_subtype(org.session_type, neighbour.session_type):
                    org.energy += cfg.interaction_gain
                    neighbour.energy -= cfg.interaction_loss // 2
                else:
                    neighbour.energy += cfg.interaction_gain
                    org.energy -= cfg.interaction_loss // 2

            else:  # Competition
                competitions += 1
                org.energy -= cfg.interaction_loss // 2
                neighbour.energy -= cfg.interaction_loss // 2

    # Phase 2: Metabolism
    for org in world.grid.values():
        if org.is_alive:
            try:
                from reticulate.statespace import build_statespace
                ss = build_statespace(org.session_type)
                cost = max(1, int(len(ss.states) * cfg.metabolism_cost))
            except Exception:
                cost = 1
            org.energy -= cost
            org.age += 1

    # Phase 3: Reproduction
    new_organisms: list[tuple[int, int, Organism]] = []
    for (x, y), org in list(world.grid.items()):
        if not org.is_alive:
            continue
        if org.energy >= cfg.reproduction_threshold:
            empties = world.empty_neighbours(x, y)
            if empties:
                target = world._rng.choice(empties)
                org.energy -= cfg.reproduction_cost

                # Mutate offspring
                child_type = org.session_type
                if world._rng.random() < cfg.mutation_rate:
                    kind = world._rng.choice([
                        MutationKind.ADD_BRANCH,
                        MutationKind.REMOVE_BRANCH,
                        MutationKind.SWAP_POLARITY,
                        MutationKind.WIDEN_BRANCH,
                    ])
                    kwargs: dict[str, str] = {}
                    if kind == MutationKind.ADD_BRANCH:
                        kwargs["label"] = world._rng.choice(cfg.labels)
                    elif kind == MutationKind.REMOVE_BRANCH:
                        if isinstance(child_type, (Branch, Select)) and child_type.choices:
                            kwargs["label"] = world._rng.choice(child_type.choices)[0]
                    elif kind == MutationKind.WIDEN_BRANCH:
                        if isinstance(child_type, (Branch, Select)) and child_type.choices:
                            kwargs["source_label"] = world._rng.choice(child_type.choices)[0]
                            kwargs["new_label"] = world._rng.choice(cfg.labels)
                    child_type, _ = mutate(child_type, kind, **kwargs)

                child_species = org.species_id
                if is_speciation(org.session_type, child_type):
                    child_species = world._next_species_id
                    world._next_species_id += 1

                child = Organism(
                    session_type=child_type,
                    energy=cfg.offspring_energy,
                    species_id=child_species,
                    generation_born=world.generation + 1,
                )
                new_organisms.append((target[0], target[1], child))
                births += 1

    for x, y, child in new_organisms:
        if world.get(x, y) is None:
            world.grid[(x, y)] = child

    # Phase 4: Death
    to_remove = []
    for pos, org in world.grid.items():
        if not org.is_alive:
            to_remove.append(pos)
            deaths += 1
    for pos in to_remove:
        del world.grid[pos]

    world.generation += 1

    # Compute stats
    living = world.organisms
    pop = len(living)
    max_e = max((o.energy for o in living), default=0)
    avg_e = sum(o.energy for o in living) / pop if pop else 0.0
    fitnesses = [fitness(o.session_type) for o in living]
    avg_f = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    species = len({o.species_id for o in living})

    stats = WorldStats(
        generation=world.generation,
        population=pop,
        max_energy=max_e,
        avg_energy=round(avg_e, 2),
        avg_fitness=round(avg_f, 2),
        species_count=species,
        births=births,
        deaths=deaths,
        predations=predations,
        mutualisms=mutualisms,
        competitions=competitions,
    )
    world.history.append(stats)
    return stats


def run(world: World, generations: int = 10) -> list[WorldStats]:
    """Run the simulation for multiple generations.

    Stops early if the population goes extinct.
    """
    results = []
    for _ in range(generations):
        stats = tick(world)
        results.append(stats)
        if stats.population == 0:
            break
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def render(world: World) -> str:
    """Render the world as an ASCII grid.

    Legend:
        . = empty cell
        S = Select organism (predator-like)
        B = Branch organism (prey-like)
        E = End/extinct organism
        * = other (recursive, parallel, etc.)

    Brightness (uppercase/lowercase) indicates energy:
        uppercase = energy >= 5
        lowercase = energy < 5
    """
    lines = [f"Generation {world.generation}  Pop: {world.population}"]
    lines.append("+" + "-" * world.config.width + "+")

    for y in range(world.config.height):
        row = "|"
        for x in range(world.config.width):
            org = world.get(x, y)
            if org is None:
                row += "."
            elif isinstance(org.session_type, Select):
                row += "S" if org.energy >= 5 else "s"
            elif isinstance(org.session_type, Branch):
                row += "B" if org.energy >= 5 else "b"
            elif isinstance(org.session_type, End):
                row += "E"
            else:
                row += "*" if org.energy >= 5 else "o"
        row += "|"
        lines.append(row)

    lines.append("+" + "-" * world.config.width + "+")
    return "\n".join(lines)


def render_stats(world: World) -> str:
    """Render population statistics as a text summary."""
    if not world.history:
        return "No history yet."

    lines = ["Gen | Pop | MaxE | AvgE | AvgF | Spp | Birth | Death | Pred | Mut | Comp"]
    lines.append("-" * 80)
    for s in world.history:
        lines.append(
            f"{s.generation:3d} | {s.population:3d} | {s.max_energy:4d} | "
            f"{s.avg_energy:5.1f} | {s.avg_fitness:4.1f} | {s.species_count:3d} | "
            f"{s.births:5d} | {s.deaths:5d} | {s.predations:4d} | "
            f"{s.mutualisms:3d} | {s.competitions:4d}"
        )
    return "\n".join(lines)
