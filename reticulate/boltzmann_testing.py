"""Boltzmann vs Uniform Testing — Experimental Validation (Step V2).

**Hypothesis**: Test paths weighted by Boltzmann probability exp(-β·E)
find more protocol defects than uniformly weighted paths.

This is an EXPERIMENT, not a theorem. The module provides the experimental
framework: mutation generation, path selection strategies, kill detection,
and statistical comparison.

**Protocol mutations** simulate defects:
- Transition removal (missing error handler)
- Target redirect (wrong state transition)
- Label swap (confused method dispatch)

**Selection strategies** (all three implemented):
A. Path-energy Boltzmann: weight by exp(-β · Σ E(states))
B. Path-length Boltzmann: weight by exp(-β · path_length)
C. Entropy-weighted: weight by exp(β · Σ H(states))  [high entropy = interesting]

**The question**: does ANY physics-inspired weighting beat uniform?
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.testgen import enumerate_valid_paths, ValidPath, Step

if TYPE_CHECKING:
    from reticulate.statespace import StateSpace


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mutant:
    """A mutated protocol (simulated defect).

    Attributes:
        kind: Mutation type ("remove", "redirect", "swap").
        original_edge: The (src, label, tgt) that was mutated.
        description: Human-readable description.
    """
    kind: str
    original_edge: tuple[int, str, int]
    description: str


@dataclass(frozen=True)
class TrialResult:
    """Result of a single trial (one strategy, one budget).

    Attributes:
        strategy: Strategy name.
        beta: Temperature parameter (None for uniform).
        budget: Number of paths selected.
        kill_rate: Fraction of mutants killed.
        kills: Number of mutants killed.
        total_mutants: Total mutants.
        transitions_covered: Number of unique transitions exercised.
        total_transitions: Total transitions in original.
    """
    strategy: str
    beta: float | None
    budget: int
    kill_rate: float
    kills: int
    total_mutants: int
    transitions_covered: int
    total_transitions: int


@dataclass(frozen=True)
class ExperimentResult:
    """Result of comparing strategies across multiple trials.

    Attributes:
        protocol_name: Name of the protocol tested.
        num_states: Number of states.
        num_paths: Total valid paths enumerated.
        num_mutants: Number of mutants generated.
        trials: List of trial results (one per strategy × budget × trial).
    """
    protocol_name: str
    num_states: int
    num_paths: int
    num_mutants: int
    trials: list[TrialResult]


@dataclass(frozen=True)
class StrategyComparison:
    """Summary comparison between strategies.

    Attributes:
        strategy: Strategy name.
        beta: Temperature (None for uniform).
        mean_kill_rate: Mean kill rate across trials.
        std_kill_rate: Standard deviation.
        mean_coverage: Mean transition coverage.
        budget: Test budget.
    """
    strategy: str
    beta: float | None
    mean_kill_rate: float
    std_kill_rate: float
    mean_coverage: float
    budget: int


# ---------------------------------------------------------------------------
# Mutation generation
# ---------------------------------------------------------------------------

def generate_mutants(
    ss: StateSpace,
    seed: int = 42,
) -> list[Mutant]:
    """Generate all single-transition mutations.

    One mutant per transition: removal of that transition.
    This is exhaustive (not random) — every transition gets mutated once.
    """
    mutants: list[Mutant] = []
    for src, label, tgt in ss.transitions:
        mutants.append(Mutant(
            kind="remove",
            original_edge=(src, label, tgt),
            description=f"Remove ({src})-{label}->({tgt})",
        ))
    return mutants


def generate_redirect_mutants(
    ss: StateSpace,
    seed: int = 42,
) -> list[Mutant]:
    """Generate redirect mutations: change target of each transition."""
    rng = random.Random(seed)
    mutants: list[Mutant] = []
    states = sorted(ss.states)
    for src, label, tgt in ss.transitions:
        # Redirect to a random different state
        candidates = [s for s in states if s != tgt]
        if candidates:
            new_tgt = rng.choice(candidates)
            mutants.append(Mutant(
                kind="redirect",
                original_edge=(src, label, tgt),
                description=f"Redirect ({src})-{label}->({tgt}) to ({new_tgt})",
            ))
    return mutants


# ---------------------------------------------------------------------------
# Path energy computation
# ---------------------------------------------------------------------------

def _rank_energy(ss: StateSpace) -> dict[int, float]:
    """Rank energy: BFS distance from bottom."""
    rev: dict[int, set[int]] = {s: set() for s in ss.states}
    for src, _, tgt in ss.transitions:
        rev.setdefault(tgt, set()).add(src)
    dist: dict[int, int] = {}
    if ss.bottom in ss.states:
        dist[ss.bottom] = 0
        queue = [ss.bottom]
        while queue:
            s = queue.pop(0)
            for pred in rev.get(s, set()):
                if pred not in dist:
                    dist[pred] = dist[s] + 1
                    queue.append(pred)
    for s in ss.states:
        if s not in dist:
            dist[s] = 0
    return {s: float(dist[s]) for s in ss.states}


def _local_entropy(ss: StateSpace, state: int) -> float:
    """Shannon entropy at a state (bits)."""
    out = [tgt for src, _, tgt in ss.transitions if src == state]
    n = len(out)
    if n <= 1:
        return 0.0
    return math.log2(n)


def path_energy(ss: StateSpace, path: ValidPath) -> float:
    """Total energy of a path: Σ E(states visited)."""
    energy = _rank_energy(ss)
    total = energy.get(ss.top, 0.0)  # Start at top
    for step in path.steps:
        total += energy.get(step.target, 0.0)
    return total


def path_length(path: ValidPath) -> int:
    """Number of steps in a path."""
    return len(path.steps)


def path_entropy(ss: StateSpace, path: ValidPath) -> float:
    """Total entropy along a path: Σ H(states visited)."""
    total = _local_entropy(ss, ss.top)
    for step in path.steps:
        total += _local_entropy(ss, step.target)
    return total


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_uniform(
    paths: list[ValidPath],
    n: int,
    rng: random.Random,
) -> list[ValidPath]:
    """Strategy: uniform random selection."""
    if n >= len(paths):
        return list(paths)
    return rng.sample(paths, n)


def select_boltzmann_energy(
    ss: StateSpace,
    paths: list[ValidPath],
    n: int,
    beta: float,
    rng: random.Random,
) -> list[ValidPath]:
    """Strategy A: Boltzmann weighting by path energy.

    Low energy paths (close to termination) get higher weight at high β.
    """
    energies = [path_energy(ss, p) for p in paths]
    # Shift for numerical stability
    min_e = min(energies) if energies else 0
    weights = [math.exp(-beta * (e - min_e)) for e in energies]
    return _weighted_sample(paths, weights, n, rng)


def select_boltzmann_length(
    paths: list[ValidPath],
    n: int,
    beta: float,
    rng: random.Random,
) -> list[ValidPath]:
    """Strategy B: Boltzmann weighting by path length.

    Short paths get higher weight at high β (prefer simple executions).
    """
    lengths = [path_length(p) for p in paths]
    min_l = min(lengths) if lengths else 0
    weights = [math.exp(-beta * (l - min_l)) for l in lengths]
    return _weighted_sample(paths, weights, n, rng)


def select_entropy_weighted(
    ss: StateSpace,
    paths: list[ValidPath],
    n: int,
    beta: float,
    rng: random.Random,
) -> list[ValidPath]:
    """Strategy C: Entropy weighting (prefer high-entropy paths).

    High entropy paths traverse more choice points — more "interesting."
    Weight ∝ exp(+β · entropy) — NOTE: positive sign (favor high entropy).
    """
    entropies = [path_entropy(ss, p) for p in paths]
    max_e = max(entropies) if entropies else 0
    weights = [math.exp(beta * (e - max_e)) for e in entropies]
    return _weighted_sample(paths, weights, n, rng)


def _weighted_sample(
    items: list,
    weights: list[float],
    n: int,
    rng: random.Random,
) -> list:
    """Weighted sampling without replacement (approximate via repeated draws)."""
    if n >= len(items):
        return list(items)
    total = sum(weights)
    if total <= 0:
        return rng.sample(items, n)
    probs = [w / total for w in weights]

    selected: list = []
    remaining = list(range(len(items)))
    rem_probs = list(probs)

    for _ in range(n):
        if not remaining:
            break
        # Normalize remaining probs
        total_r = sum(rem_probs)
        if total_r <= 0:
            idx = rng.choice(range(len(remaining)))
        else:
            r = rng.random() * total_r
            cumulative = 0.0
            idx = 0
            for i, p in enumerate(rem_probs):
                cumulative += p
                if cumulative >= r:
                    idx = i
                    break

        selected.append(items[remaining[idx]])
        remaining.pop(idx)
        rem_probs.pop(idx)

    return selected


# ---------------------------------------------------------------------------
# Kill detection
# ---------------------------------------------------------------------------

def kills_mutant(path: ValidPath, mutant: Mutant) -> bool:
    """Does this test path detect (kill) this mutant?

    A path kills a "remove" mutant if the path uses the removed transition.
    A path kills a "redirect" mutant if the path uses the redirected transition.
    """
    src, label, tgt = mutant.original_edge

    # Build the sequence of transitions in the path
    prev_state = None  # We don't know the initial state from ValidPath alone
    for step in path.steps:
        if step.label == label and step.target == tgt:
            return True  # Path traverses the mutated edge

    return False


def kills_mutant_stateful(
    ss: StateSpace,
    path: ValidPath,
    mutant: Mutant,
) -> bool:
    """Stateful kill detection: replay the path on the original, check if
    the mutated transition is traversed."""
    src, label, tgt = mutant.original_edge
    state = ss.top
    for step in path.steps:
        # Check if this step IS the mutated transition
        if state == src and step.label == label and step.target == tgt:
            return True
        state = step.target
    return False


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_trial(
    ss: StateSpace,
    paths: list[ValidPath],
    mutants: list[Mutant],
    strategy: str,
    beta: float | None,
    budget: int,
    rng: random.Random,
) -> TrialResult:
    """Run a single trial: select paths, count kills."""
    if strategy == "uniform":
        selected = select_uniform(paths, budget, rng)
    elif strategy == "energy":
        selected = select_boltzmann_energy(ss, paths, budget, beta or 1.0, rng)
    elif strategy == "length":
        selected = select_boltzmann_length(paths, budget, beta or 1.0, rng)
    elif strategy == "entropy":
        selected = select_entropy_weighted(ss, paths, budget, beta or 1.0, rng)
    elif strategy == "adaptive":
        selected = select_adaptive(ss, paths, budget, 0.1, beta or 3.0, rng)
    elif strategy == "greedy":
        selected = select_coverage_greedy(ss, paths, budget, rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Count kills
    killed = set()
    for i, mutant in enumerate(mutants):
        for path in selected:
            if kills_mutant_stateful(ss, path, mutant):
                killed.add(i)
                break

    # Count coverage
    covered_transitions: set[tuple[int, str, int]] = set()
    for path in selected:
        state = ss.top
        for step in path.steps:
            covered_transitions.add((state, step.label, step.target))
            state = step.target

    return TrialResult(
        strategy=strategy,
        beta=beta,
        budget=budget,
        kill_rate=len(killed) / len(mutants) if mutants else 0.0,
        kills=len(killed),
        total_mutants=len(mutants),
        transitions_covered=len(covered_transitions),
        total_transitions=len(ss.transitions),
    )


def run_experiment(
    ss: StateSpace,
    protocol_name: str = "unknown",
    budgets: list[int] | None = None,
    betas: list[float] | None = None,
    n_trials: int = 50,
    seed: int = 42,
    max_paths: int = 500,
) -> ExperimentResult:
    """Run the full experiment: all strategies × all budgets × n_trials."""
    if budgets is None:
        budgets = [5, 10, 20]
    if betas is None:
        betas = [0.1, 0.5, 1.0, 2.0, 5.0]

    # Enumerate paths
    paths, truncated = enumerate_valid_paths(ss, max_revisits=2, max_paths=max_paths)
    if not paths:
        return ExperimentResult(
            protocol_name=protocol_name,
            num_states=len(ss.states),
            num_paths=0,
            num_mutants=0,
            trials=[],
        )

    # Generate mutants
    mutants = generate_mutants(ss)

    rng = random.Random(seed)
    trials: list[TrialResult] = []

    for budget in budgets:
        # Uniform baseline
        for trial in range(n_trials):
            t = run_trial(ss, paths, mutants, "uniform", None, budget,
                         random.Random(rng.randint(0, 10**9)))
            trials.append(t)

        # Greedy baseline (optimal coverage)
        for trial in range(n_trials):
            t = run_trial(ss, paths, mutants, "greedy", None, budget,
                         random.Random(rng.randint(0, 10**9)))
            trials.append(t)

        # Boltzmann strategies at each β
        for beta in betas:
            for strategy in ["energy", "length", "entropy", "adaptive"]:
                for trial in range(n_trials):
                    t = run_trial(ss, paths, mutants, strategy, beta, budget,
                                 random.Random(rng.randint(0, 10**9)))
                    trials.append(t)

    return ExperimentResult(
        protocol_name=protocol_name,
        num_states=len(ss.states),
        num_paths=len(paths),
        num_mutants=len(mutants),
        trials=trials,
    )


# ---------------------------------------------------------------------------
# Analysis: summarize and compare
# ---------------------------------------------------------------------------

def summarize_experiment(result: ExperimentResult) -> list[StrategyComparison]:
    """Summarize trial results into strategy comparisons."""
    # Group by (strategy, beta, budget)
    groups: dict[tuple[str, float | None, int], list[TrialResult]] = {}
    for t in result.trials:
        key = (t.strategy, t.beta, t.budget)
        groups.setdefault(key, []).append(t)

    summaries: list[StrategyComparison] = []
    for (strategy, beta, budget), trials in sorted(groups.items()):
        kill_rates = [t.kill_rate for t in trials]
        coverages = [t.transitions_covered / t.total_transitions
                     if t.total_transitions > 0 else 0.0 for t in trials]

        mean_kr = sum(kill_rates) / len(kill_rates)
        std_kr = math.sqrt(
            sum((k - mean_kr) ** 2 for k in kill_rates) / len(kill_rates)
        ) if len(kill_rates) > 1 else 0.0

        mean_cov = sum(coverages) / len(coverages)

        summaries.append(StrategyComparison(
            strategy=strategy,
            beta=beta,
            mean_kill_rate=mean_kr,
            std_kill_rate=std_kr,
            mean_coverage=mean_cov,
            budget=budget,
        ))

    return summaries


def select_adaptive(
    ss: StateSpace,
    paths: list[ValidPath],
    n: int,
    beta_start: float,
    beta_end: float,
    rng: random.Random,
) -> list[ValidPath]:
    """Strategy D: Adaptive temperature — start hot, cool down.

    First half of budget at beta_start (explore), second half at beta_end (exploit).
    This addresses the finding that cold Boltzmann misses recursive core mutations
    at small budgets but excels at large budgets.
    """
    n_hot = n // 2
    n_cold = n - n_hot

    hot_paths = select_boltzmann_energy(ss, paths, n_hot, beta_start, rng)

    # Remove already selected from candidates for cold phase
    hot_set = set(id(p) for p in hot_paths)
    remaining = [p for p in paths if id(p) not in hot_set]
    if not remaining:
        remaining = paths

    cold_paths = select_boltzmann_energy(ss, remaining, n_cold, beta_end, rng)

    return hot_paths + cold_paths


def select_coverage_greedy(
    ss: StateSpace,
    paths: list[ValidPath],
    n: int,
    rng: random.Random,
) -> list[ValidPath]:
    """Strategy E: Greedy coverage — always pick the path covering most new transitions.

    This is the optimal baseline — the strategy a smart engineer would use.
    """
    selected: list[ValidPath] = []
    covered: set[tuple[int, str, int]] = set()

    # Precompute transitions per path
    path_trans: list[set[tuple[int, str, int]]] = []
    for p in paths:
        trans = set()
        state = ss.top
        for step in p.steps:
            trans.add((state, step.label, step.target))
            state = step.target
        path_trans.append(trans)

    remaining = list(range(len(paths)))

    for _ in range(min(n, len(paths))):
        if not remaining:
            break
        # Pick path with most uncovered transitions
        best_idx = max(remaining, key=lambda i: len(path_trans[i] - covered))
        selected.append(paths[best_idx])
        covered |= path_trans[best_idx]
        remaining.remove(best_idx)

    return selected


def format_results(summaries: list[StrategyComparison]) -> str:
    """Format results as a readable table."""
    lines = [
        f"{'Strategy':<12} {'β':>5} {'N':>4} {'Kill Rate':>10} {'± Std':>8} {'Coverage':>9}",
        "-" * 55,
    ]
    for s in summaries:
        beta_str = f"{s.beta:.1f}" if s.beta is not None else "  —"
        lines.append(
            f"{s.strategy:<12} {beta_str:>5} {s.budget:>4} "
            f"{s.mean_kill_rate:>10.3f} {s.std_kill_rate:>8.3f} "
            f"{s.mean_coverage:>9.3f}"
        )
    return "\n".join(lines)
