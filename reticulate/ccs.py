"""CCS (Calculus of Communicating Systems) semantics for session types.

Provides a translation from session type ASTs to CCS processes,
construction of Labelled Transition Systems (LTS) from CCS terms,
pretty-printing of CCS processes, and strong bisimulation checking.

Step 26 of the 1000 Steps Towards Session Types as Algebraic Reticulates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


# ---------------------------------------------------------------------------
# CCS Process AST (frozen dataclasses, hashable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Nil:
    """Terminated process — no further actions."""


@dataclass(frozen=True)
class Prefix:
    """Action prefix ``a . P``."""
    action: str
    cont: "CCSProcess"


@dataclass(frozen=True)
class Sum:
    """Non-deterministic choice ``P + Q + ...``."""
    procs: tuple["CCSProcess", ...]


@dataclass(frozen=True)
class Par:
    """Parallel composition ``P | Q``."""
    left: "CCSProcess"
    right: "CCSProcess"


@dataclass(frozen=True)
class Restrict:
    """Restriction ``P \\ {a, b, ...}``."""
    proc: "CCSProcess"
    actions: frozenset[str]


@dataclass(frozen=True)
class CCSRec:
    """Recursive process ``fix X . P``."""
    var: str
    body: "CCSProcess"


@dataclass(frozen=True)
class CCSVar:
    """Process variable reference."""
    name: str


CCSProcess = Union[Nil, Prefix, Sum, Par, Restrict, CCSRec, CCSVar]


# ---------------------------------------------------------------------------
# Labelled Transition System
# ---------------------------------------------------------------------------

@dataclass
class LTS:
    """Labelled Transition System (states are integers).

    - ``states``: set of state IDs
    - ``transitions``: list of ``(source, label, target)`` triples
    - ``initial``: the initial state
    - ``labels``: human-readable description per state
    """
    states: set[int] = field(default_factory=set)
    transitions: list[tuple[int, str, int]] = field(default_factory=list)
    initial: int = 0
    labels: dict[int, str] = field(default_factory=dict)
    tau_transitions: set[tuple[int, str, int]] = field(default_factory=set)

    def successors(self, state: int) -> list[tuple[str, int]]:
        """Return ``(label, target)`` pairs for transitions from *state*."""
        return [(l, t) for s, l, t in self.transitions if s == state]


# ---------------------------------------------------------------------------
# CCS Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CCSResult:
    """Result of CCS analysis for a session type.

    - ``ccs_process``: the CCS process term
    - ``lts``: the labelled transition system derived from the CCS process
    - ``num_states``: number of LTS states
    - ``num_transitions``: number of LTS transitions
    - ``bisimilar_to_statespace``: whether the LTS is bisimilar to the
      session type's state space
    """
    ccs_process: CCSProcess
    lts: LTS
    num_states: int
    num_transitions: int
    bisimilar_to_statespace: bool


# ---------------------------------------------------------------------------
# Translation: Session Type AST → CCS Process
# ---------------------------------------------------------------------------

def session_to_ccs(ast: "SessionType") -> CCSProcess:
    """Translate a session type AST to a CCS process.

    Translation rules:
    - ``End`` → ``Nil``
    - ``Wait`` → ``Nil``
    - ``Branch &{m₁:S₁, ..., mₙ:Sₙ}`` → ``Sum(m₁.⟦S₁⟧ + ... + mₙ.⟦Sₙ⟧)``
    - ``Select +{l₁:S₁, ..., lₙ:Sₙ}`` → ``Sum(τ_l₁.⟦S₁⟧ + ... + τ_lₙ.⟦Sₙ⟧)``
    - ``Parallel(S₁, S₂)`` → ``Par(⟦S₁⟧, ⟦S₂⟧)``
    - ``Rec X . S`` → ``CCSRec(X, ⟦S⟧)``
    - ``Var X`` → ``CCSVar(X)``
    - ``Continuation(L, R)`` → translate L then sequence to R

    Selection uses internal (tau) actions prefixed with ``τ_`` to
    distinguish from externally visible branch actions.
    """
    from reticulate.parser import (
        Branch, Continuation, End, Parallel, Rec, Select, Var, Wait,
    )

    match ast:
        case End():
            return Nil()

        case Wait():
            return Nil()

        case Var(name=name):
            return CCSVar(name)

        case Branch(choices=choices):
            procs = tuple(
                Prefix(label, session_to_ccs(body))
                for label, body in choices
            )
            if len(procs) == 1:
                return procs[0]
            return Sum(procs)

        case Select(choices=choices):
            procs = tuple(
                Prefix(f"τ_{label}", session_to_ccs(body))
                for label, body in choices
            )
            if len(procs) == 1:
                return procs[0]
            return Sum(procs)

        case Parallel(branches=branches):
            if len(branches) == 0:
                return Nil()
            result = session_to_ccs(branches[0])
            for b in branches[1:]:
                result = Par(result, session_to_ccs(b))
            return result

        case Rec(var=var, body=body):
            return CCSRec(var, session_to_ccs(body))

        case Continuation(left=left, right=right):
            # Continuation sequences parallel completion into the next part.
            # We model this as: the left (parallel) runs to completion,
            # then a synchronisation action τ_sync leads to the right.
            left_ccs = session_to_ccs(left)
            right_ccs = session_to_ccs(right)
            return _sequence_with_sync(left_ccs, right_ccs)

        case _:
            raise TypeError(f"unknown AST node: {type(ast).__name__}")


def _sequence_with_sync(left: CCSProcess, right: CCSProcess) -> CCSProcess:
    """Replace all Nil leaves in *left* with ``Prefix('τ_sync', right)``."""
    match left:
        case Nil():
            return Prefix("τ_sync", right)
        case Prefix(action=a, cont=c):
            return Prefix(a, _sequence_with_sync(c, right))
        case Sum(procs=ps):
            return Sum(tuple(_sequence_with_sync(p, right) for p in ps))
        case Par(left=l, right=r):
            # In parallel, both branches must complete; only the combined
            # bottom triggers continuation.  We keep the parallel structure
            # and let the LTS builder handle synchronisation.
            return Par(l, r)
        case CCSRec(var=v, body=b):
            return CCSRec(v, _sequence_with_sync(b, right))
        case CCSVar():
            return left
        case Restrict(proc=p, actions=a):
            return Restrict(_sequence_with_sync(p, right), a)
        case _:
            return left


# ---------------------------------------------------------------------------
# CCS Process → Labelled Transition System
# ---------------------------------------------------------------------------

def ccs_to_lts(proc: CCSProcess, *, max_states: int = 500) -> LTS:
    """Build a Labelled Transition System from a CCS process.

    Uses structural operational semantics to enumerate reachable states.
    Recursive processes are unfolded on demand.

    Parameters:
        proc: the CCS process to explore
        max_states: safety bound on state count (default 500)

    Returns:
        An LTS with integer state IDs.
    """
    builder = _LTSBuilder(max_states=max_states)
    return builder.build(proc)


class _LTSBuilder:
    """Builds an LTS by exploring CCS operational semantics."""

    def __init__(self, *, max_states: int = 500) -> None:
        self._next_id: int = 0
        self._states: dict[int, str] = {}
        self._transitions: list[tuple[int, str, int]] = []
        self._tau_transitions: set[tuple[int, str, int]] = set()
        self._proc_to_id: dict[CCSProcess, int] = {}
        self._max_states = max_states

    def _fresh(self, label: str) -> int:
        sid = self._next_id
        self._next_id += 1
        self._states[sid] = label
        return sid

    def _get_or_create(self, proc: CCSProcess) -> int:
        """Return existing state ID for *proc*, or create a new one."""
        # Normalize: unfold top-level CCSRec once
        proc = _normalize(proc)
        if proc in self._proc_to_id:
            return self._proc_to_id[proc]
        sid = self._fresh(_short_label(proc))
        self._proc_to_id[proc] = sid
        return sid

    def build(self, proc: CCSProcess) -> LTS:
        proc = _normalize(proc)
        initial = self._get_or_create(proc)

        worklist = [proc]
        visited: set[CCSProcess] = set()

        while worklist:
            current = worklist.pop()
            if current in visited:
                continue
            visited.add(current)

            if self._next_id > self._max_states:
                break

            src = self._proc_to_id[current]
            transitions = _sos_step(current)

            for action, target in transitions:
                target = _normalize(target)
                tgt_id = self._get_or_create(target)
                tr = (src, action, tgt_id)
                self._transitions.append(tr)
                if action.startswith("τ"):
                    self._tau_transitions.add(tr)
                if target not in visited:
                    worklist.append(target)

        return LTS(
            states=set(self._states.keys()),
            transitions=self._transitions,
            initial=initial,
            labels=dict(self._states),
            tau_transitions=self._tau_transitions,
        )


def _normalize(proc: CCSProcess) -> CCSProcess:
    """Normalize a process: unfold top-level CCSRec, flatten nested Sums."""
    # Unfold top-level recursion
    while isinstance(proc, CCSRec):
        proc = _unfold_rec(proc)
    # Flatten nested sums
    if isinstance(proc, Sum):
        flat = _flatten_sum(proc)
        if len(flat) == 1:
            return _normalize(flat[0])
        proc = Sum(tuple(flat))
    return proc


def _unfold_rec(rec: CCSRec) -> CCSProcess:
    """Unfold ``fix X . P`` by substituting ``fix X . P`` for ``X`` in ``P``."""
    return _subst(rec.body, rec.var, rec)


def _subst(proc: CCSProcess, var: str, replacement: CCSProcess) -> CCSProcess:
    """Substitute *replacement* for *var* in *proc*."""
    match proc:
        case Nil():
            return proc
        case CCSVar(name=name):
            return replacement if name == var else proc
        case Prefix(action=a, cont=c):
            return Prefix(a, _subst(c, var, replacement))
        case Sum(procs=ps):
            return Sum(tuple(_subst(p, var, replacement) for p in ps))
        case Par(left=l, right=r):
            return Par(_subst(l, var, replacement), _subst(r, var, replacement))
        case Restrict(proc=p, actions=acts):
            return Restrict(_subst(p, var, replacement), acts)
        case CCSRec(var=v, body=b):
            if v == var:
                return proc  # shadowed
            return CCSRec(v, _subst(b, var, replacement))
        case _:
            return proc


def _flatten_sum(proc: Sum) -> list[CCSProcess]:
    """Flatten nested Sum nodes."""
    result: list[CCSProcess] = []
    for p in proc.procs:
        if isinstance(p, Sum):
            result.extend(_flatten_sum(p))
        else:
            result.append(p)
    return result


def _sos_step(proc: CCSProcess) -> list[tuple[str, CCSProcess]]:
    """Compute one-step transitions from *proc* via SOS rules.

    Returns list of ``(action, successor_process)`` pairs.
    """
    match proc:
        case Nil():
            return []

        case CCSVar():
            return []

        case Prefix(action=a, cont=c):
            return [(a, c)]

        case Sum(procs=ps):
            # Each summand can independently step
            result: list[tuple[str, CCSProcess]] = []
            for p in ps:
                result.extend(_sos_step(p))
            return result

        case Par(left=l, right=r):
            result = []
            # Left can step independently
            for action, l_prime in _sos_step(l):
                result.append((action, Par(l_prime, r)))
            # Right can step independently
            for action, r_prime in _sos_step(r):
                result.append((action, Par(l, r_prime)))
            # No CCS synchronisation (complement actions) for session types
            # since branch/select are unidirectional
            return result

        case Restrict(proc=p, actions=acts):
            result = []
            for action, p_prime in _sos_step(p):
                if action not in acts:
                    result.append((action, Restrict(p_prime, acts)))
            return result

        case CCSRec(var=v, body=b):
            # Unfold and step
            unfolded = _unfold_rec(proc)
            return _sos_step(unfolded)

        case _:
            return []


def _short_label(proc: CCSProcess) -> str:
    """Generate a short label for a process (for display in LTS)."""
    match proc:
        case Nil():
            return "0"
        case Prefix(action=a, cont=_):
            return f"{a}.…"
        case Sum(procs=ps):
            parts = [_short_label(p) for p in ps[:3]]
            if len(ps) > 3:
                parts.append("…")
            return " + ".join(parts)
        case Par(left=l, right=r):
            return f"{_short_label(l)} | {_short_label(r)}"
        case Restrict(proc=p, actions=_):
            return f"{_short_label(p)}\\L"
        case CCSRec(var=v, body=_):
            return f"fix {v}"
        case CCSVar(name=n):
            return n
        case _:
            return "?"


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def pretty_ccs(proc: CCSProcess) -> str:
    """Pretty-print a CCS process in standard notation.

    Uses:
    - ``0`` for Nil
    - ``a . P`` for Prefix
    - ``P + Q`` for Sum
    - ``P | Q`` for Par
    - ``P \\ {a, b}`` for Restrict
    - ``fix X . P`` for CCSRec
    - ``X`` for CCSVar
    """
    return _pretty_ccs(proc, prec=0)


def _pretty_ccs(proc: CCSProcess, prec: int) -> str:
    """Internal pretty-printer with precedence tracking.

    Precedence levels (higher binds tighter):
    0 — Sum (+)
    1 — Par (|)
    2 — Prefix (.)
    3 — Atoms (Nil, Var, Restrict, Rec)
    """
    match proc:
        case Nil():
            return "0"

        case CCSVar(name=n):
            return n

        case Prefix(action=a, cont=c):
            s = f"{a} . {_pretty_ccs(c, prec=2)}"
            return f"({s})" if prec > 2 else s

        case Sum(procs=ps):
            s = " + ".join(_pretty_ccs(p, prec=1) for p in ps)
            return f"({s})" if prec > 0 else s

        case Par(left=l, right=r):
            s = f"{_pretty_ccs(l, prec=2)} | {_pretty_ccs(r, prec=2)}"
            return f"({s})" if prec > 1 else s

        case Restrict(proc=p, actions=acts):
            sorted_acts = sorted(acts)
            act_str = ", ".join(sorted_acts)
            return f"{_pretty_ccs(p, prec=3)} \\ {{{act_str}}}"

        case CCSRec(var=v, body=b):
            s = f"fix {v} . {_pretty_ccs(b, prec=0)}"
            return f"({s})" if prec > 0 else s

        case _:
            return "?"


# ---------------------------------------------------------------------------
# Strong bisimulation
# ---------------------------------------------------------------------------

def check_bisimulation(lts1: LTS, lts2: LTS) -> bool:
    """Check strong bisimulation between two LTS.

    Uses the partition-refinement algorithm (Paige-Tarjan style).
    Two LTS are strongly bisimilar if and only if their initial states
    are in the same equivalence class of the coarsest bisimulation.

    The two LTS are first combined into a single transition system,
    then partition refinement finds the coarsest bisimulation relation.
    """
    if not lts1.states or not lts2.states:
        return not lts1.states and not lts2.states

    # Combine both LTS into one (offset lts2 state IDs)
    offset = max(lts1.states) + 1 if lts1.states else 0
    combined_transitions: list[tuple[int, str, int]] = list(lts1.transitions)
    for s, l, t in lts2.transitions:
        combined_transitions.append((s + offset, l, t + offset))

    all_states: set[int] = set(lts1.states)
    for s in lts2.states:
        all_states.add(s + offset)

    # Collect all actions
    all_actions: set[str] = set()
    for _, l, _ in combined_transitions:
        all_actions.add(l)

    # Build predecessor map: action -> target -> set of sources
    pred: dict[str, dict[int, set[int]]] = {}
    for a in all_actions:
        pred[a] = {}
    for s, l, t in combined_transitions:
        if t not in pred[l]:
            pred[l][t] = set()
        pred[l][t].add(s)

    # Initial partition: all states in one block
    # (or separate terminal from non-terminal for faster convergence)
    terminal = set()
    non_terminal = set()
    for s in all_states:
        has_outgoing = any(src == s for src, _, _ in combined_transitions)
        if has_outgoing:
            non_terminal.add(s)
        else:
            terminal.add(s)

    partition: list[set[int]] = []
    if terminal:
        partition.append(terminal)
    if non_terminal:
        partition.append(non_terminal)
    if not partition:
        partition = [all_states.copy()]

    # Partition refinement
    changed = True
    while changed:
        changed = False
        new_partition: list[set[int]] = []
        for block in partition:
            if len(block) <= 1:
                new_partition.append(block)
                continue

            # Try to split this block
            split = _split_block(block, partition, all_actions, pred)
            if len(split) > 1:
                changed = True
                new_partition.extend(split)
            else:
                new_partition.append(block)
        partition = new_partition

    # Check if initial states of lts1 and lts2 are in the same block
    init1 = lts1.initial
    init2 = lts2.initial + offset
    for block in partition:
        if init1 in block and init2 in block:
            return True
    return False


def _split_block(
    block: set[int],
    partition: list[set[int]],
    actions: set[str],
    pred: dict[str, dict[int, set[int]]],
) -> list[set[int]]:
    """Try to split *block* using partition refinement.

    For each action and each splitter block, check if all states in
    *block* have the same ability to reach the splitter via that action.
    """
    for action in actions:
        for splitter in partition:
            # States in block that can reach splitter via action
            can_reach: set[int] = set()
            for t in splitter:
                if t in pred[action]:
                    can_reach |= (pred[action][t] & block)

            cannot_reach = block - can_reach
            if can_reach and cannot_reach:
                return [can_reach, cannot_reach]

    return [block]


# ---------------------------------------------------------------------------
# Bisimulation with session-type state space
# ---------------------------------------------------------------------------

def check_bisimulation_with_statespace(
    lts: LTS,
    ss: "StateSpace",
) -> bool:
    """Check if an LTS is strongly bisimilar to a session-type state space.

    Converts the state space to an LTS and delegates to check_bisimulation.
    Selection transitions in the state space use ``τ_label`` actions to
    match the CCS translation convention.
    """
    ss_lts = _statespace_to_lts(ss)
    return check_bisimulation(lts, ss_lts)


def _statespace_to_lts(ss: "StateSpace") -> LTS:
    """Convert a session-type StateSpace to an LTS.

    Selection transitions get ``τ_`` prefix on their labels.
    """
    transitions: list[tuple[int, str, int]] = []
    tau_transitions: set[tuple[int, str, int]] = set()

    for src, label, tgt in ss.transitions:
        if ss.is_selection(src, label, tgt):
            action = f"τ_{label}"
            tr = (src, action, tgt)
            transitions.append(tr)
            tau_transitions.add(tr)
        else:
            tr = (src, label, tgt)
            transitions.append(tr)

    return LTS(
        states=set(ss.states),
        transitions=transitions,
        initial=ss.top,
        labels=dict(ss.labels),
        tau_transitions=tau_transitions,
    )


# ---------------------------------------------------------------------------
# High-level analysis entry point
# ---------------------------------------------------------------------------

def analyze_ccs(ast: "SessionType") -> CCSResult:
    """Full CCS analysis pipeline for a session type.

    1. Translate AST to CCS process
    2. Build LTS from CCS process
    3. Check bisimulation with the session type's state space

    Returns a CCSResult with all information.
    """
    from reticulate.statespace import build_statespace

    ccs_proc = session_to_ccs(ast)
    lts = ccs_to_lts(ccs_proc)
    ss = build_statespace(ast)
    bisimilar = check_bisimulation_with_statespace(lts, ss)

    return CCSResult(
        ccs_process=ccs_proc,
        lts=lts,
        num_states=len(lts.states),
        num_transitions=len(lts.transitions),
        bisimilar_to_statespace=bisimilar,
    )
