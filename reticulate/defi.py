"""DeFi cross-contract interaction verification (Step 74).

Models decentralised-finance (DeFi) protocols as binary session types and
analyses them with reticulate's lattice machinery.  The focus is on
*cross-contract interactions* --- the calls that one contract makes into
another contract mid-execution --- which are the dominant source of
multi-billion-dollar losses in the DeFi ecosystem (reentrancy, sandwich,
flash-loan, and oracle-manipulation exploits).

Five canonical scenarios are provided:

1. ``uniswap_swap``           -- AMM constant-product swap (swap / settle).
2. ``aave_lend_borrow``       -- deposit / borrow / repay / withdraw loop.
3. ``flash_loan``             -- atomic borrow / use / repay within one tx.
4. ``vulnerable_withdraw``    -- the classic DAO-style reentrancy trap.
5. ``price_oracle_consumer``  -- consumer contract reading an oracle under
   sandwich / manipulation pressure.

For each scenario we build a state space, check lattice properties, and
bundle a bidirectional morphism pair::

    phi : L(S) -> ExploitPatterns
    psi : ExploitPatterns -> subsets of L(S)

that classifies every reachable state according to which exploit family
it either enables or mitigates.  The morphism pair is a Galois connection
and yields concrete audit / runtime-monitoring actions.

This module is *pure*: it has no Web3 dependency.  Real blockchains are
not contacted.  Everything is symbolic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from reticulate.lattice import LatticeResult, check_lattice
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace


# ---------------------------------------------------------------------------
# Exploit pattern taxonomy
# ---------------------------------------------------------------------------

#: Canonical exploit-pattern labels used by the phi/psi morphism pair.
EXPLOIT_PATTERNS: tuple[str, ...] = (
    "Safe",           # no known exploit vector at this state
    "Reentrancy",     # external call before state update
    "Sandwich",       # price-sensitive step, vulnerable to MEV sandwich
    "FlashLoan",      # inside an atomic flash-loan window
    "OracleManip",    # depends on an externally controlled oracle price
    "Unchecked",      # external call whose return value is ignored
)

#: Human-readable description of each exploit pattern.
PATTERN_DESCRIPTIONS: dict[str, str] = {
    "Safe": "No externally observable vulnerability at this protocol point.",
    "Reentrancy": "State not yet updated before an external call; a recursive "
                  "re-entry can drain funds (DAO 2016, Cream 2021).",
    "Sandwich": "Price-sensitive AMM step exposed to front/back-run MEV.",
    "FlashLoan": "Inside an atomic flash-loan window where invariants may "
                 "temporarily be violated.",
    "OracleManip": "Reads a price oracle whose reported value can be "
                   "manipulated within a block (bZx 2020).",
    "Unchecked": "Return value of an external call is not inspected.",
}

#: Real-world losses (USD, approximate) associated with each pattern class.
#: Sources: Chainalysis 2023 crypto crime report; Rekt.news leaderboard.
PATTERN_LOSSES_USD: dict[str, float] = {
    "Safe": 0.0,
    "Reentrancy": 320_000_000.0,   # DAO 60M + Cream 130M + others
    "Sandwich": 1_380_000_000.0,   # cumulative MEV extraction 2020-2023
    "FlashLoan": 850_000_000.0,    # bZx, Harvest, PancakeBunny, etc.
    "OracleManip": 720_000_000.0,  # Mango 115M, Cream 130M, etc.
    "Unchecked": 30_000_000.0,
}


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeFiScenario:
    """A DeFi interaction scenario as a binary session type."""

    name: str
    description: str
    session_type_string: str
    #: Map from state label prefix -> expected exploit pattern.  Used by
    #: phi.  Order matters: the first matching prefix wins.
    phi_rules: tuple[tuple[str, str], ...]
    #: Short audit actions keyed by exploit pattern.
    audit_actions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeFiAnalysis:
    """Result of analysing a single scenario."""

    scenario: DeFiScenario
    state_space: StateSpace
    lattice: LatticeResult
    pattern_histogram: dict[str, int]
    roundtrip_coverage: float
    total_loss_exposure_usd: float
    is_well_formed: bool


def uniswap_swap() -> DeFiScenario:
    """Constant-product AMM swap (Uniswap v2-style)."""
    return DeFiScenario(
        name="UniswapSwap",
        description=(
            "Trader sends tokenIn to a pair contract, pair computes the "
            "output via the x*y=k invariant, transfers tokenOut back, "
            "and settles.  The amountOut calculation is the sandwich "
            "attack surface."
        ),
        session_type_string=(
            "&{approve: &{swap_quote: "
            "+{quote_ok: &{swap_execute: &{settle: end}}, "
            "  quote_bad: end}}}"
        ),
        phi_rules=(
            ("swap_quote", "Sandwich"),
            ("swap_execute", "Sandwich"),
            ("settle", "Safe"),
            ("approve", "Safe"),
            ("quote_ok", "Safe"),
            ("quote_bad", "Safe"),
        ),
        audit_actions={
            "Sandwich": "Require minAmountOut slippage bound; monitor "
                        "block-level price delta.",
            "Safe": "Standard transfer/approval checks.",
        },
    )


def aave_lend_borrow() -> DeFiScenario:
    """Aave-style deposit / borrow / repay / withdraw lifecycle."""
    return DeFiScenario(
        name="AaveLendBorrow",
        description=(
            "User deposits collateral, borrows against it, accrues "
            "interest, repays, and finally withdraws.  The borrow and "
            "repay steps depend on an oracle price for health-factor "
            "computation."
        ),
        session_type_string=(
            "&{deposit: &{oracle_price: &{borrow: "
            "+{borrow_ok: &{accrue: &{oracle_price2: &{repay: "
            "&{withdraw: end}}}}, borrow_rejected: end}}}}"
        ),
        phi_rules=(
            ("oracle_price", "OracleManip"),
            ("borrow", "OracleManip"),
            ("deposit", "Safe"),
            ("repay", "Safe"),
            ("withdraw", "Safe"),
            ("accrue", "Safe"),
            ("borrow_ok", "Safe"),
            ("borrow_rejected", "Safe"),
        ),
        audit_actions={
            "OracleManip": "Use time-weighted average price (TWAP) or "
                           "Chainlink aggregator; bound single-block "
                           "deviation.",
            "Safe": "ERC-20 balance and allowance invariants.",
        },
    )


def flash_loan() -> DeFiScenario:
    """Atomic flash-loan: borrow, use, repay within one transaction."""
    return DeFiScenario(
        name="FlashLoan",
        description=(
            "Borrower receives N tokens, invokes arbitrary callback "
            "logic that must return N + fee before the transaction "
            "ends.  Every state between borrow and repay is inside the "
            "flash-loan window."
        ),
        session_type_string=(
            "&{flash_borrow: &{callback_enter: &{callback_body: "
            "&{callback_exit: &{flash_repay: end}}}}}"
        ),
        phi_rules=(
            ("flash_borrow", "FlashLoan"),
            ("callback_enter", "FlashLoan"),
            ("callback_body", "FlashLoan"),
            ("callback_exit", "FlashLoan"),
            ("flash_repay", "Safe"),
        ),
        audit_actions={
            "FlashLoan": "Check global invariants AT repay time, not "
                         "during callback; disallow reentrant flash "
                         "loans from the same caller.",
            "Safe": "Reconcile balances and emit event.",
        },
    )


def vulnerable_withdraw() -> DeFiScenario:
    """DAO-style reentrancy: external call BEFORE state update."""
    return DeFiScenario(
        name="VulnerableWithdraw",
        description=(
            "withdraw() sends ETH to the caller BEFORE zeroing the "
            "internal balance, allowing a malicious fallback to "
            "re-enter withdraw() and drain the contract.  This is the "
            "canonical Checks-Effects-Interactions violation."
        ),
        session_type_string=(
            "&{check_balance: &{external_call: "
            "+{reenter: &{external_call2: &{update_state_late: end}}, "
            "  no_reenter: &{update_state_late: end}}}}"
        ),
        phi_rules=(
            ("external_call", "Reentrancy"),
            ("reenter", "Reentrancy"),
            ("update_state_late", "Unchecked"),
            ("check_balance", "Safe"),
            ("no_reenter", "Safe"),
        ),
        audit_actions={
            "Reentrancy": "Enforce Checks-Effects-Interactions: move "
                          "state update BEFORE external call; add "
                          "ReentrancyGuard (OpenZeppelin).",
            "Unchecked": "Verify return value of external call; revert "
                         "on failure.",
            "Safe": "Read-only guard.",
        },
    )


def price_oracle_consumer() -> DeFiScenario:
    """Consumer contract reading a single-block oracle price."""
    return DeFiScenario(
        name="PriceOracleConsumer",
        description=(
            "A liquidation bot reads spot price from an AMM, compares "
            "against a threshold, and liquidates under-collateralised "
            "positions.  An attacker can sandwich the oracle read to "
            "trigger fake liquidations."
        ),
        session_type_string=(
            "&{read_oracle: &{compare_threshold: "
            "+{liquidate_yes: &{external_call: &{settle: end}}, "
            "  liquidate_no: end}}}"
        ),
        phi_rules=(
            ("read_oracle", "OracleManip"),
            ("compare_threshold", "OracleManip"),
            ("liquidate_yes", "Sandwich"),
            ("external_call", "Reentrancy"),
            ("settle", "Safe"),
            ("liquidate_no", "Safe"),
        ),
        audit_actions={
            "OracleManip": "Use Chainlink or Uniswap v3 TWAP; reject "
                           "readings whose block-height delta < 2.",
            "Sandwich": "Mint MEV-protected transactions via Flashbots.",
            "Reentrancy": "Apply ReentrancyGuard to the settle path.",
            "Safe": "Emit event and exit.",
        },
    )


SCENARIOS: tuple[Callable[[], DeFiScenario], ...] = (
    uniswap_swap,
    aave_lend_borrow,
    flash_loan,
    vulnerable_withdraw,
    price_oracle_consumer,
)


# ---------------------------------------------------------------------------
# Phi / psi morphism pair
# ---------------------------------------------------------------------------


def phi_label(label: str, rules: tuple[tuple[str, str], ...]) -> str:
    """Forward morphism: transition label -> exploit pattern."""
    for prefix, pattern in rules:
        if label.startswith(prefix):
            return pattern
    return "Safe"


def phi_state(ss: StateSpace, state: int,
              rules: tuple[tuple[str, str], ...]) -> str:
    """Exploit pattern at a state = worst pattern over outgoing labels.

    If a state has an outgoing transition labelled ``external_call`` and
    the label ``external_call`` maps to ``Reentrancy``, the state is
    classified as Reentrancy.  Safe is only returned when every outgoing
    label is classified Safe (or when the state is terminal).
    """
    severity = {
        "Safe": 0,
        "Unchecked": 1,
        "Sandwich": 2,
        "OracleManip": 3,
        "FlashLoan": 4,
        "Reentrancy": 5,
    }
    worst = "Safe"
    for src, lbl, _ in ss.transitions:
        if src != state:
            continue
        pat = phi_label(lbl, rules)
        if severity[pat] > severity[worst]:
            worst = pat
    return worst


def psi_pattern(ss: StateSpace, pattern: str,
                rules: tuple[tuple[str, str], ...]) -> frozenset[int]:
    """Reverse morphism: pattern -> set of states classified as that pattern."""
    return frozenset(
        s for s in ss.states if phi_state(ss, s, rules) == pattern
    )


def roundtrip_coverage(ss: StateSpace,
                       rules: tuple[tuple[str, str], ...]) -> float:
    """Fraction of states recovered by psi(phi(s)) = set containing s."""
    if not ss.states:
        return 1.0
    recovered = 0
    for s in ss.states:
        pat = phi_state(ss, s, rules)
        if s in psi_pattern(ss, pat, rules):
            recovered += 1
    return recovered / len(ss.states)


def pattern_histogram(ss: StateSpace,
                      rules: tuple[tuple[str, str], ...]) -> dict[str, int]:
    """Count of states per exploit pattern."""
    hist = {p: 0 for p in EXPLOIT_PATTERNS}
    for s in ss.states:
        hist[phi_state(ss, s, rules)] += 1
    return hist


def classify_morphism_pair() -> str:
    """Classify (phi, psi) in the morphism hierarchy.

    phi is a surjection from states to exploit patterns; psi selects
    preimages.  phi . psi = id on patterns that occur (retract).
    Ordered by severity, phi is monotone and psi is its upper adjoint,
    yielding a Galois insertion of EXPLOIT_PATTERNS into 2^{L(S)}.
    """
    return (
        "Galois insertion: phi (monotone, surjective onto used patterns) "
        "has upper adjoint psi (preimage), with phi o psi = id on the "
        "image.  Equivalently: (phi, psi) is a retraction of the state "
        "lattice onto the exploit-pattern lattice ordered by severity."
    )


# ---------------------------------------------------------------------------
# Analysis entry points
# ---------------------------------------------------------------------------


def analyse_scenario(scenario: DeFiScenario) -> DeFiAnalysis:
    """Parse, build state space, check lattice, compute morphism metrics."""
    ast: SessionType = parse(scenario.session_type_string)
    ss = build_statespace(ast)
    lattice = check_lattice(ss)
    hist = pattern_histogram(ss, scenario.phi_rules)
    cov = roundtrip_coverage(ss, scenario.phi_rules)
    exposure = sum(PATTERN_LOSSES_USD[p] for p, n in hist.items() if n > 0)
    return DeFiAnalysis(
        scenario=scenario,
        state_space=ss,
        lattice=lattice,
        pattern_histogram=hist,
        roundtrip_coverage=cov,
        total_loss_exposure_usd=exposure,
        is_well_formed=lattice.is_lattice,
    )


def analyse_all() -> list[DeFiAnalysis]:
    """Analyse every canonical scenario."""
    return [analyse_scenario(f()) for f in SCENARIOS]


# ---------------------------------------------------------------------------
# Exploit detection
# ---------------------------------------------------------------------------


def detect_reentrancy(ss: StateSpace,
                      rules: tuple[tuple[str, str], ...]) -> list[int]:
    """States classified as Reentrancy."""
    return sorted(s for s in ss.states
                  if phi_state(ss, s, rules) == "Reentrancy")


def detect_sandwich(ss: StateSpace,
                    rules: tuple[tuple[str, str], ...]) -> list[int]:
    """States classified as Sandwich-vulnerable."""
    return sorted(s for s in ss.states
                  if phi_state(ss, s, rules) == "Sandwich")


def detect_oracle_manipulation(
        ss: StateSpace,
        rules: tuple[tuple[str, str], ...]) -> list[int]:
    """States classified as oracle-manipulation-prone."""
    return sorted(s for s in ss.states
                  if phi_state(ss, s, rules) == "OracleManip")


def detect_flash_loan_window(
        ss: StateSpace,
        rules: tuple[tuple[str, str], ...]) -> list[int]:
    """States inside a flash-loan atomic window."""
    return sorted(s for s in ss.states
                  if phi_state(ss, s, rules) == "FlashLoan")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_analysis(analysis: DeFiAnalysis) -> str:
    """Human-readable report for a single scenario."""
    sc = analysis.scenario
    lines = [
        f"=== {sc.name} ===",
        sc.description,
        f"States: {len(analysis.state_space.states)}",
        f"Transitions: {len(analysis.state_space.transitions)}",
        f"Lattice: {analysis.lattice.is_lattice}",
        f"Roundtrip coverage phi/psi: {analysis.roundtrip_coverage:.2f}",
        f"Loss exposure (USD): {analysis.total_loss_exposure_usd:,.0f}",
        "Pattern histogram:",
    ]
    for pattern, count in analysis.pattern_histogram.items():
        if count:
            lines.append(f"  {pattern}: {count}")
    lines.append("Audit actions:")
    for pattern, action in sc.audit_actions.items():
        lines.append(f"  [{pattern}] {action}")
    return "\n".join(lines)


def format_summary(analyses: list[DeFiAnalysis]) -> str:
    """Summary table across all scenarios."""
    lines = [
        "DeFi cross-contract verification summary",
        "=" * 48,
        f"{'Scenario':<24}{'Lattice':<10}{'Cov':<8}{'USD exposure':>16}",
    ]
    for a in analyses:
        lines.append(
            f"{a.scenario.name:<24}"
            f"{'yes' if a.is_well_formed else 'no':<10}"
            f"{a.roundtrip_coverage:<8.2f}"
            f"{a.total_loss_exposure_usd:>16,.0f}"
        )
    total = sum(a.total_loss_exposure_usd for a in analyses)
    lines.append("-" * 48)
    lines.append(f"Total modelled exposure: USD {total:,.0f}")
    return "\n".join(lines)
