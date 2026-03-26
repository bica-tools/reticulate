"""Multiband audio processing as product lattice.

Models multiband audio processing (crossover filters splitting a signal into
frequency bands processed independently) as parallel session types whose
state spaces form product lattices.

Key insight: an N-band multiband processor is exactly the parallel composition
``band₁ ∥ band₂ ∥ ... ∥ bandₙ``, and its state space is the product lattice
``L(band₁) × L(band₂) × ... × L(bandₙ)``.

Lattice operations acquire spectral meaning:
  - Meet = shared spectral content (both configurations active)
  - Join = combined spectrum (either configuration active)
  - Distributivity = band independence (always holds for independent bands)

Step 57c of the 1000 Steps Towards Session Types as Algebraic Reticulates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reticulate.lattice import (
    LatticeResult,
    check_distributive,
    check_lattice,
    compute_join,
    compute_meet,
)
from reticulate.parser import parse, pretty
from reticulate.statespace import StateSpace, build_statespace

if TYPE_CHECKING:
    from reticulate.lattice import DistributivityResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrequencyBand:
    """A single frequency band in a multiband processor.

    Attributes:
        name: Human-readable band name (e.g. "low", "mid", "high").
        low_hz: Lower frequency bound in Hz.
        high_hz: Upper frequency bound in Hz.
        processor: Session type string for the band's effect chain.
    """

    name: str
    low_hz: float
    high_hz: float
    processor: str

    def __post_init__(self) -> None:
        if self.low_hz < 0:
            raise ValueError(f"low_hz must be non-negative, got {self.low_hz}")
        if self.high_hz <= self.low_hz:
            raise ValueError(
                f"high_hz ({self.high_hz}) must be greater than low_hz ({self.low_hz})"
            )

    @property
    def bandwidth(self) -> float:
        """Bandwidth in Hz."""
        return self.high_hz - self.low_hz

    @property
    def center_hz(self) -> float:
        """Geometric center frequency in Hz."""
        return (self.low_hz * self.high_hz) ** 0.5


@dataclass(frozen=True)
class MultibandConfig:
    """Configuration for a multiband audio processor.

    Attributes:
        bands: Ordered list of frequency bands (low to high).
        crossover_type: Type of crossover filter ("linkwitz-riley", "butterworth").
    """

    bands: tuple[FrequencyBand, ...]
    crossover_type: str = "linkwitz-riley"

    def __post_init__(self) -> None:
        if len(self.bands) < 2:
            raise ValueError(
                f"Multiband config requires at least 2 bands, got {len(self.bands)}"
            )
        # Verify bands are contiguous (each band's high == next band's low)
        for i in range(len(self.bands) - 1):
            if self.bands[i].high_hz != self.bands[i + 1].low_hz:
                raise ValueError(
                    f"Band gap between '{self.bands[i].name}' "
                    f"(high={self.bands[i].high_hz}) and "
                    f"'{self.bands[i + 1].name}' "
                    f"(low={self.bands[i + 1].low_hz})"
                )

    @property
    def num_bands(self) -> int:
        """Number of frequency bands."""
        return len(self.bands)

    @property
    def frequency_range(self) -> tuple[float, float]:
        """Overall frequency range (low, high) in Hz."""
        return (self.bands[0].low_hz, self.bands[-1].high_hz)


@dataclass(frozen=True)
class MultibandAnalysis:
    """Result of analyzing a multiband processor as a product lattice.

    Attributes:
        config: The multiband configuration analyzed.
        session_type: The parallel session type string.
        state_space: The product lattice state space.
        num_states: Total number of states in the product lattice.
        num_transitions: Total number of transitions.
        is_lattice: Whether the state space forms a lattice.
        is_distributive: Whether the lattice is distributive.
        classification: Birkhoff classification (boolean/distributive/modular/lattice).
        width: Maximum number of concurrent bands (= num_bands).
        compression_ratio: Ratio of product states to sum of individual states.
        band_sizes: Number of states per individual band.
        lattice_result: Full lattice check result.
    """

    config: MultibandConfig
    session_type: str
    state_space: StateSpace
    num_states: int
    num_transitions: int
    is_lattice: bool
    is_distributive: bool
    classification: str
    width: int
    compression_ratio: float
    band_sizes: tuple[int, ...]
    lattice_result: LatticeResult


# ---------------------------------------------------------------------------
# Session type construction
# ---------------------------------------------------------------------------

def _relabel_processor(processor: str, band_name: str) -> str:
    """Relabel a processor's session type to make labels band-unique.

    Prefixes each method/label with the band name to satisfy the WF-Par
    disjointness condition for parallel composition.
    """
    from reticulate.parser import (
        Branch,
        Continuation,
        End,
        Parallel,
        Rec,
        Select,
        Var,
        Wait,
    )

    ast = parse(processor)

    def relabel(node: object) -> object:
        if isinstance(node, (End, Wait)):
            return node
        if isinstance(node, Var):
            return Var(f"{band_name}_{node.name}")
        if isinstance(node, Branch):
            return Branch(tuple(
                (f"{band_name}_{m}", relabel(s))
                for m, s in node.choices
            ))
        if isinstance(node, Select):
            return Select(tuple(
                (f"{band_name}_{l}", relabel(s))
                for l, s in node.choices
            ))
        if isinstance(node, Rec):
            return Rec(f"{band_name}_{node.var}", relabel(node.body))
        if isinstance(node, Parallel):
            return Parallel(tuple(relabel(b) for b in node.branches))
        if isinstance(node, Continuation):
            return Continuation(relabel(node.left), relabel(node.right))
        return node  # pragma: no cover

    return pretty(relabel(ast))


def multiband_to_session_type(config: MultibandConfig) -> str:
    """Convert a multiband config to a parallel session type.

    Each band's processor becomes one branch of the parallel composition,
    with labels prefixed by the band name for disjointness.
    The result is ``(band₁ ∥ band₂ ∥ ... ∥ bandₙ) . &{sum: end}``
    where the final ``sum`` transition represents the summing junction
    that recombines the processed bands.

    Returns:
        A session type string of the form ``(B₁ ∥ B₂ ∥ ... ∥ Bₙ) . &{sum: end}``.
    """
    relabeled: list[str] = []
    for band in config.bands:
        relabeled.append(_relabel_processor(band.processor, band.name))

    # Build the parallel composition string
    # The pure product lattice L(B₁) × ... × L(Bₙ) is the timbral space.
    # No continuation needed — the product bottom IS the summing junction.
    if len(relabeled) == 2:
        return f"({relabeled[0]} || {relabeled[1]})"

    # Nest left-to-right: ((b1 || b2) || b3) || ...
    par_str = f"({relabeled[0]} || {relabeled[1]})"
    for i in range(2, len(relabeled)):
        par_str = f"({par_str} || {relabeled[i]})"
    return par_str


# ---------------------------------------------------------------------------
# State space construction
# ---------------------------------------------------------------------------

def spectral_state_space(config: MultibandConfig) -> StateSpace:
    """Build the product lattice state space for a multiband config.

    This is equivalent to ``build_statespace(parse(multiband_to_session_type(config)))``.

    Returns:
        The product lattice L(band₁) x L(band₂) x ... x L(bandₙ).
    """
    type_str = multiband_to_session_type(config)
    ast = parse(type_str)
    return build_statespace(ast)


# ---------------------------------------------------------------------------
# Spectral meet and join
# ---------------------------------------------------------------------------

def spectral_meet(ss: StateSpace, state1: int, state2: int) -> int | None:
    """Compute the spectral meet of two configurations.

    The meet represents the "shared spectral content" — the greatest
    processing state reachable from both configurations. In audio terms,
    this is the most-processed state that both configurations have
    already passed through.

    Returns:
        State ID of the meet, or None if no meet exists.
    """
    return compute_meet(ss, state1, state2)


def spectral_join(ss: StateSpace, state1: int, state2: int) -> int | None:
    """Compute the spectral join of two configurations.

    The join represents the "combined spectrum" — the least processing
    state from which both configurations are reachable. In audio terms,
    this is the earliest state from which both processing paths diverge.

    Returns:
        State ID of the join, or None if no join exists.
    """
    return compute_join(ss, state1, state2)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_multiband(config: MultibandConfig) -> MultibandAnalysis:
    """Analyze a multiband processor as a product lattice.

    Builds the product state space, checks lattice and distributivity
    properties, and computes compression ratio.

    Returns:
        MultibandAnalysis with full lattice characterization.
    """
    type_str = multiband_to_session_type(config)
    ss = spectral_state_space(config)

    lr = check_lattice(ss)
    dr = check_distributive(ss)

    # Compute individual band sizes
    band_sizes: list[int] = []
    for band in config.bands:
        band_type = _relabel_processor(band.processor, band.name)
        band_ss = build_statespace(parse(band_type))
        band_sizes.append(len(band_ss.states))

    # Compression ratio: product size / sum of individual sizes
    total_individual = sum(band_sizes)
    compression = len(ss.states) / total_individual if total_individual > 0 else 1.0

    return MultibandAnalysis(
        config=config,
        session_type=type_str,
        state_space=ss,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_lattice=lr.is_lattice,
        is_distributive=dr.is_distributive,
        classification=dr.classification,
        width=config.num_bands,
        compression_ratio=compression,
        band_sizes=tuple(band_sizes),
        lattice_result=lr,
    )


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

def standard_4band() -> MultibandConfig:
    """Standard 4-band mastering configuration.

    Bands: sub-bass (20-200 Hz), low-mid (200-2000 Hz),
    high-mid (2000-8000 Hz), high (8000-20000 Hz).
    Each band gets a compressor chain: analyze → compress → makeup → end.
    """
    return MultibandConfig(
        bands=(
            FrequencyBand(
                "sub",
                20.0,
                200.0,
                "&{analyze: &{compress: &{makeup: end}}}",
            ),
            FrequencyBand(
                "lomid",
                200.0,
                2000.0,
                "&{analyze: &{compress: &{makeup: end}}}",
            ),
            FrequencyBand(
                "himid",
                2000.0,
                8000.0,
                "&{analyze: &{compress: &{makeup: end}}}",
            ),
            FrequencyBand(
                "high",
                8000.0,
                20000.0,
                "&{analyze: &{compress: &{makeup: end}}}",
            ),
        ),
        crossover_type="linkwitz-riley",
    )


def mastering_5band() -> MultibandConfig:
    """Professional 5-band mastering configuration.

    Bands: sub (20-80 Hz), bass (80-300 Hz), mid (300-3000 Hz),
    presence (3000-8000 Hz), air (8000-20000 Hz).
    Each band gets EQ + compression: eq → compress → end.
    """
    return MultibandConfig(
        bands=(
            FrequencyBand(
                "sub",
                20.0,
                80.0,
                "&{eq: &{compress: end}}",
            ),
            FrequencyBand(
                "bass",
                80.0,
                300.0,
                "&{eq: &{compress: end}}",
            ),
            FrequencyBand(
                "mid",
                300.0,
                3000.0,
                "&{eq: &{compress: end}}",
            ),
            FrequencyBand(
                "presence",
                3000.0,
                8000.0,
                "&{eq: &{compress: end}}",
            ),
            FrequencyBand(
                "air",
                8000.0,
                20000.0,
                "&{eq: &{compress: end}}",
            ),
        ),
        crossover_type="linkwitz-riley",
    )


def de_esser_2band() -> MultibandConfig:
    """De-esser as a 2-band processor.

    Bands: body (20-5000 Hz) and sibilance (5000-20000 Hz).
    Body: passthrough (bypass → end).
    Sibilance: detect → compress → end.
    """
    return MultibandConfig(
        bands=(
            FrequencyBand(
                "body",
                20.0,
                5000.0,
                "&{bypass: end}",
            ),
            FrequencyBand(
                "sibilance",
                5000.0,
                20000.0,
                "&{detect: &{compress: end}}",
            ),
        ),
        crossover_type="butterworth",
    )
