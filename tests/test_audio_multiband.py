"""Tests for audio_multiband — multiband processing as product lattice.

Step 57c: verifies that multiband audio processors modeled as parallel
session types produce correct product lattices with spectral meet/join.
"""

from __future__ import annotations

import pytest

from reticulate.audio_multiband import (
    FrequencyBand,
    MultibandAnalysis,
    MultibandConfig,
    analyze_multiband,
    de_esser_2band,
    mastering_5band,
    multiband_to_session_type,
    spectral_join,
    spectral_meet,
    spectral_state_space,
    standard_4band,
)
from reticulate.lattice import check_distributive, check_lattice, compute_meet, compute_join
from reticulate.parser import parse
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# FrequencyBand validation
# ---------------------------------------------------------------------------

class TestFrequencyBand:
    """Tests for FrequencyBand dataclass."""

    def test_valid_band(self) -> None:
        band = FrequencyBand("low", 20.0, 200.0, "&{compress: end}")
        assert band.name == "low"
        assert band.low_hz == 20.0
        assert band.high_hz == 200.0
        assert band.bandwidth == 180.0

    def test_center_frequency(self) -> None:
        band = FrequencyBand("mid", 200.0, 2000.0, "&{eq: end}")
        # Geometric mean of 200 and 2000
        assert abs(band.center_hz - (200.0 * 2000.0) ** 0.5) < 0.01

    def test_negative_low_hz_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            FrequencyBand("bad", -10.0, 200.0, "end")

    def test_high_must_exceed_low(self) -> None:
        with pytest.raises(ValueError, match="greater than"):
            FrequencyBand("bad", 200.0, 100.0, "end")

    def test_equal_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than"):
            FrequencyBand("bad", 200.0, 200.0, "end")


# ---------------------------------------------------------------------------
# MultibandConfig validation
# ---------------------------------------------------------------------------

class TestMultibandConfig:
    """Tests for MultibandConfig dataclass."""

    def test_minimum_two_bands(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            MultibandConfig(bands=(
                FrequencyBand("only", 20.0, 20000.0, "end"),
            ))

    def test_contiguous_bands(self) -> None:
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{a: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{b: end}"),
        ))
        assert config.num_bands == 2
        assert config.frequency_range == (20.0, 20000.0)

    def test_gap_between_bands_rejected(self) -> None:
        with pytest.raises(ValueError, match="Band gap"):
            MultibandConfig(bands=(
                FrequencyBand("low", 20.0, 200.0, "&{a: end}"),
                FrequencyBand("high", 500.0, 20000.0, "&{b: end}"),
            ))

    def test_crossover_type(self) -> None:
        config = de_esser_2band()
        assert config.crossover_type == "butterworth"


# ---------------------------------------------------------------------------
# Session type construction
# ---------------------------------------------------------------------------

class TestMultibandToSessionType:
    """Tests for multiband_to_session_type."""

    def test_2band_produces_parallel(self) -> None:
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{compress: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{eq: end}"),
        ))
        st = multiband_to_session_type(config)
        assert "||" in st

    def test_labels_are_band_prefixed(self) -> None:
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{compress: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{compress: end}"),
        ))
        st = multiband_to_session_type(config)
        assert "low_compress" in st
        assert "high_compress" in st

    def test_session_type_parseable(self) -> None:
        config = de_esser_2band()
        st = multiband_to_session_type(config)
        ast = parse(st)
        assert ast is not None

    def test_4band_nested_parallel(self) -> None:
        config = standard_4band()
        st = multiband_to_session_type(config)
        # Should have 3 || operators for 4 bands
        assert st.count("||") == 3


# ---------------------------------------------------------------------------
# State space: 2-band split
# ---------------------------------------------------------------------------

class TestTwoBandStateSpace:
    """Tests for 2-band processor state space."""

    def test_de_esser_state_count(self) -> None:
        """De-esser: body has 2 states, sibilance has 3 states.
        Product = 2*3 = 6 states."""
        config = de_esser_2band()
        ss = spectral_state_space(config)
        # body: bypass->end = 2 states
        # sibilance: detect->compress->end = 3 states
        # product: 2*3 = 6 product states
        assert len(ss.states) == 6

    def test_de_esser_is_lattice(self) -> None:
        config = de_esser_2band()
        ss = spectral_state_space(config)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_simple_2band_product_size(self) -> None:
        """Two single-step bands: product should be 2x2 = 4."""
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{a: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{b: end}"),
        ))
        ss = spectral_state_space(config)
        # Each band: 2 states (entry + end). Product: 2*2 = 4 states.
        assert len(ss.states) == 4

    def test_simple_2band_transitions(self) -> None:
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{a: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{b: end}"),
        ))
        ss = spectral_state_space(config)
        labels = {lbl for _, lbl, _ in ss.transitions}
        assert "low_a" in labels
        assert "high_b" in labels


# ---------------------------------------------------------------------------
# State space: 4-band mastering
# ---------------------------------------------------------------------------

class TestFourBandStateSpace:
    """Tests for 4-band mastering processor."""

    def test_4band_is_lattice(self) -> None:
        config = standard_4band()
        ss = spectral_state_space(config)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_4band_product_size(self) -> None:
        """Each band has 4 states. Product = 4^4 = 256."""
        config = standard_4band()
        ss = spectral_state_space(config)
        # 4 bands each with 4 states: analyze->compress->makeup->end
        # Product: 4^4 = 256 states
        assert len(ss.states) == 256

    def test_4band_transitions_include_all_bands(self) -> None:
        config = standard_4band()
        ss = spectral_state_space(config)
        labels = {lbl for _, lbl, _ in ss.transitions}
        for band_name in ["sub", "lomid", "himid", "high"]:
            assert f"{band_name}_analyze" in labels
            assert f"{band_name}_compress" in labels
            assert f"{band_name}_makeup" in labels


# ---------------------------------------------------------------------------
# Different processing per band
# ---------------------------------------------------------------------------

class TestHeterogeneousBands:
    """Tests for bands with different processing chains."""

    def test_eq_vs_compressor(self) -> None:
        """Low band: EQ (2 steps), High band: compressor (3 steps)."""
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{eq_low: &{eq_high: end}}"),
            FrequencyBand("high", 200.0, 20000.0, "&{detect: &{compress: &{makeup: end}}}"),
        ))
        ss = spectral_state_space(config)
        lr = check_lattice(ss)
        assert lr.is_lattice
        # 3 * 4 = 12 product states
        assert len(ss.states) == 12

    def test_selection_band(self) -> None:
        """One band with internal choice (e.g. adaptive processing)."""
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{process: end}"),
            FrequencyBand("high", 200.0, 20000.0, "+{fast: end, slow: end}"),
        ))
        ss = spectral_state_space(config)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Spectral meet and join
# ---------------------------------------------------------------------------

class TestSpectralMeetJoin:
    """Tests for spectral_meet and spectral_join."""

    def test_meet_is_shared_content(self) -> None:
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{a: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{b: end}"),
        ))
        ss = spectral_state_space(config)
        # Top meets itself = top
        m = spectral_meet(ss, ss.top, ss.top)
        assert m == ss.top

    def test_join_is_combined_spectrum(self) -> None:
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{a: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{b: end}"),
        ))
        ss = spectral_state_space(config)
        # Bottom joins itself = bottom
        j = spectral_join(ss, ss.bottom, ss.bottom)
        assert j == ss.bottom

    def test_meet_of_top_and_bottom(self) -> None:
        config = de_esser_2band()
        ss = spectral_state_space(config)
        m = spectral_meet(ss, ss.top, ss.bottom)
        assert m == ss.bottom

    def test_join_of_top_and_bottom(self) -> None:
        config = de_esser_2band()
        ss = spectral_state_space(config)
        j = spectral_join(ss, ss.top, ss.bottom)
        assert j == ss.top

    def test_all_pairs_have_meet(self) -> None:
        config = de_esser_2band()
        ss = spectral_state_space(config)
        states = sorted(ss.states)
        for s1 in states:
            for s2 in states:
                m = spectral_meet(ss, s1, s2)
                assert m is not None, f"No meet for states {s1}, {s2}"

    def test_all_pairs_have_join(self) -> None:
        config = de_esser_2band()
        ss = spectral_state_space(config)
        states = sorted(ss.states)
        for s1 in states:
            for s2 in states:
                j = spectral_join(ss, s1, s2)
                assert j is not None, f"No join for states {s1}, {s2}"


# ---------------------------------------------------------------------------
# Distributivity
# ---------------------------------------------------------------------------

class TestDistributivity:
    """Tests for distributivity (band independence).

    Note: the reachability ordering on the product state space may NOT
    coincide with the componentwise product ordering. The product of
    chains is distributive under componentwise order, but the reachability
    poset can contain N5 sublattices when band chains have different lengths.
    Equal-length bands yield distributive lattices.
    """

    def test_equal_bands_distributive(self) -> None:
        """Equal-length bands yield a distributive product lattice."""
        config = MultibandConfig(bands=(
            FrequencyBand("low", 20.0, 200.0, "&{a: end}"),
            FrequencyBand("high", 200.0, 20000.0, "&{b: end}"),
        ))
        ss = spectral_state_space(config)
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_unequal_bands_lattice(self) -> None:
        """Unequal-length bands still form a lattice (may not be distributive)."""
        config = de_esser_2band()
        ss = spectral_state_space(config)
        lr = check_lattice(ss)
        assert lr.is_lattice

    def test_3plus_bands_product_of_chains_is_distributive(self) -> None:
        """Products of chains are always distributive lattices.

        The product of n chains under componentwise order IS distributive
        (products of distributive lattices are distributive). The state-space
        construction faithfully produces the product lattice, so the result
        is distributive regardless of the number of bands.
        """
        config = MultibandConfig(bands=(
            FrequencyBand("a", 20.0, 200.0, "&{x: end}"),
            FrequencyBand("b", 200.0, 2000.0, "&{y: end}"),
            FrequencyBand("c", 2000.0, 8000.0, "&{z: end}"),
            FrequencyBand("d", 8000.0, 20000.0, "&{w: end}"),
        ))
        ss = spectral_state_space(config)
        lr = check_lattice(ss)
        assert lr.is_lattice
        dr = check_distributive(ss)
        assert dr.is_distributive

    def test_all_presets_are_lattices(self) -> None:
        """All presets form lattices regardless of distributivity."""
        for preset_fn in [de_esser_2band, standard_4band, mastering_5band]:
            config = preset_fn()
            ss = spectral_state_space(config)
            lr = check_lattice(ss)
            assert lr.is_lattice, f"{preset_fn.__name__} is not a lattice"


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    """Tests for compression ratio via Birkhoff (irreducibles)."""

    def test_2band_compression(self) -> None:
        analysis = analyze_multiband(de_esser_2band())
        # product states > sum of individual states
        assert analysis.compression_ratio > 1.0

    def test_4band_compression_higher(self) -> None:
        """4-band has higher compression ratio than 2-band."""
        a2 = analyze_multiband(de_esser_2band())
        a4 = analyze_multiband(standard_4band())
        assert a4.compression_ratio > a2.compression_ratio

    def test_band_sizes_recorded(self) -> None:
        analysis = analyze_multiband(de_esser_2band())
        assert len(analysis.band_sizes) == 2
        assert all(s > 0 for s in analysis.band_sizes)


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

class TestAnalyzeMultiband:
    """Tests for the analyze_multiband function."""

    def test_de_esser_analysis(self) -> None:
        analysis = analyze_multiband(de_esser_2band())
        assert analysis.is_lattice
        assert analysis.width == 2
        assert analysis.num_states > 0
        assert analysis.num_transitions > 0

    def test_4band_analysis(self) -> None:
        analysis = analyze_multiband(standard_4band())
        assert analysis.is_lattice
        assert analysis.width == 4
        # 4 equal-length chains: product is distributive only for 1-step chains
        assert analysis.num_states == 256

    def test_5band_analysis(self) -> None:
        analysis = analyze_multiband(mastering_5band())
        assert analysis.is_lattice
        assert analysis.width == 5

    def test_session_type_stored(self) -> None:
        analysis = analyze_multiband(de_esser_2band())
        assert "||" in analysis.session_type


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    """Tests for preset configurations."""

    def test_standard_4band_valid(self) -> None:
        config = standard_4band()
        assert config.num_bands == 4
        assert config.frequency_range == (20.0, 20000.0)

    def test_mastering_5band_valid(self) -> None:
        config = mastering_5band()
        assert config.num_bands == 5
        assert config.frequency_range == (20.0, 20000.0)

    def test_de_esser_2band_valid(self) -> None:
        config = de_esser_2band()
        assert config.num_bands == 2
        assert config.crossover_type == "butterworth"


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

class TestBenchmarkComparison:
    """Compare multiband lattices to standard session type benchmarks."""

    def test_2band_smaller_than_smtp(self) -> None:
        """De-esser should have fewer states than a protocol with many branches."""
        # A protocol with many states (simplified mail-like protocol)
        mail_type = "&{EHLO: +{OK: &{MAIL: +{OK_MAIL: &{RCPT: +{OK_RCPT: &{DATA: end}, FAIL: end}}, FAIL: end}}, FAIL: end}, QUIT: end}"
        mail_ss = build_statespace(parse(mail_type))
        deesser = analyze_multiband(de_esser_2band())
        assert deesser.num_states < len(mail_ss.states)

    def test_4band_states_vs_oauth(self) -> None:
        """4-band mastering vs OAuth 2.0 state count."""
        oauth_type = "&{authorize: +{grant: &{token: +{access: end, deny: end}}, deny: end}}"
        oauth_ss = build_statespace(parse(oauth_type))
        mastering = analyze_multiband(standard_4band())
        # 4-band has 257 states, much larger than OAuth
        assert mastering.num_states > len(oauth_ss.states)

    def test_all_presets_are_lattices(self) -> None:
        """All preset configurations must form lattices."""
        for preset_fn in [de_esser_2band, standard_4band, mastering_5band]:
            config = preset_fn()
            analysis = analyze_multiband(config)
            assert analysis.is_lattice, f"{preset_fn.__name__} is not a lattice"
