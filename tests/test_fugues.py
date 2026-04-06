"""Tests for reticulate.fugues (Step 56 — Bach Fugues as MPST)."""

from __future__ import annotations

import pytest

from reticulate.fugues import (
    FugueSection,
    FugueStructure,
    SubjectEntry,
    bwv846_fugue,
    build_morphism,
    check_fugue,
    classify_state,
    encode_fugue,
    has_stretto,
    round_trip_sections,
    section_representative,
)
from reticulate.global_types import GEnd, GMessage, build_global_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# FugueStructure construction
# ---------------------------------------------------------------------------


def test_fugue_requires_voices():
    with pytest.raises(ValueError):
        FugueStructure(voices=(), exposition=(), episodes=0, final_entry=None)


def test_fugue_rejects_unknown_voice():
    with pytest.raises(ValueError):
        FugueStructure(
            voices=("S", "A"),
            exposition=(SubjectEntry(voice="Z"),),
        )


def test_fugue_rejects_unknown_final_voice():
    with pytest.raises(ValueError):
        FugueStructure(
            voices=("S", "A"),
            exposition=(SubjectEntry(voice="S"),),
            final_entry=SubjectEntry(voice="Q"),
        )


def test_bwv846_fugue_shape():
    f = bwv846_fugue()
    assert f.voices == ("S", "A", "T", "B")
    assert len(f.exposition) == 4
    assert f.episodes == 2
    assert f.final_entry is not None
    assert f.final_entry.voice == "B"


# ---------------------------------------------------------------------------
# encode_fugue
# ---------------------------------------------------------------------------


def test_encode_fugue_returns_global_type():
    f = bwv846_fugue()
    g = encode_fugue(f)
    assert isinstance(g, GMessage)


def test_encode_fugue_no_self_messages():
    f = bwv846_fugue()
    g = encode_fugue(f)
    node = g
    while isinstance(node, GMessage):
        assert node.sender != node.receiver, (
            f"self-message {node.sender}->{node.receiver} rejected"
        )
        assert len(node.choices) == 1
        node = node.choices[0][1]
    assert isinstance(node, GEnd)


def test_encode_fugue_minimal_two_voice():
    f = FugueStructure(
        voices=("X", "Y"),
        exposition=(
            SubjectEntry(voice="X"),
            SubjectEntry(voice="Y", cue="X"),
        ),
    )
    g = encode_fugue(f)
    assert isinstance(g, GMessage)
    # Two entries -> two messages -> GEnd
    count = 0
    node = g
    while isinstance(node, GMessage):
        count += 1
        node = node.choices[0][1]
    assert count == 2
    assert isinstance(node, GEnd)


def test_encode_fugue_with_episodes():
    f = FugueStructure(
        voices=("X", "Y"),
        exposition=(SubjectEntry(voice="X"), SubjectEntry(voice="Y", cue="X")),
        episodes=3,
    )
    g = encode_fugue(f)
    count = 0
    node = g
    while isinstance(node, GMessage):
        count += 1
        node = node.choices[0][1]
    assert count == 5  # 2 entries + 3 episodes


def test_encode_fugue_with_final_entry():
    f = FugueStructure(
        voices=("X", "Y"),
        exposition=(SubjectEntry(voice="X"), SubjectEntry(voice="Y", cue="X")),
        episodes=1,
        final_entry=SubjectEntry(voice="X"),
    )
    g = encode_fugue(f)
    # 2 + 1 + 1 = 4 messages
    count = 0
    node = g
    while isinstance(node, GMessage):
        count += 1
        node = node.choices[0][1]
    assert count == 4


# ---------------------------------------------------------------------------
# Well-formedness
# ---------------------------------------------------------------------------


def test_check_fugue_bwv846_wellformed():
    r = check_fugue(bwv846_fugue())
    assert r.is_well_formed
    assert r.errors == ()
    assert set(r.local_types.keys()) == {"S", "A", "T", "B"}
    assert r.global_size > 0


def test_check_fugue_bwv846_is_lattice():
    r = check_fugue(bwv846_fugue())
    assert r.global_is_lattice


def test_check_fugue_local_types_nonempty():
    r = check_fugue(bwv846_fugue())
    for v, lt in r.local_types.items():
        assert lt, f"empty local type for {v}"
        assert "end" in lt


def test_check_fugue_two_voice_minimal():
    f = FugueStructure(
        voices=("X", "Y"),
        exposition=(SubjectEntry(voice="X"), SubjectEntry(voice="Y", cue="X")),
    )
    r = check_fugue(f)
    assert r.is_well_formed
    assert r.global_is_lattice


# ---------------------------------------------------------------------------
# Morphism φ, ψ
# ---------------------------------------------------------------------------


def test_build_morphism_total():
    f = bwv846_fugue()
    m = build_morphism(f)
    # phi is total over all reachable + unreachable states
    assert set(m.phi.keys()) == set(m.state_space.states)


def test_build_morphism_sections_covered():
    f = bwv846_fugue()
    m = build_morphism(f)
    covered = set(m.phi.values())
    assert FugueSection.EXPOSITION in covered
    assert FugueSection.EPISODE in covered
    assert FugueSection.FINAL_ENTRY in covered


def test_build_morphism_roundtrip():
    f = bwv846_fugue()
    m = build_morphism(f)
    assert round_trip_sections(m)


def test_classify_state_and_representative():
    f = bwv846_fugue()
    m = build_morphism(f)
    for sec in [FugueSection.EXPOSITION, FugueSection.EPISODE, FugueSection.FINAL_ENTRY]:
        rep = section_representative(m, sec)
        assert classify_state(m, rep) == sec


def test_section_representative_missing_raises():
    f = FugueStructure(
        voices=("X", "Y"),
        exposition=(SubjectEntry(voice="X"), SubjectEntry(voice="Y", cue="X")),
    )
    m = build_morphism(f)
    # No episodes, no final entry -> STRETTO absent
    with pytest.raises(KeyError):
        section_representative(m, FugueSection.STRETTO)


def test_morphism_phi_partitions_states():
    f = bwv846_fugue()
    m = build_morphism(f)
    # Every state has exactly one section (phi is a function)
    for s in m.state_space.states:
        assert s in m.phi


# ---------------------------------------------------------------------------
# Stretto
# ---------------------------------------------------------------------------


def test_has_stretto_false_for_bwv846():
    # BWV 846 exposition is a clean A,S,T,B sequence.
    assert not has_stretto(bwv846_fugue())


def test_has_stretto_true_for_overlap():
    f = FugueStructure(
        voices=("S", "A", "T", "B"),
        exposition=(
            SubjectEntry(voice="A"),
            SubjectEntry(voice="A", cue="A"),  # A re-enters before S/T/B
            SubjectEntry(voice="S", cue="A"),
            SubjectEntry(voice="T", cue="S"),
        ),
    )
    assert has_stretto(f)


# ---------------------------------------------------------------------------
# Projection consistency with direct GlobalType construction
# ---------------------------------------------------------------------------


def test_global_state_space_lattice_for_bwv846():
    g = encode_fugue(bwv846_fugue())
    ss = build_global_statespace(g)
    assert check_lattice(ss).is_lattice


def test_global_state_space_size_matches_messages():
    f = bwv846_fugue()
    g = encode_fugue(f)
    ss = build_global_statespace(g)
    # 4 exposition + 2 episodes + 1 final = 7 messages, 8 states
    assert len(ss.states) == 8
