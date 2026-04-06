"""Tests for reticulate.stage_blocking (Step 58 — Stage Blocking as MPST)."""

from __future__ import annotations

import pytest

from reticulate.stage_blocking import (
    STAGE_MANAGER,
    Beat,
    BlockingMorphism,
    BlockingSnapshot,
    CueKind,
    Position,
    Scene,
    build_morphism,
    check_scene,
    classify_state,
    detect_collisions,
    encode_scene,
    macbeth_witches_scene,
    round_trip_blocking,
    verify_cue_sheet,
)
from reticulate.global_types import GEnd, GMessage, build_global_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Position / Beat / Scene construction
# ---------------------------------------------------------------------------


def test_position_rejects_unknown_code():
    with pytest.raises(ValueError):
        Position("nowhere")


def test_position_accepts_all_codes():
    for c in ["DSL", "DSC", "DSR", "USL", "USC", "USR", "CS", "OFF"]:
        assert Position(c).code == c


def test_scene_requires_actors():
    with pytest.raises(ValueError):
        Scene(title="empty", actors=(), beats=())


def test_scene_rejects_reserved_SM_name():
    with pytest.raises(ValueError):
        Scene(title="bad", actors=("SM",), beats=())


def test_scene_rejects_unknown_sender():
    with pytest.raises(ValueError):
        Scene(
            title="bad",
            actors=("A",),
            beats=(Beat("Z", "A", "x"),),
        )


def test_scene_rejects_unknown_receiver():
    with pytest.raises(ValueError):
        Scene(
            title="bad",
            actors=("A",),
            beats=(Beat("A", "Z", "x"),),
        )


def test_scene_rejects_self_loop_beat():
    with pytest.raises(ValueError):
        Scene(
            title="bad",
            actors=("A",),
            beats=(Beat("A", "A", "x"),),
        )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_encode_empty_scene_is_GEnd():
    s = Scene(title="silent", actors=("A",), beats=())
    assert isinstance(encode_scene(s), GEnd)


def test_encode_single_beat_is_single_message():
    s = Scene(
        title="one",
        actors=("A",),
        beats=(Beat(STAGE_MANAGER, "A", "cue1"),),
    )
    g = encode_scene(s)
    assert isinstance(g, GMessage)
    assert g.sender == STAGE_MANAGER
    assert g.receiver == "A"
    assert g.choices[0][0] == "cue1"
    assert isinstance(g.choices[0][1], GEnd)


def test_encode_sequence_order_preserved():
    s = Scene(
        title="two",
        actors=("A", "B"),
        beats=(
            Beat(STAGE_MANAGER, "A", "c1"),
            Beat(STAGE_MANAGER, "B", "c2"),
        ),
    )
    g = encode_scene(s)
    assert g.choices[0][0] == "c1"
    inner = g.choices[0][1]
    assert isinstance(inner, GMessage)
    assert inner.choices[0][0] == "c2"


# ---------------------------------------------------------------------------
# Well-formedness / lattice
# ---------------------------------------------------------------------------


def test_macbeth_scene_shape():
    s = macbeth_witches_scene()
    assert s.title == "Macbeth I.i"
    assert s.actors == ("W1", "W2", "W3")
    assert len(s.beats) == 10


def test_macbeth_well_formed():
    s = macbeth_witches_scene()
    wf = check_scene(s)
    assert wf.is_well_formed
    assert wf.errors == ()
    assert STAGE_MANAGER in wf.roles
    assert "W1" in wf.roles
    assert wf.global_size >= 2


def test_macbeth_global_is_lattice():
    s = macbeth_witches_scene()
    wf = check_scene(s)
    assert wf.global_is_lattice


def test_macbeth_projects_onto_every_actor():
    s = macbeth_witches_scene()
    wf = check_scene(s)
    for r in ("SM", "W1", "W2", "W3"):
        assert r in wf.local_types
        assert wf.local_types[r]


def test_empty_scene_lattice():
    s = Scene(title="silent", actors=("A",), beats=())
    wf = check_scene(s)
    assert wf.is_well_formed
    assert wf.global_is_lattice


def test_sequential_scene_is_lattice():
    s = Scene(
        title="seq",
        actors=("A", "B"),
        beats=(
            Beat(STAGE_MANAGER, "A", "c1"),
            Beat(STAGE_MANAGER, "B", "c2"),
            Beat("A", STAGE_MANAGER, "ack1"),
        ),
    )
    wf = check_scene(s)
    assert wf.is_well_formed
    assert wf.global_is_lattice


# ---------------------------------------------------------------------------
# Bidirectional morphism
# ---------------------------------------------------------------------------


def test_build_morphism_on_macbeth():
    s = macbeth_witches_scene()
    m = build_morphism(s)
    assert isinstance(m, BlockingMorphism)
    assert len(m.phi) == len(m.state_space.states)
    assert len(m.psi) >= 1


def test_morphism_initial_snapshot_all_off():
    s = macbeth_witches_scene()
    m = build_morphism(s)
    init = m.phi[m.state_space.top]
    assert init.fired_cues == ()
    for (_a, p) in init.positions:
        assert p == "OFF"


def test_morphism_round_trip():
    s = macbeth_witches_scene()
    m = build_morphism(s)
    assert round_trip_blocking(m)


def test_classify_state_returns_snapshot():
    s = macbeth_witches_scene()
    m = build_morphism(s)
    any_state = next(iter(m.state_space.states))
    sn = classify_state(m, any_state)
    assert isinstance(sn, BlockingSnapshot)


def test_snapshots_accumulate_cues():
    s = macbeth_witches_scene()
    m = build_morphism(s)
    sizes = sorted({len(sn.fired_cues) for sn in m.phi.values()})
    assert sizes[0] == 0
    assert sizes[-1] >= 1


def test_actors_on_stage_reflects_entrances():
    s = Scene(
        title="one-entry",
        actors=("A",),
        beats=(
            Beat(STAGE_MANAGER, "A", "enter", CueKind.ENTRANCE, Position("DSC")),
        ),
    )
    m = build_morphism(s)
    # Some snapshot should have A on stage.
    assert any("A" in sn.actors_on_stage() for sn in m.phi.values())


def test_exit_puts_actor_off():
    s = Scene(
        title="in-out",
        actors=("A",),
        beats=(
            Beat(STAGE_MANAGER, "A", "enter", CueKind.ENTRANCE, Position("DSC")),
            Beat(STAGE_MANAGER, "A", "exit", CueKind.EXIT),
        ),
    )
    m = build_morphism(s)
    # The last-accumulated snapshot should have A OFF.
    fullest = max(m.phi.values(), key=lambda sn: len(sn.fired_cues))
    pos_map = dict(fullest.positions)
    assert pos_map["A"] == "OFF"


# ---------------------------------------------------------------------------
# Safety: collision detection
# ---------------------------------------------------------------------------


def test_no_collisions_in_macbeth():
    s = macbeth_witches_scene()
    assert detect_collisions(s) == ()


def test_collision_detected_when_two_actors_same_position():
    s = Scene(
        title="collide",
        actors=("A", "B"),
        beats=(
            Beat(STAGE_MANAGER, "A", "eA", CueKind.ENTRANCE, Position("DSC")),
            Beat(STAGE_MANAGER, "B", "eB", CueKind.ENTRANCE, Position("DSC")),
        ),
    )
    errors = detect_collisions(s)
    assert len(errors) == 1
    assert "DSC" in errors[0]
    assert "B" in errors[0]


def test_no_collision_if_first_actor_exits_first():
    s = Scene(
        title="sequential",
        actors=("A", "B"),
        beats=(
            Beat(STAGE_MANAGER, "A", "eA", CueKind.ENTRANCE, Position("DSC")),
            Beat(STAGE_MANAGER, "A", "xA", CueKind.EXIT),
            Beat(STAGE_MANAGER, "B", "eB", CueKind.ENTRANCE, Position("DSC")),
        ),
    )
    assert detect_collisions(s) == ()


# ---------------------------------------------------------------------------
# Cue sheet verification
# ---------------------------------------------------------------------------


def test_cue_sheet_exact_match():
    s = Scene(
        title="cues",
        actors=("A",),
        beats=(
            Beat(STAGE_MANAGER, "A", "thunder", CueKind.SOUND),
            Beat(STAGE_MANAGER, "A", "lights_up", CueKind.LIGHT),
        ),
    )
    missing, extra = verify_cue_sheet(s, ["thunder", "lights_up"])
    assert missing == ()
    assert extra == ()


def test_cue_sheet_detects_missing_cue():
    s = Scene(
        title="cues",
        actors=("A",),
        beats=(Beat(STAGE_MANAGER, "A", "thunder", CueKind.SOUND),),
    )
    missing, extra = verify_cue_sheet(s, ["thunder", "lights_up"])
    assert missing == ("lights_up",)
    assert extra == ()


def test_cue_sheet_detects_extra_cue():
    s = Scene(
        title="cues",
        actors=("A",),
        beats=(
            Beat(STAGE_MANAGER, "A", "thunder", CueKind.SOUND),
            Beat(STAGE_MANAGER, "A", "lights_up", CueKind.LIGHT),
        ),
    )
    missing, extra = verify_cue_sheet(s, ["thunder"])
    assert missing == ()
    assert extra == ("lights_up",)


def test_cue_sheet_ignores_actor_to_SM_lines():
    s = Scene(
        title="lines",
        actors=("A",),
        beats=(
            Beat(STAGE_MANAGER, "A", "thunder", CueKind.SOUND),
            Beat("A", STAGE_MANAGER, "speech", CueKind.LINE),
        ),
    )
    missing, extra = verify_cue_sheet(s, ["thunder"])
    assert missing == ()
    assert extra == ()
