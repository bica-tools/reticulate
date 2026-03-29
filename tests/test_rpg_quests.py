"""Tests for rpg_quests module (Step 59)."""

from __future__ import annotations

import pytest

from reticulate.rpg_quests import (
    CompletabilityResult,
    Quest,
    QuestChainResult,
    QuestGraph,
    QuestLatticeResult,
    branching_narrative_graph,
    branching_quests,
    check_completability,
    classic_rpg_graph,
    encode_quest_chain,
    encode_quest_graph,
    open_world_graph,
    quest_dependency_order,
    quest_lattice,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Quest data type tests
# ---------------------------------------------------------------------------

class TestQuest:
    def test_creation(self) -> None:
        q = Quest("defeat_boss", "Defeat the final boss", "main",
                  ("enter_dungeon",), "magic_sword", 500, 10)
        assert q.name == "defeat_boss"
        assert q.quest_type == "main"
        assert q.prerequisites == ("enter_dungeon",)
        assert q.xp == 500

    def test_frozen(self) -> None:
        q = Quest("test")
        with pytest.raises(AttributeError):
            q.name = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        q = Quest("test")
        assert q.quest_type == "main"
        assert q.prerequisites == ()
        assert q.is_optional is False


class TestQuestGraph:
    def test_quest_map(self) -> None:
        g = QuestGraph("Test", (Quest("a"), Quest("b")))
        assert "a" in g.quest_map
        assert "b" in g.quest_map


# ---------------------------------------------------------------------------
# Dependency order tests
# ---------------------------------------------------------------------------

class TestQuestDependencyOrder:
    def test_linear(self) -> None:
        g = QuestGraph("Linear", (
            Quest("a"),
            Quest("b", prerequisites=("a",)),
            Quest("c", prerequisites=("b",)),
        ))
        order = quest_dependency_order(g)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_no_deps(self) -> None:
        g = QuestGraph("Free", (Quest("a"), Quest("b"), Quest("c")))
        order = quest_dependency_order(g)
        assert len(order) == 3

    def test_cycle_raises(self) -> None:
        g = QuestGraph("Cycle", (
            Quest("a", prerequisites=("b",)),
            Quest("b", prerequisites=("a",)),
        ))
        with pytest.raises(ValueError, match="Cycle"):
            quest_dependency_order(g)

    def test_diamond(self) -> None:
        g = QuestGraph("Diamond", (
            Quest("start"),
            Quest("left", prerequisites=("start",)),
            Quest("right", prerequisites=("start",)),
            Quest("end_q", prerequisites=("left", "right")),
        ))
        order = quest_dependency_order(g)
        assert order.index("start") < order.index("left")
        assert order.index("start") < order.index("right")
        assert order.index("left") < order.index("end_q")
        assert order.index("right") < order.index("end_q")


# ---------------------------------------------------------------------------
# Completability tests
# ---------------------------------------------------------------------------

class TestCheckCompletability:
    def test_completable(self) -> None:
        g = QuestGraph("Simple", (
            Quest("a"),
            Quest("b", prerequisites=("a",)),
        ))
        result = check_completability(g)
        assert result.is_completable
        assert len(result.completion_order) == 2

    def test_missing_prereq(self) -> None:
        g = QuestGraph("Missing", (
            Quest("b", prerequisites=("nonexistent",)),
        ))
        result = check_completability(g)
        assert not result.is_completable
        assert "b" in result.unreachable_quests

    def test_cycle_detected(self) -> None:
        g = QuestGraph("Cycle", (
            Quest("a", prerequisites=("b",)),
            Quest("b", prerequisites=("a",)),
        ))
        result = check_completability(g)
        assert not result.is_completable
        assert result.cycle_detected


# ---------------------------------------------------------------------------
# Encoding tests
# ---------------------------------------------------------------------------

class TestEncodeQuestChain:
    def test_linear_chain(self) -> None:
        quests = (Quest("prologue"), Quest("chapter1"), Quest("finale"))
        result = encode_quest_chain(quests)
        assert isinstance(result, QuestChainResult)
        assert result.chain_length == 3
        assert "prologue" in result.session_type_str

    def test_is_lattice(self) -> None:
        quests = (Quest("a"), Quest("b"))
        result = encode_quest_chain(quests)
        assert result.is_lattice

    def test_state_count(self) -> None:
        quests = (Quest("a"), Quest("b"), Quest("c"))
        result = encode_quest_chain(quests)
        assert result.state_count == 4  # 3 quests + end


class TestEncodeQuestGraph:
    def test_simple_graph(self) -> None:
        g = QuestGraph("Simple", (
            Quest("start"),
            Quest("mid", prerequisites=("start",)),
            Quest("end_q", prerequisites=("mid",)),
        ))
        result = encode_quest_graph(g)
        st = parse(result)
        ss = build_statespace(st)
        assert len(ss.states) > 1

    def test_with_side_quests(self) -> None:
        g = QuestGraph("WithSide", (
            Quest("main1"),
            Quest("side1", prerequisites=("main1",), is_optional=True),
            Quest("main2", prerequisites=("main1",)),
        ))
        result = encode_quest_graph(g)
        assert "main1" in result or "main2" in result

    def test_cycle_raises(self) -> None:
        g = QuestGraph("Cycle", (
            Quest("a", prerequisites=("b",)),
            Quest("b", prerequisites=("a",)),
        ))
        with pytest.raises(ValueError, match="cycles"):
            encode_quest_graph(g)

    def test_parseable(self) -> None:
        g = classic_rpg_graph()
        result = encode_quest_graph(g)
        st = parse(result)
        ss = build_statespace(st)
        lr = check_lattice(ss)
        assert lr.is_lattice


# ---------------------------------------------------------------------------
# Quest lattice tests
# ---------------------------------------------------------------------------

class TestQuestLattice:
    def test_classic_rpg(self) -> None:
        g = classic_rpg_graph()
        result = quest_lattice(g)
        assert isinstance(result, QuestLatticeResult)
        assert result.num_quests == 8
        assert result.is_lattice
        assert result.num_main > 0

    def test_open_world(self) -> None:
        g = open_world_graph()
        result = quest_lattice(g)
        assert result.is_lattice
        assert result.num_side > 0

    def test_branching_factor(self) -> None:
        g = classic_rpg_graph()
        result = quest_lattice(g)
        assert result.branching_factor >= 0


# ---------------------------------------------------------------------------
# Branching quests tests
# ---------------------------------------------------------------------------

class TestBranchingQuests:
    def test_main_only(self) -> None:
        result = branching_quests("boss_fight", ())
        assert "boss_fight" in result

    def test_with_side_quests(self) -> None:
        result = branching_quests("boss_fight", ("side_a", "side_b"))
        assert "boss_fight" in result
        assert "side_a" in result

    def test_with_continuation(self) -> None:
        result = branching_quests("boss_fight", (), "ending")
        assert "ending" in result

    def test_parseable(self) -> None:
        result = branching_quests("main", ("side1", "side2"))
        st = parse(result)
        ss = build_statespace(st)
        assert len(ss.states) > 1


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestClassicRpg:
    def test_structure(self) -> None:
        g = classic_rpg_graph()
        assert g.title == "Classic RPG"
        assert len(g.quests) == 8

    def test_has_prerequisites(self) -> None:
        g = classic_rpg_graph()
        boss = g.quest_map["boss_fight"]
        assert "dungeon_1" in boss.prerequisites


class TestBranchingNarrative:
    def test_exclusive_quests(self) -> None:
        g = branching_narrative_graph()
        rebels = g.quest_map["join_rebels"]
        assert "join_empire" in rebels.exclusive_with


class TestOpenWorld:
    def test_optional_quests(self) -> None:
        g = open_world_graph()
        optional = [q for q in g.quests if q.is_optional]
        assert len(optional) > 0
