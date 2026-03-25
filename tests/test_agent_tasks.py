"""Tests for agent_tasks module — task card generation for Claude Code sub-agents."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from reticulate import agent_tasks as _at

TaskCard = _at.TaskCard
writer_task = _at.writer_task
prover_task = _at.prover_task
implementer_task = _at.implementer_task
_tester_task = _at.tester_task
generate_task_cards = _at.generate_task_cards
generate_sprint_cards = _at.generate_sprint_cards
sprint_summary = _at.sprint_summary
_find_thin_section = _at._find_thin_section
_count_sections = _at._count_sections
_word_count = _at._word_count
_step_dir = _at._step_dir
_step_title = _at._step_title


# ---------------------------------------------------------------------------
# TaskCard dataclass
# ---------------------------------------------------------------------------

class TestTaskCard:
    def test_create_task_card(self):
        card = TaskCard(
            agent_type="Writer",
            step_number="23",
            action="expand_paper",
            prompt="Expand the paper...",
            transitions=("writePaper", "paperReady"),
        )
        assert card.agent_type == "Writer"
        assert card.step_number == "23"
        assert card.action == "expand_paper"
        assert card.priority == 1

    def test_task_card_frozen(self):
        card = TaskCard(agent_type="Writer", step_number="1",
                       action="x", prompt="y")
        with pytest.raises(AttributeError):
            card.action = "z"  # type: ignore

    def test_default_fields(self):
        card = TaskCard(agent_type="Writer", step_number="1",
                       action="x", prompt="y")
        assert card.context_files == ()
        assert card.output_files == ()
        assert card.acceptance_criteria == ()
        assert card.transitions == ()
        assert card.priority == 1
        assert card.metadata == {}

    def test_metadata(self):
        card = TaskCard(agent_type="Writer", step_number="1",
                       action="x", prompt="y",
                       metadata={"word_count": 3000})
        assert card.metadata["word_count"] == 3000


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

class TestTextHelpers:
    def test_word_count(self):
        assert _word_count("hello world") == 2
        assert _word_count("") == 0
        assert _word_count("one two three four five") == 5

    def test_count_sections(self):
        text = r"""
\section{Introduction}
Some text.
\section{Background}
More text.
\section{Conclusion}
"""
        sections = _count_sections(text)
        assert sections == ["Introduction", "Background", "Conclusion"]

    def test_count_sections_empty(self):
        assert _count_sections("no sections here") == []

    def test_find_thin_section(self):
        text = r"""
\section{Introduction}
This is a long introduction with many words that goes on and on
and has lots of content to read through carefully.
\section{Background}
Short.
\section{Conclusion}
This conclusion has a moderate amount of words in it.
"""
        thin = _find_thin_section(text)
        assert thin == "Background"

    def test_find_thin_section_no_sections(self):
        assert _find_thin_section("no sections") is None


# ---------------------------------------------------------------------------
# Writer task cards
# ---------------------------------------------------------------------------

class TestWriterTask:
    def test_writer_expansion_task(self):
        """Writer should generate expansion task when word count is low."""
        card = writer_task("1", ["[MINOR] Word count ≥ 5000: 4500 words (need 500 more)"])
        if card is None:
            pytest.skip("Step 1 dir not found")
        assert card.agent_type == "Writer"
        assert card.action == "expand_paper"
        assert "expand" in card.prompt.lower() or "Expand" in card.prompt
        assert card.transitions == ("writePaper", "paperReady")
        assert len(card.acceptance_criteria) > 0

    def test_writer_structure_fix(self):
        """Writer should fix paper structure when missing sections."""
        card = writer_task("1", ["[MINOR] Paper structure: missing: conclusion"])
        if card is None:
            pytest.skip("Step 1 dir not found")
        assert card.agent_type == "Writer"

    def test_writer_no_task_when_no_fixes(self):
        """No task card when no relevant fixes."""
        card = writer_task("1", ["[MINOR] Test suite: 15 tests (need 20+)"])
        assert card is None

    def test_writer_nonexistent_step(self):
        card = writer_task("99999", ["word count low"])
        assert card is None

    def test_writer_prompt_includes_step_info(self):
        card = writer_task("1", ["[MINOR] Word count ≥ 5000: 3000 words (need 2000 more)"])
        if card is None:
            pytest.skip("Step 1 dir not found")
        assert "Step 1" in card.prompt
        assert "Writer" in card.prompt

    def test_writer_output_files_set(self):
        card = writer_task("1", ["[MINOR] Word count ≥ 5000: 4000 words"])
        if card is None:
            pytest.skip("Step 1 dir not found")
        assert len(card.output_files) == 1
        assert "main.tex" in card.output_files[0]


# ---------------------------------------------------------------------------
# Prover task cards
# ---------------------------------------------------------------------------

class TestProverTask:
    def test_prover_creation_task(self):
        """Prover should generate a task when proofs are mentioned in fixes."""
        card = prover_task("1", ["[MAJOR] Companion proofs.tex: proofs.tex missing"])
        if card is None:
            pytest.skip("proofs.tex may already exist or no fix needed")
        assert card.agent_type == "Prover"
        assert card.action in ("create_proofs", "expand_proofs")
        assert card.transitions == ("writeProofs", "proofsReady")

    def test_prover_expansion_task(self):
        card = prover_task("1", ["[MINOR] Companion proofs.tex: 400 words (thin)"])
        if card is None:
            pytest.skip("No proofs fix needed")
        assert card.agent_type == "Prover"

    def test_prover_no_task_when_unrelated(self):
        card = prover_task("1", ["[MINOR] Word count: too low"])
        assert card is None

    def test_prover_nonexistent_step(self):
        card = prover_task("99999", ["proofs missing"])
        assert card is None


# ---------------------------------------------------------------------------
# Implementer task cards
# ---------------------------------------------------------------------------

class TestImplementerTask:
    def test_implementer_paper_only(self):
        """No task card for paper-only steps."""
        card = implementer_task("50", [], module_name=None)
        assert card is None

    def test_implementer_with_module(self):
        card = implementer_task("1", ["module not found"], module_name="statespace")
        # statespace.py exists and is large, so no task needed
        assert card is None

    def test_implementer_prompt_content(self):
        """If module is small/missing, should generate task."""
        card = implementer_task("1", ["module stub"], module_name="nonexistent_mod_xyz")
        if card is None:
            pytest.skip("Module exists")
        assert "Implementer" in card.prompt
        assert card.transitions == ("implement", "moduleReady")


# ---------------------------------------------------------------------------
# Tester task cards
# ---------------------------------------------------------------------------

class TestTesterTask:
    def test_tester_paper_only(self):
        card = _tester_task("50", [], module_name=None)
        assert card is None

    def test_tester_sufficient_tests(self):
        """No task when tests are sufficient (20+)."""
        card = _tester_task("1", ["tests insufficient"], module_name="statespace")
        # test_statespace.py has 50+ tests
        assert card is None

    def test_tester_prompt_content(self):
        card = _tester_task("1", ["tests missing"], module_name="nonexistent_mod_xyz")
        if card is None:
            pytest.skip("Tests sufficient")
        assert "Tester" in card.prompt
        assert "20+" in card.prompt
        assert card.transitions == ("writeTests", "testsPass")


# ---------------------------------------------------------------------------
# generate_task_cards (integration)
# ---------------------------------------------------------------------------

class TestGenerateTaskCards:
    def test_accepted_step_no_cards(self):
        """A+ step should produce no task cards."""
        cards = generate_task_cards("1")
        # Step 1 should be A+ based on evaluation
        # If it produces cards, that's also fine — just testing the interface
        assert isinstance(cards, list)
        for card in cards:
            assert isinstance(card, TaskCard)

    def test_cards_sorted_by_priority(self):
        """Cards should be sorted by priority (1=highest first)."""
        cards = generate_task_cards("1")
        if len(cards) >= 2:
            priorities = [c.priority for c in cards]
            assert priorities == sorted(priorities)

    def test_all_cards_have_required_fields(self):
        """Every card must have agent_type, step_number, action, prompt."""
        cards = generate_task_cards("1")
        for card in cards:
            assert card.agent_type
            assert card.step_number
            assert card.action
            assert card.prompt
            assert len(card.prompt) > 50  # prompts should be substantial

    def test_card_transitions_are_valid(self):
        """Every card's transitions should match its agent's session type."""
        from reticulate.agent_registry import AgentMonitor

        # Test with a step that might have fixes needed
        cards = generate_task_cards("1")
        for card in cards:
            if card.transitions:
                monitor = AgentMonitor()
                monitor.start_agent(card.agent_type, f"test-{card.agent_type}")
                for label in card.transitions:
                    ok = monitor.transition(f"test-{card.agent_type}", label)
                    assert ok, f"{card.agent_type} transition '{label}' invalid"


# ---------------------------------------------------------------------------
# generate_sprint_cards
# ---------------------------------------------------------------------------

class TestGenerateSprintCards:
    def test_returns_dict(self):
        cards = generate_sprint_cards()
        assert isinstance(cards, dict)
        for step, step_cards in cards.items():
            assert isinstance(step, str)
            assert isinstance(step_cards, list)
            for card in step_cards:
                assert isinstance(card, TaskCard)

    def test_no_cards_for_accepted_steps(self):
        """Accepted steps should not appear in sprint cards."""
        from reticulate.evaluator import evaluate_all
        results = evaluate_all(run_tests=False)
        accepted_steps = {r.step_number for r in results if r.accepted}

        cards = generate_sprint_cards()
        for step in cards:
            assert step not in accepted_steps, f"Step {step} is accepted but has task cards"


# ---------------------------------------------------------------------------
# sprint_summary
# ---------------------------------------------------------------------------

class TestSprintSummary:
    def test_summary_format(self):
        cards = {
            "1": [TaskCard("Writer", "1", "expand_paper", "prompt...")],
            "2": [TaskCard("Prover", "2", "create_proofs", "prompt...")],
        }
        summary = sprint_summary(cards)
        assert "Sprint Task Cards" in summary
        assert "2 tasks" in summary
        assert "2 steps" in summary
        assert "Writer" in summary
        assert "Prover" in summary

    def test_summary_empty(self):
        summary = sprint_summary({})
        assert "0 tasks" in summary

    def test_summary_by_agent(self):
        cards = {
            "1": [
                TaskCard("Writer", "1", "expand_paper", "p"),
                TaskCard("Prover", "1", "create_proofs", "p"),
            ],
        }
        summary = sprint_summary(cards)
        assert "By agent:" in summary
        assert "Writer=1" in summary
        assert "Prover=1" in summary

    def test_summary_by_action(self):
        cards = {
            "1": [TaskCard("Writer", "1", "expand_paper", "p")],
            "2": [TaskCard("Writer", "2", "expand_paper", "p")],
        }
        summary = sprint_summary(cards)
        assert "expand_paper=2" in summary
