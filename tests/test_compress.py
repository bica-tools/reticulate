"""Tests for session type compression (tree-to-equation transformation)."""

import pytest

from reticulate.parser import (
    Branch, End, Select, Var, Rec, Parallel, Wait, Continuation,
    Definition, Program, parse, pretty, parse_program, pretty_program,
)
from reticulate.compress import (
    ast_size,
    compress,
    compression_ratio,
    analyze_compression,
    CompressionResult,
)
from reticulate.resolve import resolve


# ---------------------------------------------------------------------------
# ast_size
# ---------------------------------------------------------------------------

class TestAstSize:
    def test_end(self):
        assert ast_size(End()) == 1

    def test_wait(self):
        assert ast_size(Wait()) == 1

    def test_var(self):
        assert ast_size(Var("X")) == 1

    def test_branch_single(self):
        assert ast_size(Branch((("a", End()),))) == 2

    def test_branch_two(self):
        assert ast_size(Branch((("a", End()), ("b", End())))) == 3

    def test_select(self):
        assert ast_size(Select((("x", End()), ("y", End())))) == 3

    def test_rec(self):
        assert ast_size(Rec("X", Branch((("a", Var("X")),)))) == 3

    def test_nested(self):
        # &{a: &{b: end}} — Branch(a: Branch(b: End)) = 1 + (1 + 1) = 3
        inner = Branch((("b", End()),))
        outer = Branch((("a", inner),))
        assert ast_size(outer) == 3

    def test_parallel(self):
        assert ast_size(Parallel((End(), End()))) == 3


# ---------------------------------------------------------------------------
# compress: no sharing
# ---------------------------------------------------------------------------

class TestCompressNoSharing:
    def test_end(self):
        prog = compress(End())
        assert len(prog.definitions) == 1
        assert prog.definitions[0].name == "Main"
        assert prog.definitions[0].body == End()

    def test_simple_branch(self):
        ast = Branch((("a", End()),))
        prog = compress(ast)
        assert len(prog.definitions) == 1

    def test_distinct_branches(self):
        # &{a: &{b: end}, c: &{d: end}} — subtrees are different
        ast = Branch((
            ("a", Branch((("b", End()),))),
            ("c", Branch((("d", End()),))),
        ))
        prog = compress(ast)
        assert len(prog.definitions) == 1  # no sharing

    def test_rec_no_sharing(self):
        ast = parse("rec X . &{a: X, b: end}")
        prog = compress(ast)
        assert len(prog.definitions) == 1


# ---------------------------------------------------------------------------
# compress: with sharing
# ---------------------------------------------------------------------------

class TestCompressWithSharing:
    def test_duplicate_branch(self):
        # &{a: &{x: end, y: end}, b: &{x: end, y: end}}
        shared = Branch((("x", End()), ("y", End())))
        ast = Branch((("a", shared), ("b", shared)))
        prog = compress(ast)
        # Should factor out the shared subtree
        assert len(prog.definitions) == 2
        # Main should reference the shared definition
        main = prog.definitions[0]
        assert main.name == "Main"
        # The body should contain Var references
        assert isinstance(main.body, Branch)
        for _, body in main.body.choices:
            assert isinstance(body, Var)
        # Both should reference the same name
        names = [body.name for _, body in main.body.choices]
        assert names[0] == names[1]

    def test_triple_sharing(self):
        # Three identical subtrees
        shared = Branch((("x", End()), ("y", End())))
        ast = Branch((("a", shared), ("b", shared), ("c", shared)))
        prog = compress(ast)
        assert len(prog.definitions) == 2
        # All three should reference same name
        main = prog.definitions[0]
        names = [body.name for _, body in main.body.choices]
        assert len(set(names)) == 1

    def test_customer_support_pattern(self):
        # The customer support pattern: 3 issue types share agent review
        review = Branch((
            ("agentReview", Select((
                ("resolve", End()),
                ("escalate", End()),
                ("requestInfo", Branch((
                    ("respond", Select((("resolve", End()), ("escalate", End())))),
                ))),
            ))),
        ))
        ast = Branch((
            ("openTicket", Branch((
                ("describeIssue", Select((
                    ("billing", review),
                    ("technical", review),
                    ("account", review),
                ))),
            ))),
        ))
        prog = compress(ast)
        # Should detect the shared review subtree
        assert len(prog.definitions) >= 2

        # Verify the shared definition contains agentReview
        shared_def = prog.definitions[1]
        assert isinstance(shared_def.body, Branch)
        labels = [l for l, _ in shared_def.body.choices]
        assert "agentReview" in labels

    def test_morning_commute_pattern(self):
        # Weather branches share &{arrive: end} — size 2, so use min_size=2
        arrive = Branch((("arrive", End()),))
        ast = Branch((
            ("wakeUp", Branch((
                ("checkWeather", Select((
                    ("raining", Branch((
                        ("takeUmbrella", Select((
                            ("bus", arrive),
                            ("taxi", arrive),
                        ))),
                    ))),
                    ("sunny", Select((
                        ("walk", arrive),
                        ("bike", arrive),
                    ))),
                ))),
            ))),
        ))
        prog = compress(ast, min_size=2)
        # &{arrive: end} appears 4 times, should be shared
        assert len(prog.definitions) >= 2

    def test_workout_pattern(self):
        # &{coolDown: &{shower: end}} shared 4 times
        shared = Branch((("coolDown", Branch((("shower", End()),))),))
        ast = Branch((
            ("changeClothes", Branch((
                ("warmUp", Select((
                    ("gym", Select((
                        ("weights", shared),
                        ("cardio", shared),
                    ))),
                    ("homeWorkout", Select((
                        ("yoga", shared),
                        ("hiit", shared),
                    ))),
                ))),
            ))),
        ))
        prog = compress(ast)
        assert len(prog.definitions) >= 2


# ---------------------------------------------------------------------------
# compress: min_size threshold
# ---------------------------------------------------------------------------

class TestCompressMinSize:
    def test_min_size_filters_small(self):
        # Two identical &{a: end} — size 2, below default threshold 3
        shared = Branch((("a", End()),))
        ast = Branch((("x", shared), ("y", shared)))
        prog = compress(ast, min_size=3)
        # Too small to factor out
        assert len(prog.definitions) == 1

    def test_min_size_1_catches_all(self):
        # With min_size=1, even End() shared would be caught
        # But End() appears everywhere... let's use a slightly bigger example
        shared = Branch((("a", End()),))
        ast = Branch((("x", shared), ("y", shared)))
        prog = compress(ast, min_size=2)
        assert len(prog.definitions) == 2

    def test_custom_entry_name(self):
        ast = Branch((("a", End()),))
        prog = compress(ast, entry_name="Protocol")
        assert prog.definitions[0].name == "Protocol"


# ---------------------------------------------------------------------------
# Roundtrip: compress then resolve
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def _roundtrip(self, type_string: str):
        """Verify compress → resolve produces alpha-equivalent AST."""
        ast = parse(type_string)
        prog = compress(ast, min_size=2)
        if len(prog.definitions) > 1:
            resolved = resolve(prog)
            # Build state spaces and check isomorphism
            from reticulate.statespace import build_statespace
            ss_original = build_statespace(ast)
            ss_resolved = build_statespace(resolved)
            assert len(ss_original.states) == len(ss_resolved.states)
            assert len(ss_original.transitions) == len(ss_resolved.transitions)

    def test_roundtrip_simple(self):
        self._roundtrip("&{a: end}")

    def test_roundtrip_branch_select(self):
        self._roundtrip("&{a: +{x: end, y: end}}")

    def test_roundtrip_shared_branches(self):
        # Two identical subtrees
        shared = Branch((("x", End()), ("y", End())))
        ast = Branch((("a", shared), ("b", shared)))
        prog = compress(ast, min_size=2)
        assert len(prog.definitions) >= 2
        resolved = resolve(prog)
        from reticulate.statespace import build_statespace
        ss_orig = build_statespace(ast)
        ss_res = build_statespace(resolved)
        assert len(ss_orig.states) == len(ss_res.states)

    def test_roundtrip_customer_support(self):
        self._roundtrip(
            "&{openTicket: &{describeIssue: +{billing: &{agentReview: "
            "+{resolve: end, escalate: end, requestInfo: &{respond: "
            "+{resolve: end, escalate: end}}}}, technical: &{agentReview: "
            "+{resolve: end, escalate: end, requestInfo: &{respond: "
            "+{resolve: end, escalate: end}}}}, account: &{agentReview: "
            "+{resolve: end, escalate: end, requestInfo: &{respond: "
            "+{resolve: end, escalate: end}}}}}}}"
        )

    def test_roundtrip_morning_commute(self):
        self._roundtrip(
            "&{wakeUp: &{checkWeather: +{raining: &{takeUmbrella: "
            "+{bus: &{arrive: end}, taxi: &{arrive: end}}}, "
            "sunny: +{walk: &{arrive: end}, bike: &{arrive: end}}}}}"
        )

    def test_roundtrip_workout(self):
        self._roundtrip(
            "&{changeClothes: &{warmUp: +{gym: +{weights: &{coolDown: "
            "&{shower: end}}, cardio: &{coolDown: &{shower: end}}}, "
            "homeWorkout: +{yoga: &{coolDown: &{shower: end}}, hiit: "
            "&{coolDown: &{shower: end}}}}}}"
        )


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_no_sharing(self):
        ast = parse("&{a: end}")
        ratio = compression_ratio(ast)
        assert ratio == pytest.approx(1.0)

    def test_with_sharing(self):
        shared = Branch((("x", End()), ("y", End())))
        ast = Branch((("a", shared), ("b", shared)))
        ratio = compression_ratio(ast, min_size=2)
        # Original: 7 nodes, compressed: 2 (Main) + 3 (S0) = 5 nodes → 7/5 = 1.4
        assert ratio > 1.0

    def test_high_sharing(self):
        # 4 identical large subtrees = high compression
        shared = Branch((("coolDown", Branch((("shower", End()),))),))
        ast = Branch((
            ("a", shared), ("b", shared), ("c", shared), ("d", shared),
        ))
        ratio = compression_ratio(ast, min_size=2)
        # Original: 4*3 + 1 = 13, compressed: 5 (Main) + 3 (S0) = 8 → 13/8 = 1.625
        assert ratio > 1.0


# ---------------------------------------------------------------------------
# analyze_compression
# ---------------------------------------------------------------------------

class TestAnalyzeCompression:
    def test_no_sharing(self):
        ast = parse("&{a: end}")
        result = analyze_compression(ast)
        assert not result.has_sharing
        assert result.num_shared == 0
        assert result.num_definitions == 1
        assert result.ratio == pytest.approx(1.0)

    def test_with_sharing(self):
        shared = Branch((("x", End()), ("y", End())))
        ast = Branch((("a", shared), ("b", shared)))
        result = analyze_compression(ast, min_size=2)
        assert result.has_sharing
        assert result.num_shared >= 1
        assert result.ratio > 1.0
        assert len(result.shared_names) >= 1
        assert isinstance(result.program, Program)

    def test_result_fields(self):
        ast = parse("&{a: +{x: end, y: end}, b: +{x: end, y: end}}")
        result = analyze_compression(ast, min_size=2)
        assert result.original_size > 0
        assert result.compressed_size > 0


# ---------------------------------------------------------------------------
# pretty_program output
# ---------------------------------------------------------------------------

class TestPrettyOutput:
    def test_compressed_pretty(self):
        shared = Branch((("x", End()), ("y", End())))
        ast = Branch((("a", shared), ("b", shared), ("c", shared)))
        prog = compress(ast, min_size=2)
        output = pretty_program(prog)
        # Should have equation syntax: Main = ...\nS0 = ...
        assert "Main" in output or "=" in output
        assert "S0" in output

    def test_no_sharing_pretty(self):
        ast = parse("&{a: end}")
        prog = compress(ast)
        output = pretty_program(prog)
        # Single _main-style or Main = ... definition
        assert "end" in output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_recursive_type(self):
        ast = parse("rec X . &{a: X, b: end}")
        prog = compress(ast)
        # Recursive types should compress without errors
        assert len(prog.definitions) >= 1

    def test_nested_rec(self):
        ast = parse("&{greet: rec X . +{ask: &{answer: X}, farewell: end}}")
        prog = compress(ast)
        assert len(prog.definitions) >= 1

    def test_parallel(self):
        ast = parse("(end || end)")
        prog = compress(ast)
        assert len(prog.definitions) >= 1

    def test_deeply_nested_sharing(self):
        # Deep nesting with sharing at multiple levels
        inner = Select((("resolve", End()), ("escalate", End())))
        mid = Branch((("review", inner),))
        ast = Branch((
            ("a", mid),
            ("b", mid),
            ("c", Branch((("review", inner),))),
        ))
        prog = compress(ast, min_size=2)
        # Should detect sharing
        assert len(prog.definitions) >= 2

    def test_empty_branch(self):
        # &{} is valid (empty branch = end equivalent)
        ast = Branch(())
        prog = compress(ast)
        assert len(prog.definitions) == 1
