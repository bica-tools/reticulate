"""Tests for the morphism hierarchy (morphism.py)."""

import pytest

from reticulate.parser import parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.morphism import (
    GaloisConnection,
    Morphism,
    classify_morphism,
    find_embedding,
    find_isomorphism,
    is_galois_connection,
    is_order_preserving,
    is_order_reflecting,
)


# ===================================================================
# Helpers: build small state spaces for testing
# ===================================================================

def _ss(source: str) -> StateSpace:
    """Parse and build state space."""
    return build_statespace(parse(source))


def _chain(n: int) -> StateSpace:
    """Build a chain (total order) with *n* states: 0 -> 1 -> ... -> n-1.

    Top=0, bottom=n-1.
    """
    states = set(range(n))
    transitions = [(i, f"s{i}", i + 1) for i in range(n - 1)]
    labels = {i: f"s{i}" for i in range(n)}
    return StateSpace(
        states=states,
        transitions=transitions,
        top=0,
        bottom=n - 1,
        labels=labels,
    )


def _diamond() -> StateSpace:
    """Build a 4-element diamond lattice:

        0 (top)
       / \\
      1   2
       \\ /
        3 (bottom)
    """
    return StateSpace(
        states={0, 1, 2, 3},
        transitions=[
            (0, "a", 1),
            (0, "b", 2),
            (1, "c", 3),
            (2, "d", 3),
        ],
        top=0,
        bottom=3,
        labels={0: "top", 1: "left", 2: "right", 3: "bottom"},
    )


def _two_chain() -> StateSpace:
    """2-element chain: 0 -> 1."""
    return _chain(2)


# ===================================================================
# TestOrderPreserving
# ===================================================================

class TestOrderPreserving:
    """Tests for is_order_preserving."""

    def test_identity_preserves(self) -> None:
        ss = _chain(3)
        mapping = {s: s for s in ss.states}
        assert is_order_preserving(ss, ss, mapping) is True

    def test_constant_map_fails(self) -> None:
        """Map all states to bottom — not preserving if source has non-trivial order."""
        ss = _chain(3)
        # Map everything to state 2 (bottom)
        mapping = {s: 2 for s in ss.states}
        # 0 >= 1, but f(1)=2 should be in reach(f(0))=reach(2)={2}. f(1)=2 in {2}. OK.
        # Actually constant-to-bottom IS preserving: reach(bottom) = {bottom},
        # and everything maps to bottom.
        assert is_order_preserving(ss, ss, mapping) is True

    def test_constant_to_top_fails(self) -> None:
        """Map everything to top — top reaches everything, so this preserves."""
        ss = _chain(3)
        mapping = {s: 0 for s in ss.states}
        # 0 >= 2 (bottom), but f(2)=0, and 0 in reach(f(0))=reach(0)={0,1,2}. OK.
        # This IS preserving since top reaches everything.
        assert is_order_preserving(ss, ss, mapping) is True

    def test_order_reversing_fails(self) -> None:
        """Reverse the chain: 0->2, 1->1, 2->0. Should fail preserving."""
        ss = _chain(3)
        mapping = {0: 2, 1: 1, 2: 0}
        # 0 >= 2 (i.e., 2 in reach(0)), so need f(2) in reach(f(0)).
        # f(2)=0, f(0)=2. reach(2)={2}. 0 not in {2}. Fails.
        assert is_order_preserving(ss, ss, mapping) is False

    def test_valid_homomorphism(self) -> None:
        """Chain(3) -> Chain(2) collapsing middle to top."""
        src = _chain(3)
        tgt = _chain(2)
        # 0->0, 1->0, 2->1
        mapping = {0: 0, 1: 0, 2: 1}
        # Preserving: 0>=1, f(1)=0 in reach(f(0))=reach(0)={0,1}. ok
        # 0>=2, f(2)=1 in reach(0)={0,1}. ok
        # 1>=2, f(2)=1 in reach(f(1))=reach(0)={0,1}. ok
        assert is_order_preserving(src, tgt, mapping) is True


# ===================================================================
# TestOrderReflecting
# ===================================================================

class TestOrderReflecting:
    """Tests for is_order_reflecting."""

    def test_identity_reflects(self) -> None:
        ss = _chain(3)
        mapping = {s: s for s in ss.states}
        assert is_order_reflecting(ss, ss, mapping) is True

    def test_embedding_reflects(self) -> None:
        """Chain(2) -> Chain(3): 0->0, 1->2. Should reflect."""
        src = _chain(2)
        tgt = _chain(3)
        mapping = {0: 0, 1: 2}
        # For reflecting: f(s2) in reach(f(s1)) => s2 in reach(s1)
        # f(1)=2 in reach(f(0))=reach(0)={0,1,2} -> 1 in reach(0)={0,1} ok
        # f(0)=0 in reach(f(1))=reach(2)={2}? No -> vacuously true ok
        assert is_order_reflecting(src, tgt, mapping) is True

    def test_non_reflecting_surjection(self) -> None:
        """Chain(3) -> Chain(2): 0->0, 1->0, 2->1. Not reflecting."""
        src = _chain(3)
        tgt = _chain(2)
        mapping = {0: 0, 1: 0, 2: 1}
        # f(0)=0, f(1)=0. f(1) in reach(f(0))=reach(0)={0,1}? 0 in {0,1}=yes.
        # So need: 1 in reach(0)={0,1,2}. Yes, ok.
        # f(0)=0 in reach(f(1))=reach(0)={0,1}? Yes.
        # So need: 0 in reach(1)={1,2}? No! Fails reflecting.
        assert is_order_reflecting(src, tgt, mapping) is False


# ===================================================================
# TestClassifyMorphism
# ===================================================================

class TestClassifyMorphism:
    """Tests for classify_morphism."""

    def test_identity_is_isomorphism(self) -> None:
        ss = _chain(3)
        mapping = {s: s for s in ss.states}
        m = classify_morphism(ss, ss, mapping)
        assert m.kind == "isomorphism"

    def test_injection_is_embedding(self) -> None:
        """Chain(2) -> Chain(3): 0->0, 1->2. Injective + reflecting, not surjective."""
        src = _chain(2)
        tgt = _chain(3)
        mapping = {0: 0, 1: 2}
        m = classify_morphism(src, tgt, mapping)
        assert m.kind == "embedding"

    def test_surjection_is_projection(self) -> None:
        """Chain(3) -> Chain(2): 0->0, 1->0, 2->1. Surjective, not reflecting."""
        src = _chain(3)
        tgt = _chain(2)
        mapping = {0: 0, 1: 0, 2: 1}
        m = classify_morphism(src, tgt, mapping)
        assert m.kind == "projection"

    def test_general_homomorphism(self) -> None:
        """Diamond -> Chain(2): collapse left/right to top. Preserving, not injective, not surjective in some cases."""
        src = _diamond()
        tgt = _chain(3)
        # 0->0, 1->1, 2->1, 3->2. Surjective, let's check reflecting.
        mapping = {0: 0, 1: 1, 2: 1, 3: 2}
        # Preserving: 0>=1->f(1)=1 in reach(f(0))=reach(0)={0,1,2} ok
        # 0>=2->f(2)=1 in reach(0) ok. 0>=3->f(3)=2 in reach(0) ok.
        # 1>=3->f(3)=2 in reach(f(1))=reach(1)={1,2} ok.
        # 2>=3->f(3)=2 in reach(f(2))=reach(1)={1,2} ok.
        # Reflecting: f(2)=1 in reach(f(1))=reach(1)={1,2}? yes. 2 in reach(1)? reach(1)={1,3}. 2 not in {1,3}. FAILS.
        # So it's a projection (surjective, not reflecting).
        m = classify_morphism(src, tgt, mapping)
        assert m.kind == "projection"

    def test_invalid_raises(self) -> None:
        """A non-order-preserving mapping should raise ValueError."""
        ss = _chain(3)
        mapping = {0: 2, 1: 1, 2: 0}  # reversal
        with pytest.raises(ValueError, match="not order-preserving"):
            classify_morphism(ss, ss, mapping)

    def test_morphism_fields(self) -> None:
        ss = _chain(2)
        mapping = {0: 0, 1: 1}
        m = classify_morphism(ss, ss, mapping)
        assert m.source is ss
        assert m.target is ss
        assert m.mapping == mapping
        assert isinstance(m, Morphism)


# ===================================================================
# TestFindIsomorphism
# ===================================================================

class TestFindIsomorphism:
    """Tests for find_isomorphism."""

    def test_isomorphic_chains(self) -> None:
        """Two chain(3)s built independently should be isomorphic."""
        ss1 = _ss("&{a: &{b: end}}")
        ss2 = _ss("&{x: &{y: end}}")
        m = find_isomorphism(ss1, ss2)
        assert m is not None
        assert m.kind == "isomorphism"
        assert m.mapping[ss1.top] == ss2.top
        assert m.mapping[ss1.bottom] == ss2.bottom

    def test_non_iso_different_sizes(self) -> None:
        """Different state counts -> not isomorphic."""
        ss1 = _ss("&{a: end}")
        ss2 = _ss("&{a: &{b: end}}")
        assert find_isomorphism(ss1, ss2) is None

    def test_non_iso_same_size_different_structure(self) -> None:
        """Chain(4) vs diamond — same state count, different structure."""
        chain4 = _chain(4)
        diamond = _diamond()
        assert find_isomorphism(chain4, diamond) is None

    def test_self_isomorphism(self) -> None:
        """Every state space is isomorphic to itself."""
        ss = _diamond()
        m = find_isomorphism(ss, ss)
        assert m is not None
        assert m.kind == "isomorphism"

    def test_end_iso_end(self) -> None:
        """Two 'end' state spaces are isomorphic."""
        ss1 = _ss("end")
        ss2 = _ss("end")
        m = find_isomorphism(ss1, ss2)
        assert m is not None

    def test_product_commutativity(self) -> None:
        """L(&{a: end} || &{b: end}) is isomorphic to L(&{b: end} || &{a: end})."""
        ss1 = _ss("(&{a: end} || &{b: end})")
        ss2 = _ss("(&{b: end} || &{a: end})")
        m = find_isomorphism(ss1, ss2)
        assert m is not None
        assert m.kind == "isomorphism"

    def test_different_transition_counts(self) -> None:
        """Same states but different transitions -> not isomorphic."""
        ss1 = _chain(3)
        # 3 states but extra transition
        ss2 = StateSpace(
            states={0, 1, 2},
            transitions=[(0, "a", 1), (0, "b", 2), (1, "c", 2)],
            top=0, bottom=2,
            labels={0: "top", 1: "mid", 2: "bot"},
        )
        assert find_isomorphism(ss1, ss2) is None


# ===================================================================
# TestFindEmbedding
# ===================================================================

class TestFindEmbedding:
    """Tests for find_embedding."""

    def test_chain_embeds_in_longer_chain(self) -> None:
        """Chain(2) -> Chain(3)."""
        ss1 = _chain(2)
        ss2 = _chain(3)
        m = find_embedding(ss1, ss2)
        assert m is not None
        assert m.kind in ("embedding", "isomorphism")
        # Top maps to top, bottom to bottom
        assert m.mapping[0] == 0
        assert m.mapping[1] == 2

    def test_chain_embeds_in_diamond(self) -> None:
        """Chain(2) -> diamond: top->top, bottom->bottom."""
        ss1 = _chain(2)
        ss2 = _diamond()
        m = find_embedding(ss1, ss2)
        assert m is not None
        assert m.kind == "embedding"

    def test_larger_into_smaller_fails(self) -> None:
        """Chain(4) cannot embed into Chain(3) (too many states)."""
        ss1 = _chain(4)
        ss2 = _chain(3)
        assert find_embedding(ss1, ss2) is None

    def test_embedding_is_injective(self) -> None:
        """Embedding mapping must be injective."""
        ss1 = _chain(2)
        ss2 = _chain(3)
        m = find_embedding(ss1, ss2)
        assert m is not None
        assert len(set(m.mapping.values())) == len(m.mapping)

    def test_self_embedding_is_isomorphism(self) -> None:
        """Embedding into self = isomorphism."""
        ss = _chain(3)
        m = find_embedding(ss, ss)
        assert m is not None
        assert m.kind == "isomorphism"

    def test_end_embeds_in_chain(self) -> None:
        """Single 'end' state embeds into any chain."""
        ss1 = _ss("end")
        ss2 = _chain(3)
        m = find_embedding(ss1, ss2)
        assert m is not None


# ===================================================================
# TestGaloisConnection
# ===================================================================

class TestGaloisConnection:
    """Tests for is_galois_connection."""

    def test_identity_pair(self) -> None:
        """(id, id) is a Galois connection on any state space."""
        ss = _chain(3)
        alpha = {s: s for s in ss.states}
        gamma = {s: s for s in ss.states}
        assert is_galois_connection(alpha, gamma, ss, ss) is True

    def test_invalid_pair(self) -> None:
        """An arbitrary non-adjoint pair should fail."""
        ss = _chain(3)
        # alpha sends everything to top, gamma sends everything to bottom
        alpha2 = {0: 0, 1: 0, 2: 0}  # everything to top
        gamma2 = {0: 2, 1: 2, 2: 2}  # everything to bottom
        # alpha(2)=0, y=2. alpha(2)<=y means 0 in reach(2)={2}. No.
        # x<=gamma(y): 2 in reach(gamma(2))=reach(2)={2}. Yes.
        # So alpha(2)<=y=False but x<=gamma(y)=True. NOT equivalent. Fails.
        assert is_galois_connection(alpha2, gamma2, ss, ss) is False

    def test_galois_connection_two_state(self) -> None:
        """On a 2-element chain, (id, id) is trivially a GC."""
        ss = _two_chain()
        alpha = {0: 0, 1: 1}
        gamma = {0: 0, 1: 1}
        assert is_galois_connection(alpha, gamma, ss, ss) is True

    def test_constant_bottom_top(self) -> None:
        """alpha=const bottom, gamma=const bottom: check."""
        ss = _chain(3)
        alpha = {0: 2, 1: 2, 2: 2}  # everything to bottom
        gamma = {0: 2, 1: 2, 2: 2}  # everything to bottom
        # alpha(x)<=y iff 2 in reach(y). True only when y can reach 2.
        # x<=gamma(y) iff x in reach(2)={2}. True only when x=2.
        # For x=0, y=0: alpha(0)<=0 -> 2 in reach(0)=yes. But 0 in reach(2)=no. Mismatch.
        assert is_galois_connection(alpha, gamma, ss, ss) is False


# ===================================================================
# TestHierarchy — strict separation examples
# ===================================================================

class TestHierarchy:
    """Verify the strict inclusions: iso < embed < project < homo."""

    def test_iso_implies_embed(self) -> None:
        """An isomorphism is also an embedding."""
        ss = _chain(3)
        mapping = {s: s for s in ss.states}
        m = classify_morphism(ss, ss, mapping)
        assert m.kind == "isomorphism"
        # An iso is trivially injective + reflecting = embedding conditions met

    def test_embed_not_project(self) -> None:
        """Chain(2) -> Chain(3) is an embedding but not a projection (not surjective)."""
        src = _chain(2)
        tgt = _chain(3)
        mapping = {0: 0, 1: 2}
        m = classify_morphism(src, tgt, mapping)
        assert m.kind == "embedding"
        # Not a projection because state 1 in target is not in image
        assert 1 not in set(mapping.values())

    def test_project_not_embed(self) -> None:
        """Chain(3) -> Chain(2) projecting: 0->0, 1->0, 2->1.
        Surjective but not injective or reflecting."""
        src = _chain(3)
        tgt = _chain(2)
        mapping = {0: 0, 1: 0, 2: 1}
        m = classify_morphism(src, tgt, mapping)
        assert m.kind == "projection"

    def test_homo_not_project_not_embed(self) -> None:
        """A homomorphism that is neither injective nor surjective nor reflecting.

        Diamond(4) -> Chain(4): 0->0, 1->1, 2->1, 3->3.
        Not injective (1,2->1). Not surjective (2 in target not hit).
        So it's just a homomorphism.
        """
        src = _diamond()
        tgt = _chain(4)
        mapping = {0: 0, 1: 1, 2: 1, 3: 3}
        m = classify_morphism(src, tgt, mapping)
        assert m.kind == "homomorphism"


# ===================================================================
# TestBenchmarks — self-isomorphism for all benchmarks
# ===================================================================

class TestBenchmarks:
    """Every benchmark state space is isomorphic to itself."""

    @pytest.fixture
    def benchmarks(self) -> list:
        from tests.benchmarks.protocols import BENCHMARKS
        return BENCHMARKS

    def test_self_isomorphism_all(self, benchmarks: list) -> None:
        for bp in benchmarks:
            ss = _ss(bp.type_string)
            m = find_isomorphism(ss, ss)
            assert m is not None, f"Benchmark {bp.name!r} should be self-isomorphic"
            assert m.kind == "isomorphism"

    def test_identity_is_order_preserving_all(self, benchmarks: list) -> None:
        for bp in benchmarks:
            ss = _ss(bp.type_string)
            identity = {s: s for s in ss.states}
            assert is_order_preserving(ss, ss, identity), (
                f"Identity on {bp.name!r} should be order-preserving"
            )

    def test_identity_is_order_reflecting_all(self, benchmarks: list) -> None:
        for bp in benchmarks:
            ss = _ss(bp.type_string)
            identity = {s: s for s in ss.states}
            assert is_order_reflecting(ss, ss, identity), (
                f"Identity on {bp.name!r} should be order-reflecting"
            )
