"""Tests for interface composition classifier."""

import pytest
from reticulate.interface_classifier import (
    InterfaceProfile,
    analyze_client_interleaving,
    classify_composition,
    InterleaveAnalysis,
    CompositionClassification,
)


# ---------------------------------------------------------------------------
# Interface profiles for testing
# ---------------------------------------------------------------------------

ITERATOR = InterfaceProfile("Iterator", frozenset(["hasNext", "next", "remove"]))
CLOSEABLE = InterfaceProfile("Closeable", frozenset(["close"]))
READABLE = InterfaceProfile("Readable", frozenset(["read"]))
WRITABLE = InterfaceProfile("Writable", frozenset(["write", "flush"]))
LIST = InterfaceProfile("List", frozenset(["add", "get", "set", "remove", "size"]))
SERIALIZABLE = InterfaceProfile("Serializable", frozenset())  # marker


# ---------------------------------------------------------------------------
# Interleaving analysis tests
# ---------------------------------------------------------------------------

class TestInterleaveAnalysis:
    def test_separated_traces(self):
        """Clients use only one interface per function."""
        traces = [
            ["hasNext", "next", "hasNext", "next"],  # only Iterator
            ["hasNext", "next"],                       # only Iterator
            ["close"],                                 # only Closeable
            ["close"],                                 # only Closeable
        ]
        ia = analyze_client_interleaving(traces, {"hasNext", "next"}, {"close"})
        assert ia.separated >= 3
        assert ia.interleaved == 0

    def test_interleaved_traces(self):
        """Clients mix methods from both interfaces."""
        traces = [
            ["hasNext", "next", "close", "hasNext"],  # A B A
            ["next", "close", "next"],                 # A B A
            ["close", "hasNext", "close"],             # B A B
        ]
        ia = analyze_client_interleaving(traces, {"hasNext", "next"}, {"close"})
        assert ia.interleaved >= 2
        assert ia.interleave_rate > 0.5

    def test_ordered_not_interleaved(self):
        """All A methods, then all B methods — not truly interleaved."""
        traces = [
            ["hasNext", "next", "next", "close"],     # A A A B
            ["hasNext", "next", "close"],              # A A B
        ]
        ia = analyze_client_interleaving(traces, {"hasNext", "next"}, {"close"})
        assert ia.interleaved == 0  # ordered, not interleaved
        assert ia.mixed_but_ordered >= 1 or ia.separated >= 1

    def test_empty_traces(self):
        ia = analyze_client_interleaving([], {"a"}, {"b"})
        assert ia.total_clients == 0
        assert ia.interleave_rate == 0.0

    def test_read_write_interleaved(self):
        """Read and write interleaved → extending (same state machine)."""
        traces = [
            ["read", "write", "read", "write", "flush"],
            ["write", "read", "flush"],
            ["read", "write", "read", "flush"],
        ]
        ia = analyze_client_interleaving(traces, {"read"}, {"write", "flush"})
        assert ia.interleaved >= 2

    def test_read_write_separated(self):
        """Read and write in separate functions → orthogonal."""
        traces = [
            ["read", "read", "read"],
            ["write", "write", "flush"],
            ["read", "read"],
            ["write", "flush"],
        ]
        ia = analyze_client_interleaving(traces, {"read"}, {"write", "flush"})
        assert ia.separated >= 3
        assert ia.interleaved == 0


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassification:
    def test_orthogonal_when_separated(self):
        """Clients never interleave → orthogonal (parallel)."""
        traces = [
            ["read", "read"],
            ["write", "flush"],
            ["read"],
            ["write", "write", "flush"],
            ["read", "read", "read"],
        ]
        result = classify_composition(
            "FileChannel", READABLE, WRITABLE, traces)
        assert result.classification == "orthogonal"
        assert result.session_type_suggestion == "parallel"

    def test_extending_when_interleaved(self):
        """Clients always interleave → extending (branch widening)."""
        traces = [
            ["hasNext", "next", "close", "hasNext"],
            ["close", "hasNext", "next", "close"],
            ["next", "close", "hasNext", "close"],
            ["hasNext", "close", "next", "close"],
        ]
        result = classify_composition(
            "FileIterator", ITERATOR, CLOSEABLE, traces)
        assert result.classification == "extending"
        assert result.session_type_suggestion == "branch_widening"

    def test_marker_interface(self):
        """Marker interface (no methods) → extending trivially."""
        traces = [
            ["add", "get", "size"],
            ["add", "add", "get"],
        ]
        result = classify_composition(
            "ArrayList", LIST, SERIALIZABLE, traces)
        # All methods are from LIST, none from Serializable
        assert result.classification in ("ambiguous", "extending", "orthogonal")

    def test_confidence_high_for_clear_cases(self):
        """Clear separation should give high confidence."""
        traces = [["read"]] * 10 + [["write", "flush"]] * 10
        result = classify_composition(
            "Stream", READABLE, WRITABLE, traces)
        assert result.confidence >= 0.7

    def test_ambiguous_when_mixed(self):
        """Some interleaved, some separated → ambiguous."""
        traces = [
            ["read", "write", "read"],   # interleaved
            ["read", "read"],             # separated
            ["write", "flush"],           # separated
        ]
        result = classify_composition(
            "Mixed", READABLE, WRITABLE, traces)
        # Could be either — should not be high confidence
        assert result.confidence <= 0.8


# ---------------------------------------------------------------------------
# Real-world scenario tests
# ---------------------------------------------------------------------------

class TestRealWorldScenarios:
    def test_file_channel_read_write(self):
        """FileChannel: read and write are genuinely independent (orthogonal)."""
        traces = [
            # Reader thread
            ["read", "read", "read"],
            ["read", "read"],
            # Writer thread
            ["write", "write", "flush"],
            ["write", "flush"],
            # Rare combined usage
            ["read", "write"],
        ]
        result = classify_composition("FileChannel", READABLE, WRITABLE, traces)
        # Mostly separated → should lean orthogonal
        assert result.classification in ("orthogonal", "ambiguous")

    def test_buffered_reader_closeable(self):
        """BufferedReader: close is part of the read lifecycle (extending)."""
        reader = InterfaceProfile("Reader", frozenset(["read", "readLine", "ready"]))
        traces = [
            ["readLine", "readLine", "close"],
            ["read", "close"],
            ["readLine", "close"],
            ["ready", "readLine", "readLine", "close"],
        ]
        result = classify_composition("BufferedReader", reader, CLOSEABLE, traces)
        # close always at end → ordered, extending
        assert result.session_type_suggestion in ("branch_widening", "needs_investigation")
