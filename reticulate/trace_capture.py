"""Capture real execution traces from Python programs (Step 80c).

Wraps real Python objects to record method calls and return values,
producing traces suitable for conformance checking against L3 session types.

Usage:
    from reticulate.trace_capture import TracingProxy, run_and_capture

    # Wrap a real object
    f = TracingProxy(open("data.txt", "r"), name="file")
    content = f.read()    # recorded: ("read", "data" or "EOF")
    f.close()             # recorded: ("close",)
    print(f.get_trace())  # ["read", "data", "close"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import traceback


# ---------------------------------------------------------------------------
# Trace recording
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    """A single entry in a captured trace."""
    method: str
    args_summary: str  # abbreviated args for debugging
    return_value: Any
    selection_label: Optional[str]  # mapped return value (TRUE/FALSE/OK/ERR etc.)
    exception: Optional[str]  # exception type if raised


@dataclass
class CapturedTrace:
    """A complete captured trace from a real execution."""
    object_name: str
    object_type: str
    entries: list[TraceEntry] = field(default_factory=list)
    complete: bool = False  # True if session ended normally

    def as_labels(self) -> list[str]:
        """Convert to label list for conformance checking.

        Interleaves method calls with selection outcomes.
        """
        labels: list[str] = []
        for entry in self.entries:
            labels.append(entry.method)
            if entry.selection_label:
                labels.append(entry.selection_label)
        return labels

    def as_method_only(self) -> list[str]:
        """Return just the method names (L2 trace format)."""
        return [e.method for e in self.entries]


# ---------------------------------------------------------------------------
# Return value → selection label mapping
# ---------------------------------------------------------------------------

# Default mappings for common return types
DEFAULT_MAPPINGS: dict[str, Callable[[Any], str]] = {
    "bool": lambda v: "TRUE" if v else "FALSE",
    "none_check": lambda v: "NULL" if v is None else "VALUE",
    "http_status": lambda v: f"s{v}" if isinstance(v, int) else str(v),
}


def bool_selection(value: Any) -> Optional[str]:
    """Map boolean return to TRUE/FALSE selection."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return None


def none_selection(value: Any) -> Optional[str]:
    """Map None/not-None to NULL/VALUE selection."""
    if value is None:
        return "NULL"
    return "VALUE"


def eof_selection(value: Any) -> Optional[str]:
    """Map empty read to EOF, non-empty to data."""
    if value == b"" or value == "" or value is None:
        return "EOF"
    return "data"


def readline_selection(value: Any) -> Optional[str]:
    """Map readline empty string to EOF."""
    if value == "" or value == b"":
        return "EOF"
    return "data"


def fetchone_selection(value: Any) -> Optional[str]:
    """Map fetchone None to NONE, otherwise ROW."""
    if value is None:
        return "NONE"
    return "ROW"


# ---------------------------------------------------------------------------
# Tracing proxy
# ---------------------------------------------------------------------------

class TracingProxy:
    """Wraps a real Python object, recording method calls and returns.

    Parameters
    ----------
    target : object
        The real object to wrap.
    name : str
        Name for the trace (e.g., "file", "conn").
    selection_map : dict
        Maps method names to functions that convert return values
        to selection labels. E.g., {"read": eof_selection}
    """

    def __init__(self, target: Any, name: str = "obj",
                 selection_map: dict[str, Callable] | None = None) -> None:
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_selection_map', selection_map or {})
        object.__setattr__(self, '_trace', CapturedTrace(
            object_name=name,
            object_type=type(target).__name__,
        ))

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, '_target')
        attr = getattr(target, name)

        if not callable(attr):
            return attr

        trace = object.__getattribute__(self, '_trace')
        sel_map = object.__getattribute__(self, '_selection_map')

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            args_str = ", ".join(repr(a)[:30] for a in args[:3])
            try:
                result = attr(*args, **kwargs)
                sel_label = None
                if name in sel_map:
                    sel_label = sel_map[name](result)

                trace.entries.append(TraceEntry(
                    method=name,
                    args_summary=args_str,
                    return_value=result,
                    selection_label=sel_label,
                    exception=None,
                ))

                # Check if this was a "close" method
                if name in ("close", "disconnect", "shutdown", "quit", "destroy"):
                    trace.complete = True

                return result
            except Exception as e:
                trace.entries.append(TraceEntry(
                    method=name,
                    args_summary=args_str,
                    return_value=None,
                    selection_label="EXCEPTION",
                    exception=type(e).__name__,
                ))
                raise

        return wrapper

    def get_trace(self) -> CapturedTrace:
        return object.__getattribute__(self, '_trace')


# ---------------------------------------------------------------------------
# Pre-configured proxies for common Python types
# ---------------------------------------------------------------------------

def trace_file(f: Any, name: str = "file") -> TracingProxy:
    """Wrap a file object with read→data/EOF selection mapping."""
    return TracingProxy(f, name=name, selection_map={
        "read": eof_selection,
        "readline": readline_selection,
        "readlines": lambda v: "EOF" if not v else "data",
    })


def trace_cursor(cursor: Any, name: str = "cursor") -> TracingProxy:
    """Wrap a DB cursor with fetchone→ROW/NONE selection mapping."""
    return TracingProxy(cursor, name=name, selection_map={
        "fetchone": fetchone_selection,
    })


def trace_iterator(it: Any, name: str = "iter") -> TracingProxy:
    """Wrap an iterator. Note: __next__ raising StopIteration becomes selection."""
    # Iterators need special handling for StopIteration
    return TracingProxy(it, name=name)


# ---------------------------------------------------------------------------
# Capture from real Python operations
# ---------------------------------------------------------------------------

def capture_file_read(path: str) -> CapturedTrace:
    """Capture a real file read operation."""
    f = trace_file(open(path, "r"), name="file")
    try:
        while True:
            chunk = f.read(1024)
            if not chunk:
                break
    finally:
        f.close()
    return f.get_trace()


def capture_file_write(path: str, content: str) -> CapturedTrace:
    """Capture a real file write operation."""
    f = trace_file(open(path, "w"), name="file")
    try:
        f.write(content)
        f.flush()
    finally:
        f.close()
    return f.get_trace()


def capture_sqlite_query(db_path: str, query: str) -> CapturedTrace:
    """Capture a real SQLite query execution."""
    import sqlite3
    conn = TracingProxy(sqlite3.connect(db_path), name="conn", selection_map={})
    try:
        cursor = conn.cursor()
        cursor_proxy = TracingProxy(cursor, name="cursor", selection_map={
            "fetchone": fetchone_selection,
        })
        cursor_proxy.execute(query)
        while True:
            row = cursor_proxy.fetchone()
            if row is None:
                break
    finally:
        conn.close()
    return conn.get_trace()


def capture_iterator(iterable: Any) -> CapturedTrace:
    """Capture iteration over any iterable."""
    trace = CapturedTrace(object_name="iter", object_type=type(iterable).__name__)
    it = iter(iterable)
    while True:
        try:
            value = next(it)
            trace.entries.append(TraceEntry(
                method="next",
                args_summary="",
                return_value=value,
                selection_label="value",
                exception=None,
            ))
        except StopIteration:
            trace.entries.append(TraceEntry(
                method="next",
                args_summary="",
                return_value=None,
                selection_label="StopIteration",
                exception="StopIteration",
            ))
            trace.complete = True
            break
    return trace
