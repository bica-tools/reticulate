"""Extract session types from Python source code (Step 97b).

Uses Python's ``ast`` module to statically extract method call sequences
on typed objects, then feeds them to the type inference engine.

Two extraction modes:
1. **AST analysis** — Parse Python source, find method calls on target objects,
   extract call sequences as traces.
2. **Protocol specs** — Manually specify known protocol patterns for standard
   library types (sqlite3.Connection, http.client, file objects, etc.)
   and validate them against real usage.

Scientific Method:
  Observe   — Python stdlib has stateful objects (files, connections, cursors)
  Question  — Do real Python stateful protocols produce distributive lattices?
  Hypothesis — Sequential/stateful protocols (unlike REST) ARE distributive
  Predict   — ≥80% of stateful Python protocols form distributive lattices
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Any

from reticulate.type_inference import Trace, TraceStep, InferenceResult, infer_from_traces
from reticulate.parser import parse, pretty, SessionType
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceExtractionResult:
    """Result of extracting a session type from Python source or protocol spec."""
    name: str
    source: str  # "ast" or "spec"
    traces: list[list[str]]  # raw method name sequences
    inferred_type: str
    num_states: int
    num_transitions: int
    is_lattice: bool
    is_distributive: bool
    is_modular: bool
    classification: str  # "boolean", "distributive", "modular", "non-modular", "non-lattice"


# ---------------------------------------------------------------------------
# Known Python protocol specifications
# ---------------------------------------------------------------------------

# Each protocol is a dict: name -> list of valid method call traces.
# These are extracted from Python documentation and real-world usage patterns.

PYTHON_PROTOCOLS: dict[str, dict[str, Any]] = {
    "file_object": {
        "description": "Python file object (open/read/write/close)",
        "traces": [
            ["open", "read", "close"],
            ["open", "readline", "readline", "close"],
            ["open", "write", "write", "flush", "close"],
            ["open", "read", "seek", "read", "close"],
            ["open", "close"],
            ["open", "write", "close"],
            ["open", "readlines", "close"],
            ["open", "writelines", "close"],
        ],
    },
    "sqlite3_connection": {
        "description": "sqlite3.Connection lifecycle",
        "traces": [
            ["connect", "cursor", "execute", "fetchall", "close"],
            ["connect", "cursor", "execute", "fetchone", "close"],
            ["connect", "cursor", "execute", "commit", "close"],
            ["connect", "cursor", "executemany", "commit", "close"],
            ["connect", "execute", "fetchall", "close"],
            ["connect", "cursor", "execute", "fetchmany", "close"],
            ["connect", "cursor", "execute", "rollback", "close"],
            ["connect", "close"],
        ],
    },
    "http_client": {
        "description": "http.client.HTTPConnection lifecycle",
        "traces": [
            ["connect", "request", "getresponse", "read", "close"],
            ["connect", "request", "getresponse", "close"],
            ["connect", "request", "getresponse", "read", "read", "close"],
            ["connect", "close"],
            ["connect", "request", "getresponse", "getheaders", "close"],
        ],
    },
    "socket_object": {
        "description": "socket.socket lifecycle",
        "traces": [
            ["create", "bind", "listen", "accept", "recv", "send", "close"],
            ["create", "connect", "send", "recv", "close"],
            ["create", "connect", "send", "recv", "send", "recv", "close"],
            ["create", "bind", "listen", "accept", "close"],
            ["create", "close"],
        ],
    },
    "threading_lock": {
        "description": "threading.Lock acquire/release cycle",
        "traces": [
            ["acquire", "release"],
            ["acquire", "release", "acquire", "release"],
            ["acquire", "release", "acquire", "release", "acquire", "release"],
        ],
    },
    "zipfile_reader": {
        "description": "zipfile.ZipFile read lifecycle",
        "traces": [
            ["open", "namelist", "read", "close"],
            ["open", "namelist", "extractall", "close"],
            ["open", "infolist", "read", "close"],
            ["open", "read", "close"],
            ["open", "close"],
        ],
    },
    "csv_reader": {
        "description": "csv.reader iteration lifecycle",
        "traces": [
            ["open", "reader", "next", "next", "next", "close"],
            ["open", "reader", "next", "close"],
            ["open", "reader", "close"],
        ],
    },
    "context_manager": {
        "description": "Generic context manager (__enter__/__exit__)",
        "traces": [
            ["enter", "use", "exit"],
            ["enter", "use", "use", "exit"],
            ["enter", "exit"],
            ["enter", "use", "use", "use", "exit"],
        ],
    },
    "iterator_protocol": {
        "description": "Python iterator (__iter__/__next__/StopIteration)",
        "traces": [
            ["iter", "next", "next", "next", "stop"],
            ["iter", "next", "stop"],
            ["iter", "stop"],
            ["iter", "next", "next", "stop"],
        ],
    },
    "queue_protocol": {
        "description": "queue.Queue put/get lifecycle",
        "traces": [
            ["put", "put", "get", "get"],
            ["put", "get"],
            ["put", "put", "put", "get", "get", "get"],
            ["put", "get", "put", "get"],
        ],
    },
}


# ---------------------------------------------------------------------------
# Java protocol specifications (Step 97c)
# ---------------------------------------------------------------------------

JAVA_PROTOCOLS: dict[str, dict[str, Any]] = {
    "java_iterator": {
        "description": "java.util.Iterator<E> protocol",
        "traces": [
            ["hasNext", "next", "hasNext", "next", "hasNext"],
            ["hasNext", "next", "hasNext"],
            ["hasNext"],
            ["hasNext", "next", "remove", "hasNext", "next", "hasNext"],
            ["hasNext", "next", "hasNext", "next", "remove", "hasNext"],
        ],
    },
    "jdbc_connection": {
        "description": "java.sql.Connection lifecycle",
        "traces": [
            ["createStatement", "executeQuery", "next", "getString", "close", "close"],
            ["createStatement", "executeUpdate", "close", "commit", "close"],
            ["createStatement", "executeQuery", "next", "close", "close"],
            ["prepareStatement", "setString", "executeQuery", "next", "close", "close"],
            ["createStatement", "executeUpdate", "close", "rollback", "close"],
            ["setAutoCommit", "createStatement", "executeUpdate", "close", "commit", "close"],
            ["close"],
        ],
    },
    "java_inputstream": {
        "description": "java.io.InputStream lifecycle",
        "traces": [
            ["read", "read", "read", "close"],
            ["read", "close"],
            ["available", "read", "close"],
            ["skip", "read", "close"],
            ["mark", "read", "read", "reset", "read", "close"],
            ["close"],
        ],
    },
    "java_socket": {
        "description": "java.net.Socket lifecycle",
        "traces": [
            ["connect", "getInputStream", "getOutputStream", "close"],
            ["connect", "getOutputStream", "close"],
            ["connect", "getInputStream", "close"],
            ["connect", "setSoTimeout", "getInputStream", "close"],
            ["close"],
        ],
    },
    "java_httpurlconnection": {
        "description": "java.net.HttpURLConnection lifecycle",
        "traces": [
            ["setRequestMethod", "connect", "getResponseCode", "getInputStream", "disconnect"],
            ["setRequestMethod", "setDoOutput", "connect", "getOutputStream", "getResponseCode", "disconnect"],
            ["setRequestMethod", "connect", "getResponseCode", "disconnect"],
            ["setRequestMethod", "addRequestProperty", "connect", "getResponseCode", "getInputStream", "disconnect"],
            ["connect", "getResponseCode", "disconnect"],
        ],
    },
    "java_servlet": {
        "description": "javax.servlet.http.HttpServlet lifecycle",
        "traces": [
            ["init", "service", "doGet", "destroy"],
            ["init", "service", "doPost", "destroy"],
            ["init", "service", "doPut", "destroy"],
            ["init", "service", "doDelete", "destroy"],
            ["init", "service", "doGet", "service", "doPost", "destroy"],
            ["init", "destroy"],
        ],
    },
    "java_outputstream": {
        "description": "java.io.OutputStream lifecycle",
        "traces": [
            ["write", "write", "flush", "close"],
            ["write", "close"],
            ["write", "write", "write", "flush", "close"],
            ["flush", "close"],
            ["close"],
        ],
    },
    "java_lock": {
        "description": "java.util.concurrent.locks.Lock protocol",
        "traces": [
            ["lock", "unlock"],
            ["lock", "unlock", "lock", "unlock"],
            ["tryLock", "unlock"],
            ["lockInterruptibly", "unlock"],
            ["lock", "unlock", "lock", "unlock", "lock", "unlock"],
        ],
    },
    "java_executorservice": {
        "description": "java.util.concurrent.ExecutorService lifecycle",
        "traces": [
            ["submit", "submit", "shutdown", "awaitTermination"],
            ["submit", "shutdown", "awaitTermination"],
            ["execute", "execute", "shutdown", "awaitTermination"],
            ["submit", "invokeAll", "shutdown", "awaitTermination"],
            ["shutdown", "awaitTermination"],
            ["shutdownNow"],
        ],
    },
    "java_bufferedreader": {
        "description": "java.io.BufferedReader lifecycle",
        "traces": [
            ["readLine", "readLine", "readLine", "close"],
            ["readLine", "close"],
            ["read", "read", "close"],
            ["ready", "readLine", "close"],
            ["close"],
        ],
    },
}


# ---------------------------------------------------------------------------
# AST-based extraction from Python source
# ---------------------------------------------------------------------------

class MethodCallExtractor(ast.NodeVisitor):
    """Extract method call sequences on a target variable from Python AST.

    Walks the AST looking for attribute calls (e.g., ``conn.execute()``)
    on a specified variable name, collecting them in order.
    """

    def __init__(self, target_var: str) -> None:
        self.target_var = target_var
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            # Check if the call is on our target variable
            value = node.func.value
            if isinstance(value, ast.Name) and value.id == self.target_var:
                self.calls.append(node.func.attr)
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                # Handle chained calls like cursor().execute()
                pass
        self.generic_visit(node)


def extract_from_source(source: str, target_var: str) -> list[str]:
    """Extract method call sequence on target_var from Python source code.

    Parameters
    ----------
    source : str
        Python source code (can be a snippet).
    target_var : str
        Variable name to track (e.g., "conn", "f", "sock").

    Returns
    -------
    list[str]
        Ordered list of method names called on target_var.
    """
    tree = ast.parse(textwrap.dedent(source))
    extractor = MethodCallExtractor(target_var)
    extractor.visit(tree)
    return extractor.calls


def extract_traces_from_source(source: str, target_var: str) -> list[Trace]:
    """Extract method traces from source, treating each function as a trace."""
    tree = ast.parse(textwrap.dedent(source))
    traces: list[Trace] = []

    # Each function body is a separate trace
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            extractor = MethodCallExtractor(target_var)
            extractor.visit(node)
            if extractor.calls:
                traces.append(Trace.from_labels(extractor.calls))

    # If no functions, treat the whole source as one trace
    if not traces:
        calls = extract_from_source(source, target_var)
        if calls:
            traces.append(Trace.from_labels(calls))

    return traces


# ---------------------------------------------------------------------------
# Protocol analysis
# ---------------------------------------------------------------------------

def analyze_protocol(name: str, raw_traces: list[list[str]]) -> SourceExtractionResult:
    """Analyze a protocol given as raw method call traces.

    Infers a session type, builds state space, checks lattice properties.
    """
    traces = [Trace.from_labels(t) for t in raw_traces]
    inferred_ast = infer_from_traces(traces)
    st_str = pretty(inferred_ast)

    # Build state space from inferred type
    parsed = parse(st_str)
    ss = build_statespace(parsed)
    lr = check_lattice(ss)

    is_dist = False
    is_mod = False
    classification = "non-lattice"

    if lr.is_lattice:
        try:
            dr = check_distributive(ss)
            is_dist = dr.is_distributive
            is_mod = dr.is_modular
            classification = dr.classification
        except Exception:
            classification = "lattice"

    return SourceExtractionResult(
        name=name,
        source="spec",
        traces=raw_traces,
        inferred_type=st_str,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_lattice=lr.is_lattice,
        is_distributive=is_dist,
        is_modular=is_mod,
        classification=classification,
    )


def analyze_all_protocols() -> list[SourceExtractionResult]:
    """Analyze all known protocol specifications (Python + Java)."""
    results: list[SourceExtractionResult] = []
    for name, spec in PYTHON_PROTOCOLS.items():
        result = analyze_protocol(name, spec["traces"])
        results.append(result)
    for name, spec in JAVA_PROTOCOLS.items():
        result = analyze_protocol(name, spec["traces"])
        results.append(result)
    return results


def analyze_java_protocols() -> list[SourceExtractionResult]:
    """Analyze all known Java protocol specifications."""
    results: list[SourceExtractionResult] = []
    for name, spec in JAVA_PROTOCOLS.items():
        result = analyze_protocol(name, spec["traces"])
        results.append(result)
    return results


def print_protocol_report(results: list[SourceExtractionResult]) -> str:
    """Format a protocol analysis report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  PYTHON PROTOCOL SESSION TYPE EXTRACTION REPORT")
    lines.append("=" * 78)

    total = len(results)
    lattice_count = sum(1 for r in results if r.is_lattice)
    dist_count = sum(1 for r in results if r.is_distributive)
    mod_count = sum(1 for r in results if r.is_modular)

    lines.append(f"  Protocols analyzed: {total}")
    lines.append(f"  Lattice rate: {lattice_count}/{total} ({lattice_count/total:.0%})")
    lines.append(f"  Distributive rate: {dist_count}/{total} ({dist_count/total:.0%})")
    lines.append(f"  Modular rate: {mod_count}/{total} ({mod_count/total:.0%})")
    lines.append("")
    lines.append(f"  {'Protocol':<22} {'States':>6} {'Trans':>6} "
                 f"{'Lattice':>8} {'Distrib':>8} {'Class':>14}")
    lines.append("  " + "-" * 72)

    for r in results:
        lat = "YES" if r.is_lattice else "NO"
        dist = "YES" if r.is_distributive else "NO"
        lines.append(f"  {r.name:<22} {r.num_states:>6} {r.num_transitions:>6} "
                     f"{lat:>8} {dist:>8} {r.classification:>14}")

    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real Python source code samples for AST extraction
# ---------------------------------------------------------------------------

PYTHON_SOURCE_SAMPLES: dict[str, dict[str, str]] = {
    "sqlite3_usage": {
        "target_var": "conn",
        "source": '''
import sqlite3

def create_user(conn, name, email):
    conn.execute("INSERT INTO users VALUES (?, ?)", (name, email))
    conn.commit()

def get_users(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    return cursor.fetchall()

def delete_user(conn, user_id):
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()

def transaction_example(conn):
    conn.execute("BEGIN")
    conn.execute("INSERT INTO log VALUES (?)", ("entry",))
    conn.commit()
    conn.close()
''',
    },
    "file_operations": {
        "target_var": "f",
        "source": '''
def read_config(f):
    f.read()
    f.close()

def write_log(f):
    f.write("entry1")
    f.write("entry2")
    f.flush()
    f.close()

def process_lines(f):
    f.readline()
    f.readline()
    f.seek(0)
    f.readline()
    f.close()
''',
    },
    "http_requests": {
        "target_var": "conn",
        "source": '''
import http.client

def fetch_page(conn):
    conn.request("GET", "/index.html")
    conn.getresponse()
    conn.close()

def post_data(conn):
    conn.request("POST", "/api/data")
    conn.getresponse()
    conn.close()

def fetch_with_headers(conn):
    conn.request("GET", "/api/info")
    conn.getresponse()
    conn.close()
''',
    },
}
