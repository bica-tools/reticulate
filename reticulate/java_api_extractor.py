"""Extract session types from Java API class definitions.

Given Java source files (or curated API specifications), extract the
lifecycle protocol as a session type.  Two extraction strategies:

1. **Source-based**: Parse Java source with ``javalang``, extract public
   method signatures from a target class, infer lifecycle from method
   names and common patterns (open/close, init/destroy, acquire/release).

2. **Spec-based**: Use curated API lifecycle specifications derived from
   Javadoc and JDK documentation, with representative usage traces.

The extraction pipeline:
  Java API → method signatures + lifecycle traces → type inference
  → session type AST → state space → lattice check → validation table

This module targets the 5 Java APIs for TOPLAS validation (Week 1):
  1. java.io.InputStream
  2. java.sql.Connection
  3. java.util.Iterator
  4. javax.net.ssl.SSLSocket
  5. java.nio.channels.SocketChannel
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import javalang
    from javalang.tree import (
        ClassDeclaration,
        InterfaceDeclaration,
        MethodDeclaration,
    )
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False

from reticulate.type_inference import Trace, infer_from_traces
from reticulate.parser import parse, pretty, SessionType
from reticulate.statespace import build_statespace, StateSpace
from reticulate.lattice import check_lattice, check_distributive, LatticeResult


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JavaMethodInfo:
    """Information about a Java method extracted from source."""
    name: str
    return_type: str
    parameters: list[str]
    is_public: bool
    is_static: bool
    throws: list[str]
    line_number: int


@dataclass(frozen=True)
class JavaAPIProfile:
    """Profile of a Java API class: its methods and lifecycle."""
    class_name: str
    package: str
    methods: list[JavaMethodInfo]
    lifecycle_phases: list[str]  # e.g., ["init", "use", "close"]
    source: str  # "javalang" or "spec"


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting and validating a session type from a Java API."""
    api_name: str
    description: str
    traces: list[list[str]]
    inferred_type: str
    session_type_ast: Optional[SessionType]
    num_states: int
    num_transitions: int
    is_lattice: bool
    is_distributive: bool
    is_modular: bool
    classification: str
    method_count: int
    trace_count: int
    source: str  # "javalang", "spec", "hybrid"


# ---------------------------------------------------------------------------
# Java API Specifications (from JDK Javadoc + real usage patterns)
# ---------------------------------------------------------------------------

JAVA_API_SPECS: dict[str, dict[str, Any]] = {
    "java.io.InputStream": {
        "description": "Abstract byte input stream (JDK core I/O)",
        "package": "java.io",
        "class": "InputStream",
        "methods": [
            "available", "read", "skip", "close",
            "mark", "reset", "markSupported",
        ],
        "lifecycle_phases": ["query", "read", "close"],
        # Return type annotations: method → possible outcomes
        # read() returns int: -1 on EOF, ≥0 on data
        # available() returns int (≥0)
        # markSupported() returns boolean
        "return_types": {
            "read": ["DATA", "EOF"],
            "available": ["READY"],
            "markSupported": ["TRUE", "FALSE"],
            "skip": ["SKIPPED"],
            "mark": ["MARKED"],
            "reset": ["RESTORED"],
            "close": [],  # void — no selection
        },
        # Directed traces: (method, "receive") for calls,
        #                   (outcome, "send") for return values
        "directed_traces": [
            # Read loop: read returns DATA, DATA, ..., EOF
            [("read", "r"), ("DATA", "s"), ("read", "r"), ("DATA", "s"), ("read", "r"), ("EOF", "s"), ("close", "r")],
            # Single read then EOF
            [("read", "r"), ("EOF", "s"), ("close", "r")],
            # Available check then read
            [("available", "r"), ("READY", "s"), ("read", "r"), ("DATA", "s"), ("read", "r"), ("EOF", "s"), ("close", "r")],
            # Skip then read
            [("skip", "r"), ("SKIPPED", "s"), ("read", "r"), ("DATA", "s"), ("read", "r"), ("EOF", "s"), ("close", "r")],
            # Mark/reset cycle
            [("markSupported", "r"), ("TRUE", "s"), ("mark", "r"), ("MARKED", "s"), ("read", "r"), ("DATA", "s"), ("reset", "r"), ("RESTORED", "s"), ("read", "r"), ("DATA", "s"), ("read", "r"), ("EOF", "s"), ("close", "r")],
            # markSupported returns false
            [("markSupported", "r"), ("FALSE", "s"), ("read", "r"), ("DATA", "s"), ("read", "r"), ("EOF", "s"), ("close", "r")],
            # Direct close (empty stream)
            [("close", "r")],
            # Read single byte
            [("read", "r"), ("DATA", "s"), ("close", "r")],
        ],
    },

    "java.sql.Connection": {
        "description": "JDBC database connection (relational DB access)",
        "package": "java.sql",
        "class": "Connection",
        "methods": [
            "createStatement", "prepareStatement", "prepareCall",
            "setAutoCommit", "commit", "rollback", "close",
            "getMetaData", "setReadOnly", "setTransactionIsolation",
        ],
        "lifecycle_phases": ["configure", "transact", "close"],
        # commit/rollback can succeed or throw SQLException
        "return_types": {
            "createStatement": ["STMT"],
            "prepareStatement": ["PSTMT"],
            "prepareCall": ["CSTMT"],
            "commit": ["OK", "SQL_ERROR"],
            "rollback": ["OK"],
            "setAutoCommit": ["CONFIGURED"],
            "getMetaData": ["METADATA"],
            "setReadOnly": ["CONFIGURED"],
            "setTransactionIsolation": ["CONFIGURED"],
            "close": [],
        },
        "directed_traces": [
            # Simple query + commit success
            [("createStatement", "r"), ("STMT", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Prepared statement + commit
            [("prepareStatement", "r"), ("PSTMT", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Transaction with rollback
            [("setAutoCommit", "r"), ("CONFIGURED", "s"), ("createStatement", "r"), ("STMT", "s"), ("rollback", "r"), ("OK", "s"), ("close", "r")],
            # Commit fails → rollback → close
            [("createStatement", "r"), ("STMT", "s"), ("commit", "r"), ("SQL_ERROR", "s"), ("rollback", "r"), ("OK", "s"), ("close", "r")],
            # Multiple statements + commit
            [("setAutoCommit", "r"), ("CONFIGURED", "s"), ("createStatement", "r"), ("STMT", "s"), ("createStatement", "r"), ("STMT", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Read-only query
            [("setReadOnly", "r"), ("CONFIGURED", "s"), ("createStatement", "r"), ("STMT", "s"), ("close", "r")],
            # Metadata inspection
            [("getMetaData", "r"), ("METADATA", "s"), ("close", "r")],
            # Set isolation + query + commit
            [("setTransactionIsolation", "r"), ("CONFIGURED", "s"), ("createStatement", "r"), ("STMT", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Just close
            [("close", "r")],
        ],
    },

    "java.util.Iterator": {
        "description": "Collection iterator (Gang of Four pattern)",
        "package": "java.util",
        "class": "Iterator",
        "methods": ["hasNext", "next", "remove"],
        "lifecycle_phases": ["check", "advance", "optional_remove"],
        # hasNext() returns boolean — THE canonical selection
        "return_types": {
            "hasNext": ["TRUE", "FALSE"],
            "next": ["ELEMENT"],
            "remove": ["REMOVED"],
        },
        "directed_traces": [
            # Standard iteration: check → true → get → check → true → get → check → false
            [("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("hasNext", "r"), ("FALSE", "s")],
            # Single element
            [("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("hasNext", "r"), ("FALSE", "s")],
            # Empty iterator
            [("hasNext", "r"), ("FALSE", "s")],
            # With remove
            [("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("remove", "r"), ("REMOVED", "s"), ("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("hasNext", "r"), ("FALSE", "s")],
            # Three elements
            [("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("hasNext", "r"), ("TRUE", "s"), ("next", "r"), ("ELEMENT", "s"), ("hasNext", "r"), ("FALSE", "s")],
        ],
    },

    "javax.net.ssl.SSLSocket": {
        "description": "SSL/TLS socket (secure network communication)",
        "package": "javax.net.ssl",
        "class": "SSLSocket",
        "methods": [
            "connect", "startHandshake", "getSession",
            "getInputStream", "getOutputStream",
            "setEnabledProtocols", "setEnabledCipherSuites",
            "close",
        ],
        "lifecycle_phases": ["configure", "handshake", "io", "close"],
        # startHandshake can succeed or fail (SSLHandshakeException)
        "return_types": {
            "connect": ["CONNECTED"],
            "startHandshake": ["OK", "HANDSHAKE_ERROR"],
            "getSession": ["SESSION"],
            "getInputStream": ["INPUT"],
            "getOutputStream": ["OUTPUT"],
            "setEnabledProtocols": ["CONFIGURED"],
            "setEnabledCipherSuites": ["CONFIGURED"],
            "close": [],
        },
        "directed_traces": [
            # Standard TLS: connect → handshake OK → get I/O → close
            [("connect", "r"), ("CONNECTED", "s"), ("startHandshake", "r"), ("OK", "s"), ("getInputStream", "r"), ("INPUT", "s"), ("getOutputStream", "r"), ("OUTPUT", "s"), ("close", "r")],
            # Handshake fails → close
            [("connect", "r"), ("CONNECTED", "s"), ("startHandshake", "r"), ("HANDSHAKE_ERROR", "s"), ("close", "r")],
            # Configure before handshake
            [("connect", "r"), ("CONNECTED", "s"), ("setEnabledProtocols", "r"), ("CONFIGURED", "s"), ("setEnabledCipherSuites", "r"), ("CONFIGURED", "s"), ("startHandshake", "r"), ("OK", "s"), ("getInputStream", "r"), ("INPUT", "s"), ("close", "r")],
            # Read-only connection
            [("connect", "r"), ("CONNECTED", "s"), ("startHandshake", "r"), ("OK", "s"), ("getInputStream", "r"), ("INPUT", "s"), ("close", "r")],
            # Write-only connection
            [("connect", "r"), ("CONNECTED", "s"), ("startHandshake", "r"), ("OK", "s"), ("getOutputStream", "r"), ("OUTPUT", "s"), ("close", "r")],
            # Session inspection
            [("connect", "r"), ("CONNECTED", "s"), ("startHandshake", "r"), ("OK", "s"), ("getSession", "r"), ("SESSION", "s"), ("close", "r")],
            # Full: session + both streams
            [("connect", "r"), ("CONNECTED", "s"), ("startHandshake", "r"), ("OK", "s"), ("getSession", "r"), ("SESSION", "s"), ("getInputStream", "r"), ("INPUT", "s"), ("getOutputStream", "r"), ("OUTPUT", "s"), ("close", "r")],
        ],
    },

    "java.nio.channels.SocketChannel": {
        "description": "Non-blocking TCP channel (Java NIO)",
        "package": "java.nio.channels",
        "class": "SocketChannel",
        "methods": [
            "open", "connect", "finishConnect", "configureBlocking",
            "read", "write", "register", "close",
        ],
        "lifecycle_phases": ["open", "connect", "io", "close"],
        # connect() in non-blocking: returns boolean (immediate or pending)
        # read() returns int: -1 on closed, 0 on no-data, >0 on data
        "return_types": {
            "open": ["CHANNEL"],
            "connect": ["IMMEDIATE", "PENDING"],
            "finishConnect": ["CONNECTED"],
            "configureBlocking": ["CONFIGURED"],
            "read": ["DATA", "NO_DATA", "CLOSED"],
            "write": ["WRITTEN"],
            "register": ["KEY"],
            "close": [],
        },
        "directed_traces": [
            # Blocking connect + read data + write + close
            [("open", "r"), ("CHANNEL", "s"), ("configureBlocking", "r"), ("CONFIGURED", "s"), ("connect", "r"), ("IMMEDIATE", "s"), ("read", "r"), ("DATA", "s"), ("write", "r"), ("WRITTEN", "s"), ("close", "r")],
            # Non-blocking connect (pending)
            [("open", "r"), ("CHANNEL", "s"), ("configureBlocking", "r"), ("CONFIGURED", "s"), ("connect", "r"), ("PENDING", "s"), ("finishConnect", "r"), ("CONNECTED", "s"), ("read", "r"), ("DATA", "s"), ("close", "r")],
            # Selector-based: non-blocking + register
            [("open", "r"), ("CHANNEL", "s"), ("configureBlocking", "r"), ("CONFIGURED", "s"), ("connect", "r"), ("PENDING", "s"), ("finishConnect", "r"), ("CONNECTED", "s"), ("register", "r"), ("KEY", "s"), ("read", "r"), ("DATA", "s"), ("write", "r"), ("WRITTEN", "s"), ("close", "r")],
            # Read returns closed (peer disconnected)
            [("open", "r"), ("CHANNEL", "s"), ("connect", "r"), ("IMMEDIATE", "s"), ("read", "r"), ("CLOSED", "s"), ("close", "r")],
            # Read-write loop
            [("open", "r"), ("CHANNEL", "s"), ("connect", "r"), ("IMMEDIATE", "s"), ("read", "r"), ("DATA", "s"), ("write", "r"), ("WRITTEN", "s"), ("read", "r"), ("DATA", "s"), ("write", "r"), ("WRITTEN", "s"), ("close", "r")],
            # Read returns no data (non-blocking)
            [("open", "r"), ("CHANNEL", "s"), ("configureBlocking", "r"), ("CONFIGURED", "s"), ("connect", "r"), ("PENDING", "s"), ("finishConnect", "r"), ("CONNECTED", "s"), ("read", "r"), ("NO_DATA", "s"), ("read", "r"), ("DATA", "s"), ("close", "r")],
            # Immediate close
            [("open", "r"), ("CHANNEL", "s"), ("close", "r")],
        ],
    },
}


# ---------------------------------------------------------------------------
# Source-based extraction (from Java source files)
# ---------------------------------------------------------------------------

def extract_api_profile_from_source(file_path: str,
                                     target_class: str) -> Optional[JavaAPIProfile]:
    """Extract an API profile from a Java source file.

    Parses the Java source, finds the target class/interface, and
    extracts all public method signatures.
    """
    if not HAS_JAVALANG:
        raise ImportError("javalang required: pip install javalang")

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = javalang.parse.parse(source)
    except Exception:
        return None

    # Find target class or interface
    for _, node in tree.filter(ClassDeclaration):
        if node.name == target_class:
            return _profile_from_class(node, source_type="javalang")

    for _, node in tree.filter(InterfaceDeclaration):
        if node.name == target_class:
            return _profile_from_class(node, source_type="javalang")

    return None


def _profile_from_class(class_node: Any, source_type: str = "javalang") -> JavaAPIProfile:
    """Build a JavaAPIProfile from a parsed class/interface node."""
    methods: list[JavaMethodInfo] = []

    for _, method_node in class_node.filter(MethodDeclaration):
        is_public = "public" in (method_node.modifiers or set())
        is_static = "static" in (method_node.modifiers or set())

        params = []
        if method_node.parameters:
            for p in method_node.parameters:
                ptype = p.type.name if hasattr(p.type, 'name') else str(p.type)
                params.append(ptype)

        throws = []
        if method_node.throws:
            throws = list(method_node.throws)

        ret_type = method_node.return_type
        ret_str = ret_type.name if ret_type and hasattr(ret_type, 'name') else "void"

        line_no = method_node.position.line if method_node.position else 0

        methods.append(JavaMethodInfo(
            name=method_node.name,
            return_type=ret_str,
            parameters=params,
            is_public=is_public,
            is_static=is_static,
            throws=throws,
            line_number=line_no,
        ))

    # Determine package from class name pattern
    package = ""
    class_name = class_node.name

    # Infer lifecycle phases from method names
    phases = _infer_lifecycle_phases([m.name for m in methods])

    return JavaAPIProfile(
        class_name=class_name,
        package=package,
        methods=methods,
        lifecycle_phases=phases,
        source=source_type,
    )


def _infer_lifecycle_phases(method_names: list[str]) -> list[str]:
    """Infer lifecycle phases from method names using common patterns."""
    phases: list[str] = []

    init_methods = {"init", "open", "connect", "start", "create", "begin",
                    "startHandshake", "setUp", "initialize"}
    close_methods = {"close", "destroy", "shutdown", "stop", "end", "finish",
                     "disconnect", "dispose", "release", "terminate"}
    query_methods = {"get", "is", "has", "contains", "size", "length",
                     "isEmpty", "available", "isOpen", "isConnected"}

    names_set = set(method_names)

    if names_set & init_methods:
        phases.append("init")
    if names_set - init_methods - close_methods - query_methods:
        phases.append("use")
    if names_set & close_methods:
        phases.append("close")

    return phases or ["use"]


# ---------------------------------------------------------------------------
# Directed trace construction
# ---------------------------------------------------------------------------

def _build_directed_traces(
    directed_traces: list[list[tuple[str, str]]],
) -> list[Trace]:
    """Convert directed trace specs to Trace objects with send/receive.

    Each step is (label, direction_code) where:
      "r" = "receive" → Branch (caller chooses method)
      "s" = "send"    → Select (object returns outcome)

    This captures return-value selections: when a method returns a
    boolean/enum/status, the return is a Select node in the session type.
    """
    DIRECTION_MAP = {"r": "receive", "s": "send"}
    traces: list[Trace] = []
    for dt in directed_traces:
        pairs = [(label, DIRECTION_MAP[d]) for label, d in dt]
        traces.append(Trace.from_pairs(pairs))
    return traces


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

def extract_session_type(api_name: str,
                         traces: Optional[list[list[str]]] = None,
                         source_file: Optional[str] = None,
                         target_class: Optional[str] = None) -> ExtractionResult:
    """Extract a session type from a Java API.

    Uses spec-based traces if available, optionally augmented with
    source-based extraction.

    Parameters
    ----------
    api_name : str
        Full API name (e.g., "java.util.Iterator").
    traces : list[list[str]], optional
        Pre-curated usage traces. If None, uses JAVA_API_SPECS.
    source_file : str, optional
        Java source file for source-based extraction.
    target_class : str, optional
        Class name to extract from source.
    """
    spec = JAVA_API_SPECS.get(api_name, {})
    description = spec.get("description", api_name)
    source = "spec"

    # Use directed traces (with return type selections) if available
    directed_traces = spec.get("directed_traces")

    if traces is None and directed_traces is None:
        raise ValueError(f"No traces available for {api_name}")

    # Optionally augment with source-based extraction
    method_count = len(spec.get("methods", []))
    if source_file and target_class:
        profile = extract_api_profile_from_source(source_file, target_class)
        if profile:
            method_count = len(profile.methods)
            source = "hybrid"

    # Build trace objects — prefer directed traces (with selections)
    if directed_traces:
        trace_objects = _build_directed_traces(directed_traces)
        # For the result, flatten directed traces to plain method lists
        if traces is None:
            traces = [
                [step[0] for step in dt]
                for dt in directed_traces
            ]
    elif traces:
        trace_objects = [Trace.from_labels(t) for t in traces]
    else:
        raise ValueError(f"No traces available for {api_name}")

    inferred_ast = infer_from_traces(trace_objects)
    st_str = pretty(inferred_ast)

    # Build state space and check lattice
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

    return ExtractionResult(
        api_name=api_name,
        description=description,
        traces=traces,
        inferred_type=st_str,
        session_type_ast=parsed,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_lattice=lr.is_lattice,
        is_distributive=is_dist,
        is_modular=is_mod,
        classification=classification,
        method_count=method_count,
        trace_count=len(traces),
        source=source,
    )


def extract_all_target_apis() -> list[ExtractionResult]:
    """Extract session types from all 5 target Java APIs."""
    results: list[ExtractionResult] = []
    for api_name in JAVA_API_SPECS:
        result = extract_session_type(api_name)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Source-based extraction from BICA project
# ---------------------------------------------------------------------------

def extract_from_bica_source(bica_dir: str) -> list[ExtractionResult]:
    """Extract session types from BICA Reborn Java classes.

    Scans the BICA project for classes with known session type annotations
    and extracts their protocols.
    """
    if not HAS_JAVALANG:
        raise ImportError("javalang required: pip install javalang")

    results: list[ExtractionResult] = []

    # Key BICA classes with session type annotations
    targets = [
        ("StateSpace", "statespace/StateSpace.java",
         [["states", "transitions", "top", "bottom"],
          ["states", "transitions", "labels"],
          ["top", "bottom", "states"],
          ["labels", "transitions"]]),
        ("LatticeChecker", "lattice/LatticeChecker.java",
         [["check", "isLattice", "getMeet", "getJoin"],
          ["check", "isLattice"],
          ["check", "getCounterexample"]]),
        ("Parser", "parser/Parser.java",
         [["parse", "getResult"],
          ["parse", "hasError", "getError"],
          ["parse", "getResult", "prettyPrint"]]),
    ]

    base_path = os.path.join(bica_dir, "com", "bica", "reborn")

    for class_name, rel_path, traces in targets:
        file_path = os.path.join(base_path, rel_path)
        try:
            result = extract_session_type(
                api_name=f"bica.{class_name}",
                traces=traces,
                source_file=file_path if os.path.exists(file_path) else None,
                target_class=class_name,
            )
            results.append(result)
        except (ValueError, Exception):
            pass

    return results


# ---------------------------------------------------------------------------
# Validation table generation
# ---------------------------------------------------------------------------

def format_validation_table(results: list[ExtractionResult]) -> str:
    """Format extraction results as a validation table."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("  JAVA API SESSION TYPE EXTRACTION — VALIDATION TABLE")
    lines.append("=" * 100)

    total = len(results)
    lattice_count = sum(1 for r in results if r.is_lattice)
    dist_count = sum(1 for r in results if r.is_distributive)

    lines.append(f"  APIs analyzed:       {total}")
    lines.append(f"  Lattice rate:        {lattice_count}/{total} "
                 f"({lattice_count/total*100:.0f}%)" if total else "  No results")
    lines.append(f"  Distributive rate:   {dist_count}/{total} "
                 f"({dist_count/total*100:.0f}%)" if total else "")
    lines.append("")

    header = (f"  {'API':<30} {'Methods':>7} {'Traces':>7} {'States':>7} "
              f"{'Trans':>7} {'Lattice':>8} {'Distrib':>8} {'Class':>14}")
    lines.append(header)
    lines.append("  " + "-" * 96)

    for r in results:
        lat = "YES" if r.is_lattice else "NO"
        dist = "YES" if r.is_distributive else "NO"
        lines.append(
            f"  {r.api_name:<30} {r.method_count:>7} {r.trace_count:>7} "
            f"{r.num_states:>7} {r.num_transitions:>7} "
            f"{lat:>8} {dist:>8} {r.classification:>14}"
        )

    lines.append("  " + "-" * 96)
    lines.append("")

    # Inferred types
    lines.append("  INFERRED SESSION TYPES:")
    lines.append("  " + "-" * 96)
    for r in results:
        type_str = r.inferred_type
        if len(type_str) > 80:
            type_str = type_str[:77] + "..."
        lines.append(f"  {r.api_name}:")
        lines.append(f"    {type_str}")
        lines.append("")

    lines.append("=" * 100)
    return "\n".join(lines)


def format_latex_table(results: list[ExtractionResult]) -> str:
    """Format extraction results as a LaTeX table for the paper."""
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Session types extracted from Java standard library APIs.}")
    lines.append(r"\label{tab:java-extraction}")
    lines.append(r"\begin{tabular}{lrrrrcc}")
    lines.append(r"\toprule")
    lines.append(r"API & Methods & Traces & $|Q|$ & $|\delta|$ & Lattice & Distrib. \\")
    lines.append(r"\midrule")

    for r in results:
        api_short = r.api_name.split(".")[-1]
        lat = r"\cmark" if r.is_lattice else r"\xmark"
        dist = r"\cmark" if r.is_distributive else r"\xmark"
        lines.append(
            f"\\texttt{{{api_short}}} & {r.method_count} & {r.trace_count} & "
            f"{r.num_states} & {r.num_transitions} & {lat} & {dist} \\\\"
        )

    total = len(results)
    lattice_count = sum(1 for r in results if r.is_lattice)
    dist_count = sum(1 for r in results if r.is_distributive)

    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & & & & & "
                 f"{lattice_count}/{total} & {dist_count}/{total} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
