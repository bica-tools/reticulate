"""Extract session types from Python standard library APIs.

Directed traces with return-value selections: method calls are Branch
(caller chooses), return outcomes are Select (object decides).

Convention:
  lowercase = method call (Branch &{})
  UPPERCASE = return outcome (Select +{})

Target APIs for TOPLAS validation (Week 2):
  1. sqlite3.Connection   — database connection lifecycle
  2. http.client.HTTPConnection — HTTP request/response
  3. smtplib.SMTP         — SMTP email sending
  4. ftplib.FTP           — FTP file transfer
  5. ssl.SSLSocket        — TLS secure socket

Unlike the Java extractor, we can actually IMPORT these modules and
validate traces against real Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from reticulate.type_inference import Trace, infer_from_traces
from reticulate.parser import parse, pretty, SessionType
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive


# ---------------------------------------------------------------------------
# Result type (shared with java_api_extractor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting and validating a session type from a Python API."""
    api_name: str
    description: str
    traces: list[list[tuple[str, str]]]  # directed traces
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


# ---------------------------------------------------------------------------
# Python API Specifications with directed traces
# ---------------------------------------------------------------------------

PYTHON_API_SPECS: dict[str, dict[str, Any]] = {
    "sqlite3.Connection": {
        "description": "SQLite database connection (Python DB-API 2.0)",
        "module": "sqlite3",
        "class": "Connection",
        "methods": [
            "cursor", "execute", "executemany", "commit",
            "rollback", "close", "isolation_level",
        ],
        "return_types": {
            "cursor": ["CURSOR"],
            "execute": ["ROWS_AFFECTED"],
            "executemany": ["ROWS_AFFECTED"],
            "commit": ["OK", "INTEGRITY_ERROR"],
            "rollback": ["OK"],
            "close": [],
            "isolation_level": ["LEVEL"],
        },
        "directed_traces": [
            # Query: cursor → execute → fetchall → close
            [("cursor", "r"), ("CURSOR", "s"), ("execute", "r"), ("ROWS_AFFECTED", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Direct execute shortcut
            [("execute", "r"), ("ROWS_AFFECTED", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Executemany + commit
            [("cursor", "r"), ("CURSOR", "s"), ("executemany", "r"), ("ROWS_AFFECTED", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Commit fails → rollback
            [("cursor", "r"), ("CURSOR", "s"), ("execute", "r"), ("ROWS_AFFECTED", "s"), ("commit", "r"), ("INTEGRITY_ERROR", "s"), ("rollback", "r"), ("OK", "s"), ("close", "r")],
            # Multiple executes + commit
            [("cursor", "r"), ("CURSOR", "s"), ("execute", "r"), ("ROWS_AFFECTED", "s"), ("execute", "r"), ("ROWS_AFFECTED", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Rollback then retry
            [("execute", "r"), ("ROWS_AFFECTED", "s"), ("rollback", "r"), ("OK", "s"), ("execute", "r"), ("ROWS_AFFECTED", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
            # Just close
            [("close", "r")],
            # Set isolation level then work
            [("isolation_level", "r"), ("LEVEL", "s"), ("execute", "r"), ("ROWS_AFFECTED", "s"), ("commit", "r"), ("OK", "s"), ("close", "r")],
        ],
    },

    "http.client.HTTPConnection": {
        "description": "HTTP/1.1 client connection",
        "module": "http.client",
        "class": "HTTPConnection",
        "methods": [
            "connect", "request", "getresponse", "close",
            "set_debuglevel", "putheader", "endheaders",
        ],
        "return_types": {
            "connect": ["CONNECTED"],
            "request": ["SENT"],
            "getresponse": ["OK_200", "NOT_FOUND_404", "SERVER_ERROR_500", "REDIRECT_301"],
            "close": [],
            "set_debuglevel": ["CONFIGURED"],
            "putheader": ["ADDED"],
            "endheaders": ["READY"],
        },
        "directed_traces": [
            # Simple GET → 200
            [("connect", "r"), ("CONNECTED", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("OK_200", "s"), ("close", "r")],
            # GET → 404
            [("connect", "r"), ("CONNECTED", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("NOT_FOUND_404", "s"), ("close", "r")],
            # GET → 500
            [("connect", "r"), ("CONNECTED", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("SERVER_ERROR_500", "s"), ("close", "r")],
            # GET → redirect → follow
            [("connect", "r"), ("CONNECTED", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("REDIRECT_301", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("OK_200", "s"), ("close", "r")],
            # POST with custom headers
            [("connect", "r"), ("CONNECTED", "s"), ("putheader", "r"), ("ADDED", "s"), ("putheader", "r"), ("ADDED", "s"), ("endheaders", "r"), ("READY", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("OK_200", "s"), ("close", "r")],
            # Debug then request
            [("set_debuglevel", "r"), ("CONFIGURED", "s"), ("connect", "r"), ("CONNECTED", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("OK_200", "s"), ("close", "r")],
            # Multiple requests (keep-alive)
            [("connect", "r"), ("CONNECTED", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("OK_200", "s"), ("request", "r"), ("SENT", "s"), ("getresponse", "r"), ("OK_200", "s"), ("close", "r")],
        ],
    },

    "smtplib.SMTP": {
        "description": "SMTP email client (RFC 5321)",
        "module": "smtplib",
        "class": "SMTP",
        "methods": [
            "connect", "ehlo", "starttls", "login",
            "sendmail", "send_message", "quit",
        ],
        "return_types": {
            "connect": ["CONNECTED"],
            "ehlo": ["EHLO_OK", "EHLO_ERROR"],
            "starttls": ["TLS_OK", "TLS_ERROR"],
            "login": ["AUTH_OK", "AUTH_FAILED"],
            "sendmail": ["SENT", "REJECTED"],
            "send_message": ["SENT"],
            "quit": [],
        },
        "directed_traces": [
            # Full TLS + auth + send
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("starttls", "r"), ("TLS_OK", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("login", "r"), ("AUTH_OK", "s"), ("sendmail", "r"), ("SENT", "s"), ("quit", "r")],
            # Auth fails
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("starttls", "r"), ("TLS_OK", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("login", "r"), ("AUTH_FAILED", "s"), ("quit", "r")],
            # No TLS (local relay)
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("sendmail", "r"), ("SENT", "s"), ("quit", "r")],
            # TLS fails → quit
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("starttls", "r"), ("TLS_ERROR", "s"), ("quit", "r")],
            # Send rejected
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("login", "r"), ("AUTH_OK", "s"), ("sendmail", "r"), ("REJECTED", "s"), ("quit", "r")],
            # send_message (modern API)
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("starttls", "r"), ("TLS_OK", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("login", "r"), ("AUTH_OK", "s"), ("send_message", "r"), ("SENT", "s"), ("quit", "r")],
            # Multiple sends
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_OK", "s"), ("login", "r"), ("AUTH_OK", "s"), ("sendmail", "r"), ("SENT", "s"), ("sendmail", "r"), ("SENT", "s"), ("quit", "r")],
            # EHLO error
            [("connect", "r"), ("CONNECTED", "s"), ("ehlo", "r"), ("EHLO_ERROR", "s"), ("quit", "r")],
        ],
    },

    "ftplib.FTP": {
        "description": "FTP client (RFC 959)",
        "module": "ftplib",
        "class": "FTP",
        "methods": [
            "connect", "login", "cwd", "pwd", "nlst", "retrbinary",
            "storbinary", "rename", "delete", "mkd", "rmd", "quit",
        ],
        "return_types": {
            "connect": ["CONNECTED"],
            "login": ["AUTH_OK", "AUTH_FAILED"],
            "cwd": ["CHANGED"],
            "pwd": ["PATH"],
            "nlst": ["LISTING"],
            "retrbinary": ["DOWNLOADED"],
            "storbinary": ["UPLOADED"],
            "rename": ["RENAMED"],
            "delete": ["DELETED"],
            "mkd": ["CREATED"],
            "rmd": ["REMOVED"],
            "quit": [],
        },
        "directed_traces": [
            # Download file
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_OK", "s"), ("cwd", "r"), ("CHANGED", "s"), ("retrbinary", "r"), ("DOWNLOADED", "s"), ("quit", "r")],
            # Upload file
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_OK", "s"), ("cwd", "r"), ("CHANGED", "s"), ("storbinary", "r"), ("UPLOADED", "s"), ("quit", "r")],
            # List directory
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_OK", "s"), ("pwd", "r"), ("PATH", "s"), ("nlst", "r"), ("LISTING", "s"), ("quit", "r")],
            # Auth fails
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_FAILED", "s"), ("quit", "r")],
            # File management: mkdir, upload, rename, delete
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_OK", "s"), ("mkd", "r"), ("CREATED", "s"), ("cwd", "r"), ("CHANGED", "s"), ("storbinary", "r"), ("UPLOADED", "s"), ("quit", "r")],
            # Rename then delete
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_OK", "s"), ("rename", "r"), ("RENAMED", "s"), ("delete", "r"), ("DELETED", "s"), ("quit", "r")],
            # Remove directory
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_OK", "s"), ("rmd", "r"), ("REMOVED", "s"), ("quit", "r")],
            # Multiple downloads
            [("connect", "r"), ("CONNECTED", "s"), ("login", "r"), ("AUTH_OK", "s"), ("retrbinary", "r"), ("DOWNLOADED", "s"), ("retrbinary", "r"), ("DOWNLOADED", "s"), ("quit", "r")],
        ],
    },

    "ssl.SSLSocket": {
        "description": "TLS/SSL wrapped socket (Python ssl module)",
        "module": "ssl",
        "class": "SSLSocket",
        "methods": [
            "connect", "do_handshake", "getpeercert",
            "recv", "send", "read", "write",
            "unwrap", "close",
        ],
        "return_types": {
            "connect": ["CONNECTED"],
            "do_handshake": ["OK", "CERT_ERROR", "HANDSHAKE_ERROR"],
            "getpeercert": ["CERT"],
            "recv": ["DATA", "CLOSED"],
            "send": ["SENT"],
            "read": ["DATA", "EOF"],
            "write": ["WRITTEN"],
            "unwrap": ["UNWRAPPED"],
            "close": [],
        },
        "directed_traces": [
            # Standard TLS: connect → handshake → verify cert → read/write → close
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("OK", "s"), ("getpeercert", "r"), ("CERT", "s"), ("recv", "r"), ("DATA", "s"), ("send", "r"), ("SENT", "s"), ("close", "r")],
            # Handshake fails (bad cert)
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("CERT_ERROR", "s"), ("close", "r")],
            # Handshake fails (protocol error)
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("HANDSHAKE_ERROR", "s"), ("close", "r")],
            # Read until EOF
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("OK", "s"), ("read", "r"), ("DATA", "s"), ("read", "r"), ("DATA", "s"), ("read", "r"), ("EOF", "s"), ("close", "r")],
            # Write-only
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("OK", "s"), ("write", "r"), ("WRITTEN", "s"), ("write", "r"), ("WRITTEN", "s"), ("close", "r")],
            # Unwrap to plain socket
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("OK", "s"), ("send", "r"), ("SENT", "s"), ("unwrap", "r"), ("UNWRAPPED", "s"), ("close", "r")],
            # Peer closes connection
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("OK", "s"), ("recv", "r"), ("CLOSED", "s"), ("close", "r")],
            # Multiple send/recv
            [("connect", "r"), ("CONNECTED", "s"), ("do_handshake", "r"), ("OK", "s"), ("send", "r"), ("SENT", "s"), ("recv", "r"), ("DATA", "s"), ("send", "r"), ("SENT", "s"), ("recv", "r"), ("DATA", "s"), ("close", "r")],
        ],
    },
}


# ---------------------------------------------------------------------------
# Directed trace construction
# ---------------------------------------------------------------------------

def _build_directed_traces(
    directed_traces: list[list[tuple[str, str]]],
) -> list[Trace]:
    """Convert directed trace specs to Trace objects with send/receive."""
    DIRECTION_MAP = {"r": "receive", "s": "send"}
    traces: list[Trace] = []
    for dt in directed_traces:
        pairs = [(label, DIRECTION_MAP[d]) for label, d in dt]
        traces.append(Trace.from_pairs(pairs))
    return traces


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------

def extract_session_type(api_name: str) -> ExtractionResult:
    """Extract a session type from a Python API specification."""
    spec = PYTHON_API_SPECS.get(api_name)
    if spec is None:
        raise ValueError(f"No spec for {api_name}")

    directed = spec["directed_traces"]
    trace_objects = _build_directed_traces(directed)

    inferred_ast = infer_from_traces(trace_objects)
    st_str = pretty(inferred_ast)

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
        description=spec["description"],
        traces=directed,
        inferred_type=st_str,
        session_type_ast=parsed,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_lattice=lr.is_lattice,
        is_distributive=is_dist,
        is_modular=is_mod,
        classification=classification,
        method_count=len(spec["methods"]),
        trace_count=len(directed),
    )


def extract_all_target_apis() -> list[ExtractionResult]:
    """Extract session types from all 5 target Python APIs."""
    return [extract_session_type(name) for name in PYTHON_API_SPECS]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_validation_table(results: list[ExtractionResult]) -> str:
    """Format extraction results as a validation table."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("  PYTHON API SESSION TYPE EXTRACTION — VALIDATION TABLE")
    lines.append("=" * 100)

    total = len(results)
    lattice_count = sum(1 for r in results if r.is_lattice)
    dist_count = sum(1 for r in results if r.is_distributive)

    lines.append(f"  APIs analyzed:       {total}")
    lines.append(f"  Lattice rate:        {lattice_count}/{total} "
                 f"({lattice_count/total*100:.0f}%)" if total else "")
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
    lines.append(r"\caption{Session types extracted from Python standard library APIs.}")
    lines.append(r"\label{tab:python-extraction}")
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
