"""Generate executable pytest tests from extracted session types.

Given a session type extracted from a Python API, generates pytest test
functions that actually execute against the real Python standard library.

The key mapping: session type paths → real API call sequences.

Each API has a APITestHarness that knows how to:
  - Create the object under test
  - Map method labels to real API calls
  - Map selection outcomes to assertions on return values
  - Tear down after each test

This is the bridge between abstract session types and concrete execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from reticulate.parser import parse, pretty
from reticulate.statespace import build_statespace, StateSpace
from reticulate.testgen import enumerate_valid_paths, ValidPath, Step


# ---------------------------------------------------------------------------
# Test harness: maps session type labels to real API calls
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodMapping:
    """Maps a session type label to real Python code."""
    label: str
    code: str  # Python expression/statement
    direction: str  # "call" or "assert"
    doc: str = ""


@dataclass
class APITestHarness:
    """Harness for testing a real Python API against its session type."""
    api_name: str
    setup_code: list[str]  # lines to create the object
    teardown_code: list[str]  # cleanup lines
    var_name: str  # variable holding the object
    imports: list[str]  # import statements
    method_map: dict[str, str]  # label → Python code
    assertion_map: dict[str, str]  # OUTCOME label → assertion code
    skip_reason: Optional[str] = None  # reason to skip if not runnable


# ---------------------------------------------------------------------------
# Harness definitions for each Python API
# ---------------------------------------------------------------------------

HARNESSES: dict[str, APITestHarness] = {
    "sqlite3.Connection": APITestHarness(
        api_name="sqlite3.Connection",
        imports=["import sqlite3", "import os", "import tempfile"],
        setup_code=[
            "db_fd, db_path = tempfile.mkstemp(suffix='.db')",
            "os.close(db_fd)",
            "conn = sqlite3.connect(db_path)",
            "conn.execute('CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)')",
            "conn.commit()",
        ],
        teardown_code=[
            "try:",
            "    conn.close()",
            "except Exception:",
            "    pass",
            "os.unlink(db_path)",
        ],
        var_name="conn",
        method_map={
            "cursor": "cur = conn.cursor()",
            "execute": "conn.execute(\"INSERT INTO t (val) VALUES ('x')\")",
            "executemany": "conn.executemany(\"INSERT INTO t (val) VALUES (?)\", [('a',), ('b',)])",
            "commit": "conn.commit()",
            "rollback": "conn.rollback()",
            "close": "conn.close()",
            "isolation_level": "_ = conn.isolation_level",
        },
        assertion_map={
            "CURSOR": "assert cur is not None",
            "ROWS_AFFECTED": "# rows affected (implicit)",
            "OK": "# commit/rollback succeeded",
            "INTEGRITY_ERROR": "# would need duplicate key to trigger",
            "LEVEL": "# isolation level retrieved",
        },
    ),

    "http.client.HTTPConnection": APITestHarness(
        api_name="http.client.HTTPConnection",
        imports=["import http.client"],
        setup_code=[
            "conn = http.client.HTTPConnection('httpbin.org', timeout=10)",
        ],
        teardown_code=[
            "try:",
            "    conn.close()",
            "except Exception:",
            "    pass",
        ],
        var_name="conn",
        method_map={
            "connect": "conn.connect()",
            "request": "conn.request('GET', '/status/200')",
            "getresponse": "resp = conn.getresponse(); _ = resp.read()",
            "close": "conn.close()",
            "set_debuglevel": "conn.set_debuglevel(0)",
            "putheader": "# putheader requires low-level request",
            "endheaders": "# endheaders requires low-level request",
        },
        assertion_map={
            "CONNECTED": "# connection established",
            "SENT": "# request sent",
            "OK_200": "assert resp.status == 200",
            "NOT_FOUND_404": "assert resp.status == 404",
            "SERVER_ERROR_500": "assert resp.status == 500",
            "REDIRECT_301": "assert resp.status in (301, 302)",
            "CONFIGURED": "# debug level set",
            "ADDED": "# header added",
            "READY": "# headers ended",
        },
        skip_reason="requires network access",
    ),

    "smtplib.SMTP": APITestHarness(
        api_name="smtplib.SMTP",
        imports=["import smtplib"],
        setup_code=[
            "smtp = smtplib.SMTP('localhost', 25, timeout=5)",
        ],
        teardown_code=[
            "try:",
            "    smtp.quit()",
            "except Exception:",
            "    pass",
        ],
        var_name="smtp",
        method_map={
            "connect": "smtp.connect('localhost', 25)",
            "ehlo": "smtp.ehlo()",
            "starttls": "smtp.starttls()",
            "login": "smtp.login('user', 'pass')",
            "sendmail": "smtp.sendmail('a@b.com', 'c@d.com', 'Subject: test\\n\\nBody')",
            "send_message": "# send_message requires email.message.EmailMessage",
            "quit": "smtp.quit()",
        },
        assertion_map={
            "CONNECTED": "# connected to SMTP server",
            "EHLO_OK": "# EHLO accepted",
            "EHLO_ERROR": "# EHLO rejected",
            "TLS_OK": "# STARTTLS succeeded",
            "TLS_ERROR": "# STARTTLS failed",
            "AUTH_OK": "# login succeeded",
            "AUTH_FAILED": "# login failed",
            "SENT": "# message sent",
            "REJECTED": "# message rejected",
        },
        skip_reason="requires SMTP server on localhost:25",
    ),

    "ftplib.FTP": APITestHarness(
        api_name="ftplib.FTP",
        imports=["import ftplib"],
        setup_code=[
            "ftp = ftplib.FTP()",
        ],
        teardown_code=[
            "try:",
            "    ftp.quit()",
            "except Exception:",
            "    pass",
        ],
        var_name="ftp",
        method_map={
            "connect": "ftp.connect('test.rebex.net')",
            "login": "ftp.login('demo', 'password')",
            "cwd": "ftp.cwd('/')",
            "pwd": "_ = ftp.pwd()",
            "nlst": "_ = ftp.nlst()",
            "retrbinary": "data = bytearray(); ftp.retrbinary('RETR readme.txt', data.extend)",
            "storbinary": "# storbinary requires write access",
            "rename": "# rename requires write access",
            "delete": "# delete requires write access",
            "mkd": "# mkd requires write access",
            "rmd": "# rmd requires write access",
            "quit": "ftp.quit()",
        },
        assertion_map={
            "CONNECTED": "# FTP connected",
            "AUTH_OK": "# login succeeded",
            "AUTH_FAILED": "# login failed",
            "CHANGED": "# directory changed",
            "PATH": "# pwd returned",
            "LISTING": "# directory listed",
            "DOWNLOADED": "assert len(data) > 0",
            "UPLOADED": "# file uploaded",
            "RENAMED": "# file renamed",
            "DELETED": "# file deleted",
            "CREATED": "# directory created",
            "REMOVED": "# directory removed",
        },
        skip_reason="requires network access to test.rebex.net",
    ),

    "ssl.SSLSocket": APITestHarness(
        api_name="ssl.SSLSocket",
        imports=["import ssl", "import socket"],
        setup_code=[
            "ctx = ssl.create_default_context()",
            "raw = socket.create_connection(('www.google.com', 443), timeout=10)",
            "sock = ctx.wrap_socket(raw, server_hostname='www.google.com')",
        ],
        teardown_code=[
            "try:",
            "    sock.close()",
            "except Exception:",
            "    pass",
        ],
        var_name="sock",
        method_map={
            "connect": "# already connected via wrap_socket",
            "do_handshake": "# handshake done by wrap_socket",
            "getpeercert": "cert = sock.getpeercert()",
            "recv": "# recv requires sent request first",
            "send": "sock.send(b'GET / HTTP/1.0\\r\\nHost: www.google.com\\r\\n\\r\\n')",
            "read": "data = sock.recv(4096)",
            "write": "sock.send(b'GET / HTTP/1.0\\r\\nHost: www.google.com\\r\\n\\r\\n')",
            "unwrap": "# unwrap returns plain socket",
            "close": "sock.close()",
        },
        assertion_map={
            "CONNECTED": "# socket connected",
            "OK": "# handshake succeeded",
            "CERT_ERROR": "# cert verification failed",
            "HANDSHAKE_ERROR": "# handshake failed",
            "CERT": "assert cert is not None",
            "DATA": "assert len(data) > 0",
            "EOF": "# connection closed",
            "CLOSED": "# peer closed",
            "SENT": "# data sent",
            "WRITTEN": "# data written",
            "UNWRAPPED": "# TLS unwrapped",
        },
        skip_reason="requires network access to www.google.com:443",
    ),
}


# ---------------------------------------------------------------------------
# Path selection: pick representative paths for testing
# ---------------------------------------------------------------------------

def select_test_paths(
    ss: StateSpace,
    max_paths: int = 10,
) -> list[ValidPath]:
    """Select representative valid paths for test generation.

    Prefers shorter paths and paths that cover different transitions.
    """
    paths, _ = enumerate_valid_paths(ss, max_revisits=2, max_paths=50)

    # Sort by length (prefer shorter, more focused tests)
    paths.sort(key=lambda p: len(p.steps))

    # Deduplicate by label sequence
    seen: set[tuple[str, ...]] = set()
    selected: list[ValidPath] = []
    for p in paths:
        key = tuple(p.labels)
        if key not in seen:
            seen.add(key)
            selected.append(p)
            if len(selected) >= max_paths:
                break

    return selected


# ---------------------------------------------------------------------------
# Pytest source generation
# ---------------------------------------------------------------------------

def generate_pytest_source(
    api_name: str,
    session_type_str: str,
    harness: Optional[APITestHarness] = None,
    max_tests: int = 10,
) -> str:
    """Generate executable pytest source for a Python API.

    Parameters
    ----------
    api_name : str
        API name (e.g., "sqlite3.Connection").
    session_type_str : str
        The inferred session type string.
    harness : APITestHarness, optional
        Custom harness. Uses HARNESSES[api_name] if None.
    max_tests : int
        Maximum number of test functions to generate.
    """
    if harness is None:
        harness = HARNESSES.get(api_name)
        if harness is None:
            raise ValueError(f"No test harness for {api_name}")

    # Parse and build state space
    ast = parse(session_type_str)
    ss = build_statespace(ast)
    paths = select_test_paths(ss, max_paths=max_tests)

    lines: list[str] = []

    # Header
    lines.append(f'"""Generated protocol conformance tests for {api_name}.')
    lines.append(f"")
    lines.append(f"Session type: {session_type_str[:80]}...")
    lines.append(f"Generated by reticulate pytest_gen module.")
    lines.append(f'"""')
    lines.append("")

    # Imports
    lines.append("import pytest")
    for imp in harness.imports:
        lines.append(imp)
    lines.append("")
    lines.append("")

    # Skip marker if needed
    if harness.skip_reason:
        lines.append(f'SKIP_REASON = "{harness.skip_reason}"')
        lines.append("")
        lines.append("")

    # Fixture
    class_short = api_name.split(".")[-1]
    lines.append(f"@pytest.fixture")
    lines.append(f"def {harness.var_name}():")
    lines.append(f'    """Create {api_name} instance for testing."""')
    for line in harness.setup_code:
        lines.append(f"    {line}")
    lines.append(f"    yield {harness.var_name}")
    for line in harness.teardown_code:
        lines.append(f"    {line}")
    lines.append("")
    lines.append("")

    # Generate test for each valid path
    for i, path in enumerate(paths):
        method_labels = [s.label for s in path.steps
                         if not s.label.isupper()]
        select_labels = [s.label for s in path.steps
                         if s.label.isupper()]

        # Build test name from method labels
        name_parts = [l for l in method_labels[:4]]
        test_name = "_".join(name_parts) if name_parts else "empty"
        test_name = test_name.replace("-", "_")

        # Skip decorator if network needed
        if harness.skip_reason:
            lines.append(f'@pytest.mark.skipif(True, reason=SKIP_REASON)')

        lines.append(f"def test_path_{i}_{test_name}({harness.var_name}):")
        lines.append(f'    """Valid path: {" → ".join(path.labels[:8])}{"..." if len(path.labels) > 8 else ""}"""')

        for step in path.steps:
            if step.label.isupper():
                # Selection outcome — generate assertion
                assertion = harness.assertion_map.get(step.label, f"# {step.label}")
                lines.append(f"    {assertion}")
            else:
                # Method call — generate real API call
                code = harness.method_map.get(step.label, f"# {step.label}()")
                lines.append(f"    {code}")

        lines.append("")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generate for all APIs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationResult:
    """Result of generating tests for one API."""
    api_name: str
    source: str
    num_tests: int
    num_paths: int
    is_runnable: bool  # False if skip_reason set
    skip_reason: Optional[str]


def generate_all_pytest_sources(
    extraction_results: list[Any],
    max_tests_per_api: int = 8,
) -> list[GenerationResult]:
    """Generate pytest sources for all extracted Python APIs."""
    results: list[GenerationResult] = []

    for er in extraction_results:
        harness = HARNESSES.get(er.api_name)
        if harness is None:
            continue

        source = generate_pytest_source(
            er.api_name,
            er.inferred_type,
            harness=harness,
            max_tests=max_tests_per_api,
        )

        # Count test functions
        num_tests = source.count("def test_")
        ast = parse(er.inferred_type)
        ss = build_statespace(ast)
        paths = select_test_paths(ss, max_paths=max_tests_per_api)

        results.append(GenerationResult(
            api_name=er.api_name,
            source=source,
            num_tests=num_tests,
            num_paths=len(paths),
            is_runnable=harness.skip_reason is None,
            skip_reason=harness.skip_reason,
        ))

    return results


# ---------------------------------------------------------------------------
# Validation table
# ---------------------------------------------------------------------------

def format_testgen_table(results: list[GenerationResult]) -> str:
    """Format test generation results as a table."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  TEST GENERATION FROM EXTRACTED SESSION TYPES")
    lines.append("=" * 90)

    total_tests = sum(r.num_tests for r in results)
    runnable = sum(1 for r in results if r.is_runnable)

    lines.append(f"  APIs:        {len(results)}")
    lines.append(f"  Tests:       {total_tests}")
    lines.append(f"  Runnable:    {runnable}/{len(results)} (without network)")
    lines.append("")

    header = (f"  {'API':<30} {'Paths':>6} {'Tests':>6} "
              f"{'Runnable':>9} {'Skip Reason':<25}")
    lines.append(header)
    lines.append("  " + "-" * 86)

    for r in results:
        runnable_str = "YES" if r.is_runnable else "SKIP"
        skip = r.skip_reason or ""
        lines.append(
            f"  {r.api_name:<30} {r.num_paths:>6} {r.num_tests:>6} "
            f"{runnable_str:>9} {skip:<25}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)
