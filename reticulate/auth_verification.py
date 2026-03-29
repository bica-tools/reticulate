"""OAuth/TLS behavioral verification via session types (Step 78).

Deep behavioral verification of OAuth 2.0 and TLS protocols using the
full reticulate toolkit: lattice checking, Gay--Hole subtyping for
backward compatibility and version evolution, morphism analysis for
structural embeddings, bisimulation for behavioral equivalence, and
conformance test generation.

Key results:
  - OAuth 2.0 PKCE is a *supertype* of the base Authorization Code flow
    (it adds a preliminary step, so the base flow is a subtype).
  - TLS 1.3 is NOT a subtype of TLS 1.2 (different handshake structure
    prevents silent downgrade via the type system).
  - mTLS extends TLS 1.3 with client certificate exchange; the base
    TLS 1.3 handshake embeds into mTLS.
  - All four OAuth flows form lattices, enabling unambiguous error recovery.

Usage:
    from reticulate.auth_verification import (
        analyze_all_auth,
        verify_oauth_backward_compat,
        verify_tls_downgrade_safety,
        verify_oauth_pkce_extension,
        generate_conformance_tests,
    )
    report = analyze_all_auth()
    print(format_auth_report(report))
"""

from __future__ import annotations

from dataclasses import dataclass

from reticulate.lattice import LatticeResult, check_lattice
from reticulate.morphism import Morphism, find_embedding, find_isomorphism
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.subtyping import SubtypingResult, check_subtype, is_subtype
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthProtocol:
    """A named authentication/security protocol as a session type.

    Attributes:
        name: Human-readable protocol name.
        version: Protocol version string (e.g. "2.0", "1.3").
        session_type_string: Session type encoding.
        category: One of "oauth", "tls", "oidc".
        description: Free-text description.
    """
    name: str
    version: str
    session_type_string: str
    category: str
    description: str


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a single auth protocol.

    Attributes:
        protocol: The protocol definition.
        ast: Parsed session type AST.
        state_space: Constructed state space.
        lattice_result: Lattice check result.
        num_states: Number of states.
        num_transitions: Number of transitions.
        is_lattice: Whether the state space forms a lattice.
        enumeration: Test enumeration result.
        test_source: Generated JUnit 5 test source code.
    """
    protocol: AuthProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    num_states: int
    num_transitions: int
    is_lattice: bool
    enumeration: EnumerationResult
    test_source: str


@dataclass(frozen=True)
class CompatibilityResult:
    """Result of backward compatibility check between two protocol versions.

    Attributes:
        old_protocol: The older version.
        new_protocol: The newer version.
        subtyping_result: Gay-Hole subtyping (old <= new).
        is_backward_compatible: True iff old is subtype of new.
        embedding: Embedding morphism from old state space into new, or None.
    """
    old_protocol: AuthProtocol
    new_protocol: AuthProtocol
    subtyping_result: SubtypingResult
    is_backward_compatible: bool
    embedding: Morphism | None


@dataclass(frozen=True)
class DowngradeSafetyResult:
    """Result of TLS downgrade safety analysis.

    Attributes:
        tls13: TLS 1.3 protocol.
        tls12: TLS 1.2 protocol.
        tls13_sub_tls12: Subtyping result for TLS 1.3 <= TLS 1.2.
        tls12_sub_tls13: Subtyping result for TLS 1.2 <= TLS 1.3.
        is_downgrade_safe: True iff TLS 1.3 is NOT a subtype of TLS 1.2.
        isomorphism: Isomorphism between state spaces, or None.
        embedding_13_into_12: Embedding of TLS 1.3 into TLS 1.2, or None.
        embedding_12_into_13: Embedding of TLS 1.2 into TLS 1.3, or None.
    """
    tls13: AuthProtocol
    tls12: AuthProtocol
    tls13_sub_tls12: SubtypingResult
    tls12_sub_tls13: SubtypingResult
    is_downgrade_safe: bool
    isomorphism: Morphism | None
    embedding_13_into_12: Morphism | None
    embedding_12_into_13: Morphism | None


@dataclass(frozen=True)
class PKCEExtensionResult:
    """Result of PKCE extension analysis.

    Attributes:
        base_flow: OAuth 2.0 Authorization Code flow.
        pkce_flow: OAuth 2.0 Authorization Code + PKCE flow.
        base_sub_pkce: Subtyping result for base <= pkce.
        pkce_sub_base: Subtyping result for pkce <= base.
        base_embeds_in_pkce: True iff base state space embeds in pkce.
        embedding: The embedding morphism, or None.
        base_is_lattice: Whether base forms a lattice.
        pkce_is_lattice: Whether PKCE forms a lattice.
    """
    base_flow: AuthProtocol
    pkce_flow: AuthProtocol
    base_sub_pkce: SubtypingResult
    pkce_sub_base: SubtypingResult
    base_embeds_in_pkce: bool
    embedding: Morphism | None
    base_is_lattice: bool
    pkce_is_lattice: bool


@dataclass(frozen=True)
class MTLSExtensionResult:
    """Result of mTLS extension analysis.

    Attributes:
        tls: Base TLS 1.3 protocol.
        mtls: Mutual TLS protocol.
        tls_sub_mtls: Subtyping for tls <= mtls.
        mtls_sub_tls: Subtyping for mtls <= tls.
        tls_embeds_in_mtls: Whether TLS embeds in mTLS.
        embedding: Embedding morphism, or None.
    """
    tls: AuthProtocol
    mtls: AuthProtocol
    tls_sub_mtls: SubtypingResult
    mtls_sub_tls: SubtypingResult
    tls_embeds_in_mtls: bool
    embedding: Morphism | None


@dataclass(frozen=True)
class AuthAnalysisReport:
    """Complete analysis report for all auth protocols.

    Attributes:
        verifications: Individual verification results for each protocol.
        compatibility: Backward compatibility results for OAuth evolution.
        downgrade_safety: TLS downgrade safety analysis.
        pkce_extension: PKCE extension analysis.
        mtls_extension: mTLS extension analysis.
        all_lattices: True iff all protocols form lattices.
        total_states: Sum of states across all protocols.
        total_transitions: Sum of transitions across all protocols.
    """
    verifications: tuple[VerificationResult, ...]
    compatibility: CompatibilityResult
    downgrade_safety: DowngradeSafetyResult
    pkce_extension: PKCEExtensionResult
    mtls_extension: MTLSExtensionResult
    all_lattices: bool
    total_states: int
    total_transitions: int


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

def oauth2_auth_code_flow() -> AuthProtocol:
    """OAuth 2.0 Authorization Code flow (RFC 6749 Section 4.1)."""
    return AuthProtocol(
        name="OAuth2-AuthCode",
        version="2.0",
        session_type_string=(
            "&{requestAuth: &{authenticate: +{GRANTED: "
            "&{issueCode: &{exchangeCode: +{TOKEN_OK: end, "
            "TOKEN_ERROR: end}}}, DENIED: end}}}"
        ),
        category="oauth",
        description=(
            "Authorization Code Grant: client requests authorization, "
            "resource owner authenticates and grants or denies, "
            "client exchanges code for tokens."
        ),
    )


def oauth2_client_credentials_flow() -> AuthProtocol:
    """OAuth 2.0 Client Credentials flow (RFC 6749 Section 4.4)."""
    return AuthProtocol(
        name="OAuth2-ClientCredentials",
        version="2.0",
        session_type_string=(
            "&{presentCredentials: &{validateClient: "
            "+{VALID: &{issueToken: end}, INVALID: end}}}"
        ),
        category="oauth",
        description=(
            "Client Credentials Grant: confidential client presents "
            "credentials directly, server validates and issues token."
        ),
    )


def oauth2_pkce_flow() -> AuthProtocol:
    """OAuth 2.0 Authorization Code + PKCE (RFC 7636)."""
    return AuthProtocol(
        name="OAuth2-PKCE",
        version="2.0-PKCE",
        session_type_string=(
            "&{generateChallenge: &{requestAuth: &{authenticate: "
            "+{GRANTED: &{issueCode: &{exchangeCodeWithVerifier: "
            "+{TOKEN_OK: end, TOKEN_ERROR: end}}}, DENIED: end}}}}"
        ),
        category="oauth",
        description=(
            "Authorization Code + PKCE: adds code_verifier/code_challenge "
            "to prevent authorization code interception in public clients."
        ),
    )


def oauth2_device_code_flow() -> AuthProtocol:
    """OAuth 2.0 Device Authorization Grant (RFC 8628)."""
    return AuthProtocol(
        name="OAuth2-DeviceCode",
        version="2.0-Device",
        session_type_string=(
            "&{requestDeviceCode: &{displayUserCode: "
            "&{pollToken: +{AUTHORIZED: &{issueToken: end}, "
            "PENDING: &{waitInterval: &{pollToken: +{AUTHORIZED: "
            "&{issueToken: end}, DENIED: end}}}, "
            "DENIED: end}}}}"
        ),
        category="oauth",
        description=(
            "Device Code Grant: device requests code, user authorizes "
            "on separate device, client polls for token."
        ),
    )


def tls13_handshake() -> AuthProtocol:
    """TLS 1.3 handshake (RFC 8446)."""
    return AuthProtocol(
        name="TLS-1.3",
        version="1.3",
        session_type_string=(
            "&{clientHello: &{serverHello: &{serverCert: "
            "+{CERT_VALID: &{serverFinished: &{clientFinished: end}}, "
            "CERT_INVALID: end}}}}"
        ),
        category="tls",
        description=(
            "TLS 1.3 handshake: 1-RTT with ephemeral Diffie-Hellman, "
            "server certificate verification, forward secrecy."
        ),
    )


def tls12_handshake() -> AuthProtocol:
    """TLS 1.2 handshake (RFC 5246) -- simplified model.

    Models the 2-RTT handshake with separate key exchange and cipher
    negotiation steps that TLS 1.3 eliminated.
    """
    return AuthProtocol(
        name="TLS-1.2",
        version="1.2",
        session_type_string=(
            "&{clientHello: &{serverHello: &{serverCert: "
            "&{serverKeyExchange: &{serverHelloDone: "
            "&{clientKeyExchange: &{changeCipherSpec: "
            "+{HANDSHAKE_OK: &{serverChangeCipherSpec: end}, "
            "HANDSHAKE_FAIL: end}}}}}}}}"
        ),
        category="tls",
        description=(
            "TLS 1.2 handshake: 2-RTT with separate key exchange, "
            "cipher spec change, no mandatory forward secrecy."
        ),
    )


def mtls_handshake() -> AuthProtocol:
    """Mutual TLS handshake (RFC 8446 Section 4.4.2)."""
    return AuthProtocol(
        name="mTLS",
        version="1.3-mutual",
        session_type_string=(
            "&{clientHello: &{serverHello: &{serverCert: "
            "+{CERT_VALID: &{certRequest: &{clientCert: "
            "+{CLIENT_VALID: &{serverFinished: &{clientFinished: end}}, "
            "CLIENT_INVALID: end}}}, CERT_INVALID: end}}}}"
        ),
        category="tls",
        description=(
            "Mutual TLS: extends TLS 1.3 with client certificate "
            "authentication for zero-trust and service mesh."
        ),
    )


def oidc_auth_code_flow() -> AuthProtocol:
    """OpenID Connect Authorization Code flow."""
    return AuthProtocol(
        name="OIDC-AuthCode",
        version="1.0",
        session_type_string=(
            "&{requestAuthOpenID: &{authenticateUser: "
            "+{GRANTED: &{issueCode: &{exchangeForTokens: "
            "&{validateIDToken: +{CLAIMS_VALID: end, "
            "CLAIMS_INVALID: end}}}}, DENIED: end}}}"
        ),
        category="oidc",
        description=(
            "OpenID Connect: extends OAuth 2.0 with ID token for "
            "user authentication and claim validation."
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_AUTH_PROTOCOLS: tuple[AuthProtocol, ...] = (
    oauth2_auth_code_flow(),
    oauth2_client_credentials_flow(),
    oauth2_pkce_flow(),
    oauth2_device_code_flow(),
    tls13_handshake(),
    tls12_handshake(),
    mtls_handshake(),
    oidc_auth_code_flow(),
)
"""All pre-defined authentication protocols."""


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------

def _parse_and_build(protocol: AuthProtocol) -> tuple[SessionType, StateSpace]:
    """Parse a protocol and build its state space."""
    ast = parse(protocol.session_type_string)
    ss = build_statespace(ast)
    return ast, ss


def verify_protocol(protocol: AuthProtocol) -> VerificationResult:
    """Run the full verification pipeline on an auth protocol.

    Parses the session type, builds the state space, checks lattice
    properties, enumerates test paths, and generates test source.

    Args:
        protocol: The auth protocol to verify.

    Returns:
        Complete VerificationResult.
    """
    ast, ss = _parse_and_build(protocol)
    lr = check_lattice(ss)
    config = TestGenConfig(
        class_name=f"{protocol.name.replace('-', '')}Test",
        package_name="com.auth.conformance",
    )
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.session_type_string)

    return VerificationResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        is_lattice=lr.is_lattice,
        enumeration=enum,
        test_source=test_src,
    )


def verify_oauth_backward_compat(
    old: AuthProtocol,
    new: AuthProtocol,
) -> CompatibilityResult:
    """Check backward compatibility between two OAuth versions.

    Uses Gay-Hole subtyping: old <= new means the new version accepts
    all interactions the old version accepted.  Also searches for a
    state-space embedding.

    Args:
        old: The older protocol version.
        new: The newer protocol version.

    Returns:
        CompatibilityResult with subtyping verdict and embedding.
    """
    old_ast, old_ss = _parse_and_build(old)
    new_ast, new_ss = _parse_and_build(new)

    sub = check_subtype(old_ast, new_ast)
    emb = find_embedding(old_ss, new_ss)

    return CompatibilityResult(
        old_protocol=old,
        new_protocol=new,
        subtyping_result=sub,
        is_backward_compatible=sub.is_subtype,
        embedding=emb,
    )


def verify_tls_downgrade_safety() -> DowngradeSafetyResult:
    """Verify that TLS 1.3 is NOT a subtype of TLS 1.2.

    Downgrade safety means that a TLS 1.3 client cannot silently fall
    back to a TLS 1.2 handshake pattern.  In session type terms, TLS 1.3
    and TLS 1.2 have incompatible structures (different handshake steps),
    so neither is a subtype of the other.

    Returns:
        DowngradeSafetyResult with subtyping verdicts and morphisms.
    """
    tls13 = tls13_handshake()
    tls12 = tls12_handshake()

    ast13, ss13 = _parse_and_build(tls13)
    ast12, ss12 = _parse_and_build(tls12)

    sub_13_12 = check_subtype(ast13, ast12)
    sub_12_13 = check_subtype(ast12, ast13)

    iso = find_isomorphism(ss13, ss12)
    emb_13_12 = find_embedding(ss13, ss12)
    emb_12_13 = find_embedding(ss12, ss13)

    return DowngradeSafetyResult(
        tls13=tls13,
        tls12=tls12,
        tls13_sub_tls12=sub_13_12,
        tls12_sub_tls13=sub_12_13,
        is_downgrade_safe=not sub_13_12.is_subtype,
        isomorphism=iso,
        embedding_13_into_12=emb_13_12,
        embedding_12_into_13=emb_12_13,
    )


def verify_oauth_pkce_extension() -> PKCEExtensionResult:
    """Verify that PKCE extends the base Authorization Code flow.

    PKCE adds a preliminary generateChallenge step before the standard
    auth code flow.  The base flow is a subtype of PKCE (base accepts
    fewer initial steps).  The base state space embeds into the PKCE
    state space (the shared suffix is structurally identical).

    Returns:
        PKCEExtensionResult with subtyping and embedding analysis.
    """
    base = oauth2_auth_code_flow()
    pkce = oauth2_pkce_flow()

    base_ast, base_ss = _parse_and_build(base)
    pkce_ast, pkce_ss = _parse_and_build(pkce)

    base_sub_pkce = check_subtype(base_ast, pkce_ast)
    pkce_sub_base = check_subtype(pkce_ast, base_ast)

    emb = find_embedding(base_ss, pkce_ss)

    base_lr = check_lattice(base_ss)
    pkce_lr = check_lattice(pkce_ss)

    return PKCEExtensionResult(
        base_flow=base,
        pkce_flow=pkce,
        base_sub_pkce=base_sub_pkce,
        pkce_sub_base=pkce_sub_base,
        base_embeds_in_pkce=emb is not None,
        embedding=emb,
        base_is_lattice=base_lr.is_lattice,
        pkce_is_lattice=pkce_lr.is_lattice,
    )


def verify_mtls_extension() -> MTLSExtensionResult:
    """Verify that mTLS extends TLS 1.3 with client authentication.

    The base TLS 1.3 handshake state space should embed into the mTLS
    state space.  TLS is a subtype of mTLS because mTLS requires
    additional client certificate steps (more methods in the branch).

    Returns:
        MTLSExtensionResult with subtyping and embedding.
    """
    tls = tls13_handshake()
    mtls_proto = mtls_handshake()

    tls_ast, tls_ss = _parse_and_build(tls)
    mtls_ast, mtls_ss = _parse_and_build(mtls_proto)

    tls_sub_mtls = check_subtype(tls_ast, mtls_ast)
    mtls_sub_tls = check_subtype(mtls_ast, tls_ast)

    emb = find_embedding(tls_ss, mtls_ss)

    return MTLSExtensionResult(
        tls=tls,
        mtls=mtls_proto,
        tls_sub_mtls=tls_sub_mtls,
        mtls_sub_tls=mtls_sub_tls,
        tls_embeds_in_mtls=emb is not None,
        embedding=emb,
    )


def generate_conformance_tests(protocol: AuthProtocol) -> str:
    """Generate JUnit 5 conformance test source for a protocol.

    Enumerates valid paths, violations, and incomplete prefixes,
    then produces a complete test class.

    Args:
        protocol: The auth protocol.

    Returns:
        JUnit 5 test source code as a string.
    """
    ast, ss = _parse_and_build(protocol)
    config = TestGenConfig(
        class_name=f"{protocol.name.replace('-', '')}ConformanceTest",
        package_name="com.auth.conformance",
    )
    return generate_test_source(ss, config, protocol.session_type_string)


def analyze_all_auth() -> AuthAnalysisReport:
    """Run the full analysis suite on all auth protocols.

    Verifies each protocol individually, then performs cross-protocol
    analyses: backward compatibility, downgrade safety, PKCE extension,
    and mTLS extension.

    Returns:
        Complete AuthAnalysisReport.
    """
    verifications = tuple(verify_protocol(p) for p in ALL_AUTH_PROTOCOLS)

    compat = verify_oauth_backward_compat(
        oauth2_auth_code_flow(),
        oauth2_pkce_flow(),
    )

    downgrade = verify_tls_downgrade_safety()
    pkce = verify_oauth_pkce_extension()
    mtls_ext = verify_mtls_extension()

    all_lattice = all(v.is_lattice for v in verifications)
    total_s = sum(v.num_states for v in verifications)
    total_t = sum(v.num_transitions for v in verifications)

    return AuthAnalysisReport(
        verifications=verifications,
        compatibility=compat,
        downgrade_safety=downgrade,
        pkce_extension=pkce,
        mtls_extension=mtls_ext,
        all_lattices=all_lattice,
        total_states=total_s,
        total_transitions=total_t,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_auth_report(report: AuthAnalysisReport) -> str:
    """Format an AuthAnalysisReport as structured text."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("  AUTH PROTOCOL BEHAVIORAL VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Individual protocols
    lines.append("--- Protocol Verification ---")
    header = f"  {'Protocol':<28} {'States':>6} {'Trans':>6} {'Lattice':>8} {'Paths':>6}"
    lines.append(header)
    lines.append("  " + "-" * 58)
    for v in report.verifications:
        lat = "YES" if v.is_lattice else "NO"
        lines.append(
            f"  {v.protocol.name:<28} {v.num_states:>6} "
            f"{v.num_transitions:>6} {lat:>8} "
            f"{len(v.enumeration.valid_paths):>6}"
        )
    lines.append("")
    lines.append(f"  All form lattices: {'YES' if report.all_lattices else 'NO'}")
    lines.append(f"  Total states: {report.total_states}")
    lines.append(f"  Total transitions: {report.total_transitions}")
    lines.append("")

    # Backward compatibility
    c = report.compatibility
    lines.append("--- OAuth Backward Compatibility ---")
    lines.append(f"  {c.old_protocol.name} <= {c.new_protocol.name}: "
                 f"{c.is_backward_compatible}")
    if c.embedding:
        lines.append(f"  State-space embedding: FOUND ({c.embedding.kind})")
    else:
        lines.append("  State-space embedding: NOT FOUND")
    lines.append("")

    # Downgrade safety
    d = report.downgrade_safety
    lines.append("--- TLS Downgrade Safety ---")
    lines.append(f"  TLS 1.3 <= TLS 1.2: {d.tls13_sub_tls12.is_subtype}")
    lines.append(f"  TLS 1.2 <= TLS 1.3: {d.tls12_sub_tls13.is_subtype}")
    lines.append(f"  Downgrade safe: {d.is_downgrade_safe}")
    if d.isomorphism:
        lines.append("  WARNING: state spaces are isomorphic!")
    lines.append("")

    # PKCE extension
    p = report.pkce_extension
    lines.append("--- PKCE Extension ---")
    lines.append(f"  Base <= PKCE: {p.base_sub_pkce.is_subtype}")
    lines.append(f"  PKCE <= Base: {p.pkce_sub_base.is_subtype}")
    lines.append(f"  Base embeds in PKCE: {p.base_embeds_in_pkce}")
    lines.append(f"  Both form lattices: "
                 f"{p.base_is_lattice and p.pkce_is_lattice}")
    lines.append("")

    # mTLS extension
    m = report.mtls_extension
    lines.append("--- mTLS Extension ---")
    lines.append(f"  TLS <= mTLS: {m.tls_sub_mtls.is_subtype}")
    lines.append(f"  mTLS <= TLS: {m.mtls_sub_tls.is_subtype}")
    lines.append(f"  TLS embeds in mTLS: {m.tls_embeds_in_mtls}")
    lines.append("")

    return "\n".join(lines)
