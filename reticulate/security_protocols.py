"""Security protocol verification via lattice properties (Step 89).

Models security protocols (OAuth 2.0, TLS, mTLS, Kerberos, SAML) as
session types and verifies their correctness through lattice analysis,
subtyping for version evolution, and coverage computation.

Security protocols are fundamentally stateful: they progress through
handshakes, challenges, and grants in a strict order.  Session types
capture this structure precisely.  The lattice property on the
resulting state space guarantees that every pair of protocol states
has a well-defined join (least common continuation) and meet (greatest
common predecessor), which ensures unambiguous protocol recovery from
any reachable state.

Protocol evolution (e.g., OAuth 2.0 adding PKCE, TLS 1.2 to 1.3)
is modelled via Gay--Hole subtyping: a newer protocol version is
backward-compatible iff the old version is a subtype of the new one
(the new version offers at least the same methods).

Usage:
    from reticulate.security_protocols import (
        oauth2_auth_code,
        tls_handshake,
        verify_security_protocol,
        check_protocol_evolution,
    )
    proto = oauth2_auth_code()
    result = verify_security_protocol(proto)
    print(format_security_report(result))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from reticulate.coverage import CoverageResult, compute_coverage
from reticulate.lattice import (
    DistributivityResult,
    LatticeResult,
    check_distributive,
    check_lattice,
)
from reticulate.parser import SessionType, parse
from reticulate.statespace import StateSpace, build_statespace
from reticulate.subtyping import SubtypingResult, check_subtype, is_subtype
from reticulate.testgen import (
    EnumerationResult,
    TestGenConfig,
    enumerate as enumerate_tests,
    generate_test_source,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecurityProtocol:
    """A named security protocol modelled as a session type.

    Attributes:
        name: Protocol name (e.g., "OAuth2AuthCode").
        roles: Participants in the protocol (e.g., ("Client", "AuthServer")).
        session_type_string: Session type encoding of the protocol.
        security_properties: Properties the protocol aims to guarantee
            (e.g., ("confidentiality", "authentication")).
        description: Free-text description of the protocol.
    """
    name: str
    roles: tuple[str, ...]
    session_type_string: str
    security_properties: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class SecurityAnalysisResult:
    """Complete analysis result for a security protocol.

    Attributes:
        protocol: The analysed protocol definition.
        ast: Parsed session type AST.
        state_space: Constructed state space (reticulate).
        lattice_result: Lattice property check.
        distributivity: Distributivity check result.
        enumeration: Test path enumeration.
        test_source: Generated JUnit 5 test source.
        coverage: Coverage analysis.
        num_states: Number of states in the state space.
        num_transitions: Number of transitions.
        num_valid_paths: Count of valid execution paths.
        num_violations: Count of protocol violation points.
        is_well_formed: True iff state space is a lattice.
    """
    protocol: SecurityProtocol
    ast: SessionType
    state_space: StateSpace
    lattice_result: LatticeResult
    distributivity: DistributivityResult
    enumeration: EnumerationResult
    test_source: str
    coverage: CoverageResult
    num_states: int
    num_transitions: int
    num_valid_paths: int
    num_violations: int
    is_well_formed: bool


@dataclass(frozen=True)
class ProtocolEvolutionResult:
    """Result of checking backward compatibility between protocol versions.

    Attributes:
        old_protocol: The older protocol version.
        new_protocol: The newer protocol version.
        subtyping_result: Gay-Hole subtyping check (old <= new).
        is_backward_compatible: True iff old is subtype of new.
        old_analysis: Full analysis of old protocol.
        new_analysis: Full analysis of new protocol.
    """
    old_protocol: SecurityProtocol
    new_protocol: SecurityProtocol
    subtyping_result: SubtypingResult
    is_backward_compatible: bool
    old_analysis: SecurityAnalysisResult
    new_analysis: SecurityAnalysisResult


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------

def oauth2_auth_code() -> SecurityProtocol:
    """OAuth 2.0 Authorization Code flow (RFC 6749 Section 4.1).

    Models the full authorization code grant:
    1. Client requests authorization
    2. User authenticates and grants/denies
    3. If granted, auth server issues authorization code
    4. Client exchanges code for tokens
    5. Token exchange succeeds or fails

    Participants: Client, Authorization Server, Resource Owner.
    """
    return SecurityProtocol(
        name="OAuth2AuthCode",
        roles=("Client", "AuthorizationServer", "ResourceOwner"),
        session_type_string=(
            "&{requestAuth: &{authenticate: +{GRANTED: "
            "&{issueCode: &{exchangeCode: +{TOKEN_OK: end, "
            "TOKEN_ERROR: end}}}, DENIED: end}}}"
        ),
        security_properties=(
            "authentication",
            "authorization",
            "token_confidentiality",
            "csrf_protection",
        ),
        description=(
            "OAuth 2.0 Authorization Code Grant: the client requests "
            "authorization, the resource owner authenticates and grants "
            "or denies, then the client exchanges the authorization code "
            "for access and refresh tokens."
        ),
    )


def oauth2_client_credentials() -> SecurityProtocol:
    """OAuth 2.0 Client Credentials flow (RFC 6749 Section 4.4).

    Models the client credentials grant where a confidential client
    authenticates directly with the authorization server:
    1. Client presents credentials
    2. Server validates credentials
    3. Token issued or error returned

    Simpler than auth code: no user interaction.
    """
    return SecurityProtocol(
        name="OAuth2ClientCredentials",
        roles=("Client", "AuthorizationServer"),
        session_type_string=(
            "&{presentCredentials: &{validateClient: "
            "+{VALID: &{issueToken: end}, INVALID: end}}}"
        ),
        security_properties=(
            "client_authentication",
            "token_confidentiality",
            "credential_protection",
        ),
        description=(
            "OAuth 2.0 Client Credentials Grant: the client presents "
            "its credentials directly, the server validates them, and "
            "issues an access token or returns an error."
        ),
    )


def tls_handshake() -> SecurityProtocol:
    """TLS 1.3 handshake protocol (RFC 8446).

    Models the simplified TLS 1.3 handshake:
    1. Client sends ClientHello (with key share)
    2. Server responds with ServerHello (with key share)
    3. Server sends certificate and finished
    4. Handshake completes or fails

    TLS 1.3 reduced round-trips from 2 to 1 compared to TLS 1.2.
    """
    return SecurityProtocol(
        name="TLS13Handshake",
        roles=("Client", "Server"),
        session_type_string=(
            "&{clientHello: &{serverHello: &{serverCert: "
            "+{CERT_VALID: &{serverFinished: &{clientFinished: end}}, "
            "CERT_INVALID: end}}}}"
        ),
        security_properties=(
            "confidentiality",
            "integrity",
            "server_authentication",
            "forward_secrecy",
        ),
        description=(
            "TLS 1.3 Handshake: ClientHello with key share, ServerHello "
            "with key share, server certificate verification, and "
            "handshake completion with forward secrecy via ephemeral "
            "Diffie-Hellman."
        ),
    )


def mtls_handshake() -> SecurityProtocol:
    """Mutual TLS handshake with client certificate (RFC 8446 Section 4.4.2).

    Extends TLS 1.3 with client certificate authentication:
    1. Standard TLS handshake begins
    2. Server requests client certificate
    3. Client presents certificate
    4. Server validates client certificate
    5. Mutual authentication established

    Used in zero-trust architectures and service mesh.
    """
    return SecurityProtocol(
        name="MutualTLS",
        roles=("Client", "Server"),
        session_type_string=(
            "&{clientHello: &{serverHello: &{serverCert: "
            "+{CERT_VALID: &{certRequest: &{clientCert: "
            "+{CLIENT_VALID: &{serverFinished: &{clientFinished: end}}, "
            "CLIENT_INVALID: end}}}, CERT_INVALID: end}}}}"
        ),
        security_properties=(
            "confidentiality",
            "integrity",
            "mutual_authentication",
            "forward_secrecy",
            "client_identity_verification",
        ),
        description=(
            "Mutual TLS: extends TLS 1.3 with client certificate "
            "authentication.  Both client and server present and "
            "validate certificates, establishing mutual trust."
        ),
    )


def kerberos_auth() -> SecurityProtocol:
    """Kerberos authentication protocol (RFC 4120).

    Models the three exchanges of Kerberos:
    1. AS Exchange: client -> KDC for TGT (AS-REQ -> AS-REP)
    2. TGS Exchange: client -> KDC for service ticket (TGS-REQ -> TGS-REP)
    3. AP Exchange: client -> service (AP-REQ -> AP-REP)

    Each exchange can succeed or fail (wrong password, expired ticket, etc.)
    """
    return SecurityProtocol(
        name="Kerberos",
        roles=("Client", "KDC", "ServiceServer"),
        session_type_string=(
            "&{asReq: +{AS_OK: &{asRep: &{tgsReq: "
            "+{TGS_OK: &{tgsRep: &{apReq: "
            "+{AP_OK: &{apRep: end}, AP_FAIL: end}}}, "
            "TGS_FAIL: end}}}, AS_FAIL: end}}"
        ),
        security_properties=(
            "mutual_authentication",
            "single_sign_on",
            "ticket_delegation",
            "replay_protection",
        ),
        description=(
            "Kerberos authentication: AS Exchange (obtain TGT), "
            "TGS Exchange (obtain service ticket), AP Exchange "
            "(authenticate to service).  Each exchange can succeed "
            "or fail independently."
        ),
    )


def saml_sso() -> SecurityProtocol:
    """SAML 2.0 Single Sign-On flow (Web Browser SSO Profile).

    Models the SAML SSO flow:
    1. User accesses service provider (SP)
    2. SP redirects to identity provider (IdP)
    3. User authenticates at IdP
    4. IdP issues SAML assertion
    5. SP validates assertion and grants/denies access

    Covers the HTTP POST binding (most common).
    """
    return SecurityProtocol(
        name="SAML_SSO",
        roles=("User", "ServiceProvider", "IdentityProvider"),
        session_type_string=(
            "&{accessSP: &{redirectToIdP: &{authenticateIdP: "
            "+{AUTH_OK: &{issueAssertion: &{validateAssertion: "
            "+{VALID_ASSERTION: &{grantAccess: end}, "
            "INVALID_ASSERTION: end}}}, AUTH_FAIL: end}}}}"
        ),
        security_properties=(
            "single_sign_on",
            "identity_federation",
            "assertion_integrity",
            "replay_protection",
        ),
        description=(
            "SAML 2.0 SSO: user accesses SP, is redirected to IdP, "
            "authenticates, IdP issues signed SAML assertion, SP "
            "validates and grants or denies access."
        ),
    )


def oauth2_pkce() -> SecurityProtocol:
    """OAuth 2.0 Authorization Code with PKCE (RFC 7636).

    Extends the Authorization Code flow with Proof Key for Code Exchange.
    This is the recommended flow for public clients (SPAs, mobile apps)
    that cannot securely store a client secret.

    Adds code_verifier/code_challenge to prevent authorization code
    interception attacks.
    """
    return SecurityProtocol(
        name="OAuth2PKCE",
        roles=("Client", "AuthorizationServer", "ResourceOwner"),
        session_type_string=(
            "&{generateChallenge: &{requestAuth: &{authenticate: "
            "+{GRANTED: &{issueCode: &{exchangeCodeWithVerifier: "
            "+{TOKEN_OK: end, TOKEN_ERROR: end}}}, DENIED: end}}}}"
        ),
        security_properties=(
            "authentication",
            "authorization",
            "token_confidentiality",
            "code_interception_protection",
            "csrf_protection",
        ),
        description=(
            "OAuth 2.0 with PKCE: extends the authorization code flow "
            "with a code verifier/challenge pair to protect public "
            "clients against authorization code interception."
        ),
    )


def oidc_auth_code() -> SecurityProtocol:
    """OpenID Connect Authorization Code Flow (built on OAuth 2.0).

    Extends OAuth 2.0 auth code flow with ID token for authentication:
    1. Client requests authorization with openid scope
    2. User authenticates
    3. Auth server returns code
    4. Client exchanges code for ID token + access token
    5. Client validates ID token claims
    """
    return SecurityProtocol(
        name="OIDCAuthCode",
        roles=("RelyingParty", "OpenIDProvider", "EndUser"),
        session_type_string=(
            "&{requestAuthOpenID: &{authenticateUser: "
            "+{GRANTED: &{issueCode: &{exchangeForTokens: "
            "&{validateIDToken: +{CLAIMS_VALID: end, "
            "CLAIMS_INVALID: end}}}}, DENIED: end}}}"
        ),
        security_properties=(
            "authentication",
            "identity_verification",
            "token_confidentiality",
            "nonce_protection",
        ),
        description=(
            "OpenID Connect Auth Code: extends OAuth 2.0 with an ID "
            "token for user authentication.  The relying party obtains "
            "both access token and ID token, then validates claims."
        ),
    )


# ---------------------------------------------------------------------------
# Registry of all protocols
# ---------------------------------------------------------------------------

ALL_SECURITY_PROTOCOLS: tuple[SecurityProtocol, ...] = (
    oauth2_auth_code(),
    oauth2_client_credentials(),
    oauth2_pkce(),
    oidc_auth_code(),
    tls_handshake(),
    mtls_handshake(),
    kerberos_auth(),
    saml_sso(),
)
"""All pre-defined security protocols."""


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def security_to_session_type(protocol: SecurityProtocol) -> SessionType:
    """Parse the protocol's session type string into an AST."""
    return parse(protocol.session_type_string)


def verify_security_protocol(
    protocol: SecurityProtocol,
    config: TestGenConfig | None = None,
) -> SecurityAnalysisResult:
    """Run the full verification pipeline on a security protocol.

    Parses the protocol's session type, builds the state space,
    checks lattice properties, generates conformance tests, and
    computes coverage.

    Args:
        protocol: The security protocol to verify.
        config: Optional test generation configuration.

    Returns:
        A complete SecurityAnalysisResult.
    """
    if config is None:
        config = TestGenConfig(
            class_name=f"{protocol.name}ProtocolTest",
            package_name="com.security.conformance",
        )

    # 1. Parse and build state space
    ast = security_to_session_type(protocol)
    ss = build_statespace(ast)

    # 2. Lattice analysis
    lr = check_lattice(ss)
    dist = check_distributive(ss)

    # 3. Test generation
    enum = enumerate_tests(ss, config)
    test_src = generate_test_source(ss, config, protocol.session_type_string)

    # 4. Coverage
    cov = compute_coverage(ss, result=enum)

    return SecurityAnalysisResult(
        protocol=protocol,
        ast=ast,
        state_space=ss,
        lattice_result=lr,
        distributivity=dist,
        enumeration=enum,
        test_source=test_src,
        coverage=cov,
        num_states=len(ss.states),
        num_transitions=len(ss.transitions),
        num_valid_paths=len(enum.valid_paths),
        num_violations=len(enum.violations),
        is_well_formed=lr.is_lattice,
    )


def verify_all_security_protocols() -> list[SecurityAnalysisResult]:
    """Verify all pre-defined security protocols and return results."""
    return [verify_security_protocol(p) for p in ALL_SECURITY_PROTOCOLS]


def check_protocol_evolution(
    old: SecurityProtocol,
    new: SecurityProtocol,
) -> ProtocolEvolutionResult:
    """Check whether a new protocol version is backward-compatible with an old one.

    Uses Gay-Hole subtyping: old <= new means the new version accepts
    all interactions that the old version accepted (the new version may
    offer additional methods, but must support all old ones).

    For security protocols, backward compatibility means that clients
    built for the old protocol version can still interact with servers
    implementing the new version.

    Args:
        old: The older protocol version.
        new: The newer protocol version.

    Returns:
        ProtocolEvolutionResult with compatibility verdict.
    """
    old_ast = security_to_session_type(old)
    new_ast = security_to_session_type(new)

    sub_result = check_subtype(old_ast, new_ast)

    old_analysis = verify_security_protocol(old)
    new_analysis = verify_security_protocol(new)

    return ProtocolEvolutionResult(
        old_protocol=old,
        new_protocol=new,
        subtyping_result=sub_result,
        is_backward_compatible=sub_result.is_subtype,
        old_analysis=old_analysis,
        new_analysis=new_analysis,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_security_report(result: SecurityAnalysisResult) -> str:
    """Format a SecurityAnalysisResult as structured text for terminal output."""
    lines: list[str] = []
    proto = result.protocol

    lines.append("=" * 70)
    lines.append(f"  SECURITY PROTOCOL REPORT: {proto.name}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  {proto.description}")
    lines.append("")

    # Roles
    lines.append("--- Protocol Roles ---")
    for r in proto.roles:
        lines.append(f"  - {r}")
    lines.append("")

    # Security properties
    lines.append("--- Security Properties ---")
    for p in proto.security_properties:
        lines.append(f"  - {p}")
    lines.append("")

    # Session type
    lines.append("--- Session Type ---")
    lines.append(f"  {proto.session_type_string}")
    lines.append("")

    # State space
    lines.append("--- State Space ---")
    lines.append(f"  States:      {result.num_states}")
    lines.append(f"  Transitions: {result.num_transitions}")
    lines.append(f"  Top (init):  {result.state_space.top}")
    lines.append(f"  Bottom (end):{result.state_space.bottom}")
    lines.append("")

    # Lattice
    lines.append("--- Lattice Analysis ---")
    lines.append(f"  Is lattice:      {result.lattice_result.is_lattice}")
    lines.append(f"  Distributive:    {result.distributivity.is_distributive}")
    if not result.lattice_result.is_lattice and result.lattice_result.counterexample:
        lines.append(f"  Counterexample:  {result.lattice_result.counterexample}")
    lines.append("")

    # Test generation
    lines.append("--- Test Generation ---")
    lines.append(f"  Valid paths:     {result.num_valid_paths}")
    lines.append(f"  Violation points:{result.num_violations}")
    lines.append("")

    # Coverage
    lines.append("--- Coverage ---")
    lines.append(f"  State coverage:      {result.coverage.state_coverage:.1%}")
    lines.append(f"  Transition coverage: {result.coverage.transition_coverage:.1%}")
    lines.append("")

    # Verdict
    lines.append("--- Verdict ---")
    if result.is_well_formed:
        lines.append("  PASS: Protocol forms a valid lattice.")
        lines.append("  Every pair of protocol states has a well-defined")
        lines.append("  join and meet, ensuring unambiguous recovery.")
    else:
        lines.append("  FAIL: Protocol does NOT form a valid lattice.")
        lines.append("  Some protocol states lack a join or meet,")
        lines.append("  which may lead to ambiguous recovery scenarios.")
    lines.append("")

    return "\n".join(lines)


def format_evolution_report(result: ProtocolEvolutionResult) -> str:
    """Format a ProtocolEvolutionResult as structured text."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append(f"  PROTOCOL EVOLUTION: {result.old_protocol.name} -> "
                 f"{result.new_protocol.name}")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"  Old protocol: {result.old_protocol.name}")
    lines.append(f"    States: {result.old_analysis.num_states}, "
                 f"Transitions: {result.old_analysis.num_transitions}")
    lines.append(f"  New protocol: {result.new_protocol.name}")
    lines.append(f"    States: {result.new_analysis.num_states}, "
                 f"Transitions: {result.new_analysis.num_transitions}")
    lines.append("")

    lines.append("--- Subtyping Analysis ---")
    lines.append(f"  {result.old_protocol.name} <= {result.new_protocol.name}: "
                 f"{result.subtyping_result.is_subtype}")
    if result.subtyping_result.reason:
        lines.append(f"  Reason: {result.subtyping_result.reason}")
    lines.append("")

    lines.append("--- Backward Compatibility ---")
    if result.is_backward_compatible:
        lines.append("  COMPATIBLE: The new protocol version is backward-compatible.")
        lines.append("  Clients built for the old version can interact with")
        lines.append("  servers implementing the new version.")
    else:
        lines.append("  INCOMPATIBLE: The new protocol version is NOT backward-compatible.")
        lines.append("  Clients built for the old version may fail when")
        lines.append("  interacting with servers implementing the new version.")
    lines.append("")

    return "\n".join(lines)


def format_security_summary(results: list[SecurityAnalysisResult]) -> str:
    """Format a summary table of all verified security protocols."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("  SECURITY PROTOCOL VERIFICATION SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    header = f"  {'Protocol':<25} {'States':>6} {'Trans':>6} {'Lattice':>8} {'Dist':>6} {'Paths':>6}"
    lines.append(header)
    lines.append("  " + "-" * 65)

    for r in results:
        lattice_str = "YES" if r.is_well_formed else "NO"
        dist_str = "YES" if r.distributivity.is_distributive else "NO"
        row = (
            f"  {r.protocol.name:<25} "
            f"{r.num_states:>6} "
            f"{r.num_transitions:>6} "
            f"{lattice_str:>8} "
            f"{dist_str:>6} "
            f"{r.num_valid_paths:>6}"
        )
        lines.append(row)

    lines.append("")
    all_lattice = all(r.is_well_formed for r in results)
    lines.append(f"  All protocols form lattices: {'YES' if all_lattice else 'NO'}")
    lines.append("")

    return "\n".join(lines)
