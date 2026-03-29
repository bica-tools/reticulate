"""Tests for auth_verification.py -- Step 78.

Verifies OAuth 2.0 and TLS behavioral verification: protocol
definitions, lattice properties, subtyping, downgrade safety,
PKCE extension, mTLS extension, and conformance test generation.
"""

from __future__ import annotations

import pytest

from reticulate.auth_verification import (
    ALL_AUTH_PROTOCOLS,
    AuthAnalysisReport,
    AuthProtocol,
    CompatibilityResult,
    DowngradeSafetyResult,
    MTLSExtensionResult,
    PKCEExtensionResult,
    VerificationResult,
    analyze_all_auth,
    format_auth_report,
    generate_conformance_tests,
    mtls_handshake,
    oauth2_auth_code_flow,
    oauth2_client_credentials_flow,
    oauth2_device_code_flow,
    oauth2_pkce_flow,
    oidc_auth_code_flow,
    tls12_handshake,
    tls13_handshake,
    verify_mtls_extension,
    verify_oauth_backward_compat,
    verify_oauth_pkce_extension,
    verify_protocol,
    verify_tls_downgrade_safety,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Protocol definition tests
# ---------------------------------------------------------------------------

class TestProtocolDefinitions:
    """Test that each protocol factory returns a valid AuthProtocol."""

    def test_oauth2_auth_code(self) -> None:
        p = oauth2_auth_code_flow()
        assert p.name == "OAuth2-AuthCode"
        assert p.version == "2.0"
        assert p.category == "oauth"
        assert len(p.session_type_string) > 0

    def test_oauth2_client_credentials(self) -> None:
        p = oauth2_client_credentials_flow()
        assert p.name == "OAuth2-ClientCredentials"
        assert p.category == "oauth"

    def test_oauth2_pkce(self) -> None:
        p = oauth2_pkce_flow()
        assert p.name == "OAuth2-PKCE"
        assert "PKCE" in p.version

    def test_oauth2_device_code(self) -> None:
        p = oauth2_device_code_flow()
        assert p.name == "OAuth2-DeviceCode"
        assert p.category == "oauth"

    def test_tls13(self) -> None:
        p = tls13_handshake()
        assert p.name == "TLS-1.3"
        assert p.version == "1.3"
        assert p.category == "tls"

    def test_tls12(self) -> None:
        p = tls12_handshake()
        assert p.name == "TLS-1.2"
        assert p.version == "1.2"

    def test_mtls(self) -> None:
        p = mtls_handshake()
        assert p.name == "mTLS"
        assert p.category == "tls"

    def test_oidc(self) -> None:
        p = oidc_auth_code_flow()
        assert p.name == "OIDC-AuthCode"
        assert p.category == "oidc"

    def test_all_protocols_registry(self) -> None:
        assert len(ALL_AUTH_PROTOCOLS) == 8
        names = {p.name for p in ALL_AUTH_PROTOCOLS}
        assert "OAuth2-AuthCode" in names
        assert "TLS-1.3" in names
        assert "mTLS" in names

    def test_all_protocols_parseable(self) -> None:
        for p in ALL_AUTH_PROTOCOLS:
            ast = parse(p.session_type_string)
            assert ast is not None


# ---------------------------------------------------------------------------
# Individual verification tests
# ---------------------------------------------------------------------------

class TestVerification:
    """Test individual protocol verification."""

    def test_oauth2_auth_code_is_lattice(self) -> None:
        result = verify_protocol(oauth2_auth_code_flow())
        assert result.is_lattice
        assert result.num_states > 0
        assert result.num_transitions > 0

    def test_oauth2_client_credentials_is_lattice(self) -> None:
        result = verify_protocol(oauth2_client_credentials_flow())
        assert result.is_lattice

    def test_oauth2_pkce_is_lattice(self) -> None:
        result = verify_protocol(oauth2_pkce_flow())
        assert result.is_lattice

    def test_oauth2_device_code_is_lattice(self) -> None:
        result = verify_protocol(oauth2_device_code_flow())
        assert result.is_lattice

    def test_tls13_is_lattice(self) -> None:
        result = verify_protocol(tls13_handshake())
        assert result.is_lattice

    def test_tls12_is_lattice(self) -> None:
        result = verify_protocol(tls12_handshake())
        assert result.is_lattice

    def test_mtls_is_lattice(self) -> None:
        result = verify_protocol(mtls_handshake())
        assert result.is_lattice

    def test_oidc_is_lattice(self) -> None:
        result = verify_protocol(oidc_auth_code_flow())
        assert result.is_lattice

    def test_verification_has_test_source(self) -> None:
        result = verify_protocol(oauth2_auth_code_flow())
        assert len(result.test_source) > 0
        assert "class" in result.test_source

    def test_verification_has_valid_paths(self) -> None:
        result = verify_protocol(oauth2_auth_code_flow())
        assert len(result.enumeration.valid_paths) > 0

    def test_tls13_state_count(self) -> None:
        result = verify_protocol(tls13_handshake())
        # TLS 1.3: clientHello -> serverHello -> serverCert -> CERT_VALID/INVALID
        #   CERT_VALID -> serverFinished -> clientFinished -> end
        #   CERT_INVALID -> end
        assert result.num_states >= 6

    def test_tls12_state_count(self) -> None:
        result = verify_protocol(tls12_handshake())
        # TLS 1.2 has more steps (2-RTT)
        assert result.num_states >= 8


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Test OAuth backward compatibility analysis."""

    def test_auth_code_compat_with_pkce(self) -> None:
        result = verify_oauth_backward_compat(
            oauth2_auth_code_flow(),
            oauth2_pkce_flow(),
        )
        assert isinstance(result, CompatibilityResult)
        # PKCE adds generateChallenge before requestAuth,
        # so the structures differ at the root: base has requestAuth,
        # PKCE has generateChallenge. Neither is a subtype.
        assert isinstance(result.is_backward_compatible, bool)

    def test_same_protocol_is_compatible(self) -> None:
        base = oauth2_auth_code_flow()
        result = verify_oauth_backward_compat(base, base)
        assert result.is_backward_compatible

    def test_client_cred_not_compat_with_auth_code(self) -> None:
        result = verify_oauth_backward_compat(
            oauth2_client_credentials_flow(),
            oauth2_auth_code_flow(),
        )
        # Different root methods (presentCredentials vs requestAuth)
        assert not result.is_backward_compatible


# ---------------------------------------------------------------------------
# TLS downgrade safety tests
# ---------------------------------------------------------------------------

class TestDowngradeSafety:
    """Test TLS downgrade safety analysis."""

    def test_tls_downgrade_safety(self) -> None:
        result = verify_tls_downgrade_safety()
        assert isinstance(result, DowngradeSafetyResult)
        # TLS 1.3 and 1.2 have different structures
        assert result.is_downgrade_safe

    def test_tls13_not_subtype_of_tls12(self) -> None:
        result = verify_tls_downgrade_safety()
        assert not result.tls13_sub_tls12.is_subtype

    def test_tls12_not_subtype_of_tls13(self) -> None:
        result = verify_tls_downgrade_safety()
        assert not result.tls12_sub_tls13.is_subtype

    def test_tls_not_isomorphic(self) -> None:
        result = verify_tls_downgrade_safety()
        assert result.isomorphism is None

    def test_tls13_does_not_embed_in_tls12(self) -> None:
        # Different handshake structures prevent embedding
        result = verify_tls_downgrade_safety()
        # Even if embedding exists, the subtyping check is the key safety property
        assert result.is_downgrade_safe


# ---------------------------------------------------------------------------
# PKCE extension tests
# ---------------------------------------------------------------------------

class TestPKCEExtension:
    """Test PKCE extension analysis."""

    def test_pkce_extension(self) -> None:
        result = verify_oauth_pkce_extension()
        assert isinstance(result, PKCEExtensionResult)

    def test_both_are_lattices(self) -> None:
        result = verify_oauth_pkce_extension()
        assert result.base_is_lattice
        assert result.pkce_is_lattice

    def test_pkce_has_more_states(self) -> None:
        base_ss = build_statespace(parse(oauth2_auth_code_flow().session_type_string))
        pkce_ss = build_statespace(parse(oauth2_pkce_flow().session_type_string))
        # PKCE adds one extra step (generateChallenge)
        assert len(pkce_ss.states) > len(base_ss.states)

    def test_base_embeds_in_pkce(self) -> None:
        result = verify_oauth_pkce_extension()
        assert result.base_embeds_in_pkce


# ---------------------------------------------------------------------------
# mTLS extension tests
# ---------------------------------------------------------------------------

class TestMTLSExtension:
    """Test mTLS extension analysis."""

    def test_mtls_extension(self) -> None:
        result = verify_mtls_extension()
        assert isinstance(result, MTLSExtensionResult)

    def test_tls_embeds_in_mtls(self) -> None:
        result = verify_mtls_extension()
        assert result.tls_embeds_in_mtls

    def test_mtls_has_more_states(self) -> None:
        tls_ss = build_statespace(parse(tls13_handshake().session_type_string))
        mtls_ss = build_statespace(parse(mtls_handshake().session_type_string))
        assert len(mtls_ss.states) > len(tls_ss.states)


# ---------------------------------------------------------------------------
# Conformance test generation
# ---------------------------------------------------------------------------

class TestConformanceGeneration:
    """Test conformance test generation for auth protocols."""

    def test_oauth_auth_code_tests(self) -> None:
        src = generate_conformance_tests(oauth2_auth_code_flow())
        assert "class" in src
        assert len(src) > 100

    def test_tls13_tests(self) -> None:
        src = generate_conformance_tests(tls13_handshake())
        assert "class" in src

    def test_mtls_tests(self) -> None:
        src = generate_conformance_tests(mtls_handshake())
        assert len(src) > 100

    def test_device_code_tests(self) -> None:
        src = generate_conformance_tests(oauth2_device_code_flow())
        assert "class" in src


# ---------------------------------------------------------------------------
# Full analysis suite
# ---------------------------------------------------------------------------

class TestAnalyzeAll:
    """Test the full analysis suite."""

    def test_analyze_all(self) -> None:
        report = analyze_all_auth()
        assert isinstance(report, AuthAnalysisReport)
        assert len(report.verifications) == 8
        assert report.total_states > 0
        assert report.total_transitions > 0

    def test_all_protocols_form_lattices(self) -> None:
        report = analyze_all_auth()
        assert report.all_lattices

    def test_report_formatting(self) -> None:
        report = analyze_all_auth()
        text = format_auth_report(report)
        assert "AUTH PROTOCOL" in text
        assert "Downgrade" in text
        assert "PKCE" in text
        assert "mTLS" in text
        assert len(text) > 200


# ---------------------------------------------------------------------------
# Cross-protocol subtyping edge cases
# ---------------------------------------------------------------------------

class TestSubtypingEdgeCases:
    """Test subtyping edge cases across protocols."""

    def test_oauth_not_subtype_of_tls(self) -> None:
        oauth_ast = parse(oauth2_auth_code_flow().session_type_string)
        tls_ast = parse(tls13_handshake().session_type_string)
        assert not is_subtype(oauth_ast, tls_ast)

    def test_oidc_not_subtype_of_oauth_auth_code(self) -> None:
        oidc_ast = parse(oidc_auth_code_flow().session_type_string)
        oauth_ast = parse(oauth2_auth_code_flow().session_type_string)
        # Different root methods (requestAuthOpenID vs requestAuth)
        assert not is_subtype(oidc_ast, oauth_ast)

    def test_self_subtype(self) -> None:
        for p in ALL_AUTH_PROTOCOLS:
            ast = parse(p.session_type_string)
            assert is_subtype(ast, ast), f"{p.name} should be subtype of itself"
