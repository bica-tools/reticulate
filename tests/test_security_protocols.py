"""Tests for security_protocols.py — Step 89.

Verifies security protocol definitions, lattice properties,
subtyping for protocol evolution, and analysis pipeline.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.security_protocols import (
    ALL_SECURITY_PROTOCOLS,
    ProtocolEvolutionResult,
    SecurityAnalysisResult,
    SecurityProtocol,
    check_protocol_evolution,
    format_evolution_report,
    format_security_report,
    format_security_summary,
    kerberos_auth,
    mtls_handshake,
    oauth2_auth_code,
    oauth2_client_credentials,
    oauth2_pkce,
    oidc_auth_code,
    saml_sso,
    security_to_session_type,
    tls_handshake,
    verify_all_security_protocols,
    verify_security_protocol,
)
from reticulate.statespace import build_statespace
from reticulate.subtyping import is_subtype


# ---------------------------------------------------------------------------
# Protocol definition tests
# ---------------------------------------------------------------------------

class TestProtocolDefinitions:
    """Test that each protocol factory returns a valid SecurityProtocol."""

    def test_oauth2_auth_code_structure(self) -> None:
        proto = oauth2_auth_code()
        assert proto.name == "OAuth2AuthCode"
        assert len(proto.roles) == 3
        assert "Client" in proto.roles
        assert "authentication" in proto.security_properties
        assert len(proto.session_type_string) > 0

    def test_oauth2_client_credentials_structure(self) -> None:
        proto = oauth2_client_credentials()
        assert proto.name == "OAuth2ClientCredentials"
        assert len(proto.roles) == 2
        assert "client_authentication" in proto.security_properties

    def test_tls_handshake_structure(self) -> None:
        proto = tls_handshake()
        assert proto.name == "TLS13Handshake"
        assert "Client" in proto.roles and "Server" in proto.roles
        assert "forward_secrecy" in proto.security_properties

    def test_mtls_handshake_structure(self) -> None:
        proto = mtls_handshake()
        assert proto.name == "MutualTLS"
        assert "mutual_authentication" in proto.security_properties
        assert "client_identity_verification" in proto.security_properties

    def test_kerberos_structure(self) -> None:
        proto = kerberos_auth()
        assert proto.name == "Kerberos"
        assert len(proto.roles) == 3
        assert "single_sign_on" in proto.security_properties
        assert "replay_protection" in proto.security_properties

    def test_saml_sso_structure(self) -> None:
        proto = saml_sso()
        assert proto.name == "SAML_SSO"
        assert "IdentityProvider" in proto.roles
        assert "single_sign_on" in proto.security_properties

    def test_oauth2_pkce_structure(self) -> None:
        proto = oauth2_pkce()
        assert proto.name == "OAuth2PKCE"
        assert "code_interception_protection" in proto.security_properties

    def test_oidc_auth_code_structure(self) -> None:
        proto = oidc_auth_code()
        assert proto.name == "OIDCAuthCode"
        assert "identity_verification" in proto.security_properties


class TestProtocolParsing:
    """Test that each protocol's session type string parses without error."""

    @pytest.mark.parametrize("protocol", ALL_SECURITY_PROTOCOLS,
                             ids=[p.name for p in ALL_SECURITY_PROTOCOLS])
    def test_parseable(self, protocol: SecurityProtocol) -> None:
        ast = security_to_session_type(protocol)
        assert ast is not None

    @pytest.mark.parametrize("protocol", ALL_SECURITY_PROTOCOLS,
                             ids=[p.name for p in ALL_SECURITY_PROTOCOLS])
    def test_builds_statespace(self, protocol: SecurityProtocol) -> None:
        ast = security_to_session_type(protocol)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2
        assert len(ss.transitions) >= 1


# ---------------------------------------------------------------------------
# Lattice property tests
# ---------------------------------------------------------------------------

class TestLatticeProperties:
    """All security protocols must form lattices."""

    @pytest.mark.parametrize("protocol", ALL_SECURITY_PROTOCOLS,
                             ids=[p.name for p in ALL_SECURITY_PROTOCOLS])
    def test_is_lattice(self, protocol: SecurityProtocol) -> None:
        result = verify_security_protocol(protocol)
        assert result.is_well_formed, (
            f"{protocol.name} does not form a lattice: "
            f"{result.lattice_result.counterexample}"
        )

    @pytest.mark.parametrize("protocol", ALL_SECURITY_PROTOCOLS,
                             ids=[p.name for p in ALL_SECURITY_PROTOCOLS])
    def test_is_distributive(self, protocol: SecurityProtocol) -> None:
        result = verify_security_protocol(protocol)
        assert result.distributivity.is_distributive, (
            f"{protocol.name} is not distributive"
        )


# ---------------------------------------------------------------------------
# Verification pipeline tests
# ---------------------------------------------------------------------------

class TestVerificationPipeline:
    """Test the full verification pipeline."""

    def test_verify_oauth2_auth_code(self) -> None:
        result = verify_security_protocol(oauth2_auth_code())
        assert isinstance(result, SecurityAnalysisResult)
        assert result.num_states > 0
        assert result.num_transitions > 0
        assert result.num_valid_paths > 0
        assert result.num_violations >= 0
        assert len(result.test_source) > 0

    def test_verify_tls_handshake(self) -> None:
        result = verify_security_protocol(tls_handshake())
        assert result.is_well_formed
        assert result.num_states >= 5  # At least 5 states in TLS

    def test_verify_kerberos(self) -> None:
        result = verify_security_protocol(kerberos_auth())
        assert result.is_well_formed
        assert result.num_states >= 5

    def test_verify_all(self) -> None:
        results = verify_all_security_protocols()
        assert len(results) == len(ALL_SECURITY_PROTOCOLS)
        for r in results:
            assert r.is_well_formed

    def test_coverage_positive(self) -> None:
        result = verify_security_protocol(oauth2_auth_code())
        assert result.coverage.state_coverage > 0
        assert result.coverage.transition_coverage > 0

    def test_custom_config(self) -> None:
        from reticulate.testgen import TestGenConfig
        config = TestGenConfig(
            class_name="CustomOAuth2Test",
            package_name="com.custom.test",
        )
        result = verify_security_protocol(oauth2_auth_code(), config=config)
        assert "CustomOAuth2Test" in result.test_source


# ---------------------------------------------------------------------------
# Protocol evolution / subtyping tests
# ---------------------------------------------------------------------------

class TestProtocolEvolution:
    """Test backward compatibility checks via subtyping."""

    def test_oauth2_to_pkce_evolution(self) -> None:
        """PKCE extends auth code with generateChallenge; old is NOT subtype of new."""
        result = check_protocol_evolution(oauth2_auth_code(), oauth2_pkce())
        assert isinstance(result, ProtocolEvolutionResult)
        # Old auth code is NOT subtype of PKCE (different first step)
        assert not result.is_backward_compatible

    def test_tls_to_mtls_evolution(self) -> None:
        """mTLS extends TLS; they diverge after server cert validation."""
        result = check_protocol_evolution(tls_handshake(), mtls_handshake())
        assert isinstance(result, ProtocolEvolutionResult)

    def test_self_evolution_is_compatible(self) -> None:
        """Any protocol is backward-compatible with itself (reflexivity)."""
        proto = oauth2_client_credentials()
        old_ast = security_to_session_type(proto)
        new_ast = security_to_session_type(proto)
        assert is_subtype(old_ast, new_ast)

    def test_client_credentials_subtype_of_auth_code(self) -> None:
        """Client credentials is NOT a subtype of auth code (different structure)."""
        cc = security_to_session_type(oauth2_client_credentials())
        ac = security_to_session_type(oauth2_auth_code())
        assert not is_subtype(cc, ac)

    def test_evolution_result_has_analyses(self) -> None:
        result = check_protocol_evolution(
            oauth2_client_credentials(),
            oauth2_client_credentials(),
        )
        assert result.old_analysis.is_well_formed
        assert result.new_analysis.is_well_formed


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------

class TestReportFormatting:
    """Test that report formatters produce non-empty, structured output."""

    def test_security_report_format(self) -> None:
        result = verify_security_protocol(oauth2_auth_code())
        report = format_security_report(result)
        assert "OAuth2AuthCode" in report
        assert "SECURITY PROTOCOL REPORT" in report
        assert "Lattice Analysis" in report
        assert "Verdict" in report

    def test_evolution_report_format(self) -> None:
        result = check_protocol_evolution(
            oauth2_client_credentials(),
            oauth2_client_credentials(),
        )
        report = format_evolution_report(result)
        assert "PROTOCOL EVOLUTION" in report
        assert "Backward Compatibility" in report

    def test_summary_format(self) -> None:
        results = verify_all_security_protocols()
        summary = format_security_summary(results)
        assert "SUMMARY" in summary
        assert "OAuth2AuthCode" in summary
        assert "Kerberos" in summary

    def test_report_includes_security_properties(self) -> None:
        result = verify_security_protocol(tls_handshake())
        report = format_security_report(result)
        assert "forward_secrecy" in report
        assert "confidentiality" in report


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test the ALL_SECURITY_PROTOCOLS registry."""

    def test_registry_count(self) -> None:
        assert len(ALL_SECURITY_PROTOCOLS) == 8

    def test_unique_names(self) -> None:
        names = [p.name for p in ALL_SECURITY_PROTOCOLS]
        assert len(names) == len(set(names))

    def test_all_have_descriptions(self) -> None:
        for p in ALL_SECURITY_PROTOCOLS:
            assert len(p.description) > 0

    def test_all_have_security_properties(self) -> None:
        for p in ALL_SECURITY_PROTOCOLS:
            assert len(p.security_properties) >= 2
