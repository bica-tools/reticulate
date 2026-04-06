"""Tests for reticulate.opcua_mqtt (Step 79)."""

import pytest

from reticulate.opcua_mqtt import (
    IEC62443_FRS,
    IndustrialProtocol,
    SafetyCheck,
    all_industrial_protocols,
    format_industrial_report,
    is_galois_pair,
    is_phi_total,
    is_psi_total,
    mqtt_sparkplug_edge,
    mqtt_tls_secured,
    opcua_client_session,
    opcua_secure_channel_renew,
    opcua_subscription,
    phi_lattice_to_checks,
    psi_checks_to_lattice,
    verify_industrial_protocol,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def test_iec62443_has_seven_frs():
    assert len(IEC62443_FRS) == 7
    assert all(fr.startswith("FR") for fr in IEC62443_FRS)


def test_opcua_client_session_fields():
    p = opcua_client_session()
    assert p.name == "OPCUA_ClientSession"
    assert p.family == "OPCUA"
    assert "hello" in p.session_type_string
    assert "closeSession" in p.session_type_string


def test_opcua_subscription_fields():
    p = opcua_subscription()
    assert p.family == "OPCUA"
    assert "createSubscription" in p.session_type_string


def test_opcua_secure_channel_renew_fields():
    p = opcua_secure_channel_renew()
    assert "renewSecureChannel" in p.session_type_string


def test_mqtt_sparkplug_edge_fields():
    p = mqtt_sparkplug_edge()
    assert p.family == "MQTT"
    assert "nbirth" in p.session_type_string
    assert "ndeath" in p.session_type_string


def test_mqtt_tls_secured_fields():
    p = mqtt_tls_secured()
    assert "tlsHandshake" in p.session_type_string
    assert "FR4_DataConfidentiality" in p.iec62443_scope


def test_all_industrial_protocols_count():
    assert len(all_industrial_protocols()) == 5


def test_all_protocols_parse():
    for p in all_industrial_protocols():
        ast = parse(p.session_type_string)
        assert ast is not None


# ---------------------------------------------------------------------------
# Lattice properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("proto_fn", [
    opcua_client_session,
    opcua_subscription,
    opcua_secure_channel_renew,
    mqtt_sparkplug_edge,
    mqtt_tls_secured,
])
def test_all_protocols_are_lattices(proto_fn):
    r = verify_industrial_protocol(proto_fn())
    assert r.is_well_formed
    assert r.lattice_result.is_lattice


def test_sparkplug_is_distributive():
    r = verify_industrial_protocol(mqtt_sparkplug_edge())
    assert r.distributivity.is_distributive


def test_state_counts_positive():
    for p in all_industrial_protocols():
        r = verify_industrial_protocol(p)
        assert r.num_states >= 3
        assert r.num_transitions >= 2


# ---------------------------------------------------------------------------
# phi / psi morphisms
# ---------------------------------------------------------------------------

def test_phi_is_total_on_all_protocols():
    for p in all_industrial_protocols():
        ss = build_statespace(parse(p.session_type_string))
        checks = phi_lattice_to_checks(ss)
        assert is_phi_total(ss, checks)


def test_psi_is_total_on_phi_image():
    for p in all_industrial_protocols():
        ss = build_statespace(parse(p.session_type_string))
        checks = phi_lattice_to_checks(ss)
        assert is_psi_total(ss, checks)


def test_psi_phi_recovers_reachable():
    for p in all_industrial_protocols():
        ss = build_statespace(parse(p.session_type_string))
        checks = phi_lattice_to_checks(ss)
        recovered = set(psi_checks_to_lattice(checks, ss))
        # All transition targets plus the top state
        expected = {ss.top}
        for src, _, tgt in ss.transitions:
            expected.add(tgt)
        assert recovered == expected


def test_galois_pair_holds_for_all():
    for p in all_industrial_protocols():
        ss = build_statespace(parse(p.session_type_string))
        assert is_galois_pair(ss)


def test_phi_maps_tls_to_authentication_fr():
    ss = build_statespace(parse("&{tlsHandshake: end}"))
    checks = phi_lattice_to_checks(ss)
    frs = {c.fr for c in checks}
    assert "FR1_IdentificationAndAuthenticationControl" in frs


def test_phi_maps_publish_to_integrity():
    ss = build_statespace(parse("&{publish: end}"))
    checks = phi_lattice_to_checks(ss)
    assert any(c.fr == "FR3_SystemIntegrity" for c in checks)


def test_phi_severity_critical_on_failure_label():
    ss = build_statespace(parse("&{connect: +{CONNACK_FAIL: end, OK: end}}"))
    checks = phi_lattice_to_checks(ss)
    assert any(c.severity == "critical" for c in checks)


def test_phi_severity_info_for_benign_label():
    ss = build_statespace(parse("&{read: end}"))
    checks = phi_lattice_to_checks(ss)
    # the read transition has info severity
    assert any(c.severity == "info" and "read" in c.action for c in checks)


def test_safety_check_immutable():
    c = SafetyCheck(state_id=1, fr="FR3_SystemIntegrity",
                    action="noop", severity="info")
    with pytest.raises(Exception):
        c.state_id = 99  # type: ignore


def test_psi_filters_out_unreachable_state_ids():
    ss = build_statespace(parse("&{a: end}"))
    fake = (SafetyCheck(state_id=999, fr="FR1_IdentificationAndAuthenticationControl",
                        action="x", severity="info"),)
    recovered = psi_checks_to_lattice(fake, ss)
    assert 999 not in recovered


# ---------------------------------------------------------------------------
# Verification driver + report
# ---------------------------------------------------------------------------

def test_verify_returns_complete_result():
    r = verify_industrial_protocol(opcua_client_session())
    assert r.protocol.name == "OPCUA_ClientSession"
    assert r.ast is not None
    assert r.num_states == len(r.state_space.states)
    assert len(r.safety_checks) > 0


def test_format_industrial_report_mentions_protocol_name():
    r = verify_industrial_protocol(opcua_client_session())
    s = format_industrial_report(r)
    assert "OPCUA_ClientSession" in s
    assert "lattice:" in s
    assert "phi total" in s


def test_opcua_client_session_emits_authentication_checks():
    r = verify_industrial_protocol(opcua_client_session())
    auth_checks = [c for c in r.safety_checks
                   if c.fr == "FR1_IdentificationAndAuthenticationControl"]
    assert len(auth_checks) >= 2


def test_sparkplug_emits_timely_response_checks():
    r = verify_industrial_protocol(mqtt_sparkplug_edge())
    timely = [c for c in r.safety_checks if c.fr == "FR6_TimelyResponseToEvents"]
    assert len(timely) >= 1


def test_tls_mqtt_emits_confidentiality_scope():
    p = mqtt_tls_secured()
    assert "FR4_DataConfidentiality" in p.iec62443_scope


def test_phi_total_count_matches_transitions_plus_one():
    for p in all_industrial_protocols():
        ss = build_statespace(parse(p.session_type_string))
        checks = phi_lattice_to_checks(ss)
        assert len(checks) == len(ss.transitions) + 1


def test_verification_result_galois_flag():
    for p in all_industrial_protocols():
        r = verify_industrial_protocol(p)
        assert r.galois is True
        assert r.phi_total is True
        assert r.psi_total is True
