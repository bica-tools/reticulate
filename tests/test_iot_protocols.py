"""Tests for iot_protocols.py — Step 86.

Verifies IoT protocol definitions, lattice properties,
QoS level comparison, analysis pipeline, and report formatting.
"""

from __future__ import annotations

import pytest

from reticulate.parser import parse
from reticulate.iot_protocols import (
    ALL_IOT_PROTOCOLS,
    IoTAnalysisResult,
    IoTProtocol,
    coap_block_transfer,
    coap_observe,
    coap_request_response,
    compare_qos_levels,
    format_iot_report,
    format_iot_summary,
    iot_to_session_type,
    mqtt_last_will,
    mqtt_publish_subscribe,
    mqtt_retained_messages,
    verify_all_iot_protocols,
    verify_iot_protocol,
    zigbee_data_transfer,
    zigbee_join,
)
from reticulate.statespace import build_statespace


# ---------------------------------------------------------------------------
# Protocol definition tests
# ---------------------------------------------------------------------------

class TestMQTTDefinitions:
    """Test MQTT protocol factory functions."""

    def test_mqtt_qos0_structure(self) -> None:
        proto = mqtt_publish_subscribe(qos=0)
        assert proto.name == "MQTT_QoS0"
        assert proto.qos_level == 0
        assert proto.transport == "TCP"
        assert "fire-and-forget" in proto.description.lower() or "at most once" in proto.description.lower()

    def test_mqtt_qos1_structure(self) -> None:
        proto = mqtt_publish_subscribe(qos=1)
        assert proto.name == "MQTT_QoS1"
        assert proto.qos_level == 1
        assert "puback" in proto.description.lower() or "at least once" in proto.description.lower()

    def test_mqtt_qos2_structure(self) -> None:
        proto = mqtt_publish_subscribe(qos=2)
        assert proto.name == "MQTT_QoS2"
        assert proto.qos_level == 2
        assert "exactly once" in proto.description.lower() or "pubcomp" in proto.description.lower()

    def test_mqtt_invalid_qos_raises(self) -> None:
        with pytest.raises(ValueError, match="QoS must be 0, 1, or 2"):
            mqtt_publish_subscribe(qos=3)

    def test_mqtt_retained_structure(self) -> None:
        proto = mqtt_retained_messages()
        assert proto.name == "MQTT_Retained"
        assert "retained" in proto.description.lower()

    def test_mqtt_last_will_structure(self) -> None:
        proto = mqtt_last_will()
        assert proto.name == "MQTT_LastWill"
        assert "will" in proto.description.lower()

    def test_mqtt_qos_state_space_grows_with_qos(self) -> None:
        """Higher QoS levels produce larger state spaces."""
        sizes = []
        for qos in (0, 1, 2):
            proto = mqtt_publish_subscribe(qos=qos)
            ast = iot_to_session_type(proto)
            ss = build_statespace(ast)
            sizes.append(len(ss.states))
        assert sizes[0] < sizes[1] < sizes[2]


class TestCoAPDefinitions:
    """Test CoAP protocol factory functions."""

    def test_coap_con_structure(self) -> None:
        proto = coap_request_response(confirmable=True)
        assert proto.name == "CoAP_CON"
        assert proto.transport == "UDP"
        assert proto.constrained is True
        assert proto.qos_level == 1

    def test_coap_non_structure(self) -> None:
        proto = coap_request_response(confirmable=False)
        assert proto.name == "CoAP_NON"
        assert proto.qos_level == 0
        assert proto.constrained is True

    def test_coap_observe_structure(self) -> None:
        proto = coap_observe()
        assert proto.name == "CoAP_Observe"
        assert "observe" in proto.description.lower()
        assert proto.constrained is True

    def test_coap_block_transfer_structure(self) -> None:
        proto = coap_block_transfer()
        assert proto.name == "CoAP_BlockTransfer"
        assert "block" in proto.description.lower()

    def test_coap_non_smaller_than_con(self) -> None:
        """Non-confirmable has fewer states than confirmable."""
        con = coap_request_response(confirmable=True)
        non = coap_request_response(confirmable=False)
        con_ss = build_statespace(iot_to_session_type(con))
        non_ss = build_statespace(iot_to_session_type(non))
        assert len(non_ss.states) < len(con_ss.states)


class TestZigbeeDefinitions:
    """Test Zigbee protocol factory functions."""

    def test_zigbee_join_structure(self) -> None:
        proto = zigbee_join()
        assert proto.name == "Zigbee_Join"
        assert proto.transport == "IEEE802.15.4"
        assert proto.constrained is True
        assert proto.qos_level == -1

    def test_zigbee_data_transfer_structure(self) -> None:
        proto = zigbee_data_transfer()
        assert proto.name == "Zigbee_DataTransfer"
        assert proto.constrained is True


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestProtocolParsing:
    """Test that each protocol's session type string parses without error."""

    @pytest.mark.parametrize(
        "protocol",
        ALL_IOT_PROTOCOLS,
        ids=[p.name for p in ALL_IOT_PROTOCOLS],
    )
    def test_parseable(self, protocol: IoTProtocol) -> None:
        ast = iot_to_session_type(protocol)
        assert ast is not None

    @pytest.mark.parametrize(
        "protocol",
        ALL_IOT_PROTOCOLS,
        ids=[p.name for p in ALL_IOT_PROTOCOLS],
    )
    def test_state_space_buildable(self, protocol: IoTProtocol) -> None:
        ast = iot_to_session_type(protocol)
        ss = build_statespace(ast)
        assert len(ss.states) >= 2  # at least top and bottom
        assert ss.top in ss.states
        assert ss.bottom in ss.states


# ---------------------------------------------------------------------------
# Lattice property tests
# ---------------------------------------------------------------------------

class TestLatticeProperties:
    """Test that all IoT protocols form lattices."""

    @pytest.mark.parametrize(
        "protocol",
        ALL_IOT_PROTOCOLS,
        ids=[p.name for p in ALL_IOT_PROTOCOLS],
    )
    def test_is_lattice(self, protocol: IoTProtocol) -> None:
        result = verify_iot_protocol(protocol)
        assert result.lattice_result.is_lattice, (
            f"{protocol.name} does not form a lattice: "
            f"{result.lattice_result.counterexample}"
        )

    @pytest.mark.parametrize(
        "protocol",
        ALL_IOT_PROTOCOLS,
        ids=[p.name for p in ALL_IOT_PROTOCOLS],
    )
    def test_is_well_formed(self, protocol: IoTProtocol) -> None:
        result = verify_iot_protocol(protocol)
        assert result.is_well_formed


# ---------------------------------------------------------------------------
# Verification pipeline tests
# ---------------------------------------------------------------------------

class TestVerification:
    """Test the full verification pipeline."""

    def test_verify_mqtt_qos0(self) -> None:
        result = verify_iot_protocol(mqtt_publish_subscribe(qos=0))
        assert isinstance(result, IoTAnalysisResult)
        assert result.num_states > 0
        assert result.num_transitions > 0
        assert result.num_valid_paths > 0

    def test_verify_coap_con(self) -> None:
        result = verify_iot_protocol(coap_request_response(confirmable=True))
        assert result.is_well_formed
        assert result.num_violations > 0

    def test_verify_zigbee_join(self) -> None:
        result = verify_iot_protocol(zigbee_join())
        assert result.is_well_formed
        assert len(result.test_source) > 0

    def test_verify_generates_monitor(self) -> None:
        result = verify_iot_protocol(mqtt_publish_subscribe(qos=1))
        assert result.monitor is not None
        assert len(result.monitor.source_code) > 0

    def test_verify_all(self) -> None:
        results = verify_all_iot_protocols()
        assert len(results) == len(ALL_IOT_PROTOCOLS)
        assert all(r.is_well_formed for r in results)

    def test_coverage_positive(self) -> None:
        result = verify_iot_protocol(mqtt_publish_subscribe(qos=1))
        assert result.coverage.state_coverage > 0
        assert result.coverage.transition_coverage > 0


# ---------------------------------------------------------------------------
# QoS comparison tests
# ---------------------------------------------------------------------------

class TestQoSComparison:
    """Test MQTT QoS level comparison."""

    def test_compare_qos_levels_returns_three(self) -> None:
        results = compare_qos_levels()
        assert len(results) == 3
        assert "QoS0" in results
        assert "QoS1" in results
        assert "QoS2" in results

    def test_qos_state_space_monotonic(self) -> None:
        results = compare_qos_levels()
        assert results["QoS0"].num_states < results["QoS1"].num_states
        assert results["QoS1"].num_states < results["QoS2"].num_states

    def test_all_qos_levels_are_lattices(self) -> None:
        results = compare_qos_levels()
        for key, r in results.items():
            assert r.is_well_formed, f"{key} is not a lattice"


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------

class TestFormatting:
    """Test report formatting functions."""

    def test_format_iot_report_contains_sections(self) -> None:
        result = verify_iot_protocol(mqtt_publish_subscribe(qos=1))
        report = format_iot_report(result)
        assert "IOT PROTOCOL REPORT" in report
        assert "MQTT_QoS1" in report
        assert "Lattice Analysis" in report
        assert "Coverage" in report
        assert "Runtime Monitor" in report
        assert "Verdict" in report

    def test_format_iot_report_pass_verdict(self) -> None:
        result = verify_iot_protocol(coap_observe())
        report = format_iot_report(result)
        assert "PASS" in report

    def test_format_iot_summary_table(self) -> None:
        results = verify_all_iot_protocols()
        summary = format_iot_summary(results)
        assert "IOT PROTOCOL VERIFICATION SUMMARY" in summary
        assert "MQTT_QoS0" in summary
        assert "CoAP_CON" in summary
        assert "Zigbee_Join" in summary
        assert "All protocols form lattices: YES" in summary

    def test_format_report_shows_transport(self) -> None:
        result = verify_iot_protocol(coap_request_response(confirmable=True))
        report = format_iot_report(result)
        assert "UDP" in report

    def test_format_report_shows_constrained(self) -> None:
        result = verify_iot_protocol(zigbee_join())
        report = format_iot_report(result)
        assert "Yes" in report  # constrained = Yes


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test the ALL_IOT_PROTOCOLS registry."""

    def test_registry_has_eleven_protocols(self) -> None:
        assert len(ALL_IOT_PROTOCOLS) == 11

    def test_registry_names_unique(self) -> None:
        names = [p.name for p in ALL_IOT_PROTOCOLS]
        assert len(names) == len(set(names))

    def test_registry_includes_mqtt(self) -> None:
        names = {p.name for p in ALL_IOT_PROTOCOLS}
        assert "MQTT_QoS0" in names
        assert "MQTT_QoS1" in names
        assert "MQTT_QoS2" in names

    def test_registry_includes_coap(self) -> None:
        names = {p.name for p in ALL_IOT_PROTOCOLS}
        assert "CoAP_CON" in names
        assert "CoAP_NON" in names
        assert "CoAP_Observe" in names

    def test_registry_includes_zigbee(self) -> None:
        names = {p.name for p in ALL_IOT_PROTOCOLS}
        assert "Zigbee_Join" in names
        assert "Zigbee_DataTransfer" in names
