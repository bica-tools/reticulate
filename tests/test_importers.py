"""Tests for protocol importers (OpenAPI, gRPC, AsyncAPI → session types)."""

import pytest

from reticulate.importers import (
    IMPORTER_SESSION_TYPE,
    from_asyncapi,
    from_grpc,
    from_openapi,
    from_spec,
    _sanitize_label,
    _response_selection,
)
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice


# ---------------------------------------------------------------------------
# Helper: round-trip check (import → parse → statespace → lattice)
# ---------------------------------------------------------------------------

def _assert_is_lattice(session_type_str: str) -> None:
    """Parse a session type string, build its state space, and assert lattice."""
    ast = parse(session_type_str)
    ss = build_statespace(ast)
    result = check_lattice(ss)
    assert result.is_lattice, (
        f"Expected lattice for {session_type_str!r}, "
        f"got counterexample: {result.counterexample}"
    )


# ===========================================================================
# Sanitize helpers
# ===========================================================================


class TestSanitizeLabel:
    """Tests for _sanitize_label."""

    def test_simple(self):
        assert _sanitize_label("hello") == "hello"

    def test_with_slashes(self):
        assert _sanitize_label("/users/{id}") == "users_id"

    def test_uppercase(self):
        assert _sanitize_label("GET") == "get"

    def test_special_chars(self):
        assert _sanitize_label("foo-bar.baz") == "foo_bar_baz"

    def test_empty(self):
        assert _sanitize_label("") == "unknown"

    def test_only_special(self):
        assert _sanitize_label("///") == "unknown"


class TestResponseSelection:
    """Tests for _response_selection."""

    def test_empty_responses(self):
        assert _response_selection({}) == "+{OK: end}"

    def test_single_code(self):
        result = _response_selection({"200": {}})
        assert result == "+{s200: end}"

    def test_multiple_codes_sorted(self):
        result = _response_selection({"404": {}, "200": {}, "500": {}})
        assert result == "+{s200: end, s404: end, s500: end}"

    def test_non_numeric_code(self):
        result = _response_selection({"default": {}})
        assert "default" in result


# ===========================================================================
# OpenAPI importer
# ===========================================================================


class TestFromOpenAPI:
    """Tests for from_openapi."""

    def test_empty_spec(self):
        result = from_openapi({})
        assert result == {"default": "end"}

    def test_empty_paths(self):
        result = from_openapi({"paths": {}})
        assert result == {"default": "end"}

    def test_single_endpoint(self):
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "responses": {"200": {}, "500": {}}
                    }
                }
            }
        }
        result = from_openapi(spec)
        assert "default" in result
        st = result["default"]
        assert "&{" in st
        assert "get_users" in st

    def test_multiple_endpoints_same_tag(self):
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "tags": ["users"],
                        "responses": {"200": {}}
                    },
                    "post": {
                        "tags": ["users"],
                        "responses": {"201": {}, "400": {}}
                    }
                }
            }
        }
        result = from_openapi(spec)
        assert "users" in result
        st = result["users"]
        assert "get_users" in st
        assert "post_users" in st

    def test_multiple_tags(self):
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "tags": ["users"],
                        "responses": {"200": {}}
                    }
                },
                "/items": {
                    "get": {
                        "tags": ["items"],
                        "responses": {"200": {}}
                    }
                }
            }
        }
        result = from_openapi(spec)
        assert "users" in result
        assert "items" in result

    def test_endpoint_no_tags_defaults(self):
        spec = {
            "paths": {
                "/health": {
                    "get": {
                        "responses": {"200": {}}
                    }
                }
            }
        }
        result = from_openapi(spec)
        assert "default" in result

    def test_endpoint_empty_tags_defaults(self):
        spec = {
            "paths": {
                "/health": {
                    "get": {
                        "tags": [],
                        "responses": {"200": {}}
                    }
                }
            }
        }
        result = from_openapi(spec)
        assert "default" in result

    def test_endpoint_no_responses(self):
        spec = {
            "paths": {
                "/fire": {
                    "post": {
                        "tags": ["events"]
                    }
                }
            }
        }
        result = from_openapi(spec)
        assert "events" in result
        # No responses → +{OK: end}
        assert "OK" in result["events"]

    def test_non_http_keys_ignored(self):
        spec = {
            "paths": {
                "/users": {
                    "summary": "User operations",
                    "get": {
                        "responses": {"200": {}}
                    }
                }
            }
        }
        result = from_openapi(spec)
        st = result["default"]
        assert "summary" not in st

    def test_realistic_petstore(self):
        """Simplified Petstore-like spec."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Petstore", "version": "1.0"},
            "paths": {
                "/pets": {
                    "get": {
                        "tags": ["pets"],
                        "operationId": "listPets",
                        "responses": {"200": {}, "500": {}}
                    },
                    "post": {
                        "tags": ["pets"],
                        "operationId": "createPet",
                        "responses": {"201": {}, "400": {}, "500": {}}
                    }
                },
                "/pets/{petId}": {
                    "get": {
                        "tags": ["pets"],
                        "operationId": "showPetById",
                        "responses": {"200": {}, "404": {}, "500": {}}
                    },
                    "delete": {
                        "tags": ["pets"],
                        "operationId": "deletePet",
                        "responses": {"204": {}, "404": {}}
                    }
                }
            }
        }
        result = from_openapi(spec)
        assert "pets" in result
        st = result["pets"]
        assert "&{" in st
        # Should contain 4 methods
        assert "get_pets" in st
        assert "post_pets" in st
        assert "delete_pets_petid" in st


# ===========================================================================
# gRPC importer
# ===========================================================================


class TestFromGRPC:
    """Tests for from_grpc."""

    def test_empty_service(self):
        result = from_grpc({"name": "Empty", "methods": []})
        assert result == "end"

    def test_no_methods_key(self):
        result = from_grpc({"name": "Bare"})
        assert result == "end"

    def test_unary_method(self):
        service = {
            "name": "Greeter",
            "methods": [
                {"name": "SayHello", "type": "unary"}
            ]
        }
        result = from_grpc(service)
        assert "sayhello" in result
        assert "+{OK: end, ERROR: end}" in result

    def test_server_streaming(self):
        service = {
            "name": "Stream",
            "methods": [
                {"name": "ListFeatures", "type": "server_streaming"}
            ]
        }
        result = from_grpc(service)
        assert "rec X" in result
        assert "data: X" in result
        assert "done: end" in result

    def test_client_streaming(self):
        service = {
            "name": "Upload",
            "methods": [
                {"name": "RecordRoute", "type": "client_streaming"}
            ]
        }
        result = from_grpc(service)
        assert "rec X" in result
        assert "send" in result
        assert "continue: X" in result
        assert "finish" in result

    def test_bidi_streaming(self):
        service = {
            "name": "Chat",
            "methods": [
                {"name": "RouteChat", "type": "bidi_streaming"}
            ]
        }
        result = from_grpc(service)
        assert "rec X" in result
        assert "send: X" in result
        assert "recv: X" in result
        assert "done: end" in result

    def test_mixed_methods(self):
        service = {
            "name": "RouteGuide",
            "methods": [
                {"name": "GetFeature", "type": "unary"},
                {"name": "ListFeatures", "type": "server_streaming"},
                {"name": "RecordRoute", "type": "client_streaming"},
                {"name": "RouteChat", "type": "bidi_streaming"}
            ]
        }
        result = from_grpc(service)
        assert "&{" in result
        assert "getfeature" in result
        assert "listfeatures" in result
        assert "recordroute" in result
        assert "routechat" in result

    def test_unknown_type_defaults_to_unary(self):
        service = {
            "name": "Mystery",
            "methods": [
                {"name": "DoThing", "type": "whatevs"}
            ]
        }
        result = from_grpc(service)
        assert "+{OK: end, ERROR: end}" in result


# ===========================================================================
# AsyncAPI importer
# ===========================================================================


class TestFromAsyncAPI:
    """Tests for from_asyncapi."""

    def test_empty_spec(self):
        result = from_asyncapi({})
        assert result == {"default": "end"}

    def test_empty_channels(self):
        result = from_asyncapi({"channels": {}})
        assert result == {"default": "end"}

    def test_publish_only(self):
        spec = {
            "channels": {
                "user/signedup": {
                    "publish": {"message": {"payload": {}}}
                }
            }
        }
        result = from_asyncapi(spec)
        assert "publishers" in result
        assert "+{" in result["publishers"]
        assert "user_signedup" in result["publishers"]

    def test_subscribe_only(self):
        spec = {
            "channels": {
                "order/placed": {
                    "subscribe": {"message": {"payload": {}}}
                }
            }
        }
        result = from_asyncapi(spec)
        assert "subscribers" in result
        assert "&{" in result["subscribers"]

    def test_both_pub_and_sub(self):
        spec = {
            "channels": {
                "events/send": {
                    "publish": {}
                },
                "events/receive": {
                    "subscribe": {}
                }
            }
        }
        result = from_asyncapi(spec)
        assert "publishers" in result
        assert "subscribers" in result

    def test_channel_with_both_operations(self):
        spec = {
            "channels": {
                "chat/messages": {
                    "publish": {},
                    "subscribe": {}
                }
            }
        }
        result = from_asyncapi(spec)
        assert "publishers" in result
        assert "subscribers" in result

    def test_multiple_publish_channels(self):
        spec = {
            "channels": {
                "notifications/email": {"publish": {}},
                "notifications/sms": {"publish": {}},
                "notifications/push": {"publish": {}}
            }
        }
        result = from_asyncapi(spec)
        assert "publishers" in result
        st = result["publishers"]
        assert "notifications_email" in st
        assert "notifications_sms" in st
        assert "notifications_push" in st


# ===========================================================================
# Dispatcher
# ===========================================================================


class TestFromSpec:
    """Tests for from_spec dispatcher."""

    def test_openapi_dispatch(self):
        spec = {"paths": {"/x": {"get": {"responses": {"200": {}}}}}}
        result = from_spec(spec, "openapi")
        assert "default" in result

    def test_grpc_dispatch(self):
        spec = {"name": "Svc", "methods": [{"name": "Do", "type": "unary"}]}
        result = from_spec(spec, "grpc")
        assert "svc" in result

    def test_asyncapi_dispatch(self):
        spec = {"channels": {"ch": {"subscribe": {}}}}
        result = from_spec(spec, "asyncapi")
        assert "subscribers" in result

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown spec format"):
            from_spec({}, "graphql")

    def test_case_insensitive(self):
        result = from_spec({"paths": {}}, "OpenAPI")
        assert "default" in result


# ===========================================================================
# Round-trip: import → parse → build_statespace → check_lattice
# ===========================================================================


class TestRoundTripLattice:
    """Every imported session type must parse and form a lattice."""

    def test_openapi_single_endpoint_is_lattice(self):
        spec = {
            "paths": {
                "/users": {
                    "get": {"responses": {"200": {}, "500": {}}}
                }
            }
        }
        for st in from_openapi(spec).values():
            _assert_is_lattice(st)

    def test_openapi_multi_endpoint_is_lattice(self):
        spec = {
            "paths": {
                "/users": {
                    "get": {
                        "tags": ["api"],
                        "responses": {"200": {}}
                    },
                    "post": {
                        "tags": ["api"],
                        "responses": {"201": {}, "400": {}}
                    }
                }
            }
        }
        for st in from_openapi(spec).values():
            _assert_is_lattice(st)

    def test_grpc_unary_is_lattice(self):
        service = {
            "name": "Test",
            "methods": [{"name": "Call", "type": "unary"}]
        }
        _assert_is_lattice(from_grpc(service))

    def test_grpc_server_stream_is_lattice(self):
        service = {
            "name": "Test",
            "methods": [{"name": "Stream", "type": "server_streaming"}]
        }
        _assert_is_lattice(from_grpc(service))

    def test_grpc_client_stream_is_lattice(self):
        service = {
            "name": "Test",
            "methods": [{"name": "Upload", "type": "client_streaming"}]
        }
        _assert_is_lattice(from_grpc(service))

    def test_grpc_bidi_stream_is_lattice(self):
        service = {
            "name": "Test",
            "methods": [{"name": "Chat", "type": "bidi_streaming"}]
        }
        _assert_is_lattice(from_grpc(service))

    def test_grpc_mixed_is_lattice(self):
        service = {
            "name": "RouteGuide",
            "methods": [
                {"name": "GetFeature", "type": "unary"},
                {"name": "ListFeatures", "type": "server_streaming"},
            ]
        }
        _assert_is_lattice(from_grpc(service))

    def test_asyncapi_pub_is_lattice(self):
        spec = {
            "channels": {
                "events/a": {"publish": {}},
                "events/b": {"publish": {}}
            }
        }
        for st in from_asyncapi(spec).values():
            _assert_is_lattice(st)

    def test_asyncapi_sub_is_lattice(self):
        spec = {
            "channels": {
                "events/a": {"subscribe": {}},
                "events/b": {"subscribe": {}}
            }
        }
        for st in from_asyncapi(spec).values():
            _assert_is_lattice(st)

    def test_petstore_full_is_lattice(self):
        spec = {
            "paths": {
                "/pets": {
                    "get": {"tags": ["pets"], "responses": {"200": {}, "500": {}}},
                    "post": {"tags": ["pets"], "responses": {"201": {}, "400": {}}}
                },
                "/pets/{petId}": {
                    "get": {"tags": ["pets"], "responses": {"200": {}, "404": {}}},
                    "delete": {"tags": ["pets"], "responses": {"204": {}, "404": {}}}
                }
            }
        }
        for st in from_openapi(spec).values():
            _assert_is_lattice(st)


# ===========================================================================
# Self-referencing benchmark (P104)
# ===========================================================================


class TestImporterSelfReference:
    """The Importer's own session type must be a valid lattice."""

    def test_importer_session_type_parses(self):
        ast = parse(IMPORTER_SESSION_TYPE)
        assert ast is not None

    def test_importer_session_type_is_lattice(self):
        _assert_is_lattice(IMPORTER_SESSION_TYPE)

    def test_importer_session_type_has_three_branches(self):
        ast = parse(IMPORTER_SESSION_TYPE)
        from reticulate.parser import Branch
        assert isinstance(ast, Branch)
        assert len(ast.choices) == 3

    def test_importer_session_type_labels(self):
        ast = parse(IMPORTER_SESSION_TYPE)
        labels = {name for name, _ in ast.choices}
        assert labels == {"import_openapi", "import_grpc", "import_asyncapi"}
