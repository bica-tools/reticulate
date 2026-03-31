"""Protocol importers: convert external API specs into session types.

Three importers convert plain Python dicts (no external dependencies) into
session type strings that can be parsed by ``reticulate.parser.parse``:

1. **OpenAPI 3.x** — ``from_openapi(spec)`` groups endpoints by tag.
2. **gRPC / Protobuf** — ``from_grpc(service)`` converts RPC methods.
3. **AsyncAPI** — ``from_asyncapi(spec)`` converts pub/sub channels.

Dispatcher: ``from_spec(spec, format)`` routes to the right importer.

Self-referencing session type for the Importer protocol itself::

    &{import_openapi: +{OK: end, ERROR: end},
      import_grpc: +{OK: end, ERROR: end},
      import_asyncapi: +{OK: end, ERROR: end}}
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Importer's own session type (P104 dogfooding)
# ---------------------------------------------------------------------------

IMPORTER_SESSION_TYPE: str = (
    "&{import_openapi: +{OK: end, ERROR: end}, "
    "import_grpc: +{OK: end, ERROR: end}, "
    "import_asyncapi: +{OK: end, ERROR: end}}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    """Sanitize a string to be a valid session type label.

    Replaces non-alphanumeric characters (except underscore) with underscore,
    strips leading/trailing underscores, and lowercases.
    """
    result: list[str] = []
    for ch in label:
        if ch.isalnum() or ch == "_":
            result.append(ch)
        else:
            result.append("_")
    sanitized = "".join(result).strip("_").lower()
    # Collapse multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized if sanitized else "unknown"


def _response_selection(responses: dict[str, Any]) -> str:
    """Build a selection type from HTTP response codes.

    Each response code becomes a selection label like ``s200``, ``s404``, etc.
    If no responses, returns ``+{OK: end}``.
    """
    if not responses:
        return "+{OK: end}"
    labels: list[str] = []
    for code in sorted(responses.keys()):
        label = f"s{code}" if str(code).isdigit() else _sanitize_label(str(code))
        labels.append(f"{label}: end")
    return "+{" + ", ".join(labels) + "}"


# ---------------------------------------------------------------------------
# OpenAPI 3.x importer
# ---------------------------------------------------------------------------

def from_openapi(spec: dict[str, Any]) -> dict[str, str]:
    """Convert an OpenAPI 3.x spec dict to session types, one per tag.

    Parameters
    ----------
    spec : dict
        OpenAPI spec as a plain Python dict. Must have ``paths`` key.

    Returns
    -------
    dict[str, str]
        Mapping from tag name to session type string.
        Endpoints without tags go into the ``"default"`` group.
    """
    paths: dict[str, Any] = spec.get("paths", {})
    if not paths:
        return {"default": "end"}

    # Collect methods per tag
    tag_methods: dict[str, list[str]] = {}
    http_methods = {"get", "post", "put", "delete", "patch", "head", "options", "trace"}

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in http_methods:
                continue
            if not isinstance(operation, dict):
                continue
            # Determine tag
            tags = operation.get("tags", ["default"])
            if not tags:
                tags = ["default"]
            # Build label from method + path
            label = _sanitize_label(f"{method}_{path}")
            # Build response selection
            responses = operation.get("responses", {})
            selection = _response_selection(responses)
            branch_entry = f"{label}: {selection}"

            for tag in tags:
                tag_key = _sanitize_label(tag) if tag != "default" else "default"
                if tag_key not in tag_methods:
                    tag_methods[tag_key] = []
                tag_methods[tag_key].append(branch_entry)

    if not tag_methods:
        return {"default": "end"}

    result: dict[str, str] = {}
    for tag, methods in tag_methods.items():
        if len(methods) == 1:
            result[tag] = "&{" + methods[0] + "}"
        else:
            result[tag] = "&{" + ", ".join(methods) + "}"
    return result


# ---------------------------------------------------------------------------
# gRPC / Protobuf importer
# ---------------------------------------------------------------------------

def from_grpc(service: dict[str, Any]) -> str:
    """Convert a simplified protobuf service definition to a session type.

    Parameters
    ----------
    service : dict
        Keys: ``name`` (str), ``methods`` (list of dicts).
        Each method dict has:
        - ``name`` (str): RPC method name
        - ``type`` (str): one of ``"unary"``, ``"server_streaming"``,
          ``"client_streaming"``, ``"bidi_streaming"``

    Returns
    -------
    str
        Session type string for the service.
    """
    methods: list[dict[str, Any]] = service.get("methods", [])
    if not methods:
        return "end"

    branch_entries: list[str] = []
    for m in methods:
        name = _sanitize_label(m.get("name", "unknown"))
        rpc_type = m.get("type", "unary").lower()

        if rpc_type == "unary":
            entry = f"{name}: +{{OK: end, ERROR: end}}"
        elif rpc_type == "server_streaming":
            entry = f"{name}: rec X . +{{data: X, done: end, error: end}}"
        elif rpc_type == "client_streaming":
            entry = (
                f"{name}: rec X . &{{send: +{{continue: X, "
                f"finish: +{{OK: end, ERROR: end}}}}}}"
            )
        elif rpc_type == "bidi_streaming":
            # Bidirectional: interleave send and receive
            entry = (
                f"{name}: rec X . &{{send: X, recv: X, done: end}}"
            )
        else:
            # Unknown type, treat as unary
            entry = f"{name}: +{{OK: end, ERROR: end}}"

        branch_entries.append(entry)

    return "&{" + ", ".join(branch_entries) + "}"


# ---------------------------------------------------------------------------
# AsyncAPI importer
# ---------------------------------------------------------------------------

def from_asyncapi(spec: dict[str, Any]) -> dict[str, str]:
    """Convert an AsyncAPI spec dict to session types.

    Parameters
    ----------
    spec : dict
        AsyncAPI spec as a plain Python dict. Must have ``channels`` key.
        Each channel has ``publish`` and/or ``subscribe`` operations.

    Returns
    -------
    dict[str, str]
        Mapping from channel group name to session type string.
        Uses ``"publishers"`` for publish channels and ``"subscribers"``
        for subscribe channels.
    """
    channels: dict[str, Any] = spec.get("channels", {})
    if not channels:
        return {"default": "end"}

    publish_labels: list[str] = []
    subscribe_labels: list[str] = []

    for channel_name, channel_def in channels.items():
        if not isinstance(channel_def, dict):
            continue
        label = _sanitize_label(channel_name)

        if "publish" in channel_def:
            # Client publishes → selection (client chooses to send)
            publish_labels.append(f"{label}: end")

        if "subscribe" in channel_def:
            # Client subscribes → branch (client receives)
            subscribe_labels.append(f"{label}: end")

    result: dict[str, str] = {}

    if publish_labels:
        result["publishers"] = "+{" + ", ".join(publish_labels) + "}"

    if subscribe_labels:
        result["subscribers"] = "&{" + ", ".join(subscribe_labels) + "}"

    if not result:
        return {"default": "end"}

    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def from_spec(spec: dict[str, Any], format: str) -> dict[str, str]:
    """Dispatch to the appropriate importer based on format string.

    Parameters
    ----------
    spec : dict
        API specification as a plain Python dict.
    format : str
        One of ``"openapi"``, ``"grpc"``, ``"asyncapi"``.

    Returns
    -------
    dict[str, str]
        Mapping from group name to session type string.

    Raises
    ------
    ValueError
        If *format* is not recognised.
    """
    fmt = format.lower().strip()
    if fmt == "openapi":
        return from_openapi(spec)
    elif fmt == "grpc":
        service_name = spec.get("name", "service")
        return {_sanitize_label(service_name): from_grpc(spec)}
    elif fmt == "asyncapi":
        return from_asyncapi(spec)
    else:
        raise ValueError(
            f"Unknown spec format {format!r}; "
            f"expected 'openapi', 'grpc', or 'asyncapi'"
        )
