"""REST API for modularity analysis (P104).

Provides session type analysis as a FastAPI service.

Self-referencing session type (the API's own protocol):
    rec X . &{submit: +{accepted: &{poll: rec Y . +{pending: Y, complete: &{fetch: end}, failed: end}}, rejected: X}}

Endpoints:
    POST /api/v1/analyze      -- Full modularity report
    POST /api/v1/modularity   -- Quick modularity check
    POST /api/v1/import       -- Import from external spec (openapi|grpc|asyncapi)
    GET  /api/v1/health       -- Health check
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reticulate.modular_report import ModularityReport, generate_report
from reticulate.parser import ParseError, parse, pretty
from reticulate.statespace import build_statespace

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Reticulate Modularity API",
    version="0.1.0",
    description="Session type modularity analysis as a REST service.",
)

# Self-referencing protocol for the API itself
API_PROTOCOL = (
    "rec X . &{submit: +{accepted: &{poll: rec Y . +{pending: Y, "
    "complete: &{fetch: end}, failed: end}}, rejected: X}}"
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/v1/analyze."""

    type_string: str = Field(..., description="Session type string to analyze")
    protocol_name: str = Field(
        "Protocol", description="Human-readable protocol name"
    )


class ModularityRequest(BaseModel):
    """Request body for POST /api/v1/modularity."""

    type_string: str = Field(..., description="Session type string to check")


class ImportRequest(BaseModel):
    """Request body for POST /api/v1/import."""

    spec: dict[str, Any] = Field(..., description="External specification")
    format: str = Field(
        ..., description="Specification format: openapi, grpc, or asyncapi"
    )


class ModularityResponse(BaseModel):
    """Response for POST /api/v1/modularity."""

    is_modular: bool
    is_distributive: bool
    verdict: str
    classification: str


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health."""

    status: str
    version: str


class ImportResponse(BaseModel):
    """Response for POST /api/v1/import."""

    session_types: dict[str, str]


class ErrorDetail(BaseModel):
    """Error detail for 422 responses."""

    error: str
    detail: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_and_build(type_string: str) -> tuple[Any, Any]:
    """Parse a type string and build its state space.

    Returns (ast, state_space).
    Raises HTTPException(422) on parse errors.
    """
    try:
        ast = parse(type_string)
    except (ParseError, ValueError, Exception) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "parse_error", "detail": str(exc)},
        )
    try:
        ss = build_statespace(ast)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "statespace_error", "detail": str(exc)},
        )
    return ast, ss


def _import_openapi(spec: dict[str, Any]) -> dict[str, str]:
    """Extract session types from an OpenAPI-style spec.

    Expects spec to contain 'paths' with state annotations in
    'x-precondition' and 'x-postcondition' extension fields, or
    falls back to simple endpoint-to-branch conversion.
    """
    from reticulate.openapi import OpenAPIEndpoint, api_to_contract

    paths = spec.get("paths", {})
    if not paths:
        raise HTTPException(
            status_code=422,
            detail={"error": "import_error", "detail": "No paths found in spec"},
        )

    endpoints: list[OpenAPIEndpoint] = []
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, details in methods.items():
            method_upper = method.upper()
            if method_upper not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
            if not isinstance(details, dict):
                continue
            pre = details.get("x-precondition", "init")
            post = details.get("x-postcondition", "end")
            desc = details.get("summary", "")
            endpoints.append(
                OpenAPIEndpoint(
                    method=method_upper,
                    path=path,
                    precondition_state=pre,
                    postcondition_state=post,
                    description=desc,
                )
            )

    if not endpoints:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "import_error",
                "detail": "No valid endpoints extracted from spec",
            },
        )

    contract = api_to_contract(endpoints)
    return {"main": contract.session_type_string}


def _import_grpc(spec: dict[str, Any]) -> dict[str, str]:
    """Extract session types from a gRPC-style spec.

    Expects 'services' with 'methods' containing request/response types.
    """
    services = spec.get("services", {})
    if not services:
        raise HTTPException(
            status_code=422,
            detail={"error": "import_error", "detail": "No services found in spec"},
        )

    result: dict[str, str] = {}
    for svc_name, svc_def in services.items():
        methods = svc_def.get("methods", [])
        if not methods:
            continue
        if len(methods) == 1:
            result[svc_name] = f"&{{{methods[0]}: end}}"
        else:
            branches = ", ".join(f"{m}: end" for m in methods)
            result[svc_name] = f"&{{{branches}}}"

    if not result:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "import_error",
                "detail": "No methods extracted from gRPC spec",
            },
        )
    return result


def _import_asyncapi(spec: dict[str, Any]) -> dict[str, str]:
    """Extract session types from an AsyncAPI-style spec.

    Expects 'channels' with publish/subscribe operations.
    """
    channels = spec.get("channels", {})
    if not channels:
        raise HTTPException(
            status_code=422,
            detail={"error": "import_error", "detail": "No channels found in spec"},
        )

    result: dict[str, str] = {}
    for ch_name, ch_def in channels.items():
        if not isinstance(ch_def, dict):
            continue
        ops: list[str] = []
        if "publish" in ch_def:
            ops.append("publish")
        if "subscribe" in ch_def:
            ops.append("subscribe")
        if ops:
            if len(ops) == 1:
                result[ch_name] = f"&{{{ops[0]}: end}}"
            else:
                branches = ", ".join(f"{op}: end" for op in ops)
                result[ch_name] = f"&{{{branches}}}"

    if not result:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "import_error",
                "detail": "No operations extracted from AsyncAPI spec",
            },
        )
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/api/v1/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """Full modularity analysis of a session type.

    Returns the complete modularity report as a dictionary.
    """
    ast, ss = _parse_and_build(req.type_string)
    report = generate_report(
        req.type_string, ss, protocol_name=req.protocol_name
    )
    return report.to_dict()


@app.post("/api/v1/modularity", response_model=ModularityResponse)
def modularity(req: ModularityRequest) -> ModularityResponse:
    """Quick modularity check of a session type.

    Returns a lightweight verdict without the full report.
    """
    ast, ss = _parse_and_build(req.type_string)
    report = generate_report(req.type_string, ss)
    d = report.to_dict()
    return ModularityResponse(
        is_modular=d["is_modular"],
        is_distributive=d["is_distributive"],
        verdict=d["verdict"],
        classification=d["classification"],
    )


@app.post("/api/v1/import", response_model=ImportResponse)
def import_spec(req: ImportRequest) -> ImportResponse:
    """Import session types from an external specification format.

    Supported formats: openapi, grpc, asyncapi.
    """
    fmt = req.format.lower()
    if fmt == "openapi":
        types = _import_openapi(req.spec)
    elif fmt == "grpc":
        types = _import_grpc(req.spec)
    elif fmt == "asyncapi":
        types = _import_asyncapi(req.spec)
    else:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unsupported_format",
                "detail": f"Format '{req.format}' not supported. Use: openapi, grpc, asyncapi",
            },
        )
    return ImportResponse(session_types=types)
