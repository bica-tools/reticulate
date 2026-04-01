"""Real-world session type extraction from public API specifications.

Applies the reticulate extraction pipeline to real open-source API specs
(OpenAPI, gRPC, AsyncAPI) and analyzes the resulting session type lattices.

This module implements Step 71b: validating that real-world APIs produce
well-formed lattice structures when modeled as session types.

Scientific Method:
  Observe   — Real APIs have stateful lifecycles (auth → use → close)
  Question  — Do real-world API specs produce lattices when imported?
  Hypothesis — Most well-designed APIs produce distributive lattices
  Predict   — ≥80% of real API session types form lattices
  Experiment — Extract from 10+ real specs, check lattice properties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reticulate.importers import from_openapi
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice, check_distributive


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting and analyzing a single API tag's session type."""
    api_name: str
    tag: str
    session_type: str
    num_states: int
    num_transitions: int
    is_lattice: bool
    is_distributive: bool
    num_endpoints: int  # number of branch methods


@dataclass(frozen=True)
class AnalysisReport:
    """Aggregate report across multiple API extractions."""
    results: list[ExtractionResult]
    total_apis: int
    total_tags: int
    lattice_count: int
    distributive_count: int
    lattice_rate: float
    distributive_rate: float


# ---------------------------------------------------------------------------
# Real-world API specs (embedded as Python dicts — no network dependency)
# ---------------------------------------------------------------------------

def petstore_spec() -> dict[str, Any]:
    """Swagger Petstore — the canonical OpenAPI example."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Petstore", "version": "1.0.0"},
        "paths": {
            "/pets": {
                "get": {
                    "tags": ["pets"],
                    "operationId": "listPets",
                    "responses": {"200": {}, "default": {}}
                },
                "post": {
                    "tags": ["pets"],
                    "operationId": "createPets",
                    "responses": {"201": {}, "default": {}}
                }
            },
            "/pets/{petId}": {
                "get": {
                    "tags": ["pets"],
                    "operationId": "showPetById",
                    "responses": {"200": {}, "404": {}, "default": {}}
                },
                "put": {
                    "tags": ["pets"],
                    "operationId": "updatePet",
                    "responses": {"200": {}, "404": {}}
                },
                "delete": {
                    "tags": ["pets"],
                    "operationId": "deletePet",
                    "responses": {"200": {}, "404": {}}
                }
            }
        }
    }


def github_repos_spec() -> dict[str, Any]:
    """GitHub API — repository operations (simplified)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "GitHub Repos", "version": "1.0.0"},
        "paths": {
            "/repos/{owner}/{repo}": {
                "get": {
                    "tags": ["repos"],
                    "responses": {"200": {}, "404": {}, "403": {}}
                },
                "patch": {
                    "tags": ["repos"],
                    "responses": {"200": {}, "403": {}, "422": {}}
                },
                "delete": {
                    "tags": ["repos"],
                    "responses": {"204": {}, "403": {}, "404": {}}
                }
            },
            "/repos/{owner}/{repo}/branches": {
                "get": {
                    "tags": ["repos"],
                    "responses": {"200": {}, "404": {}}
                }
            },
            "/repos/{owner}/{repo}/tags": {
                "get": {
                    "tags": ["repos"],
                    "responses": {"200": {}, "404": {}}
                }
            }
        }
    }


def stripe_payments_spec() -> dict[str, Any]:
    """Stripe API — payment intent lifecycle (simplified)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Stripe Payments", "version": "1.0.0"},
        "paths": {
            "/v1/payment_intents": {
                "post": {
                    "tags": ["payment_intents"],
                    "responses": {"200": {}, "400": {}, "401": {}}
                },
                "get": {
                    "tags": ["payment_intents"],
                    "responses": {"200": {}, "401": {}}
                }
            },
            "/v1/payment_intents/{id}": {
                "get": {
                    "tags": ["payment_intents"],
                    "responses": {"200": {}, "404": {}}
                },
                "post": {
                    "tags": ["payment_intents"],
                    "responses": {"200": {}, "400": {}, "404": {}}
                }
            },
            "/v1/payment_intents/{id}/confirm": {
                "post": {
                    "tags": ["payment_intents"],
                    "responses": {"200": {}, "400": {}, "402": {}}
                }
            },
            "/v1/payment_intents/{id}/cancel": {
                "post": {
                    "tags": ["payment_intents"],
                    "responses": {"200": {}, "400": {}}
                }
            },
            "/v1/payment_intents/{id}/capture": {
                "post": {
                    "tags": ["payment_intents"],
                    "responses": {"200": {}, "400": {}, "402": {}}
                }
            }
        }
    }


def twilio_messaging_spec() -> dict[str, Any]:
    """Twilio API — messaging operations (simplified)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Twilio Messaging", "version": "1.0.0"},
        "paths": {
            "/Messages": {
                "post": {
                    "tags": ["messages"],
                    "responses": {"201": {}, "400": {}, "401": {}}
                },
                "get": {
                    "tags": ["messages"],
                    "responses": {"200": {}, "401": {}}
                }
            },
            "/Messages/{Sid}": {
                "get": {
                    "tags": ["messages"],
                    "responses": {"200": {}, "404": {}}
                },
                "post": {
                    "tags": ["messages"],
                    "responses": {"200": {}, "404": {}}
                },
                "delete": {
                    "tags": ["messages"],
                    "responses": {"204": {}, "404": {}}
                }
            }
        }
    }


def kubernetes_pods_spec() -> dict[str, Any]:
    """Kubernetes API — pod lifecycle (simplified)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Kubernetes Pods", "version": "1.0.0"},
        "paths": {
            "/api/v1/namespaces/{ns}/pods": {
                "get": {
                    "tags": ["pods"],
                    "responses": {"200": {}, "401": {}, "403": {}}
                },
                "post": {
                    "tags": ["pods"],
                    "responses": {"201": {}, "400": {}, "401": {}, "409": {}}
                }
            },
            "/api/v1/namespaces/{ns}/pods/{name}": {
                "get": {
                    "tags": ["pods"],
                    "responses": {"200": {}, "404": {}}
                },
                "put": {
                    "tags": ["pods"],
                    "responses": {"200": {}, "404": {}, "409": {}}
                },
                "delete": {
                    "tags": ["pods"],
                    "responses": {"200": {}, "404": {}}
                },
                "patch": {
                    "tags": ["pods"],
                    "responses": {"200": {}, "404": {}, "422": {}}
                }
            },
            "/api/v1/namespaces/{ns}/pods/{name}/log": {
                "get": {
                    "tags": ["pods"],
                    "responses": {"200": {}, "404": {}}
                }
            },
            "/api/v1/namespaces/{ns}/pods/{name}/exec": {
                "post": {
                    "tags": ["pods"],
                    "responses": {"200": {}, "400": {}, "404": {}}
                }
            }
        }
    }


def docker_containers_spec() -> dict[str, Any]:
    """Docker Engine API — container lifecycle (simplified)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Docker Containers", "version": "1.0.0"},
        "paths": {
            "/containers/json": {
                "get": {
                    "tags": ["containers"],
                    "responses": {"200": {}, "400": {}, "500": {}}
                }
            },
            "/containers/create": {
                "post": {
                    "tags": ["containers"],
                    "responses": {"201": {}, "400": {}, "404": {}, "409": {}, "500": {}}
                }
            },
            "/containers/{id}/start": {
                "post": {
                    "tags": ["containers"],
                    "responses": {"204": {}, "304": {}, "404": {}, "500": {}}
                }
            },
            "/containers/{id}/stop": {
                "post": {
                    "tags": ["containers"],
                    "responses": {"204": {}, "304": {}, "404": {}, "500": {}}
                }
            },
            "/containers/{id}/restart": {
                "post": {
                    "tags": ["containers"],
                    "responses": {"204": {}, "404": {}, "500": {}}
                }
            },
            "/containers/{id}": {
                "delete": {
                    "tags": ["containers"],
                    "responses": {"204": {}, "400": {}, "404": {}, "409": {}, "500": {}}
                }
            },
            "/containers/{id}/logs": {
                "get": {
                    "tags": ["containers"],
                    "responses": {"200": {}, "404": {}, "500": {}}
                }
            }
        }
    }


def elasticsearch_index_spec() -> dict[str, Any]:
    """Elasticsearch API — index operations (simplified)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Elasticsearch Index", "version": "1.0.0"},
        "paths": {
            "/{index}": {
                "put": {
                    "tags": ["index"],
                    "responses": {"200": {}, "400": {}, "404": {}}
                },
                "get": {
                    "tags": ["index"],
                    "responses": {"200": {}, "404": {}}
                },
                "delete": {
                    "tags": ["index"],
                    "responses": {"200": {}, "404": {}}
                }
            },
            "/{index}/_doc/{id}": {
                "put": {
                    "tags": ["index"],
                    "responses": {"200": {}, "201": {}, "400": {}}
                },
                "get": {
                    "tags": ["index"],
                    "responses": {"200": {}, "404": {}}
                },
                "delete": {
                    "tags": ["index"],
                    "responses": {"200": {}, "404": {}}
                }
            },
            "/{index}/_search": {
                "get": {
                    "tags": ["index"],
                    "responses": {"200": {}, "400": {}, "404": {}}
                },
                "post": {
                    "tags": ["index"],
                    "responses": {"200": {}, "400": {}, "404": {}}
                }
            }
        }
    }


def redis_commands_spec() -> dict[str, Any]:
    """Redis API — key-value operations (simplified as REST-like)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Redis Commands", "version": "1.0.0"},
        "paths": {
            "/keys/{key}": {
                "get": {
                    "tags": ["keys"],
                    "responses": {"200": {}, "404": {}}
                },
                "put": {
                    "tags": ["keys"],
                    "responses": {"200": {}, "400": {}}
                },
                "delete": {
                    "tags": ["keys"],
                    "responses": {"200": {}, "404": {}}
                }
            },
            "/keys/{key}/expire": {
                "post": {
                    "tags": ["keys"],
                    "responses": {"200": {}, "404": {}}
                }
            },
            "/keys/{key}/ttl": {
                "get": {
                    "tags": ["keys"],
                    "responses": {"200": {}, "404": {}}
                }
            }
        }
    }


def auth0_spec() -> dict[str, Any]:
    """Auth0 API — authentication flows (simplified)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Auth0", "version": "1.0.0"},
        "paths": {
            "/oauth/token": {
                "post": {
                    "tags": ["auth"],
                    "responses": {"200": {}, "401": {}, "403": {}}
                }
            },
            "/userinfo": {
                "get": {
                    "tags": ["auth"],
                    "responses": {"200": {}, "401": {}}
                }
            },
            "/oauth/revoke": {
                "post": {
                    "tags": ["auth"],
                    "responses": {"200": {}, "400": {}}
                }
            },
            "/dbconnections/signup": {
                "post": {
                    "tags": ["auth"],
                    "responses": {"200": {}, "400": {}}
                }
            },
            "/dbconnections/change_password": {
                "post": {
                    "tags": ["auth"],
                    "responses": {"200": {}, "400": {}}
                }
            }
        }
    }


def s3_objects_spec() -> dict[str, Any]:
    """AWS S3 API — object operations (simplified as REST)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "S3 Objects", "version": "1.0.0"},
        "paths": {
            "/{bucket}": {
                "get": {
                    "tags": ["objects"],
                    "responses": {"200": {}, "403": {}, "404": {}}
                },
                "put": {
                    "tags": ["objects"],
                    "responses": {"200": {}, "403": {}}
                },
                "delete": {
                    "tags": ["objects"],
                    "responses": {"204": {}, "403": {}, "404": {}}
                }
            },
            "/{bucket}/{key}": {
                "get": {
                    "tags": ["objects"],
                    "responses": {"200": {}, "403": {}, "404": {}}
                },
                "put": {
                    "tags": ["objects"],
                    "responses": {"200": {}, "403": {}}
                },
                "delete": {
                    "tags": ["objects"],
                    "responses": {"204": {}, "403": {}}
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Registry of all real-world specs
# ---------------------------------------------------------------------------

REAL_WORLD_SPECS: dict[str, callable] = {
    "Petstore": petstore_spec,
    "GitHub Repos": github_repos_spec,
    "Stripe Payments": stripe_payments_spec,
    "Twilio Messaging": twilio_messaging_spec,
    "Kubernetes Pods": kubernetes_pods_spec,
    "Docker Containers": docker_containers_spec,
    "Elasticsearch": elasticsearch_index_spec,
    "Redis": redis_commands_spec,
    "Auth0": auth0_spec,
    "S3 Objects": s3_objects_spec,
}


# ---------------------------------------------------------------------------
# Extraction and analysis
# ---------------------------------------------------------------------------

def extract_and_analyze(api_name: str, spec: dict[str, Any]) -> list[ExtractionResult]:
    """Extract session types from an OpenAPI spec and analyze lattice properties."""
    tag_types = from_openapi(spec)
    results: list[ExtractionResult] = []

    for tag, st_str in tag_types.items():
        ast = parse(st_str)
        ss = build_statespace(ast)
        lr = check_lattice(ss)

        # Check distributivity if it's a lattice
        is_dist = False
        if lr.is_lattice:
            try:
                dr = check_distributive(ss)
                is_dist = dr.is_distributive
            except Exception:
                is_dist = False

        # Count endpoints (branches at top level)
        num_endpoints = len([t for t in ss.transitions if t[0] == ss.top])

        results.append(ExtractionResult(
            api_name=api_name,
            tag=tag,
            session_type=st_str,
            num_states=len(ss.states),
            num_transitions=len(ss.transitions),
            is_lattice=lr.is_lattice,
            is_distributive=is_dist,
            num_endpoints=num_endpoints,
        ))

    return results


def analyze_all_specs() -> AnalysisReport:
    """Run extraction and analysis on all real-world API specs."""
    all_results: list[ExtractionResult] = []

    for name, spec_fn in REAL_WORLD_SPECS.items():
        spec = spec_fn()
        results = extract_and_analyze(name, spec)
        all_results.extend(results)

    lattice_count = sum(1 for r in all_results if r.is_lattice)
    distributive_count = sum(1 for r in all_results if r.is_distributive)
    total = len(all_results)

    return AnalysisReport(
        results=all_results,
        total_apis=len(REAL_WORLD_SPECS),
        total_tags=total,
        lattice_count=lattice_count,
        distributive_count=distributive_count,
        lattice_rate=lattice_count / total if total > 0 else 0.0,
        distributive_rate=distributive_count / total if total > 0 else 0.0,
    )


def print_report(report: AnalysisReport) -> str:
    """Format an analysis report as a readable string."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  REAL-WORLD API SESSION TYPE EXTRACTION REPORT")
    lines.append("=" * 72)
    lines.append(f"  APIs analyzed: {report.total_apis}")
    lines.append(f"  Tags extracted: {report.total_tags}")
    lines.append(f"  Lattice rate: {report.lattice_count}/{report.total_tags} "
                 f"({report.lattice_rate:.0%})")
    lines.append(f"  Distributive rate: {report.distributive_count}/{report.total_tags} "
                 f"({report.distributive_rate:.0%})")
    lines.append("")
    lines.append(f"  {'API':<22} {'Tag':<20} {'States':>6} {'Trans':>6} "
                 f"{'Lattice':>8} {'Distrib':>8} {'Endpoints':>9}")
    lines.append("  " + "-" * 68)

    for r in report.results:
        lat = "YES" if r.is_lattice else "NO"
        dist = "YES" if r.is_distributive else "NO"
        lines.append(f"  {r.api_name:<22} {r.tag:<20} {r.num_states:>6} "
                     f"{r.num_transitions:>6} {lat:>8} {dist:>8} {r.num_endpoints:>9}")

    lines.append("=" * 72)
    return "\n".join(lines)
