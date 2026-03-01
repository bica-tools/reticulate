"""reticulate — Session types as algebraic reticulates.

Public API re-exports from the parser, statespace, product, lattice,
termination, morphism, and visualize modules.
"""

from reticulate.lattice import LatticeResult, check_lattice, compute_join, compute_meet
from reticulate.morphism import (
    GaloisConnection,
    Morphism,
    classify_morphism,
    find_embedding,
    find_isomorphism,
    is_galois_connection,
    is_order_preserving,
    is_order_reflecting,
)
from reticulate.parser import (
    Branch,
    End,
    Parallel,
    ParseError,
    Rec,
    Select,
    Sequence,
    SessionType,
    Var,
    parse,
    pretty,
    tokenize,
)
from reticulate.product import product_statespace
from reticulate.statespace import StateSpace, build_statespace
from reticulate.termination import (
    TerminationResult,
    WFParallelResult,
    check_termination,
    check_wf_parallel,
    is_terminating,
)
from reticulate.testgen import (
    ClientProgram,
    MethodCallNode,
    SelectionSwitchNode,
    TerminalNode,
    TestGenConfig,
    enumerate,
    enumerate_client_programs,
    generate_test_source,
)
from reticulate.visualize import dot_source, hasse_diagram, render_hasse

__all__ = [
    "Branch",
    "End",
    "GaloisConnection",
    "LatticeResult",
    "Morphism",
    "Parallel",
    "ParseError",
    "Rec",
    "Select",
    "Sequence",
    "SessionType",
    "StateSpace",
    "TerminationResult",
    "ClientProgram",
    "MethodCallNode",
    "SelectionSwitchNode",
    "TerminalNode",
    "TestGenConfig",
    "enumerate_client_programs",
    "Var",
    "WFParallelResult",
    "build_statespace",
    "check_lattice",
    "check_termination",
    "check_wf_parallel",
    "classify_morphism",
    "compute_join",
    "compute_meet",
    "dot_source",
    "find_embedding",
    "find_isomorphism",
    "generate_test_source",
    "hasse_diagram",
    "is_galois_connection",
    "is_order_preserving",
    "is_order_reflecting",
    "is_terminating",
    "parse",
    "pretty",
    "product_statespace",
    "render_hasse",
    "tokenize",
]
