"""reticulate — Session types as algebraic reticulates.

Public API re-exports from the parser, statespace, product, and lattice modules.
"""

from reticulate.lattice import LatticeResult, check_lattice, compute_join, compute_meet
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
from reticulate.visualize import dot_source, hasse_diagram, render_hasse

__all__ = [
    "Branch",
    "End",
    "LatticeResult",
    "Parallel",
    "ParseError",
    "Rec",
    "Select",
    "Sequence",
    "SessionType",
    "StateSpace",
    "Var",
    "build_statespace",
    "check_lattice",
    "compute_join",
    "compute_meet",
    "dot_source",
    "hasse_diagram",
    "parse",
    "pretty",
    "product_statespace",
    "render_hasse",
    "tokenize",
]
