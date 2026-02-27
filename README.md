# Reticulate

A Python tool for lattice analysis of session type state spaces.

Given a session type definition, Reticulate constructs the state-space labeled transition system, computes its SCC quotient, checks whether the quotient forms a bounded lattice, and optionally generates a Hasse diagram.

## Requirements

- Python 3.11+
- Optional: `graphviz` Python package (for Hasse diagram rendering)

## Installation

```bash
# No installation needed — run directly from the source directory
cd reticulate

# Optional: install graphviz for diagram rendering
pip install graphviz
```

## Usage

### Command Line

```bash
# Basic lattice check
python -m reticulate "rec X.&{a:X, b:end}"

# Generate DOT output for Hasse diagram
python -m reticulate --dot "rec X.&{a:X, b:end}"

# Render Hasse diagram to file
python -m reticulate --hasse output.png "&{m: a.end, n: b.end}"

# Options
python -m reticulate --help
```

### As a Library

```python
from reticulate.parser import parse
from reticulate.statespace import build_statespace
from reticulate.lattice import check_lattice

# Parse a session type
st = parse("rec X.&{a:X, b:end}")

# Build state space
ss = build_statespace(st)

# Check lattice property
result = check_lattice(ss)
print(f"Is lattice: {result.is_lattice}")
print(f"States: {len(ss.states)}, SCCs: {len(set(result.scc_map.values()))}")
```

## Session Type Grammar

```
S  ::=  &{ m1 : S1 , ... , mn : Sn }    -- branch (external choice)
     |  +{ l1 : S1 , ... , ln : Sn }    -- selection (internal choice)
     |  ( S1 || S2 )                     -- parallel composition
     |  rec X . S                        -- recursion
     |  X                                -- variable
     |  end                              -- terminated
     |  m . S                            -- sequencing (sugar for &{m : S})
```

Unicode alternatives: `&` = `&`, `+` = `oplus`, `||` = `parallel`, `rec` = `mu`.

## Modules

| Module | Description |
|--------|-------------|
| `parser.py` | Recursive-descent parser, AST nodes, pretty-printer |
| `statespace.py` | State-space construction by structural induction |
| `product.py` | Product construction for parallel composition |
| `lattice.py` | SCC quotient, reachability, lattice checking |
| `termination.py` | Termination checking, WF-Par well-formedness |
| `morphism.py` | Morphism hierarchy (isomorphism, embedding, projection, Galois) |
| `visualize.py` | Hasse diagram generation (DOT/Graphviz) |
| `cli.py` | Command-line interface |

## Tests

```bash
# Run all tests (642 tests)
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_lattice.py -v

# Run benchmarks only
python -m pytest tests/benchmarks/ -v
```

## Benchmarks

30 real-world protocol benchmarks are included in `tests/benchmarks/`, spanning networking (SMTP, HTTP, DNS, TLS, MQTT), databases, distributed systems (Raft, 2PC, Saga), security (OAuth 2.0), AI agents (MCP, A2A), and more. All 30 form bounded lattices.

## Reference

This tool accompanies the paper:

> A. Z. Caldeira and V. T. Vasconcelos. "Session Type State Spaces Form Lattices." Submitted to CONCUR 2026.

## License

Research software. See repository root for license information.
