# Reticulate

A Python tool for lattice analysis of session type state spaces.

Given a session type definition, Reticulate constructs the state-space labeled transition system, computes its SCC quotient, checks whether the quotient forms a bounded lattice, and optionally generates a Hasse diagram. Part of the *Session Types as Algebraic Reticulates* research project.

## Requirements

- Python 3.11+
- Optional: `graphviz` Python package + system binary (for Hasse diagram rendering)

## Installation

```bash
# Install from source
pip install .

# With visualisation support
pip install .[viz]

# Development (editable + pytest)
pip install -e .[dev]
```

This installs the `session2lattice` command (plus aliases `s2l`, `sess2lat`, `lattice-check`, `bica`).

## Usage

### Command Line

```bash
# Basic lattice check
session2lattice "rec X.&{a:X, b:end}"

# Show version
session2lattice --version

# Generate DOT output for Hasse diagram
session2lattice --dot "rec X.&{a:X, b:end}"

# Render Hasse diagram to file
session2lattice --hasse output "&{m: end, n: end}"

# Check distributivity (Birkhoff classification)
session2lattice --distributive "(&{a: end} || &{b: end})"

# Pretty-print lattice tables (states, transitions, meet/join)
session2lattice --lattice "&{a: end, b: end}"

# Generate JUnit test source
session2lattice --test-gen --class-name FileHandle "&{open: &{read: end, close: end}}"

# All options
session2lattice --help
```

### As a Library

```python
from reticulate import parse, build_statespace, check_lattice

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
```

Unicode alternatives: `⊕` for `+`, `∥` for `||`, `μ` for `rec`.

## Modules

### Core

| Module | Description |
|--------|-------------|
| `parser.py` | Recursive-descent parser, 8 AST nodes, pretty-printer |
| `sugar.py` | Syntactic sugar: desugar / ensugar transformations |
| `statespace.py` | State-space construction by structural induction |
| `product.py` | Product construction for parallel composition |
| `lattice.py` | SCC quotient, reachability, lattice + distributivity checking |
| `termination.py` | Termination checking, WF-Par well-formedness |
| `recursion.py` | Guardedness, contractivity, unfolding, SCC analysis |

### Analysis

| Module | Description |
|--------|-------------|
| `morphism.py` | Morphism hierarchy (isomorphism, embedding, projection, Galois) |
| `subtyping.py` | Gay–Hole subtyping on ASTs, width embedding verification |
| `duality.py` | Session type duality (Branch ↔ Select), involution check |
| `endomorphism.py` | Transition endomorphism analysis (order/meet/join preservation) |
| `reticular.py` | Reticular form characterisation and reconstruction |
| `enumerate_types.py` | Exhaustive session type enumeration (universality check) |
| `context_free.py` | Chomsky classification (regular vs context-free) |
| `polarity.py` | Polarity analysis, concept lattice, Galois pairs |
| `realizability.py` | Realizability conditions, obstruction detection |

### Multiparty

| Module | Description |
|--------|-------------|
| `global_types.py` | Multiparty global type AST, parser, state-space |
| `projection.py` | MPST projection: global → local types |
| `composition.py` | Binary/n-ary composition, synchronized product |
| `composition_viz.py` | Composition dashboard visualisation |
| `channel.py` | Synchronous channel construction |
| `async_channel.py` | Asynchronous channel with buffer semantics |

### Output

| Module | Description |
|--------|-------------|
| `visualize.py` | Hasse diagram generation (DOT/Graphviz) |
| `testgen.py` | JUnit/TestNG test generation from state spaces |
| `coverage.py` | Test coverage analysis and storyboard rendering |
| `cli.py` | Command-line interface (13 flags) |

## Tests

```bash
# Run all tests (2,481 tests)
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_lattice.py -v

# Run benchmarks only
python -m pytest tests/benchmarks/ -v
```

## Benchmarks

79 binary and 24 multiparty real-world protocol benchmarks are included in `tests/benchmarks/`, spanning networking (SMTP, HTTP, DNS, TLS, MQTT), databases (JDBC, Redis), distributed systems (Raft, 2PC, Saga), security (OAuth 2.0), AI agents (MCP, A2A), and more. All form bounded lattices.

## Reference

This tool accompanies the paper:

> A. Z. Caldeira and V. T. Vasconcelos. "Session Type State Spaces Form Lattices." ICE 2026.

## License

MIT License. See [LICENSE](LICENSE).
