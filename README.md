# Reticulate

Python library for lattice analysis of session type state spaces.

Part of the [Session Types as Algebraic Reticulates](https://github.com/bica-tools) research programme.

## Install

```bash
pip install graphviz  # optional, for Hasse diagrams
```

## Usage

```python
from reticulate import parse, build_statespace, check_lattice

S = parse('&{open: &{read: end, write: end}}')
L = build_statespace(S)
result = check_lattice(L)
print(result.is_lattice)  # True
```

## CLI

```bash
python -m reticulate '&{open: &{read: end, write: end}}'
python -m reticulate --dot '&{open: &{read: end, write: end}}'
python -m reticulate --hasse '&{open: &{read: end, write: end}}' -o output.svg
```

## Modules

| Module | Step | Description |
|--------|------|-------------|
| `parser.py` | 1 | AST + tokenizer + recursive-descent parser |
| `statespace.py` | 1 | State-space construction |
| `product.py` | 1 | Product state space for parallel (∥) |
| `lattice.py` | 5 | Lattice checking (SCC quotient + meet/join) |
| `termination.py` | 1 | Termination + WF-Par checking |
| `visualize.py` | 1 | Hasse diagram generation (DOT/graphviz) |
| `subtyping.py` | 7 | Gay-Hole coinductive subtyping |
| `duality.py` | 8 | Session type duality |
| `reticular.py` | 9 | Reticular form + reconstruction |
| `endomorphism.py` | 10 | Transition endomorphism analysis |
| `global_types.py` | 11 | Multiparty global types |
| `projection.py` | 12 | MPST projection |
| `recursion.py` | 13 | Recursive type analysis |
| `context_free.py` | 14 | Chomsky classification |
| `testgen.py` | 15 | Test generation from state spaces |
| `coverage.py` | 15 | Test coverage analysis |
| `enumerate_types.py` | 6 | Exhaustive type enumeration |
| `morphism.py` | 151 | Morphism hierarchy |
| `cli.py` | 15 | CLI entry point |

## Tests

```bash
python -m pytest tests/ -v  # 2,428 tests
```

## License

Apache 2.0

## Author

Alexandre Zua Caldeira — Independent Researcher
