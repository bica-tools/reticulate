# Reticulate — User Manual

**Version 1.0 — March 2026**
Alexandre Zua Caldeira, Independent Researcher

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Session Type Syntax](#4-session-type-syntax)
5. [Command-Line Interface](#5-command-line-interface)
6. [Library API](#6-library-api)
7. [Understanding Results](#7-understanding-results)
8. [Hasse Diagrams](#8-hasse-diagrams)
9. [Test Generation](#9-test-generation)
10. [Advanced Topics](#10-advanced-topics)
11. [Troubleshooting](#11-troubleshooting)
12. [API Reference](#12-api-reference)

---

## 1. Overview

Reticulate is a Python library and CLI tool for analysing session types. Given a session type definition, it:

1. **Parses** the type string into an abstract syntax tree (AST).
2. **Constructs** the state-space labelled transition system (LTS).
3. **Checks** whether the state space forms a bounded lattice.
4. **Visualises** the Hasse diagram of the state space.
5. **Checks termination** and well-formedness of parallel composition.
6. **Classifies morphisms** between state spaces.
7. **Generates JUnit 5 tests** from the state space.

The central result: the state space of every well-formed session type, quotiented by strongly connected components, forms a bounded lattice — an algebraic structure called a *reticulate*.

### Architecture

```
type string ──→ parser ──→ AST ──→ statespace ──→ StateSpace
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                     lattice      termination     visualize
                         │              │              │
                         ▼              ▼              ▼
                  LatticeResult  TerminationResult  DOT/SVG
                         │
                    ┌────┴────┐
                    ▼         ▼
                morphism   testgen
```

### Modules

| Module | Purpose |
|--------|---------|
| `parser` | Tokenizer, recursive-descent parser, AST nodes, pretty-printer |
| `statespace` | State-space construction by structural induction |
| `product` | Product construction for parallel composition (∥) |
| `lattice` | SCC quotient, reachability, meet/join, lattice checking |
| `termination` | Termination checking and WF-Par well-formedness |
| `morphism` | Morphism hierarchy between state spaces |
| `visualize` | Hasse diagram generation (DOT/Graphviz) |
| `testgen` | Test suite generation (valid paths, violations, incomplete prefixes) |
| `cli` | Command-line interface |

---

## 2. Installation

### Requirements

- Python 3.11 or later
- Optional: `graphviz` Python package and system `dot` binary (for diagram rendering)

### Setup

```bash
# Clone the repository
git clone https://github.com/zuacaldeira/SessionTypesResearch.git
cd SessionTypesResearch/reticulate

# No installation needed — run directly from source
python3 -m reticulate --help

# Optional: install graphviz for diagram rendering
pip3 install graphviz

# Verify graphviz system binary
dot -V
```

### Running Tests

```bash
python3 -m pytest tests/ -v          # All 761 tests
python3 -m pytest tests/benchmarks/  # 34 benchmark protocols
```

---

## 3. Quick Start

### CLI: Check if a type forms a lattice

```bash
python3 -m reticulate "open . read . close . end"
```

Output:
```
Session type: open . read . close . end
States: 4, Transitions: 3
SCCs: 4
Is lattice: True
Top: 0, Bottom: 3
```

### CLI: Generate a Hasse diagram

```bash
python3 -m reticulate --hasse diagram.png "&{read: data.end, write: ack.end}"
```

### Python: Full analysis pipeline

```python
from reticulate import parse, build_statespace, check_lattice, dot_source

# 1. Parse
ast = parse("rec X . &{send: X, quit: end}")

# 2. Build state space
ss = build_statespace(ast)
print(f"States: {len(ss.states)}, Transitions: {len(ss.transitions)}")

# 3. Check lattice
result = check_lattice(ss)
print(f"Is lattice: {result.is_lattice}")
print(f"SCCs: {len(set(result.scc_map.values()))}")

# 4. Visualise
print(dot_source(ss, result))
```

---

## 4. Session Type Syntax

### Grammar

```
S  ::=  &{ m₁ : S₁ , … , mₙ : Sₙ }    — branch (external choice)
     |  +{ l₁ : S₁ , … , lₙ : Sₙ }    — selection (internal choice)
     |  ( S₁ || S₂ )                     — parallel composition
     |  rec X . S                        — recursion
     |  X                                — type variable
     |  end                              — terminated
     |  m . S                            — sequencing (sugar for &{m:S})
```

### Constructors Explained

| Constructor | Syntax | Meaning |
|-------------|--------|---------|
| **Branch** | `&{m1:S1, m2:S2}` | Client chooses which method to call |
| **Selection** | `+{OK:S1, ERR:S2}` | Object returns a label; continuation depends on it |
| **Parallel** | `(S1 \|\| S2)` | Two sub-protocols execute concurrently |
| **Recursion** | `rec X . S` | Loop: variable X refers back to this point |
| **Variable** | `X` | Reference to enclosing recursion binder |
| **End** | `end` | Protocol is complete |
| **Sequencing** | `m . S` | Sugar for `&{m : S}` (single-method branch) |

### Unicode Alternatives

The parser accepts both ASCII and Unicode syntax:

| ASCII | Unicode | Meaning |
|-------|---------|---------|
| `&{...}` | `&{...}` | branch |
| `+{...}` | `⊕{...}` | selection |
| `(S1 \|\| S2)` | `(S1 ∥ S2)` | parallel |
| `rec X . S` | `μ X . S` | recursion |

### Parallel Brace Notation

An alternative notation for parallel composition:

```
||{S1, S2}      equivalent to    (S1 || S2)
∥{S1, S2}       equivalent to    (S1 ∥ S2)
```

### Examples

**Simple chain** — a file object:
```
open . read . close . end
```

**Branch with selection** — a server response:
```
open . &{read: +{DATA: end, EOF: close.end}, write: ack.end}
```

**Recursion** — an iterator:
```
rec X . &{hasNext: +{TRUE: &{next: X}, FALSE: end}}
```

**SMTP connection**:
```
connect . ehlo . rec X . &{mail: rcpt . data . +{OK: X, ERR: X}, quit: end}
```

**Parallel composition** — concurrent read/write:
```
open . (read.end || write.end) . close . end
```

**Complex parallel** — bidirectional channel:
```
(rec X . &{send: X, done: end}) || (rec Y . +{msg: Y, fin: end})
```

### Common Mistakes

| Mistake | Error | Fix |
|---------|-------|-----|
| `a.b.c` | Missing `end` | `a.b.c.end` |
| `&{m: end n: end}` | Missing comma | `&{m: end, n: end}` |
| `rec X . X` | Unguarded recursion | `rec X . &{m: X, n: end}` |
| `((a.end \|\| b.end) \|\| c.end)` | Nested parallel | Flatten to single level |
| `rec X . (a.X \|\| b.end)` | Cross-branch variable | Ensure each branch has its own rec |

---

## 5. Command-Line Interface

### Synopsis

```
python3 -m reticulate [OPTIONS] TYPE_STRING
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--help` | Show help message | — |
| `--dot` | Print DOT source to stdout | — |
| `--hasse [PATH]` | Render Hasse diagram to file | `hasse_output` |
| `--fmt {png,svg,pdf,dot}` | Output format for `--hasse` | `png` |
| `--no-labels` | Hide state labels (show IDs only) | show labels |
| `--no-edge-labels` | Hide transition labels on edges | show labels |
| `--title TITLE` | Title for the diagram | — |

### Output Modes

**1. Text summary (default)**

```bash
$ python3 -m reticulate "rec X.&{a:X, b:end}"
Session type: rec X . &{a: X, b: end}
States: 2, Transitions: 2
SCCs: 2
Is lattice: True
Top: 0, Bottom: 1
```

**2. DOT source (`--dot`)**

```bash
$ python3 -m reticulate --dot "&{m:end, n:end}"
digraph {
  rankdir=TB;
  node [shape=ellipse];
  0 [label="0" style=filled fillcolor=lightblue];
  1 [label="1" style=filled fillcolor=lightgreen];
  0 -> 1 [label="m"];
  0 -> 1 [label="n"];
}
```

Pipe to `dot` for rendering without the graphviz Python package:

```bash
python3 -m reticulate --dot "a.b.end" | dot -Tpng -o diagram.png
```

**3. Hasse diagram (`--hasse`)**

```bash
# Render to PNG (default)
python3 -m reticulate --hasse diagram.png "rec X.&{a:X, b:end}"

# Render to SVG
python3 -m reticulate --hasse diagram.svg --fmt svg "a.b.end"

# Render with title, no edge labels
python3 -m reticulate --hasse out.png --title "SMTP" --no-edge-labels \
  "connect.ehlo.rec X.&{mail:rcpt.data.+{OK:X,ERR:X},quit:end}"
```

Requires the `graphviz` Python package. If not installed, use `--dot` instead.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Parse error or invalid type |

### Batch Processing

```bash
# Check all types in a file (one per line)
while IFS= read -r type; do
  echo -n "$type → "
  python3 -m reticulate "$type" 2>&1 | grep "Is lattice"
done < types.txt
```

---

## 6. Library API

### Workflow 1: Parse and Inspect

```python
from reticulate import parse, pretty, End, Branch, Rec, Var

# Parse a type string
ast = parse("rec X . &{a: X, b: end}")

# Inspect the AST
assert isinstance(ast, Rec)
assert ast.var == "X"
assert isinstance(ast.body, Branch)

# Pretty-print back to string
print(pretty(ast))  # "rec X . &{a: X, b: end}"
```

### Workflow 2: Build and Explore State Space

```python
from reticulate import parse, build_statespace

ast = parse("&{read: data.end, write: ack.end}")
ss = build_statespace(ast)

# State space properties
print(f"States: {ss.states}")          # {0, 1, 2, 3}
print(f"Top: {ss.top}, Bottom: {ss.bottom}")  # 0, 3
print(f"Transitions: {ss.transitions}")  # {(0,'read',1), (1,'data',3), ...}

# Query enabled methods at a state
print(ss.enabled_methods(0))   # {'read', 'write'}
print(ss.enabled_methods(1))   # {'data'}

# Check if a transition is a selection (internal choice)
print(ss.is_selection(0, 'read'))  # False (branch)
```

### Workflow 3: Check Lattice Property

```python
from reticulate import parse, build_statespace, check_lattice

ast = parse("rec X . &{send: X, quit: end}")
ss = build_statespace(ast)
result = check_lattice(ss)

print(f"Is lattice: {result.is_lattice}")     # True
print(f"SCCs: {len(set(result.scc_map.values()))}")  # 2
print(f"Counterexample: {result.counterexample}")     # None

# Compute specific meets and joins
from reticulate import compute_meet, compute_join

meet = compute_meet(result, 0, 1)  # Meet of states 0 and 1
join = compute_join(result, 0, 1)  # Join of states 0 and 1
```

### Workflow 4: Check Termination

```python
from reticulate import parse, is_terminating, check_termination, check_wf_parallel

# Check if a recursive type terminates
ast = parse("rec X . &{a: X, b: end}")
print(is_terminating(ast))  # True

ast_bad = parse("rec X . &{a: X, b: X}")
print(is_terminating(ast_bad))  # False (no exit path)

# Detailed termination result
result = check_termination(ast)
print(result.is_terminating)  # True
print(result.reason)          # Explanation string

# Check well-formedness of parallel composition
ast_par = parse("(a.end || b.end)")
wf = check_wf_parallel(ast_par)
print(wf.is_well_formed)     # True
print(wf.termination_ok)     # True
print(wf.no_cross_vars)      # True
print(wf.no_nested_parallel) # True
```

### Workflow 5: Compare State Spaces (Morphisms)

```python
from reticulate import (
    parse, build_statespace, check_lattice,
    find_isomorphism, find_embedding, classify_morphism,
    is_galois_connection,
)

# Build two state spaces
ss1 = build_statespace(parse("a.b.end"))
ss2 = build_statespace(parse("c.d.end"))

# Find isomorphism (if exists)
iso = find_isomorphism(ss1, ss2)
if iso:
    print(f"Isomorphic! Mapping: {iso.mapping}")

# Find embedding (injective, order-preserving + reflecting)
emb = find_embedding(ss1, ss2)

# Classify an explicit mapping
mapping = {0: 0, 1: 1, 2: 2}
r1 = check_lattice(ss1)
r2 = check_lattice(ss2)
morph = classify_morphism(ss1, ss2, r1, r2, mapping)
print(morph.kind)  # "isomorphism", "embedding", "projection", or "homomorphism"

# Check Galois connection
gc = is_galois_connection(r1, r2, alpha=mapping, gamma={0:0, 1:1, 2:2})
print(gc)  # True/False
```

### Workflow 6: Generate Hasse Diagram

```python
from reticulate import parse, build_statespace, check_lattice, dot_source, render_hasse

ast = parse("&{m: a.end, n: b.end}")
ss = build_statespace(ast)
result = check_lattice(ss)

# Get DOT source as string (no dependencies)
dot = dot_source(ss, result)
print(dot)

# Render to file (requires graphviz)
render_hasse(ss, result, filename="my_diagram", fmt="svg")

# Get graphviz.Source object for Jupyter notebooks
from reticulate import hasse_diagram
src = hasse_diagram(ss, result)
src  # Renders inline in Jupyter
```

### Workflow 7: Generate Tests

```python
from reticulate import (
    parse, build_statespace, check_lattice,
    TestGenConfig, enumerate, enumerate_client_programs,
    generate_test_source,
)

ast = parse("&{open: &{read: close.end, write: close.end}}")
ss = build_statespace(ast)
lr = check_lattice(ss)

# Configure test generation
config = TestGenConfig(
    class_name="FileHandle",
    package_name="com.example",
    var_name="file",
    max_revisits=2,
    max_paths=100,
)

# Enumerate paths, violations, incomplete prefixes
result = enumerate(ss, config)
print(f"Valid paths: {len(result.valid_paths)}")
print(f"Violations: {len(result.violations)}")
print(f"Incomplete prefixes: {len(result.incomplete_prefixes)}")
print(f"Truncated: {result.truncated}")

# Generate JUnit 5 source code
source = generate_test_source(ss, lr, config)
print(source)
```

Output (JUnit 5):
```java
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FileHandleProtocolTest {

    @Test
    void validPath_open_read_close() {
        FileHandle file = new FileHandle();
        file.open();
        file.read();
        file.close();
    }

    @Test
    void violation_read_atInitial() {
        FileHandle file = new FileHandle();
        assertThrows(Exception.class, () -> file.read());
    }
    // ...
}
```

---

## 7. Understanding Results

### LatticeResult

The `check_lattice()` function returns a `LatticeResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `is_lattice` | `bool` | Whether the SCC quotient forms a bounded lattice |
| `counterexample` | `tuple[int,int] \| None` | Pair of states without meet or join (if not a lattice) |
| `scc_map` | `dict[int,int]` | Maps each state to its SCC representative |
| `quotient_states` | `set[int]` | The SCC representatives (quotient nodes) |
| `quotient_transitions` | `set[tuple[int,str,int]]` | Transitions between SCC representatives |
| `reachable_from` | `dict[int,set[int]]` | Reachability sets in the quotient |

### What "Is Lattice" Means

A state space is a **lattice** when:

1. It has a **top** (initial state) that reaches all other states.
2. It has a **bottom** (terminal state) reachable from all states.
3. Every pair of states has a **meet** (greatest lower bound) — the latest point reachable from both.
4. Every pair of states has a **join** (least upper bound) — the earliest point from which both are reachable.

### When a State Space Is NOT a Lattice

If `is_lattice` is `False`, `counterexample` gives a pair of states `(a, b)` that lack a meet or join. This happens with non-standard session type constructions (all well-formed session types produce lattices).

### SCC Quotient

Recursive types create cycles (back-edges from variable references). The SCC quotient collapses mutually reachable states into single representatives, restoring the partial order needed for lattice analysis.

Example: `rec X . &{a: X, b: end}` creates a cycle between the branch state and itself. After quotienting, these merge into one representative.

---

## 8. Hasse Diagrams

### Reading a Hasse Diagram

A Hasse diagram is a visual representation of the state space ordered by reachability:

- **Nodes** = states (SCC representatives after quotienting)
- **Edges** = transitions (directed top-to-bottom)
- **Top node** (blue) = initial protocol state
- **Bottom node** (green) = terminal state (`end`)
- **Red nodes** = counterexample pair (when not a lattice)

### Node Labels

By default, each node shows its state ID and the enabled methods:

```
[0] open, write    ← state 0 with two enabled methods
```

Use `--no-labels` to show only state IDs, or `--no-edge-labels` to hide transition labels.

### SCC Collapsing

When recursion creates cycles, the diagram shows SCC representatives. Multiple states collapsed into one SCC appear as a single node with all their combined transitions.

### Examples

**Chain (total order):**
```
open . read . close . end

    [0] open
       |
    [1] read
       |
    [2] close
       |
    [3] end
```

**Diamond (branch):**
```
&{read: data.end, write: ack.end}

       [0]
      /   \
   [1]     [2]
      \   /
       [3]
```

**Product (parallel):**
```
(a.end || b.end)

     (0,0)
     /   \
  (1,0) (0,1)
     \   /
     (1,1)
```

---

## 9. Test Generation

### Overview

Reticulate generates three categories of protocol conformance tests:

| Category | Purpose | What it tests |
|----------|---------|---------------|
| **Valid paths** | Complete correct executions | Every top-to-bottom path is a valid usage |
| **Violations** | Illegal method calls | Calling a disabled method at a reachable state |
| **Incomplete prefixes** | Premature termination | Stopping before reaching the terminal state |

### Configuration

```python
TestGenConfig(
    class_name="MyProtocol",     # Name of the class under test
    package_name="com.example",  # Java package (optional)
    var_name="obj",              # Variable name in generated tests
    max_revisits=2,              # Max times to revisit a state (for recursion)
    max_paths=100,               # Cap on total valid paths
)
```

### Selection-Aware Tests

When the protocol has selection (internal choice), the generated tests use `switch` statements:

```java
@Test
void validPath_mail_response() {
    SmtpConnection smtp = new SmtpConnection();
    smtp.connect();
    smtp.ehlo();
    smtp.mail();
    smtp.rcpt();
    smtp.data();
    switch (smtp.response()) {
        case OK -> { smtp.quit(); }
        case ERR -> { smtp.quit(); }
    }
}
```

### Enumeration Result

```python
result = enumerate(ss, config)

# Valid paths: list of ValidPath objects
for path in result.valid_paths:
    print([step.label for step in path.steps])

# Violations: list of ViolationPoint objects
for v in result.violations:
    print(f"At state {v.state}, method '{v.disabled}' is disabled")
    print(f"Reached via: {v.prefix_labels()}")

# Incomplete prefixes: list of IncompletePrefix objects
for ip in result.incomplete_prefixes:
    print(f"Stops at state {ip.state}: {ip.labels()}")
```

---

## 10. Advanced Topics

### Product Construction

The parallel constructor `(S1 || S2)` builds a product state space:

```python
from reticulate import parse, build_statespace

ast = parse("(a.b.end || c.d.end)")
ss = build_statespace(ast)
print(f"States: {len(ss.states)}")  # 9 (3 × 3)
```

The product has `|Q₁| × |Q₂|` states with componentwise transitions. Each thread can advance independently; both must reach `end` for the product to terminate.

### Direct Product Construction

For fine-grained control:

```python
from reticulate import parse, build_statespace, product_statespace

ss1 = build_statespace(parse("a.b.end"))
ss2 = build_statespace(parse("c.d.end"))
product = product_statespace(ss1, ss2)
print(f"Product states: {len(product.states)}")  # 9
```

### Morphism Hierarchy

The four levels of structural relationship:

```
Isomorphism ⊂ Embedding ⊂ Projection ⊂ Homomorphism
```

| Level | Order-preserving | Order-reflecting | Bijective |
|-------|:---:|:---:|:---:|
| Isomorphism | Yes | Yes | Yes |
| Embedding | Yes | Yes | Injective |
| Projection | Yes | — | Surjective |
| Homomorphism | Yes | — | — |

Use cases:
- **Isomorphism**: Two protocols are structurally identical (same state space up to renaming).
- **Embedding**: One protocol is a sub-protocol of another (strict refinement).
- **Projection**: One protocol is an abstraction of another (states merged).
- **Galois connection**: An abstraction-concretisation pair relating two protocols.

### Working with SCCs

```python
from reticulate import parse, build_statespace, check_lattice

ast = parse("rec X . &{a: X, b: end}")
ss = build_statespace(ast)
result = check_lattice(ss)

# SCC map: state → representative
for state, rep in result.scc_map.items():
    print(f"State {state} → SCC {rep}")

# Quotient graph
print(f"Quotient states: {result.quotient_states}")
print(f"Quotient transitions: {result.quotient_transitions}")
```

### Performance Considerations

- State spaces are small for typical protocols (< 100 states).
- Product construction can be expensive: `|Q₁| × |Q₂|` states.
- Lattice checking is O(n²) in the number of quotient states.
- Test generation with high `max_revisits` on recursive types can produce many paths — use `max_paths` to cap.
- The 34 benchmarks range from 3 to 19 states, all checking in < 1 second.

---

## 11. Troubleshooting

### Parse Errors

**"Unexpected token"**
Check your syntax. Common issues:
- Missing commas between branches: `&{a:end b:end}` → `&{a:end, b:end}`
- Missing `end` at the end: `a.b` → `a.b.end`
- Unmatched braces: `&{a:end` → `&{a:end}`

**"Unexpected end of input"**
The type string is incomplete. Ensure all constructs are closed.

### Graphviz Errors

**"ImportError: graphviz package not installed"**
The `--hasse` flag requires the graphviz Python package:
```bash
pip3 install graphviz
```

**"ExecutableNotFound: failed to execute 'dot'"**
The system graphviz binary is missing:
```bash
# Ubuntu/Debian
sudo apt install graphviz

# macOS
brew install graphviz
```

**Workaround**: Use `--dot` to get DOT source, then render manually:
```bash
python3 -m reticulate --dot "a.end" | dot -Tpng -o out.png
```

### Lattice Check Fails

If `is_lattice` is `False` for a well-formed type, this indicates a bug. All well-formed session types should produce lattices. Please report the type string and output.

### Non-Terminating Types

If a parallel composition fails WF-Par, check:
1. Both branches must have at least one path to `end`.
2. No recursion variable from one branch is used in the other.
3. No nested `||` inside a parallel branch.

### Large State Spaces

If analysis is slow (> 5 seconds), the type likely has deep recursion with parallel. Consider:
- Reducing `max_revisits` for test generation.
- Using `--dot` instead of `--hasse` (avoids graphviz layout computation).

---

## 12. API Reference

### parser

| Function/Class | Signature | Description |
|-------|-----------|-------------|
| `parse(s)` | `str → SessionType` | Parse a session type string into an AST |
| `pretty(st)` | `SessionType → str` | Pretty-print an AST back to a string |
| `tokenize(s)` | `str → list[str]` | Tokenize a type string |
| `End` | dataclass | Terminal type |
| `Var(name)` | dataclass | Type variable |
| `Branch(choices)` | dataclass | External choice `&{...}` |
| `Select(choices)` | dataclass | Internal choice `+{...}` |
| `Parallel(left, right)` | dataclass | Parallel `(S1 \|\| S2)` |
| `Rec(var, body)` | dataclass | Recursion `rec X . S` |
| `Sequence(left, right)` | dataclass | Sequencing `S1 . S2` |
| `ParseError` | exception | Raised on invalid syntax |

### statespace

| Function/Class | Signature | Description |
|-------|-----------|-------------|
| `build_statespace(st)` | `SessionType → StateSpace` | Build LTS from AST |
| `StateSpace` | dataclass | State space with `states`, `transitions`, `top`, `bottom` |
| `StateSpace.enabled_methods(s)` | `int → set[str]` | Methods enabled at state `s` |
| `StateSpace.enabled_selections(s)` | `int → set[str]` | Selection labels at state `s` |
| `StateSpace.is_selection(s, label)` | `int, str → bool` | Is the transition a selection? |

### product

| Function | Signature | Description |
|----------|-----------|-------------|
| `product_statespace(left, right)` | `StateSpace, StateSpace → StateSpace` | Compute L₁ × L₂ |

### lattice

| Function/Class | Signature | Description |
|-------|-----------|-------------|
| `check_lattice(ss)` | `StateSpace → LatticeResult` | Check if SCC quotient is a lattice |
| `compute_meet(lr, a, b)` | `LatticeResult, int, int → int \| None` | Meet of two states |
| `compute_join(lr, a, b)` | `LatticeResult, int, int → int \| None` | Join of two states |
| `LatticeResult` | dataclass | Result with `is_lattice`, `counterexample`, `scc_map`, etc. |

### termination

| Function/Class | Signature | Description |
|-------|-----------|-------------|
| `is_terminating(st)` | `SessionType → bool` | Quick termination check |
| `check_termination(st)` | `SessionType → TerminationResult` | Detailed result with reason |
| `check_wf_parallel(st)` | `SessionType → WFParallelResult` | WF-Par well-formedness check |

### morphism

| Function/Class | Signature | Description |
|-------|-----------|-------------|
| `is_order_preserving(ss1, ss2, lr1, lr2, f)` | `..., dict → bool` | Check if mapping preserves order |
| `is_order_reflecting(ss1, ss2, lr1, lr2, f)` | `..., dict → bool` | Check if mapping reflects order |
| `classify_morphism(ss1, ss2, lr1, lr2, f)` | `..., dict → Morphism` | Classify as iso/embedding/projection/homo |
| `find_isomorphism(ss1, ss2)` | `StateSpace, StateSpace → Morphism \| None` | Find isomorphism by backtracking |
| `find_embedding(ss1, ss2)` | `StateSpace, StateSpace → Morphism \| None` | Find embedding by backtracking |
| `is_galois_connection(lr1, lr2, alpha, gamma)` | `..., dict, dict → bool` | Check adjunction α(x)≤y ⟺ x≤γ(y) |

### visualize

| Function | Signature | Description |
|----------|-----------|-------------|
| `dot_source(ss, lr, **opts)` | `StateSpace, LatticeResult → str` | Generate DOT source string |
| `hasse_diagram(ss, lr, **opts)` | `StateSpace, LatticeResult → graphviz.Source` | Get graphviz Source object |
| `render_hasse(ss, lr, filename, fmt, **opts)` | `... → str` | Render to file, return path |

Options: `title=str`, `no_labels=bool`, `no_edge_labels=bool`.

### testgen

| Function/Class | Signature | Description |
|-------|-----------|-------------|
| `TestGenConfig(...)` | dataclass | Configuration (class_name, package, max_revisits, max_paths) |
| `enumerate(ss, config)` | `StateSpace, TestGenConfig → EnumerationResult` | Enumerate all paths, violations, prefixes |
| `enumerate_client_programs(ss, config)` | `StateSpace, TestGenConfig → list[ClientProgram]` | Tree-shaped programs with selection switches |
| `generate_test_source(ss, lr, config)` | `..., TestGenConfig → str` | Generate JUnit 5 Java source |
| `ValidPath` | dataclass | Sequence of `Step` from top to bottom |
| `ViolationPoint` | dataclass | Disabled method at reachable state |
| `IncompletePrefix` | dataclass | Path stopping before bottom |
| `EnumerationResult` | dataclass | Combined result with `truncated` flag |

---

## Appendix: 34 Benchmark Protocols

All 34 benchmarks are defined in `tests/benchmarks/protocols.py` and can be run with:

```bash
python3 -m pytest tests/benchmarks/ -v
```

Each benchmark specifies:
- Session type string
- Expected state count, transition count, SCC count
- Whether it uses the parallel constructor

All 34 form bounded lattices, validated by both Reticulate (Python) and BICA Reborn (Java).

---

*Reticulate — Session Types as Algebraic Reticulates*
*Alexandre Zua Caldeira, Independent Researcher, 2026*
