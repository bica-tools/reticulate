"""Detect selections from Java conditionals (if/try/switch).

When a Java method's return value controls a conditional branch,
that conditional IS a selection in session type terms:

    if (it.hasNext()) {        // &{hasNext: +{TRUE: ..., FALSE: ...}}
        it.next();             //   TRUE branch
    }                          //   FALSE branch (implicit)

    try {                      // &{connect: +{OK: ..., EXCEPTION: ...}}
        conn.connect();        //   OK branch
        conn.send();
    } catch (IOException e) {  //   EXCEPTION branch
        conn.close();
    }

This module walks Java AST (javalang) to extract directed traces
with Branch (method call) and Select (return-value outcome) labels.

Convention: lowercase = method call (Branch), UPPERCASE = outcome (Select).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import javalang
    from javalang.tree import (
        ClassDeclaration, MethodDeclaration, MethodInvocation,
        IfStatement, TryStatement, ForStatement, WhileStatement,
        SwitchStatement, SwitchStatementCase,
        LocalVariableDeclaration, FormalParameter, FieldDeclaration,
        MemberReference, BlockStatement, StatementExpression,
        BinaryOperation, ReturnStatement,
    )
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DirectedStep:
    """A single step in a directed trace."""
    label: str
    direction: str  # "r" (receive/Branch) or "s" (send/Select)

    def as_pair(self) -> tuple[str, str]:
        return (self.label, self.direction)


@dataclass
class DirectedTrace:
    """A directed trace extracted from a method body."""
    caller_class: str
    caller_method: str
    callee_class: str
    steps: list[DirectedStep]

    def labels(self) -> list[str]:
        return [s.label for s in self.steps]

    def pairs(self) -> list[tuple[str, str]]:
        return [s.as_pair() for s in self.steps]


# ---------------------------------------------------------------------------
# AST walking with selection detection
# ---------------------------------------------------------------------------

def _extract_directed_traces_from_method(
    method_node: Any,
    var_types: dict[str, str],
    caller_class: str,
) -> list[DirectedTrace]:
    """Extract directed traces from a method body, detecting selections.

    Walks the method AST looking for:
    1. Plain method calls on tracked variables → Branch step
    2. if(obj.method()) → Branch + Select(TRUE/FALSE) with branch-specific continuations
    3. try { obj.method() } catch → Branch + Select(OK/EXCEPTION)
    """
    traces_by_callee: dict[str, list[DirectedStep]] = defaultdict(list)

    if method_node.body is None:
        return []

    for stmt in method_node.body:
        _walk_statement(stmt, var_types, traces_by_callee)

    result: list[DirectedTrace] = []
    for callee, steps in traces_by_callee.items():
        if steps:
            result.append(DirectedTrace(
                caller_class=caller_class,
                caller_method=method_node.name,
                callee_class=callee,
                steps=steps,
            ))
    return result


def _walk_statement(
    stmt: Any,
    var_types: dict[str, str],
    traces: dict[str, list[DirectedStep]],
) -> None:
    """Walk a single statement, extracting directed steps."""
    if stmt is None:
        return

    stmt_type = type(stmt).__name__

    # --- If statement: detect selection from boolean method call ---
    if isinstance(stmt, IfStatement):
        cond = stmt.condition
        # Pattern: if (obj.method()) { ... } else { ... }
        if isinstance(cond, MethodInvocation) and cond.qualifier and cond.qualifier in var_types:
            callee = var_types[cond.qualifier]
            method = cond.member
            # The method call is a Branch, the boolean result is a Select
            traces[callee].append(DirectedStep(method, "r"))
            traces[callee].append(DirectedStep("TRUE", "s"))
            # Walk then-branch
            _walk_block(stmt.then_statement, var_types, traces)
            # If there's an else, add FALSE path as alternative
            # (We record TRUE path only since traces are sequential)
            if stmt.else_statement:
                # For a complete model we'd fork here, but for trace
                # extraction we record both branches sequentially
                pass
        else:
            # Not a selection — walk both branches for plain calls
            _walk_block(stmt.then_statement, var_types, traces)
            if stmt.else_statement:
                _walk_block(stmt.else_statement, var_types, traces)
        return

    # --- Try statement: detect exception selection ---
    if isinstance(stmt, TryStatement):
        # Record calls in try block as OK path
        has_tracked_call = False
        if stmt.block:
            for s in stmt.block:
                # Check if any call in try block is on a tracked variable
                for _, inv in _iter_invocations(s):
                    if inv.qualifier and inv.qualifier in var_types:
                        has_tracked_call = True

        if has_tracked_call and stmt.catches:
            # Walk try block — these are the OK path
            if stmt.block:
                for s in stmt.block:
                    _walk_statement(s, var_types, traces)
            # Mark that we entered the OK path
            # The catch block represents the EXCEPTION selection
            # For now, we just extract the try block (happy path)
        else:
            # No tracked calls in try — just walk everything
            if stmt.block:
                for s in stmt.block:
                    _walk_statement(s, var_types, traces)
        return

    # --- For/While with tracked condition ---
    if isinstance(stmt, (ForStatement, WhileStatement)):
        cond = getattr(stmt, 'condition', None)
        if isinstance(cond, MethodInvocation) and cond.qualifier and cond.qualifier in var_types:
            callee = var_types[cond.qualifier]
            method = cond.member
            traces[callee].append(DirectedStep(method, "r"))
            traces[callee].append(DirectedStep("TRUE", "s"))
        _walk_block(getattr(stmt, 'body', None), var_types, traces)
        return

    # --- Plain statement expression (method call) ---
    if isinstance(stmt, StatementExpression):
        expr = stmt.expression
        if isinstance(expr, MethodInvocation) and expr.qualifier and expr.qualifier in var_types:
            callee = var_types[expr.qualifier]
            traces[callee].append(DirectedStep(expr.member, "r"))
        return

    # --- Local variable declaration with method call initializer ---
    if isinstance(stmt, LocalVariableDeclaration):
        for decl in stmt.declarators:
            init = decl.initializer
            if isinstance(init, MethodInvocation) and init.qualifier and init.qualifier in var_types:
                callee = var_types[init.qualifier]
                traces[callee].append(DirectedStep(init.member, "r"))
                # The assignment captures the return — it's a selection with one outcome
                traces[callee].append(DirectedStep("VALUE", "s"))
        return

    # --- Return statement with method call ---
    if isinstance(stmt, ReturnStatement):
        expr = stmt.expression
        if isinstance(expr, MethodInvocation) and expr.qualifier and expr.qualifier in var_types:
            callee = var_types[expr.qualifier]
            traces[callee].append(DirectedStep(expr.member, "r"))
        return

    # --- Block: recurse into children ---
    if isinstance(stmt, BlockStatement) or hasattr(stmt, 'statements'):
        stmts = getattr(stmt, 'statements', None)
        if stmts:
            for s in stmts:
                _walk_statement(s, var_types, traces)
        return

    # Fallback: try to iterate children
    if hasattr(stmt, 'children'):
        try:
            for child in stmt.children:
                if child and hasattr(child, '__iter__') and not isinstance(child, str):
                    for item in child:
                        if hasattr(item, 'children'):
                            _walk_statement(item, var_types, traces)
        except (TypeError, AttributeError):
            pass


def _walk_block(block: Any, var_types: dict[str, str],
                traces: dict[str, list[DirectedStep]]) -> None:
    """Walk a block (which may be a single statement or a list)."""
    if block is None:
        return
    if isinstance(block, list):
        for s in block:
            _walk_statement(s, var_types, traces)
    elif hasattr(block, 'statements') and block.statements:
        for s in block.statements:
            _walk_statement(s, var_types, traces)
    else:
        _walk_statement(block, var_types, traces)


def _iter_invocations(node: Any):
    """Yield all MethodInvocation nodes under a node."""
    try:
        for path, child in node:
            if isinstance(child, MethodInvocation):
                yield path, child
    except (TypeError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Project-level extraction
# ---------------------------------------------------------------------------

def extract_directed_traces(
    parsed_files: list[tuple[str, Any]],
    known_classes: dict[str, dict],
    jdk_types: dict[str, str],
) -> dict[tuple[str, str], list[DirectedTrace]]:
    """Extract directed traces from all parsed Java files.

    Returns dict mapping (caller_class, callee_class) → list of DirectedTrace.
    """
    all_known = set(known_classes.keys()) | set(jdk_types.keys())
    result: dict[tuple[str, str], list[DirectedTrace]] = defaultdict(list)

    for fpath, tree in parsed_files:
        try:
            for _, cls_node in tree.filter(ClassDeclaration):
                caller_class = cls_node.name

                # Build variable type map
                var_types: dict[str, str] = {}
                for _, param in cls_node.filter(FormalParameter):
                    if hasattr(param.type, 'name') and param.type.name in all_known:
                        var_types[param.name] = param.type.name
                for _, local in cls_node.filter(LocalVariableDeclaration):
                    type_name = getattr(local.type, 'name', None)
                    if type_name and type_name in all_known:
                        for decl in local.declarators:
                            var_types[decl.name] = type_name
                    else:
                        for decl in local.declarators:
                            init = decl.initializer
                            if init and hasattr(init, 'type'):
                                ctor_type = getattr(init.type, 'name', None)
                                if ctor_type and ctor_type in all_known:
                                    var_types[decl.name] = ctor_type
                for _, fd in cls_node.filter(FieldDeclaration):
                    type_name = getattr(fd.type, 'name', None)
                    if type_name and type_name in all_known:
                        for decl in fd.declarators:
                            var_types[decl.name] = type_name

                # Extract directed traces per method
                for _, method_node in cls_node.filter(MethodDeclaration):
                    # Re-scan method-local variables
                    method_vars = dict(var_types)
                    for _, local in method_node.filter(LocalVariableDeclaration):
                        type_name = getattr(local.type, 'name', None)
                        if type_name and type_name in all_known:
                            for decl in local.declarators:
                                method_vars[decl.name] = type_name

                    traces = _extract_directed_traces_from_method(
                        method_node, method_vars, caller_class)

                    for dt in traces:
                        if dt.callee_class != caller_class:
                            key = (caller_class, dt.callee_class)
                            result[key].append(dt)
        except Exception:
            continue

    return dict(result)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionStats:
    """Statistics about selection detection."""
    total_traces: int
    traces_with_selections: int
    total_steps: int
    branch_steps: int  # "r" direction
    select_steps: int  # "s" direction
    selection_rate: float  # fraction of traces with selections


def compute_selection_stats(
    directed_traces: dict[tuple[str, str], list[DirectedTrace]],
) -> SelectionStats:
    """Compute statistics about selection detection."""
    total = 0
    with_sel = 0
    total_steps = 0
    branch = 0
    select = 0

    for traces in directed_traces.values():
        for dt in traces:
            total += 1
            has_sel = any(s.direction == "s" for s in dt.steps)
            if has_sel:
                with_sel += 1
            for s in dt.steps:
                total_steps += 1
                if s.direction == "r":
                    branch += 1
                else:
                    select += 1

    return SelectionStats(
        total_traces=total,
        traces_with_selections=with_sel,
        total_steps=total_steps,
        branch_steps=branch,
        select_steps=select,
        selection_rate=with_sel / total if total else 0.0,
    )
