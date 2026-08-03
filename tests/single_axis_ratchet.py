"""AST support for counting layout functions that encode only one axis."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cache
from pathlib import Path

X_FIELDS = frozenset({"bbox_x", "bbox_w", "grid_col", "grid_col_span", "offset_x"})
Y_FIELDS = frozenset({"bbox_y", "bbox_h", "grid_row", "grid_row_span", "offset_y"})


@dataclass(frozen=True)
class SingleAxisSite:
    axis: str
    fields: frozenset[str]
    line: int


class _FieldReads(ast.NodeVisitor):
    def __init__(self) -> None:
        self.fields: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self.fields.add(node.attr)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _classify(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> tuple[str, frozenset[str]] | None:
    reads = _FieldReads()
    if isinstance(node, ast.Lambda):
        reads.visit(node.body)
    else:
        for statement in node.body:
            reads.visit(statement)
    x_fields = frozenset(reads.fields & X_FIELDS)
    y_fields = frozenset(reads.fields & Y_FIELDS)
    if len(x_fields) >= 2 and not y_fields:
        return ("x", x_fields)
    if len(y_fields) >= 2 and not x_fields:
        return ("y", y_fields)
    return None


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.sites: dict[str, SingleAxisSite] = {}
        self.lambda_ordinals: dict[str, int] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_default(
        self, function_name: str, argument: ast.arg, default: ast.expr
    ) -> None:
        if isinstance(default, ast.Lambda):
            self._visit_lambda(default, f"default:{function_name}.{argument.arg}")
        else:
            self.visit(default)

    def _visit_argument_defaults(self, owner: str, arguments: ast.arguments) -> None:
        positional = [*arguments.posonlyargs, *arguments.args]
        for argument, default in zip(
            positional[-len(arguments.defaults) :], arguments.defaults
        ):
            self._visit_default(owner, argument, default)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            if default is not None:
                self._visit_default(owner, argument, default)

    def _visit_definition_expressions(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_argument_defaults(node.name, node.args)
        positional = [*node.args.posonlyargs, *node.args.args]
        arguments = [*positional, *node.args.kwonlyargs]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._visit_definition_expressions(node)
        name_parts = [*self.scope, node.name]
        qualname = ".".join(name_parts)
        if result := _classify(node):
            axis, fields = result
            self.sites[qualname] = SingleAxisSite(axis, fields, node.lineno)
        self.scope.extend((node.name, "<locals>"))
        for statement in node.body:
            self.visit(statement)
        del self.scope[-2:]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_lambda(self, node: ast.Lambda, role: str | None = None) -> None:
        parent = ".".join(self.scope)
        base = f"<lambda:{role}>" if role else "<lambda>"
        ordinal_key = f"{parent}.{base}"
        ordinal = self.lambda_ordinals.get(ordinal_key, 0) + 1
        self.lambda_ordinals[ordinal_key] = ordinal
        name = base if role and ordinal == 1 else f"{base}#{ordinal}"
        qualname = f"{parent}.{name}" if parent else name
        owner = role.removeprefix("default:") if role else name
        self._visit_argument_defaults(owner, node.args)
        if result := _classify(node):
            axis, fields = result
            self.sites[qualname] = SingleAxisSite(axis, fields, node.lineno)
        self.scope.extend((name, "<locals>"))
        self.visit(node.body)
        del self.scope[-2:]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_lambda(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Lambda) and len(node.targets) == 1:
            target = node.targets[0]
            role = f"assigned:{target.id}" if isinstance(target, ast.Name) else None
            self._visit_lambda(node.value, role)
            return
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if isinstance(node.value, ast.Lambda) and node.arg is not None:
            self._visit_lambda(node.value, f"keyword:{node.arg}")
            return
        self.generic_visit(node)


def _sites_from_source(
    source: str, filename: str = "<unknown>"
) -> dict[str, SingleAxisSite]:
    collector = _FunctionCollector()
    collector.visit(ast.parse(source, filename=filename))
    return collector.sites


def single_axis_sites_from_source(
    source: str,
) -> dict[str, tuple[str, frozenset[str]]]:
    """Classify functions in an in-memory source used by detector unit tests."""
    return {
        name: (site.axis, site.fields)
        for name, site in _sites_from_source(source).items()
    }


@cache
def single_axis_sites(layout_dir: Path) -> dict[str, SingleAxisSite]:
    """Every one-axis function below *layout_dir*, keyed by stable qualname."""
    sites: dict[str, SingleAxisSite] = {}
    for path in sorted(layout_dir.rglob("*.py")):
        relative = path.relative_to(layout_dir).as_posix()
        for qualname, site in _sites_from_source(
            path.read_text(), filename=str(path)
        ).items():
            sites[f"{relative}::{qualname}"] = site
    return sites
