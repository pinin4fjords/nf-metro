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

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
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

    def visit_Lambda(self, node: ast.Lambda) -> None:
        parent = ".".join(self.scope)
        ordinal = self.lambda_ordinals.get(parent, 0) + 1
        self.lambda_ordinals[parent] = ordinal
        name = f"<lambda>#{ordinal}"
        qualname = f"{parent}.{name}" if parent else name
        if result := _classify(node):
            axis, fields = result
            self.sites[qualname] = SingleAxisSite(axis, fields, node.lineno)
        self.scope.extend((name, "<locals>"))
        self.visit(node.body)
        del self.scope[-2:]


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
