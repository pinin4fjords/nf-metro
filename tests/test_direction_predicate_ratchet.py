"""Ratchet on direction-keyed predicates in the geometry-bearing packages.

``test_tb_branch_ratchet`` counts bare ``"TB"`` references.  The other way a
heuristic gets keyed to an orientation is a membership test or lookup table over
a *proper subset* of the flow directions -- ``direction in ("LR", "RL")``,
``_FLIP_HORIZONTAL = {"LR": "RL", "RL": "LR"}`` -- or a predicate that hides
the same distinction behind a name such as ``tb_positive_fan``.  Each such
subset is some axis property spelled out by hand, and a partial spelling is how
one orientation of a shape ends up on a different code path from another.

``AxisFrame`` (``layout/geometry.py``) already supplies the properties:

===================  =========================================================
subset               the property it is standing in for
===================  =========================================================
``{LR, RL}``         flow runs along X: ``lanes_run_along_y(direction)``
``{TB, BT}``         flow runs along Y: ``lanes_run_along_x(direction)``
``{RL, BT}``         flow is reversed: ``AxisFrame.flow_sign(direction) < 0``
``{TB, RL}``         two properties at once, and needs splitting: a horizontal
                     flow reversed on X, or a vertical flow whose lanes fan to
                     -X (``AxisFrame.secondary_sign_for(direction) < 0``)
===================  =========================================================

``layout/geometry.py`` is exempt: it is where those accessors are defined, so
its direction literals are the vocabulary rather than a use of it.

The bounds are upper ones. Lower them whenever a call site migrates onto the
accessors, and never raise them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable

from nf_metro.parser.model import FLOW_DIRECTIONS

_SRC = Path(__file__).resolve().parents[1] / "src" / "nf_metro"

# Packages whose code makes geometric decisions and so can be orientation-keyed.
_SCANNED_PACKAGES = ("layout", "parser", "render")

# The accessors' own definitions, not a use of them.
_EXEMPT = frozenset({"layout/geometry.py"})

# Lower these (never raise them) when a call site migrates onto AxisFrame.
_LITERAL_BASELINE = 64
_NAMED_BASELINE = 139

_FLOWS = frozenset(FLOW_DIRECTIONS)
_FLOW_NAME_TOKENS = frozenset(flow.lower() for flow in _FLOWS)


def _direction_subset(node: ast.expr) -> frozenset[str]:
    """The flow directions a container literal is keyed on, if it is keyed only on them.

    A dict contributes its keys, a set/tuple/list its elements.  Returns empty
    for anything holding a non-direction string, so an unrelated lookup table
    that happens to contain ``"LR"`` is not counted.
    """
    if isinstance(node, ast.Dict):
        elements = [k for k in node.keys if k is not None]
    elif isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        elements = list(node.elts)
    else:
        return frozenset()
    values = {
        e.value
        for e in elements
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    }
    if not values or not values <= _FLOWS:
        return frozenset()
    return frozenset(values)


def _is_partial(subset: frozenset[str]) -> bool:
    """A subset that discriminates between orientations without covering them all."""
    return 2 <= len(subset) < len(_FLOWS)


def _qualified_name(node: ast.expr) -> str | None:
    """The dotted identifier named by *node*, excluding calls and expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _direction_named(node: ast.expr) -> str | None:
    """A direction-keyed identifier carried by *node*, if one is explicit."""
    name = _qualified_name(node)
    if name is None:
        return None
    tokens = {part.lower() for part in re.findall(r"[A-Za-z]+", name)}
    if tokens & _FLOW_NAME_TOKENS:
        return name.rsplit(".", 1)[-1]
    return None


def _literal_sites_in(path: Path) -> list[tuple[int, str]]:
    """Every literal direction-keyed table or membership test in *path*."""
    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            subset = _direction_subset(node.value)
            if _is_partial(subset):
                names = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))
                found.append(
                    (node.lineno, f"table {names or '<unnamed>'} over {sorted(subset)}")
                )
        elif isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        ):
            for comparator in node.comparators:
                subset = _direction_subset(comparator)
                if _is_partial(subset):
                    found.append((node.lineno, f"membership in {sorted(subset)}"))
    return sorted(found)


def _named_sites_in(path: Path) -> list[tuple[int, str]]:
    """Every direction-named membership or helper call in *path*."""
    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        ):
            for comparator in node.comparators:
                if name := _direction_named(comparator):
                    found.append((node.lineno, f"membership in named predicate {name}"))
        elif isinstance(node, ast.Call) and (name := _direction_named(node.func)):
            found.append((node.lineno, f"call to direction-keyed helper {name}"))
    return sorted(set(found))


def _scan(
    detector: Callable[[Path], list[tuple[int, str]]],
) -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    for package in _SCANNED_PACKAGES:
        for path in sorted((_SRC / package).rglob("*.py")):
            relative = path.relative_to(_SRC).as_posix()
            if relative in _EXEMPT:
                continue
            if sites := detector(path):
                out[relative] = sites
    return out


def direction_predicate_sites() -> dict[str, list[tuple[int, str]]]:
    """Map each scanned module to literal direction-container sites."""
    return _scan(_literal_sites_in)


def named_direction_predicate_sites() -> dict[str, list[tuple[int, str]]]:
    """Map each scanned module to direction-keyed named sites."""
    return _scan(_named_sites_in)


def _breakdown(sites: dict[str, list[tuple[int, str]]]) -> str:
    return "\n  ".join(
        f"{len(found):>3}  {module}"
        for module, found in sorted(sites.items(), key=lambda kv: -len(kv[1]))
    )


def test_no_new_literal_direction_keyed_predicates() -> None:
    total = sum(len(v) for v in direction_predicate_sites().values())

    # Guard against the counter silently matching nothing (packages moved, the
    # AST walk broken): the engine genuinely carries dozens of these today.
    assert total >= 40, (
        f"expected many direction-keyed predicates, found {total} - the counter "
        "may be broken or the packages restructured"
    )

    assert total <= _LITERAL_BASELINE, (
        "literal direction-keyed predicate count rose to "
        f"{total} (baseline {_LITERAL_BASELINE}).\n"
        "A heuristic that needs to know a section's axis or flow sense should ask "
        "AxisFrame (layout/geometry.py: lanes_run_along_x/y, AxisFrame.flow_sign, "
        "AxisFrame.secondary_sign_for) rather than testing membership of a subset "
        "of the flow directions. A partial subset is how one orientation of a "
        f"geometry reaches a different code path from another.\n  "
        f"{_breakdown(direction_predicate_sites())}"
    )


def test_no_new_named_direction_keyed_predicates() -> None:
    sites = named_direction_predicate_sites()
    total = sum(len(found) for found in sites.values())

    assert total <= _NAMED_BASELINE, (
        f"named direction-keyed site count rose to {total} "
        f"(baseline {_NAMED_BASELINE}).\n"
        "A direction embedded in a helper or collection name hides the same "
        "orientation split as a literal. Express the property through AxisFrame "
        "or add an orientation-neutral classifier instead.\n  "
        f"{_breakdown(sites)}"
    )


def test_direction_predicate_baselines_are_current() -> None:
    literal_total = sum(len(v) for v in direction_predicate_sites().values())
    named_total = sum(len(v) for v in named_direction_predicate_sites().values())
    assert literal_total == _LITERAL_BASELINE
    assert named_total == _NAMED_BASELINE


def test_exempt_modules_exist() -> None:
    """The exemption names a real module, so a rename cannot silently widen scope."""
    for relative in _EXEMPT:
        assert (_SRC / relative).is_file(), f"exempt module {relative} not found"


def test_geometry_defines_the_accessors_the_ratchet_points_at() -> None:
    """The suggested replacements exist, so the failure message stays actionable."""
    from nf_metro.layout import geometry

    for name in ("lanes_run_along_x", "lanes_run_along_y", "AxisFrame"):
        assert hasattr(geometry, name), f"geometry.{name} is missing"
    for name in ("flow_sign", "secondary_sign_for", "axes_for_direction"):
        assert hasattr(geometry.AxisFrame, name), f"AxisFrame.{name} is missing"


def test_named_direction_membership_is_counted(tmp_path: Path) -> None:
    source = tmp_path / "named_membership.py"
    source.write_text(
        "def selected(section_id, tb_positive_fan):\n"
        "    return section_id in tb_positive_fan\n"
    )

    assert _named_sites_in(source) == [
        (2, "membership in named predicate tb_positive_fan")
    ]


def test_named_direction_predicate_call_is_counted(tmp_path: Path) -> None:
    source = tmp_path / "named_call.py"
    source.write_text(
        "def selected(section):\n"
        "    if is_tb_positive_fan(section):\n"
        "        return True\n"
        "    return False\n"
    )

    assert _named_sites_in(source) == [
        (2, "call to direction-keyed helper is_tb_positive_fan")
    ]


def test_named_direction_classifier_call_is_counted(tmp_path: Path) -> None:
    source = tmp_path / "named_classifier.py"
    source.write_text(
        "def selected(graph):\n    return tb_positive_fan_sections(graph)\n"
    )

    assert _named_sites_in(source) == [
        (2, "call to direction-keyed helper tb_positive_fan_sections")
    ]
