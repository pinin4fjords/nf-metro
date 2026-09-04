"""Regression test: duplicate SVG ids across inlined render-diff panels.

The render-diff page (``scripts/build_render_diff.py``) inlines two copies of
the same pipeline's SVG side by side (base + PR). When both copies share the
same source (e.g. the mask markup is byte-identical), their ``id``s collide.
The base/PR/side-by-side toggle buttons switch panels via ``display:none``,
and a browser resolving ``url(#id)`` picks the first same-id element in
document order - if that first element sits inside a now-hidden subtree, the
reference stops working and the mask is dropped entirely, so both the light
and dark logo variants render unmasked on top of each other.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from nf_metro.api import render_string

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_render_diff import _ID_ATTR_RE as _ID_RE  # noqa: E402
from build_render_diff import _URL_REF_RE, _inline_svg, build_diff  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

_CLASS_ATTR_TOKENS_RE = re.compile(r'class="([^"]*)"')
_STYLE_SELECTOR_TOKEN_RE = re.compile(r"\.([A-Za-z_][\w-]*)")
_MUTED_RULE_PAIR_RE = re.compile(
    r"\.nf-metro-station-label\.nf-metro-muted\s*\{[^}]*\}\n?"
    r"\.nf-metro-marker-stroke\.nf-metro-muted\s*\{[^}]*\}\n?"
)


def _render_inactive_lines_svg() -> str:
    text = (EXAMPLES / "inactive_lines.mmd").read_text()
    return render_string(text, self_color_scheme=False)


def _strip_muted_rule(svg_text: str) -> str:
    """Delete the muted CSS rule pair from *svg_text*'s own stylesheet, leaving
    its ``nf-metro-muted``-classed elements with no local rule to give that
    class a meaning."""
    stripped, n = _MUTED_RULE_PAIR_RE.subn("", svg_text)
    assert n == 1, (
        "fixture must declare the muted CSS rule pair for this test to be meaningful"
    )
    return stripped


def _classes_used_by_elements(fragment: str) -> set[str]:
    """Every class token that appears on an element in *fragment*."""
    tokens: set[str] = set()
    for m in _CLASS_ATTR_TOKENS_RE.finditer(fragment):
        tokens.update(m.group(1).split())
    return tokens


def _classes_named_in_style_selectors(fragment: str) -> set[str]:
    """Every class name a selector inside *fragment*'s <style> block(s) references."""
    names: set[str] = set()
    for style_match in re.finditer(r"<style>(.*?)</style>", fragment, re.DOTALL):
        names.update(_STYLE_SELECTOR_TOKEN_RE.findall(style_match.group(1)))
    return names


def _render_adaptive_logo_svg() -> str:
    text = (EXAMPLES / "sarek_metro.mmd").read_text()
    return render_string(text, self_color_scheme=False)


def test_url_referenced_ids_are_unique_across_inlined_panels(tmp_path):
    """No two url(#...)-referenced ids may collide once base+PR are inlined together."""
    svg_text = _render_adaptive_logo_svg()
    assert _URL_REF_RE.search(svg_text), (
        "fixture must contain a url(#...) reference (e.g. an adaptive logo mask) "
        "for this test to be meaningful"
    )

    base_path = tmp_path / "sarek_metro.svg"
    pr_path = tmp_path / "sarek_metro_pr.svg"
    base_path.write_text(svg_text)
    pr_path.write_text(svg_text)

    base_inlined = _inline_svg(base_path, "sarek_metro-base")
    pr_inlined = _inline_svg(pr_path, "sarek_metro-pr")
    combined = base_inlined + pr_inlined

    referenced_ids = _URL_REF_RE.findall(combined)
    assert referenced_ids, "expected url(#...) references to survive inlining"

    defined_ids = _ID_RE.findall(combined)
    for ref in referenced_ids:
        assert defined_ids.count(ref) == 1, (
            f"id {ref!r} referenced by url(#{ref}) must be unique across "
            "inlined panels, or a display:none toggle on one panel breaks "
            "mask resolution for the other"
        )


def test_each_panel_still_resolves_its_own_references(tmp_path):
    """Renaming ids must not break a panel's own url(#...) -> id(...) resolution."""
    svg_text = _render_adaptive_logo_svg()
    path = tmp_path / "sarek_metro.svg"
    path.write_text(svg_text)

    inlined = _inline_svg(path, "sarek_metro-pr")
    defined_ids = set(_ID_RE.findall(inlined))
    for ref in _URL_REF_RE.findall(inlined):
        assert ref in defined_ids, (
            f"url(#{ref}) no longer resolves within its own panel"
        )


def test_build_diff_output_has_no_id_collisions(tmp_path):
    """End-to-end: the generated diff page itself must not collide any panel's ids."""
    svg_text = _render_adaptive_logo_svg()
    base_dir = tmp_path / "base"
    pr_dir = tmp_path / "pr"
    base_dir.mkdir()
    pr_dir.mkdir()
    (base_dir / "sarek_metro.svg").write_text(svg_text)
    # A trivial byte difference is enough to classify this as a "changed" render,
    # matching the case that renders both panels onto the page together.
    (pr_dir / "sarek_metro.svg").write_text(svg_text + "<!-- pr -->")

    output_dir = tmp_path / "out"
    assert build_diff(base_dir, pr_dir, output_dir)

    page = (output_dir / "index.html").read_text()
    referenced_ids = _URL_REF_RE.findall(page)
    assert referenced_ids
    defined_ids = _ID_RE.findall(page)
    for ref in referenced_ids:
        assert defined_ids.count(ref) == 1, (
            f"id {ref!r} referenced by url(#{ref}) collides in the generated page"
        )


def test_one_panels_stylesheet_cannot_select_the_other_panels_elements(tmp_path):
    """A CSS rule scoped to one panel's ``<style>`` block must not match the
    other panel's elements once both share one HTML document.

    Constructs a base/PR pair where only the base copy carries the
    ``nf-metro-muted`` rule; the PR copy's markup applies the class with no
    local rule to give it meaning. Inline SVG ``<style>`` is document-global,
    so an un-namespaced base rule matches the PR panel's identically-classed
    elements too, silently repainting them and masking the very stylesheet
    difference the two panels exist to surface.
    """
    svg_text = _render_inactive_lines_svg()
    defective_svg_text = _strip_muted_rule(svg_text)

    base_path = tmp_path / "inactive_lines.svg"
    pr_path = tmp_path / "inactive_lines_pr.svg"
    base_path.write_text(svg_text)
    pr_path.write_text(defective_svg_text)

    base_inlined = _inline_svg(base_path, "inactive_lines-base")
    pr_inlined = _inline_svg(pr_path, "inactive_lines-pr")

    base_selectors = _classes_named_in_style_selectors(base_inlined)
    pr_classes = _classes_used_by_elements(pr_inlined)
    matched = base_selectors & pr_classes
    assert not matched, (
        f"base panel's <style> selects PR panel element(s) via shared class "
        f"name(s) {matched}"
    )

    pr_selectors = _classes_named_in_style_selectors(pr_inlined)
    base_classes = _classes_used_by_elements(base_inlined)
    matched = pr_selectors & base_classes
    assert not matched, (
        f"PR panel's <style> selects base panel element(s) via shared class "
        f"name(s) {matched}"
    )


def test_build_diff_isolates_each_panels_css_and_leaves_the_corpus_untouched(tmp_path):
    """End-to-end: a genuine stylesheet difference between base and PR renders
    is not masked once both are inlined onto the same diff page, and the
    on-disk corpus SVGs the change-detection gate compares are untouched by
    page generation."""
    svg_text = _render_inactive_lines_svg()
    defective_svg_text = _strip_muted_rule(svg_text)

    base_dir = tmp_path / "base"
    pr_dir = tmp_path / "pr"
    base_dir.mkdir()
    pr_dir.mkdir()
    base_svg_path = base_dir / "inactive_lines.svg"
    pr_svg_path = pr_dir / "inactive_lines.svg"
    base_svg_path.write_text(svg_text)
    pr_svg_path.write_text(defective_svg_text)

    output_dir = tmp_path / "out"
    assert build_diff(base_dir, pr_dir, output_dir)

    assert base_svg_path.read_text() == svg_text
    assert pr_svg_path.read_text() == defective_svg_text

    page = (output_dir / "index.html").read_text()
    base_start = page.index('class="side side-base"')
    pr_start = page.index('class="side side-pr"')
    side_base = page[base_start:pr_start]
    side_pr = page[pr_start:]

    base_selectors = _classes_named_in_style_selectors(side_base)
    pr_classes = _classes_used_by_elements(side_pr)
    matched = base_selectors & pr_classes
    assert not matched, (
        f"base panel's <style> selects PR panel element(s) via shared class "
        f"name(s) {matched} on the generated diff page"
    )
