#!/usr/bin/env python3
"""Compare two directories of SVG renders and generate an HTML diff page.

Usage:
    python scripts/build_render_diff.py BASE_DIR PR_DIR OUTPUT_DIR [--pr NUMBER]

Compares SVG files in BASE_DIR (main branch renders) against PR_DIR (PR branch
renders). Generates a self-contained HTML page showing side-by-side before/after
for changed outputs only. Copies changed SVGs into OUTPUT_DIR for deployment.

Exit codes:
    0 - changes detected, diff page written
    1 - error
    2 - no changes detected
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Render diff{title_suffix}</title>
<style>
  :root {{
    --bg: #1e1e2e;
    --surface: #2a2a3c;
    --border: #3a3a4c;
    --text: #e0e0e0;
    --muted: #888;
    --accent: #4ec9b0;
    --added: #2ea04370;
    --removed: #da363470;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    color: var(--accent);
  }}
  .summary {{
    color: var(--muted);
    margin-bottom: 2rem;
    font-size: 0.95rem;
  }}
  .toc {{
    margin-bottom: 2rem;
    padding: 1rem 1.5rem;
    background: var(--surface);
    border-radius: 8px;
    border: 1px solid var(--border);
  }}
  .toc h2 {{
    font-size: 1rem;
    margin-bottom: 0.5rem;
    color: var(--accent);
  }}
  .toc ul {{
    list-style: none;
    columns: 2;
    column-gap: 2rem;
  }}
  .toc li {{
    margin-bottom: 0.25rem;
  }}
  .toc a {{
    color: var(--text);
    text-decoration: none;
  }}
  .toc a:hover {{
    color: var(--accent);
    text-decoration: underline;
  }}
  .badge {{
    display: inline-block;
    font-size: 0.7rem;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    margin-left: 0.4rem;
    vertical-align: middle;
  }}
  .badge-changed {{ background: var(--border); }}
  .badge-added {{ background: var(--added); }}
  .badge-removed {{ background: var(--removed); }}
  .diff-entry {{
    margin-bottom: 3rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .diff-entry h2 {{
    font-size: 1.1rem;
    padding: 0.75rem 1rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}
  .comparison {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
  }}
  .side {{
    padding: 1rem;
    overflow-x: auto;
  }}
  .side:first-child {{
    border-right: 1px solid var(--border);
  }}
  .side h3 {{
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .side img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
  }}
  .side-only {{
    padding: 1rem;
  }}
  .side-only img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
  }}
  .empty {{
    color: var(--muted);
    font-style: italic;
    padding: 2rem;
    text-align: center;
  }}
  .toggle-bar {{
    display: flex;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}
  .toggle-bar button {{
    background: var(--border);
    color: var(--text);
    border: none;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
  }}
  .toggle-bar button.active {{
    background: var(--accent);
    color: var(--bg);
  }}
</style>
</head>
<body>
<h1>Render diff{title_suffix}</h1>
<p class="summary">{summary}</p>
{toc}
{entries}
<script>
document.querySelectorAll('.toggle-bar button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const entry = btn.closest('.diff-entry');
    const mode = btn.dataset.mode;
    const buttons = entry.querySelectorAll('.toggle-bar button');
    buttons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const comp = entry.querySelector('.comparison');
    const sideBase = comp.querySelector('.side-base');
    const sidePr = comp.querySelector('.side-pr');
    if (mode === 'side-by-side') {{
      comp.style.gridTemplateColumns = '1fr 1fr';
      sideBase.style.display = '';
      sidePr.style.display = '';
    }} else if (mode === 'base') {{
      comp.style.gridTemplateColumns = '1fr';
      sideBase.style.display = '';
      sidePr.style.display = 'none';
    }} else if (mode === 'pr') {{
      comp.style.gridTemplateColumns = '1fr';
      sideBase.style.display = 'none';
      sidePr.style.display = '';
    }}
  }});
}});
</script>
</body>
</html>
"""


def build_diff(
    base_dir: Path, pr_dir: Path, output_dir: Path, pr_number: str | None = None
) -> bool:
    """Compare renders and generate diff page. Returns True if changes found."""
    base_svgs = {p.name for p in base_dir.glob("*.svg")} if base_dir.exists() else set()
    pr_svgs = {p.name for p in pr_dir.glob("*.svg")} if pr_dir.exists() else set()
    all_names = sorted(base_svgs | pr_svgs)

    changed: list[tuple[str, str]] = []  # (name, kind)
    for name in all_names:
        base_path = base_dir / name
        pr_path = pr_dir / name
        if name in base_svgs and name in pr_svgs:
            if base_path.read_bytes() != pr_path.read_bytes():
                changed.append((name, "changed"))
        elif name in pr_svgs:
            changed.append((name, "added"))
        else:
            changed.append((name, "removed"))

    if not changed:
        return False

    # Set up output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    out_base = output_dir / "base"
    out_pr = output_dir / "pr"
    out_base.mkdir(exist_ok=True)
    out_pr.mkdir(exist_ok=True)

    # Copy changed SVGs
    for name, kind in changed:
        if kind in ("changed", "removed"):
            shutil.copy2(base_dir / name, out_base / name)
        if kind in ("changed", "added"):
            shutil.copy2(pr_dir / name, out_pr / name)

    # Build HTML
    title_suffix = f" - PR #{pr_number}" if pr_number else ""
    n_changed = sum(1 for _, k in changed if k == "changed")
    n_added = sum(1 for _, k in changed if k == "added")
    n_removed = sum(1 for _, k in changed if k == "removed")
    parts = []
    if n_changed:
        parts.append(f"{n_changed} changed")
    if n_added:
        parts.append(f"{n_added} added")
    if n_removed:
        parts.append(f"{n_removed} removed")
    summary = f"{', '.join(parts)} out of {len(all_names)} total renders."

    # Table of contents
    toc_items = []
    for name, kind in changed:
        stem = name.removesuffix(".svg")
        badge_class = f"badge-{kind}"
        toc_items.append(
            f'<li><a href="#{stem}">{stem}</a>'
            f'<span class="badge {badge_class}">{kind}</span></li>'
        )
    toc = (
        '<div class="toc">\n'
        "<h2>Changed renders</h2>\n"
        f"<ul>{''.join(toc_items)}</ul>\n"
        "</div>"
    )

    # Diff entries
    entries_html = []
    for name, kind in changed:
        stem = name.removesuffix(".svg")
        heading = stem.replace("_", " ").title()
        badge_class = f"badge-{kind}"

        if kind == "changed":
            entry = (
                f'<div class="diff-entry" id="{stem}">\n'
                f'<h2>{heading} <span class="badge {badge_class}">{kind}</span></h2>\n'
                f'<div class="toggle-bar">'
                f'<button class="active" data-mode="side-by-side">Side by side</button>'
                f'<button data-mode="base">Base only</button>'
                f'<button data-mode="pr">PR only</button>'
                f"</div>\n"
                f'<div class="comparison">\n'
                f'<div class="side side-base"><h3>Base (main)</h3>'
                f'<img src="base/{name}" alt="Base: {stem}"></div>\n'
                f'<div class="side side-pr"><h3>PR</h3>'
                f'<img src="pr/{name}" alt="PR: {stem}"></div>\n'
                f"</div>\n"
                f"</div>"
            )
        elif kind == "added":
            entry = (
                f'<div class="diff-entry" id="{stem}">\n'
                f'<h2>{heading} <span class="badge {badge_class}">{kind}</span></h2>\n'
                f'<div class="side-only"><h3>New in PR</h3>'
                f'<img src="pr/{name}" alt="PR: {stem}"></div>\n'
                f"</div>"
            )
        else:  # removed
            entry = (
                f'<div class="diff-entry" id="{stem}">\n'
                f'<h2>{heading} <span class="badge {badge_class}">{kind}</span></h2>\n'
                f'<div class="side-only"><h3>Removed (was in base)</h3>'
                f'<img src="base/{name}" alt="Base: {stem}"></div>\n'
                f"</div>"
            )
        entries_html.append(entry)

    html = HTML_TEMPLATE.format(
        title_suffix=title_suffix,
        summary=summary,
        toc=toc,
        entries="\n\n".join(entries_html),
    )
    (output_dir / "index.html").write_text(html)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate render diff page")
    parser.add_argument("base_dir", type=Path, help="Base branch render directory")
    parser.add_argument("pr_dir", type=Path, help="PR branch render directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for diff site")
    parser.add_argument("--pr", default=None, help="PR number for title")
    args = parser.parse_args()

    has_changes = build_diff(args.base_dir, args.pr_dir, args.output_dir, args.pr)
    if has_changes:
        print(f"Diff page written to {args.output_dir}/index.html")
        sys.exit(0)
    else:
        print("No render changes detected.")
        sys.exit(2)


if __name__ == "__main__":
    main()
