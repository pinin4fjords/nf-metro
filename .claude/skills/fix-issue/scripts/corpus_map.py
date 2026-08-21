#!/usr/bin/env python3
"""Emit `output_name<TAB>source_path` for every render-diff entry.

The render preview keys its diff-entry divs by *output* name, which is not always
the entry id: pipelines carry a `pipeline_` prefix and nextflow conversions
declare an explicit `output`. Resolving those by searching for a matching
basename guesses at data `scripts/gallery.yaml` already states, and gets it wrong
for duplicate basenames, ids containing regex metacharacters, and any entry whose
output differs from its id.

Group semantics mirror scripts/build_gallery.py, which owns them:
  gallery              -> id, source_dir from the entry
  pipelines            -> pipeline_<id>, the gallery entry's source_dir if the id
                          also appears there, else examples/
  guide_examples       -> id, source dir examples/
  nextflow_conversions -> entry["output"], source examples/<id>.mmd
  test_fixtures        -> id, source dir tests/fixtures/

build_gallery.py silently skips an entry whose source is absent. This map emits it
anyway, so render_pairs.sh reports it as UNRESOLVED: for a review gate a missing
source should be loud, not invisible. --self-test asserts every source exists, so
the two only diverge when something is genuinely wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def corpus_map(repo: Path) -> dict[str, Path]:
    cfg = yaml.safe_load((repo / "scripts" / "gallery.yaml").read_text())
    out: dict[str, Path] = {}
    for entry in cfg.get("gallery", []):
        out[entry["id"]] = Path(entry["source_dir"]) / f"{entry['id']}.mmd"
    gallery_dirs = {e["id"]: Path(e["source_dir"]) for e in cfg.get("gallery", [])}
    for entry in cfg.get("pipelines", []):
        stem = entry["id"]
        out[f"pipeline_{stem}"] = gallery_dirs.get(stem, Path("examples")) / f"{stem}.mmd"
    groups = cfg.get("render_only", {})
    for stem in groups.get("guide_examples", []) or []:
        out[stem] = Path("examples") / f"{stem}.mmd"
    for entry in groups.get("nextflow_conversions", []) or []:
        out[entry["output"]] = Path("examples") / f"{entry['id']}.mmd"
    for stem in groups.get("test_fixtures", []) or []:
        out[stem] = Path("tests/fixtures") / f"{stem}.mmd"
    return out


def self_test(repo: Path) -> int:
    mapping = corpus_map(repo)
    failures = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok   ' if cond else 'FAIL '} {label}")
        if not cond:
            failures.append(label)

    check("map is non-empty", len(mapping) > 200)
    # A single positional argument must be honoured: it was previously read from
    # argv[1], so `corpus_map.py REPO` silently used the cwd instead.
    check("positional repo argument is honoured",
          parse_args(["/somewhere"]) == (False, Path("/somewhere")))
    check("--self-test takes an optional repo",
          parse_args(["--self-test", "/somewhere"]) == (True, Path("/somewhere")))
    check("no arguments falls back to cwd", parse_args([]) == (False, Path.cwd()))
    check("a pipelines entry carries the pipeline_ prefix",
          any(k.startswith("pipeline_") for k in mapping))
    check("a nextflow conversion maps output name, not id",
          mapping.get("nf_variant_calling_tuned") == Path("examples/variant_calling.mmd"))
    check("a pipelines entry follows its gallery source_dir",
          mapping.get("pipeline_seqinspector") == Path("examples/showcase/seqinspector.mmd"))
    missing = [f"{k} -> {v}" for k, v in mapping.items() if not (repo / v).exists()]
    check(f"every mapped source exists ({len(missing)} missing)", not missing)
    for m in missing[:5]:
        print(f"        {m}")
    return 1 if failures else 0


def parse_args(argv: list[str]) -> tuple[bool, Path]:
    """Returns (self_test, repo). Usage:
    corpus_map.py [REPO]              -> emit the map
    corpus_map.py --self-test [REPO]  -> run the self-test
    """
    if argv and argv[0] == "--self-test":
        return True, Path(argv[1]) if len(argv) > 1 else Path.cwd()
    return False, Path(argv[0]) if argv else Path.cwd()


if __name__ == "__main__":
    want_self_test, repo_root = parse_args(sys.argv[1:])
    if want_self_test:
        sys.exit(self_test(repo_root))
    for name, rel in sorted(corpus_map(repo_root).items()):
        print(f"{name}\t{rel}")
