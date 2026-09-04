"""The shipped examples must render from any working directory.

Every other test and CI invocation renders from the repository root, where a
path written relative to the root resolves by accident. These tests render from
a directory that is not the root, so an example that only works from the root
reds here instead of reaching a reader.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from nf_metro.cli import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.mmd")) + sorted(
    (EXAMPLES_DIR / "showcase").glob("*.mmd")
)
LOGO_EXAMPLES = [p for p in EXAMPLE_FILES if "%%metro logo:" in p.read_text()]


def _render(source: Path, out: Path, cwd: Path, monkeypatch) -> str:
    """Render *source* by absolute path with the process cwd at *cwd*."""
    monkeypatch.chdir(cwd)
    result = CliRunner().invoke(cli, ["render", str(source), "-o", str(out)])
    assert result.exit_code == 0, (
        f"{source.name} failed to render from {cwd}:\n{result.output}"
    )
    return out.read_text()


def test_example_corpus_is_not_empty():
    """A corpus that globs to nothing would make every test below vacuous."""
    assert EXAMPLE_FILES
    assert LOGO_EXAMPLES


@pytest.mark.parametrize("source", EXAMPLE_FILES, ids=lambda p: p.stem)
def test_example_renders_from_a_foreign_cwd(source, tmp_path, monkeypatch):
    svg = _render(source, tmp_path / "out.svg", tmp_path, monkeypatch)
    assert svg.lstrip().startswith(("<?xml", "<svg"))


@pytest.mark.parametrize("source", LOGO_EXAMPLES, ids=lambda p: p.stem)
def test_logo_example_embeds_its_logo_from_a_foreign_cwd(source, tmp_path, monkeypatch):
    """An unresolvable logo path is an error, so a render proves resolution.

    The embedded asset is asserted as well, so downgrading that error to a
    warning cannot quietly ship these maps with the logo missing.
    """
    svg = _render(source, tmp_path / "out.svg", tmp_path, monkeypatch)
    assert "data:image/png;base64," in svg


@pytest.mark.parametrize("source", LOGO_EXAMPLES, ids=lambda p: p.stem)
def test_logo_example_renders_the_same_bytes_from_any_cwd(
    source, tmp_path, monkeypatch
):
    """Nothing about where the render runs may reach the rendered bytes."""
    from_root = _render(source, tmp_path / "root.svg", REPO_ROOT, monkeypatch)
    from_elsewhere = _render(source, tmp_path / "elsewhere.svg", tmp_path, monkeypatch)
    assert from_root == from_elsewhere
