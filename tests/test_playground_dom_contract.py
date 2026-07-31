import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
APP_PATH = ROOT / "website/public/playground/app.js"
MARKUP_PATHS = [
    ROOT / "website/src/pages/playground.astro",
    ROOT / "website/public/playground/harness.html",
]


def _block(source: str, name: str) -> str:
    match = re.search(rf"const {name} = \[(.*?)\];", source, re.DOTALL)
    assert match is not None
    return match.group(1)


def _required_element_ids(source: str) -> set[str]:
    direct_ids = set(re.findall(r'\bel\("([^"]+)"\)', source)) - {"btn-theme"}
    directive_ids = set(
        re.findall(r'\["([^"]+)",', _block(source, "DIRECTIVE_CONTROLS"))
    )
    snippet_match = re.search(r"const SNIPPETS = \{(.*?)\n\};", source, re.DOTALL)
    assert snippet_match is not None
    snippet_ids = set(
        re.findall(r'^\s*"([^"]+)":', snippet_match.group(1), re.MULTILINE)
    )
    return direct_ids | directive_ids | snippet_ids


@pytest.mark.parametrize("markup_path", MARKUP_PATHS, ids=lambda path: path.name)
def test_playground_markup_provides_required_elements(markup_path: Path) -> None:
    required_ids = _required_element_ids(APP_PATH.read_text())
    markup_ids = set(re.findall(r'\bid="([^"]+)"', markup_path.read_text()))

    assert required_ids <= markup_ids, (
        f"{markup_path.name} lacks required element IDs: "
        f"{sorted(required_ids - markup_ids)}"
    )
