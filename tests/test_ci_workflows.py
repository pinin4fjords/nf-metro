"""Repository invariants for GitHub Actions resource usage."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
JOB_HEADER = re.compile(r"^  (?P<name>[A-Za-z0-9_-]+):\n", re.MULTILINE)


def _jobs(path: Path) -> dict[str, str]:
    text = path.read_text()
    _, jobs = text.split("\njobs:\n", maxsplit=1)
    matches = list(JOB_HEADER.finditer(jobs))
    return {
        match.group("name"): jobs[
            match.start() : matches[index + 1].start()
            if index + 1 < len(matches)
            else None
        ]
        for index, match in enumerate(matches)
    }


@pytest.mark.parametrize(
    "workflow",
    sorted(WORKFLOWS.glob("*.yml")),
    ids=lambda path: path.name,
)
def test_every_workflow_job_has_a_finite_timeout(workflow: Path):
    for job_name, job in _jobs(workflow).items():
        match = re.search(r"^    timeout-minutes: (?P<minutes>\d+)$", job, re.MULTILINE)
        if match:
            assert int(match.group("minutes")) > 0
            continue
        assert re.search(
            r"^    uses: \./\.github/workflows/[A-Za-z0-9_-]+\.yml$",
            job,
            re.MULTILINE,
        ), f"{workflow.name}:{job_name} has no timeout-minutes"


@pytest.mark.parametrize("job_name", ["test", "routing-gates"])
def test_expensive_ci_jobs_wait_for_fast_checks(job_name: str):
    job = _jobs(WORKFLOWS / "ci.yml")[job_name]
    assert re.search(r"^    needs: \[lint, format\]$", job, re.MULTILINE)


def test_test_matrix_uses_committed_timings_and_reports_slow_tests():
    job = _jobs(WORKFLOWS / "ci.yml")["test"]
    assert (ROOT / ".test_durations").is_file()
    assert "fail-fast: true" in job
    assert "--splitting-algorithm least_duration" in job
    assert "--durations=25" in job


def test_render_diff_waits_for_fast_checks_and_uses_content_cache_key():
    caller = _jobs(WORKFLOWS / "ci.yml")["render-diff"]
    assert re.search(r"^    needs: \[lint, format\]$", caller, re.MULTILINE)
    assert "uses: ./.github/workflows/pr-renders.yml" in caller

    render_job = _jobs(WORKFLOWS / "pr-renders.yml")["render-diff"]
    assert "base_render_key=" in render_job
    assert "tests/layout_metrics.py" in render_job
    assert (
        "key: render-base-${{ steps.strategy.outputs.base_render_key }}" in render_job
    )


def test_render_publisher_follows_ci_artifact_instead_of_ci_conclusion():
    workflow = (WORKFLOWS / "pr-render-publish.yml").read_text()
    assert 'workflows: ["CI"]' in workflow
    assert "actions/runs/${RUN_ID}/artifacts?name=render-preview" in workflow
    assert "needs.resolve.outputs.artifact == 'true'" in workflow
