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
        assert match, f"{workflow.name}:{job_name} has no timeout-minutes"
        assert int(match.group("minutes")) > 0


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
    workflow = WORKFLOWS / "pr-renders.yml"
    jobs = _jobs(workflow)
    assert re.search(
        r"^    needs: \[lint, format\]$", jobs["render-diff"], re.MULTILINE
    )
    assert "base_render_key=" in jobs["render-diff"]
    assert (
        "key: render-base-${{ steps.strategy.outputs.base_render_key }}"
        in jobs["render-diff"]
    )
