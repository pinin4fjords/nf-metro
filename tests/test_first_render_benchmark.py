"""Synthetic contract tests for the offline first-render benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "first_render_benchmark.py"
SPEC = importlib.util.spec_from_file_location("first_render_benchmark", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


SIMPLE_MMD = """\
%%metro title: Synthetic benchmark case
%%metro line: main | Main | #2db572
graph LR
    subgraph one [One]
        a[A] -->|main| b[B]
    end
"""


def test_human_rubric_uses_locked_issue_values() -> None:
    assert benchmark.VERDICTS == {
        "accepted_without_correction",
        "minor_polish_only",
        "major_layout_correction_required",
        "unusable_or_aborting",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _case(
    root: Path,
    case_id: str,
    status: str,
    *,
    with_human: bool = True,
) -> dict[str, object]:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    source_rel: str | None = None
    digest: str | None = None
    if status != "unavailable":
        source = case_dir / "source.mmd"
        source.write_text(SIMPLE_MMD)
        source_rel = source.relative_to(root).as_posix()
        digest = benchmark.sha256_file(source)
    provenance_rel = f"cases/{case_id}/provenance.json"
    _write_json(
        root / provenance_rel,
        {
            "schema_version": 1,
            "canonical_pipeline": f"synthetic/{case_id}",
            "original_timestamp": "2026-01-01T00:00:00Z",
            "source_url_or_ref": "synthetic:test",
            "retrieval_date": "2026-07-31",
            "source_sha256": digest,
            "assets": [],
            "reconstruction_status": status,
            "transformations": [] if status != "derived" else ["synthetic rewrite"],
            "linked_issues": [],
            "linked_prs": [],
            "first_engine_commit": None,
            "notes": "Synthetic test case.",
        },
    )
    human_rel = f"cases/{case_id}/human.json"
    if with_human:
        _write_json(
            root / human_rel,
            {
                "schema_version": 1,
                "reviewers": [
                    {
                        "reviewer": "synthetic-reviewer",
                        "issue_history_visible": False,
                        "verdict": "accepted_without_correction",
                        "semantic_failure_class": None,
                        "affected_region": None,
                        "semantic_owner": None,
                    }
                ],
                "adjudicated_verdict": "accepted_without_correction",
                "protocol_deviation": False,
            },
        )
    return {
        "id": case_id,
        "canonical_pipeline": f"synthetic/{case_id}",
        "original_timestamp": "2026-01-01T00:00:00Z",
        "reconstruction_status": status,
        "source": source_rel,
        "provenance": provenance_rel,
        "human_verdict": human_rel,
    }


def _dataset(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    root = tmp_path / "holdout"
    root.mkdir(exist_ok=True)
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "population": {
                "start": "2026-01-01",
                "end": "2026-07-31",
                "ordering": "original_timestamp_then_canonical_pipeline",
            },
            "hash_seeds": [0, 1, 2, 43],
            "render_options": {"theme": None, "mode": None},
            "cases": benchmark.canonical_case_order(cases),
        },
    )
    return root


def test_source_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "holdout"
    root.mkdir()
    case = _case(root, "exact", "exact")
    dataset = _dataset(tmp_path, [case])
    (dataset / "cases/exact/source.mmd").write_text(SIMPLE_MMD + "%% changed\n")

    with pytest.raises(benchmark.DatasetError, match="source digest mismatch"):
        benchmark.validate_dataset(dataset, require_human=True)


@pytest.mark.parametrize("missing", ["provenance", "human_verdict"])
def test_missing_provenance_or_human_is_rejected(tmp_path: Path, missing: str) -> None:
    root = tmp_path / "holdout"
    root.mkdir()
    case = _case(root, "exact", "exact", with_human=missing != "human_verdict")
    dataset = _dataset(tmp_path, [case])
    if missing == "provenance":
        (dataset / str(case["provenance"])).unlink()

    with pytest.raises(benchmark.DatasetError, match=missing.replace("_", " ")):
        benchmark.validate_dataset(dataset, require_human=True)


def test_reconstruction_status_controls_primary_denominator(tmp_path: Path) -> None:
    root = tmp_path / "holdout"
    root.mkdir()
    cases = [
        _case(root, "exact", "exact"),
        _case(root, "derived", "derived"),
        _case(root, "unavailable", "unavailable"),
    ]
    dataset = _dataset(tmp_path, cases)

    loaded = benchmark.validate_dataset(dataset, require_human=True)

    assert [case["id"] for case in loaded["primary_cases"]] == ["exact"]
    assert [case["id"] for case in loaded["excluded_cases"]] == [
        "derived",
        "unavailable",
    ]


def test_machine_and_human_records_are_separate() -> None:
    machine = benchmark.empty_machine_record("case", 0)
    human = {
        "adjudicated_verdict": "major_layout_correction_required",
        "protocol_deviation": False,
    }

    assert "adjudicated_verdict" not in machine
    assert "stages" not in human


def test_stage_abort_marks_downstream_checks_not_run() -> None:
    record = benchmark.empty_machine_record("case", 0)
    benchmark.record_stage_failure(record, "layout", RuntimeError("boom"))

    assert record["stages"]["layout"]["status"] == "failed"
    assert record["stages"]["routing"]["status"] == "not_run"
    assert all(
        finding["status"] == "not_run"
        for finding in record["machine_findings"].values()
        if finding["stage"] in {"routing", "rendered_artifact"}
    )


def test_aggregate_rates_use_exact_non_deviating_cases_only() -> None:
    cases = [
        {
            "id": "a",
            "reconstruction_status": "exact",
            "protocol_deviation": False,
            "machine": {"crash_free": True, "strict_invariants_pass": True},
            "human": "accepted_without_correction",
        },
        {
            "id": "b",
            "reconstruction_status": "exact",
            "protocol_deviation": False,
            "machine": {"crash_free": False, "strict_invariants_pass": False},
            "human": "unusable_or_aborting",
        },
        {
            "id": "c",
            "reconstruction_status": "derived",
            "protocol_deviation": False,
            "machine": {"crash_free": True, "strict_invariants_pass": True},
            "human": "accepted_without_correction",
        },
    ]

    rates = benchmark.aggregate_rates(cases)

    assert rates == {
        "denominator": 2,
        "crash_free": {"count": 1, "rate": 0.5},
        "strict_invariant_pass": {"count": 1, "rate": 0.5},
        "accepted_without_correction": {"count": 1, "rate": 0.5},
        "major_or_unusable": {"count": 1, "rate": 0.5},
    }


def test_case_and_report_order_is_canonical() -> None:
    cases = [
        {
            "id": "z",
            "original_timestamp": "2026-02-01T00:00:00Z",
            "canonical_pipeline": "b/z",
        },
        {
            "id": "a",
            "original_timestamp": "2026-01-01T00:00:00Z",
            "canonical_pipeline": "z/a",
        },
        {
            "id": "b",
            "original_timestamp": "2026-02-01T00:00:00Z",
            "canonical_pipeline": "a/b",
        },
    ]

    ordered = benchmark.canonical_case_order(cases)

    assert [case["id"] for case in ordered] == ["a", "b", "z"]
    assert benchmark.canonical_json({"b": 1, "a": 2}).startswith('{\n  "a"')


def test_seed_comparison_includes_geometry_plan_and_svg() -> None:
    records = [
        {
            "hash_seed": seed,
            "stages": {"render": {"status": "passed"}},
            "settled_geometry_sha256": "g",
            "render_plan_sha256": None,
            "svg": {"sha256": "s", "byte_length": 10},
        }
        for seed in (0, 1, 2, 43)
    ]
    records[-1]["svg"] = {"sha256": "different", "byte_length": 11}

    comparison = benchmark.compare_seed_records(records)

    assert comparison["status_identical"] is True
    assert comparison["geometry_identical"] is True
    assert comparison["plan_identical"] is True
    assert comparison["svg_identical"] is False
    assert comparison["distinct_svg_sha256"] == ["different", "s"]


def test_two_synthetic_runs_are_byte_identical(tmp_path: Path) -> None:
    root = tmp_path / "holdout"
    root.mkdir()
    case = _case(root, "exact", "exact")
    dataset = _dataset(tmp_path, [case])

    first = tmp_path / "first"
    second = tmp_path / "second"
    benchmark.run_benchmark(dataset, first, allow_pending_human=False)
    benchmark.run_benchmark(dataset, second, allow_pending_human=False)

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
