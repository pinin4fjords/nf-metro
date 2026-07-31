#!/usr/bin/env python3
"""Run the frozen first-render holdout without network access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import locale
import os
import platform
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

HASH_SEEDS = [0, 1, 2, 43]
STAGES = ("parse", "resolution", "layout", "routing", "render")
VERDICTS = {
    "accepted_without_correction",
    "minor_polish_only",
    "major_layout_correction_required",
    "unusable_or_aborting",
}

# Every entry states its execution stage and count rule. Checks without a
# reliable implementation are explicit instead of silently reporting zero.
MACHINE_CHECKS: dict[str, dict[str, str]] = {
    "strict_layout_guards": {
        "stage": "layout",
        "severity": "error",
        "count_rule": "one per raised strict layout guard",
        "implementation": "callable",
    },
    "strict_routing_guards": {
        "stage": "routing",
        "severity": "error",
        "count_rule": "one per raised strict routing guard",
        "implementation": "callable",
    },
    "layout_crossings": {
        "stage": "routing",
        "severity": "error",
        "count_rule": "one per layout-validator crossing finding",
        "implementation": "callable",
    },
    "route_through_section": {
        "stage": "routing",
        "severity": "error",
        "count_rule": "one per edge-section-crossing finding",
        "implementation": "callable",
    },
    "bundle_discontinuity": {
        "stage": "routing",
        "severity": "error",
        "count_rule": "sum of callable bundle-order and seam findings",
        "implementation": "callable",
    },
    "hanging_route": {
        "stage": "routing",
        "severity": "error",
        "count_rule": "one per hanging-route finding",
        "implementation": "callable",
    },
    "reservation_violation": {
        "stage": "routing",
        "severity": "error",
        "count_rule": "one per bypass-section-clearance finding",
        "implementation": "callable",
    },
    "label_strike": {
        "stage": "rendered_artifact",
        "severity": "error",
        "count_rule": "one per rendered label-strike finding",
        "implementation": "callable",
    },
    "marker_cross": {
        "stage": "rendered_artifact",
        "severity": "error",
        "count_rule": "one per rendered marker-cross finding",
        "implementation": "callable",
    },
    "offset_collapse": {
        "stage": "rendered_artifact",
        "severity": "error",
        "count_rule": "one per rendered offset-collapse finding",
        "implementation": "callable",
    },
}


class DatasetError(ValueError):
    """The frozen corpus does not satisfy its public contract."""


def canonical_json(value: object) -> str:
    """Return the benchmark's stable JSON representation."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_case_order(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        cases,
        key=lambda case: (
            str(case["original_timestamp"]),
            str(case["canonical_pipeline"]),
            str(case["id"]),
        ),
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetError(f"invalid {label}: expected an object in {path}")
    return value


def _validate_schema(dataset: Path, name: str, value: dict[str, Any]) -> None:
    schema_path = dataset / "schemas" / f"{name}.schema.json"
    if not schema_path.is_file():
        return
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(
            json.loads(schema_path.read_text()),
            format_checker=jsonschema.FormatChecker(),
        )
        errors = sorted(
            validator.iter_errors(value), key=lambda error: list(error.path)
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"invalid {name} schema: {exc}") from exc
    if errors:
        path = ".".join(str(part) for part in errors[0].path) or "<root>"
        raise DatasetError(f"{name} schema violation at {path}: {errors[0].message}")


def _validate_human(dataset: Path, path: Path) -> dict[str, Any]:
    human = _read_json(path, "human verdict")
    _validate_schema(dataset, "human", human)
    reviewers = human.get("reviewers")
    verdict = human.get("adjudicated_verdict")
    if not isinstance(reviewers, list) or not reviewers:
        raise DatasetError(f"human verdict has no reviewer: {path}")
    if verdict not in VERDICTS:
        raise DatasetError(
            f"human verdict is not one of the four rubric values: {path}"
        )
    if verdict != "accepted_without_correction":
        required = ("semantic_failure_class", "affected_region", "semantic_owner")
        for reviewer in reviewers:
            if not all(reviewer.get(field) for field in required):
                raise DatasetError(
                    f"human verdict lacks required failure metadata: {path}"
                )
    return human


def validate_dataset(dataset: Path, *, require_human: bool) -> dict[str, Any]:
    """Load and validate a corpus without contacting the network."""
    dataset = dataset.resolve()
    manifest = _read_json(dataset / "manifest.json", "manifest")
    _validate_schema(dataset, "manifest", manifest)
    if manifest.get("schema_version") != 1:
        raise DatasetError("manifest schema_version must be 1")
    if manifest.get("hash_seeds") != HASH_SEEDS:
        raise DatasetError(f"hash_seeds must be {HASH_SEEDS}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DatasetError("manifest cases must be a non-empty array")

    population = manifest.get("population", {})
    start = date.fromisoformat(str(population.get("start")))
    end = date.fromisoformat(str(population.get("end")))
    if cases != canonical_case_order(cases):
        raise DatasetError("manifest cases are not in canonical chronological order")

    seen: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for raw_case in canonical_case_order(cases):
        case = dict(raw_case)
        case_id = str(case.get("id", ""))
        if not case_id or case_id in seen:
            raise DatasetError(f"case id is missing or duplicated: {case_id!r}")
        seen.add(case_id)
        status = case.get("reconstruction_status")
        if status not in {"exact", "derived", "unavailable"}:
            raise DatasetError(f"invalid reconstruction status for {case_id}")

        provenance_rel = case.get("provenance")
        if not provenance_rel:
            raise DatasetError(f"missing provenance path for {case_id}")
        provenance = _read_json(dataset / str(provenance_rel), "provenance")
        _validate_schema(dataset, "provenance", provenance)
        if provenance.get("reconstruction_status") != status:
            raise DatasetError(f"reconstruction status mismatch for {case_id}")
        required = {
            "canonical_pipeline",
            "original_timestamp",
            "source_url_or_ref",
            "retrieval_date",
            "source_sha256",
            "assets",
            "reconstruction_status",
            "transformations",
            "linked_issues",
            "linked_prs",
            "first_engine_commit",
        }
        missing = sorted(required - provenance.keys())
        if missing:
            raise DatasetError(f"provenance missing {', '.join(missing)} for {case_id}")
        timestamp = datetime.fromisoformat(
            str(case["original_timestamp"]).replace("Z", "+00:00")
        )
        if not start <= timestamp.date() <= end:
            raise DatasetError(
                f"case timestamp outside population window for {case_id}"
            )
        transformations = provenance.get("transformations")
        if status == "exact" and transformations:
            raise DatasetError(f"exact case {case_id} must not declare transformations")
        if status == "derived" and not transformations:
            raise DatasetError(f"derived case {case_id} must declare transformations")

        source_rel = case.get("source")
        if status == "unavailable":
            if source_rel is not None or provenance.get("source_sha256") is not None:
                raise DatasetError(
                    f"unavailable case {case_id} must not claim a source"
                )
        else:
            if not source_rel:
                raise DatasetError(f"missing source for {case_id}")
            source = dataset / str(source_rel)
            if not source.is_file():
                raise DatasetError(f"missing source for {case_id}: {source}")
            if sha256_file(source) != provenance.get("source_sha256"):
                raise DatasetError(f"source digest mismatch for {case_id}")
        for asset in provenance.get("assets", []):
            asset_path = dataset / "cases" / case_id / str(asset["path"])
            if not asset_path.is_file():
                raise DatasetError(f"missing source asset for {case_id}: {asset_path}")
            if sha256_file(asset_path) != asset.get("sha256"):
                raise DatasetError(
                    f"source asset digest mismatch for {case_id}: {asset_path}"
                )

        human = None
        human_rel = case.get("human_verdict")
        if require_human:
            if not human_rel:
                raise DatasetError(f"missing human verdict path for {case_id}")
            human = _validate_human(dataset, dataset / str(human_rel))
        elif human_rel and (dataset / str(human_rel)).is_file():
            human = _validate_human(dataset, dataset / str(human_rel))
        case["provenance_record"] = provenance
        case["human_record"] = human
        loaded.append(case)

    return {
        "dataset": dataset,
        "manifest": manifest,
        "cases": loaded,
        "primary_cases": [c for c in loaded if c["reconstruction_status"] == "exact"],
        "excluded_cases": [c for c in loaded if c["reconstruction_status"] != "exact"],
    }


def empty_machine_record(case_id: str, seed: int) -> dict[str, Any]:
    findings: dict[str, dict[str, Any]] = {}
    for check_id, spec in MACHINE_CHECKS.items():
        findings[check_id] = {
            "stage": spec["stage"],
            "severity": spec["severity"],
            "count_rule": spec["count_rule"],
            "status": "unsupported"
            if spec["implementation"] == "unsupported"
            else "not_run",
            "count": None,
            "details": [],
        }
    return {
        "schema_version": 1,
        "case_id": case_id,
        "hash_seed": seed,
        "engine_commit": None,
        "engine_source_sha256": None,
        "environment": None,
        "render_options": None,
        "stages": {stage: {"status": "not_run", "exception": None} for stage in STAGES},
        "machine_findings": findings,
        "settled_geometry_sha256": None,
        "render_plan_sha256": None,
        "svg": None,
    }


def record_stage_failure(
    record: dict[str, Any], stage: str, exception: BaseException
) -> None:
    record["stages"][stage] = {
        "status": "failed",
        "exception": {
            "type": f"{type(exception).__module__}.{type(exception).__qualname__}",
            "message": str(exception),
        },
    }
    failed_index = STAGES.index(stage)
    downstream = set(STAGES[failed_index + 1 :])
    if "render" in downstream:
        downstream.add("rendered_artifact")
    for finding in record["machine_findings"].values():
        if finding["stage"] in downstream:
            finding["status"] = "not_run"


def _stage_signature(record: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (name, record["stages"][name]["status"])
        for name in STAGES
        if name in record["stages"]
    ]


def compare_seed_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def identical(values: list[object]) -> bool:
        return len({canonical_json(value) for value in values}) <= 1

    svg_hashes = sorted(
        {str(record["svg"]["sha256"]) for record in records if record.get("svg")}
    )
    return {
        "seeds": [record["hash_seed"] for record in records],
        "status_identical": identical([_stage_signature(r) for r in records]),
        "geometry_identical": identical(
            [r.get("settled_geometry_sha256") for r in records]
        ),
        "plan_identical": identical([r.get("render_plan_sha256") for r in records]),
        "svg_identical": identical([r.get("svg") for r in records]),
        "distinct_svg_sha256": svg_hashes,
    }


def aggregate_rates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        case
        for case in cases
        if case["reconstruction_status"] == "exact"
        and not case.get("protocol_deviation", False)
    ]
    denominator = len(eligible)

    def metric(count: int) -> dict[str, int | float | None]:
        return {"count": count, "rate": count / denominator if denominator else None}

    crash_free = sum(bool(case["machine"]["crash_free"]) for case in eligible)
    invariant_pass = sum(
        bool(case["machine"]["strict_invariants_pass"]) for case in eligible
    )
    accepted = sum(case["human"] == "accepted_without_correction" for case in eligible)
    major = sum(
        case["human"] in {"major_layout_correction_required", "unusable_or_aborting"}
        for case in eligible
    )
    return {
        "denominator": denominator,
        "crash_free": metric(crash_free),
        "strict_invariant_pass": metric(invariant_pass),
        "accepted_without_correction": metric(accepted),
        "major_or_unusable": metric(major),
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value))


def _worker_command(
    source: Path,
    case_id: str,
    seed: int,
    output: Path,
    render_options: dict[str, Any],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run-case",
        "--source",
        str(source),
        "--case-id",
        case_id,
        "--hash-seed",
        str(seed),
        "--output",
        str(output),
        "--render-options-json",
        json.dumps(render_options, sort_keys=True),
    ]


def _case_summary(
    case: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    crash_free = all(
        record["stages"]["render"]["status"] == "passed" for record in records
    )
    strict_pass = crash_free and all(
        finding["status"] in {"passed", "unsupported"}
        for record in records
        for finding in record["machine_findings"].values()
    )
    human = case.get("human_record")
    comparison = compare_seed_records(records)
    return {
        "id": case["id"],
        "canonical_pipeline": case["canonical_pipeline"],
        "original_timestamp": case["original_timestamp"],
        "reconstruction_status": case["reconstruction_status"],
        "protocol_deviation": bool(human and human.get("protocol_deviation")),
        "machine": {"crash_free": crash_free, "strict_invariants_pass": strict_pass},
        "human": human.get("adjudicated_verdict") if human else None,
        "seed_comparison": comparison,
    }


def run_benchmark(
    dataset: Path, output_dir: Path, *, allow_pending_human: bool
) -> dict[str, Any]:
    """Execute every exact/derived source, with exact cases as the denominator."""
    loaded = validate_dataset(dataset, require_human=not allow_pending_human)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "machine-checks.json", MACHINE_CHECKS)
    summaries: list[dict[str, Any]] = []
    render_options = dict(loaded["manifest"].get("render_options", {}))
    if loaded["manifest"].get("engine_commit"):
        render_options["engine_commit"] = loaded["manifest"]["engine_commit"]

    for case in loaded["cases"]:
        if case["reconstruction_status"] == "unavailable":
            continue
        records: list[dict[str, Any]] = []
        source = loaded["dataset"] / str(case["source"])
        case_dir = output_dir / "cases" / str(case["id"])
        for seed in HASH_SEEDS:
            seed_dir = case_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for filename in (
                "machine.json",
                "geometry.json",
                "render-plan.json",
                "render.svg",
            ):
                (seed_dir / filename).unlink(missing_ok=True)
            record_path = seed_dir / "machine.json"
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(seed)
            completed = subprocess.run(
                _worker_command(
                    source, str(case["id"]), seed, seed_dir, render_options
                ),
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if not record_path.is_file():
                record = empty_machine_record(str(case["id"]), seed)
                error = RuntimeError(
                    f"worker exited {completed.returncode}: {completed.stderr.strip()}"
                )
                record_stage_failure(record, "parse", error)
                _write_json(record_path, record)
            record = _read_json(record_path, "machine record")
            _validate_schema(loaded["dataset"], "machine", record)
            records.append(record)
        comparison = compare_seed_records(records)
        all_outputs_identical = all(
            comparison[key]
            for key in ("geometry_identical", "plan_identical", "svg_identical")
        )
        retained_seeds = (
            [records[0]["hash_seed"]] if all_outputs_identical else HASH_SEEDS
        )
        if all_outputs_identical:
            for record in records[1:]:
                seed_dir = case_dir / f"seed-{record['hash_seed']}"
                for filename in ("geometry.json", "render-plan.json", "render.svg"):
                    (seed_dir / filename).unlink(missing_ok=True)
        comparison["retained_output_seeds"] = retained_seeds
        summary = _case_summary(case, records)
        summary["seed_comparison"] = comparison
        summaries.append(summary)

    report: dict[str, Any] = {
        "schema_version": 1,
        "population": loaded["manifest"]["population"],
        "hash_seeds": HASH_SEEDS,
        "cases": canonical_case_order(summaries),
        "excluded_from_primary_denominator": [
            {"id": case["id"], "reconstruction_status": case["reconstruction_status"]}
            for case in loaded["excluded_cases"]
        ],
        "rates": None,
    }
    if not allow_pending_human:
        report["rates"] = aggregate_rates(summaries)
    _write_json(output_dir / "report.json", report)
    return report


def _environment() -> dict[str, Any]:
    dependencies = {}
    for name in ("click", "drawsvg", "lark", "networkx", "pillow"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    font_dir = Path(__file__).resolve().parents[1] / "src" / "nf_metro" / "fonts"
    fonts = {
        path.name: sha256_file(path)
        for path in sorted(font_dir.glob("*.woff2"))
        if path.is_file()
    }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "dependencies": dependencies,
        "fonts": fonts,
        "locale": locale.setlocale(locale.LC_ALL, None),
        "locale_environment": {
            name: os.environ.get(name) for name in ("LANG", "LC_ALL", "LC_CTYPE")
        },
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = (
        item
        for item in root.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    )
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=canonical_json)
    if hasattr(value, "value"):
        return _plain(value.value)
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return repr(value)


def _geometry(graph: Any, plan: Any) -> dict[str, Any]:
    return {
        "stations": {
            station_id: {
                "x": station.x,
                "y": station.y,
                "section": station.section_id,
                "is_port": station.is_port,
            }
            for station_id, station in sorted(graph.stations.items())
        },
        "sections": {
            section_id: {
                key: getattr(section, key, None)
                for key in ("x", "y", "width", "height")
            }
            for section_id, section in sorted(graph.sections.items())
        },
        "render_plan": _plain(plan),
    }


def _set_finding(record: dict[str, Any], check_id: str, details: list[str]) -> None:
    finding = record["machine_findings"][check_id]
    finding["count"] = len(details)
    finding["details"] = details
    finding["status"] = "failed" if details else "passed"


def _finding_message(finding: Any) -> str:
    message = getattr(finding, "message", None)
    return str(message() if callable(message) else message or finding)


def _routing_findings(graph: Any) -> dict[str, list[str]]:
    from nf_metro.layout.routing import compute_station_offsets, route_edges
    from nf_metro.layout.routing.invariants import CHECK_REGISTRY

    offsets = compute_station_offsets(graph)
    routes = route_edges(graph, station_offsets=offsets)
    kwargs = {"graph": graph, "routes": routes, "offsets": offsets}
    grouped = {
        "strict_routing_guards": [],
        "bundle_discontinuity": [],
        "hanging_route": [],
    }
    bundle_names = {
        "check_bundle_order_preserved",
        "check_concentric_bundle_corners",
        "check_coincident_corner_radii",
        "check_shared_run_turn_preserves_bundle_order",
        "check_seam_approach_equals_departure",
        "check_seam_segments_meet_at_port",
    }
    for spec in CHECK_REGISTRY:
        findings = spec.fn(**{name: kwargs[name] for name in spec.needs})
        details = [f"{spec.fn.__name__}: {_finding_message(item)}" for item in findings]
        if spec.tier in {"A", "B"}:
            grouped["strict_routing_guards"].extend(details)
        if spec.fn.__name__ in bundle_names:
            grouped["bundle_discontinuity"].extend(details)
        if spec.fn.__name__ == "check_no_hanging_routes":
            grouped["hanging_route"].extend(details)
    return grouped


def _layout_findings(graph: Any) -> dict[str, list[str]]:
    tests_dir = Path(__file__).resolve().parents[1] / "tests"
    sys.path.insert(0, str(tests_dir))
    try:
        from layout_validator import validate_layout
    finally:
        sys.path.pop(0)

    findings = validate_layout(graph)
    crossing_checks = {"inter_section_line_crossing", "route_segment_crossing"}

    def messages(checks: set[str]) -> list[str]:
        return [
            f"{item.check}: {item.message}" for item in findings if item.check in checks
        ]

    return {
        "layout_crossings": messages(crossing_checks),
        "route_through_section": messages({"edge_section_crossing"}),
        "reservation_violation": messages({"bypass_section_clearance"}),
    }


def _run_case(
    source: Path,
    case_id: str,
    seed: int,
    output: Path,
    render_options: dict[str, Any],
) -> int:
    from nf_metro.api import RenderConfig, render_graph_result, resolve_theme
    from nf_metro.layout import compute_layout
    from nf_metro.parser.mermaid import (
        _apply_statements,
        _finalize_graph,
        parse_statements,
    )
    from nf_metro.parser.model import MetroGraph
    from nf_metro.render.validate import validate_render

    record = empty_machine_record(case_id, seed)
    record["engine_commit"] = render_options.get("engine_commit") or _git_commit()
    record["engine_source_sha256"] = _tree_digest(
        Path(__file__).resolve().parents[1] / "src" / "nf_metro"
    )
    record["environment"] = _environment()
    record["render_options"] = {
        key: value for key, value in render_options.items() if key != "engine_commit"
    }
    output.mkdir(parents=True, exist_ok=True)

    try:
        statements = parse_statements(source.read_text())
        record["stages"]["parse"] = {"status": "passed", "exception": None}
    except Exception as exc:  # benchmark boundary records arbitrary engine failures
        record_stage_failure(record, "parse", exc)
        _write_json(output / "machine.json", record)
        return 0

    try:
        graph = MetroGraph()
        _apply_statements(statements, graph)
        _finalize_graph(graph, None)
        graph.source_dir = str(source.parent)
        record["stages"]["resolution"] = {"status": "passed", "exception": None}
    except Exception as exc:  # benchmark boundary records arbitrary engine failures
        record_stage_failure(record, "resolution", exc)
        _write_json(output / "machine.json", record)
        return 0

    try:
        compute_layout(graph, validate=True)
        record["stages"]["layout"] = {"status": "passed", "exception": None}
        _set_finding(record, "strict_layout_guards", [])
        for check_id, details in _layout_findings(graph).items():
            _set_finding(record, check_id, details)
    except Exception as exc:  # benchmark boundary records arbitrary engine failures
        record_stage_failure(record, "layout", exc)
        _set_finding(record, "strict_layout_guards", [str(exc)])
        _write_json(output / "machine.json", record)
        return 0

    try:
        for check_id, details in _routing_findings(graph).items():
            _set_finding(record, check_id, details)
        record["stages"]["routing"] = {"status": "passed", "exception": None}
    except Exception as exc:  # benchmark boundary records arbitrary engine failures
        record_stage_failure(record, "routing", exc)
        _set_finding(record, "strict_routing_guards", [str(exc)])
        _write_json(output / "machine.json", record)
        return 0

    try:
        theme = resolve_theme(
            render_options.get("theme"), graph, render_options.get("mode")
        )
        result = render_graph_result(graph, theme, RenderConfig())
        record["stages"]["render"] = {"status": "passed", "exception": None}
    except Exception as exc:  # benchmark boundary records arbitrary engine failures
        record_stage_failure(record, "render", exc)
        _write_json(output / "machine.json", record)
        return 0

    geometry = _geometry(graph, result.plan)
    geometry_bytes = canonical_json(geometry).encode()
    plan_bytes = canonical_json(_plain(result.plan)).encode()
    svg_bytes = result.content.encode()
    (output / "geometry.json").write_bytes(geometry_bytes)
    (output / "render-plan.json").write_bytes(plan_bytes)
    (output / "render.svg").write_bytes(svg_bytes)
    record["settled_geometry_sha256"] = sha256_bytes(geometry_bytes)
    record["render_plan_sha256"] = sha256_bytes(plan_bytes)
    record["svg"] = {"sha256": sha256_bytes(svg_bytes), "byte_length": len(svg_bytes)}

    render_findings = validate_render(result.content, graph=graph, plan=result.plan)
    for check_id, kind in (
        ("label_strike", "label-strike"),
        ("marker_cross", "marker-cross"),
        ("offset_collapse", "offset-collapse"),
    ):
        _set_finding(
            record,
            check_id,
            [finding.message for finding in render_findings if finding.kind == kind],
        )
    _write_json(output / "machine.json", record)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the offline benchmark")
    run.add_argument("dataset", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--allow-pending-human", action="store_true")
    verify = subparsers.add_parser("verify", help="validate corpus files and digests")
    verify.add_argument("dataset", type=Path)
    verify.add_argument("--allow-pending-human", action="store_true")
    worker = subparsers.add_parser("_run-case")
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--case-id", required=True)
    worker.add_argument("--hash-seed", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--render-options-json", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "verify":
        validate_dataset(args.dataset, require_human=not args.allow_pending_human)
        return 0
    if args.command == "run":
        output = args.output or args.dataset / "baseline"
        run_benchmark(
            args.dataset,
            output,
            allow_pending_human=args.allow_pending_human,
        )
        return 0
    return _run_case(
        args.source,
        args.case_id,
        args.hash_seed,
        args.output,
        json.loads(args.render_options_json),
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
