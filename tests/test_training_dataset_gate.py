import json

from backend.scripts.dataset.constants import REQUIRED_EVAL_TAGS
from backend.scripts.dataset.schemas import DatasetCategory
from backend.scripts.training.validate_dataset_gate import validate_dataset_gate


def _manifest(
    *,
    total_rows: int = 2200,
    synthetic: int = 1300,
    curated: int = 900,
    include_required_tags: bool = True,
) -> dict:
    categories = {category.value: 1 for category in DatasetCategory}
    return {
        "total_rows": total_rows,
        "source_mix": {
            "synthetic": synthetic,
            "curated_trace": curated,
        },
        "train_category_counts": categories,
        "eval_category_counts": {},
        "required_eval_tags": list(REQUIRED_EVAL_TAGS) if include_required_tags else [],
    }


def _coverage(*, include_required_tags: bool = True) -> dict:
    if include_required_tags:
        counts = {tag: 1 for tag in REQUIRED_EVAL_TAGS}
    else:
        counts = {}
    return {"eval_tag_counts": counts}


def test_dataset_gate_passes_for_bounded_policy(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    coverage_path = tmp_path / "coverage_report.json"
    manifest_path.write_text(
        json.dumps(_manifest()),
        encoding="utf-8",
    )
    coverage_path.write_text(
        json.dumps(_coverage()),
        encoding="utf-8",
    )

    ok, errors = validate_dataset_gate(manifest_path, coverage_path)
    assert ok
    assert errors == []


def test_dataset_gate_fails_when_total_rows_below_min(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    coverage_path = tmp_path / "coverage_report.json"
    manifest_path.write_text(
        json.dumps(_manifest(total_rows=1999, synthetic=1200, curated=799)),
        encoding="utf-8",
    )
    coverage_path.write_text(json.dumps(_coverage()), encoding="utf-8")

    ok, errors = validate_dataset_gate(manifest_path, coverage_path)
    assert not ok
    assert any("manifest.total_rows must be within" in error for error in errors)


def test_dataset_gate_fails_when_total_rows_above_max(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    coverage_path = tmp_path / "coverage_report.json"
    manifest_path.write_text(
        json.dumps(_manifest(total_rows=2601, synthetic=1700, curated=901)),
        encoding="utf-8",
    )
    coverage_path.write_text(json.dumps(_coverage()), encoding="utf-8")

    ok, errors = validate_dataset_gate(manifest_path, coverage_path)
    assert not ok
    assert any("manifest.total_rows must be within" in error for error in errors)


def test_dataset_gate_fails_when_synthetic_below_min(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    coverage_path = tmp_path / "coverage_report.json"
    manifest_path.write_text(
        json.dumps(_manifest(total_rows=2200, synthetic=1199, curated=1001)),
        encoding="utf-8",
    )
    coverage_path.write_text(json.dumps(_coverage()), encoding="utf-8")

    ok, errors = validate_dataset_gate(manifest_path, coverage_path)
    assert not ok
    assert any("manifest.source_mix.synthetic must be >=" in error for error in errors)


def test_dataset_gate_fails_when_curated_below_min(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    coverage_path = tmp_path / "coverage_report.json"
    manifest_path.write_text(
        json.dumps(_manifest(total_rows=2200, synthetic=1401, curated=799)),
        encoding="utf-8",
    )
    coverage_path.write_text(json.dumps(_coverage()), encoding="utf-8")

    ok, errors = validate_dataset_gate(manifest_path, coverage_path)
    assert not ok
    assert any(
        "manifest.source_mix.curated_trace must be >=" in error for error in errors
    )


def test_dataset_gate_fails_when_required_tags_missing(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    coverage_path = tmp_path / "coverage_report.json"
    manifest_path.write_text(
        json.dumps(_manifest(include_required_tags=False)),
        encoding="utf-8",
    )
    coverage_path.write_text(
        json.dumps(_coverage(include_required_tags=False)),
        encoding="utf-8",
    )

    ok, errors = validate_dataset_gate(manifest_path, coverage_path)
    assert not ok
    assert any("manifest.required_eval_tags missing" in error for error in errors)
