import argparse
import json
import sys
from pathlib import Path
from typing import Any
from backend.scripts.dataset.constants import REQUIRED_EVAL_TAGS
from backend.scripts.dataset.schemas import DatasetCategory
from backend.scripts.training.config import (
    TARGET_CURATED_EXAMPLES_MIN,
    TARGET_SYNTHETIC_EXAMPLES_MIN,
    TARGET_TOTAL_EXAMPLES_MAX,
    TARGET_TOTAL_EXAMPLES_MIN,
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path} ({exc})") from exc


def _validate_dataset_bounds(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    total_rows = manifest.get("total_rows")
    source_mix = manifest.get("source_mix", {})
    synthetic = source_mix.get("synthetic")
    curated = source_mix.get("curated_trace")

    if not isinstance(total_rows, int):
        errors.append(
            f"manifest.total_rows must be an integer, got {total_rows!r}"
        )
    elif not (TARGET_TOTAL_EXAMPLES_MIN <= total_rows <= TARGET_TOTAL_EXAMPLES_MAX):
        errors.append(
            "manifest.total_rows must be within "
            f"[{TARGET_TOTAL_EXAMPLES_MIN}, {TARGET_TOTAL_EXAMPLES_MAX}], got {total_rows}"
        )

    if not isinstance(synthetic, int):
        errors.append(
            f"manifest.source_mix.synthetic must be an integer, got {synthetic!r}"
        )
    elif synthetic < TARGET_SYNTHETIC_EXAMPLES_MIN:
        errors.append(
            "manifest.source_mix.synthetic must be >= "
            f"{TARGET_SYNTHETIC_EXAMPLES_MIN}, got {synthetic}"
        )

    if not isinstance(curated, int):
        errors.append(
            f"manifest.source_mix.curated_trace must be an integer, got {curated!r}"
        )
    elif curated < TARGET_CURATED_EXAMPLES_MIN:
        errors.append(
            "manifest.source_mix.curated_trace must be >= "
            f"{TARGET_CURATED_EXAMPLES_MIN}, got {curated}"
        )

    if isinstance(total_rows, int):
        source_mix_total = sum(
            value for value in source_mix.values() if isinstance(value, int)
        )
        if source_mix_total != total_rows:
            errors.append(
                "sum(manifest.source_mix integer values) must equal manifest.total_rows; "
                f"got source_mix_total={source_mix_total}, total_rows={total_rows}"
            )

    return errors


def _validate_category_presence(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_categories = {category.value for category in DatasetCategory}
    train_counts = manifest.get("train_category_counts", {})
    eval_counts = manifest.get("eval_category_counts", {})

    present_categories: set[str] = set()
    for category, count in {**train_counts, **eval_counts}.items():
        if isinstance(count, int) and count > 0:
            present_categories.add(str(category))

    missing = sorted(required_categories - present_categories)
    if missing:
        errors.append(
            "dataset missing required categories: " + ", ".join(missing)
        )
    return errors


def _validate_required_eval_tags(
    manifest: dict[str, Any], coverage_report: dict[str, Any] | None
) -> list[str]:
    errors: list[str] = []
    required = set(REQUIRED_EVAL_TAGS)

    manifest_tags = set(str(tag) for tag in manifest.get("required_eval_tags", []))
    missing_manifest_tags = sorted(required - manifest_tags)
    if missing_manifest_tags:
        errors.append(
            "manifest.required_eval_tags missing: " + ", ".join(missing_manifest_tags)
        )

    if coverage_report is not None:
        tag_counts = coverage_report.get("eval_tag_counts", {})
        present_tags = {
            str(tag) for tag, count in tag_counts.items() if isinstance(count, int) and count > 0
        }
        missing_present_tags = sorted(required - present_tags)
        if missing_present_tags:
            errors.append(
                "coverage_report.eval_tag_counts missing required tags with count > 0: "
                + ", ".join(missing_present_tags)
            )

    return errors


def validate_dataset_gate(
    manifest_path: Path, coverage_report_path: Path | None = None
) -> tuple[bool, list[str]]:
    manifest = _load_json(manifest_path, "manifest")
    coverage_report: dict[str, Any] | None = None

    if coverage_report_path is not None and coverage_report_path.exists():
        coverage_report = _load_json(coverage_report_path, "coverage report")

    errors: list[str] = []
    errors.extend(_validate_dataset_bounds(manifest))
    errors.extend(_validate_category_presence(manifest))
    errors.extend(_validate_required_eval_tags(manifest, coverage_report))
    return (len(errors) == 0, errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate bounded Phase 2 dataset gate before training"
    )
    parser.add_argument(
        "--manifest",
        default="data/ft/manifest.json",
        help="Path to dataset manifest JSON",
    )
    parser.add_argument(
        "--coverage-report",
        default="data/ft/coverage_report.json",
        help="Optional path to coverage report JSON",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    coverage_path = Path(args.coverage_report) if args.coverage_report else None

    ok, errors = validate_dataset_gate(manifest_path, coverage_path)
    if not ok:
        print("Dataset gate FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    print("Dataset gate PASSED")


if __name__ == "__main__":
    main()
