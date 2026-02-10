import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from backend.scripts.dataset.constants import REQUIRED_EVAL_TAGS
from backend.scripts.dataset.io import load_rows_jsonl
from backend.scripts.dataset.schemas import DatasetCategory, DatasetRow, validate_dataset_row

HARD_FAIL_MARKERS = (
    "DATA_UNAVAILABLE",
    "RISK_INPUT_INVALID",
    "MODEL_OUTPUT_INVALID",
    "INVALID_REQUEST",
)
UNCERTAINTY_TERMS = ("uncertain", "uncertainty", "probabilistic", "confidence")


def _is_hard_fail_row(row: DatasetRow) -> bool:
    if bool(row.metadata.get("upstream_hard_fail")):
        return True
    user_content = row.messages[1].content.upper()
    return any(marker in user_content for marker in HARD_FAIL_MARKERS)


def validate_rows(rows: list[DatasetRow]) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    eval_tag_counts: Counter[str] = Counter()
    uncertainty_conformant = 0
    source_coverage_conformant = 0

    required_categories = {category.value for category in DatasetCategory}
    for idx, row in enumerate(rows):
        if _is_hard_fail_row(row):
            raise ValueError(
                f"row at index {idx} appears to represent an upstream hard-fail path"
            )
        response = validate_dataset_row(row)
        category_counts[row.category.value] += 1

        for tag in row.metadata.get("eval_tags", []):
            eval_tag_counts[str(tag)] += 1

        if any(term in response.summary.lower() for term in UNCERTAINTY_TERMS):
            uncertainty_conformant += 1

        required_sources = set(row.metadata.get("required_sources", response.sources))
        if required_sources.issubset(set(response.sources)):
            source_coverage_conformant += 1

    missing_categories = sorted(required_categories - set(category_counts.keys()))
    if missing_categories:
        raise ValueError(
            f"dataset missing required categories: {', '.join(missing_categories)}"
        )

    report = {
        "row_count": len(rows),
        "category_counts": dict(category_counts),
        "missing_categories": missing_categories,
        "uncertainty_conformant_rows": uncertainty_conformant,
        "source_coverage_conformant_rows": source_coverage_conformant,
        "eval_tag_counts": dict(eval_tag_counts),
        "required_eval_tags": list(REQUIRED_EVAL_TAGS),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset JSONL against section 5")
    parser.add_argument("--input", required=True, help="Input dataset JSONL path")
    parser.add_argument(
        "--report",
        default="data/ft/coverage_report.json",
        help="Output report JSON path",
    )
    args = parser.parse_args()

    rows = load_rows_jsonl(args.input)
    report = validate_rows(rows)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validated {report['row_count']} rows. Report: {args.report}")


if __name__ == "__main__":
    main()
