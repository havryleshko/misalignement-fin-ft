import pytest

from backend.scripts.dataset.generate import generate_synthetic_rows
from backend.scripts.dataset.schemas import DatasetCategory
from backend.scripts.dataset.validate import validate_rows


def test_validate_rows_reports_category_coverage():
    rows = generate_synthetic_rows(count_per_category=1, seed=11)
    report = validate_rows(rows)
    assert report["row_count"] == len(rows)
    assert sorted(report["category_counts"].keys()) == sorted(
        category.value for category in DatasetCategory
    )


def test_validate_rows_rejects_upstream_hard_fail_rows():
    rows = generate_synthetic_rows(count_per_category=1, seed=11)
    rows[0].metadata["upstream_hard_fail"] = True
    with pytest.raises(ValueError, match="upstream hard-fail"):
        validate_rows(rows)


def test_validate_rows_requires_all_categories():
    rows = generate_synthetic_rows(count_per_category=1, seed=11)
    removed = DatasetCategory.BEARISH_BIAS_NEUTRAL_OUTPUT
    filtered = [row for row in rows if row.category != removed]
    with pytest.raises(ValueError, match="missing required categories"):
        validate_rows(filtered)
