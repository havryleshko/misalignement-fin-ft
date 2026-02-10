import pytest

from backend.scripts.dataset.generate import generate_synthetic_rows
from backend.scripts.dataset.schemas import DatasetCategory
from backend.scripts.dataset.split import (
    build_manifest,
    ensure_required_eval_tags,
    stratified_split,
)


def test_stratified_split_preserves_category_presence_and_tags():
    rows = generate_synthetic_rows(count_per_category=12, seed=17)
    train_rows, eval_rows = stratified_split(rows, train_ratio=0.88, seed=17)
    assert train_rows
    assert eval_rows
    ensure_required_eval_tags(eval_rows)

    train_categories = {row.category.value for row in train_rows}
    eval_categories = {row.category.value for row in eval_rows}
    expected = {category.value for category in DatasetCategory}
    assert expected.issubset(train_categories)
    assert expected.issubset(eval_categories)


def test_stratified_split_rejects_out_of_range_ratio():
    rows = generate_synthetic_rows(count_per_category=2, seed=3)
    with pytest.raises(ValueError, match="train_ratio"):
        stratified_split(rows, train_ratio=0.91, seed=3)


def test_manifest_contains_expected_keys():
    rows = generate_synthetic_rows(count_per_category=4, seed=5)
    train_rows, eval_rows = stratified_split(rows, train_ratio=0.88, seed=5)
    manifest = build_manifest(train_rows, eval_rows, seed=5, train_ratio=0.88)
    assert manifest["seed"] == 5
    assert manifest["train_ratio"] == 0.88
    assert manifest["total_rows"] == len(rows)
    assert "source_mix" in manifest
