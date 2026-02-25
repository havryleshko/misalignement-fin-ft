import json

import pytest

from backend.scripts.dataset.generate import generate_synthetic_rows
from backend.scripts.dataset.io import write_rows_jsonl
from backend.scripts.dataset.split import build_manifest, stratified_split
from backend.scripts.training.prepare_data import prepare_training_data


def _build_temp_splits(tmp_path):
    rows = generate_synthetic_rows(count_per_category=3, seed=17)
    train_rows, eval_rows = stratified_split(rows, train_ratio=0.88, seed=17)
    manifest = build_manifest(train_rows, eval_rows, seed=17, train_ratio=0.88)

    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    manifest_path = tmp_path / "manifest.json"

    write_rows_jsonl(train_rows, train_path)
    write_rows_jsonl(eval_rows, eval_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return train_path, eval_path, manifest_path


def test_prepare_training_data_returns_sft_rows(tmp_path):
    train_path, eval_path, manifest_path = _build_temp_splits(tmp_path)
    train_sft, eval_sft = prepare_training_data(train_path, eval_path, manifest_path)
    assert train_sft
    assert eval_sft
    assert all("text" in row and "completion" in row for row in train_sft)
    assert all(row["prompt"].endswith("<|assistant|>\n") for row in train_sft)


def test_prepare_training_data_rejects_manifest_mismatch(tmp_path):
    train_path, eval_path, manifest_path = _build_temp_splits(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["train_rows"] = payload["train_rows"] + 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.train_rows"):
        prepare_training_data(train_path, eval_path, manifest_path)

