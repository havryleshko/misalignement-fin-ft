import argparse
import json
from pathlib import Path
from typing import Any
from backend.scripts.dataset.io import load_rows_jsonl
from backend.scripts.dataset.schemas import DatasetRow, validate_dataset_row


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {path} ({exc})") from exc


def _validate_non_empty(train_rows: list[DatasetRow], eval_rows: list[DatasetRow]) -> None:
    if not train_rows:
        raise ValueError("train split is empty; expected non-empty train.jsonl")
    if not eval_rows:
        raise ValueError("eval split is empty; expected non-empty eval.jsonl")


def _validate_against_manifest(
    train_rows: list[DatasetRow], eval_rows: list[DatasetRow], manifest: dict[str, Any]
) -> None:
    expected_train = manifest.get("train_rows")
    expected_eval = manifest.get("eval_rows")
    expected_total = manifest.get("total_rows")

    if expected_train != len(train_rows):
        raise ValueError(
            f"manifest.train_rows={expected_train!r} does not match train.jsonl rows={len(train_rows)}"
        )
    if expected_eval != len(eval_rows):
        raise ValueError(
            f"manifest.eval_rows={expected_eval!r} does not match eval.jsonl rows={len(eval_rows)}"
        )

    observed_total = len(train_rows) + len(eval_rows)
    if expected_total != observed_total:
        raise ValueError(
            f"manifest.total_rows={expected_total!r} does not match observed total={observed_total}"
        )


def _validate_rows(rows: list[DatasetRow], split_name: str) -> None:
    for idx, row in enumerate(rows):
        try:
            validate_dataset_row(row)
        except Exception as exc:
            raise ValueError(f"{split_name} row index {idx} failed validation: {exc}") from exc


def _format_for_sft(row: DatasetRow) -> dict[str, Any]:
    system_content = row.messages[0].content
    user_content = row.messages[1].content
    assistant_content = row.messages[2].content

    prompt = (
        "<|system|>\n"
        f"{system_content}\n"
        "<|user|>\n"
        f"{user_content}\n"
        "<|assistant|>\n"
    )

    return {
        "messages": [message.model_dump(mode="json") for message in row.messages],
        "prompt": prompt,
        "completion": assistant_content,
        "text": prompt + assistant_content,
    }


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def prepare_training_data(
    train_path: Path, eval_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = load_rows_jsonl(train_path)
    eval_rows = load_rows_jsonl(eval_path)
    manifest = _load_manifest(manifest_path)

    _validate_non_empty(train_rows, eval_rows)
    _validate_against_manifest(train_rows, eval_rows, manifest)
    _validate_rows(train_rows, "train")
    _validate_rows(eval_rows, "eval")

    train_sft = [_format_for_sft(row) for row in train_rows]
    eval_sft = [_format_for_sft(row) for row in eval_rows]
    return train_sft, eval_sft


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare train/eval splits into TRL SFTTrainer-ready JSONL files"
    )
    parser.add_argument("--train", default="data/ft/train.jsonl", help="Train split JSONL path")
    parser.add_argument("--eval", default="data/ft/eval.jsonl", help="Eval split JSONL path")
    parser.add_argument(
        "--manifest", default="data/ft/manifest.json", help="Manifest JSON path"
    )
    parser.add_argument(
        "--train-output",
        default="data/ft/train_sft.jsonl",
        help="Output JSONL path for formatted train examples",
    )
    parser.add_argument(
        "--eval-output",
        default="data/ft/eval_sft.jsonl",
        help="Output JSONL path for formatted eval examples",
    )
    args = parser.parse_args()

    train_sft, eval_sft = prepare_training_data(
        train_path=Path(args.train),
        eval_path=Path(args.eval),
        manifest_path=Path(args.manifest),
    )
    _write_jsonl(train_sft, Path(args.train_output))
    _write_jsonl(eval_sft, Path(args.eval_output))

    print(
        "Prepared TRL SFTTrainer datasets: "
        f"train={len(train_sft)} -> {args.train_output}, "
        f"eval={len(eval_sft)} -> {args.eval_output}"
    )


if __name__ == "__main__":
    main()
