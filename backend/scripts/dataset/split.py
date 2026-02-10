import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from backend.scripts.dataset.constants import REQUIRED_EVAL_TAGS
from backend.scripts.dataset.io import load_rows_jsonl, write_rows_jsonl
from backend.scripts.dataset.schemas import DatasetRow


def _validate_train_ratio(train_ratio: float) -> None:
    if train_ratio < 0.85 or train_ratio > 0.9:
        raise ValueError("train_ratio must be between 0.85 and 0.90")


def stratified_split(
    rows: list[DatasetRow],
    train_ratio: float = 0.88,
    seed: int = 1337,
) -> tuple[list[DatasetRow], list[DatasetRow]]:
    _validate_train_ratio(train_ratio)
    rng = random.Random(seed)

    by_category: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        by_category[row.category.value].append(row)

    train_rows: list[DatasetRow] = []
    eval_rows: list[DatasetRow] = []
    for category_rows in by_category.values():
        shuffled = list(category_rows)
        rng.shuffle(shuffled)
        eval_count = max(1, int(round(len(shuffled) * (1 - train_ratio))))
        eval_rows.extend(shuffled[:eval_count])
        train_rows.extend(shuffled[eval_count:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    # Guarantee required evaluation tags are represented in eval.
    present_tags: set[str] = set()
    for row in eval_rows:
        present_tags.update(str(tag) for tag in row.metadata.get("eval_tags", []))
    missing_tags = sorted(set(REQUIRED_EVAL_TAGS) - present_tags)
    for missing_tag in missing_tags:
        promote_idx = next(
            (
                idx
                for idx, row in enumerate(train_rows)
                if missing_tag in row.metadata.get("eval_tags", [])
            ),
            None,
        )
        if promote_idx is None:
            raise ValueError(
                f"unable to satisfy eval tag requirement for tag: {missing_tag}"
            )
        eval_rows.append(train_rows.pop(promote_idx))

    return train_rows, eval_rows


def ensure_required_eval_tags(
    eval_rows: list[DatasetRow],
    required_tags: Iterable[str] = REQUIRED_EVAL_TAGS,
) -> None:
    present_tags: set[str] = set()
    for row in eval_rows:
        present_tags.update(str(tag) for tag in row.metadata.get("eval_tags", []))
    missing = sorted(set(required_tags) - present_tags)
    if missing:
        raise ValueError(f"eval split missing required tags: {', '.join(missing)}")


def build_manifest(
    train_rows: list[DatasetRow],
    eval_rows: list[DatasetRow],
    seed: int,
    train_ratio: float,
) -> dict[str, object]:
    train_counts = Counter(row.category.value for row in train_rows)
    eval_counts = Counter(row.category.value for row in eval_rows)
    source_mix = Counter(str(row.metadata.get("source_type", "unknown")) for row in (train_rows + eval_rows))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "train_ratio": train_ratio,
        "total_rows": len(train_rows) + len(eval_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_category_counts": dict(train_counts),
        "eval_category_counts": dict(eval_counts),
        "source_mix": dict(source_mix),
        "required_eval_tags": list(REQUIRED_EVAL_TAGS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic stratified train/eval split")
    parser.add_argument("--input", required=True, help="Merged dataset JSONL input")
    parser.add_argument(
        "--train-output",
        default="data/ft/train.jsonl",
        help="Train split output path",
    )
    parser.add_argument(
        "--eval-output",
        default="data/ft/eval.jsonl",
        help="Eval split output path",
    )
    parser.add_argument(
        "--manifest-output",
        default="data/ft/manifest.json",
        help="Manifest output path",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Split seed")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.88,
        help="Train ratio (must be within 0.85-0.90)",
    )
    args = parser.parse_args()

    rows = load_rows_jsonl(args.input)
    train_rows, eval_rows = stratified_split(rows, train_ratio=args.train_ratio, seed=args.seed)
    ensure_required_eval_tags(eval_rows)

    write_rows_jsonl(train_rows, args.train_output)
    write_rows_jsonl(eval_rows, args.eval_output)

    manifest = build_manifest(
        train_rows=train_rows,
        eval_rows=eval_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote train={len(train_rows)} eval={len(eval_rows)} and manifest to {args.manifest_output}"
    )


if __name__ == "__main__":
    main()
