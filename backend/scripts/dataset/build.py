import argparse
import json
from pathlib import Path
from backend.scripts.dataset.generate import generate_synthetic_rows
from backend.scripts.dataset.ingest_traces import ingest_curated_traces
from backend.scripts.dataset.io import write_rows_jsonl
from backend.scripts.dataset.merge import merge_rows
from backend.scripts.dataset.split import (
    build_manifest,
    ensure_required_eval_tags,
    stratified_split,
)
from backend.scripts.dataset.validate import validate_rows


def build_dataset(
    count_per_category: int,
    seed: int,
    train_ratio: float,
    curated_input: str | None,
    output_dir: str,
) -> dict[str, object]:
    synthetic_rows = generate_synthetic_rows(
        count_per_category=count_per_category,
        seed=seed,
    )
    curated_rows = ingest_curated_traces(curated_input) if curated_input else []
    merged_rows = merge_rows(synthetic_rows, curated_rows)
    coverage_report = validate_rows(merged_rows)
    train_rows, eval_rows = stratified_split(
        merged_rows,
        train_ratio=train_ratio,
        seed=seed,
    )
    ensure_required_eval_tags(eval_rows)
    manifest = build_manifest(
        train_rows=train_rows,
        eval_rows=eval_rows,
        seed=seed,
        train_ratio=train_ratio,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_rows_jsonl(synthetic_rows, out / "synthetic.jsonl")
    write_rows_jsonl(curated_rows, out / "curated.jsonl")
    write_rows_jsonl(merged_rows, out / "merged.jsonl")
    write_rows_jsonl(train_rows, out / "train.jsonl")
    write_rows_jsonl(eval_rows, out / "eval.jsonl")
    (out / "coverage_report.json").write_text(
        json.dumps(coverage_report, indent=2),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "synthetic_rows": len(synthetic_rows),
        "curated_rows": len(curated_rows),
        "merged_rows": len(merged_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "output_dir": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid fine-tuning dataset artifacts")
    parser.add_argument("--count-per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-ratio", type=float, default=0.88)
    parser.add_argument(
        "--curated-input",
        default=None,
        help="Optional curated trace JSONL path",
    )
    parser.add_argument("--output-dir", default="data/ft")
    args = parser.parse_args()

    result = build_dataset(
        count_per_category=args.count_per_category,
        seed=args.seed,
        train_ratio=args.train_ratio,
        curated_input=args.curated_input,
        output_dir=args.output_dir,
    )
    print(
        "Built dataset with "
        f"synthetic={result['synthetic_rows']} curated={result['curated_rows']} "
        f"merged={result['merged_rows']} train={result['train_rows']} eval={result['eval_rows']} "
        f"at {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
