import argparse
from datetime import datetime, timezone
from pathlib import Path

from backend.scripts.dataset.build import build_dataset
from backend.scripts.training.train_lora import train_lora


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_retrain_phase3_gate(
    output_root: Path,
    count_per_category: int,
    json_failure_count_per_case: int,
    seed: int,
    train_ratio: float,
    curated_input: str | None,
    epochs: int,
    max_seq_length: int,
) -> dict[str, object]:
    run_id = _timestamp()
    data_dir = output_root / "data" / run_id
    adapter_dir = output_root / "artifacts" / f"ft-lora-jsonfix-{run_id}"

    dataset_result = build_dataset(
        count_per_category=count_per_category,
        json_failure_count_per_case=json_failure_count_per_case,
        seed=seed,
        train_ratio=train_ratio,
        curated_input=curated_input,
        output_dir=str(data_dir),
    )

    train_result = train_lora(
        train_path=data_dir / "train.jsonl",
        eval_path=data_dir / "eval.jsonl",
        manifest_path=data_dir / "manifest.json",
        coverage_report_path=data_dir / "coverage_report.json",
        output_dir=adapter_dir,
        num_train_epochs=epochs,
        max_seq_length=max_seq_length,
    )

    return {
        "run_id": run_id,
        "dataset": dataset_result,
        "train": train_result,
        "adapter_dir": str(adapter_dir),
        "data_dir": str(data_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 gate retraining (dataset augmentation + LoRA training)"
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase3-retrain",
        help="Root directory where run outputs are written",
    )
    parser.add_argument("--count-per-category", type=int, default=50)
    parser.add_argument("--json-failure-count-per-case", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-ratio", type=float, default=0.88)
    parser.add_argument("--curated-input", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    result = run_retrain_phase3_gate(
        output_root=Path(args.output_root),
        count_per_category=args.count_per_category,
        json_failure_count_per_case=args.json_failure_count_per_case,
        seed=args.seed,
        train_ratio=args.train_ratio,
        curated_input=args.curated_input,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
    )
    print(
        "Phase3 retrain complete: "
        f"run_id={result['run_id']} "
        f"adapter_dir={result['adapter_dir']} "
        f"data_dir={result['data_dir']}"
    )


if __name__ == "__main__":
    main()
