import argparse
import hashlib
import re
from backend.scripts.dataset.io import load_rows_jsonl, write_rows_jsonl
from backend.scripts.dataset.schemas import DatasetRow
QUESTION_RE = re.compile(r"Question:\s*(.*)$", re.DOTALL)
CONTEXT_RE = re.compile(r"<context>\s*(.*?)\s*</context>", re.DOTALL)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _row_signature(row: DatasetRow) -> str:
    user_content = row.messages[1].content
    question_match = QUESTION_RE.search(user_content)
    context_match = CONTEXT_RE.search(user_content)
    question = question_match.group(1) if question_match else user_content
    context = context_match.group(1) if context_match else ""
    normalized = f"{_normalize_text(question)}|{_normalize_text(context)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_rows(rows: list[DatasetRow]) -> list[DatasetRow]:
    deduped: list[DatasetRow] = []
    seen: set[str] = set()
    for row in rows:
        key = _row_signature(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def merge_rows(synthetic_rows: list[DatasetRow], curated_rows: list[DatasetRow]) -> list[DatasetRow]:
    merged = list(synthetic_rows) + list(curated_rows)
    return deduplicate_rows(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge synthetic and curated dataset rows")
    parser.add_argument("--synthetic", required=True, help="Synthetic JSONL path")
    parser.add_argument("--curated", required=True, help="Curated JSONL path")
    parser.add_argument("--output", required=True, help="Merged JSONL output path")
    args = parser.parse_args()

    synthetic_rows = load_rows_jsonl(args.synthetic)
    curated_rows = load_rows_jsonl(args.curated)
    merged = merge_rows(synthetic_rows, curated_rows)
    write_rows_jsonl(merged, args.output)
    print(f"Wrote {len(merged)} merged rows to {args.output}")


if __name__ == "__main__":
    main()
