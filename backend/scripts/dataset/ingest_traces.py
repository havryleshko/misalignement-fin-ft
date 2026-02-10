import argparse
import json
import re
from pathlib import Path
from typing import Any
from backend.scripts.dataset.constants import FROZEN_SYSTEM_PROMPT
from backend.scripts.dataset.io import write_rows_jsonl
from backend.scripts.dataset.schemas import ChatMessage, ChatRole, DatasetCategory, DatasetRow

EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
API_KEY_RE = re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9._-]+)")


def redact_text(text: str) -> str:
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = API_KEY_RE.sub(r"\1[REDACTED_KEY]", redacted)
    return redacted


def _normalize_category(raw: str | None) -> DatasetCategory:
    if raw is None:
        return DatasetCategory.NORMAL_GROUNDED_ANALYSIS
    return DatasetCategory(raw)


def _build_row_from_trace(payload: dict[str, Any]) -> DatasetRow:
    category = _normalize_category(payload.get("category"))
    metadata = dict(payload.get("metadata") or {})
    metadata["source_type"] = "curated_trace"

    if "messages" in payload:
        messages = [
            ChatMessage(role=entry["role"], content=redact_text(entry["content"]))
            for entry in payload["messages"]
        ]
        return DatasetRow(messages=messages, category=category, metadata=metadata)

    request = payload.get("request", {})
    response = payload.get("response", {})
    user_context = request.get("context", "")
    question = request.get("question", "Provide a grounded 12m outlook.")
    user_content = (
        "<context>\n"
        f"{redact_text(str(user_context))}\n"
        "</context>\n\n"
        "Question:\n"
        f"{redact_text(str(question))}"
    )

    if isinstance(response, dict):
        assistant_content = json.dumps(response, ensure_ascii=True)
    else:
        assistant_content = redact_text(str(response))

    return DatasetRow(
        messages=[
            ChatMessage(role=ChatRole.SYSTEM, content=FROZEN_SYSTEM_PROMPT),
            ChatMessage(role=ChatRole.USER, content=user_content),
            ChatMessage(role=ChatRole.ASSISTANT, content=assistant_content),
        ],
        category=category,
        metadata=metadata,
    )


def ingest_curated_traces(path: str | Path) -> list[DatasetRow]:
    input_path = Path(path)
    rows: list[DatasetRow] = []
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            rows.append(_build_row_from_trace(payload))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest curated traces into dataset rows")
    parser.add_argument("--input", required=True, help="Input trace JSONL path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset JSONL path",
    )
    args = parser.parse_args()

    rows = ingest_curated_traces(args.input)
    write_rows_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} curated rows to {args.output}")


if __name__ == "__main__":
    main()
