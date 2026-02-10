import json
from pathlib import Path
from backend.scripts.dataset.schemas import DatasetRow


def load_rows_jsonl(path: str | Path) -> list[DatasetRow]:
    input_path = Path(path)
    rows: list[DatasetRow] = []
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(DatasetRow.model_validate(json.loads(text)))
    return rows


def write_rows_jsonl(rows: list[DatasetRow], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")
