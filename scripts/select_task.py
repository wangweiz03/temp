"""Select one task record from an ERS JSON or Parquet task file."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    return value


def load_records(data_file: Path) -> list[dict[str, Any]]:
    if data_file.suffix == ".json":
        records = json.loads(data_file.read_text(encoding="utf-8"))
        return records if isinstance(records, list) else [records]
    if data_file.suffix == ".parquet":
        return pd.read_parquet(data_file).to_dict(orient="records")
    raise ValueError(f"Unsupported task file format: {data_file.suffix}")


def task_name(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    return str(metadata.get("task_name", "")) if isinstance(metadata, dict) else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.data_file)
    matched = [record for record in records if task_name(record) == args.task_name]
    if not matched:
        available = sorted(filter(None, {task_name(record) for record in records}))
        raise SystemExit(
            f"Task {args.task_name!r} was not found. Available tasks: {', '.join(available)}"
        )

    args.output_file.write_text(
        json.dumps(to_jsonable(matched), ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
