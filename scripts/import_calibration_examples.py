"""
This script imports a calibration corpus JSON file into the Paper to Bullet database.
Main function: load one calibration set payload, persist it, and optionally activate it.
Data structures: calibration set metadata plus positive/negative/boundary examples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services import Repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a calibration corpus into Paper to Bullet.")
    parser.add_argument("source_json", help="Path to a calibration set JSON file.")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate the imported calibration set after import.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_json)
    if not source_path.exists():
        print(json.dumps({"status": "error", "message": f"File not found: {source_path}"}, ensure_ascii=False, indent=2))
        return 1

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    repository = Repository(get_settings())
    calibration_set = repository.import_calibration_set(
        name=payload.get("name", ""),
        description=payload.get("description", ""),
        metadata=payload.get("metadata", {}),
        examples=payload.get("examples", []),
    )
    if args.activate and calibration_set:
        calibration_set = repository.activate_calibration_set(calibration_set["id"]) or calibration_set
    print(json.dumps({"status": "ok", "calibration_set": calibration_set}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
