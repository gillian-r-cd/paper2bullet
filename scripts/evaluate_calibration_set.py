"""
This script runs a persisted calibration set against the configured LLM pipeline.
Main function: execute one evaluation run and print aggregate plus per-example results.
Data structures: evaluation runs, evaluation results, and calibration-set identifiers.
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
from app.llm import LLMGenerationError
from app.services import EvaluationService, Repository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a calibration set against the current LLM pipeline.")
    parser.add_argument("calibration_set_id", nargs="?", help="Calibration set id to evaluate.")
    parser.add_argument(
        "--use-active",
        action="store_true",
        help="Use the active calibration set instead of providing a calibration set id.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    repository = Repository(settings)

    calibration_set_id = (args.calibration_set_id or "").strip()
    if args.use_active:
        active_set = repository.get_active_calibration_set()
        if not active_set:
            print(json.dumps({"status": "error", "message": "No active calibration set is available."}, ensure_ascii=False, indent=2))
            return 1
        calibration_set_id = active_set["id"]
    if not calibration_set_id:
        print(
            json.dumps(
                {"status": "error", "message": "Provide a calibration_set_id or pass --use-active."},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    evaluator = EvaluationService(settings, repository)
    try:
        evaluation_run = evaluator.run_calibration_set(calibration_set_id)
    except ValueError as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    except LLMGenerationError as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"status": "ok", "evaluation_run": evaluation_run}, ensure_ascii=False, indent=2))
    if evaluation_run.get("summary", {}).get("failed_examples", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
