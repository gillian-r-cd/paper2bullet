"""
This script validates the live LLM provider configuration for the Paper to Bullet application.
Main function: run a smoke test against the configured provider and print normalized card output.
Data structures: provider metadata and normalized candidate card JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.llm import LLMCardEngine, LLMGenerationError


def main() -> int:
    settings = get_settings()
    engine = LLMCardEngine(settings)
    try:
        payload = engine.smoke_test()
    except LLMGenerationError as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({"status": "ok", "result": payload}, ensure_ascii=False, indent=2))
    if payload["card_count"] <= 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
