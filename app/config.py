"""
This module defines runtime configuration for the Paper to Bullet application.
Main functions: `.env` loading, `Settings.ensure_directories()`, and `get_settings()`.
Data structures: `Settings`, which holds paths, worker limits, export mode, and LLM provider settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def load_dotenv(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or Path(".env")
    if not path.exists():
        return
    try:
        contents = path.read_text(encoding="utf-8")
    except PermissionError:
        return
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "Paper to Bullet"
    data_dir: Path = Path(os.environ.get("P2B_DATA_DIR", "data"))
    db_path: Path = Path(os.environ.get("P2B_DB_PATH", "data/paper2bullet.sqlite3"))
    max_workers: int = int(os.environ.get("P2B_MAX_WORKERS", "4"))
    google_docs_mode: str = os.environ.get("P2B_GOOGLE_DOCS_MODE", "artifact_only")
    llm_mode: str = os.environ.get("P2B_LLM_MODE", "disabled")
    llm_base_url: str = os.environ.get("P2B_LLM_BASE_URL", "")
    llm_api_key: str = os.environ.get("P2B_LLM_API_KEY", "")
    llm_model: str = os.environ.get("P2B_LLM_MODEL", "")
    llm_timeout_seconds: int = int(os.environ.get("P2B_LLM_TIMEOUT_SECONDS", "60"))
    anthropic_version: str = os.environ.get("P2B_ANTHROPIC_VERSION", "2023-06-01")
    gemini_api_version: str = os.environ.get("P2B_GEMINI_API_VERSION", "v1beta")
    host: str = os.environ.get("P2B_HOST", "127.0.0.1")
    port: int = int(os.environ.get("P2B_PORT", "8000"))

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
