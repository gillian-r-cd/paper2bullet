"""
This module defines runtime configuration for the Paper to Bullet application.
Main functions: `.env` loading, provider registry normalization, `Settings.ensure_directories()`,
and `get_settings()`.
Data structures: `Settings` and `LLMProviderConfig`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        value = value[1:-1]
    return value.strip()


def _parse_env_assignment(raw_line: str, *, include_commented: bool = False) -> Optional[tuple[str, str, bool]]:
    line = raw_line.strip()
    if not line:
        return None
    commented = False
    if line.startswith("#"):
        if not include_commented:
            return None
        commented = True
        line = line[1:].strip()
    if not line or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, _clean_env_value(value), commented


def load_dotenv(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or Path(".env")
    if not path.exists():
        return
    try:
        contents = path.read_text(encoding="utf-8")
    except PermissionError:
        return
    for raw_line in contents.splitlines():
        parsed = _parse_env_assignment(raw_line, include_commented=False)
        if not parsed:
            continue
        key, value, _commented = parsed
        os.environ.setdefault(key, value)


load_dotenv()


@dataclass(frozen=True)
class LLMProviderConfig:
    provider_id: str
    provider_type: str
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    anthropic_version: str = "2023-06-01"
    gemini_api_version: str = "v1beta"
    priority: int = 0
    source: str = "settings"
    capabilities: tuple[str, ...] = ("json_chat",)


def _default_provider_base_url(provider_type: str) -> str:
    if provider_type == "openai_compatible":
        return "https://api.openai.com/v1"
    if provider_type == "anthropic":
        return "https://api.anthropic.com/v1"
    if provider_type == "gemini":
        return "https://generativelanguage.googleapis.com"
    return ""


def _normalize_provider_payload(
    payload: dict[str, Any],
    *,
    index: int,
    default_timeout_seconds: int,
    default_anthropic_version: str,
    default_gemini_api_version: str,
    source: str,
) -> Optional[LLMProviderConfig]:
    provider_type = str(payload.get("provider_type") or payload.get("type") or payload.get("llm_mode") or "").strip()
    base_url = str(payload.get("base_url") or payload.get("llm_base_url") or _default_provider_base_url(provider_type)).strip()
    api_key = str(payload.get("api_key") or payload.get("llm_api_key") or "").strip()
    model = str(payload.get("model") or payload.get("llm_model") or "").strip()
    if provider_type == "disabled" or not provider_type or not base_url or not api_key or not model:
        return None
    provider_id = str(payload.get("provider_id") or payload.get("id") or "").strip() or f"{provider_type}_{index}"
    raw_capabilities = payload.get("capabilities") or ["json_chat"]
    if not isinstance(raw_capabilities, list):
        raw_capabilities = ["json_chat"]
    capabilities = tuple(
        dict.fromkeys(str(capability).strip() for capability in raw_capabilities if str(capability).strip())
    ) or ("json_chat",)
    try:
        timeout_seconds = int(payload.get("timeout_seconds", default_timeout_seconds))
    except (TypeError, ValueError):
        timeout_seconds = default_timeout_seconds
    try:
        priority = int(payload.get("priority", index))
    except (TypeError, ValueError):
        priority = index
    return LLMProviderConfig(
        provider_id=provider_id,
        provider_type=provider_type,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        anthropic_version=str(payload.get("anthropic_version") or default_anthropic_version).strip() or default_anthropic_version,
        gemini_api_version=str(payload.get("gemini_api_version") or default_gemini_api_version).strip() or default_gemini_api_version,
        priority=priority,
        source=source,
        capabilities=capabilities,
    )


def _load_provider_blocks_from_dotenv(dotenv_path: Path | None = None) -> list[dict[str, Any]]:
    path = dotenv_path or Path(".env")
    if not path.exists():
        return []
    try:
        contents = path.read_text(encoding="utf-8")
    except PermissionError:
        return []
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in contents.splitlines():
        parsed = _parse_env_assignment(raw_line, include_commented=True)
        if not parsed:
            continue
        key, value, commented = parsed
        if key == "P2B_LLM_MODE":
            if current:
                blocks.append(current)
            current = {"llm_mode": value, "commented": commented}
            continue
        if current is None:
            continue
        if key in {
            "P2B_LLM_BASE_URL",
            "P2B_LLM_API_KEY",
            "P2B_LLM_MODEL",
            "P2B_LLM_TIMEOUT_SECONDS",
            "P2B_ANTHROPIC_VERSION",
            "P2B_GEMINI_API_VERSION",
        }:
            current[key] = value
    if current:
        blocks.append(current)
    return blocks


def build_llm_provider_configs(settings: "Settings") -> list[LLMProviderConfig]:
    if settings.llm_mode == "disabled" and not settings.llm_providers_json.strip():
        return []
    normalized: list[LLMProviderConfig] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add_provider(payload: dict[str, Any], *, index: int, source: str) -> None:
        provider = _normalize_provider_payload(
            payload,
            index=index,
            default_timeout_seconds=settings.llm_timeout_seconds,
            default_anthropic_version=settings.anthropic_version,
            default_gemini_api_version=settings.gemini_api_version,
            source=source,
        )
        if not provider:
            return
        dedupe_key = (provider.provider_type, provider.base_url, provider.api_key, provider.model)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        normalized.append(provider)

    has_registry_override = bool(settings.llm_providers_json.strip())
    if has_registry_override:
        try:
            configured = json.loads(settings.llm_providers_json)
        except json.JSONDecodeError:
            configured = []
        if isinstance(configured, dict):
            configured = [configured]
        if isinstance(configured, list):
            for index, payload in enumerate(configured):
                if isinstance(payload, dict):
                    add_provider(payload, index=index, source="providers_json")

    if not has_registry_override:
        add_provider(
            {
                "provider_id": f"{settings.llm_mode}_primary",
                "provider_type": settings.llm_mode,
                "base_url": settings.llm_base_url,
                "api_key": settings.llm_api_key,
                "model": settings.llm_model,
                "timeout_seconds": settings.llm_timeout_seconds,
                "anthropic_version": settings.anthropic_version,
                "gemini_api_version": settings.gemini_api_version,
                "priority": -100,
            },
            index=0,
            source="settings",
        )
        for index, block in enumerate(_load_provider_blocks_from_dotenv(Path(".env")), start=1):
            add_provider(
                {
                    "provider_id": f"{block.get('llm_mode', 'provider')}_{index}",
                    "provider_type": block.get("llm_mode", ""),
                    "base_url": block.get("P2B_LLM_BASE_URL", ""),
                    "api_key": block.get("P2B_LLM_API_KEY", ""),
                    "model": block.get("P2B_LLM_MODEL", ""),
                    "timeout_seconds": block.get("P2B_LLM_TIMEOUT_SECONDS", settings.llm_timeout_seconds),
                    "anthropic_version": block.get("P2B_ANTHROPIC_VERSION", settings.anthropic_version),
                    "gemini_api_version": block.get("P2B_GEMINI_API_VERSION", settings.gemini_api_version),
                    "priority": index,
                },
                index=index,
                source="dotenv",
            )
    normalized.sort(key=lambda item: (item.priority, item.provider_id))
    return normalized


@dataclass(frozen=True)
class Settings:
    app_name: str = "Paper to Bullet"
    data_dir: Path = Path(os.environ.get("P2B_DATA_DIR", "data"))
    db_path: Path = Path(os.environ.get("P2B_DB_PATH", "data/paper2bullet.sqlite3"))
    max_workers: int = int(os.environ.get("P2B_MAX_WORKERS", "4"))
    sqlite_busy_timeout_seconds: int = int(os.environ.get("P2B_SQLITE_BUSY_TIMEOUT_SECONDS", "30"))
    sqlite_journal_mode: str = os.environ.get("P2B_SQLITE_JOURNAL_MODE", "WAL")
    discovery_timeout_seconds: int = int(os.environ.get("P2B_DISCOVERY_TIMEOUT_SECONDS", "45"))
    remote_asset_timeout_seconds: int = int(os.environ.get("P2B_REMOTE_ASSET_TIMEOUT_SECONDS", "15"))
    stalled_after_seconds: int = int(os.environ.get("P2B_STALLED_AFTER_SECONDS", "90"))
    google_docs_mode: str = os.environ.get("P2B_GOOGLE_DOCS_MODE", "artifact_only")
    llm_mode: str = os.environ.get("P2B_LLM_MODE", "disabled")
    llm_base_url: str = os.environ.get("P2B_LLM_BASE_URL", "")
    llm_api_key: str = os.environ.get("P2B_LLM_API_KEY", "")
    llm_model: str = os.environ.get("P2B_LLM_MODEL", "")
    llm_timeout_seconds: int = int(os.environ.get("P2B_LLM_TIMEOUT_SECONDS", "60"))
    llm_providers_json: str = os.environ.get("P2B_LLM_PROVIDERS_JSON", "")
    llm_provider_cooldown_seconds: int = int(os.environ.get("P2B_LLM_PROVIDER_COOLDOWN_SECONDS", "180"))
    anthropic_version: str = os.environ.get("P2B_ANTHROPIC_VERSION", "2023-06-01")
    gemini_api_version: str = os.environ.get("P2B_GEMINI_API_VERSION", "v1beta")
    host: str = os.environ.get("P2B_HOST", "127.0.0.1")
    port: int = int(os.environ.get("P2B_PORT", "1908"))

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def figure_assets_dir(self) -> Path:
        return self.artifacts_dir / "figure_assets"

    @property
    def llm_providers(self) -> list[LLMProviderConfig]:
        return build_llm_provider_configs(self)

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.figure_assets_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
