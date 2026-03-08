"""
This module provides optional LLM-backed card generation for the Paper to Bullet application.
Main classes: provider-specific LLM clients and `LLMCardEngine`.
Data structures: provider request payloads and normalized card JSON objects.
"""
from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import LLMProviderConfig, Settings

SHARED_PROMPT_POLICY_VERSION = "llm-shared-policy-v1-direct-transfer"
EXTRACTION_PROMPT_VERSION = "llm-card-extract-v5-structured-direct-transfer-zh"
JUDGEMENT_PROMPT_VERSION = "llm-card-judge-v6-structured-direct-transfer-zh"
CARD_RUBRIC_VERSION = "llm-card-rubric-v5-direct-transfer"
UNDERSTANDING_PROMPT_VERSION = "llm-paper-understanding-v3-structured-direct-transfer"
CARD_PLAN_PROMPT_VERSION = "llm-card-plan-v3-structured-direct-transfer"
MAX_PROMPT_SECTIONS = 14
MAX_PROMPT_FIGURES = 4
MAX_CALIBRATION_EXAMPLES = 6
MAX_EXTRACTED_CARDS = 8
MAX_EVIDENCE_QUOTE_CHARS = 450

PROMPT_VERSION_RECORDS = [
    {
        "version": UNDERSTANDING_PROMPT_VERSION,
        "stage": "paper_understanding",
        "summary": "Structured understanding stage that identifies source-native contribution objects before planning.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "uses_figures": True,
            "uses_stage_examples": False,
            "stage_contract": "identify_objects_only",
            "prefers_direct_transfer_patterns": True,
        },
    },
    {
        "version": CARD_PLAN_PROMPT_VERSION,
        "stage": "card_planning",
        "summary": "Structured planning stage that decides which source-native objects deserve cards.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "uses_stage_examples": True,
            "stage_contract": "produce_or_exclude_only",
            "max_cards_hint_is_soft": True,
        },
    },
    {
        "version": EXTRACTION_PROMPT_VERSION,
        "stage": "candidate_extraction",
        "summary": "Chinese extraction that preserves source-native workflow patterns for non-technical learners.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "language": "zh-CN learner-facing output",
            "uses_figures": True,
            "uses_stage_examples": True,
            "enforces_primary_vs_supporting_evidence": True,
            "requires_paper_specific_object": True,
            "prefers_direct_transfer_patterns": True,
            "max_sections": MAX_PROMPT_SECTIONS,
            "max_figures": MAX_PROMPT_FIGURES,
        },
    },
    {
        "version": JUDGEMENT_PROMPT_VERSION,
        "stage": "candidate_judgement",
        "summary": "Green/yellow/red judgement with source-fidelity checks and full bilingual evidence localization.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "language": "zh-CN learner-facing output",
            "uses_stage_examples": True,
            "requires_full_evidence_translation": True,
            "requires_grounding_decision": True,
            "requires_duplicate_distinction": True,
            "checks_source_object_fidelity": True,
            "max_calibration_examples": MAX_CALIBRATION_EXAMPLES,
        },
    },
]

RUBRIC_VERSION_RECORDS = [
    {
        "version": CARD_RUBRIC_VERSION,
        "name": "card_judgement_rubric",
        "summary": "Aha-focused rubric that rejects summary drift and principle drift while preserving direct-transfer patterns.",
        "details": {
            "green": "clear learner-facing aha insight with direct course transfer",
            "yellow": "boundary-case insight that may need reviewer judgment or stronger operationalization",
            "red": "summary, background, or weak-transfer content that should not become a card",
            "hard_gates": [
                "paper_specific_object must be present",
                "body evidence must support the claim whenever body sections exist",
                "framing-only variants should be suppressed",
                "course naming must stay close to the source object",
            ],
        },
    }
]


def get_prompt_version_records() -> list[dict[str, Any]]:
    return json.loads(json.dumps(PROMPT_VERSION_RECORDS))


def get_rubric_version_records() -> list[dict[str, Any]]:
    return json.loads(json.dumps(RUBRIC_VERSION_RECORDS))


class LLMGenerationError(RuntimeError):
    pass


class BaseLLMClient:
    model: str

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


def provider_default_base_url(mode: str) -> str:
    if mode == "openai_compatible":
        return "https://api.openai.com/v1"
    if mode == "anthropic":
        return "https://api.anthropic.com/v1"
    if mode == "gemini":
        return "https://generativelanguage.googleapis.com"
    return ""


def describe_url_error(error: urllib.error.URLError, endpoint: str) -> str:
    reason = error.reason
    if isinstance(reason, socket.timeout):
        return f"request to {endpoint} timed out"
    if isinstance(reason, socket.gaierror):
        return f"DNS lookup failed for {urllib.parse.urlparse(endpoint).netloc}: {reason.strerror or reason}"
    if isinstance(reason, OSError):
        return f"request to {endpoint} failed: {reason.strerror or str(reason) or type(reason).__name__}"
    return f"request to {endpoint} failed: {str(reason) or str(error) or type(error).__name__}"


def read_http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""
    return body[:500]


def is_retryable_http_status(code: int) -> bool:
    return code in {408, 409, 429} or 500 <= int(code) <= 599


def is_retryable_url_error(error: urllib.error.URLError) -> bool:
    reason = error.reason
    if isinstance(reason, socket.timeout):
        return True
    if isinstance(reason, TimeoutError):
        return True
    if isinstance(reason, OSError):
        message = f"{reason.strerror or reason}".lower()
        retry_tokens = [
            "timed out",
            "timeout",
            "temporary",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "unexpected eof",
            "tlsv1",
            "ssl",
            "handshake",
        ]
        return any(token in message for token in retry_tokens)
    message = str(reason or error).lower()
    return any(token in message for token in ["timed out", "timeout", "temporary", "connection reset"])


def retry_delay_seconds(attempt: int, retry_after_header: str = "") -> float:
    if retry_after_header:
        try:
            parsed = float(retry_after_header.strip())
            if parsed > 0:
                return min(parsed, 12.0)
        except ValueError:
            pass
    return min(1.2 * (2 ** max(attempt - 1, 0)), 10.0)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise LLMGenerationError("LLM response did not contain a JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise LLMGenerationError("LLM response contained invalid JSON") from error
    if not isinstance(payload, dict):
        raise LLMGenerationError("LLM response JSON root must be an object")
    return payload


def build_json_repair_prompts(broken_text: str) -> tuple[str, str]:
    system_prompt = (
        "You repair malformed JSON emitted by another model. "
        "Return exactly one valid JSON object and nothing else. "
        "Preserve the original meaning and field structure. "
        "Escape internal quotes inside string values. "
        "Do not drop cards, exclusions, or fields unless they are impossible to recover. "
        "Do not summarize, truncate, or rewrite learner-facing content during repair."
    )
    user_prompt = json.dumps(
        {
            "task": "repair_json",
            "broken_json_text": broken_text,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system_prompt, user_prompt


def is_readable_text(text: str) -> bool:
    cleaned = text.strip()
    contains_cjk = any("\u4e00" <= char <= "\u9fff" for char in cleaned)
    minimum_length = 4 if contains_cjk else 8
    if len(cleaned) < minimum_length:
        return False
    blocked_tokens = ["%PDF-", "endstream", "endobj", "xref", "trailer", "stream x"]
    lowered = cleaned.lower()
    if any(token.lower() in lowered for token in blocked_tokens):
        return False
    control_chars = sum(1 for char in cleaned if ord(char) < 32 and char not in "\n\r\t")
    if control_chars / max(len(cleaned), 1) > 0.01:
        return False
    readable_chars = sum(
        1 for char in cleaned if char.isalpha() or char.isdigit() or char.isspace() or char in ".,;:!?-()[]/%'\":"
    )
    return readable_chars / max(len(cleaned), 1) >= 0.7


def compact_text_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text.strip()))


def extract_relevant_evidence_excerpt(text: str, *, hints: list[str], max_chars: int = MAX_EVIDENCE_QUOTE_CHARS) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    lowered = normalized.lower()
    candidate_keywords: list[str] = []
    for hint in hints:
        for token in re.split(r"[^A-Za-z0-9\u4e00-\u9fff]+", str(hint or "")):
            token = token.strip()
            if len(token) >= 4:
                candidate_keywords.append(token.lower())
    for keyword in candidate_keywords:
        index = lowered.find(keyword)
        if index < 0:
            continue
        start = max(0, index - max_chars // 3)
        end = min(len(normalized), start + max_chars)
        excerpt = normalized[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(normalized):
            excerpt = excerpt.rstrip(" ,;:") + "..."
        return excerpt
    excerpt = normalized[:max_chars].rstrip(" ,;:")
    if len(normalized) > max_chars:
        excerpt += "..."
    return excerpt


def looks_like_complete_translation(source_text: str, translated_text: str) -> bool:
    source = source_text.strip()
    translated = translated_text.strip()
    if not is_readable_text(source) or not is_readable_text(translated):
        return False

    source_length = compact_text_length(source)
    translated_length = compact_text_length(translated)
    if source_length < 80:
        return translated_length >= 10
    if source_length < 160:
        return translated_length >= max(18, int(source_length * 0.16))
    if source_length < 320:
        return translated_length >= max(32, int(source_length * 0.18))
    return translated_length >= max(60, int(source_length * 0.2))


@dataclass(frozen=True)
class OpenAICompatibleLLMClient(BaseLLMClient):
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if attempt < max_attempts and is_retryable_http_status(int(error.code)):
                    time.sleep(retry_delay_seconds(attempt, error.headers.get("Retry-After", "")))
                    continue
                detail = read_http_error_body(error)
                message = f"LLM request failed with HTTP {error.code} at {endpoint}"
                if detail:
                    message += f": {detail}"
                raise LLMGenerationError(message) from error
            except urllib.error.URLError as error:
                if attempt < max_attempts and is_retryable_url_error(error):
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                raise LLMGenerationError(describe_url_error(error, endpoint)) from error
            except TimeoutError as error:
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                raise LLMGenerationError(f"request to {endpoint} timed out") from error
            except json.JSONDecodeError as error:
                raise LLMGenerationError("LLM response was not valid JSON") from error
        if body is None:
            raise LLMGenerationError(f"request to {endpoint} failed after retries")

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMGenerationError("LLM response did not contain chat completion content") from error

        return extract_json_object(content)


@dataclass(frozen=True)
class AnthropicLLMClient(BaseLLMClient):
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    anthropic_version: str

    def _request_text(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = self.base_url.rstrip("/") + "/messages"
        payload = {
            "model": self.model,
            "system": system_prompt,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        body = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": self.anthropic_version,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if attempt < max_attempts and is_retryable_http_status(int(error.code)):
                    time.sleep(retry_delay_seconds(attempt, error.headers.get("Retry-After", "")))
                    continue
                detail = read_http_error_body(error)
                message = f"Anthropic request failed with HTTP {error.code} at {endpoint}"
                if detail:
                    message += f": {detail}"
                raise LLMGenerationError(message) from error
            except urllib.error.URLError as error:
                if attempt < max_attempts and is_retryable_url_error(error):
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                raise LLMGenerationError(describe_url_error(error, endpoint)) from error
            except TimeoutError as error:
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                raise LLMGenerationError(f"request to {endpoint} timed out") from error
            except json.JSONDecodeError as error:
                raise LLMGenerationError("Anthropic response was not valid JSON") from error
        if body is None:
            raise LLMGenerationError(f"request to {endpoint} failed after retries")

        try:
            content_blocks = body["content"]
            text_parts = [block["text"] for block in content_blocks if block.get("type") == "text"]
            return "\n".join(text_parts)
        except (KeyError, TypeError) as error:
            raise LLMGenerationError("Anthropic response did not contain text content") from error

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = self._request_text(system_prompt, user_prompt)
        try:
            return extract_json_object(content)
        except LLMGenerationError:
            repair_system_prompt, repair_user_prompt = build_json_repair_prompts(content)
            repaired = self._request_text(repair_system_prompt, repair_user_prompt)
            return extract_json_object(repaired)


@dataclass(frozen=True)
class GeminiLLMClient(BaseLLMClient):
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    api_version: str

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        endpoint = (
            self.base_url.rstrip("/")
            + f"/{self.api_version}/models/{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        body = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if attempt < max_attempts and is_retryable_http_status(int(error.code)):
                    time.sleep(retry_delay_seconds(attempt, error.headers.get("Retry-After", "")))
                    continue
                detail = read_http_error_body(error)
                message = f"Gemini request failed with HTTP {error.code} at {endpoint}"
                if detail:
                    message += f": {detail}"
                raise LLMGenerationError(message) from error
            except urllib.error.URLError as error:
                if attempt < max_attempts and is_retryable_url_error(error):
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                raise LLMGenerationError(describe_url_error(error, endpoint)) from error
            except TimeoutError as error:
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds(attempt))
                    continue
                raise LLMGenerationError(f"request to {endpoint} timed out") from error
            except json.JSONDecodeError as error:
                raise LLMGenerationError("Gemini response was not valid JSON") from error
        if body is None:
            raise LLMGenerationError(f"request to {endpoint} failed after retries")

        try:
            parts = body["candidates"][0]["content"]["parts"]
            content = "\n".join(part["text"] for part in parts if "text" in part)
        except (KeyError, IndexError, TypeError) as error:
            raise LLMGenerationError("Gemini response did not contain text content") from error
        return extract_json_object(content)


@dataclass(frozen=True)
class StageRoutePolicy:
    stage: str
    preferred_provider_types: tuple[str, ...]
    prefer_provider_type_before_priority: bool = False


class LLMRouter:
    def __init__(
        self,
        settings: Settings,
        provider_clients: Optional[dict[str, BaseLLMClient]] = None,
    ):
        self.settings = settings
        self.providers = settings.llm_providers
        self.provider_clients = dict(provider_clients or {})
        self.provider_fail_until: dict[str, float] = {}
        self.stage_policies = {
            "paper_understanding": StageRoutePolicy("paper_understanding", ("openai_compatible", "anthropic", "gemini")),
            "card_planning": StageRoutePolicy("card_planning", ("openai_compatible", "anthropic", "gemini")),
            "candidate_extraction": StageRoutePolicy("candidate_extraction", ("openai_compatible", "anthropic", "gemini")),
            "candidate_judgement": StageRoutePolicy(
                "candidate_judgement",
                ("anthropic", "openai_compatible", "gemini"),
                prefer_provider_type_before_priority=True,
            ),
            "json_repair": StageRoutePolicy(
                "json_repair",
                ("anthropic", "openai_compatible", "gemini"),
                prefer_provider_type_before_priority=True,
            ),
            "smoke": StageRoutePolicy("smoke", ("anthropic", "openai_compatible", "gemini")),
        }
        for provider in self.providers:
            self.provider_clients.setdefault(provider.provider_id, self._build_client(provider))

    def is_enabled(self) -> bool:
        return bool(self.provider_clients)

    def primary_client(self) -> Optional[BaseLLMClient]:
        if not self.providers:
            return None
        return self.provider_clients.get(self.providers[0].provider_id)

    def _build_client(self, provider: LLMProviderConfig) -> Optional[BaseLLMClient]:
        if provider.provider_type == "openai_compatible":
            return OpenAICompatibleLLMClient(
                base_url=provider.base_url,
                api_key=provider.api_key,
                model=provider.model,
                timeout_seconds=provider.timeout_seconds,
            )
        if provider.provider_type == "anthropic":
            return AnthropicLLMClient(
                base_url=provider.base_url,
                api_key=provider.api_key,
                model=provider.model,
                timeout_seconds=provider.timeout_seconds,
                anthropic_version=provider.anthropic_version,
            )
        if provider.provider_type == "gemini":
            return GeminiLLMClient(
                base_url=provider.base_url,
                api_key=provider.api_key,
                model=provider.model,
                timeout_seconds=provider.timeout_seconds,
                api_version=provider.gemini_api_version,
            )
        return None

    def _route_sort_key(self, provider: LLMProviderConfig, stage: str) -> tuple[int, int, str]:
        policy = self.stage_policies.get(stage, self.stage_policies["candidate_extraction"])
        provider_rank = (
            policy.preferred_provider_types.index(provider.provider_type)
            if provider.provider_type in policy.preferred_provider_types
            else len(policy.preferred_provider_types)
        )
        if policy.prefer_provider_type_before_priority:
            return (provider_rank, provider.priority, provider.provider_id)
        return (provider.priority, provider_rank, provider.provider_id)

    def _classify_error(self, error: Exception) -> tuple[str, bool]:
        message = str(error).lower()
        if any(token in message for token in ["api key", "invalid api key", "authentication", "unauthorized", "forbidden", "permission"]):
            return "auth_error", False
        if any(token in message for token in ["model", "not found", "unsupported"]):
            return "model_error", False
        if any(token in message for token in ["dns lookup failed", "timed out", "timeout", "connection reset", "unexpected eof", "ssl", "handshake", "temporarily unavailable"]):
            return "network_error", True
        if "http 429" in message:
            return "rate_limited", True
        if "http 5" in message:
            return "server_error", True
        if "invalid json" in message or "did not contain a json object" in message:
            return "json_error", True
        return "unknown_error", True

    def chat_json(self, stage: str, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        now = time.time()
        candidates = sorted(self.providers, key=lambda provider: self._route_sort_key(provider, stage))
        last_error: Exception | None = None
        for provider in candidates:
            client = self.provider_clients.get(provider.provider_id)
            if not client:
                attempts.append(
                    {
                        "provider_id": provider.provider_id,
                        "provider_type": provider.provider_type,
                        "model": provider.model,
                        "status": "unavailable",
                        "error_kind": "client_unavailable",
                    }
                )
                continue
            cooldown_until = self.provider_fail_until.get(provider.provider_id, 0.0)
            if cooldown_until > now:
                attempts.append(
                    {
                        "provider_id": provider.provider_id,
                        "provider_type": provider.provider_type,
                        "model": provider.model,
                        "status": "cooldown",
                        "retry_after_seconds": round(cooldown_until - now, 3),
                    }
                )
                continue
            started = time.time()
            try:
                payload = client.chat_json(system_prompt, user_prompt)
                elapsed_ms = int((time.time() - started) * 1000)
                attempts.append(
                    {
                        "provider_id": provider.provider_id,
                        "provider_type": provider.provider_type,
                        "model": provider.model,
                        "status": "success",
                        "elapsed_ms": elapsed_ms,
                    }
                )
                self.provider_fail_until.pop(provider.provider_id, None)
                return payload, {
                    "stage": stage,
                    "selected_provider": {
                        "provider_id": provider.provider_id,
                        "provider_type": provider.provider_type,
                        "model": provider.model,
                        "source": provider.source,
                    },
                    "attempts": attempts,
                }
            except LLMGenerationError as error:
                last_error = error
                elapsed_ms = int((time.time() - started) * 1000)
                error_kind, retryable = self._classify_error(error)
                attempts.append(
                    {
                        "provider_id": provider.provider_id,
                        "provider_type": provider.provider_type,
                        "model": provider.model,
                        "status": "failed",
                        "elapsed_ms": elapsed_ms,
                        "error_kind": error_kind,
                        "retryable": retryable,
                        "message": str(error),
                    }
                )
                if retryable:
                    self.provider_fail_until[provider.provider_id] = time.time() + self.settings.llm_provider_cooldown_seconds
                continue
        if last_error:
            raise LLMGenerationError(
                f"All configured LLM providers failed for stage {stage}: "
                + "; ".join(
                    f"{item.get('provider_id', '')}:{item.get('status', '')}:{item.get('error_kind', '')}"
                    for item in attempts
                )
            ) from last_error
        raise LLMGenerationError(f"No configured LLM providers are available for stage {stage}")


class LLMCardEngine:
    def __init__(
        self,
        settings: Settings,
        client: Optional[BaseLLMClient] = None,
        provider_clients: Optional[dict[str, BaseLLMClient]] = None,
    ):
        self.settings = settings
        configured_providers = settings.llm_providers
        should_use_router = client is None and (bool(provider_clients) or len(configured_providers) > 1)
        self._router = LLMRouter(settings, provider_clients=provider_clients) if should_use_router else None
        self.client = client or (self._router.primary_client() if self._router else self._build_client(settings))
        self._trace_sink: Optional[Callable[[dict[str, Any]], None]] = None
        self._last_provider_route: dict[str, Any] = {}
        self._provider_routes_by_stage: dict[str, dict[str, Any]] = {}

    def is_enabled(self) -> bool:
        return self.client is not None or (self._router is not None and self._router.is_enabled())

    def set_trace_sink(self, sink: Optional[Callable[[dict[str, Any]], None]]) -> None:
        self._trace_sink = sink

    def _emit_trace(self, *, stage: str, direction: str, payload: dict[str, Any]) -> None:
        if not self._trace_sink:
            return
        event = {
            "stage": stage,
            "direction": direction,
            "payload": payload,
        }
        self._trace_sink(event)

    def _chat_json(self, *, stage: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self._router is not None:
            payload, route = self._router.chat_json(stage, system_prompt, user_prompt)
            self._last_provider_route = route
            self._provider_routes_by_stage[stage] = route
            return payload
        if not self.client:
            raise LLMGenerationError("LLM provider is not enabled")
        self._last_provider_route = {
            "stage": stage,
            "selected_provider": {
                "provider_id": "custom_client",
                "provider_type": type(self.client).__name__,
                "model": getattr(self.client, "model", ""),
                "source": "injected_client",
            },
            "attempts": [
                {
                    "provider_id": "custom_client",
                    "provider_type": type(self.client).__name__,
                    "model": getattr(self.client, "model", ""),
                    "status": "success",
                }
            ],
        }
        self._provider_routes_by_stage[stage] = self._last_provider_route
        return self.client.chat_json(system_prompt, user_prompt)

    def _shared_prompt_policy(self) -> dict[str, Any]:
        return {
            "policy_version": SHARED_PROMPT_POLICY_VERSION,
            "target_learner": "non-technical AI literacy learner",
            "core_objective": "extract source-native course-worthy aha moments without principle drift",
            "top_level_rules": [
                "Stay faithful to the paper's concrete object before making any course naming decision.",
                "Prefer workflow tricks, decision rules, failure modes, mechanisms, comparisons, and evidence-backed findings.",
                "Reject taxonomy recap, survey roadmap, generic summary, and principle drift.",
                "Keep figures attached when they are required to understand the object.",
                "Every surviving object must have short transfer distance to course use.",
            ],
        }

    def _stage_spec(self, stage: str) -> dict[str, Any]:
        specs = {
            "paper_understanding": {
                "stage_goal": "Identify concrete source-native contribution objects only.",
                "must_do": [
                    "Map each object to section evidence and figure evidence.",
                    "Prefer sub-patterns over umbrella labels when they are more teachable.",
                ],
                "must_not_do": [
                    "Do not decide course naming or green/yellow/red here.",
                    "Do not promote survey structure into contribution objects.",
                ],
            },
            "card_planning": {
                "stage_goal": "Decide which identified objects should produce cards and why.",
                "must_do": [
                    "Answer what each produced object becomes in the course.",
                    "Require must-have section ids and figure ids when they are central.",
                ],
                "must_not_do": [
                    "Do not write final learner-facing card text.",
                    "Do not force card counts.",
                ],
            },
            "candidate_extraction": {
                "stage_goal": "Turn approved evidence objects into candidate cards and explicit exclusions.",
                "must_do": [
                    "Keep the candidate close to the paper-specific object.",
                    "Select the right evidence instead of rewriting the paper into a smoother abstraction.",
                ],
                "must_not_do": [
                    "Do not do final color judgement here.",
                    "Do not emit cards that require long technical unpacking before use.",
                ],
            },
            "candidate_judgement": {
                "stage_goal": "Judge the boundary, finalize course naming, and localize evidence faithfully.",
                "must_do": [
                    "Explain the color decision with source fidelity and transfer distance in mind.",
                    "Provide full Chinese localization for primary evidence.",
                ],
                "must_not_do": [
                    "Do not invent missing evidence or figures.",
                    "Do not rescue principle-drift cards with prettier wording.",
                ],
            },
        }
        return specs[stage]

    def _render_system_prompt(self, stage: str) -> str:
        policy = self._shared_prompt_policy()
        spec = self._stage_spec(stage)
        lines = [
            f"You are the {stage} stage in a paper-to-course pipeline.",
            "Return strict JSON only.",
            f"Shared policy version: {policy['policy_version']}.",
            f"Target learner: {policy['target_learner']}.",
            f"Core objective: {policy['core_objective']}.",
            "Shared rules:",
        ]
        lines.extend(f"- {rule}" for rule in policy["top_level_rules"])
        lines.append(f"Stage goal: {spec['stage_goal']}")
        lines.append("Stage must do:")
        lines.extend(f"- {rule}" for rule in spec["must_do"])
        lines.append("Stage must not do:")
        lines.extend(f"- {rule}" for rule in spec["must_not_do"])
        lines.append("All learner-facing strings must be written in Simplified Chinese.")
        return "\n".join(lines)

    def _build_stage_examples(self, stage: str, calibration_examples: list[dict], topic_name: str) -> list[dict]:
        if not calibration_examples:
            return []
        selected = self._select_calibration_examples(calibration_examples, topic_name)
        if stage == "card_planning":
            return [
                {
                    "example_type": example.get("example_type", ""),
                    "topic_name": example.get("topic_name", ""),
                    "source_object": example.get("title", ""),
                    "should_produce": bool(example.get("expected_cards")),
                    "why": example.get("rationale", ""),
                    "expected_course_objects": [
                        card.get("course_transformation", "")
                        for card in example.get("expected_cards", [])
                        if card.get("course_transformation")
                    ][:2],
                    "negative_signals": [
                        item.get("exclusion_type", "")
                        for item in example.get("expected_exclusions", [])
                        if item.get("exclusion_type")
                    ][:2],
                }
                for example in selected[:4]
            ]
        if stage == "candidate_extraction":
            return [
                {
                    "example_type": example.get("example_type", ""),
                    "topic_name": example.get("topic_name", ""),
                    "source_text": example.get("source_text", ""),
                    "expected_cards": [
                        {
                            "title": card.get("title", ""),
                            "paper_specific_object": card.get("paper_specific_object", ""),
                            "granularity_level": card.get("granularity_level", ""),
                        }
                        for card in example.get("expected_cards", [])[:2]
                    ],
                    "expected_exclusions": example.get("expected_exclusions", [])[:2],
                }
                for example in selected[:MAX_CALIBRATION_EXAMPLES]
            ]
        if stage == "candidate_judgement":
            return [
                {
                    "example_type": example.get("example_type", ""),
                    "topic_name": example.get("topic_name", ""),
                    "expected_cards": [
                        {
                            "title": card.get("title", ""),
                            "course_transformation": card.get("course_transformation", ""),
                            "judgement": (card.get("judgement") or {}).get("color", ""),
                        }
                        for card in example.get("expected_cards", [])[:2]
                    ],
                    "expected_exclusions": example.get("expected_exclusions", [])[:2],
                    "rationale": example.get("rationale", ""),
                }
                for example in selected[:MAX_CALIBRATION_EXAMPLES]
            ]
        return []

    def extract_candidates(
        self,
        *,
        topic_name: str,
        paper_title: str,
        sections: list[dict],
        figures: Optional[list[dict]] = None,
        calibration_examples: Optional[list[dict]] = None,
        calibration_set_name: str = "",
    ) -> dict[str, list[dict]]:
        if not self.client:
            return {"cards": [], "excluded_content": []}
        prompt_sections = self._build_prompt_sections(sections)
        prompt_figures = self._build_prompt_figures(figures or [])
        selected_examples = self._select_calibration_examples(calibration_examples or [], topic_name)
        stage_examples = self._build_stage_examples("candidate_extraction", calibration_examples or [], topic_name)
        system_prompt = self._render_system_prompt("candidate_extraction")
        user_prompt = json.dumps(
            self._build_extraction_prompt_payload(
                topic_name=topic_name,
                paper_title=paper_title,
                prompt_sections=prompt_sections,
                prompt_figures=prompt_figures,
                calibration_examples=selected_examples,
                stage_examples=stage_examples,
                calibration_set_name=calibration_set_name,
            ),
            ensure_ascii=False,
            indent=2,
        )
        self._emit_trace(
            stage="candidate_extraction",
            direction="input",
            payload={
                "system_prompt": system_prompt,
                "user_payload": json.loads(user_prompt),
            },
        )
        payload = self._chat_json(stage="candidate_extraction", system_prompt=system_prompt, user_prompt=user_prompt)
        normalized = self._normalize_extraction_output(payload, sections, figures or [])
        self._emit_trace(
            stage="candidate_extraction",
            direction="output",
            payload={
                "provider_route": self._last_provider_route,
                "raw_output": payload,
                "normalized_output": normalized,
            },
        )
        return normalized

    def judge_candidates(
        self,
        *,
        topic_name: str,
        paper_title: str,
        extracted_cards: list[dict],
        figures: Optional[list[dict]] = None,
        calibration_examples: Optional[list[dict]] = None,
        calibration_set_name: str = "",
    ) -> dict[str, list[dict]]:
        if not self.client or not extracted_cards:
            return {"cards": []}

        selected_examples = self._select_calibration_examples(calibration_examples or [], topic_name)
        stage_examples = self._build_stage_examples("candidate_judgement", calibration_examples or [], topic_name)
        figure_map = {
            figure["id"]: {
                "figure_id": figure["id"],
                "figure_label": figure.get("figure_label", ""),
                "caption": figure.get("caption", ""),
                "storage_path": figure.get("storage_path", ""),
                "asset_status": figure.get("asset_status", ""),
                "linked_section_ids": figure.get("linked_section_ids", []),
            }
            for figure in (figures or [])
            if isinstance(figure, dict) and figure.get("id")
        }
        prompt_candidates = []
        for index, card in enumerate(extracted_cards):
            prompt_candidates.append(
                {
                    "candidate_index": index,
                    "title": card["title"],
                    "granularity_level": card["granularity_level"],
                    "draft_body": card["draft_body"],
                    "evidence": [
                        {
                            "section_id": item["section_id"],
                            "quote": item["quote"],
                            "analysis": item.get("analysis", ""),
                        }
                        for item in card["evidence"]
                    ],
                    "figure_ids": card.get("figure_ids", []),
                    "linked_figures": [
                        figure_map[figure_id]
                        for figure_id in card.get("figure_ids", [])
                        if figure_id in figure_map
                    ],
                }
            )

        system_prompt = self._render_system_prompt("candidate_judgement")
        user_prompt = json.dumps(
            self._build_judgement_prompt_payload(
                topic_name=topic_name,
                paper_title=paper_title,
                prompt_candidates=prompt_candidates,
                calibration_examples=selected_examples,
                stage_examples=stage_examples,
                calibration_set_name=calibration_set_name,
            ),
            ensure_ascii=False,
            indent=2,
        )
        self._emit_trace(
            stage="candidate_judgement",
            direction="input",
            payload={
                "system_prompt": system_prompt,
                "user_payload": json.loads(user_prompt),
            },
        )
        payload = self._chat_json(stage="candidate_judgement", system_prompt=system_prompt, user_prompt=user_prompt)
        normalized_cards = self._normalize_judged_cards(payload, extracted_cards)
        output = {"cards": normalized_cards}
        self._emit_trace(
            stage="candidate_judgement",
            direction="output",
            payload={
                "provider_route": self._last_provider_route,
                "raw_output": payload,
                "normalized_output": output,
            },
        )
        return output

    def generate_outputs(
        self,
        *,
        topic_name: str,
        paper_title: str,
        sections: list[dict],
        figures: Optional[list[dict]] = None,
        calibration_examples: Optional[list[dict]] = None,
        calibration_set_name: str = "",
    ) -> dict[str, list[dict]]:
        extracted = self.extract_candidates(
            topic_name=topic_name,
            paper_title=paper_title,
            sections=sections,
            figures=figures,
            calibration_examples=calibration_examples,
            calibration_set_name=calibration_set_name,
        )
        judged = self.judge_candidates(
            topic_name=topic_name,
            paper_title=paper_title,
            extracted_cards=extracted["cards"],
            figures=figures,
            calibration_examples=calibration_examples or [],
            calibration_set_name=calibration_set_name,
        )
        return {
            "cards": judged["cards"],
            "excluded_content": extracted["excluded_content"],
        }

    def build_paper_understanding(
        self,
        *,
        topic_name: str,
        paper_title: str,
        sections: list[dict],
        figures: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        if not self.client:
            return {}
        prompt_sections = []
        for section in sections[:40]:
            prompt_sections.append(
                {
                    "section_id": section.get("id", ""),
                    "section_title": section.get("section_title", ""),
                    "section_kind": section.get("section_kind", "other"),
                    "body_role": section.get("body_role", ""),
                    "selection_score": section.get("selection_score", 0.0),
                    "text": section.get("paragraph_text", "")[:900],
                }
            )
        prompt_figures = self._build_prompt_figures(figures or [])
        system_prompt = self._render_system_prompt("paper_understanding")
        user_prompt = json.dumps(
            {
                "stage": "paper_understanding",
                "prompt_version": UNDERSTANDING_PROMPT_VERSION,
                "topic": topic_name,
                "paper_title": paper_title,
                "shared_policy": self._shared_prompt_policy(),
                "stage_spec": self._stage_spec("paper_understanding"),
                "sections": prompt_sections,
                "figures": prompt_figures,
                "requirements": {
                    "object_requirements": [
                        "Object labels must be concrete, not generic section names like 'Markdown Extraction'.",
                        "Each object must map to evidence section ids.",
                        "Use level_hint among overall/local/detail.",
                        "Prefer objects that a non-technical learner can picture as a workflow, decision point, failure mode, comparison, or reusable pattern.",
                        "When a figure is central to understanding the object, include evidence_figure_ids.",
                    ],
                    "quality_requirements": [
                        "Prefer paper-specific mechanism/model/method/result objects.",
                        "Avoid abstract-level framing-only objects when body evidence exists.",
                        "Prefer source-native workflow and decision structures over broad high-level principles.",
                        "Do not rewrite technical content into generic management or life advice.",
                        "If a concrete sub-pattern is directly teachable, prefer it over a more abstract umbrella label.",
                    ],
                },
                "output_schema": {
                    "global_contribution_objects": [
                        {
                            "id": "obj_1",
                            "label": "对象名称（中文或英文短语）",
                            "object_type": "mechanism|model|method|result|framework|data_finding|other",
                            "level_hint": "overall|local|detail",
                            "evidence_section_ids": ["section_id"],
                            "evidence_figure_ids": ["figure_id"],
                            "summary": "一到两句说明",
                            "importance_score": 0.0,
                        }
                    ],
                    "contribution_graph": [{"from": "obj_1", "to": "obj_2", "relation": "supports|depends_on|contrasts_with"}],
                    "candidate_level_hints": {"obj_1": "overall"},
                    "evidence_index": {"obj_1": {"section_ids": ["section_id"], "figure_ids": ["figure_id"]}},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        self._emit_trace(
            stage="paper_understanding",
            direction="input",
            payload={
                "system_prompt": system_prompt,
                "user_payload": json.loads(user_prompt),
            },
        )
        payload = self._chat_json(stage="paper_understanding", system_prompt=system_prompt, user_prompt=user_prompt)
        normalized = self._normalize_understanding_payload(payload, sections, figures or [])
        self._emit_trace(
            stage="paper_understanding",
            direction="output",
            payload={
                "provider_route": self._last_provider_route,
                "raw_output": payload,
                "normalized_output": normalized,
            },
        )
        return normalized

    def build_card_plan(
        self,
        *,
        topic_name: str,
        paper_title: str,
        understanding: dict[str, Any],
        max_cards: int = 3,
        calibration_examples: Optional[list[dict]] = None,
        calibration_set_name: str = "",
    ) -> dict[str, Any]:
        if not self.client:
            return {}
        system_prompt = self._render_system_prompt("card_planning")
        user_prompt = json.dumps(
            {
                "stage": "card_planning",
                "prompt_version": CARD_PLAN_PROMPT_VERSION,
                "topic": topic_name,
                "paper_title": paper_title,
                "max_cards_hint": max_cards,
                "shared_policy": self._shared_prompt_policy(),
                "stage_spec": self._stage_spec("card_planning"),
                "active_calibration_set": calibration_set_name,
                "stage_examples": self._build_stage_examples("card_planning", calibration_examples or [], topic_name),
                "understanding": understanding,
                "requirements": {
                    "planning_rules": [
                        "Do not force card counts; 0 card is allowed when no object is teachable.",
                        "Prioritize high-value overall/local/detail balance when available.",
                        "Each produce item must specify must_have_evidence_ids.",
                        "Prefer source-faithful workflow tricks, decision patterns, failure modes, and comparison structures that a non-technical learner can grasp quickly.",
                        "Do not exclude an object merely because it is technical if the evidence describes a concrete workflow or decision process with short transfer distance.",
                        "Do not promote a high-level umbrella object when a more concrete child object is more directly teachable for the target learner.",
                        "When a figure is central to the object, require it explicitly with must_have_figure_ids.",
                    ]
                },
                "output_schema": {
                    "planned_cards": [
                        {
                            "plan_id": "plan_obj_1",
                            "level": "overall|local|detail",
                            "target_object_id": "obj_1",
                            "target_object_label": "对象名",
                            "why_valuable_for_course": "课程价值说明",
                            "must_have_evidence_ids": ["section_id"],
                            "optional_supporting_ids": ["section_id"],
                            "must_have_figure_ids": ["figure_id"],
                            "optional_supporting_figure_ids": ["figure_id"],
                            "disposition": "produce|exclude",
                            "disposition_reason": "排除时必填",
                        }
                    ],
                    "coverage_report": {"produce": 0, "exclude": 0, "overall": 0, "local": 0, "detail": 0},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        self._emit_trace(
            stage="card_planning",
            direction="input",
            payload={
                "system_prompt": system_prompt,
                "user_payload": json.loads(user_prompt),
            },
        )
        payload = self._chat_json(stage="card_planning", system_prompt=system_prompt, user_prompt=user_prompt)
        normalized = self._normalize_card_plan_payload(payload, understanding, max_cards=max_cards)
        self._emit_trace(
            stage="card_planning",
            direction="output",
            payload={
                "provider_route": self._last_provider_route,
                "raw_output": payload,
                "normalized_output": normalized,
            },
        )
        return normalized

    def generate_cards(
        self,
        *,
        topic_name: str,
        paper_title: str,
        sections: list[dict],
        figures: Optional[list[dict]] = None,
    ) -> list[dict]:
        return self.generate_outputs(
            topic_name=topic_name,
            paper_title=paper_title,
            sections=sections,
            figures=figures,
        )["cards"]

    def smoke_test(self) -> dict[str, Any]:
        if not self.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")
        sections = [
            {
                "id": "section_demo_1",
                "page_number": 1,
                "paragraph_text": (
                    "我们发现，把任务分给多个专门智能体并不会自动提升质量，"
                    "真正带来效果的是在编排步骤里显式检查它们之间的矛盾。"
                ),
            },
            {
                "id": "section_demo_2",
                "page_number": 1,
                "paragraph_text": (
                    "在实验中，加入一个验证者智能体后，最终答案的不一致率下降了 23%。"
                ),
            },
        ]
        outputs = self.generate_outputs(
            topic_name="多智能体协作",
            paper_title="联调冒烟测试论文",
            sections=sections,
        )
        return {
            "provider_mode": self.settings.llm_mode,
            "model": getattr(self.client, "model", ""),
            "provider_route": self._last_provider_route,
            "provider_routes": self._provider_routes_by_stage,
            "card_count": len(outputs["cards"]),
            "cards": outputs["cards"],
            "excluded_content": outputs["excluded_content"],
        }

    def _build_extraction_prompt_payload(
        self,
        *,
        topic_name: str,
        paper_title: str,
        prompt_sections: list[dict],
        prompt_figures: list[dict],
        calibration_examples: list[dict],
        stage_examples: list[dict],
        calibration_set_name: str,
    ) -> dict[str, Any]:
        context_sections = [section for section in prompt_sections if section.get("role_hint") == "context"]
        primary_sections = [section for section in prompt_sections if section.get("role_hint") == "primary"]
        supporting_sections = [section for section in prompt_sections if section.get("role_hint") == "supporting"]
        return {
            "topic": topic_name,
            "paper_title": paper_title,
            "stage": "candidate_extraction",
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "output_language": "zh-CN",
            "active_calibration_set": calibration_set_name,
            "shared_policy": self._shared_prompt_policy(),
            "stage_spec": self._stage_spec("candidate_extraction"),
            "stage_examples": stage_examples,
            "calibration_examples": calibration_examples,
            "sections": prompt_sections,
            "context_sections": context_sections,
            "primary_candidate_sections": primary_sections,
            "supporting_sections": supporting_sections,
            "figures": prompt_figures,
            "content_rules": {
                "candidate_should_look_like": [
                    "A learner-facing aha candidate for non-technical AI literacy learners that stays faithful to the paper's actual object.",
                    "Something teachable in a course with short transfer distance, not a generic summary or generalized life principle.",
                    "Prefer a workflow, decision point, failure mode, comparison structure, mechanism, or evidence-backed data point that the learner can picture directly.",
                    "Grounded in the provided section_ids, and include figure_ids when a figure materially supports or anchors the idea.",
                ],
                "card_shape_rules": [
                    "Write all card-facing strings in Simplified Chinese.",
                    "Keep the original paper evidence as the primary body material by selecting the right section_ids rather than rewriting the paper.",
                    "Use evidence_analysis to add only very short 1-2 sentence explanations of why each cited part matters for teaching.",
                    "Prefer a strong specific title over a paper-topic label.",
                    "Primary evidence should come from mechanism/model/method/result/failure sections; abstract or framing sections can only be supporting context.",
                    "If body evidence is available in sections, do not use abstract or front matter as the only primary evidence.",
                    "If you cannot name the paper-specific object being taught, emit no card.",
                    "If you cannot articulate a learner-facing candidate, emit no card.",
                    "Do not convert technical evidence into broad management slogans, motivational advice, or generic productivity principles.",
                    "If a candidate requires multiple layers of explanation before a non-technical learner can use it, reject it in extraction.",
                    "course_transformation will later name how the source object is presented in the course; do not pre-abstract the object here.",
                ],
                "judgement_boundary_hints": [
                    "Look for belief-gap, counterintuitive, tacit-to-explicit, or highly actionable insight candidates.",
                    "Prefer ideas with commercial relevance and presentation usefulness, not only academic validity.",
                    "Also prefer direct-transfer source patterns even when they are not maximally dramatic, as long as a non-technical learner can understand and reuse them quickly.",
                    "A mere taxonomy recap, literature background, or weak-transfer technical detail should be rejected here.",
                ],
                "should_be_rejected_here": [
                    "Background theory or literature review context.",
                    "Classification or taxonomy recap without a sharp insight.",
                    "Generic paper summary or conclusion.",
                    "Technical detail that is academically valid but too far from course use.",
                    "A generalized principle whose wording drifts above the source object described in the paper.",
                    "Policy or management recommendation aimed at the wrong audience.",
                    "Old low-hanging-fruit claims that are already obvious to the target learner.",
                    "Claims that only describe what this paper is generally about without naming its concrete contribution object.",
                ],
            },
            "output_schema": {
                "cards": [
                    {
                        "title": "中文卡片标题",
                        "primary_section_ids": ["section_id"],
                        "supporting_section_ids": ["section_id"],
                        "section_ids": ["section_id"],
                        "figure_ids": ["figure_id"],
                        "granularity_level": "framework|subpattern|detail",
                        "claim_type": "mechanism|model|method|result|failure_mode|framework|data_finding|other",
                        "paper_specific_object": "这篇论文的具体对象（模型/机制/方法/结果）",
                        "body_grounding_reason": "一句话说明为什么这张卡是由正文证据支撑的",
                        "evidence_level": "strong|medium|weak",
                        "possible_duplicate_signature": "用于同论文去重的短签名",
                        "draft_body": "基于证据的中文简短说明",
                        "evidence_analysis": [
                            {
                                "section_id": "section_id",
                                "analysis": "1-2句中文说明，解释这段证据为什么值得教",
                            }
                        ],
                    }
                ],
                "excluded_content": [
                    {
                        "label": "中文排除项名称",
                        "section_ids": ["section_id"],
                        "exclusion_type": "background|summary|weak_transfer|wrong_audience|replaced_by_stronger_card|insufficient_evidence|other",
                        "reason": "中文简短理由，说明为什么这部分不应该出卡",
                    }
                ],
            },
        }

    def _build_judgement_prompt_payload(
        self,
        *,
        topic_name: str,
        paper_title: str,
        prompt_candidates: list[dict],
        calibration_examples: list[dict],
        stage_examples: list[dict],
        calibration_set_name: str,
    ) -> dict[str, Any]:
        return {
            "topic": topic_name,
            "paper_title": paper_title,
            "stage": "candidate_judgement",
            "prompt_version": JUDGEMENT_PROMPT_VERSION,
            "output_language": "zh-CN",
            "active_calibration_set": calibration_set_name,
            "shared_policy": self._shared_prompt_policy(),
            "stage_spec": self._stage_spec("candidate_judgement"),
            "stage_examples": stage_examples,
            "calibration_examples": calibration_examples,
            "candidates": prompt_candidates,
            "judgement_rules": {
                "must_be_true_for_a_card": [
                    "The candidate expresses a real learner-facing cognitive shift instead of a paper takeaway.",
                    "The candidate names the paper-specific object (model/mechanism/method/result/failure/framework/data finding), not only topic framing.",
                    "The idea can be named as a concrete course object, framework, pattern, story, or evidence-backed talking point.",
                    "The transfer distance to course use is short enough to teach.",
                    "The evidence strength matches the claim strength.",
                    "At least one of these qualities is present: belief-gap, counterintuitive, tacit-to-explicit, or highly actionable.",
                    "The card stays close to the paper's source object instead of drifting into a broader principle.",
                    "A non-technical learner should be able to understand what to picture or do after one slide of explanation.",
                ],
                "business_and_teaching_rules": [
                    "Judge from the learner and course-design perspective, not only from academic importance.",
                    "Prefer ideas that would make sense on one slide with one strong line and, when available, one supporting figure.",
                    "If you cannot say what this becomes in the course, the card should not pass as green.",
                    "Explicitly state whether primary evidence is body evidence or merely abstract/front-matter framing.",
                    "course_transformation should name the course presentation form of the source object, not rewrite it into a more generic doctrine.",
                    "If the strongest version of the card sounds like a principle that could have been written without this paper, downgrade or reject it.",
                    "If linked_figures materially support the candidate, keep the figure attachment instead of silently dropping it.",
                ],
                "evidence_translation_rules": [
                    "For every evidence quote, provide a complete Simplified Chinese translation in quote_zh.",
                    "Translate the full quoted evidence, not only the claim-relevant fragment.",
                    "Do not summarize, compress, paraphrase away caveats, or keep only the punchline.",
                    "If the quote contains multiple sentences, lists, or numbered parts, translate all of them in order.",
                    "Preserve informational scope, caveats, enumerations, and logical structure from the English quote.",
                ],
                "color_rules": {
                    "green": "Clear learner belief conflict or tacit-to-explicit shift, evidence is strong, and course use is obvious.",
                    "yellow": "Potentially valuable but boundary-like: learner prior is uncertain, evidence is partial, or course use needs human judgement.",
                    "red": "Mostly aligned with common knowledge, generic summary, wrong audience, or too indirect to teach.",
                },
                "must_be_downgraded_or_rejected": [
                    "The candidate is merely background, summary, taxonomy, or weak-transfer detail.",
                    "The audience fit is wrong.",
                    "The idea is academically valid but not teachable for the target learner.",
                    "The claim feels outdated or already obvious to the target learner.",
                    "The candidate is a framing variant that overlaps with a sibling candidate from the same paper and evidence.",
                    "The card relies on principle drift: it sounds smoother after being generalized away from the source evidence.",
                ],
            },
            "required_judgement_questions": [
                "What is the paper's unique contribution object here?",
                "Is this claim grounded in body evidence or only paper framing?",
                "Does this candidate remain distinct from sibling candidates within the same paper/topic?",
                "Is the evidence strength proportional to the claim strength?",
                "Would a non-technical learner still understand the source object without extra technical unpacking?",
                "Has the course naming stayed close to the source object instead of abstracting it upward?",
            ],
            "output_schema": {
                "cards": [
                    {
                        "candidate_index": 0,
                        "title": "最终中文卡片标题",
                        "course_transformation": "它在课程里变成什么（中文，最好是可命名对象）",
                        "teachable_one_liner": "老师可以直接说出来的一句中文",
                        "draft_body": "基于证据的中文简短说明",
                        "evidence_localization": [
                            {
                                "section_id": "section_id",
                                "quote_zh": "对应证据原文的完整中文译文，必须覆盖整段证据，不得压缩、提炼或只保留其中一句",
                            }
                        ],
                        "judgement": {
                            "color": "green|yellow|red",
                            "reason": "中文简短理由，说明为什么落在这个颜色边界",
                        },
                        "grounding_quality": "strong|medium|weak",
                        "paper_specific_object": "可选补充，若提取阶段遗漏可在此补充",
                        "claim_type": "mechanism|model|method|result|failure_mode|framework|data_finding|other",
                        "evidence_level": "strong|medium|weak",
                        "body_grounding_reason": "中文，说明正文证据支撑程度",
                    }
                ],
            },
        }

    def _build_prompt_sections(self, sections: list[dict]) -> list[dict]:
        return [
            {
                "section_id": section["id"],
                "section_title": section.get("section_title", ""),
                "page_number": section["page_number"],
                "text": section["paragraph_text"],
                "section_kind": section.get("section_kind", "other"),
                "section_label": section.get("section_label", section.get("section_title", "")),
                "is_front_matter": bool(section.get("is_front_matter", False)),
                "is_abstract": bool(section.get("is_abstract", False)),
                "is_body": bool(section.get("is_body", False)),
                "body_role": section.get("body_role", ""),
                "role_hint": (
                    "primary"
                    if bool(section.get("is_body", False))
                    else ("context" if bool(section.get("is_abstract", False)) else "supporting")
                ),
                "selection_score": section.get("selection_score", 0.0),
            }
            for section in sections[:MAX_PROMPT_SECTIONS]
        ]

    def _build_prompt_figures(self, figures: list[dict]) -> list[dict]:
        return [
            {
                "figure_id": figure["id"],
                "figure_label": figure.get("figure_label", ""),
                "caption": figure.get("caption", ""),
                "storage_path": figure.get("storage_path", ""),
                "asset_status": figure.get("asset_status", "metadata_only"),
                "has_asset": figure.get("asset_status", "") == "validated_local_asset",
                "linked_section_ids": figure.get("linked_section_ids", []),
            }
            for figure in figures[:MAX_PROMPT_FIGURES]
        ]

    def _normalize_extraction_output(
        self,
        payload: dict[str, Any],
        sections: list[dict],
        figures: list[dict],
    ) -> dict[str, list[dict]]:
        raw_cards = payload.get("cards", [])
        raw_excluded = payload.get("excluded_content", [])
        if not isinstance(raw_cards, list):
            raise LLMGenerationError("LLM card payload must contain a list named 'cards'")
        if not isinstance(raw_excluded, list):
            raise LLMGenerationError("LLM card payload must contain a list named 'excluded_content'")

        section_map = {section["id"]: section for section in sections}
        figure_id_set = {figure["id"] for figure in figures}
        normalized = []
        for raw_card in raw_cards[:MAX_EXTRACTED_CARDS]:
            if not isinstance(raw_card, dict):
                continue
            title = str(raw_card.get("title", "")).strip()
            draft_body = str(raw_card.get("draft_body", "")).strip()
            granularity_level = str(raw_card.get("granularity_level", "subpattern")).strip().lower()
            primary_section_ids = raw_card.get("primary_section_ids", [])
            supporting_section_ids = raw_card.get("supporting_section_ids", [])
            section_ids = raw_card.get("section_ids", [])
            if not primary_section_ids and isinstance(section_ids, list):
                primary_section_ids = section_ids[:1]
            if not section_ids and isinstance(primary_section_ids, list):
                section_ids = list(primary_section_ids) + [
                    sid for sid in supporting_section_ids if sid not in set(primary_section_ids)
                ]
            raw_figure_ids = raw_card.get("figure_ids", [])
            raw_evidence_analysis = raw_card.get("evidence_analysis", [])
            claim_type = str(raw_card.get("claim_type", "other")).strip().lower()
            paper_specific_object = str(raw_card.get("paper_specific_object", "")).strip()
            body_grounding_reason = str(raw_card.get("body_grounding_reason", "")).strip()
            evidence_level = str(raw_card.get("evidence_level", "medium")).strip().lower()
            possible_duplicate_signature = str(raw_card.get("possible_duplicate_signature", "")).strip()

            if (
                not isinstance(section_ids, list)
                or not isinstance(raw_evidence_analysis, list)
                or not isinstance(raw_figure_ids, list)
                or not isinstance(primary_section_ids, list)
                or not isinstance(supporting_section_ids, list)
            ):
                continue
            analysis_by_section = {}
            for item in raw_evidence_analysis:
                if not isinstance(item, dict):
                    continue
                analysis_section_id = str(item.get("section_id", "")).strip()
                analysis_text = str(item.get("analysis", "")).strip()
                if analysis_section_id and analysis_text:
                    analysis_by_section[analysis_section_id] = analysis_text
            evidence = []
            for section_id in section_ids:
                section = section_map.get(str(section_id))
                if not section:
                    continue
                analysis_text = analysis_by_section.get(section["id"], "")
                evidence.append(
                    {
                        "section_id": section["id"],
                        "quote": extract_relevant_evidence_excerpt(
                            section["paragraph_text"],
                            hints=[title, paper_specific_object, analysis_text],
                        ),
                        "quote_zh": "",
                        "page_number": section["page_number"],
                        "analysis": analysis_text,
                    }
                )

            if not title or not evidence:
                continue
            figure_ids = [
                str(figure_id).strip()
                for figure_id in raw_figure_ids
                if str(figure_id).strip() in figure_id_set
            ]
            if not is_readable_text(title):
                continue
            if not is_readable_text(draft_body):
                continue
            if not all(is_readable_text(item["quote"]) for item in evidence):
                continue
            if not all((not item["analysis"]) or is_readable_text(item["analysis"]) for item in evidence):
                continue
            if claim_type not in {"mechanism", "model", "method", "result", "failure_mode", "framework", "data_finding", "other"}:
                claim_type = "other"
            if evidence_level not in {"strong", "medium", "weak"}:
                evidence_level = "medium"

            normalized.append(
                {
                    "title": title,
                    "granularity_level": granularity_level if granularity_level in {"framework", "subpattern", "detail"} else "subpattern",
                    "draft_body": draft_body,
                    "evidence": evidence,
                    "figure_ids": figure_ids,
                    "status": "candidate",
                    "primary_section_ids": [sid for sid in [str(item).strip() for item in primary_section_ids] if sid in section_map],
                    "supporting_section_ids": [sid for sid in [str(item).strip() for item in supporting_section_ids] if sid in section_map],
                    "claim_type": claim_type,
                    "paper_specific_object": paper_specific_object,
                    "body_grounding_reason": body_grounding_reason,
                    "evidence_level": evidence_level,
                    "possible_duplicate_signature": possible_duplicate_signature,
                }
            )

        normalized_excluded = []
        for raw_item in raw_excluded[:8]:
            if not isinstance(raw_item, dict):
                continue
            label = str(raw_item.get("label", "")).strip()
            exclusion_type = str(raw_item.get("exclusion_type", "other")).strip().lower()
            reason = str(raw_item.get("reason", "")).strip()
            section_ids = raw_item.get("section_ids", [])
            if not label or not reason or not isinstance(section_ids, list):
                continue
            valid_section_ids = [str(section_id) for section_id in section_ids if str(section_id) in section_map]
            if not valid_section_ids:
                continue
            if not is_readable_text(label) or not is_readable_text(reason):
                continue
            if exclusion_type not in {
                "background",
                "summary",
                "weak_transfer",
                "wrong_audience",
                "replaced_by_stronger_card",
                "insufficient_evidence",
                "other",
            }:
                exclusion_type = "other"
            normalized_excluded.append(
                {
                    "label": label,
                    "exclusion_type": exclusion_type,
                    "reason": reason,
                    "section_ids": valid_section_ids,
                }
            )

        return {"cards": normalized, "excluded_content": normalized_excluded}

    def _normalize_judged_cards(self, payload: dict[str, Any], extracted_cards: list[dict]) -> list[dict]:
        raw_cards = payload.get("cards", [])
        if not isinstance(raw_cards, list):
            raise LLMGenerationError("LLM judgement payload must contain a list named 'cards'")

        normalized = []
        for raw_card in raw_cards[: len(extracted_cards)]:
            if not isinstance(raw_card, dict):
                continue
            try:
                candidate_index = int(raw_card.get("candidate_index"))
            except (TypeError, ValueError):
                continue
            if candidate_index < 0 or candidate_index >= len(extracted_cards):
                continue

            extracted = extracted_cards[candidate_index]
            title = str(raw_card.get("title", extracted["title"])).strip()
            course_transformation = str(raw_card.get("course_transformation", "")).strip()
            teachable_one_liner = str(raw_card.get("teachable_one_liner", "")).strip()
            draft_body = str(raw_card.get("draft_body", extracted["draft_body"])).strip()
            raw_evidence_localization = raw_card.get("evidence_localization", [])
            judgement = raw_card.get("judgement", {})
            if not isinstance(raw_evidence_localization, list):
                continue
            if not all(
                [
                    is_readable_text(title),
                    is_readable_text(course_transformation),
                    is_readable_text(teachable_one_liner),
                    is_readable_text(draft_body),
                ]
            ):
                continue

            color = str(judgement.get("color", "yellow")).strip().lower()
            if color not in {"green", "yellow", "red"}:
                color = "yellow"
            reason = str(judgement.get("reason", "")).strip() or "LLM judgement did not provide a reason."
            if not is_readable_text(reason):
                continue

            quote_zh_by_section = {}
            for item in raw_evidence_localization:
                if not isinstance(item, dict):
                    continue
                section_id = str(item.get("section_id", "")).strip()
                quote_zh = str(item.get("quote_zh", "")).strip()
                if section_id and is_readable_text(quote_zh):
                    quote_zh_by_section[section_id] = quote_zh

            primary_section_ids = {
                str(section_id).strip()
                for section_id in extracted.get("primary_section_ids", [])
                if str(section_id).strip()
            }
            evidence = []
            for evidence_item in extracted["evidence"]:
                section_id = evidence_item["section_id"]
                quote_zh = quote_zh_by_section.get(section_id, "")
                if quote_zh:
                    if not looks_like_complete_translation(evidence_item["quote"], quote_zh):
                        evidence = []
                        break
                elif section_id in primary_section_ids:
                    evidence = []
                    break
                evidence.append(
                    {
                        **evidence_item,
                        "quote_zh": quote_zh,
                    }
                )
            if not evidence:
                continue
            claim_type = str(raw_card.get("claim_type", extracted.get("claim_type", "other"))).strip().lower() or "other"
            if claim_type not in {"mechanism", "model", "method", "result", "failure_mode", "framework", "data_finding", "other"}:
                claim_type = "other"
            evidence_level = str(raw_card.get("evidence_level", extracted.get("evidence_level", "medium"))).strip().lower() or "medium"
            if evidence_level not in {"strong", "medium", "weak"}:
                evidence_level = "medium"
            grounding_quality = str(raw_card.get("grounding_quality", "")).strip().lower() or evidence_level
            if grounding_quality not in {"strong", "medium", "weak"}:
                grounding_quality = "medium"

            normalized.append(
                {
                    "title": title,
                    "granularity_level": extracted["granularity_level"],
                    "course_transformation": course_transformation,
                    "teachable_one_liner": teachable_one_liner,
                    "draft_body": draft_body,
                    "evidence": evidence,
                    "figure_ids": extracted.get("figure_ids", []),
                    "status": extracted.get("status", "candidate"),
                    "primary_section_ids": extracted.get("primary_section_ids", []),
                    "supporting_section_ids": extracted.get("supporting_section_ids", []),
                    "paper_specific_object": str(raw_card.get("paper_specific_object", extracted.get("paper_specific_object", ""))).strip(),
                    "claim_type": claim_type,
                    "evidence_level": evidence_level,
                    "body_grounding_reason": str(raw_card.get("body_grounding_reason", extracted.get("body_grounding_reason", ""))).strip(),
                    "grounding_quality": grounding_quality,
                    "possible_duplicate_signature": extracted.get("possible_duplicate_signature", ""),
                    "judgement": {
                        "color": color,
                        "reason": reason,
                        "model_version": self.client.model,
                        "prompt_version": JUDGEMENT_PROMPT_VERSION,
                        "rubric_version": CARD_RUBRIC_VERSION,
                    },
                }
            )
        return normalized

    def _normalize_understanding_payload(
        self,
        payload: dict[str, Any],
        sections: list[dict],
        figures: list[dict],
    ) -> dict[str, Any]:
        section_id_set = {section.get("id", "") for section in sections}
        figure_id_set = {figure.get("id", "") for figure in figures}
        figure_ids_by_section: dict[str, list[str]] = {}
        for figure in figures:
            figure_id = str(figure.get("id", "")).strip()
            if not figure_id:
                continue
            for section_id in figure.get("linked_section_ids", []) or []:
                normalized_section_id = str(section_id).strip()
                if normalized_section_id:
                    figure_ids_by_section.setdefault(normalized_section_id, []).append(figure_id)
        raw_objects = payload.get("global_contribution_objects", [])
        raw_graph = payload.get("contribution_graph", [])
        raw_hints = payload.get("candidate_level_hints", {})
        if not isinstance(raw_objects, list):
            raw_objects = []
        if not isinstance(raw_graph, list):
            raw_graph = []
        if not isinstance(raw_hints, dict):
            raw_hints = {}
        objects = []
        for index, item in enumerate(raw_objects[:8], start=1):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("id", "")).strip() or f"obj_{index}"
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            if label.lower() in {"markdown extraction", "html snapshot"}:
                continue
            level_hint = str(item.get("level_hint", "detail")).strip().lower()
            if level_hint not in {"overall", "local", "detail"}:
                level_hint = "detail"
            object_type = str(item.get("object_type", "other")).strip().lower()
            if object_type not in {"mechanism", "model", "method", "result", "framework", "data_finding", "other"}:
                object_type = "other"
            evidence_section_ids = [sid for sid in [str(s).strip() for s in item.get("evidence_section_ids", [])] if sid in section_id_set]
            evidence_figure_ids = [fid for fid in [str(f).strip() for f in item.get("evidence_figure_ids", [])] if fid in figure_id_set]
            if not evidence_figure_ids:
                inferred_figure_ids = [
                    figure_id
                    for section_id in evidence_section_ids
                    for figure_id in figure_ids_by_section.get(section_id, [])
                    if figure_id in figure_id_set
                ]
                evidence_figure_ids = list(dict.fromkeys(inferred_figure_ids))[:2]
            try:
                importance_score = float(item.get("importance_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                importance_score = 0.0
            if not evidence_section_ids:
                continue
            objects.append(
                {
                    "id": object_id,
                    "label": label,
                    "object_type": object_type,
                    "level_hint": level_hint,
                    "evidence_section_ids": evidence_section_ids,
                    "evidence_figure_ids": evidence_figure_ids,
                    "summary": str(item.get("summary", "")).strip(),
                    "importance_score": round(max(0.0, min(1.0, importance_score)), 4),
                }
            )
        if not objects:
            return {
                "global_contribution_objects": [],
                "contribution_graph": [],
                "candidate_level_hints": {},
                "evidence_index": {},
            }
        object_ids = {item["id"] for item in objects}
        graph = []
        for edge in raw_graph[:20]:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("from", "")).strip()
            target = str(edge.get("to", "")).strip()
            relation = str(edge.get("relation", "")).strip().lower() or "supports"
            if source in object_ids and target in object_ids and source != target:
                graph.append({"from": source, "to": target, "relation": relation})
        hints = {}
        for object_id, level in raw_hints.items():
            if object_id not in object_ids:
                continue
            normalized_level = str(level).strip().lower()
            if normalized_level not in {"overall", "local", "detail"}:
                continue
            hints[object_id] = normalized_level
        for item in objects:
            hints.setdefault(item["id"], item["level_hint"])
        evidence_index = {
            item["id"]: {"section_ids": item["evidence_section_ids"], "figure_ids": item["evidence_figure_ids"]}
            for item in objects
        }
        return {
            "global_contribution_objects": objects,
            "contribution_graph": graph,
            "candidate_level_hints": hints,
            "evidence_index": evidence_index,
        }

    def _normalize_card_plan_payload(
        self,
        payload: dict[str, Any],
        understanding: dict[str, Any],
        *,
        max_cards: int,
    ) -> dict[str, Any]:
        objects = understanding.get("global_contribution_objects", [])
        object_map = {item.get("id"): item for item in objects if isinstance(item, dict)}
        raw_cards = payload.get("planned_cards", [])
        if not isinstance(raw_cards, list):
            raw_cards = []
        planned_cards = []
        produce_count = 0
        unlimited_produce = max_cards <= 0
        for item in raw_cards[: max(len(objects), max_cards + 2)]:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("target_object_id", "")).strip()
            if object_id not in object_map:
                continue
            level = str(item.get("level", object_map[object_id].get("level_hint", "detail"))).strip().lower()
            if level not in {"overall", "local", "detail"}:
                level = "detail"
            must_have = [sid for sid in [str(s).strip() for s in item.get("must_have_evidence_ids", [])] if sid in set(object_map[object_id].get("evidence_section_ids", []))]
            optional = [sid for sid in [str(s).strip() for s in item.get("optional_supporting_ids", [])] if sid and sid not in must_have]
            object_figure_ids = set(object_map[object_id].get("evidence_figure_ids", []))
            must_have_figure_ids = [
                fid
                for fid in [str(f).strip() for f in item.get("must_have_figure_ids", [])]
                if fid in object_figure_ids
            ]
            optional_supporting_figure_ids = [
                fid
                for fid in [str(f).strip() for f in item.get("optional_supporting_figure_ids", [])]
                if fid in object_figure_ids and fid not in must_have_figure_ids
            ]
            if object_figure_ids and not must_have_figure_ids and not optional_supporting_figure_ids:
                must_have_figure_ids = list(object_figure_ids)[:1]
            disposition = str(item.get("disposition", "exclude")).strip().lower()
            if disposition not in {"produce", "exclude"}:
                disposition = "exclude"
            if disposition == "produce":
                if ((not unlimited_produce) and produce_count >= max_cards) or not must_have:
                    disposition = "exclude"
                else:
                    produce_count += 1
            planned_cards.append(
                {
                    "plan_id": str(item.get("plan_id", "")).strip() or f"plan_{object_id}",
                    "level": level,
                    "target_object_id": object_id,
                    "target_object_label": str(item.get("target_object_label", object_map[object_id].get("label", ""))).strip(),
                    "why_valuable_for_course": str(item.get("why_valuable_for_course", "")).strip(),
                    "must_have_evidence_ids": must_have,
                    "optional_supporting_ids": optional,
                    "must_have_figure_ids": must_have_figure_ids,
                    "optional_supporting_figure_ids": optional_supporting_figure_ids,
                    "disposition": disposition,
                    "disposition_reason": str(item.get("disposition_reason", "")).strip(),
                }
            )
        if not planned_cards:
            fallback_unlimited = max_cards <= 0
            for index, obj in enumerate(sorted(objects, key=lambda x: float(x.get("importance_score", 0.0)), reverse=True), start=1):
                produce = (fallback_unlimited or index <= max_cards) and bool(obj.get("evidence_section_ids"))
                planned_cards.append(
                    {
                        "plan_id": f"plan_{obj.get('id', index)}",
                        "level": obj.get("level_hint", "detail"),
                        "target_object_id": obj.get("id", f"obj_{index}"),
                        "target_object_label": obj.get("label", ""),
                        "why_valuable_for_course": "",
                        "must_have_evidence_ids": list(obj.get("evidence_section_ids", []))[:2],
                        "optional_supporting_ids": [],
                        "must_have_figure_ids": list(obj.get("evidence_figure_ids", []))[:1],
                        "optional_supporting_figure_ids": [],
                        "disposition": "produce" if produce else "exclude",
                        "disposition_reason": "" if produce else "Fallback planner excluded due to slot/evidence constraints.",
                    }
                )
        produced_cards = [item for item in planned_cards if item["disposition"] == "produce"]
        has_figure_backed_produce = any(item.get("must_have_figure_ids") for item in produced_cards)
        if produced_cards and not has_figure_backed_produce:
            figure_candidates = [
                item
                for item in planned_cards
                if item.get("must_have_figure_ids")
                and item.get("target_object_id") in object_map
                and object_map[item["target_object_id"]].get("evidence_section_ids")
            ]
            if figure_candidates:
                best_figure_candidate = max(
                    figure_candidates,
                    key=lambda item: float(object_map[item["target_object_id"]].get("importance_score", 0.0)),
                )
                weakest_non_figure_produce = min(
                    produced_cards,
                    key=lambda item: (
                        1 if item.get("must_have_figure_ids") else 0,
                        float(object_map[item["target_object_id"]].get("importance_score", 0.0)),
                    ),
                )
                if best_figure_candidate is not weakest_non_figure_produce:
                    weakest_non_figure_produce["disposition"] = "exclude"
                    weakest_non_figure_produce["disposition_reason"] = (
                        "Rebalanced planner coverage to keep at least one figure-backed card candidate."
                    )
                    best_figure_candidate["disposition"] = "produce"
                    best_figure_candidate["disposition_reason"] = ""
        return {
            "planned_cards": planned_cards,
            "coverage_report": {
                "produce": sum(1 for item in planned_cards if item["disposition"] == "produce"),
                "exclude": sum(1 for item in planned_cards if item["disposition"] != "produce"),
                "overall": sum(1 for item in planned_cards if item["level"] == "overall"),
                "local": sum(1 for item in planned_cards if item["level"] == "local"),
                "detail": sum(1 for item in planned_cards if item["level"] == "detail"),
            },
        }

    def _select_calibration_examples(
        self,
        calibration_examples: list[dict],
        topic_name: str,
        limit: int = MAX_CALIBRATION_EXAMPLES,
    ) -> list[dict]:
        if not calibration_examples:
            return []
        target = topic_name.strip().lower()
        ranked: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
        for example in calibration_examples:
            audience = str(example.get("audience", "")).strip()
            tags = [str(tag).strip() for tag in example.get("tags", []) if str(tag).strip()]
            compact = {
                "example_type": example.get("example_type", ""),
                "topic_name": example.get("topic_name", ""),
                "audience": audience,
                "title": example.get("title", ""),
                "source_text": example.get("source_text", ""),
                "evidence": example.get("evidence", [])[:2],
                "expected_cards": example.get("expected_cards", []),
                "expected_exclusions": example.get("expected_exclusions", []),
                "rationale": example.get("rationale", ""),
                "tags": tags,
            }
            lowered_tags = {tag.lower() for tag in tags}
            lowered_audience = audience.lower()
            same_topic_score = 1 if str(example.get("topic_name", "")).strip().lower() == target else 0
            audience_score = 1 if any(token in lowered_audience for token in ["ai literacy", "non-technical", "operator", "learner"]) else 0
            transfer_score = 1 if lowered_tags.intersection({"direct-transfer", "nontechnical-audience", "principle-drift-negative"}) else 0
            boundary_score = 1 if compact["example_type"] == "boundary" else 0
            ranked.append(((same_topic_score, audience_score, transfer_score, boundary_score), compact))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:limit]]

    def _build_client(self, settings: Settings) -> Optional[BaseLLMClient]:
        if settings.llm_mode == "disabled":
            return None
        base_url = settings.llm_base_url or provider_default_base_url(settings.llm_mode)
        if not base_url or not settings.llm_api_key or not settings.llm_model:
            return None
        if settings.llm_mode == "openai_compatible":
            return OpenAICompatibleLLMClient(
                base_url=base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        if settings.llm_mode == "anthropic":
            return AnthropicLLMClient(
                base_url=base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                anthropic_version=settings.anthropic_version,
            )
        if settings.llm_mode == "gemini":
            return GeminiLLMClient(
                base_url=base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                api_version=settings.gemini_api_version,
            )
        return None
