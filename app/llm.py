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

SHARED_PROMPT_POLICY_VERSION = "llm-shared-policy-v6-causal-reconstruction"
EXTRACTION_PROMPT_VERSION = "llm-card-extract-v9-causal-reconstruction-zh"
JUDGEMENT_PROMPT_VERSION = "llm-card-judge-v10-causal-reconstruction-zh"
CARD_RUBRIC_VERSION = "llm-card-rubric-v8-causal-reconstruction"
UNDERSTANDING_PROMPT_VERSION = "llm-paper-understanding-v6-causal-reconstruction"
CARD_PLAN_PROMPT_VERSION = "llm-card-plan-v8-causal-reconstruction"
RESEARCH_PLAN_PROMPT_VERSION = "llm-research-plan-v1-dual-mode"
CLAIM_EVIDENCE_PROMPT_VERSION = "llm-claim-evidence-v4-strict-evidence-translation"
PAPER_QA_PROMPT_VERSION = "llm-paper-qa-v1-grounded"
PREFERENCE_MEMORY_PROMPT_VERSION = "llm-preference-memory-v1"
MAX_PROMPT_SECTIONS = 14
MAX_PROMPT_FIGURES = 4
MAX_CALIBRATION_EXAMPLES = 6
MAX_EXTRACTED_CARDS = 8
PROMPT_VERSION_RECORDS = [
    {
        "version": UNDERSTANDING_PROMPT_VERSION,
        "stage": "paper_understanding",
        "summary": "Understanding stage that decides paper worthiness by checking whether the paper contains a real causal-reconstruction aha before any course packaging.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "uses_figures": True,
            "uses_stage_examples": False,
            "stage_contract": "conditions_1_to_3_then_identify_objects",
            "prefers_direct_transfer_patterns": True,
            "reportable_aha_gate": True,
            "ontology_primary_axis": "causal_reconstruction",
        },
    },
    {
        "version": CARD_PLAN_PROMPT_VERSION,
        "stage": "card_planning",
        "summary": "Planning stage that inherits the causal-reconstruction verdict and decides whether the surviving aha can be operationalized into the course.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "uses_stage_examples": True,
            "stage_contract": "condition_4_operationalization_planning",
            "max_cards_hint_is_soft": True,
            "defaults_to_single_strong_aha": True,
            "ontology_primary_axis": "course_operationalizability_after_true_aha",
        },
    },
    {
        "version": EXTRACTION_PROMPT_VERSION,
        "stage": "candidate_extraction",
        "summary": "Chinese extraction that assembles quote-first cards only for planned causal-reconstruction objects and logs explicit exclusions for everything else.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "language": "zh-CN learner-facing output",
            "uses_figures": True,
            "uses_stage_examples": True,
            "enforces_primary_vs_supporting_evidence": True,
            "requires_paper_specific_object": True,
            "prefers_direct_transfer_patterns": True,
            "suppresses_weaker_same_paper_rephrasings": True,
            "ontology_primary_axis": "evidence_for_causal_reconstruction",
            "max_sections": MAX_PROMPT_SECTIONS,
            "max_figures": MAX_PROMPT_FIGURES,
        },
    },
    {
        "version": JUDGEMENT_PROMPT_VERSION,
        "stage": "candidate_judgement",
        "summary": "Final card judgement with causal-reconstruction-first reasoning, plan-slot fidelity, full evidence translation, and green/yellow/red decisions.",
        "details": {
            "shared_policy_version": SHARED_PROMPT_POLICY_VERSION,
            "language": "zh-CN learner-facing output",
            "uses_stage_examples": True,
            "requires_full_evidence_translation": True,
            "requires_grounding_decision": True,
            "requires_duplicate_distinction": True,
            "checks_source_object_fidelity": True,
            "checks_reportable_aha_strength": True,
            "ontology_primary_axis": "old_model_to_new_model_then_course_use",
            "max_calibration_examples": MAX_CALIBRATION_EXAMPLES,
        },
    },
    {
        "version": RESEARCH_PLAN_PROMPT_VERSION,
        "stage": "research_planning",
        "summary": "Dual-mode research planning that classifies the task, drafts a structured search plan, and keeps the output directly executable.",
        "details": {
            "supports_dual_mode": True,
            "modes": ["aha_exploration", "claim_evidence"],
            "requires_confirmable_plan": True,
        },
    },
    {
        "version": CLAIM_EVIDENCE_PROMPT_VERSION,
        "stage": "claim_evidence_generation",
        "summary": "Grounded claim-evidence generation that turns one parsed paper into evidence matrix items while rejecting cross-domain analogies that are not direct evidence for the claim context.",
        "details": {
            "output_type": "matrix_item",
            "requires_section_grounding": True,
            "surfaces_limitations": True,
            "surfaces_contradictions": True,
            "rejects_cross_domain_analogies": True,
        },
    },
    {
        "version": PAPER_QA_PROMPT_VERSION,
        "stage": "paper_qa",
        "summary": "Single-paper grounded question answering over retrieved sections and linked figures.",
        "details": {
            "grounded_only": True,
            "single_paper_only": True,
            "requires_section_ids": True,
        },
    },
    {
        "version": PREFERENCE_MEMORY_PROMPT_VERSION,
        "stage": "preference_memory",
        "summary": "Preference-memory distillation from explicit accept/reject/comment signals into a human-confirmable memory draft.",
        "details": {
            "human_confirmed_loop": True,
            "no_automatic_activation": True,
            "source_of_truth": "review decisions and comments",
        },
    },
]

RUBRIC_VERSION_RECORDS = [
    {
        "version": CARD_RUBRIC_VERSION,
        "name": "card_judgement_rubric",
        "summary": "Rubric for deciding whether a planned concrete paper object is a real causal-reconstruction aha and, only then, a usable course card.",
        "details": {
            "green": "clear causal reconstruction for an active learner prior, credible evidence, and direct course transfer with low technical overhang",
            "yellow": "boundary-case insight that may have partial causal reconstruction, uncertain active prior, or weaker operationalization and still needs reviewer judgment",
            "red": "information gain, summary, background, weak-transfer, duplicate-framing, or technical-but-not-reportable content that should not become a card",
            "hard_gates": [
                "paper_specific_object must be present",
                "body evidence must support the claim whenever body sections exist",
                "framing-only variants should be suppressed",
                "course naming must stay close to the source object",
                "technical specificity alone is not enough without a real causal reconstruction",
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


def normalize_evidence_paragraph(text: str) -> str:
    return str(text or "").strip()


def normalize_paper_relevance_verdict(value: Any, default: str = "on_topic") -> str:
    verdict = str(value or "").strip().lower() or default
    if verdict not in {"on_topic", "borderline_reject", "off_topic_hard"}:
        verdict = default
    return verdict


def normalize_relevance_failure_type(value: Any, verdict: str) -> str:
    failure_type = str(value or "").strip().lower()
    allowed = {
        "pure_technical_mismatch",
        "cannot_name_course_object",
        "long_transfer_distance",
        "taxonomy_not_insight",
        "weak_method_or_data",
        "low_hanging_fruit",
        "topic_word_overlap_only",
        "other",
    }
    if verdict == "on_topic":
        return ""
    if failure_type not in allowed:
        return "other"
    return failure_type


def looks_like_complete_translation(source_text: str, translated_text: str) -> bool:
    source = source_text.strip()
    translated = translated_text.strip()
    if not is_readable_text(source) or not is_readable_text(translated):
        return False

    normalized_translated = re.sub(r"\s+", "", translated)
    normalized_source = re.sub(r"\s+", " ", source.lower())
    meta_summary_prefixes = (
        "文中提到",
        "文中指出",
        "文章提到",
        "文章指出",
        "本文提到",
        "本文指出",
        "该文提到",
        "该文指出",
        "作者提到",
        "作者指出",
        "研究提到",
        "研究指出",
        "研究发现",
        "该研究提到",
        "该研究指出",
        "该研究发现",
        "这段话提到",
        "这段证据提到",
    )
    source_has_self_reference = normalized_source.startswith(
        (
            "this study",
            "the study",
            "this paper",
            "the paper",
            "this article",
            "the article",
            "authors",
            "the authors",
            "we ",
            "our ",
        )
    )
    if any(normalized_translated.startswith(prefix) for prefix in meta_summary_prefixes) and not source_has_self_reference:
        return False

    source_length = compact_text_length(source)
    translated_length = compact_text_length(translated)
    if source_length < 80:
        return translated_length >= 12
    if source_length < 160:
        return translated_length >= max(24, int(source_length * 0.2))
    if source_length < 320:
        return translated_length >= max(40, int(source_length * 0.23))

    source_sentence_count = len([part for part in re.split(r"[.!?;:]+", source) if compact_text_length(part) >= 24])
    translated_sentence_count = len([part for part in re.split(r"[。！？；：]+", translated) if compact_text_length(part) >= 8])
    if source_sentence_count >= 3 and translated_sentence_count < max(2, int(source_sentence_count * 0.5)):
        return False
    return translated_length >= max(72, int(source_length * 0.26))


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
            "research_planning": StageRoutePolicy("research_planning", ("anthropic", "openai_compatible", "gemini")),
            "claim_evidence_generation": StageRoutePolicy("claim_evidence_generation", ("anthropic", "openai_compatible", "gemini")),
            "paper_qa": StageRoutePolicy("paper_qa", ("anthropic", "openai_compatible", "gemini")),
            "preference_memory": StageRoutePolicy("preference_memory", ("anthropic", "openai_compatible", "gemini")),
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

    def _ideal_aha_ontology(self) -> dict[str, Any]:
        return {
            "ideal_aha_definition": (
                "A true Aha happens when a learner's causal explanation of an already-lived experience gets reconstructed by the paper's evidence."
            ),
            "essence": "causal reconstruction of an already-lived learner experience",
            "four_conditions": [
                {
                    "id": 1,
                    "name": "causal_reconstruction",
                    "definition": "The learner had a prior causal explanation for something they have actually experienced, and the paper replaces or reorganizes that explanation.",
                },
                {
                    "id": 2,
                    "name": "methodological_credibility",
                    "definition": "The new causal model has enough evidence strength that it can beat the learner's old explanation rather than merely sounding interesting.",
                },
                {
                    "id": 3,
                    "name": "active_prior_belief",
                    "definition": "The old belief is still alive for the target learner in 2026; reconstructing a dead prior does not create a real aha.",
                },
                {
                    "id": 4,
                    "name": "course_operationalizability",
                    "definition": "After the aha is real, the new model can still become a teachable course object such as a tool, framework, reconstructive narrative, or evidence-backed talking point.",
                },
            ],
            "named_dilemma_as_proxy": [
                "A named dilemma is a strong proxy for a true aha because it usually implies lived experience, an insufficient old explanation, and a newly clarified structure.",
                "But named dilemma is still a proxy; the essence remains causal reconstruction.",
            ],
            "information_gain_vs_causal_reconstruction": [
                "Classification, taxonomy, landscape description, and theory compilation add information but do not create aha unless they replace a causal explanation the learner was using.",
                "Knowing more categories is not the same as understanding a familiar experience differently.",
            ],
            "evaluation_split": {
                "stage_a": "Conditions 1-3 decide whether this is a true aha at all.",
                "stage_b": "Condition 4 decides whether the true aha survives into the course pipeline.",
            },
            "operationalization_modes": [
                "method_or_tool",
                "evidence_or_argument",
                "reconstructive_narrative",
                "framework_or_model",
            ],
        }

    def _shared_prompt_policy(self) -> dict[str, Any]:
        ontology = self._ideal_aha_ontology()
        return {
            "policy_version": SHARED_PROMPT_POLICY_VERSION,
            "target_learner": "a practical learner with real work experience who has the potential to use AI and Agents to improve the efficiency of their production process or the value of their production output; they have hands-on practical familiarity but do not need deep technical expertise, and they have low patience for abstract theory",
            "core_objective": "find evidence-backed causal-reconstruction aha candidates that can later survive course operationalization without drifting away from the paper",
            "ideal_aha_definition": ontology["ideal_aha_definition"],
            "four_conditions": ontology["four_conditions"],
            "named_dilemma_as_proxy": ontology["named_dilemma_as_proxy"],
            "information_gain_vs_causal_reconstruction": ontology["information_gain_vs_causal_reconstruction"],
            "evaluation_split": ontology["evaluation_split"],
            "operationalization_modes": ontology["operationalization_modes"],
            "top_level_rules": [
                "A card is one atomic pattern or one atomic data finding, not a summary, topic label, or outline slot.",
                "A true aha is causal reconstruction first and course packaging second.",
                "Start from the learner's already-lived experience and the paper's concrete source object before doing any course naming.",
                "Treat a paper as invalid for this pipeline when it cannot reconstruct an active learner causal model, not only when topic words mismatch.",
                "A card survives only if it first creates a real old-model-to-new-model shift and then has short enough distance to course use.",
                "The shift should come from a better causal explanation: deeper mechanism, counterintuitive structure, or something learners feel but cannot name clearly.",
                "Every surviving card must be actionable in understanding, attitude, or method, but actionability alone does not make it a true aha.",
                "Use body evidence as the main material whenever body evidence exists.",
                "Keep the original figure when the figure is needed to understand the point.",
                "Reject recap, taxonomy, survey framing, generic advice, obvious 2026 claims, weak evidence, and wording that drifts above the source object or above the causal reconstruction.",
                "Do not treat repeated keyword hits or repeated topic attachment for the same paper as evidence that the paper is more important.",
                "Default to preserving the strongest single aha in a paper; a second one survives only when it is clearly independent in object, learner shift, and course use.",
                "Downgrade or reject content that is technically specific yet still feels like an internal architecture part, benchmark detail, or framework component rather than a tellable learner shift.",
                "Downgrade or reject content that needs long technical unpacking before a practical learner would care.",
                "Prefer candidates that can be expressed as 'the learner thought A, this paper makes them realize B' rather than as an internal paper takeaway.",
            ],
        }

    def _stage_spec(self, stage: str) -> dict[str, Any]:
        specs = {
            "paper_understanding": {
                "stage_goal": "Judge whether the paper contains a true aha under conditions 1-3, then identify only the source objects that anchor that causal reconstruction.",
                "must_do": [
                    "Return a paper-level verdict before proposing any object: on_topic, borderline_reject, or off_topic_hard.",
                    "Name the exact failure type when the paper should not continue.",
                    "Identify the learner's already-lived experience, the likely old causal model, and the new causal model supplied by the paper.",
                    "Judge methodological credibility and active-prior plausibility before letting course use influence the verdict.",
                    "Map each object to section evidence and figure evidence.",
                    "Prefer a single complete causal-reconstruction pattern or data finding over a broad umbrella label.",
                    "Mark the object at the right size: overall, local, or detail.",
                    "Identify the strongest true aha candidate in the paper before considering a second one.",
                ],
                "must_not_do": [
                    "Do not decide course naming or green/yellow/red here.",
                    "Do not turn literature structure, survey framing, or broad themes into objects.",
                    "Do not rescue a topic-adjacent paper just because you can imagine a course angle if the causal reconstruction is missing.",
                    "Do not keep a paper alive just because it is technically rich, benchmark-heavy, or attached to many overlapping topics.",
                ],
            },
            "card_planning": {
                "stage_goal": "Inherit the paper-level causal-reconstruction verdict and decide whether the strongest surviving aha can be operationalized into the course under condition 4.",
                "must_do": [
                    "Respect the understanding verdict before planning any card.",
                    "Assume conditions 1-3 were screened upstream and use this stage to test condition 4: course operationalizability.",
                    "Answer what each produced object becomes in the course and which operationalization mode it belongs to.",
                    "State the learner shift that makes the card worth keeping and keep it tied to the old-model-to-new-model change.",
                    "Require must-have section ids and figure ids when they are central.",
                    "Rank sibling candidates and prefer the strongest single aha by default.",
                ],
                "must_not_do": [
                    "Do not write final learner-facing card text.",
                    "Do not force card counts.",
                    "Do not keep a card plan that still sounds like a theme bucket instead of one card.",
                    "Do not let course usefulness rescue an object that was not a true aha upstream.",
                    "Do not let the same paper spawn multiple planned cards that merely restate the same core insight in different framings.",
                ],
            },
            "candidate_extraction": {
                "stage_goal": "Extract evidence-backed candidate cards from approved causal-reconstruction objects and log explicit exclusions for everything else.",
                "must_do": [
                    "Keep the candidate close to the paper-specific object.",
                    "Preserve the already-identified causal reconstruction instead of turning it into a cleaner but looser principle.",
                    "Use quotes and source evidence as the main body material.",
                    "Add only brief analysis instead of rewriting the paper into a cleaner doctrine.",
                    "When no aha candidates survive planning review, emit zero cards and log explicit exclusions only.",
                    "Extract only the planned strongest aha objects, not weaker same-paper rephrasings.",
                ],
                "must_not_do": [
                    "Do not do final color judgement here.",
                    "Do not emit cards that require long technical unpacking before use.",
                    "Do not hide excluded content that should be logged explicitly.",
                ],
            },
            "candidate_judgement": {
                "stage_goal": "Make the final keep or border call by judging causal reconstruction first, then course survivability, while finalizing course naming and translating evidence faithfully.",
                "must_do": [
                    "Explain the color decision with old model, new model, source fidelity, evidence credibility, active prior, and transfer distance in mind.",
                    "Provide full Chinese localization for primary evidence.",
                    "Check that the card still names one concrete object instead of a vague lesson.",
                    "Check whether this is the strongest tellable true aha in the paper rather than merely a valid technical object.",
                ],
                "must_not_do": [
                    "Do not invent missing evidence or figures.",
                    "Do not rescue weak cards with prettier wording.",
                    "Do not let course naming drift above the source object.",
                    "Do not keep a candidate just because it is technically correct or operationally useful if the causal reconstruction is weak, over-technical, or duplicated by a stronger sibling candidate.",
                ],
            },
        }
        return specs[stage]

    def _render_system_prompt(self, stage: str) -> str:
        policy = self._shared_prompt_policy()
        spec = self._stage_spec(stage)
        ontology = self._ideal_aha_ontology()
        _SHARED_RULE_PERSONALITIES = [
            "You are certain that a card is one atomic pattern or one atomic data finding — never a summary, topic label, or outline placeholder.",
            "You believe deeply that a true aha is causal reconstruction first and course packaging second.",
            "You always start from the learner's already-lived experience and the paper's concrete source object before doing any course naming.",
            "You treat any paper that cannot reconstruct an active learner causal model as invalid for this pipeline — topic word overlap is not enough.",
            "You believe a card earns its place only by first producing a real old-model-to-new-model shift and then having short enough distance to course use.",
            "You are drawn to shifts that come from a better causal explanation: deeper mechanism, counterintuitive structure, or something learners feel but cannot name clearly.",
            "You require every surviving card to be actionable in understanding, attitude, or method — but actionability alone is not enough to make it a true aha.",
            "You always use body evidence as the main material when body evidence exists.",
            "You always keep the original figure when it is needed to understand the point.",
            "You are deeply averse to recap, taxonomy, survey framing, generic advice, obvious 2026 claims, weak evidence, and wording that drifts above the source object or the causal reconstruction.",
            "You never treat repeated keyword hits or repeated topic attachment for the same paper as evidence that the paper is more important.",
            "Your instinct is to preserve the single strongest aha in a paper; a second one survives only when its object, learner shift, and course use are clearly independent.",
            "You actively downgrade or reject content that is technically specific yet still feels like an internal architecture part, benchmark detail, or framework component rather than a tellable learner shift.",
            "You actively downgrade or reject content that needs long technical unpacking before a practical learner would care.",
            "You prefer candidates expressible as 'the learner thought A, this paper makes them realize B' over internal paper takeaways.",
        ]
        lines = [
            f"You are the {stage} specialist in a paper-to-course pipeline — ruthlessly exacting about aha quality.",
            "You have a formatting obsession: output strict JSON only. Any text outside the JSON structure makes you uncomfortable.",
            f"Shared policy version: {policy['policy_version']}.",
            f"Target learner: {policy['target_learner']}.",
            f"Core objective: {policy['core_objective']}.",
        ]
        if stage == "candidate_extraction":
            lines.append(
                "Aha conditions and causal reconstruction were already evaluated by the upstream understanding and planning stages. "
                "Your task here is faithful extraction from the approved planned objects, not re-adjudication of aha validity."
            )
            lines.append(f"Ideal aha definition (for context only, do not re-litigate): {ontology['ideal_aha_definition']}")
        else:
            lines.append(f"Ideal aha definition: {ontology['ideal_aha_definition']}")
            lines.append("Ideal aha conditions:")
            lines.extend(f"- {item['name']}: {item['definition']}" for item in ontology["four_conditions"])
            lines.append("Named dilemma as proxy:")
            lines.extend(f"- {item}" for item in ontology["named_dilemma_as_proxy"])
            lines.append("Information gain vs causal reconstruction:")
            lines.extend(f"- {item}" for item in ontology["information_gain_vs_causal_reconstruction"])
            lines.append(f"Evaluation split: stage A = {ontology['evaluation_split']['stage_a']}")
            lines.append(f"Evaluation split: stage B = {ontology['evaluation_split']['stage_b']}")
        lines.append("Your core beliefs and behavioral tendencies (internalized as instincts):")
        for i, _rule in enumerate(policy["top_level_rules"]):
            personality = _SHARED_RULE_PERSONALITIES[i] if i < len(_SHARED_RULE_PERSONALITIES) else f"- {_rule}"
            lines.append(f"- {personality}")
        lines.append(f"Stage goal: {spec['stage_goal']}")
        lines.append("In this stage, you instinctively:")
        lines.extend(f"- {rule}" for rule in spec["must_do"])
        lines.append("In this stage, you are deeply averse to and will not do:")
        lines.extend(f"- {rule}" for rule in spec["must_not_do"])
        lines.append("All learner-facing strings you produce are naturally in Simplified Chinese.")
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
                    "ontology_lesson": self._build_example_ontology_lesson(example),
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
                    "ontology_lesson": self._build_example_ontology_lesson(example),
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
                    "ontology_lesson": self._build_example_ontology_lesson(example),
                }
                for example in selected[:MAX_CALIBRATION_EXAMPLES]
            ]
        return []

    def _build_example_ontology_lesson(self, example: dict[str, Any]) -> str:
        example_type = str(example.get("example_type", "")).strip().lower()
        expected_cards = example.get("expected_cards", []) or []
        expected_exclusions = example.get("expected_exclusions", []) or []
        exclusion_types = {
            str(item.get("exclusion_type", "")).strip().lower()
            for item in expected_exclusions
            if isinstance(item, dict)
        }
        if expected_cards and example_type == "positive":
            return "Positive example: the paper replaces an old causal explanation with a better new one, and the result can later be operationalized in the course."
        if example_type == "boundary":
            return "Boundary example: check whether this is true causal reconstruction with enough evidence and an active prior, or merely useful packaging / partial insight."
        if exclusion_types.intersection({"taxonomy_not_insight", "summary", "background"}):
            return "Negative example: this adds information or structure without replacing the learner's causal explanation."
        if exclusion_types.intersection({"weak_method_or_data", "low_hanging_fruit"}):
            return "Negative example: a possible causal claim exists, but it fails because the evidence is too weak or the old prior is no longer active."
        if expected_exclusions:
            return "Negative example: this may be interesting or useful, but it should not survive because it is not a true course-worthy aha under the causal-reconstruction ontology."
        return "Use this example to distinguish true causal reconstruction from mere information gain or generic course packaging."

    def extract_candidates(
        self,
        *,
        topic_name: str,
        paper_title: str,
        sections: list[dict],
        figures: Optional[list[dict]] = None,
        planned_cards: Optional[list[dict]] = None,
        planning_context: Optional[dict[str, Any]] = None,
        calibration_examples: Optional[list[dict]] = None,
        calibration_set_name: str = "",
        active_memory: Optional[dict[str, Any]] = None,
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
                planned_cards=planned_cards or [],
                planning_context=planning_context or {},
                calibration_examples=selected_examples,
                stage_examples=stage_examples,
                calibration_set_name=calibration_set_name,
                active_memory=active_memory or {},
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
        normalized = self._normalize_extraction_output(payload, sections, figures or [], planned_cards or [])
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
        active_memory: Optional[dict[str, Any]] = None,
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
                    "source_plan_id": card.get("source_plan_id", ""),
                    "planned_object_label": card.get("planned_object_label", ""),
                    "planned_why_valuable": card.get("planned_why_valuable", ""),
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
                active_memory=active_memory or {},
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
                "sections": prompt_sections,
                "figures": prompt_figures,
                "requirements": {
                    "paper_relevance_rules": [
                        "First decide whether this paper contains a true aha under conditions 1-3, not just whether topic words overlap.",
                        "Ask what lived learner experience this paper speaks to, what old causal model the learner likely holds, and what new causal model the paper supplies.",
                        "Reject papers that provide information gain without replacing a learner's causal explanation.",
                        "Use off_topic_hard only for clear cases: pure technical mismatch, obvious topic-word-overlap-only, or clearly unusable input for this learner and topic.",
                        "Use borderline_reject for close-but-not-worth-it cases where the paper seems related yet still does not create a convincing causal reconstruction or still fails active-prior / credibility checks.",
                        "Treat a paper as not worth keeping when it is technically detailed yet still fails to produce one true aha for this learner.",
                        "Do not let repeated keyword overlap or multi-topic attachment rescue a paper whose strongest candidate still feels internal, over-technical, or weakly teachable.",
                        "If verdict is not on_topic, return global_contribution_objects as an empty list.",
                    ],
                    "object_requirements": [
                        "Object labels must be concrete and plain, not generic section names like 'Markdown Extraction'.",
                        "Each object should be one atomic pattern or one atomic data finding that anchors the causal reconstruction, not a mixed basket.",
                        "Each object must map to evidence section ids.",
                        "Use level_hint among overall/local/detail.",
                        "Prefer objects that a practical learner can picture as a workflow, decision point, failure mode, comparison, reusable pattern, or data point tied to the new causal model.",
                        "When a figure is central to understanding the object, include evidence_figure_ids.",
                    ],
                    "quality_requirements": [
                        "Prefer paper-specific mechanism, model, method, result, workflow, comparison, or failure-mode objects that explain why a familiar experience happens differently than the learner assumed.",
                        "Avoid framing-only objects when body evidence exists.",
                        "Prefer source-native workflow and decision structures over broad principles.",
                        "Do not rewrite technical content into generic management or life advice.",
                        "If a concrete sub-pattern is directly teachable, prefer it over a more abstract umbrella label.",
                        "If the object would still need heavy unpacking before a learner can picture it, leave it out.",
                        "Look for the strongest single old-model-to-new-model shift in the paper before preserving additional objects.",
                        "Downgrade framework components, benchmark internals, or architecture parts when they are technically valid but not naturally reportable as an aha.",
                    ],
                },
                "output_schema": {
                    "paper_relevance_verdict": "on_topic|borderline_reject|off_topic_hard",
                    "paper_relevance_reason": "一句话说明为什么这篇论文值得继续或不值得继续",
                    "relevance_failure_type": "pure_technical_mismatch|cannot_name_course_object|long_transfer_distance|taxonomy_not_insight|weak_method_or_data|low_hanging_fruit|topic_word_overlap_only|technical_but_not_reportable|same_paper_multi_topic_overlap|framework_without_shift|benchmark_without_teachable_delta|other",
                    "global_contribution_objects": [
                        {
                            "id": "obj_1",
                            "label": "对象名称，直白、具体，可用中文或英文短语",
                            "object_type": "mechanism|model|method|result|framework|data_finding|other",
                            "level_hint": "overall|local|detail",
                            "evidence_section_ids": ["section_id"],
                            "evidence_figure_ids": ["figure_id"],
                            "summary": "一到两句说明，讲清这到底是什么对象",
                            "importance_score": 0.0,
                            "reportable_aha_rank": 1,
                        }
                    ],
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
                "active_calibration_set": calibration_set_name,
                "stage_examples": self._build_stage_examples("card_planning", calibration_examples or [], topic_name),
                "understanding": understanding,
                "requirements": {
                    "planning_rules": [
                        "Inherit the paper_relevance_verdict from understanding before you plan anything.",
                        "If paper_relevance_verdict is off_topic_hard, produce zero cards.",
                        "If paper_relevance_verdict is borderline_reject, default to zero cards unless the understanding payload already contains a clearly teachable concrete object with short transfer distance.",
                        "Do not force card counts; 0 card is allowed when no object is teachable.",
                        "Treat this stage as condition 4 only: ask how a true aha becomes a course object after conditions 1-3 were screened upstream.",
                        "A planned card must be one card, not a theme bucket or a paper recap.",
                        "A planned card should center on one atomic pattern or one atomic data finding.",
                        "A produce item is valid only if you can state both the learner shift and what it becomes in the course.",
                        "A produce item is strong only if you can also say why a leader or practical learner would care in one sentence.",
                        "Default to one strong aha per paper; produce a second one only if its object, learner shift, and course use are clearly independent from the first.",
                        "If a second candidate is only a submechanism, applied slice, or zoomed-in restatement of the main causal chain, exclude it instead of producing a second card.",
                        "A second card is independent only when the lived experience, old model, new model, and course action each remain meaningfully distinct from the first card.",
                        "A result card or evidence card is not a second aha by default; keep it only when the result itself changes a different active learner explanation rather than merely adding proof or numbers to the main mechanism card.",
                        "Prefer the smallest object that is complete enough to teach well.",
                        "Use overall/local/detail only when it matches the real size of the object, not to fill slots.",
                        "Each produce item must specify must_have_evidence_ids.",
                        "Prefer source-faithful workflow steps, decision rules, failure modes, comparison structures, mechanisms, and data findings that preserve the paper's causal reconstruction and that a practical learner can grasp quickly.",
                        "Choose an operationalization mode mentally: method_or_tool, evidence_or_argument, reconstructive_narrative, or framework_or_model.",
                        "Do not exclude an object merely because it is technical if the evidence describes a concrete workflow or decision process with short transfer distance.",
                        "Do not promote a high-level umbrella object when a more concrete child object is more directly teachable for the target learner.",
                        "Do not let course usefulness rescue an object that still does not feel like a true causal reconstruction.",
                        "Reject background theory, taxonomy recap, old obvious claims, weak-transfer details, and anything that still needs heavy unpacking before use.",
                        "When a figure is central to the object, require it explicitly with must_have_figure_ids.",
                        "Explicitly compare sibling candidates and suppress weaker same-paper rephrasings.",
                    ]
                },
                "output_schema": {
                    "paper_relevance_verdict": "on_topic|borderline_reject|off_topic_hard",
                    "paper_relevance_reason": "沿用或补充 paper-level verdict 的原因",
                    "relevance_failure_type": "pure_technical_mismatch|cannot_name_course_object|long_transfer_distance|taxonomy_not_insight|weak_method_or_data|low_hanging_fruit|topic_word_overlap_only|technical_but_not_reportable|same_paper_multi_topic_overlap|framework_without_shift|benchmark_without_teachable_delta|other",
                    "planned_cards": [
                        {
                            "plan_id": "plan_obj_1",
                            "level": "overall|local|detail",
                            "target_object_id": "obj_1",
                            "target_object_label": "对象名，必须具体到一个卡片对象",
                            "why_valuable_for_course": "课程价值说明，讲清 learner shift 和课程用法",
                            "must_have_evidence_ids": ["section_id"],
                            "optional_supporting_ids": ["section_id"],
                            "must_have_figure_ids": ["figure_id"],
                            "optional_supporting_figure_ids": ["figure_id"],
                            "disposition": "produce|exclude",
                            "disposition_reason": "排除时必填，直接说明为什么不该有这张卡",
                        }
                    ],
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

    def recommend_search_terms(self, research_goal: str, *, max_terms: int = 6) -> dict[str, Any]:
        if not self.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")
        normalized_goal = str(research_goal or "").strip()
        if not normalized_goal:
            raise ValueError("research_goal is required")
        bounded_max_terms = max(1, min(int(max_terms or 6), 12))
        system_prompt = "\n".join(
            [
                "You help an internal research operator turn one research goal into practical academic topic anchors for literature discovery.",
                "Return strict JSON only.",
                "Do not output any explanation, rationale, summary, priority, or metadata.",
                "Output only a flat list of short topic phrases that can be pasted directly into the Topics textarea.",
                "The current system will later expand each topic into mechanism/application/recency discovery variants, so the topic itself must stay short and central.",
                "Prefer mid-level academic anchors that are broad enough to retrieve papers but specific enough to stay on the intended problem.",
                "Prefer 1-4 words per topic phrase; 2-3 words is ideal.",
                "Avoid long-tail compositional phrases, full claims, full questions, or phrases that already include too much context.",
                "Avoid weak peripheral directions that are only loosely related to the user's goal.",
                "Do not include boolean operators, quotation marks, site filters, or provider-specific search syntax.",
                "Avoid near-duplicate phrasings that only swap one synonym or add one modifier.",
                "Prefer terms that are likely to work across common academic search providers.",
                "Output schema:",
                "{",
                '  "recommended_topics": ["string"]',
                "}",
            ]
        )
        user_prompt = json.dumps(
            {
                "stage": "search_term_recommendation",
                "research_goal": normalized_goal,
                "max_terms": bounded_max_terms,
                "operator_context": {
                    "downstream_use": "Each returned topic will later be expanded by the system into core / mechanism / application / recency search strategies.",
                    "preferred_shape": "short paste-ready academic topic phrases for the Topics box",
                },
                "requirements": [
                    "Return 4 to max_terms recommended topics unless the goal is extremely narrow.",
                    "Each topic should be 1-4 words whenever possible, and must not exceed 4 words unless the phrase is impossible to shorten without losing the topic.",
                    "Prefer central search anchors over detailed long-tail variants.",
                    "Do not explain the terms; just output the terms.",
                    "If the goal contains many ideas, choose the few highest-leverage search anchors instead of trying to cover everything.",
                    "If the goal is ambiguous, resolve it into likely academic topic anchors instead of asking a follow-up question.",
                ],
            },
            ensure_ascii=False,
        )
        payload = self._chat_json(
            stage="search_term_recommendation",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        raw_topics = payload.get("recommended_topics", [])
        if not isinstance(raw_topics, list):
            raw_topics = []
        normalized_topics: list[str] = []
        seen_topics: set[str] = set()
        for item in raw_topics:
            if isinstance(item, dict):
                topic = str(item.get("topic", "")).strip()
            else:
                topic = str(item or "").strip()
            if not topic:
                continue
            topic = topic.strip().strip("\"'`")
            topic = re.sub(r"\s+", " ", topic)
            topic = re.sub(r"\b(and|or|not|site)\b", " ", topic, flags=re.IGNORECASE)
            topic = re.sub(r"[,:;!?()\[\]{}]+", " ", topic)
            topic = re.sub(r"\s+", " ", topic).strip()
            word_count = len(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", topic))
            if word_count > 4:
                continue
            normalized_key = re.sub(r"\s+", " ", topic.lower())
            if normalized_key in seen_topics:
                continue
            seen_topics.add(normalized_key)
            normalized_topics.append(topic)
            if len(normalized_topics) >= bounded_max_terms:
                break
        return {
            "research_goal": normalized_goal,
            "recommended_topics": normalized_topics,
            "suggested_topics_text": "\n".join(normalized_topics),
            "provider_route": self._last_provider_route,
        }

    def draft_research_plan(
        self,
        research_brief: str,
        *,
        requested_task_type: str = "auto",
        max_terms: int = 6,
        active_memory: Optional[dict[str, Any]] = None,
        also_generate_aha_cards: bool = False,
    ) -> dict[str, Any]:
        if not self.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")
        normalized_brief = str(research_brief or "").strip()
        if not normalized_brief:
            raise ValueError("research_brief is required")
        bounded_max_terms = max(1, min(int(max_terms or 6), 12))
        requested_mode = str(requested_task_type or "auto").strip().lower() or "auto"
        system_prompt = "\n".join(
            [
                "You turn one internal research brief into a confirmable execution plan for a paper research workflow.",
                "Return strict JSON only.",
                "First decide whether the task is aha_exploration or claim_evidence unless the requested mode is already explicit.",
                "aha_exploration means open-ended discovery of high-value learner-facing aha moments.",
                "claim_evidence means the user already has a claim or framework and wants supporting, mixed, contradictory, and limiting evidence.",
                "Keep the plan minimal, executable, and directly useful for the downstream search pipeline.",
                "For aha_exploration, output short search topics only; do not output long explanations.",
                "For claim_evidence, output search_topics as structured entries with dimension labels, query anchors, and outcome terms.",
                "Do not output prose outside the JSON schema.",
                "Output schema:",
                "{",
                '  "suggested_task_type": "aha_exploration|claim_evidence",',
                '  "summary": "string",',
                '  "claim": "string",',
                '  "recommended_topics": ["string"],',
                '  "search_topics": [',
                "    {",
                '      "topic_name": "string",',
                '      "dimension_key": "string",',
                '      "dimension_label": "string",',
                '      "query_anchor": "string",',
                '      "outcome_terms": ["string"]',
                "    }",
                "  ],",
                '  "outcomes": ["string"],',
                '  "evidence_policy": {',
                '    "surface_contradictions": true,',
                '    "minimum_supporting_papers_per_dimension": 3',
                "  },",
                '  "also_generate_aha_cards": false',
                "}",
            ]
        )
        user_prompt = json.dumps(
            {
                "stage": "research_planning",
                "prompt_version": RESEARCH_PLAN_PROMPT_VERSION,
                "research_brief": normalized_brief,
                "requested_task_type": requested_mode,
                "max_terms": bounded_max_terms,
                "also_generate_aha_cards": also_generate_aha_cards,
                "active_preference_memory": active_memory or {},
                "requirements": [
                    "If the brief asks to prove, support, validate, or challenge a known framework or claim, choose claim_evidence.",
                    "If the brief asks to discover unknown high-value insights, choose aha_exploration.",
                    "For aha_exploration, prefer 3 to max_terms short academic anchors.",
                    "For claim_evidence, prefer one search topic per claim dimension whenever possible.",
                    "For claim_evidence, outcome_terms should contain the most decision-relevant measurable outcomes, not generic filler.",
                    "Do not invent more than max_terms search topics unless the brief explicitly contains more hard dimensions.",
                    "Stay as close as possible to the user's phrasing, but convert it into search-ready academic anchors.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._chat_json(stage="research_planning", system_prompt=system_prompt, user_prompt=user_prompt)
        normalized = {
            "suggested_task_type": str(payload.get("suggested_task_type", "")).strip().lower(),
            "summary": str(payload.get("summary", "")).strip(),
            "claim": str(payload.get("claim", "")).strip(),
            "recommended_topics": payload.get("recommended_topics", []),
            "search_topics": payload.get("search_topics", []),
            "outcomes": payload.get("outcomes", []),
            "evidence_policy": payload.get("evidence_policy", {}),
            "also_generate_aha_cards": bool(payload.get("also_generate_aha_cards", also_generate_aha_cards)),
            "provider_route": self._last_provider_route,
        }
        return normalized

    def generate_matrix_items(
        self,
        *,
        claim_text: str,
        topic_name: str,
        paper_title: str,
        dimension: dict[str, Any],
        sections: list[dict],
        figures: Optional[list[dict]] = None,
        active_memory: Optional[dict[str, Any]] = None,
        evidence_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[dict]]:
        if not self.client:
            return {"items": []}
        prompt_sections = self._build_prompt_sections(sections)
        prompt_figures = self._build_prompt_figures(figures or [])
        dimension_alias_guidance = self._build_claim_evidence_dimension_alias_guidance(dimension)
        system_prompt_lines = [
            "You convert one paper into grounded evidence matrix items for a known claim.",
            "Return strict JSON only.",
            "Do not summarize the whole paper. Only emit items that directly matter for the given dimension and outcomes.",
            "Each item must be grounded in section ids from the provided paper.",
            "Use verdict among supporting, mixed, contradictory, or context_only.",
            "Always surface limitations when they materially affect how strongly this paper supports the claim.",
            "Write learner-facing and report-facing strings in Simplified Chinese.",
            "Before extracting items, judge whether the paper is within the same subject domain as the claim. If the paper is primarily about a different domain, emit zero items rather than stretching evidence across domains.",
            "Cross-domain analogies do not count as direct evidence for a claim. A paper from a different domain may provide context_only items only when the mechanism is explicitly argued to transfer and the paper itself makes that argument.",
            "Proxy constructs and operationalizations that are within the same domain as the claim count as direct evidence when the paper makes the behavior-outcome link explicit.",
            "Dimension labels and claim constructs should be interpreted by their substantive meaning, not literal wording only; valid domain-appropriate proxy measures count as direct evidence when the paper makes the link explicit.",
            "For every cited section, quote_zh must be a faithful Simplified Chinese translation of the cited evidence passages from that section; preserve all caveats, enumerations, and logical structure.",
            "Do not turn quote_zh into a summary, evidence interpretation, paraphrase, or selective restatement of the cited passages.",
            "If a cited section is long, translate the directly relevant passages faithfully and note where the section continues; do not silently compress or omit material parts.",
            "If the provided evidence is abstract-only or metadata-derived rather than full text, emit at most one matrix item, keep the claim modest, and treat the evidence as weaker than full-text evidence.",
        ]
        if dimension_alias_guidance:
            system_prompt_lines.append(dimension_alias_guidance)
        system_prompt = "\n".join(system_prompt_lines)
        user_prompt = json.dumps(
            {
                "stage": "claim_evidence_generation",
                "prompt_version": CLAIM_EVIDENCE_PROMPT_VERSION,
                "claim_text": claim_text,
                "topic_name": topic_name,
                "paper_title": paper_title,
                "dimension": dimension,
                "dimension_alias_guidance": dimension_alias_guidance,
                "sections": prompt_sections,
                "figures": prompt_figures,
                "active_preference_memory": active_memory or {},
                "evidence_policy": evidence_policy or {},
                "requirements": [
                    "Emit zero items if the paper only gives broad background without a usable link to the claim dimension or outcomes.",
                    "Prefer papers that directly support or challenge the claim over papers that only provide distant context.",
                    "If a paper is from a clearly different domain than the claim, treat it as off-domain and emit zero items rather than forcing analogical evidence into the matrix.",
                    "If a paper is in the same domain as the claim but mainly supports a mechanism or intermediate outcome, mixed or context_only is acceptable; do not force zero items unless the paper is clearly off-domain.",
                    "Use dimension_alias_guidance when provided to map domain-appropriate proxy constructs onto the target dimension; do not require the paper to repeat the exact dimension label literally.",
                    "quote_zh must faithfully translate the cited evidence passages, preserving caveats, enumerations, and logical structure; for long sections, translate the directly relevant passages and note the boundary.",
                    "Do not prepend meta-summary language such as '文中提到' or '作者指出' unless those words are explicitly present in the source evidence.",
                    "If the available evidence is only an abstract or metadata summary, emit at most one weak item and explicitly state in limitation_text that the full paper was not available for verification.",
                    "A supporting item should make the direction of support clear and name the relevant outcome.",
                    "A mixed item should state both the support and the limitation or boundary condition.",
                    "A contradictory item should only be used when the paper materially undermines the claim or reverses the expected direction.",
                    "A context_only item is allowed only for near-domain framing within the same subject domain as the claim, not for cross-domain analogies.",
                    "Use primary_section_ids for the strongest evidence, supporting_section_ids for secondary evidence.",
                    "The summary should be one concise Chinese paragraph suitable for the downstream audience.",
                ],
                "output_schema": {
                    "items": [
                        {
                            "dimension_key": "string",
                            "dimension_label": "string",
                            "outcome_key": "string",
                            "outcome_label": "string",
                            "claim_text": "string",
                            "verdict": "supporting|mixed|contradictory|context_only",
                            "evidence_strength": "strong|medium|weak",
                            "summary": "中文简短总结",
                            "limitation_text": "中文限制或边界条件",
                            "citation_text": "作者与年份",
                            "primary_section_ids": ["section_id"],
                            "supporting_section_ids": ["section_id"],
                            "figure_ids": ["figure_id"],
                            "evidence_analysis": [
                                {
                                    "section_id": "section_id",
                                    "quote_zh": "对应 section_id 原文整段内容的完整中文译文，不得总结、提炼、改写或只翻译其中一部分",
                                    "analysis": "为什么这段证据支持或限制该命题",
                                }
                            ],
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._chat_json(stage="claim_evidence_generation", system_prompt=system_prompt, user_prompt=user_prompt)
        return self._normalize_matrix_items_output(payload, sections, figures or [], dimension, claim_text)

    def _build_claim_evidence_dimension_alias_guidance(self, dimension: dict[str, Any]) -> str:
        dimension_key = str(dimension.get("dimension_key", "") or "").strip().lower()
        guidance_by_dimension = {
            "expression": (
                "For the expression dimension, leadership communication clarity, goal clarity, sensegiving, framing,"
                " expectation-setting, supervisory communication quality, and message clarity are valid direct evidence"
                " proxies when the paper links them to employee or team outcomes."
            ),
            "listening": (
                "For the listening dimension, leader listening behavior, perceived listening quality, receptiveness to"
                " employee voice, supportive listening, and leader responsiveness count as direct evidence proxies when"
                " the paper links them to trust, psychological safety, voice, satisfaction, or performance."
            ),
            "questioning": (
                "For the questioning dimension, managerial coaching, coaching leadership, inquiry-based leadership,"
                " developmental feedback conversations, reflective inquiry, and coach-like prompting count as direct"
                " evidence proxies when the paper shows that leader dialogue prompts employee thinking, learning,"
                " role clarity, problem solving, development, or performance. Do not require the literal word"
                " 'questioning' if the coaching or inquiry mechanism is explicit."
            ),
            "empathy": (
                "For the empathy dimension, leader compassion, individualized consideration, empathic concern,"
                " supportive concern, mindful communication, and emotionally attuned leadership count as direct"
                " evidence proxies when linked to relationship quality, trust, burnout, well-being, or performance."
            ),
            "action_facilitation": (
                "For the action facilitation dimension, action planning, accountability, follow-up, implementation"
                " intentions, goal setting, feedback-to-action, and behavioral commitment mechanisms count as direct"
                " evidence proxies when they connect leader dialogue to execution, goal attainment, or task performance."
            ),
            "integrative_framework": (
                "For the integrative framework dimension, combined leadership behaviors, multidimensional leadership"
                " communication, mediated or moderated leadership pathways, and interaction or bundle effects count as"
                " relevant evidence even if the paper does not name the exact five-dimension framework."
            ),
            "boundary_contradictions": (
                "For the boundary and contradiction dimension, null effects, inconsistent findings, curvilinear results,"
                " moderator effects, task or culture contingencies, and remote-work limitations count as direct evidence"
                " even when they challenge only part of the broader claim."
            ),
            "measurement_methods": (
                "For the measurement and methods dimension, longitudinal designs, diary studies, experience sampling,"
                " matched supervisor-subordinate data, multilevel models, behavioral coding, text analysis, and field"
                " experiments count as direct evidence about evidence quality even if they do not test the full claim."
            ),
        }
        return guidance_by_dimension.get(dimension_key, "")

    def answer_paper_question(
        self,
        *,
        paper_title: str,
        question: str,
        sections: list[dict],
        figures: Optional[list[dict]] = None,
        active_memory: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")
        system_prompt = "\n".join(
            [
                "You answer a question about one specific paper using only the provided retrieved sections and linked figures.",
                "Return strict JSON only.",
                "If the provided evidence is insufficient, say so explicitly instead of guessing.",
                "Do not use external knowledge beyond the retrieved paper evidence.",
                "Write the answer in Simplified Chinese.",
            ]
        )
        user_prompt = json.dumps(
            {
                "stage": "paper_qa",
                "prompt_version": PAPER_QA_PROMPT_VERSION,
                "paper_title": paper_title,
                "question": question,
                "sections": self._build_prompt_sections(sections),
                "figures": self._build_prompt_figures(figures or []),
                "active_preference_memory": active_memory or {},
                "requirements": [
                    "Only cite section ids that appear in the provided sections list.",
                    "Answer directly and concisely.",
                    "If the question asks for causality but the paper only shows correlation or qualitative evidence, say that clearly.",
                ],
                "output_schema": {
                    "answer": "中文回答",
                    "confidence_note": "中文，说明证据是否充分",
                    "cannot_answer_from_paper": False,
                    "used_section_ids": ["section_id"],
                    "used_figure_ids": ["figure_id"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._chat_json(stage="paper_qa", system_prompt=system_prompt, user_prompt=user_prompt)
        normalized = {
            "answer": str(payload.get("answer", "")).strip(),
            "confidence_note": str(payload.get("confidence_note", "")).strip(),
            "cannot_answer_from_paper": bool(payload.get("cannot_answer_from_paper", False)),
            "used_section_ids": [str(item).strip() for item in payload.get("used_section_ids", []) if str(item).strip()],
            "used_figure_ids": [str(item).strip() for item in payload.get("used_figure_ids", []) if str(item).strip()],
            "provider_route": self._last_provider_route,
        }
        section_map = {section["id"]: section for section in sections}
        figure_map = {figure["id"]: figure for figure in (figures or []) if figure.get("id")}
        normalized["used_sections"] = [
            {
                "section_id": section_id,
                "section_title": section_map[section_id].get("section_title", ""),
                "page_number": section_map[section_id].get("page_number"),
                "paragraph_text": section_map[section_id].get("paragraph_text", ""),
            }
            for section_id in normalized["used_section_ids"]
            if section_id in section_map
        ]
        normalized["used_figures"] = [figure_map[figure_id] for figure_id in normalized["used_figure_ids"] if figure_id in figure_map]
        return normalized

    def distill_preference_memory(
        self,
        review_items: list[dict],
        *,
        task_type: str = "",
        active_memory: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")
        condensed_items = []
        for item in review_items[:80]:
            condensed_items.append(
                {
                    "object_type": item.get("object_type", ""),
                    "topic_name": item.get("topic_name", ""),
                    "paper_title": item.get("paper_title", ""),
                    "display_title": item.get("display_title", ""),
                    "review_status": item.get("review_status", ""),
                    "comment_text": item.get("comment_text", ""),
                    "verdict": item.get("verdict", ""),
                    "evidence_strength": item.get("evidence_strength", ""),
                    "dimension_label": item.get("dimension_label", ""),
                    "outcome_label": item.get("outcome_label", ""),
                }
            )
        system_prompt = "\n".join(
            [
                "You distill explicit user review behavior into a small, human-confirmable preference memory draft.",
                "Return strict JSON only.",
                "This is not model retraining and not a hidden automatic optimization loop.",
                "Summarize only stable, review-supported preferences that should influence future planning and filtering.",
                "Do not invent preferences that are not visible in the review data.",
            ]
        )
        user_prompt = json.dumps(
            {
                "stage": "preference_memory",
                "prompt_version": PREFERENCE_MEMORY_PROMPT_VERSION,
                "task_type": task_type,
                "active_preference_memory": active_memory or {},
                "review_items": condensed_items,
                "requirements": [
                    "Prefer stable rules over one-off anecdotes.",
                    "Write prefer and avoid as direct filtering or planning instructions.",
                    "review_signals should cite the repeated behavior pattern, not restate the whole dataset.",
                    "Keep the draft compact and human-reviewable.",
                ],
                "output_schema": {
                    "scope": "project",
                    "mode": "aha_exploration|claim_evidence|mixed",
                    "summary": "一句话总结",
                    "prefer": ["string"],
                    "avoid": ["string"],
                    "review_signals": ["string"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self._chat_json(stage="preference_memory", system_prompt=system_prompt, user_prompt=user_prompt)
        return {
            "scope": str(payload.get("scope", "project")).strip() or "project",
            "mode": str(payload.get("mode", task_type or "mixed")).strip() or (task_type or "mixed"),
            "summary": str(payload.get("summary", "")).strip(),
            "prefer": [str(item).strip() for item in payload.get("prefer", []) if str(item).strip()],
            "avoid": [str(item).strip() for item in payload.get("avoid", []) if str(item).strip()],
            "review_signals": [str(item).strip() for item in payload.get("review_signals", []) if str(item).strip()],
            "provider_route": self._last_provider_route,
        }

    def _normalize_matrix_items_output(
        self,
        payload: dict[str, Any],
        sections: list[dict],
        figures: list[dict],
        dimension: dict[str, Any],
        claim_text: str,
    ) -> dict[str, list[dict]]:
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []
        section_map = {section["id"]: section for section in sections}
        figure_id_set = {figure["id"] for figure in figures}
        normalized_items = []
        for raw_item in raw_items[:8]:
            if not isinstance(raw_item, dict):
                continue
            primary_section_ids = [
                str(section_id).strip()
                for section_id in raw_item.get("primary_section_ids", [])
                if str(section_id).strip() in section_map
            ]
            supporting_section_ids = [
                str(section_id).strip()
                for section_id in raw_item.get("supporting_section_ids", [])
                if str(section_id).strip() in section_map and str(section_id).strip() not in primary_section_ids
            ]
            if not primary_section_ids:
                continue
            analysis_by_section_id = {}
            for item in raw_item.get("evidence_analysis", []) if isinstance(raw_item.get("evidence_analysis", []), list) else []:
                if not isinstance(item, dict):
                    continue
                section_id = str(item.get("section_id", "")).strip()
                if section_id and section_id in section_map:
                    analysis_by_section_id[section_id] = {
                        "quote_zh": str(item.get("quote_zh", "")).strip(),
                        "analysis": str(item.get("analysis", "")).strip(),
                    }
            evidence = []
            for section_id in primary_section_ids + supporting_section_ids:
                section = section_map[section_id]
                localized = analysis_by_section_id.get(section_id, {})
                quote_zh = localized.get("quote_zh", "")
                if not looks_like_complete_translation(section.get("paragraph_text", ""), quote_zh):
                    evidence = []
                    break
                evidence.append(
                    {
                        "section_id": section_id,
                        "quote": section.get("paragraph_text", ""),
                        "quote_zh": quote_zh,
                        "analysis": localized.get("analysis", ""),
                        "page_number": section.get("page_number"),
                    }
                )
            figure_ids = [
                str(figure_id).strip()
                for figure_id in raw_item.get("figure_ids", [])
                if str(figure_id).strip() in figure_id_set
            ]
            normalized_items.append(
                {
                    "dimension_key": str(raw_item.get("dimension_key", "") or dimension.get("dimension_key", "")).strip(),
                    "dimension_label": str(raw_item.get("dimension_label", "") or dimension.get("dimension_label", "")).strip(),
                    "outcome_key": str(raw_item.get("outcome_key", "")).strip(),
                    "outcome_label": str(raw_item.get("outcome_label", "")).strip(),
                    "claim_text": str(raw_item.get("claim_text", "") or claim_text).strip(),
                    "verdict": str(raw_item.get("verdict", "")).strip().lower() or "context_only",
                    "evidence_strength": str(raw_item.get("evidence_strength", "")).strip().lower() or "medium",
                    "summary": str(raw_item.get("summary", "")).strip(),
                    "limitation_text": str(raw_item.get("limitation_text", "")).strip(),
                    "citation_text": str(raw_item.get("citation_text", "")).strip(),
                    "evidence": evidence,
                    "figure_ids": figure_ids,
                    "supporting_section_ids": supporting_section_ids,
                }
            )
        return {"items": normalized_items}

    def _build_extraction_prompt_payload(
        self,
        *,
        topic_name: str,
        paper_title: str,
        prompt_sections: list[dict],
        prompt_figures: list[dict],
        planned_cards: list[dict],
        planning_context: dict[str, Any],
        calibration_examples: list[dict],
        stage_examples: list[dict],
        calibration_set_name: str,
        active_memory: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        context_sections = [section for section in prompt_sections if section.get("role_hint") == "context"]
        primary_sections = [section for section in prompt_sections if section.get("role_hint") == "primary"]
        supporting_sections = [section for section in prompt_sections if section.get("role_hint") == "supporting"]
        paper_relevance_verdict = normalize_paper_relevance_verdict(
            planning_context.get("paper_relevance_verdict"),
            default="on_topic",
        )
        paper_relevance_reason = str(planning_context.get("paper_relevance_reason", "")).strip()
        relevance_failure_type = normalize_relevance_failure_type(
            planning_context.get("relevance_failure_type"),
            paper_relevance_verdict,
        )
        return {
            "topic": topic_name,
            "paper_title": paper_title,
            "stage": "candidate_extraction",
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "output_language": "zh-CN",
            "active_calibration_set": calibration_set_name,
            "stage_examples": stage_examples,
            "calibration_examples": calibration_examples,
            "active_preference_memory": active_memory or {},
            "paper_relevance_verdict": paper_relevance_verdict,
            "paper_relevance_reason": paper_relevance_reason,
            "relevance_failure_type": relevance_failure_type,
            "sections": prompt_sections,
            "context_sections": context_sections,
            "primary_candidate_sections": primary_sections,
            "supporting_sections": supporting_sections,
            "figures": prompt_figures,
            "planned_card_slots": [
                {
                    "plan_id": str(item.get("plan_id", "")).strip(),
                    "level": str(item.get("level", "")).strip(),
                    "target_object_id": str(item.get("target_object_id", "")).strip(),
                    "target_object_label": str(item.get("target_object_label", "")).strip(),
                    "why_valuable_for_course": str(item.get("why_valuable_for_course", "")).strip(),
                    "must_have_evidence_ids": [str(section_id).strip() for section_id in item.get("must_have_evidence_ids", [])],
                    "optional_supporting_ids": [str(section_id).strip() for section_id in item.get("optional_supporting_ids", [])],
                    "must_have_figure_ids": [str(figure_id).strip() for figure_id in item.get("must_have_figure_ids", [])],
                }
                for item in planned_cards
                if str(item.get("plan_id", "")).strip()
            ],
            "content_rules": {
                "candidate_should_look_like": [
                    "A learner-facing aha candidate that stays faithful to one concrete paper object and one causal reconstruction.",
                    "One card should focus on one atomic pattern or one atomic data finding.",
                    "The card should be teachable with short transfer distance, not a summary or a generalized life principle.",
                    "Prefer a workflow, decision point, failure mode, comparison structure, mechanism, or evidence-backed data point that the learner can picture directly as the new explanation for a familiar experience.",
                    "Ground the card in the provided section_ids, and include figure_ids when a figure materially supports or anchors the idea.",
                    "If planned_card_slots are provided, each surviving candidate must clearly match exactly one slot.",
                    "If planned_card_slots are empty, emit zero cards.",
                ],
                "card_shape_rules": [
                    "Write all card-facing strings in Simplified Chinese.",
                    "Keep the original paper evidence as the main body material by selecting the right section_ids rather than rewriting the paper.",
                    "Use evidence_analysis to add only very short 1-2 sentence explanations of why each cited part matters for teaching.",
                    "Treat draft_body as a short bridge, not the main body of the card.",
                    "Prefer a strong specific title over a paper-topic label.",
                    "Primary evidence should come from mechanism, model, method, result, failure-mode, or data sections; abstract or framing sections can only be supporting context.",
                    "If body evidence is available in sections, do not use abstract or front matter as the only primary evidence.",
                    "If you cannot name the paper-specific object being taught, emit no card.",
                    "If you cannot state the learner's old model and the paper's new model in plain words, emit no card.",
                    "Do not convert technical evidence into broad management slogans, motivational advice, or generic productivity principles.",
                    "If a candidate requires multiple layers of explanation before a practical learner can use it, reject it in extraction.",
                    "course_transformation will later name how the source object is presented in the course; do not abstract the object upward here.",
                ],
                "judgement_boundary_hints": [
                    "Look for a clear learner shift: deeper mechanism, counterintuitive claim, tacit-to-explicit explanation, or strong action value, but remember the ontology is causal reconstruction first.",
                    "Prefer ideas with business relevance and presentation usefulness, not only academic validity.",
                    "Keep direct-transfer source patterns even when they are not flashy, as long as a practical learner can understand and reuse them quickly.",
                    "A mere taxonomy recap, literature background, or weak-transfer technical detail should be rejected here because it is information gain without causal reconstruction.",
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
                    "Candidates that do not clearly match any provided planned_card_slot.",
                ],
                "zero_slot_behavior": [
                    "When planned_card_slots are empty, do not invent a rescue card.",
                    "Use excluded_content to record why this paper or this remaining content should not continue.",
                    "Reuse the paper_relevance_verdict and relevance_failure_type when they already explain the exclusion.",
                ],
            },
            "output_schema": {
                "cards": [
                    {
                        "source_plan_id": "plan_obj_1",
                        "title": "中文卡片标题",
                        "primary_section_ids": ["section_id"],
                        "supporting_section_ids": ["section_id"],
                        "figure_ids": ["figure_id"],
                        "granularity_level": "framework|subpattern|detail",
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
                        "exclusion_type": "background|summary|weak_transfer|wrong_audience|replaced_by_stronger_card|insufficient_evidence|pure_technical_mismatch|cannot_name_course_object|long_transfer_distance|taxonomy_not_insight|weak_method_or_data|low_hanging_fruit|topic_word_overlap_only|other",
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
        active_memory: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return {
            "topic": topic_name,
            "paper_title": paper_title,
            "stage": "candidate_judgement",
            "prompt_version": JUDGEMENT_PROMPT_VERSION,
            "output_language": "zh-CN",
            "active_calibration_set": calibration_set_name,
            "stage_examples": stage_examples,
            "calibration_examples": calibration_examples,
            "active_preference_memory": active_memory or {},
            "candidates": prompt_candidates,
            "judgement_rules": {
                "must_be_true_for_a_card": [
                    "The candidate expresses a real learner-facing shift instead of a paper takeaway.",
                    "The shift is a causal reconstruction of an already-lived learner experience, not just an interesting observation.",
                    "The candidate names one paper-specific object: model, mechanism, method, result, failure mode, framework, or data finding.",
                    "The candidate lets you state the learner's old causal model and the paper's new causal model in plain language.",
                    "The evidence strength matches the claim strength.",
                    "The target learner plausibly still holds the old model in 2026; otherwise the aha is weak even if the paper is correct.",
                    "At least one of these is clearly present: deeper mechanism, counterintuitive claim, tacit-to-explicit explanation, or strong action value.",
                    "The card stays close to the paper's source object instead of drifting into a broader principle.",
                    "If the candidate came from a planned slot, it still matches that planned object instead of sliding to a sibling object from the same section.",
                    "Only after the aha is real should the card be named as a concrete course object, pattern, story, exercise, comparison, or evidence-backed talking point.",
                    "The transfer distance to course use is short enough to teach.",
                    "A practical learner should be able to understand what to picture, say, or do after one slide of explanation.",
                    "The candidate is not merely a technically correct structure; it produces a reportable learner shift for a low-patience non-specialist audience.",
                    "Among sibling candidates from the same paper, this candidate is one of the strongest tellable ahas rather than a weaker rephrasing.",
                    "A same-paper child mechanism, sub-step, or local slice of the main causal chain is not an independent aha unless it changes a different lived experience with a different old model and a different course use.",
                    "A metrics-only, benchmark-only, or evidence-only card is usually support for a mechanism card, not a separate aha, unless it independently reconstructs a different active learner belief.",
                ],
                "business_and_teaching_rules": [
                    "Judge from the learner and course-buying perspective, not only from academic importance.",
                    "Business usefulness and one-slide clarity are secondary filters after the causal reconstruction is already real.",
                    "Prefer ideas that work on one slide with one strong line and, when available, one supporting figure.",
                    "If you cannot say what this becomes in the course, the card should not pass as green.",
                    "If you cannot say why a leader would care in one sentence, the card should not pass as green.",
                    "Explicitly state whether primary evidence is body evidence or merely abstract or front-matter framing.",
                    "course_transformation should name the course form of the source object, not rewrite it into a more generic doctrine.",
                    "If the strongest version of the card sounds like a principle that could have been written without this paper, downgrade or reject it.",
                    "If the card would need minutes of setup before the learner cares, downgrade or reject it.",
                    "If linked_figures materially support the candidate, keep the figure attachment instead of silently dropping it.",
                ],
                "evidence_translation_rules": [
                    "For every evidence quote, provide a complete Simplified Chinese translation in quote_zh.",
                    "Translate the full quoted evidence, not only the claim-relevant fragment.",
                    "Do not summarize, compress, paraphrase away caveats, or keep only the punchline.",
                    "If the quote contains multiple sentences, lists, or numbered parts, translate all of them in order.",
                    "Preserve informational scope, caveats, enumerations, and logical structure from the English quote.",
                ],
                "color_decision_sequence": [
                    "Step 1 (hard gate): Is there genuine causal reconstruction — a real old-model-to-new-model shift grounded in paper evidence? If No → red, stop. No other dimension can rescue a missing causal reconstruction.",
                    "Step 2 (quality gate): Is the evidence strength proportional to the claim? If evidence is weak relative to the strength of the claimed shift → yellow or red, stop.",
                    "Step 3 (relevance gate): Is the learner's old model still an active prior for the target audience in 2026? If the prior is likely already dead or obvious → yellow, stop.",
                    "Step 4 (usability gate): Is there a clear, short-distance path to course operationalization? If the course use is vague or needs too many extra steps → yellow, stop.",
                    "Step 5: All four gates pass → green.",
                    "Downgrade from green to yellow if any of the following: primary evidence comes only from abstract or front matter rather than body sections; a competing sibling candidate from the same paper is clearly stronger; the course use requires one extra reframing step.",
                    "Evaluate in sequence; stop at the first gate that fails. Do not average across dimensions.",
                ],
                "color_rules": {
                    "green": "All four gates pass: clear old-model-to-new-model reconstruction, evidence proportional to claim, active prior, and clear course use.",
                    "yellow": "Passes the causal reconstruction gate but fails or is uncertain on evidence quality, active prior, or course operationalization.",
                    "red": "Fails the causal reconstruction gate, or is mostly information gain, obviousness, genericity, wrong-audience fit, or too-indirect-to-teach content.",
                },
                "must_be_downgraded_or_rejected": [
                    "The candidate is merely background, summary, taxonomy, or weak-transfer detail.",
                    "The audience fit is wrong.",
                    "The idea is academically valid but does not reconstruct a learner's causal explanation.",
                    "The idea is academically valid but not teachable for the target learner.",
                    "The claim feels outdated or already obvious to the target learner.",
                    "The candidate is a framing variant that overlaps with a sibling candidate from the same paper and evidence.",
                    "The candidate is only a child mechanism or zoomed-in subpattern of a stronger same-paper causal chain.",
                    "The candidate mainly adds metrics, benchmark wins, or proof strength to a stronger mechanism card without creating a distinct old-model-to-new-model shift.",
                    "The card relies on principle drift: it sounds smoother after being generalized away from the source evidence.",
                    "The technical-overhang is too high: the learner would need long specialist unpacking before caring.",
                    "The candidate is a same-paper rephrasing of a stronger sibling card.",
                    "The framework or architecture is clear, but the causal reconstruction is weak.",
                    "The benchmark or result is specific but still not naturally teachable to this audience.",
                ],
            },
            "required_judgement_questions": [
                "What is the concrete paper object here?",
                "If a planned slot exists, does this card still match that specific planned object?",
                "What already-lived learner experience is this card about?",
                "What old causal model would the learner likely use here?",
                "What new causal model does the paper supply instead?",
                "Is this claim grounded in body evidence or only paper framing?",
                "Is the learner's old model still active in 2026 for this audience?",
                "What learner belief or vague intuition does this card change or clarify?",
                "Does this candidate remain distinct from sibling candidates within the same paper or topic?",
                "Is this candidate a truly independent aha, or only a submechanism / zoomed-in slice of a stronger same-paper causal chain?",
                "Is the evidence strength proportional to the claim strength?",
                "Would a practical learner still understand the source object without extra technical unpacking?",
                "Has the course naming stayed close to the source object instead of abstracting it upward?",
                "Is this one of the strongest true ahas in the paper, or just a technically valid leftover candidate?",
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
        planned_cards: list[dict],
    ) -> dict[str, list[dict]]:
        raw_cards = payload.get("cards", [])
        raw_excluded = payload.get("excluded_content", [])
        if not isinstance(raw_cards, list):
            raise LLMGenerationError("LLM card payload must contain a list named 'cards'")
        if not isinstance(raw_excluded, list):
            raise LLMGenerationError("LLM card payload must contain a list named 'excluded_content'")

        section_map = {section["id"]: section for section in sections}
        figure_id_set = {figure["id"] for figure in figures}
        plan_map = {
            str(item.get("plan_id", "")).strip(): item
            for item in planned_cards
            if isinstance(item, dict) and str(item.get("plan_id", "")).strip()
        }
        normalized = []
        for raw_card in raw_cards[:MAX_EXTRACTED_CARDS]:
            if not isinstance(raw_card, dict):
                continue
            title = str(raw_card.get("title", "")).strip()
            draft_body = str(raw_card.get("draft_body", "")).strip()
            granularity_level = str(raw_card.get("granularity_level", "subpattern")).strip().lower()
            primary_section_ids = raw_card.get("primary_section_ids", [])
            supporting_section_ids = raw_card.get("supporting_section_ids", [])
            section_ids = list(primary_section_ids) + [
                sid for sid in supporting_section_ids if sid not in set(primary_section_ids)
            ] if isinstance(primary_section_ids, list) and isinstance(supporting_section_ids, list) else []
            raw_figure_ids = raw_card.get("figure_ids", [])
            raw_evidence_analysis = raw_card.get("evidence_analysis", [])
            source_plan_id = str(raw_card.get("source_plan_id", "")).strip()
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
                        "quote": normalize_evidence_paragraph(section["paragraph_text"]),
                        "quote_zh": "",
                        "page_number": section["page_number"],
                        "analysis": analysis_text,
                    }
                )

            if not title or not evidence:
                continue
            if source_plan_id and source_plan_id not in plan_map:
                source_plan_id = ""
            if not source_plan_id and len(plan_map) == 1:
                source_plan_id = next(iter(plan_map.keys()))
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
                    "claim_type": "other",
                    "paper_specific_object": "",
                    "body_grounding_reason": "",
                    "evidence_level": "medium",
                    "possible_duplicate_signature": possible_duplicate_signature,
                    "source_plan_id": source_plan_id,
                    "planned_object_label": str((plan_map.get(source_plan_id) or {}).get("target_object_label", "")).strip(),
                    "planned_why_valuable": str((plan_map.get(source_plan_id) or {}).get("why_valuable_for_course", "")).strip(),
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
                "pure_technical_mismatch",
                "cannot_name_course_object",
                "long_transfer_distance",
                "taxonomy_not_insight",
                "weak_method_or_data",
                "low_hanging_fruit",
                "topic_word_overlap_only",
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
            claim_type = str(raw_card.get("claim_type", "other")).strip().lower() or "other"
            if claim_type not in {"mechanism", "model", "method", "result", "failure_mode", "framework", "data_finding", "other"}:
                claim_type = "other"
            evidence_level = str(raw_card.get("evidence_level", "medium")).strip().lower() or "medium"
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
                    "paper_specific_object": str(raw_card.get("paper_specific_object", "")).strip(),
                    "claim_type": claim_type,
                    "evidence_level": evidence_level,
                    "body_grounding_reason": str(raw_card.get("body_grounding_reason", "")).strip(),
                    "grounding_quality": grounding_quality,
                    "possible_duplicate_signature": extracted.get("possible_duplicate_signature", ""),
                    "source_plan_id": extracted.get("source_plan_id", ""),
                    "planned_object_label": extracted.get("planned_object_label", ""),
                    "planned_why_valuable": extracted.get("planned_why_valuable", ""),
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
        raw_objects = payload.get("global_contribution_objects", [])
        paper_relevance_verdict = normalize_paper_relevance_verdict(
            payload.get("paper_relevance_verdict"),
            default="on_topic" if isinstance(raw_objects, list) and raw_objects else "borderline_reject",
        )
        paper_relevance_reason = str(payload.get("paper_relevance_reason", "")).strip()
        relevance_failure_type = normalize_relevance_failure_type(
            payload.get("relevance_failure_type"),
            paper_relevance_verdict,
        )
        figure_ids_by_section: dict[str, list[str]] = {}
        for figure in figures:
            figure_id = str(figure.get("id", "")).strip()
            if not figure_id:
                continue
            for section_id in figure.get("linked_section_ids", []) or []:
                normalized_section_id = str(section_id).strip()
                if normalized_section_id:
                    figure_ids_by_section.setdefault(normalized_section_id, []).append(figure_id)
        raw_graph = payload.get("contribution_graph", [])
        raw_hints = payload.get("candidate_level_hints", {})
        if not isinstance(raw_objects, list):
            raw_objects = []
        if not isinstance(raw_graph, list):
            raw_graph = []
        if not isinstance(raw_hints, dict):
            raw_hints = {}
        if paper_relevance_verdict != "on_topic":
            raw_objects = []
            raw_graph = []
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
                "paper_relevance_verdict": paper_relevance_verdict,
                "paper_relevance_reason": paper_relevance_reason,
                "relevance_failure_type": relevance_failure_type,
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
            "paper_relevance_verdict": paper_relevance_verdict,
            "paper_relevance_reason": paper_relevance_reason,
            "relevance_failure_type": relevance_failure_type,
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
        paper_relevance_verdict = normalize_paper_relevance_verdict(
            payload.get("paper_relevance_verdict", understanding.get("paper_relevance_verdict")),
            default=normalize_paper_relevance_verdict(
                understanding.get("paper_relevance_verdict"),
                default="on_topic" if understanding.get("global_contribution_objects") else "borderline_reject",
            ),
        )
        paper_relevance_reason = str(
            payload.get("paper_relevance_reason", understanding.get("paper_relevance_reason", ""))
        ).strip()
        relevance_failure_type = normalize_relevance_failure_type(
            payload.get("relevance_failure_type", understanding.get("relevance_failure_type")),
            paper_relevance_verdict,
        )
        if paper_relevance_verdict != "on_topic":
            return {
                "paper_relevance_verdict": paper_relevance_verdict,
                "paper_relevance_reason": paper_relevance_reason,
                "relevance_failure_type": relevance_failure_type,
                "planned_cards": [],
                "coverage_report": {"produce": 0, "exclude": 0, "overall": 0, "local": 0, "detail": 0},
            }
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
            "paper_relevance_verdict": paper_relevance_verdict,
            "paper_relevance_reason": paper_relevance_reason,
            "relevance_failure_type": relevance_failure_type,
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
