"""
This module provides optional LLM-backed card generation for the Paper to Bullet application.
Main classes: provider-specific LLM clients and `LLMCardEngine`.
Data structures: provider request payloads and normalized card JSON objects.
"""
from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from .config import Settings


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


def is_readable_text(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 8:
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
        except urllib.error.HTTPError as error:
            detail = read_http_error_body(error)
            message = f"LLM request failed with HTTP {error.code} at {endpoint}"
            if detail:
                message += f": {detail}"
            raise LLMGenerationError(message) from error
        except urllib.error.URLError as error:
            raise LLMGenerationError(describe_url_error(error, endpoint)) from error
        except json.JSONDecodeError as error:
            raise LLMGenerationError("LLM response was not valid JSON") from error

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

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        endpoint = self.base_url.rstrip("/") + "/messages"
        payload = {
            "model": self.model,
            "system": system_prompt,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": user_prompt}],
        }
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
        except urllib.error.HTTPError as error:
            detail = read_http_error_body(error)
            message = f"Anthropic request failed with HTTP {error.code} at {endpoint}"
            if detail:
                message += f": {detail}"
            raise LLMGenerationError(message) from error
        except urllib.error.URLError as error:
            raise LLMGenerationError(describe_url_error(error, endpoint)) from error
        except json.JSONDecodeError as error:
            raise LLMGenerationError("Anthropic response was not valid JSON") from error

        try:
            content_blocks = body["content"]
            text_parts = [block["text"] for block in content_blocks if block.get("type") == "text"]
            content = "\n".join(text_parts)
        except (KeyError, TypeError) as error:
            raise LLMGenerationError("Anthropic response did not contain text content") from error
        return extract_json_object(content)


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
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = read_http_error_body(error)
            message = f"Gemini request failed with HTTP {error.code} at {endpoint}"
            if detail:
                message += f": {detail}"
            raise LLMGenerationError(message) from error
        except urllib.error.URLError as error:
            raise LLMGenerationError(describe_url_error(error, endpoint)) from error
        except json.JSONDecodeError as error:
            raise LLMGenerationError("Gemini response was not valid JSON") from error

        try:
            parts = body["candidates"][0]["content"]["parts"]
            content = "\n".join(part["text"] for part in parts if "text" in part)
        except (KeyError, IndexError, TypeError) as error:
            raise LLMGenerationError("Gemini response did not contain text content") from error
        return extract_json_object(content)


class LLMCardEngine:
    def __init__(self, settings: Settings, client: Optional[BaseLLMClient] = None):
        self.settings = settings
        self.client = client or self._build_client(settings)

    def is_enabled(self) -> bool:
        return self.client is not None

    def generate_outputs(self, *, topic_name: str, paper_title: str, sections: list[dict]) -> dict[str, list[dict]]:
        if not self.client:
            return {"cards": [], "excluded_content": []}
        prompt_sections = [
            {
                "section_id": section["id"],
                "page_number": section["page_number"],
                "text": section["paragraph_text"],
            }
            for section in sections[:8]
        ]
        system_prompt = (
            "You extract atomic, evidence-backed aha-moment candidate cards from academic paper evidence for course design. "
            "You are not a paper summarizer. Return strict JSON only. Never invent evidence. Prefer 0-3 cards. "
            "Only return a card when the evidence supports a learner-facing insight with a clear cognitive shift and a short transfer distance to teaching use. "
            "Reject content that is merely background theory, literature review, taxonomy recap, generic summary, or technical detail that is too far from course use. "
            "For important content that should not become a card, record it in excluded_content with a short reason."
        )
        user_prompt = json.dumps(
            {
                "topic": topic_name,
                "paper_title": paper_title,
                "sections": prompt_sections,
                "content_rules": {
                    "must_be_true_for_a_card": [
                        "The idea is a learner-facing aha, not just a paper takeaway.",
                        "The card can be taught as a concrete course object, frame, pattern, or evidence-backed talking point.",
                        "The evidence directly supports the claim.",
                        "The card can be explained in one sentence to a learner.",
                    ],
                    "must_be_rejected": [
                        "Background theory or literature review context.",
                        "Classification or taxonomy recap without a sharp insight.",
                        "Generic paper summary or conclusion.",
                        "Technical detail that is academically valid but too far from course use.",
                        "Policy or management recommendation aimed at the wrong audience.",
                    ],
                },
                "output_schema": {
                    "cards": [
                        {
                            "title": "human-readable card title",
                            "section_ids": ["section_id"],
                            "granularity_level": "framework|subpattern|detail",
                            "course_transformation": "what it becomes in the course",
                            "teachable_one_liner": "one sentence a teacher can say out loud",
                            "draft_body": "short explanation grounded in evidence, focused on the learner-facing insight",
                            "evidence_analysis": [
                                {
                                    "section_id": "section_id",
                                    "analysis": "1-2 sentence explanation of why this quoted evidence matters",
                                }
                            ],
                            "judgement": {
                                "color": "green|yellow|red",
                                "reason": "short reason",
                            },
                        }
                    ],
                    "excluded_content": [
                        {
                            "label": "short name for rejected content",
                            "section_ids": ["section_id"],
                            "exclusion_type": "background|summary|weak_transfer|wrong_audience|replaced_by_stronger_card|insufficient_evidence|other",
                            "reason": "short explanation of why this should not become a card",
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        payload = self.client.chat_json(system_prompt, user_prompt)
        return self._normalize_outputs(payload, sections)

    def generate_cards(self, *, topic_name: str, paper_title: str, sections: list[dict]) -> list[dict]:
        return self.generate_outputs(
            topic_name=topic_name,
            paper_title=paper_title,
            sections=sections,
        )["cards"]

    def smoke_test(self) -> dict[str, Any]:
        if not self.client:
            raise LLMGenerationError("LLM provider is not enabled")
        sections = [
            {
                "id": "section_demo_1",
                "page_number": 1,
                "paragraph_text": (
                    "We find that delegating subtasks to specialized agents improves solution quality, "
                    "but only when the orchestration step explicitly checks contradictions between agents."
                ),
            },
            {
                "id": "section_demo_2",
                "page_number": 1,
                "paragraph_text": (
                    "In our experiments, adding a verifier agent reduced final-answer inconsistency by 23 percent."
                ),
            },
        ]
        outputs = self.generate_outputs(
            topic_name="LLM agent",
            paper_title="Smoke Test Paper",
            sections=sections,
        )
        return {
            "provider_mode": self.settings.llm_mode,
            "model": self.client.model,
            "card_count": len(outputs["cards"]),
            "cards": outputs["cards"],
            "excluded_content": outputs["excluded_content"],
        }

    def _normalize_outputs(self, payload: dict[str, Any], sections: list[dict]) -> dict[str, list[dict]]:
        raw_cards = payload.get("cards", [])
        raw_excluded = payload.get("excluded_content", [])
        if not isinstance(raw_cards, list):
            raise LLMGenerationError("LLM card payload must contain a list named 'cards'")
        if not isinstance(raw_excluded, list):
            raise LLMGenerationError("LLM card payload must contain a list named 'excluded_content'")

        section_map = {section["id"]: section for section in sections}
        normalized = []
        for raw_card in raw_cards[:3]:
            if not isinstance(raw_card, dict):
                continue
            title = str(raw_card.get("title", "")).strip()
            course_transformation = str(raw_card.get("course_transformation", "")).strip()
            teachable_one_liner = str(raw_card.get("teachable_one_liner", "")).strip()
            draft_body = str(raw_card.get("draft_body", "")).strip()
            granularity_level = str(raw_card.get("granularity_level", "subpattern")).strip().lower()
            section_ids = raw_card.get("section_ids", [])
            raw_evidence_analysis = raw_card.get("evidence_analysis", [])
            judgement = raw_card.get("judgement", {})

            if not isinstance(section_ids, list) or not isinstance(raw_evidence_analysis, list):
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
                evidence.append(
                    {
                        "section_id": section["id"],
                        "quote": section["paragraph_text"],
                        "page_number": section["page_number"],
                        "analysis": analysis_by_section.get(section["id"], ""),
                    }
                )

            if not title or not evidence:
                continue
            if not is_readable_text(title):
                continue
            if not is_readable_text(course_transformation):
                continue
            if not is_readable_text(teachable_one_liner):
                continue
            if not is_readable_text(draft_body):
                continue
            if not all(is_readable_text(item["quote"]) for item in evidence):
                continue
            if not all(is_readable_text(item["analysis"]) for item in evidence):
                continue

            color = str(judgement.get("color", "yellow")).strip().lower()
            if color not in {"green", "yellow", "red"}:
                color = "yellow"
            reason = str(judgement.get("reason", "")).strip() or "LLM judgement did not provide a reason."

            normalized.append(
                {
                    "title": title,
                    "granularity_level": granularity_level if granularity_level in {"framework", "subpattern", "detail"} else "subpattern",
                    "course_transformation": course_transformation,
                    "teachable_one_liner": teachable_one_liner,
                    "draft_body": draft_body,
                    "evidence": evidence,
                    "figure_ids": [],
                    "status": "candidate",
                    "judgement": {
                        "color": color,
                        "reason": reason,
                        "model_version": self.client.model,
                        "prompt_version": "llm-card-generator-v1",
                        "rubric_version": "llm-card-rubric-v1",
                    },
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
