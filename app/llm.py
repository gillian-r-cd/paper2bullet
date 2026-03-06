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

EXTRACTION_PROMPT_VERSION = "llm-card-extract-v2-zh"
JUDGEMENT_PROMPT_VERSION = "llm-card-judge-v3-zh"
CARD_RUBRIC_VERSION = "llm-card-rubric-v3"
MAX_PROMPT_SECTIONS = 8
MAX_PROMPT_FIGURES = 4
MAX_CALIBRATION_EXAMPLES = 6


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
        system_prompt = (
            "You extract learner-facing aha-moment candidates from academic paper evidence for course design. "
            "You are not a paper summarizer and you must not output generic takeaways. "
            "Return strict JSON only. Never invent evidence, figures, or course use. "
            "All learner-facing strings must be written in Simplified Chinese. "
            "Prefer 0-3 candidate cards. At this stage, identify candidate cards and explicit excluded content only. "
            "Do not assign final green/yellow/red judgement yet."
        )
        user_prompt = json.dumps(
            self._build_extraction_prompt_payload(
                topic_name=topic_name,
                paper_title=paper_title,
                prompt_sections=prompt_sections,
                prompt_figures=prompt_figures,
                calibration_examples=selected_examples,
                calibration_set_name=calibration_set_name,
            ),
            ensure_ascii=False,
            indent=2,
        )
        payload = self.client.chat_json(system_prompt, user_prompt)
        return self._normalize_extraction_output(payload, sections, figures or [])

    def judge_candidates(
        self,
        *,
        topic_name: str,
        paper_title: str,
        extracted_cards: list[dict],
        calibration_examples: list[dict],
        calibration_set_name: str = "",
    ) -> dict[str, list[dict]]:
        if not self.client or not extracted_cards:
            return {"cards": []}

        selected_examples = self._select_calibration_examples(calibration_examples, topic_name)
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
                }
            )

        system_prompt = (
            "You judge candidate aha-moment cards for course design. "
            "You receive extracted candidates plus calibration examples that define the desired judgement boundary. "
            "Return strict JSON only. Never invent evidence. "
            "All learner-facing strings must be written in Simplified Chinese. "
            "Use the calibration examples to decide whether each candidate creates a real learner-facing cognitive shift, "
            "what it becomes in the course, and whether it is green, yellow, or red under the rubric."
        )
        user_prompt = json.dumps(
            self._build_judgement_prompt_payload(
                topic_name=topic_name,
                paper_title=paper_title,
                prompt_candidates=prompt_candidates,
                calibration_examples=selected_examples,
                calibration_set_name=calibration_set_name,
            ),
            ensure_ascii=False,
            indent=2,
        )
        payload = self.client.chat_json(system_prompt, user_prompt)
        return {"cards": self._normalize_judged_cards(payload, extracted_cards)}

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
            calibration_examples=calibration_examples or [],
            calibration_set_name=calibration_set_name,
        )
        return {
            "cards": judged["cards"],
            "excluded_content": extracted["excluded_content"],
        }

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
        if not self.client:
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
            "model": self.client.model,
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
        calibration_set_name: str,
    ) -> dict[str, Any]:
        return {
            "topic": topic_name,
            "paper_title": paper_title,
            "stage": "candidate_extraction",
            "output_language": "zh-CN",
            "active_calibration_set": calibration_set_name,
            "calibration_examples": calibration_examples,
            "sections": prompt_sections,
            "figures": prompt_figures,
            "content_rules": {
                "candidate_should_look_like": [
                    "A learner-facing aha candidate that reveals a gap between what the learner would normally assume and what the paper evidence implies instead.",
                    "Something teachable in a course with short transfer distance, not a generic summary.",
                    "A pattern, framework, mechanism, or evidence-backed data point that can later become a concrete course object.",
                    "Grounded in the provided section_ids, and optionally linked to provided figure_ids when a figure materially supports the idea.",
                ],
                "card_shape_rules": [
                    "Write all card-facing strings in Simplified Chinese.",
                    "Keep the original paper evidence as the primary body material by selecting the right section_ids rather than rewriting the paper.",
                    "Use evidence_analysis to add only very short 1-2 sentence explanations of why each cited part matters for teaching.",
                    "Prefer a strong specific title over a paper-topic label.",
                    "If you cannot articulate a learner-facing candidate, emit no card.",
                ],
                "judgement_boundary_hints": [
                    "Look for belief-gap, counterintuitive, tacit-to-explicit, or highly actionable insight candidates.",
                    "Prefer ideas with commercial relevance and presentation usefulness, not only academic validity.",
                    "A mere taxonomy recap, literature background, or weak-transfer technical detail should be rejected here.",
                ],
                "should_be_rejected_here": [
                    "Background theory or literature review context.",
                    "Classification or taxonomy recap without a sharp insight.",
                    "Generic paper summary or conclusion.",
                    "Technical detail that is academically valid but too far from course use.",
                    "Policy or management recommendation aimed at the wrong audience.",
                    "Old low-hanging-fruit claims that are already obvious to the target learner.",
                ],
            },
            "output_schema": {
                "cards": [
                    {
                        "title": "中文卡片标题",
                        "section_ids": ["section_id"],
                        "figure_ids": ["figure_id"],
                        "granularity_level": "framework|subpattern|detail",
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
        calibration_set_name: str,
    ) -> dict[str, Any]:
        return {
            "topic": topic_name,
            "paper_title": paper_title,
            "stage": "candidate_judgement",
            "output_language": "zh-CN",
            "active_calibration_set": calibration_set_name,
            "calibration_examples": calibration_examples,
            "candidates": prompt_candidates,
            "judgement_rules": {
                "must_be_true_for_a_card": [
                    "The candidate expresses a real learner-facing cognitive shift instead of a paper takeaway.",
                    "The idea can be named as a concrete course object, framework, pattern, story, or evidence-backed talking point.",
                    "The transfer distance to course use is short enough to teach.",
                    "The evidence strength matches the claim strength.",
                    "At least one of these qualities is present: belief-gap, counterintuitive, tacit-to-explicit, or highly actionable.",
                ],
                "business_and_teaching_rules": [
                    "Judge from the learner and course-design perspective, not only from academic importance.",
                    "Prefer ideas that would make sense on one slide with one strong line and, when available, one supporting figure.",
                    "If you cannot say what this becomes in the course, the card should not pass as green.",
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
                ],
            },
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
            }
            for section in sections[:MAX_PROMPT_SECTIONS]
        ]

    def _build_prompt_figures(self, figures: list[dict]) -> list[dict]:
        return [
            {
                "figure_id": figure["id"],
                "figure_label": figure.get("figure_label", ""),
                "caption": figure.get("caption", ""),
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
        for raw_card in raw_cards[:3]:
            if not isinstance(raw_card, dict):
                continue
            title = str(raw_card.get("title", "")).strip()
            draft_body = str(raw_card.get("draft_body", "")).strip()
            granularity_level = str(raw_card.get("granularity_level", "subpattern")).strip().lower()
            section_ids = raw_card.get("section_ids", [])
            raw_figure_ids = raw_card.get("figure_ids", [])
            raw_evidence_analysis = raw_card.get("evidence_analysis", [])

            if not isinstance(section_ids, list) or not isinstance(raw_evidence_analysis, list) or not isinstance(raw_figure_ids, list):
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
                        "quote_zh": "",
                        "page_number": section["page_number"],
                        "analysis": analysis_by_section.get(section["id"], ""),
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
            if not all(is_readable_text(item["analysis"]) for item in evidence):
                continue

            normalized.append(
                {
                    "title": title,
                    "granularity_level": granularity_level if granularity_level in {"framework", "subpattern", "detail"} else "subpattern",
                    "draft_body": draft_body,
                    "evidence": evidence,
                    "figure_ids": figure_ids,
                    "status": "candidate",
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

            evidence = []
            for evidence_item in extracted["evidence"]:
                quote_zh = quote_zh_by_section.get(evidence_item["section_id"], evidence_item.get("analysis", "").strip())
                if not looks_like_complete_translation(evidence_item["quote"], quote_zh):
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

    def _select_calibration_examples(
        self,
        calibration_examples: list[dict],
        topic_name: str,
        limit: int = MAX_CALIBRATION_EXAMPLES,
    ) -> list[dict]:
        if not calibration_examples:
            return []
        same_topic = []
        fallback = []
        target = topic_name.strip().lower()
        for example in calibration_examples:
            compact = {
                "example_type": example.get("example_type", ""),
                "topic_name": example.get("topic_name", ""),
                "audience": example.get("audience", ""),
                "title": example.get("title", ""),
                "source_text": example.get("source_text", ""),
                "evidence": example.get("evidence", [])[:2],
                "expected_cards": example.get("expected_cards", []),
                "expected_exclusions": example.get("expected_exclusions", []),
                "rationale": example.get("rationale", ""),
                "tags": example.get("tags", []),
            }
            if str(example.get("topic_name", "")).strip().lower() == target:
                same_topic.append(compact)
            else:
                fallback.append(compact)
        selected = same_topic[:limit]
        if len(selected) < limit:
            selected.extend(fallback[: limit - len(selected)])
        return selected

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
