"""
This module implements the Phase 0 workflow for the Paper to Bullet application.
Main classes: `Repository`, `PaperPipeline`, and `RunCoordinator`.
Data structures: runs, topic jobs, papers, sections, candidate cards, judgements, access queue items, and export artifacts.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import mimetypes
import re
import shutil
import sqlite3
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional
from uuid import uuid4

from .config import Settings
from .db import db_cursor, db_read_cursor
from .llm import (
    CARD_RUBRIC_VERSION,
    CARD_PLAN_PROMPT_VERSION,
    EXTRACTION_PROMPT_VERSION,
    JUDGEMENT_PROMPT_VERSION,
    LLMCardEngine,
    LLMGenerationError,
    get_prompt_version_records,
    get_rubric_version_records,
    looks_like_complete_translation,
)

try:
    from markitdown import MarkItDown
except ImportError:  # pragma: no cover - optional dependency
    MarkItDown = None

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


PAPER_CONTENT_BASIS_PARSED_FULLTEXT = "parsed_fulltext"
PAPER_CONTENT_BASIS_ABSTRACT_ONLY = "abstract_only"
PAPER_CONTENT_BASIS_UNAVAILABLE = "unavailable"

PAPER_QA_STATUS_READY = "ready"
PAPER_QA_STATUS_BLOCKED_ABSTRACT_ONLY = "blocked_abstract_only"
PAPER_QA_STATUS_BLOCKED_NO_PARSED_SECTIONS = "blocked_no_parsed_sections"

PAPER_QA_STATUS_MESSAGES = {
    PAPER_QA_STATUS_READY: "Full-paper QA is available.",
    PAPER_QA_STATUS_BLOCKED_ABSTRACT_ONLY: (
        "This paper currently only has abstract-level evidence. Full-paper QA is unavailable until full text is acquired and parsed."
    ),
    PAPER_QA_STATUS_BLOCKED_NO_PARSED_SECTIONS: (
        "This paper does not yet have parsed full-text sections, so full-paper QA is unavailable."
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "item"


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_identifier(value: str) -> str:
    cleaned = urllib.parse.unquote(str(value or "").strip()).lower()
    cleaned = re.sub(r"^https?://(dx\.)?doi\.org/", "", cleaned)
    cleaned = re.sub(r"^doi:", "", cleaned)
    cleaned = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", cleaned)
    cleaned = cleaned.removesuffix(".pdf")
    cleaned = cleaned.split("?", 1)[0].strip().strip("/")
    return cleaned


def normalize_title_key(title: str) -> str:
    lowered = str(title or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def has_concept_belief_gap_signal(*texts: str) -> bool:
    combined = " ".join(str(text or "").lower() for text in texts if str(text or "").strip())
    if not combined:
        return False
    direct_markers = [
        "不是",
        "而是",
        "不代表",
        "误以为",
        "以为",
        "但",
        "却",
        "反而",
        "别只",
        "instead",
        "rather than",
        "not ",
        "but ",
        "however",
    ]
    if any(marker in combined for marker in direct_markers):
        return True
    if re.search(r"从.+到.+", combined):
        return True
    return False


def _normalized_signal_text(*texts: str) -> str:
    combined = " ".join(str(text or "") for text in texts if str(text or "").strip())
    combined = combined.lower()
    combined = re.sub(r"\s+", "", combined)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", combined)


def _signal_ngrams(text: str) -> set[str]:
    cleaned = _normalized_signal_text(text)
    grams: set[str] = set()
    for token in re.findall(r"[a-z0-9]{3,}", cleaned):
        grams.add(token)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
    for run in cjk_runs:
        max_size = min(4, len(run))
        for size in range(2, max_size + 1):
            for index in range(0, len(run) - size + 1):
                grams.add(run[index : index + size])
    return grams


def has_direct_transfer_signal(*texts: str) -> bool:
    combined = " ".join(str(text or "").lower() for text in texts if str(text or "").strip())
    if not combined:
        return False
    direct_markers = [
        "流程",
        "步骤",
        "闭环",
        "回路",
        "工作流",
        "模板",
        "检查表",
        "决策树",
        "打分",
        "排序",
        "分工",
        "并行",
        "写回",
        "检索",
        "对比",
        "失败模式",
        "接口",
        "playbook",
        "workflow",
        "pipeline",
        "loop",
        "checklist",
        "decision",
        "scoring",
        "ranking",
        "retrieval",
        "write-back",
    ]
    if any(marker in combined for marker in direct_markers):
        return True
    return bool(re.search(r"(→|->|=>|vs\.?|versus|从.+到.+)", combined))


def has_source_object_fidelity_signal(
    course_transformation: str,
    title: str,
    paper_specific_object: str,
) -> bool:
    source_grams = _signal_ngrams(f"{title} {paper_specific_object}")
    course_grams = _signal_ngrams(course_transformation)
    if not source_grams or not course_grams:
        return False
    shared = source_grams.intersection(course_grams)
    strong_shared = {token for token in shared if len(token) >= 3}
    if len(strong_shared) >= 2:
        return True
    if any(re.fullmatch(r"[a-z0-9-]{4,}", token) for token in shared):
        return True
    return False


def has_named_course_object_signal(course_transformation: str) -> bool:
    text = str(course_transformation or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(token in lowered for token in ["摘要", "概述", "背景", "综述", "taxonomy recap", "summary only"]):
        return False
    if "：" in text or ":" in text:
        return True
    object_tokens = [
        "框架",
        "模式",
        "方法",
        "模板",
        "检查表",
        "决策树",
        "清单",
        "机制",
        "策略",
        "流程",
        "模型",
        "回路",
        "闭环",
        "打分器",
        "对比讲解",
        "story",
    ]
    return any(token in lowered for token in object_tokens)


def compute_plan_object_match_score(plan_item: dict[str, Any], card: dict[str, Any]) -> float:
    plan_text = " ".join(
        [
            str(plan_item.get("target_object_label", "")).strip(),
            str(plan_item.get("why_valuable_for_course", "")).strip(),
        ]
    )
    card_text = " ".join(
        [
            str(card.get("title", "")).strip(),
            str(card.get("paper_specific_object", "")).strip(),
            str(card.get("course_transformation", "")).strip(),
            str(card.get("teachable_one_liner", "")).strip(),
            str(card.get("planned_object_label", "")).strip(),
        ]
    )
    plan_grams = _signal_ngrams(plan_text)
    card_grams = _signal_ngrams(card_text)
    if not plan_grams or not card_grams:
        return 0.0
    shared = plan_grams.intersection(card_grams)
    strong_shared = {
        token
        for token in shared
        if len(token) >= 3 or any("\u4e00" <= char <= "\u9fff" for char in token)
    }
    score = float(len(strong_shared))
    normalized_plan = _normalized_signal_text(plan_text)
    normalized_card = _normalized_signal_text(card_text)
    if normalized_plan and normalized_plan in normalized_card:
        score += 3.0
    return score


def build_quote_first_blocks(card: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for evidence in card.get("evidence", []) or []:
        if not isinstance(evidence, dict):
            continue
        quote_zh = str(evidence.get("quote_zh", "")).strip()
        quote_en = str(evidence.get("quote", "")).strip()
        analysis = str(evidence.get("analysis", "")).strip()
        primary_quote = quote_zh or quote_en
        if not primary_quote:
            continue
        blocks.append(
            {
                "section_id": str(evidence.get("section_id", "")).strip(),
                "quote": primary_quote,
                "quote_en": quote_en,
                "analysis": analysis,
            }
        )
    return blocks


def render_quote_first_markdown(card: dict[str, Any]) -> str:
    blocks = build_quote_first_blocks(card)
    lines = ["原文（穿插分析）："]
    if not blocks:
        summary = str(card.get("draft_body", "")).strip()
        if summary:
            lines.append(f"- {summary}")
        return "\n".join(lines)
    for block in blocks:
        lines.append(f"- {block['quote']}")
        if block.get("analysis"):
            lines.append(f"  *→ {block['analysis']}*")
    return "\n".join(lines)


def build_discovery_identity(title: str, publication_year: Optional[int], authors: list[str], ids: dict[str, str]) -> str:
    doi = normalize_identifier(ids.get("doi", ""))
    if doi:
        return f"doi::{doi}"
    arxiv_id = normalize_identifier(ids.get("arxiv", ""))
    if arxiv_id:
        return f"arxiv::{arxiv_id}"
    openalex_id = normalize_identifier(ids.get("openalex", ""))
    if openalex_id:
        return f"openalex::{openalex_id}"
    semantic_scholar_id = normalize_identifier(ids.get("semantic_scholar", ""))
    if semantic_scholar_id:
        return f"semanticscholar::{semantic_scholar_id}"
    title_key = normalize_title_key(title)
    author_key = "|".join(slugify(author) for author in authors[:2] if author)
    return f"title::{stable_hash(f'{title_key}|{publication_year or 0}|{author_key}')}"


def initial_topic_run_stats() -> dict[str, Any]:
    return {
        "discovered": 0,
        "accessible": 0,
        "cards": 0,
        "matrix_items": 0,
        "parsed_papers": 0,
        "card_generation_attempts": 0,
        "discovered_raw": 0,
        "deduped_candidates": 0,
        "duplicate_candidates_collapsed": 0,
        "discovery_strategy_count": 0,
        "queued_for_access": 0,
        "provider_summary": {},
        "acquisition_errors": 0,
        "paper_processing_errors": 0,
        "processing_warnings": [],
        "failure_log": [],
        "current_stage": "pending",
        "stage_started_at": "",
        "last_progress_at": "",
    }


def parse_iso_datetime(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def seconds_since(value: str) -> Optional[int]:
    parsed = parse_iso_datetime(value)
    if not parsed:
        return None
    return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))


def append_failure_log(stats: dict[str, Any], *, stage: str, code: str, message: str, retryable: bool = True) -> None:
    stats.setdefault("failure_log", []).append(
        {
            "created_at": utc_now(),
            "stage": stage,
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    )


def summarize_latest_failures(stats: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    failures = stats.get("failure_log", []) if isinstance(stats, dict) else []
    if not isinstance(failures, list):
        return []
    return failures[-limit:]


def default_saturation_stop_policy() -> dict[str, Any]:
    return {
        "policy_version": "saturation-stop-v1",
        "minimum_snapshot_count": 2,
        "minimum_duplication_ratio": 0.5,
        "maximum_duplication_ratio_drop": 0.05,
        "require_flattening_signal": True,
        "require_zero_tail": True,
    }


def evaluate_saturation_stop(
    *,
    current_metrics: dict[str, Any],
    previous_snapshots: list[dict[str, Any]],
    policy: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    resolved_policy = policy or default_saturation_stop_policy()
    flattening = current_metrics.get("flattening_signal", {}) if isinstance(current_metrics, dict) else {}
    tail = []
    if isinstance(flattening, dict):
        tail = flattening.get("tail_incremental_new_aha_classes", []) or flattening.get("tail_incremental_new_cards", [])
    latest_duplication_ratio = float(
        current_metrics.get("aha_class_duplication_ratio", current_metrics.get("semantic_duplication_ratio", 0.0)) or 0.0
    )
    previous_snapshot = previous_snapshots[0] if previous_snapshots else {}
    previous_saturation_metrics = previous_snapshot.get("saturation_metrics", {}) if isinstance(previous_snapshot, dict) else {}
    previous_duplication_ratio = float(
        (
            previous_saturation_metrics.get(
                "aha_class_duplication_ratio",
                previous_snapshot.get("semantic_duplication_ratio", 0.0),
            )
            if isinstance(previous_saturation_metrics, dict)
            else previous_snapshot.get("semantic_duplication_ratio", 0.0)
        )
        or 0.0
    )
    duplication_ratio_delta = round(latest_duplication_ratio - previous_duplication_ratio, 4) if previous_snapshots else 0.0
    snapshot_count = len(previous_snapshots) + 1
    enough_history = snapshot_count >= int(resolved_policy.get("minimum_snapshot_count", 2))
    flattening_met = bool(flattening.get("likely_flattening", False))
    duplication_met = latest_duplication_ratio >= float(resolved_policy.get("minimum_duplication_ratio", 0.5))
    zero_tail_met = bool(tail) and all(int(value) == 0 for value in tail)
    allowed_drop = float(resolved_policy.get("maximum_duplication_ratio_drop", 0.05))
    stable_duplication_met = (not previous_snapshots) or duplication_ratio_delta >= (-1 * allowed_drop)

    checks = {
        "snapshot_count": snapshot_count,
        "enough_history": enough_history,
        "flattening_met": flattening_met,
        "duplication_met": duplication_met,
        "zero_tail_met": zero_tail_met,
        "stable_duplication_met": stable_duplication_met,
        "latest_duplication_ratio": latest_duplication_ratio,
        "previous_duplication_ratio": previous_duplication_ratio,
        "duplication_ratio_delta": duplication_ratio_delta,
        "tail_incremental_new_cards": tail,
        "tail_metric": "incremental_new_aha_classes" if flattening.get("tail_incremental_new_aha_classes") else "incremental_new_cards",
    }
    current_metrics["stop_policy"] = resolved_policy
    current_metrics["stop_checks"] = checks

    if not enough_history:
        decision = "insufficient_history"
        reason = f"Need at least {resolved_policy['minimum_snapshot_count']} snapshots before recommending a stop."
    elif (
        (not resolved_policy.get("require_flattening_signal", True) or flattening_met)
        and duplication_met
        and (not resolved_policy.get("require_zero_tail", True) or zero_tail_met)
        and stable_duplication_met
    ):
        decision = "candidate_stop"
        reason = "Recent strategies yielded no new independent aha classes while duplication stayed high and stable."
    else:
        decision = "continue_search"
        blockers = []
        if resolved_policy.get("require_flattening_signal", True) and not flattening_met:
            blockers.append("flattening signal not met")
        if not duplication_met:
            blockers.append("duplication ratio still below threshold")
        if resolved_policy.get("require_zero_tail", True) and not zero_tail_met:
            blockers.append("recent tail still produced new aha classes")
        if not stable_duplication_met:
            blockers.append("duplication ratio dropped too much versus previous run")
        reason = "; ".join(blockers) if blockers else "Signals still suggest further retrieval may pay off."

    decision_payload = {
        "decision": decision,
        "reason": reason,
        "policy": resolved_policy,
        "checks": checks,
    }
    current_metrics["stop_decision"] = decision_payload
    return decision_payload


def build_topic_search_strategies(topic: str, *, current_year: Optional[int] = None) -> list[dict[str, Any]]:
    normalized_topic = str(topic or "").strip()
    if not normalized_topic:
        return []
    year = current_year or datetime.now(timezone.utc).year
    recent_year_from = max(1900, year - 3)
    return [
        {
            "strategy_family": "core",
            "strategy_type": "topic_query",
            "strategy_order": 1,
            "query_text": normalized_topic,
            "params": {},
        },
        {
            "strategy_family": "mechanism",
            "strategy_type": "mechanism_focus",
            "strategy_order": 2,
            "query_text": f"{normalized_topic} mechanism evidence",
            "params": {},
        },
        {
            "strategy_family": "application",
            "strategy_type": "application_focus",
            "strategy_order": 3,
            "query_text": f"{normalized_topic} case study",
            "params": {},
        },
        {
            "strategy_family": "recency",
            "strategy_type": "recent_window",
            "strategy_order": 4,
            "query_text": normalized_topic,
            "params": {"year_from": recent_year_from},
        },
    ]


WORKPLACE_CLAIM_HINT_TOKENS = (
    "leadership",
    "leader",
    "manager",
    "managerial",
    "employee",
    "employees",
    "team",
    "teams",
    "workplace",
    "organization",
    "organizational",
    "management",
    "supervisor",
    "subordinate",
    "follower",
    "followers",
    "staff",
    "job",
    "jobs",
    "engagement",
    "psychological safety",
    "voice",
    "burnout",
    "lmx",
    "领导",
    "管理",
    "经理",
    "员工",
    "团队",
    "组织",
    "绩效",
    "心理安全",
)

WORKPLACE_CONTEXT_QUERY_TERMS = (
    "leadership",
    "manager",
    "employee",
    "workplace",
    "organizational",
)

WORKPLACE_DIMENSION_QUERY_ANCHORS = {
    "expression": "leadership communication sensegiving role clarity goal clarity employee engagement",
    "listening": "supervisor listening employee commitment trust in leader psychological safety communication satisfaction",
    "questioning": "managerial coaching leader questioning inquiry employee development problem solving",
    "empathy": "leader empathy supervisor support emotional intelligence burnout job satisfaction leader-member exchange",
    "action_facilitation": "manager praise goal setting performance feedback accountability follow-up employee performance goal attainment",
    "integrative_framework": "organizational communication leadership employee commitment team performance workplace well-being",
    "boundary_contradictions": "leadership communication boundary conditions culture remote work task interdependence null effects",
    "measurement_methods": "experience sampling daily diary workplace communication supervisor employee conversation behavior coding",
}

WORKPLACE_DIMENSION_QUERY_EXPANSIONS = {
    "expression": (
        "supervisor communication goal clarity role clarity employee engagement",
        "transparent communication sensegiving trust in leader team performance",
    ),
    "listening": (
        "leader listening trust in leader psychological safety employee voice",
        "active empathic listening supervisor employee trust psychological safety",
        "perceived leader listening supervisor listening quality employee voice trust in leader",
    ),
    "questioning": (
        "managerial coaching feedback leadership development employee learning",
        "coach-like questioning inquiry problem solving employee development",
        "executive coaching feedback inquiry leadership development work performance",
        "leader questioning inquiry learning behavior problem solving innovation",
    ),
    "empathy": (
        "servant leadership workplace trust employee wellbeing burnout",
        "leader emotional intelligence trust in supervisor job satisfaction",
    ),
    "action_facilitation": (
        "performance appraisal goal setting organizational performance manager feedback",
        "manager praise feedback goal setting accountability follow-up employee performance",
    ),
    "integrative_framework": (
        "organizational communication leadership employee engagement performance mediation",
        "leadership style workplace wellbeing commitment employee performance",
    ),
    "boundary_contradictions": (
        "virtual teams participative leadership job satisfaction remote work",
        "leadership communication culture task interdependence null effects workplace",
    ),
    "measurement_methods": (
        "experience sampling supervisor behavior daily diary employee wellbeing",
        "family supportive supervisor behavior multilevel workplace interactions",
    ),
}

CLAIM_EVIDENCE_PROVIDER_RESULT_LIMIT = 8
CLAIM_EVIDENCE_DISCOVERY_CANDIDATE_CAP = 30
CLAIM_EVIDENCE_MAX_ABSTRACT_FALLBACK_ITEMS_NO_FULLTEXT = 2
CLAIM_EVIDENCE_MAX_ABSTRACT_FALLBACK_ITEMS_WITH_FULLTEXT = 1
CLAIM_EVIDENCE_MAX_ABSTRACT_FALLBACK_PAPER_ATTEMPTS = 8

LOW_SIGNAL_CLAIM_EVIDENCE_TITLE_TOKENS = (
    "decision letter",
    "review for",
    "review of",
    "retracted:",
    "retraction",
    "corrigendum",
    "erratum",
    "editorial",
    "book review",
    "preface",
)

WORKPLACE_DIRECT_EVIDENCE_TOKENS = (
    "leadership",
    "leader",
    "manager",
    "managerial",
    "employee",
    "employees",
    "workplace",
    "organizational",
    "organization",
    "supervisor",
    "subordinate",
    "follower",
    "followers",
    "staff",
    "worker",
    "workers",
    "management",
    "firm",
    "firms",
    "job satisfaction",
    "trust in leader",
    "employee voice",
    "psychological safety",
    "supervisor support",
    "team performance",
    "leader-member exchange",
    "lmx",
    "burnout",
    "service innovation",
)

OFFDOMAIN_CLAIM_EVIDENCE_TITLE_TOKENS = (
    "robot",
    "robotic",
    "artificial intelligence",
    " ai ",
    "llm",
    "language model",
    "reinforcement learning",
    "goal-conditioned",
    "video model",
    "benchmark",
    "dataset",
    "maze",
    "physics-conditioned",
    "consumer",
    "customer",
    "restaurant",
    "marketing",
    "fitness",
    "physical activity",
    "driver",
    "medical student",
    "medical students",
    "physiotherapy",
    "patient",
    "therapy",
    "music listening",
    "cochlear",
    "hearing",
    "virtual reality",
    "madrasah",
    "crowd worker",
    "crowd workers",
    "tennis",
    "supernova",
    "black hole",
    "qed",
    "qcd",
)


def _joined_lower_text(*parts: Any) -> str:
    return " ".join(str(part or "").strip() for part in parts if str(part or "").strip()).lower()


def infer_claim_evidence_context(
    query_anchor: str,
    *,
    outcome_terms: Optional[list[str]] = None,
    claim_text: str = "",
    research_brief: str = "",
) -> dict[str, Any]:
    combined = _joined_lower_text(query_anchor, claim_text, research_brief, " ".join(outcome_terms or []))
    requires_workplace_context = any(token in combined for token in WORKPLACE_CLAIM_HINT_TOKENS)
    context_terms: list[str] = []
    if requires_workplace_context:
        for term in WORKPLACE_CONTEXT_QUERY_TERMS:
            if term in combined or term in str(query_anchor or "").lower():
                context_terms.append(term)
        for fallback in WORKPLACE_CONTEXT_QUERY_TERMS:
            if fallback not in context_terms:
                context_terms.append(fallback)
    return {
        "requires_workplace_context": requires_workplace_context,
        "context_terms": context_terms[:5],
        "provider_allowlist": ["openalex", "crossref", "semantic_scholar"] if requires_workplace_context else [],
    }


def scope_claim_evidence_query(anchor: str, context_terms: list[str]) -> str:
    normalized_anchor = str(anchor or "").strip()
    if not normalized_anchor or not context_terms:
        return normalized_anchor
    lowered_anchor = normalized_anchor.lower()
    suffix_terms = [term for term in context_terms if term and term.lower() not in lowered_anchor]
    if not suffix_terms:
        return normalized_anchor
    return f"{normalized_anchor} {' '.join(suffix_terms)}".strip()


def _clean_metadata_abstract(text: str) -> str:
    normalized = html.unescape(str(text or ""))
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _reconstruct_openalex_abstract(abstract_index: Any) -> str:
    if not isinstance(abstract_index, dict):
        return ""
    positions: dict[int, str] = {}
    for token, indexes in abstract_index.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions[index] = str(token or "")
    if not positions:
        return ""
    return " ".join(positions[index] for index in sorted(positions))


def _extract_source_metadata_abstract(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    direct_abstract = _clean_metadata_abstract(str(metadata.get("abstract", "") or ""))
    if direct_abstract:
        return direct_abstract
    source_metadata = metadata.get("source_metadata", {})
    if isinstance(source_metadata, dict):
        source_abstract = _clean_metadata_abstract(str(source_metadata.get("abstract", "") or ""))
        if source_abstract:
            return source_abstract
        openalex_abstract = _reconstruct_openalex_abstract(source_metadata.get("abstract_inverted_index"))
        if openalex_abstract:
            return _clean_metadata_abstract(openalex_abstract)
    openalex_abstract = _reconstruct_openalex_abstract(metadata.get("abstract_inverted_index"))
    if openalex_abstract:
        return _clean_metadata_abstract(openalex_abstract)
    return ""


def _extract_candidate_metadata_abstract(candidate: dict[str, Any]) -> str:
    candidates: list[str] = []
    for source in candidate.get("discovery_sources", []):
        abstract_text = _extract_source_metadata_abstract(source.get("metadata", {}))
        if abstract_text:
            candidates.append(abstract_text)
    abstract_text = _extract_source_metadata_abstract(candidate.get("metadata", {}))
    if abstract_text:
        candidates.append(abstract_text)
    if not candidates:
        return ""
    return max(candidates, key=len)


def _keywordize_claim_text(*parts: Any) -> list[str]:
    stopwords = {
        "and",
        "the",
        "for",
        "with",
        "that",
        "from",
        "into",
        "over",
        "using",
        "use",
        "role",
        "effects",
        "effect",
        "study",
        "workplace",
        "organizational",
        "leadership",
        "manager",
        "employee",
    }
    keywords: list[str] = []
    for part in parts:
        cleaned = _clean_metadata_abstract(str(part or "")).lower()
        for token in re.findall(r"[a-z][a-z\-]{2,}", cleaned):
            if token in stopwords:
                continue
            keywords.append(token)
    return list(dict.fromkeys(keywords))


def _title_has_claim_evidence_noise(title: str) -> bool:
    lowered = _clean_metadata_abstract(title).lower()
    if not lowered:
        return False
    if any(token in lowered for token in LOW_SIGNAL_CLAIM_EVIDENCE_TITLE_TOKENS):
        return True
    return False


def build_claim_evidence_search_strategies(
    query_anchor: str,
    *,
    outcome_terms: Optional[list[str]] = None,
    dimension_key: str = "",
    claim_text: str = "",
    research_brief: str = "",
    current_year: Optional[int] = None,
) -> list[dict[str, Any]]:
    normalized_anchor = str(query_anchor or "").strip()
    if not normalized_anchor:
        return []
    normalized_outcomes = []
    for raw_term in outcome_terms or []:
        term = str(raw_term or "").strip()
        if term and term.lower() not in {str(item or "").lower() for item in normalized_outcomes}:
            normalized_outcomes.append(term)
    year = current_year or datetime.now(timezone.utc).year
    recent_year_from = max(1900, year - 3)
    primary_outcome = normalized_outcomes[0] if normalized_outcomes else ""
    secondary_outcome = normalized_outcomes[1] if len(normalized_outcomes) >= 2 else primary_outcome
    context = infer_claim_evidence_context(
        normalized_anchor,
        outcome_terms=normalized_outcomes,
        claim_text=claim_text,
        research_brief=research_brief,
    )
    normalized_dimension_key = str(dimension_key or "").strip().lower()
    dimension_anchor = WORKPLACE_DIMENSION_QUERY_ANCHORS.get(normalized_dimension_key, "")
    dimension_expansions = WORKPLACE_DIMENSION_QUERY_EXPANSIONS.get(normalized_dimension_key, ())
    effective_anchor = dimension_anchor if context["requires_workplace_context"] and dimension_anchor else normalized_anchor
    scoped_anchor = scope_claim_evidence_query(effective_anchor, context["context_terms"])
    provider_allowlist = context["provider_allowlist"]
    evidence_suffix = "empirical study" if context["requires_workplace_context"] else "empirical evidence"
    result_limit = CLAIM_EVIDENCE_PROVIDER_RESULT_LIMIT if context["requires_workplace_context"] else 5
    strategies = [
        {
            "strategy_family": "core",
            "strategy_type": "topic_query",
            "strategy_order": 1,
            "query_text": scoped_anchor,
            "params": {"result_limit": result_limit},
            "provider_allowlist": provider_allowlist,
        },
        {
            "strategy_family": "evidence",
            "strategy_type": "evidence_focus",
            "strategy_order": 2,
            "query_text": f"{scoped_anchor} {evidence_suffix}",
            "params": {"result_limit": result_limit},
            "provider_allowlist": provider_allowlist,
        },
    ]
    for expansion in dimension_expansions:
        scoped_expansion = scope_claim_evidence_query(str(expansion or "").strip(), context["context_terms"])
        if not scoped_expansion or scoped_expansion.lower() == scoped_anchor.lower():
            continue
        strategies.append(
            {
                "strategy_family": "dimension",
                "strategy_type": "dimension_focus",
                "strategy_order": len(strategies) + 1,
                "query_text": scoped_expansion,
                "params": {"result_limit": result_limit},
                "provider_allowlist": provider_allowlist,
            }
        )
    if primary_outcome:
        strategies.append(
            {
                "strategy_family": "outcome",
                "strategy_type": "outcome_focus",
                "strategy_order": 3,
                "query_text": f"{scoped_anchor} {primary_outcome}",
                "params": {"outcome": primary_outcome, "result_limit": result_limit},
                "provider_allowlist": provider_allowlist,
            }
        )
    if secondary_outcome and secondary_outcome != primary_outcome:
        strategies.append(
            {
                "strategy_family": "outcome",
                "strategy_type": "outcome_focus",
                "strategy_order": 4,
                "query_text": f"{scoped_anchor} {secondary_outcome}",
                "params": {"outcome": secondary_outcome, "result_limit": result_limit},
                "provider_allowlist": provider_allowlist,
            }
        )
    strategies.append(
        {
            "strategy_family": "recency",
            "strategy_type": "recent_window",
            "strategy_order": len(strategies) + 1,
            "query_text": scoped_anchor,
            "params": {"year_from": recent_year_from, "result_limit": max(4, result_limit - 2)},
            "provider_allowlist": provider_allowlist,
        }
    )
    unique_strategies: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for strategy in strategies:
        key = (
            str(strategy.get("strategy_family", "")).strip().lower(),
            str(strategy.get("strategy_type", "")).strip().lower(),
            str(strategy.get("query_text", "")).strip().lower(),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        strategy["strategy_order"] = len(unique_strategies) + 1
        unique_strategies.append(strategy)
    return unique_strategies


def derive_topics_from_confirmed_plan(task_type: str, confirmed_plan: dict[str, Any]) -> list[str]:
    normalized_task_type = normalize_task_type(task_type)
    if normalized_task_type == "claim_evidence":
        entries = confirmed_plan.get("search_topics", [])
        topics = []
        seen: set[str] = set()
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict):
                topic_name = str(entry.get("topic_name", "")).strip()
            else:
                topic_name = str(entry or "").strip()
            if not topic_name:
                continue
            lowered = topic_name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            topics.append(topic_name)
        return topics
    recommended_topics = confirmed_plan.get("recommended_topics", [])
    topics = []
    seen_topics: set[str] = set()
    for entry in recommended_topics if isinstance(recommended_topics, list) else []:
        if isinstance(entry, dict):
            topic_name = str(entry.get("topic_name", "") or entry.get("topic", "")).strip()
        else:
            topic_name = str(entry or "").strip()
        if not topic_name:
            continue
        lowered = topic_name.lower()
        if lowered in seen_topics:
            continue
        seen_topics.add(lowered)
        topics.append(topic_name)
    return topics


def classify_neighbor_relationship(similarity: float) -> str:
    if similarity >= 0.92:
        return "near_duplicate"
    if similarity >= 0.84:
        return "same_pattern"
    return "related"


def neighbor_relationship_reason(relationship: str) -> str:
    if relationship == "near_duplicate":
        return "Likely repeats the same teaching point and should be compared before review."
    if relationship == "same_pattern":
        return "Likely covers the same underlying pattern with different evidence or phrasing."
    return "Related card worth checking, but not an obvious duplicate."


def _aha_signal_text(card: dict[str, Any]) -> str:
    return " ".join(
        [
            str(card.get("paper_specific_object", "")).strip(),
            str(card.get("course_transformation", "")).strip(),
            str(card.get("teachable_one_liner", "")).strip(),
            str(card.get("title", "")).strip(),
        ]
    ).strip()


def _aha_shared_signal_score(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_grams = _signal_ngrams(_aha_signal_text(left))
    right_grams = _signal_ngrams(_aha_signal_text(right))
    if not left_grams or not right_grams:
        return 0
    return len(left_grams.intersection(right_grams))


def cards_share_aha_class(left: dict[str, Any], right: dict[str, Any]) -> bool:
    similarity = cosine_similarity(left.get("embedding", []), right.get("embedding", []))
    shared_signal_score = _aha_shared_signal_score(left, right)
    left_object = _normalized_signal_text(left.get("paper_specific_object", ""))
    right_object = _normalized_signal_text(right.get("paper_specific_object", ""))
    left_course = _normalized_signal_text(left.get("course_transformation", ""))
    right_course = _normalized_signal_text(right.get("course_transformation", ""))
    same_object = bool(left_object and left_object == right_object)
    same_course_form = bool(left_course and left_course == right_course)
    if similarity >= 0.94:
        return True
    if similarity >= 0.88 and shared_signal_score >= 2:
        return True
    if similarity >= 0.82 and same_object and same_course_form:
        return True
    return False


def _aha_class_representative_sort_key(card: dict[str, Any]) -> tuple[int, int, int, int]:
    judgement = card.get("judgement") or {}
    return (
        {"green": 3, "yellow": 2, "red": 1}.get(str(judgement.get("color", "yellow")).strip().lower(), 2),
        {"strong": 3, "medium": 2, "weak": 1}.get(str(card.get("evidence_level", "medium")).strip().lower(), 2),
        len(card.get("primary_section_ids", [])),
        len(card.get("evidence", [])),
    )


def cluster_cards_into_aha_classes(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    ordered_cards = sorted(cards, key=_aha_class_representative_sort_key, reverse=True)
    for card in ordered_cards:
        target_cluster = None
        for cluster in clusters:
            if cards_share_aha_class(cluster["representative"], card):
                target_cluster = cluster
                break
        if target_cluster is None:
            target_cluster = {
                "aha_class_id": f"aha_class_{len(clusters) + 1}",
                "representative": card,
                "members": [],
            }
            clusters.append(target_cluster)
        target_cluster["members"].append(card)
        if _aha_class_representative_sort_key(card) > _aha_class_representative_sort_key(target_cluster["representative"]):
            target_cluster["representative"] = card
    return clusters


def normalize_topics(topics_text: str) -> list[str]:
    seen: set[str] = set()
    topics: list[str] = []
    for raw in topics_text.splitlines():
        topic = raw.strip()
        if not topic:
            continue
        lowered = topic.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        topics.append(topic)
    return topics


def normalize_calibration_example(example: dict) -> dict:
    example_type = str(example.get("example_type", "")).strip().lower()
    if example_type not in {"positive", "negative", "boundary"}:
        raise ValueError("Calibration example_type must be one of: positive, negative, boundary.")

    title = str(example.get("title", "")).strip()
    source_text = str(example.get("source_text", "")).strip()
    topic_name = str(example.get("topic_name", "")).strip()
    audience = str(example.get("audience", "")).strip()
    rationale = str(example.get("rationale", "")).strip()
    if not title:
        raise ValueError("Calibration example title is required.")
    if not source_text:
        raise ValueError("Calibration example source_text is required.")
    if not topic_name:
        raise ValueError("Calibration example topic_name is required.")

    evidence = example.get("evidence", [])
    expected_cards = example.get("expected_cards", [])
    expected_exclusions = example.get("expected_exclusions", [])
    tags = example.get("tags", [])
    if not isinstance(evidence, list) or not isinstance(expected_cards, list) or not isinstance(expected_exclusions, list):
        raise ValueError("Calibration example evidence, expected_cards, and expected_exclusions must be lists.")
    if not isinstance(tags, list):
        raise ValueError("Calibration example tags must be a list.")

    return {
        "example_type": example_type,
        "topic_name": topic_name,
        "audience": audience,
        "title": title,
        "source_text": source_text,
        "evidence": evidence,
        "expected_cards": expected_cards,
        "expected_exclusions": expected_exclusions,
        "rationale": rationale,
        "tags": [str(tag).strip() for tag in tags if str(tag).strip()],
    }


def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\u00ad", "")
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = re.sub(
        r"([.!?])\s+(Abstract|Introduction|Background|Method|Methods|Results|Discussion|Conclusion|References)\b",
        r"\1\n\n\2",
        text,
    )
    blocks = re.split(r"\n\s*\n", text)
    paragraphs = []
    for block in blocks:
        paragraph = " ".join(line.strip() for line in block.splitlines() if line.strip())
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) >= 30:
            paragraphs.append(paragraph)
    return paragraphs


def classify_section_metadata(
    *,
    section_title: str,
    paragraph_text: str,
    section_order: int,
    total_sections: int,
    source_format: str,
) -> dict[str, Any]:
    title = str(section_title or "").strip()
    text = str(paragraph_text or "").strip()
    lowered_title = title.lower()
    lowered_text = text.lower()
    snippet = f"{lowered_title}\n{lowered_text[:320]}"

    section_kind = "other"
    body_role = ""
    if "abstract" in lowered_title or lowered_text.startswith("abstract "):
        section_kind = "abstract"
    elif any(token in lowered_title for token in ("keyword", "ccs concepts", "index terms")):
        section_kind = "keywords"
    elif any(token in lowered_title for token in ("author", "affiliation", "corresponding")):
        section_kind = "author_affiliation"
    elif any(token in lowered_title for token in ("reference", "bibliography")):
        section_kind = "references"
    elif any(token in lowered_title for token in ("appendix", "supplementary")):
        section_kind = "appendix"
    elif "introduction" in lowered_title:
        section_kind = "introduction"
        body_role = "introduction"
    elif any(token in lowered_title for token in ("method", "approach", "experimental setup", "materials and methods")):
        section_kind = "methods"
        body_role = "methods"
    elif any(token in lowered_title for token in ("result", "finding", "evaluation", "experiment")):
        section_kind = "results"
        body_role = "results"
    elif any(token in lowered_title for token in ("discussion", "analysis")):
        section_kind = "discussion"
        body_role = "discussion"
    elif any(token in lowered_title for token in ("conclusion", "limitations", "future work")):
        section_kind = "conclusion"
        body_role = "conclusion"

    if section_kind == "other":
        if any(token in snippet for token in ("we propose", "our method", "architecture", "algorithm", "pipeline")):
            section_kind = "methods"
            body_role = "methods"
        elif any(token in snippet for token in ("experiment", "ablation", "improves by", "outperforms", "accuracy")):
            section_kind = "results"
            body_role = "results"
        elif any(token in snippet for token in ("in summary", "we conclude", "this work shows")):
            section_kind = "conclusion"
            body_role = "conclusion"

    is_abstract = section_kind == "abstract"
    is_front_matter = section_kind in {"author_affiliation", "keywords"} or (section_order <= 2 and not body_role and not is_abstract)
    is_body = bool(body_role) or section_kind in {"methods", "results", "discussion", "conclusion", "introduction"}
    has_figure_reference = bool(re.search(r"\b(fig(?:ure)?\.?\s*\d+|table\s*\d+)\b", snippet))

    if section_order == 1 and not is_abstract and not is_body and source_format in {"pdf_markitdown", "pdf_fallback", "html"}:
        section_kind = "title_block"
        is_front_matter = True

    section_label = title or text[:64]
    return {
        "section_kind": section_kind,
        "section_label": section_label,
        "is_front_matter": is_front_matter,
        "is_abstract": is_abstract,
        "is_body": is_body,
        "body_role": body_role,
        "has_figure_reference": has_figure_reference,
        "source_format": source_format,
    }


def enrich_sections_with_structure(sections: list[dict], source_format: str) -> list[dict]:
    total_sections = len(sections)
    for section in sections:
        metadata = classify_section_metadata(
            section_title=section.get("section_title", ""),
            paragraph_text=section.get("paragraph_text", ""),
            section_order=int(section.get("section_order", 0) or 0),
            total_sections=total_sections,
            source_format=source_format,
        )
        section.update(metadata)
        section.setdefault("selection_score", 0.0)
        section.setdefault("selection_reason", {})
    return sections


def embedding_for_text(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        vector[bucket] += sign
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / length, 6) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class ParseFailure(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


TASK_TYPES = frozenset({"aha_exploration", "claim_evidence"})
CARD_REVIEW_DECISIONS = frozenset({"accepted", "rejected", "keep_for_later", "needs_manual_check"})
MATRIX_REVIEW_DECISIONS = CARD_REVIEW_DECISIONS
EXCLUDED_REVIEW_DECISIONS = frozenset({"accepted", "reopened", "needs_manual_check"})


def normalize_task_type(value: str, *, default: str = "aha_exploration") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in TASK_TYPES:
        return normalized
    return default


def review_status_sort_value(decision: str) -> int:
    return {
        "accepted": 4,
        "keep_for_later": 3,
        "needs_manual_check": 2,
        "rejected": 1,
        "reopened": 1,
    }.get(str(decision or "").strip().lower(), 0)


def allowed_review_decisions(target_type: str) -> frozenset[str]:
    if target_type == "card":
        return CARD_REVIEW_DECISIONS
    if target_type == "matrix_item":
        return MATRIX_REVIEW_DECISIONS
    if target_type == "excluded":
        return EXCLUDED_REVIEW_DECISIONS
    return frozenset()


def is_card_export_eligible(review_decision: str) -> bool:
    return review_decision == "accepted"


class Repository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()

    def _fetchone(self, query: str, params: tuple = ()) -> Optional[dict]:
        with db_read_cursor(self.settings.db_path) as connection:
            return connection.execute(query, params).fetchone()

    def _fetchall(self, query: str, params: tuple = ()) -> list[dict]:
        with db_read_cursor(self.settings.db_path) as connection:
            return connection.execute(query, params).fetchall()

    def _hydrate_evidence(self, evidence: list[dict]) -> list[dict]:
        hydrated = []
        for item in evidence:
            quote_en = str(item.get("quote", "")).strip()
            quote_zh = str(item.get("quote_zh", "")).strip()
            if quote_zh and quote_en and not looks_like_complete_translation(quote_en, quote_zh):
                quote_zh = ""
            hydrated.append(
                {
                    **item,
                    "quote_zh": quote_zh,
                }
            )
        return hydrated

    def _hydrate_card_row(self, row: dict) -> dict:
        row["evidence"] = self._hydrate_evidence(json.loads(row["evidence_json"]))
        row["figure_ids"] = json.loads(row["figure_ids_json"])
        row["embedding"] = json.loads(row["embedding_json"])
        row["primary_section_ids"] = json.loads(row.get("primary_section_ids_json", "[]") or "[]")
        row["supporting_section_ids"] = json.loads(row.get("supporting_section_ids_json", "[]") or "[]")
        row["paper_url"] = row.get("paper_url", "")
        row["source_excluded_content_id"] = row.get("source_excluded_content_id")
        return row

    def _build_paper_qa_capability(
        self,
        paper: dict[str, Any],
        *,
        section_count: int,
        has_abstract_backed_matrix_items: bool,
    ) -> dict[str, Any]:
        if section_count > 0:
            paper_content_basis = PAPER_CONTENT_BASIS_PARSED_FULLTEXT
            qa_available = True
            qa_status = PAPER_QA_STATUS_READY
        elif has_abstract_backed_matrix_items:
            paper_content_basis = PAPER_CONTENT_BASIS_ABSTRACT_ONLY
            qa_available = False
            qa_status = PAPER_QA_STATUS_BLOCKED_ABSTRACT_ONLY
        else:
            paper_content_basis = PAPER_CONTENT_BASIS_UNAVAILABLE
            qa_available = False
            qa_status = PAPER_QA_STATUS_BLOCKED_NO_PARSED_SECTIONS
        return {
            "paper_id": paper["id"],
            "paper_title": str(paper.get("title", "")).strip(),
            "access_status": str(paper.get("access_status", "")).strip(),
            "parse_status": str(paper.get("parse_status", "")).strip(),
            "section_count": int(section_count or 0),
            "paper_content_basis": paper_content_basis,
            "qa_available": qa_available,
            "qa_status": qa_status,
            "qa_message": PAPER_QA_STATUS_MESSAGES[qa_status],
            "has_abstract_backed_matrix_items": bool(has_abstract_backed_matrix_items),
        }

    def _matrix_item_is_abstract_only(self, item: dict[str, Any]) -> bool:
        supporting_ids = list(item.get("supporting_section_ids", []) or [])
        return bool(supporting_ids and all(str(section_id).startswith("abstract_meta::") for section_id in supporting_ids))

    def get_paper_qa_capability(
        self,
        paper_id: str,
        *,
        matrix_items: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[dict[str, Any]]:
        paper = self.get_paper(paper_id)
        if not paper:
            return None
        sections = self.get_sections(paper_id)
        resolved_matrix_items = matrix_items
        if resolved_matrix_items is None:
            resolved_matrix_items = self.list_matrix_items(paper_id=paper_id, include_paper_qa=False)
        has_abstract_backed_matrix_items = any(
            self._matrix_item_is_abstract_only(item)
            for item in resolved_matrix_items
        )
        return self._build_paper_qa_capability(
            paper,
            section_count=len(sections),
            has_abstract_backed_matrix_items=has_abstract_backed_matrix_items,
        )

    def _hydrate_matrix_row(
        self,
        row: dict,
        *,
        include_paper_qa: bool = True,
        paper_qa_capability_cache: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict:
        row["evidence"] = self._hydrate_evidence(json.loads(row["evidence_json"]))
        row["figure_ids"] = json.loads(row.get("figure_ids_json", "[]") or "[]")
        row["supporting_section_ids"] = json.loads(row.get("supporting_section_ids_json", "[]") or "[]")
        row["paper_url"] = row.get("paper_url", "")
        row["matrix_evidence_basis"] = "abstract_only" if self._matrix_item_is_abstract_only(row) else "full_text"
        if include_paper_qa and row.get("paper_id"):
            cache = paper_qa_capability_cache if paper_qa_capability_cache is not None else {}
            paper_id = str(row["paper_id"])
            capability = cache.get(paper_id)
            if capability is None:
                matrix_items = self.list_matrix_items(paper_id=paper_id, include_paper_qa=False)
                capability = self.get_paper_qa_capability(paper_id, matrix_items=matrix_items)
                if capability is not None:
                    cache[paper_id] = capability
            if capability:
                row["paper_access_status"] = str(row.get("paper_access_status", "")).strip() or capability["access_status"]
                row["paper_parse_status"] = str(row.get("paper_parse_status", "")).strip() or capability["parse_status"]
                row["paper_content_basis"] = capability["paper_content_basis"]
                row["paper_qa_available"] = capability["qa_available"]
                row["paper_qa_status"] = capability["qa_status"]
                row["paper_qa_message"] = capability["qa_message"]
                row["paper_has_parsed_sections"] = capability["section_count"] > 0
                row["paper_section_count"] = capability["section_count"]
                row["paper_has_abstract_backed_matrix_items"] = capability["has_abstract_backed_matrix_items"]
        return row

    def _hydrate_figure_row(self, row: dict) -> dict:
        row["linked_section_ids"] = json.loads(row.get("linked_section_ids_json", "[]") or "[]")
        row["page_number"] = row.get("page_number")
        row["byte_size"] = int(row.get("byte_size", 0) or 0)
        row["width"] = int(row["width"]) if row.get("width") is not None else None
        row["height"] = int(row["height"]) if row.get("height") is not None else None
        row["has_validated_asset"] = str(row.get("asset_status", "")).strip() == "validated_local_asset"
        return row

    def _hydrate_excluded_row(self, row: dict) -> dict:
        row["section_ids"] = json.loads(row["section_ids_json"])
        row["paper_url"] = row.get("paper_url", "")
        return row

    def _insert_candidate_card(self, connection, paper_id: str, topic_id: str, run_id: str, card: dict, source_excluded_content_id: Optional[str] = None) -> None:
        connection.execute(
            """
            INSERT INTO candidate_cards(
                id, paper_id, topic_id, run_id, title, granularity_level, course_transformation,
                teachable_one_liner, draft_body, evidence_json, figure_ids_json, status, embedding_json,
                source_excluded_content_id, primary_section_ids_json, supporting_section_ids_json,
                paper_specific_object, claim_type, evidence_level, body_grounding_reason, grounding_quality,
                duplicate_cluster_id, duplicate_rank, duplicate_disposition, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card["id"],
                paper_id,
                topic_id,
                run_id,
                card["title"],
                card["granularity_level"],
                card["course_transformation"],
                card["teachable_one_liner"],
                card["draft_body"],
                json.dumps(card["evidence"], ensure_ascii=False),
                json.dumps(card["figure_ids"]),
                card["status"],
                json.dumps(card["embedding"]),
                source_excluded_content_id,
                json.dumps(card.get("primary_section_ids", []), ensure_ascii=False),
                json.dumps(card.get("supporting_section_ids", []), ensure_ascii=False),
                card.get("paper_specific_object", ""),
                card.get("claim_type", ""),
                card.get("evidence_level", ""),
                card.get("body_grounding_reason", ""),
                card.get("grounding_quality", ""),
                card.get("duplicate_cluster_id", ""),
                int(card.get("duplicate_rank", 0) or 0),
                card.get("duplicate_disposition", ""),
                card["created_at"],
            ),
        )

    def _insert_evidence_matrix_item(self, connection, paper_id: str, topic_id: str, run_id: str, item: dict) -> None:
        connection.execute(
            """
            INSERT INTO evidence_matrix_items(
                id, paper_id, topic_id, run_id, dimension_key, dimension_label, outcome_key, outcome_label,
                claim_text, verdict, evidence_strength, summary, limitation_text, citation_text, evidence_json,
                figure_ids_json, supporting_section_ids_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["id"],
                paper_id,
                topic_id,
                run_id,
                item["dimension_key"],
                item["dimension_label"],
                item["outcome_key"],
                item["outcome_label"],
                item["claim_text"],
                item["verdict"],
                item["evidence_strength"],
                item["summary"],
                item.get("limitation_text", ""),
                item.get("citation_text", ""),
                json.dumps(item.get("evidence", []), ensure_ascii=False),
                json.dumps(item.get("figure_ids", []), ensure_ascii=False),
                json.dumps(item.get("supporting_section_ids", []), ensure_ascii=False),
                item["created_at"],
            ),
        )

    def _insert_judgement(self, connection, card_id: str, judgement: dict) -> None:
        connection.execute(
            """
            INSERT INTO judgements(id, card_id, color, reason, model_version, prompt_version, rubric_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("judge"),
                card_id,
                judgement["color"],
                judgement["reason"],
                judgement["model_version"],
                judgement["prompt_version"],
                judgement["rubric_version"],
                utc_now(),
            ),
        )

    def create_or_get_topic(self, name: str) -> dict:
        existing = self._fetchone("SELECT * FROM topics WHERE lower(name) = lower(?)", (name,))
        if existing:
            return existing
        topic = {
            "id": new_id("topic"),
            "name": name,
            "description": "",
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                "INSERT INTO topics(id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (topic["id"], topic["name"], topic["description"], topic["created_at"]),
            )
        return topic

    def get_topic(self, topic_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM topics WHERE id = ?", (topic_id,))

    def get_paper(self, paper_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM papers WHERE id = ?", (paper_id,))

    def sync_governance_records(self) -> None:
        self.sync_prompt_versions(get_prompt_version_records())
        self.sync_rubric_versions(get_rubric_version_records())

    def sync_prompt_versions(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        active_versions = {record["version"] for record in records}
        now = utc_now()
        with db_cursor(self.settings.db_path) as connection:
            for record in records:
                existing = connection.execute(
                    "SELECT * FROM prompt_versions WHERE version = ?",
                    (record["version"],),
                ).fetchone()
                if existing:
                    connection.execute(
                        """
                        UPDATE prompt_versions
                        SET prompt_stage = ?, summary = ?, details_json = ?, status = ?, activated_at = ?
                        WHERE version = ?
                        """,
                        (
                            record["stage"],
                            record["summary"],
                            json.dumps(record["details"], ensure_ascii=False),
                            "active",
                            now,
                            record["version"],
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO prompt_versions(id, prompt_stage, version, summary, details_json, status, created_at, activated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("pver"),
                            record["stage"],
                            record["version"],
                            record["summary"],
                            json.dumps(record["details"], ensure_ascii=False),
                            "active",
                            now,
                            now,
                        ),
                    )
            placeholders = ",".join("?" for _ in active_versions)
            if placeholders:
                connection.execute(
                    f"UPDATE prompt_versions SET status = 'archived' WHERE version NOT IN ({placeholders})",
                    tuple(active_versions),
                )

    def sync_rubric_versions(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        active_versions = {record["version"] for record in records}
        now = utc_now()
        with db_cursor(self.settings.db_path) as connection:
            for record in records:
                existing = connection.execute(
                    "SELECT * FROM rubric_versions WHERE version = ?",
                    (record["version"],),
                ).fetchone()
                if existing:
                    connection.execute(
                        """
                        UPDATE rubric_versions
                        SET rubric_name = ?, summary = ?, details_json = ?, status = ?, activated_at = ?
                        WHERE version = ?
                        """,
                        (
                            record["name"],
                            record["summary"],
                            json.dumps(record["details"], ensure_ascii=False),
                            "active",
                            now,
                            record["version"],
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO rubric_versions(id, rubric_name, version, summary, details_json, status, created_at, activated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id("rver"),
                            record["name"],
                            record["version"],
                            record["summary"],
                            json.dumps(record["details"], ensure_ascii=False),
                            "active",
                            now,
                            now,
                        ),
                    )
            placeholders = ",".join("?" for _ in active_versions)
            if placeholders:
                connection.execute(
                    f"UPDATE rubric_versions SET status = 'archived' WHERE version NOT IN ({placeholders})",
                    tuple(active_versions),
                )

    def list_prompt_versions(self, status: str = "") -> list[dict]:
        if status:
            rows = self._fetchall(
                "SELECT * FROM prompt_versions WHERE status = ? ORDER BY activated_at DESC, created_at DESC",
                (status,),
            )
        else:
            rows = self._fetchall("SELECT * FROM prompt_versions ORDER BY activated_at DESC, created_at DESC")
        for row in rows:
            row["details"] = json.loads(row["details_json"])
        return rows

    def list_rubric_versions(self, status: str = "") -> list[dict]:
        if status:
            rows = self._fetchall(
                "SELECT * FROM rubric_versions WHERE status = ? ORDER BY activated_at DESC, created_at DESC",
                (status,),
            )
        else:
            rows = self._fetchall("SELECT * FROM rubric_versions ORDER BY activated_at DESC, created_at DESC")
        for row in rows:
            row["details"] = json.loads(row["details_json"])
        return rows

    def get_calibration_workflow_status(self) -> dict[str, Any]:
        active_set = self.get_active_calibration_set()
        latest_evaluation = self.get_latest_evaluation_run()
        failed_results = []
        failed_boundary_results = []
        if latest_evaluation:
            failed_results = [item for item in latest_evaluation.get("results", []) if item.get("verdict") != "passed"]
            failed_boundary_results = [item for item in failed_results if item.get("example_type") == "boundary"]
        example_counts = {"positive": 0, "negative": 0, "boundary": 0}
        boundary_examples = []
        if active_set:
            for example in active_set.get("examples", []):
                example_type = str(example.get("example_type", "")).strip().lower()
                if example_type in example_counts:
                    example_counts[example_type] += 1
                if example_type == "boundary":
                    boundary_examples.append(
                        {
                            "id": example["id"],
                            "title": example["title"],
                            "tags": example.get("tags", []),
                            "rationale": example.get("rationale", ""),
                        }
                    )
        return {
            "active_calibration_set": active_set,
            "example_counts": example_counts,
            "boundary_examples": boundary_examples,
            "active_prompt_versions": self.list_prompt_versions(status="active"),
            "active_rubric_versions": self.list_rubric_versions(status="active"),
            "latest_evaluation_run": latest_evaluation,
            "failed_examples": failed_results,
            "failed_boundary_examples": failed_boundary_results,
        }

    def import_calibration_set(self, name: str, description: str, metadata: dict, examples: list[dict]) -> dict:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Calibration set name is required.")
        if not isinstance(metadata, dict):
            raise ValueError("Calibration set metadata must be an object.")
        normalized_examples = [normalize_calibration_example(example) for example in examples]
        if not normalized_examples:
            raise ValueError("Calibration set import requires at least one example.")

        existing = self._fetchone("SELECT * FROM calibration_sets WHERE lower(name) = lower(?)", (normalized_name,))
        calibration_set_id = existing["id"] if existing else new_id("cset")
        created_at = existing["created_at"] if existing else utc_now()
        activated_at = existing["activated_at"] if existing else ""
        status = existing["status"] if existing else "draft"

        with db_cursor(self.settings.db_path) as connection:
            if existing:
                connection.execute(
                    """
                    UPDATE calibration_sets
                    SET description = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (description, json.dumps(metadata, ensure_ascii=False), calibration_set_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO calibration_sets(id, name, description, status, metadata_json, created_at, activated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        calibration_set_id,
                        normalized_name,
                        description,
                        status,
                        json.dumps(metadata, ensure_ascii=False),
                        created_at,
                        activated_at,
                    ),
                )
            connection.execute("DELETE FROM calibration_examples WHERE calibration_set_id = ?", (calibration_set_id,))
            for example in normalized_examples:
                connection.execute(
                    """
                    INSERT INTO calibration_examples(
                        id, calibration_set_id, example_type, topic_name, audience, title, source_text,
                        evidence_json, expected_cards_json, expected_exclusions_json, rationale, tags_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id("cex"),
                        calibration_set_id,
                        example["example_type"],
                        example["topic_name"],
                        example["audience"],
                        example["title"],
                        example["source_text"],
                        json.dumps(example["evidence"], ensure_ascii=False),
                        json.dumps(example["expected_cards"], ensure_ascii=False),
                        json.dumps(example["expected_exclusions"], ensure_ascii=False),
                        example["rationale"],
                        json.dumps(example["tags"], ensure_ascii=False),
                        utc_now(),
                    ),
                )
        return self.get_calibration_set(calibration_set_id) or {}

    def activate_calibration_set(self, calibration_set_id: str) -> Optional[dict]:
        target = self._fetchone("SELECT * FROM calibration_sets WHERE id = ?", (calibration_set_id,))
        if not target:
            return None
        activated_at = utc_now()
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                "UPDATE calibration_sets SET status = 'draft', activated_at = '' WHERE status = 'active'"
            )
            connection.execute(
                "UPDATE calibration_sets SET status = 'active', activated_at = ? WHERE id = ?",
                (activated_at, calibration_set_id),
            )
        return self.get_calibration_set(calibration_set_id)

    def list_calibration_sets(self) -> list[dict]:
        rows = self._fetchall("SELECT * FROM calibration_sets ORDER BY created_at DESC")
        for row in rows:
            row["metadata"] = json.loads(row["metadata_json"])
            counts = {"positive": 0, "negative": 0, "boundary": 0}
            example_rows = self._fetchall(
                """
                SELECT example_type, COUNT(*) AS count
                FROM calibration_examples
                WHERE calibration_set_id = ?
                GROUP BY example_type
                """,
                (row["id"],),
            )
            row["example_count"] = 0
            for item in example_rows:
                row["example_count"] += item["count"]
                example_type = str(item["example_type"]).strip().lower()
                if example_type in counts:
                    counts[example_type] = item["count"]
            row["example_type_counts"] = counts
        return rows

    def get_calibration_set(self, calibration_set_id: str) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM calibration_sets WHERE id = ?", (calibration_set_id,))
        if not row:
            return None
        row["metadata"] = json.loads(row["metadata_json"])
        row["examples"] = self.list_calibration_examples(calibration_set_id)
        return row

    def get_active_calibration_set(self) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM calibration_sets WHERE status = 'active' ORDER BY activated_at DESC LIMIT 1")
        if not row:
            return None
        return self.get_calibration_set(row["id"])

    def create_evaluation_run(
        self,
        *,
        calibration_set: dict,
        llm_mode: str,
        model_name: str,
        extraction_prompt_version: str,
        judgement_prompt_version: str,
        rubric_version: str,
    ) -> dict:
        record = {
            "id": new_id("evalrun"),
            "calibration_set_id": calibration_set["id"],
            "calibration_set_name": calibration_set["name"],
            "llm_mode": llm_mode,
            "model_name": model_name,
            "extraction_prompt_version": extraction_prompt_version,
            "judgement_prompt_version": judgement_prompt_version,
            "rubric_version": rubric_version,
            "status": "running",
            "summary_json": json.dumps({}, ensure_ascii=False),
            "created_at": utc_now(),
            "completed_at": "",
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs(
                    id, calibration_set_id, calibration_set_name, llm_mode, model_name,
                    extraction_prompt_version, judgement_prompt_version, rubric_version,
                    status, summary_json, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["calibration_set_id"],
                    record["calibration_set_name"],
                    record["llm_mode"],
                    record["model_name"],
                    record["extraction_prompt_version"],
                    record["judgement_prompt_version"],
                    record["rubric_version"],
                    record["status"],
                    record["summary_json"],
                    record["created_at"],
                    record["completed_at"],
                ),
            )
        return record

    def finalize_evaluation_run(self, evaluation_run_id: str, status: str, summary: dict[str, Any]) -> Optional[dict]:
        completed_at = utc_now()
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                UPDATE evaluation_runs
                SET status = ?, summary_json = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, json.dumps(summary, ensure_ascii=False), completed_at, evaluation_run_id),
            )
        return self.get_evaluation_run(evaluation_run_id)

    def create_evaluation_result(
        self,
        *,
        evaluation_run_id: str,
        calibration_example: dict,
        extraction_output: dict[str, Any],
        judgement_output: dict[str, Any],
        expected: dict[str, Any],
        actual: dict[str, Any],
        verdict: str,
        regression_type: str,
        reason: str,
    ) -> dict:
        record = {
            "id": new_id("evalresult"),
            "evaluation_run_id": evaluation_run_id,
            "calibration_example_id": calibration_example["id"],
            "example_type": calibration_example["example_type"],
            "title": calibration_example["title"],
            "source_text": calibration_example["source_text"],
            "extraction_json": json.dumps(extraction_output, ensure_ascii=False),
            "judgement_json": json.dumps(judgement_output, ensure_ascii=False),
            "expected_json": json.dumps(expected, ensure_ascii=False),
            "actual_json": json.dumps(actual, ensure_ascii=False),
            "verdict": verdict,
            "regression_type": regression_type,
            "reason": reason,
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO evaluation_results(
                    id, evaluation_run_id, calibration_example_id, example_type, title, source_text,
                    extraction_json, judgement_json, expected_json, actual_json,
                    verdict, regression_type, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["evaluation_run_id"],
                    record["calibration_example_id"],
                    record["example_type"],
                    record["title"],
                    record["source_text"],
                    record["extraction_json"],
                    record["judgement_json"],
                    record["expected_json"],
                    record["actual_json"],
                    record["verdict"],
                    record["regression_type"],
                    record["reason"],
                    record["created_at"],
                ),
            )
        return self.get_evaluation_result(record["id"]) or record

    def list_evaluation_runs(self, limit: int = 10) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT * FROM evaluation_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        for row in rows:
            row["summary"] = json.loads(row["summary_json"])
        return rows

    def get_evaluation_run(self, evaluation_run_id: str) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM evaluation_runs WHERE id = ?", (evaluation_run_id,))
        if not row:
            return None
        row["summary"] = json.loads(row["summary_json"])
        row["results"] = self.list_evaluation_results(evaluation_run_id)
        return row

    def get_latest_evaluation_run(self) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT 1")
        if not row:
            return None
        return self.get_evaluation_run(row["id"])

    def create_paper_understanding_record(
        self,
        *,
        paper_id: str,
        topic_id: str,
        run_id: str,
        version: str,
        understanding: dict[str, Any],
    ) -> dict:
        record = {
            "id": new_id("understanding"),
            "paper_id": paper_id,
            "topic_id": topic_id,
            "run_id": run_id,
            "version": version,
            "understanding_json": json.dumps(understanding, ensure_ascii=False),
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO paper_understanding_records(
                    id, paper_id, topic_id, run_id, version, understanding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["paper_id"],
                    record["topic_id"],
                    record["run_id"],
                    record["version"],
                    record["understanding_json"],
                    record["created_at"],
                ),
            )
        return self.get_paper_understanding_record(record["id"]) or record

    def get_paper_understanding_record(self, understanding_record_id: str) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM paper_understanding_records WHERE id = ?", (understanding_record_id,))
        if not row:
            return None
        row["understanding"] = json.loads(row["understanding_json"])
        return row

    def get_latest_paper_understanding(self, paper_id: str, topic_id: str, run_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT * FROM paper_understanding_records
            WHERE paper_id = ? AND topic_id = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (paper_id, topic_id, run_id),
        )
        if not row:
            return None
        row["understanding"] = json.loads(row["understanding_json"])
        return row

    def create_card_plan(
        self,
        *,
        paper_id: str,
        topic_id: str,
        run_id: str,
        version: str,
        plan: dict[str, Any],
    ) -> dict:
        record = {
            "id": new_id("cardplan"),
            "paper_id": paper_id,
            "topic_id": topic_id,
            "run_id": run_id,
            "version": version,
            "plan_json": json.dumps(plan, ensure_ascii=False),
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO card_plans(
                    id, paper_id, topic_id, run_id, version, plan_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["paper_id"],
                    record["topic_id"],
                    record["run_id"],
                    record["version"],
                    record["plan_json"],
                    record["created_at"],
                ),
            )
        return self.get_card_plan(record["id"]) or record

    def get_card_plan(self, card_plan_id: str) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM card_plans WHERE id = ?", (card_plan_id,))
        if not row:
            return None
        row["plan"] = json.loads(row["plan_json"])
        return row

    def get_latest_card_plan(self, paper_id: str, topic_id: str, run_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT * FROM card_plans
            WHERE paper_id = ? AND topic_id = ? AND run_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (paper_id, topic_id, run_id),
        )
        if not row:
            return None
        row["plan"] = json.loads(row["plan_json"])
        return row

    def list_evaluation_results(self, evaluation_run_id: str) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT * FROM evaluation_results
            WHERE evaluation_run_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (evaluation_run_id,),
        )
        for row in rows:
            row["extraction"] = json.loads(row["extraction_json"])
            row["judgement"] = json.loads(row["judgement_json"])
            row["expected"] = json.loads(row["expected_json"])
            row["actual"] = json.loads(row["actual_json"])
        return rows

    def get_evaluation_result(self, evaluation_result_id: str) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM evaluation_results WHERE id = ?", (evaluation_result_id,))
        if not row:
            return None
        row["extraction"] = json.loads(row["extraction_json"])
        row["judgement"] = json.loads(row["judgement_json"])
        row["expected"] = json.loads(row["expected_json"])
        row["actual"] = json.loads(row["actual_json"])
        return row

    def list_calibration_examples(self, calibration_set_id: str) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT * FROM calibration_examples
            WHERE calibration_set_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (calibration_set_id,),
        )
        for row in rows:
            row["evidence"] = json.loads(row["evidence_json"])
            row["expected_cards"] = json.loads(row["expected_cards_json"])
            row["expected_exclusions"] = json.loads(row["expected_exclusions_json"])
            row["tags"] = json.loads(row["tags_json"])
        return rows

    def create_run(self, topics_text: str, metadata: dict) -> dict:
        run = {
            "id": new_id("run"),
            "created_at": utc_now(),
            "topics_text": topics_text,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "status": "pending",
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                "INSERT INTO runs(id, created_at, topics_text, metadata_json, status) VALUES (?, ?, ?, ?, ?)",
                (run["id"], run["created_at"], run["topics_text"], run["metadata_json"], run["status"]),
            )
        return run

    def update_run_status(self, run_id: str, status: str) -> None:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))

    def create_topic_run(self, run_id: str, topic_id: str) -> dict:
        topic_run = {
            "id": new_id("topicrun"),
            "run_id": run_id,
            "topic_id": topic_id,
            "status": "pending",
            "started_at": "",
            "completed_at": "",
            "stats_json": json.dumps(initial_topic_run_stats()),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO topic_runs(id, run_id, topic_id, status, started_at, completed_at, stats_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    topic_run["id"],
                    topic_run["run_id"],
                    topic_run["topic_id"],
                    topic_run["status"],
                    topic_run["started_at"],
                    topic_run["completed_at"],
                    topic_run["stats_json"],
                ),
            )
        return topic_run

    def update_topic_run(self, topic_run_id: str, status: str, stats: Optional[dict] = None, started: bool = False) -> None:
        current = self._fetchone("SELECT * FROM topic_runs WHERE id = ?", (topic_run_id,))
        if not current:
            return
        started_at = current["started_at"] or (utc_now() if started else "")
        completed_at = utc_now() if status in {"completed", "failed"} else current["completed_at"]
        stats_json = json.dumps(stats if stats is not None else json.loads(current["stats_json"]))
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                UPDATE topic_runs
                SET status = ?, started_at = ?, completed_at = ?, stats_json = ?
                WHERE id = ?
                """,
                (status, started_at, completed_at, stats_json, topic_run_id),
            )

    def create_discovery_strategy(
        self,
        *,
        run_id: str,
        topic_run_id: str,
        topic_id: str,
        provider: str,
        strategy_family: str,
        strategy_type: str,
        strategy_order: int,
        query_text: str,
        result_count: int,
        metadata: dict,
    ) -> dict:
        record = {
            "id": new_id("dstrat"),
            "run_id": run_id,
            "topic_run_id": topic_run_id,
            "topic_id": topic_id,
            "provider": provider,
            "strategy_family": strategy_family,
            "strategy_type": strategy_type,
            "strategy_order": strategy_order,
            "query_text": query_text,
            "status": "completed",
            "result_count": result_count,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO discovery_strategies(
                    id, run_id, topic_run_id, topic_id, provider, strategy_family, strategy_type,
                    strategy_order, query_text, status, result_count, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["run_id"],
                    record["topic_run_id"],
                    record["topic_id"],
                    record["provider"],
                    record["strategy_family"],
                    record["strategy_type"],
                    record["strategy_order"],
                    record["query_text"],
                    record["status"],
                    record["result_count"],
                    record["metadata_json"],
                    record["created_at"],
                ),
            )
        record["metadata"] = metadata
        return record

    def create_discovery_result(
        self,
        *,
        run_id: str,
        topic_run_id: str,
        strategy_id: str,
        dedupe_key: str,
        provider: str,
        source_external_id: str,
        paper_title: str,
        authors: list[str],
        publication_year: Optional[int],
        original_url: str,
        asset_url: str,
        confidence: float,
        dedupe_status: str,
        paper_id: str,
        metadata: dict,
    ) -> dict:
        record = {
            "id": new_id("dresult"),
            "run_id": run_id,
            "topic_run_id": topic_run_id,
            "strategy_id": strategy_id,
            "dedupe_key": dedupe_key,
            "provider": provider,
            "source_external_id": source_external_id,
            "paper_title": paper_title,
            "authors_json": json.dumps(authors, ensure_ascii=False),
            "publication_year": publication_year,
            "original_url": original_url,
            "asset_url": asset_url,
            "confidence": confidence,
            "dedupe_status": dedupe_status,
            "paper_id": paper_id,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO discovery_results(
                    id, run_id, topic_run_id, strategy_id, dedupe_key, provider, source_external_id,
                    paper_title, authors_json, publication_year, original_url, asset_url, confidence,
                    dedupe_status, paper_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["run_id"],
                    record["topic_run_id"],
                    record["strategy_id"],
                    record["dedupe_key"],
                    record["provider"],
                    record["source_external_id"],
                    record["paper_title"],
                    record["authors_json"],
                    record["publication_year"],
                    record["original_url"],
                    record["asset_url"],
                    record["confidence"],
                    record["dedupe_status"],
                    record["paper_id"],
                    record["metadata_json"],
                    record["created_at"],
                ),
            )
        record["authors"] = authors
        record["metadata"] = metadata
        return record

    def list_discovery_strategies(self, run_id: Optional[str] = None, topic_run_id: Optional[str] = None) -> list[dict]:
        params: list[str] = []
        clauses: list[str] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if topic_run_id:
            clauses.append("topic_run_id = ?")
            params.append(topic_run_id)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetchall(
            f"""
            SELECT discovery_strategies.*, topics.name AS topic_name
            FROM discovery_strategies
            JOIN topics ON topics.id = discovery_strategies.topic_id
            {where_clause}
            ORDER BY discovery_strategies.strategy_order ASC, discovery_strategies.created_at ASC, discovery_strategies.id ASC
            """,
            tuple(params),
        )
        for row in rows:
            row["metadata"] = json.loads(row["metadata_json"])
        return rows

    def list_discovery_results(self, run_id: Optional[str] = None, topic_run_id: Optional[str] = None) -> list[dict]:
        params: list[str] = []
        clauses: list[str] = []
        if run_id:
            clauses.append("discovery_results.run_id = ?")
            params.append(run_id)
        if topic_run_id:
            clauses.append("discovery_results.topic_run_id = ?")
            params.append(topic_run_id)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetchall(
            f"""
            SELECT discovery_results.*, discovery_strategies.query_text, discovery_strategies.strategy_type,
                   discovery_strategies.strategy_family, discovery_strategies.strategy_order
            FROM discovery_results
            JOIN discovery_strategies ON discovery_strategies.id = discovery_results.strategy_id
            {where_clause}
            ORDER BY discovery_strategies.strategy_order ASC, discovery_results.created_at ASC, discovery_results.id ASC
            """,
            tuple(params),
        )
        for row in rows:
            row["authors"] = json.loads(row["authors_json"])
            row["metadata"] = json.loads(row["metadata_json"])
        return rows

    def create_topic_saturation_snapshot(
        self,
        *,
        run_id: str,
        topic_run_id: str,
        topic_id: str,
        saturation_metrics: dict[str, Any],
    ) -> dict:
        flattening = saturation_metrics.get("flattening_signal", {}) if isinstance(saturation_metrics, dict) else {}
        search_strategy_comparison = saturation_metrics.get("search_strategy_comparison", []) if isinstance(saturation_metrics, dict) else []
        stop_decision = saturation_metrics.get("stop_decision", {}) if isinstance(saturation_metrics, dict) else {}
        record = {
            "id": new_id("tsnap"),
            "run_id": run_id,
            "topic_run_id": topic_run_id,
            "topic_id": topic_id,
            "card_count": int(saturation_metrics.get("card_count", 0)) if isinstance(saturation_metrics, dict) else 0,
            "near_duplicate_cards": int(saturation_metrics.get("near_duplicate_cards", 0)) if isinstance(saturation_metrics, dict) else 0,
            "same_pattern_cards": int(saturation_metrics.get("same_pattern_cards", 0)) if isinstance(saturation_metrics, dict) else 0,
            "novel_cards": int(saturation_metrics.get("novel_cards", 0)) if isinstance(saturation_metrics, dict) else 0,
            "semantic_duplication_ratio": float(saturation_metrics.get("semantic_duplication_ratio", 0.0)) if isinstance(saturation_metrics, dict) else 0.0,
            "likely_flattening": 1 if flattening.get("likely_flattening") else 0,
            "stop_decision": str(stop_decision.get("decision", "insufficient_history")).strip() or "insufficient_history",
            "stop_reason": str(stop_decision.get("reason", "")).strip(),
            "stop_policy_json": json.dumps(stop_decision.get("policy", {}), ensure_ascii=False),
            "tail_incremental_new_cards_json": json.dumps(flattening.get("tail_incremental_new_cards", []), ensure_ascii=False),
            "search_strategy_comparison_json": json.dumps(search_strategy_comparison, ensure_ascii=False),
            "saturation_metrics_json": json.dumps(saturation_metrics if isinstance(saturation_metrics, dict) else {}, ensure_ascii=False),
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO topic_saturation_snapshots(
                    id, run_id, topic_run_id, topic_id, card_count, near_duplicate_cards, same_pattern_cards, novel_cards,
                    semantic_duplication_ratio, likely_flattening, stop_decision, stop_reason, stop_policy_json,
                    tail_incremental_new_cards_json, search_strategy_comparison_json, saturation_metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["run_id"],
                    record["topic_run_id"],
                    record["topic_id"],
                    record["card_count"],
                    record["near_duplicate_cards"],
                    record["same_pattern_cards"],
                    record["novel_cards"],
                    record["semantic_duplication_ratio"],
                    record["likely_flattening"],
                    record["stop_decision"],
                    record["stop_reason"],
                    record["stop_policy_json"],
                    record["tail_incremental_new_cards_json"],
                    record["search_strategy_comparison_json"],
                    record["saturation_metrics_json"],
                    record["created_at"],
                ),
            )
        record["tail_incremental_new_cards"] = json.loads(record["tail_incremental_new_cards_json"])
        record["stop_policy"] = json.loads(record["stop_policy_json"])
        record["search_strategy_comparison"] = json.loads(record["search_strategy_comparison_json"])
        record["saturation_metrics"] = json.loads(record["saturation_metrics_json"])
        return record

    def list_topic_saturation_snapshots(self, topic: str = "", limit: int = 50) -> list[dict]:
        params: list[Any] = []
        where = ""
        topic_filter = topic.strip()
        if topic_filter:
            where = "WHERE LOWER(topics.name) = LOWER(?)"
            params.append(topic_filter)
        params.append(limit)
        rows = self._fetchall(
            f"""
            SELECT topic_saturation_snapshots.*, topics.name AS topic_name
            FROM topic_saturation_snapshots
            JOIN topics ON topics.id = topic_saturation_snapshots.topic_id
            {where}
            ORDER BY topic_saturation_snapshots.created_at DESC, topic_saturation_snapshots.id DESC
            LIMIT ?
            """,
            tuple(params),
        )
        for row in rows:
            row["tail_incremental_new_cards"] = json.loads(row["tail_incremental_new_cards_json"])
            row["stop_policy"] = json.loads(row["stop_policy_json"])
            row["search_strategy_comparison"] = json.loads(row["search_strategy_comparison_json"])
            row["saturation_metrics"] = json.loads(row["saturation_metrics_json"])
            row["likely_flattening"] = bool(row["likely_flattening"])
        return rows

    def list_topic_saturation_trends(self, topic: str = "", history_limit: int = 5) -> list[dict]:
        snapshots = self.list_topic_saturation_snapshots(topic=topic, limit=200)
        grouped: dict[str, list[dict]] = {}
        for snapshot in snapshots:
            grouped.setdefault(snapshot["topic_name"], []).append(snapshot)
        trends = []
        for topic_name, topic_snapshots in grouped.items():
            ordered = sorted(topic_snapshots, key=lambda item: item["created_at"], reverse=True)
            history = ordered[:history_limit]
            latest = history[0]
            previous = history[1] if len(history) > 1 else None
            latest_ratio = float(latest.get("semantic_duplication_ratio", 0.0))
            previous_ratio = float(previous.get("semantic_duplication_ratio", latest_ratio)) if previous else latest_ratio
            trends.append(
                {
                    "topic_name": topic_name,
                    "latest_run_id": latest["run_id"],
                    "latest_topic_run_id": latest["topic_run_id"],
                    "latest_created_at": latest["created_at"],
                    "latest_card_count": latest.get("card_count", 0),
                    "latest_duplication_ratio": latest_ratio,
                    "previous_duplication_ratio": previous_ratio,
                    "duplication_ratio_delta": round(latest_ratio - previous_ratio, 4),
                    "latest_likely_flattening": bool(latest.get("likely_flattening", False)),
                    "latest_stop_decision": latest.get("stop_decision", "insufficient_history"),
                    "latest_stop_reason": latest.get("stop_reason", ""),
                    "latest_stop_policy": latest.get("stop_policy", {}),
                    "latest_tail_incremental_new_cards": latest.get("tail_incremental_new_cards", []),
                    "history": history,
                }
            )
        trends.sort(key=lambda item: item["latest_created_at"], reverse=True)
        return trends

    def create_or_get_paper(
        self,
        *,
        title: str,
        authors: list[str],
        publication_year: Optional[int],
        external_id: str,
        source_type: str,
        local_path: str = "",
        original_url: str = "",
        access_status: str = "metadata_only",
        ingestion_status: str = "discovered",
        parse_status: str = "pending",
        artifact_path: str = "",
    ) -> dict:
        paper_key = external_id or stable_hash(f"{title}|{publication_year}|{source_type}")
        existing = self._fetchone("SELECT * FROM papers WHERE external_id = ?", (paper_key,))
        if existing:
            if existing.get("publication_year") is None and publication_year is not None:
                self.update_paper(existing["id"], publication_year=publication_year)
                existing["publication_year"] = publication_year
            return existing
        paper = {
            "id": new_id("paper"),
            "title": title,
            "authors_json": json.dumps(authors, ensure_ascii=False),
            "publication_year": publication_year,
            "external_id": paper_key,
            "source_type": source_type,
            "local_path": local_path,
            "original_url": original_url,
            "access_status": access_status,
            "ingestion_status": ingestion_status,
            "parse_status": parse_status,
            "parse_failure_reason": "",
            "card_generation_status": "pending",
            "card_generation_failure_reason": "",
            "artifact_path": artifact_path,
            "created_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO papers(
                    id, title, authors_json, publication_year, external_id, source_type,
                    local_path, original_url, access_status, ingestion_status, parse_status,
                    parse_failure_reason,
                    card_generation_status, card_generation_failure_reason,
                    artifact_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper["id"],
                    paper["title"],
                    paper["authors_json"],
                    paper["publication_year"],
                    paper["external_id"],
                    paper["source_type"],
                    paper["local_path"],
                    paper["original_url"],
                    paper["access_status"],
                    paper["ingestion_status"],
                    paper["parse_status"],
                    paper["parse_failure_reason"],
                    paper["card_generation_status"],
                    paper["card_generation_failure_reason"],
                    paper["artifact_path"],
                    paper["created_at"],
                ),
            )
        return paper

    def update_paper(self, paper_id: str, **fields: str) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [paper_id]
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(f"UPDATE papers SET {assignments} WHERE id = ?", values)

    def backfill_missing_publication_years(self) -> int:
        with db_cursor(self.settings.db_path) as connection:
            cursor = connection.execute(
                """
                UPDATE papers
                SET publication_year = (
                    SELECT MAX(discovery_results.publication_year)
                    FROM discovery_results
                    WHERE discovery_results.paper_id = papers.id
                      AND discovery_results.publication_year IS NOT NULL
                )
                WHERE papers.publication_year IS NULL
                  AND EXISTS (
                    SELECT 1
                    FROM discovery_results
                    WHERE discovery_results.paper_id = papers.id
                      AND discovery_results.publication_year IS NOT NULL
                  )
                """
            )
            return int(cursor.rowcount or 0)

    def add_paper_source(self, paper_id: str, provider: str, confidence: float, metadata: dict) -> None:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO paper_sources(id, paper_id, provider, confidence, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_id("psrc"), paper_id, provider, confidence, json.dumps(metadata, ensure_ascii=False), utc_now()),
            )

    def list_paper_sources(self, paper_id: str) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT *
            FROM paper_sources
            WHERE paper_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (paper_id,),
        )
        for row in rows:
            row["metadata"] = json.loads(row.get("metadata_json", "{}") or "{}")
        return rows

    def link_paper_to_topic(self, paper_id: str, topic_id: str, run_id: str, source_kind: str) -> None:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO paper_topics(paper_id, topic_id, run_id, source_kind)
                VALUES (?, ?, ?, ?)
                """,
                (paper_id, topic_id, run_id, source_kind),
            )

    def create_access_queue_item(self, paper_id: str, run_id: str, reason: str, priority: str = "medium") -> None:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO access_queue(id, paper_id, run_id, reason, priority, owner, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (new_id("aq"), paper_id, run_id, reason, priority, "", "open", utc_now()),
            )

    def get_access_queue_item(self, queue_item_id: str) -> Optional[dict]:
        return self._fetchone(
            """
            SELECT access_queue.*, papers.title AS paper_title, papers.original_url, papers.access_status, papers.parse_status
            FROM access_queue
            JOIN papers ON papers.id = access_queue.paper_id
            WHERE access_queue.id = ?
            """,
            (queue_item_id,),
        )

    def update_access_queue_item(self, queue_item_id: str, *, status: str, owner: str = "") -> Optional[dict]:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                "UPDATE access_queue SET status = ?, owner = ? WHERE id = ?",
                (status, owner, queue_item_id),
            )
        return self.get_access_queue_item(queue_item_id)

    def list_topic_runs_for_paper_run(self, paper_id: str, run_id: str) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT topic_runs.*, topics.name AS topic_name
            FROM topic_runs
            JOIN topics ON topics.id = topic_runs.topic_id
            JOIN paper_topics ON paper_topics.topic_id = topic_runs.topic_id AND paper_topics.run_id = topic_runs.run_id
            WHERE paper_topics.paper_id = ? AND topic_runs.run_id = ?
            ORDER BY topic_runs.started_at DESC, topic_runs.id DESC
            """,
            (paper_id, run_id),
        )
        for row in rows:
            self._decorate_topic_run(row)
        return rows

    def count_open_access_queue_for_topic(self, run_id: str, topic_id: str) -> int:
        return self._fetchone(
            """
            SELECT COUNT(*) AS count
            FROM access_queue
            JOIN paper_topics ON paper_topics.paper_id = access_queue.paper_id AND paper_topics.run_id = access_queue.run_id
            WHERE access_queue.run_id = ? AND paper_topics.topic_id = ? AND access_queue.status = 'open'
            """,
            (run_id, topic_id),
        )["count"]

    def _replace_sections_in_connection(self, connection: Any, paper_id: str, sections: list[dict]) -> None:
        connection.execute("DELETE FROM paper_sections WHERE paper_id = ?", (paper_id,))
        for section in sections:
            connection.execute(
                """
                INSERT INTO paper_sections(
                    id, paper_id, section_order, section_title, paragraph_text, page_number,
                    section_kind, section_label, is_front_matter, is_abstract, is_body, body_role,
                    has_figure_reference, source_format, selection_score, selection_reason_json, embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section["id"],
                    paper_id,
                    section["section_order"],
                    section["section_title"],
                    section["paragraph_text"],
                    section["page_number"],
                    section.get("section_kind", "other"),
                    section.get("section_label", ""),
                    int(section.get("is_front_matter", False)),
                    int(section.get("is_abstract", False)),
                    int(section.get("is_body", False)),
                    section.get("body_role", ""),
                    int(section.get("has_figure_reference", False)),
                    section.get("source_format", ""),
                    float(section.get("selection_score", 0.0) or 0.0),
                    json.dumps(section.get("selection_reason", {}), ensure_ascii=False),
                    json.dumps(section["embedding"]),
                ),
            )

    def _replace_figures_in_connection(self, connection: Any, paper_id: str, figures: list[dict]) -> None:
        connection.execute("DELETE FROM figures WHERE paper_id = ?", (paper_id,))
        for figure in figures:
            connection.execute(
                """
                INSERT INTO figures(
                    id, paper_id, figure_label, caption, page_number, storage_path, asset_status,
                    asset_kind, asset_local_path, asset_source_url, mime_type, byte_size,
                    sha256, width, height, validation_error, linked_section_ids_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    figure["id"],
                    paper_id,
                    figure["figure_label"],
                    figure["caption"],
                    figure.get("page_number"),
                    figure["storage_path"],
                    figure.get("asset_status", "metadata_only"),
                    figure.get("asset_kind", ""),
                    figure.get("asset_local_path", ""),
                    figure.get("asset_source_url", ""),
                    figure.get("mime_type", ""),
                    int(figure.get("byte_size", 0) or 0),
                    figure.get("sha256", ""),
                    figure.get("width"),
                    figure.get("height"),
                    figure.get("validation_error", ""),
                    json.dumps(figure["linked_section_ids"]),
                ),
            )

    def replace_sections(self, paper_id: str, sections: list[dict]) -> None:
        with db_cursor(self.settings.db_path) as connection:
            self._replace_sections_in_connection(connection, paper_id, sections)

    def replace_figures(self, paper_id: str, figures: list[dict]) -> None:
        with db_cursor(self.settings.db_path) as connection:
            self._replace_figures_in_connection(connection, paper_id, figures)

    def persist_parse_result(
        self,
        *,
        paper_id: str,
        sections: list[dict],
        figures: list[dict],
        parse_status: str,
        ingestion_status: str,
        parse_failure_reason: str,
        card_generation_status: str,
        card_generation_failure_reason: str,
        artifact_path: str = "",
    ) -> None:
        with db_cursor(self.settings.db_path) as connection:
            self._replace_sections_in_connection(connection, paper_id, sections)
            self._replace_figures_in_connection(connection, paper_id, figures)
            assignments = [
                ("parse_status", parse_status),
                ("ingestion_status", ingestion_status),
                ("parse_failure_reason", parse_failure_reason),
                ("card_generation_status", card_generation_status),
                ("card_generation_failure_reason", card_generation_failure_reason),
            ]
            if artifact_path:
                assignments.append(("artifact_path", artifact_path))
            sql = ", ".join(f"{name} = ?" for name, _ in assignments)
            values = [value for _, value in assignments] + [paper_id]
            connection.execute(f"UPDATE papers SET {sql} WHERE id = ?", values)

    def replace_generation_outputs_for_paper_topic(
        self,
        paper_id: str,
        topic_id: str,
        run_id: str,
        cards: list[dict],
        excluded_items: list[dict],
    ) -> None:
        with db_cursor(self.settings.db_path) as connection:
            card_rows = connection.execute(
                "SELECT id FROM candidate_cards WHERE paper_id = ? AND topic_id = ? AND run_id = ?",
                (paper_id, topic_id, run_id),
            ).fetchall()
            for row in card_rows:
                connection.execute("DELETE FROM judgements WHERE card_id = ?", (row["id"],))
                connection.execute("DELETE FROM review_decisions WHERE card_id = ?", (row["id"],))
            connection.execute(
                "DELETE FROM candidate_cards WHERE paper_id = ? AND topic_id = ? AND run_id = ?",
                (paper_id, topic_id, run_id),
            )
            connection.execute(
                "DELETE FROM paper_excluded_content WHERE paper_id = ? AND topic_id = ? AND run_id = ?",
                (paper_id, topic_id, run_id),
            )
            for card in cards:
                self._insert_candidate_card(connection, paper_id, topic_id, run_id, card)
                self._insert_judgement(connection, card["id"], card["judgement"])
            for item in excluded_items:
                connection.execute(
                    """
                    INSERT INTO paper_excluded_content(
                        id, paper_id, topic_id, run_id, label, exclusion_type, reason, section_ids_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["id"],
                        paper_id,
                        topic_id,
                        run_id,
                        item["label"],
                        item["exclusion_type"],
                        item["reason"],
                        json.dumps(item["section_ids"]),
                        item["created_at"],
                    ),
                )

    def replace_cards_for_paper_topic(self, paper_id: str, topic_id: str, run_id: str, cards: list[dict]) -> None:
        self.replace_generation_outputs_for_paper_topic(paper_id, topic_id, run_id, cards, [])

    def replace_matrix_items_for_paper_topic(
        self,
        paper_id: str,
        topic_id: str,
        run_id: str,
        matrix_items: list[dict],
    ) -> None:
        with db_cursor(self.settings.db_path) as connection:
            existing_rows = connection.execute(
                "SELECT id FROM evidence_matrix_items WHERE paper_id = ? AND topic_id = ? AND run_id = ?",
                (paper_id, topic_id, run_id),
            ).fetchall()
            for row in existing_rows:
                connection.execute(
                    "DELETE FROM review_decisions WHERE target_type = 'matrix_item' AND target_id = ?",
                    (row["id"],),
                )
            connection.execute(
                "DELETE FROM evidence_matrix_items WHERE paper_id = ? AND topic_id = ? AND run_id = ?",
                (paper_id, topic_id, run_id),
            )
            for item in matrix_items:
                self._insert_evidence_matrix_item(connection, paper_id, topic_id, run_id, item)

    def create_review_decision(self, target_type: str, target_id: str, reviewer: str, decision: str, note: str) -> None:
        card_id = target_id if target_type == "card" else None
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO review_decisions(id, target_type, target_id, card_id, reviewer, decision, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (new_id("review"), target_type, target_id, card_id, reviewer, decision, note, utc_now()),
            )

    def get_review_item_comment(self, target_type: str, target_id: str) -> Optional[dict]:
        return self._fetchone(
            """
            SELECT * FROM review_item_comments
            WHERE target_type = ? AND target_id = ?
            LIMIT 1
            """,
            (target_type, target_id),
        )

    def upsert_review_item_comment(self, target_type: str, target_id: str, reviewer: str, comment: str) -> Optional[dict]:
        normalized_comment = comment.strip()
        with db_cursor(self.settings.db_path) as connection:
            if not normalized_comment:
                connection.execute(
                    "DELETE FROM review_item_comments WHERE target_type = ? AND target_id = ?",
                    (target_type, target_id),
                )
                return None
            existing = connection.execute(
                """
                SELECT id, created_at
                FROM review_item_comments
                WHERE target_type = ? AND target_id = ?
                LIMIT 1
                """,
                (target_type, target_id),
            ).fetchone()
            now = utc_now()
            if existing:
                connection.execute(
                    """
                    UPDATE review_item_comments
                    SET reviewer = ?, comment = ?, updated_at = ?
                    WHERE target_type = ? AND target_id = ?
                    """,
                    (reviewer, normalized_comment, now, target_type, target_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO review_item_comments(id, target_type, target_id, reviewer, comment, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (new_id("comment"), target_type, target_id, reviewer, normalized_comment, now, now),
                )
        return self.get_review_item_comment(target_type, target_id)

    def create_promoted_candidate_card(self, excluded_content_id: str, card: dict) -> dict:
        excluded_item = self.get_excluded_content_summary(excluded_content_id)
        if not excluded_item:
            raise ValueError("Excluded content not found")
        with db_cursor(self.settings.db_path) as connection:
            self._insert_candidate_card(
                connection,
                excluded_item["paper_id"],
                excluded_item["topic_id"],
                excluded_item["run_id"],
                card,
                source_excluded_content_id=excluded_content_id,
            )
            self._insert_judgement(connection, card["id"], card["judgement"])
        return self.get_card(card["id"]) or card

    def get_latest_review_decision(self, target_type: str, target_id: str) -> Optional[dict]:
        return self._fetchone(
            """
            SELECT * FROM review_decisions
            WHERE target_type = ? AND target_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (target_type, target_id),
        )

    def get_promoted_card_summary(self, excluded_content_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT candidate_cards.*, papers.title AS paper_title, papers.original_url AS paper_url, topics.name AS topic_name
            FROM candidate_cards
            JOIN papers ON papers.id = candidate_cards.paper_id
            JOIN topics ON topics.id = candidate_cards.topic_id
            WHERE candidate_cards.source_excluded_content_id = ?
            """,
            (excluded_content_id,),
        )
        if not row:
            return None
        row["review"] = self.get_latest_review_decision("card", row["id"])
        row["judgement"] = self._fetchone(
            "SELECT * FROM judgements WHERE card_id = ? ORDER BY created_at DESC LIMIT 1",
            (row["id"],),
        )
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "title": row["title"],
            "paper_title": row["paper_title"],
            "paper_url": row.get("paper_url", ""),
            "topic_name": row["topic_name"],
            "review_status": (row["review"] or {}).get("decision", ""),
            "color": (row["judgement"] or {}).get("color", ""),
            "created_at": row["created_at"],
        }

    def create_export(
        self,
        run_id: str,
        destination_type: str,
        export_mode: str,
        google_doc_id: str,
        export_status: str,
        error_message: str,
        artifact_path: str,
        request_payload: dict,
    ) -> dict:
        record = {
            "id": new_id("export"),
            "run_id": run_id,
            "destination_type": destination_type,
            "export_mode": export_mode,
            "google_doc_id": google_doc_id,
            "export_status": export_status,
            "error_message": error_message,
            "artifact_path": artifact_path,
            "request_json": json.dumps(request_payload, ensure_ascii=False),
            "created_at": utc_now(),
            "completed_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO exports(id, run_id, destination_type, export_mode, google_doc_id, export_status, error_message, artifact_path, request_json, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["run_id"],
                    record["destination_type"],
                    record["export_mode"],
                    record["google_doc_id"],
                    record["export_status"],
                    record["error_message"],
                    record["artifact_path"],
                    record["request_json"],
                    record["created_at"],
                    record["completed_at"],
                ),
            )
        return record

    def get_run_progress_summary(self, run_id: str) -> dict[str, Any]:
        paper_rows = self._fetchall(
            """
            SELECT papers.*
            FROM papers
            JOIN paper_topics ON paper_topics.paper_id = papers.id
            WHERE paper_topics.run_id = ?
            """,
            (run_id,),
        )
        card_rows = self._fetchall("SELECT id FROM candidate_cards WHERE run_id = ?", (run_id,))
        matrix_rows = self._fetchall("SELECT id FROM evidence_matrix_items WHERE run_id = ?", (run_id,))
        judged_count = self._fetchone(
            """
            SELECT COUNT(DISTINCT judgements.card_id) AS count
            FROM judgements
            JOIN candidate_cards ON candidate_cards.id = judgements.card_id
            WHERE candidate_cards.run_id = ?
            """,
            (run_id,),
        )["count"]
        reviewed_count = self._fetchone(
            """
            SELECT COUNT(DISTINCT review_decisions.target_id) AS count
            FROM review_decisions
            JOIN candidate_cards ON candidate_cards.id = review_decisions.target_id
            WHERE review_decisions.target_type = 'card' AND candidate_cards.run_id = ?
            """,
            (run_id,),
        )["count"]
        reviewed_matrix_count = self._fetchone(
            """
            SELECT COUNT(DISTINCT review_decisions.target_id) AS count
            FROM review_decisions
            JOIN evidence_matrix_items ON evidence_matrix_items.id = review_decisions.target_id
            WHERE review_decisions.target_type = 'matrix_item' AND evidence_matrix_items.run_id = ?
            """,
            (run_id,),
        )["count"]
        export_rows = self._fetchall("SELECT request_json FROM exports WHERE run_id = ?", (run_id,))
        exported_card_ids: set[str] = set()
        exported_matrix_item_ids: set[str] = set()
        for row in export_rows:
            try:
                request_payload = json.loads(row["request_json"] or "{}")
            except json.JSONDecodeError:
                continue
            for card_id in request_payload.get("resolved_card_ids", []):
                if card_id:
                    exported_card_ids.add(str(card_id))
            for item_id in request_payload.get("resolved_matrix_item_ids", []):
                if item_id:
                    exported_matrix_item_ids.add(str(item_id))
        topic_runs = self.list_topic_runs(run_id)
        return {
            "topic_total": len(topic_runs),
            "topic_completed": sum(1 for item in topic_runs if item["status"] == "completed"),
            "topic_failed": sum(1 for item in topic_runs if item["status"] == "failed"),
            "topic_active": sum(1 for item in topic_runs if item["status"] == "running"),
            "discovered": len({row["id"] for row in paper_rows}),
            "accessible": sum(1 for row in paper_rows if row["access_status"] == "open_fulltext"),
            "parsed": sum(1 for row in paper_rows if row["parse_status"] == "parsed"),
            "carded": len(card_rows),
            "matrix_items": len(matrix_rows),
            "judged": int(judged_count),
            "reviewed": int(reviewed_count),
            "reviewed_matrix_items": int(reviewed_matrix_count),
            "exported": len(exported_card_ids),
            "exported_matrix_items": len(exported_matrix_item_ids),
        }

    def _decorate_run(self, row: dict) -> dict:
        try:
            row["metadata"] = json.loads(row.get("metadata_json", "{}") or "{}")
        except json.JSONDecodeError:
            row["metadata"] = {}
        row["progress_summary"] = self.get_run_progress_summary(row["id"])
        return row

    def _decorate_topic_run(self, row: dict) -> dict:
        stats = json.loads(row["stats_json"])
        elapsed_seconds = seconds_since(row.get("started_at", ""))
        stage_elapsed_seconds = seconds_since(stats.get("stage_started_at", ""))
        last_progress_seconds_ago = seconds_since(stats.get("last_progress_at", ""))
        derived_status = row["status"]
        if row["status"] == "running":
            waiting_for_access = (
                stats.get("current_stage") == "acquisition"
                and int(stats.get("queued_for_access", 0) or 0) > 0
                and int(stats.get("accessible", 0) or 0) == 0
            )
            if waiting_for_access:
                derived_status = "waiting_for_access"
            elif (
                last_progress_seconds_ago is not None
                and last_progress_seconds_ago >= self.settings.stalled_after_seconds
            ):
                derived_status = "stalled"
        row["stats"] = stats
        row["current_stage"] = stats.get("current_stage", "")
        row["elapsed_seconds"] = elapsed_seconds or 0
        row["stage_elapsed_seconds"] = stage_elapsed_seconds or 0
        row["last_progress_seconds_ago"] = last_progress_seconds_ago
        row["derived_status"] = derived_status
        row["latest_failures"] = summarize_latest_failures(stats)
        return row

    def get_run(self, run_id: str) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not row:
            return None
        return self._decorate_run(row)

    def list_runs(self) -> list[dict]:
        rows = self._fetchall("SELECT * FROM runs ORDER BY created_at DESC")
        for row in rows:
            self._decorate_run(row)
        return rows

    def list_topic_runs(self, run_id: Optional[str] = None) -> list[dict]:
        if run_id:
            rows = self._fetchall(
                """
                SELECT topic_runs.*, topics.name AS topic_name
                FROM topic_runs
                JOIN topics ON topics.id = topic_runs.topic_id
                WHERE topic_runs.run_id = ?
                ORDER BY topic_runs.started_at DESC, topic_runs.id DESC
                """,
                (run_id,),
            )
        else:
            rows = self._fetchall(
                """
                SELECT topic_runs.*, topics.name AS topic_name
                FROM topic_runs
                JOIN topics ON topics.id = topic_runs.topic_id
                ORDER BY topic_runs.started_at DESC, topic_runs.id DESC
                """
            )
        for row in rows:
            self._decorate_topic_run(row)
        return rows

    def get_topic_run(self, topic_run_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT topic_runs.*, topics.name AS topic_name
            FROM topic_runs
            JOIN topics ON topics.id = topic_runs.topic_id
            WHERE topic_runs.id = ?
            """,
            (topic_run_id,),
        )
        if row:
            self._decorate_topic_run(row)
        return row

    def list_papers_for_topic_run(self, run_id: str, topic_id: str) -> list[dict]:
        return self._fetchall(
            """
            SELECT papers.*
            FROM papers
            JOIN paper_topics ON paper_topics.paper_id = papers.id
            WHERE paper_topics.run_id = ? AND paper_topics.topic_id = ?
            ORDER BY papers.created_at DESC
            """,
            (run_id, topic_id),
        )

    def list_local_papers_for_topic_run(self, run_id: str, topic_id: str) -> list[dict]:
        return self._fetchall(
            """
            SELECT papers.*
            FROM papers
            JOIN paper_topics ON paper_topics.paper_id = papers.id
            WHERE paper_topics.run_id = ? AND paper_topics.topic_id = ? AND paper_topics.source_kind = 'local_pdf'
            ORDER BY papers.created_at DESC
            """,
            (run_id, topic_id),
        )

    def get_sections(self, paper_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM paper_sections WHERE paper_id = ? ORDER BY section_order ASC",
            (paper_id,),
        )
        for row in rows:
            row["embedding"] = json.loads(row["embedding_json"])
            row["selection_reason"] = json.loads(row.get("selection_reason_json", "{}") or "{}")
            row["is_front_matter"] = bool(row.get("is_front_matter", 0))
            row["is_abstract"] = bool(row.get("is_abstract", 0))
            row["is_body"] = bool(row.get("is_body", 0))
            row["has_figure_reference"] = bool(row.get("has_figure_reference", 0))
        return rows

    def update_section_selection_diagnostics(self, paper_id: str, diagnostics_by_section_id: dict[str, dict[str, Any]]) -> None:
        if not diagnostics_by_section_id:
            return
        with db_cursor(self.settings.db_path) as connection:
            for section_id, diagnostics in diagnostics_by_section_id.items():
                connection.execute(
                    """
                    UPDATE paper_sections
                    SET selection_score = ?, selection_reason_json = ?
                    WHERE paper_id = ? AND id = ?
                    """,
                    (
                        float(diagnostics.get("score", 0.0) or 0.0),
                        json.dumps(diagnostics, ensure_ascii=False),
                        paper_id,
                        section_id,
                    ),
                )

    def get_figures(self, paper_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM figures WHERE paper_id = ? ORDER BY id ASC",
            (paper_id,),
        )
        for row in rows:
            self._hydrate_figure_row(row)
        return rows

    def get_figure(self, figure_id: str) -> Optional[dict]:
        row = self._fetchone("SELECT * FROM figures WHERE id = ?", (figure_id,))
        if not row:
            return None
        return self._hydrate_figure_row(row)

    def get_figures_by_ids(self, paper_id: str, figure_ids: list[str]) -> list[dict]:
        if not figure_ids:
            return []
        placeholders = ",".join("?" for _ in figure_ids)
        rows = self._fetchall(
            f"SELECT * FROM figures WHERE paper_id = ? AND id IN ({placeholders}) ORDER BY id ASC",
            tuple([paper_id] + figure_ids),
        )
        row_map: dict[str, dict] = {}
        for row in rows:
            hydrated = self._hydrate_figure_row(row)
            row_map[hydrated["id"]] = hydrated
        return [row_map[figure_id] for figure_id in figure_ids if figure_id in row_map]

    def get_card(self, card_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT candidate_cards.*, papers.title AS paper_title, papers.original_url AS paper_url, topics.name AS topic_name
            FROM candidate_cards
            JOIN papers ON papers.id = candidate_cards.paper_id
            JOIN topics ON topics.id = candidate_cards.topic_id
            WHERE candidate_cards.id = ?
            """,
            (card_id,),
        )
        if not row:
            return None
        row = self._hydrate_card_row(row)
        row["judgement"] = self._fetchone(
            "SELECT * FROM judgements WHERE card_id = ? ORDER BY created_at DESC LIMIT 1",
            (card_id,),
        )
        row["review"] = self.get_latest_review_decision("card", card_id)
        row["comment"] = self.get_review_item_comment("card", card_id)
        row["excluded_content"] = self.list_excluded_content(
            paper_id=row["paper_id"],
            topic_id=row["topic_id"],
            run_id=row["run_id"],
        )
        row["figures"] = self.get_figures_by_ids(row["paper_id"], row.get("figure_ids", []))
        row["grounding_diagnostics"] = self._build_card_grounding_diagnostics(row)
        return row

    def get_matrix_item(self, matrix_item_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT evidence_matrix_items.*, papers.title AS paper_title, papers.original_url AS paper_url,
                   papers.publication_year AS publication_year, papers.access_status AS paper_access_status,
                   papers.parse_status AS paper_parse_status, topics.name AS topic_name
            FROM evidence_matrix_items
            JOIN papers ON papers.id = evidence_matrix_items.paper_id
            JOIN topics ON topics.id = evidence_matrix_items.topic_id
            WHERE evidence_matrix_items.id = ?
            """,
            (matrix_item_id,),
        )
        if not row:
            return None
        row = self._hydrate_matrix_row(row, paper_qa_capability_cache={})
        row["review"] = self.get_latest_review_decision("matrix_item", matrix_item_id)
        row["comment"] = self.get_review_item_comment("matrix_item", matrix_item_id)
        row["figures"] = self.get_figures_by_ids(row["paper_id"], row.get("figure_ids", []))
        evidence_section_ids = [
            str(item.get("section_id", "")).strip()
            for item in row.get("evidence", [])
            if str(item.get("section_id", "")).strip()
        ]
        row["evidence_sections"] = self._fetchall(
            f"""
            SELECT id, section_title, paragraph_text, page_number
            FROM paper_sections
            WHERE paper_id = ? AND id IN ({",".join("?" for _ in evidence_section_ids) or "''"})
            ORDER BY section_order ASC
            """,
            tuple([row["paper_id"]] + evidence_section_ids),
        )
        return row

    def _build_card_grounding_diagnostics(self, card: dict) -> dict[str, Any]:
        evidence_section_ids = [item.get("section_id") for item in card.get("evidence", []) if item.get("section_id")]
        if evidence_section_ids:
            placeholders = ",".join("?" for _ in evidence_section_ids)
            section_rows = self._fetchall(
                f"""
                SELECT id, section_kind, is_front_matter, is_abstract, is_body, body_role
                FROM paper_sections
                WHERE paper_id = ? AND id IN ({placeholders})
                """,
                tuple([card["paper_id"]] + evidence_section_ids),
            )
        else:
            section_rows = []
        section_type_mix = {
            "front_matter": 0,
            "abstract": 0,
            "body": 0,
        }
        for section in section_rows:
            if section.get("is_front_matter"):
                section_type_mix["front_matter"] += 1
            if section.get("is_abstract"):
                section_type_mix["abstract"] += 1
            if section.get("is_body"):
                section_type_mix["body"] += 1
        sibling_rows = self._fetchall(
            """
            SELECT id, title, duplicate_cluster_id, duplicate_rank, duplicate_disposition
            FROM candidate_cards
            WHERE run_id = ? AND paper_id = ? AND topic_id = ? AND id != ?
            ORDER BY created_at DESC
            """,
            (card["run_id"], card["paper_id"], card["topic_id"], card["id"]),
        )
        return {
            "section_type_mix": section_type_mix,
            "primary_section_ids": card.get("primary_section_ids", []),
            "supporting_section_ids": card.get("supporting_section_ids", []),
            "paper_specific_object": card.get("paper_specific_object", ""),
            "claim_type": card.get("claim_type", ""),
            "evidence_level": card.get("evidence_level", ""),
            "body_grounding_reason": card.get("body_grounding_reason", ""),
            "grounding_quality": card.get("grounding_quality", ""),
            "duplicate_cluster_id": card.get("duplicate_cluster_id", ""),
            "duplicate_rank": card.get("duplicate_rank", 0),
            "duplicate_disposition": card.get("duplicate_disposition", ""),
            "same_paper_siblings": sibling_rows,
        }

    def list_cards(
        self,
        run_id: Optional[str] = None,
        topic: str = "",
        paper_id: Optional[str] = None,
        topic_id: Optional[str] = None,
    ) -> list[dict]:
        params: list[str] = []
        filters = []
        if run_id:
            filters.append("candidate_cards.run_id = ?")
            params.append(run_id)
        if paper_id:
            filters.append("candidate_cards.paper_id = ?")
            params.append(paper_id)
        if topic_id:
            filters.append("candidate_cards.topic_id = ?")
            params.append(topic_id)
        if topic:
            filters.append("lower(topics.name) = lower(?)")
            params.append(topic)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self._fetchall(
            f"""
            SELECT
                candidate_cards.*,
                papers.title AS paper_title,
                papers.original_url AS paper_url,
                papers.publication_year AS publication_year,
                topics.name AS topic_name,
                (
                    SELECT color FROM judgements
                    WHERE judgements.card_id = candidate_cards.id
                    ORDER BY created_at DESC LIMIT 1
                ) AS color,
                (
                    SELECT decision FROM review_decisions
                    WHERE review_decisions.target_type = 'card'
                      AND review_decisions.target_id = candidate_cards.id
                    ORDER BY created_at DESC LIMIT 1
                ) AS review_decision
            FROM candidate_cards
            JOIN papers ON papers.id = candidate_cards.paper_id
            JOIN topics ON topics.id = candidate_cards.topic_id
            {where_clause}
            ORDER BY candidate_cards.created_at DESC
            """,
            tuple(params),
        )
        for row in rows:
            self._hydrate_card_row(row)
        return rows

    def list_matrix_items(
        self,
        *,
        run_id: Optional[str] = None,
        topic: str = "",
        paper_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        include_paper_qa: bool = True,
    ) -> list[dict]:
        params: list[str] = []
        filters = []
        if run_id:
            filters.append("evidence_matrix_items.run_id = ?")
            params.append(run_id)
        if paper_id:
            filters.append("evidence_matrix_items.paper_id = ?")
            params.append(paper_id)
        if topic_id:
            filters.append("evidence_matrix_items.topic_id = ?")
            params.append(topic_id)
        if topic:
            filters.append("lower(topics.name) = lower(?)")
            params.append(topic)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self._fetchall(
            f"""
            SELECT
                evidence_matrix_items.*,
                papers.title AS paper_title,
                papers.original_url AS paper_url,
                papers.publication_year AS publication_year,
                papers.access_status AS paper_access_status,
                papers.parse_status AS paper_parse_status,
                topics.name AS topic_name,
                (
                    SELECT decision FROM review_decisions
                    WHERE review_decisions.target_type = 'matrix_item'
                      AND review_decisions.target_id = evidence_matrix_items.id
                    ORDER BY created_at DESC LIMIT 1
                ) AS review_decision
            FROM evidence_matrix_items
            JOIN papers ON papers.id = evidence_matrix_items.paper_id
            JOIN topics ON topics.id = evidence_matrix_items.topic_id
            {where_clause}
            ORDER BY evidence_matrix_items.created_at DESC
            """,
            tuple(params),
        )
        paper_qa_capability_cache: dict[str, dict[str, Any]] = {}
        for row in rows:
            self._hydrate_matrix_row(
                row,
                include_paper_qa=include_paper_qa,
                paper_qa_capability_cache=paper_qa_capability_cache,
            )
        return rows

    def get_quality_metrics(self, run_id: Optional[str] = None) -> dict[str, Any]:
        cards = self.list_cards(run_id=run_id)
        total_cards = len(cards)
        if total_cards == 0:
            return {
                "run_id": run_id or "",
                "total_cards": 0,
                "abstract_backed_card_rate": 0.0,
                "front_matter_primary_rate": 0.0,
                "body_grounded_card_rate": 0.0,
                "same_evidence_duplicate_escape_rate": 0.0,
                "paper_specific_object_presence_rate": 0.0,
                "accepted_card_body_grounding_rate": 0.0,
                "accepted_card_duplicate_conflict_rate": 0.0,
            }
        abstract_backed = 0
        front_primary = 0
        body_grounded = 0
        duplicate_escape = 0
        object_present = 0
        accepted_cards = 0
        accepted_body_grounded = 0
        accepted_duplicate_conflicts = 0
        for card in cards:
            diagnostics = self._build_card_grounding_diagnostics(card)
            section_mix = diagnostics["section_type_mix"]
            if section_mix["abstract"] > 0:
                abstract_backed += 1
            primary_ids = diagnostics["primary_section_ids"] or []
            if primary_ids:
                placeholders = ",".join("?" for _ in primary_ids)
                primary_sections = self._fetchall(
                    f"SELECT is_front_matter, is_abstract, is_body FROM paper_sections WHERE paper_id = ? AND id IN ({placeholders})",
                    tuple([card["paper_id"]] + primary_ids),
                )
            else:
                primary_sections = []
            if primary_sections and all(item.get("is_front_matter") or item.get("is_abstract") for item in primary_sections):
                front_primary += 1
            if section_mix["body"] > 0:
                body_grounded += 1
            if str(card.get("paper_specific_object", "")).strip():
                object_present += 1
            if card.get("duplicate_disposition", "") not in {"", "kept"}:
                duplicate_escape += 1
            if (card.get("review_decision") or "") == "accepted":
                accepted_cards += 1
                if section_mix["body"] > 0:
                    accepted_body_grounded += 1
                if card.get("duplicate_disposition", "") not in {"", "kept"}:
                    accepted_duplicate_conflicts += 1
        return {
            "run_id": run_id or "",
            "total_cards": total_cards,
            "abstract_backed_card_rate": round(abstract_backed / total_cards, 4),
            "front_matter_primary_rate": round(front_primary / total_cards, 4),
            "body_grounded_card_rate": round(body_grounded / total_cards, 4),
            "same_evidence_duplicate_escape_rate": round(duplicate_escape / total_cards, 4),
            "paper_specific_object_presence_rate": round(object_present / total_cards, 4),
            "accepted_card_body_grounding_rate": round(accepted_body_grounded / max(accepted_cards, 1), 4),
            "accepted_card_duplicate_conflict_rate": round(accepted_duplicate_conflicts / max(accepted_cards, 1), 4),
            "accepted_cards": accepted_cards,
        }

    def get_excluded_content_summary(self, excluded_content_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT paper_excluded_content.*, papers.title AS paper_title, papers.original_url AS paper_url, papers.publication_year AS publication_year, topics.name AS topic_name
            FROM paper_excluded_content
            JOIN papers ON papers.id = paper_excluded_content.paper_id
            JOIN topics ON topics.id = paper_excluded_content.topic_id
            WHERE paper_excluded_content.id = ?
            """,
            (excluded_content_id,),
        )
        if not row:
            return None
        row = self._hydrate_excluded_row(row)
        row["review"] = self.get_latest_review_decision("excluded", excluded_content_id)
        row["comment"] = self.get_review_item_comment("excluded", excluded_content_id)
        return row

    def list_excluded_content(
        self,
        *,
        paper_id: Optional[str] = None,
        topic_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[dict]:
        filters = []
        params: list[str] = []
        if paper_id:
            filters.append("paper_excluded_content.paper_id = ?")
            params.append(paper_id)
        if topic_id:
            filters.append("paper_excluded_content.topic_id = ?")
            params.append(topic_id)
        if run_id:
            filters.append("paper_excluded_content.run_id = ?")
            params.append(run_id)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        rows = self._fetchall(
            f"""
            SELECT paper_excluded_content.*, papers.title AS paper_title, papers.original_url AS paper_url, papers.publication_year AS publication_year, topics.name AS topic_name
            FROM paper_excluded_content
            JOIN papers ON papers.id = paper_excluded_content.paper_id
            JOIN topics ON topics.id = paper_excluded_content.topic_id
            {where_clause}
            ORDER BY paper_excluded_content.created_at ASC
            """,
            tuple(params),
        )
        for row in rows:
            self._hydrate_excluded_row(row)
            row["review"] = self.get_latest_review_decision("excluded", row["id"])
            row["comment"] = self.get_review_item_comment("excluded", row["id"])
            row["promoted_card"] = self.get_promoted_card_summary(row["id"])
        return rows

    def get_excluded_content(self, excluded_content_id: str) -> Optional[dict]:
        row = self.get_excluded_content_summary(excluded_content_id)
        if not row:
            return None
        row["evidence_sections"] = self._fetchall(
            f"""
            SELECT id, section_title, paragraph_text, page_number
            FROM paper_sections
            WHERE paper_id = ? AND id IN ({",".join("?" for _ in row["section_ids"]) or "''"})
            ORDER BY section_order ASC
            """,
            tuple([row["paper_id"]] + row["section_ids"]),
        )
        row["promoted_card"] = self.get_promoted_card_summary(excluded_content_id)
        return row

    def list_review_items(
        self,
        *,
        run_id: Optional[str] = None,
        topic: str = "",
        item_type: str = "cards",
        review_status: str = "",
        exclusion_type: str = "",
    ) -> list[dict]:
        normalized_item_type = (item_type or "cards").strip().lower()
        items = []
        if normalized_item_type in {"cards", "both", "all"}:
            for card in self.list_cards(run_id=run_id, topic=topic):
                if review_status and (card.get("review_decision") or "") != review_status:
                    continue
                comment = self.get_review_item_comment("card", card["id"]) or {}
                items.append(
                    {
                        "object_type": "card",
                        "object_id": card["id"],
                        "run_id": card["run_id"],
                        "topic_name": card["topic_name"],
                        "paper_title": card["paper_title"],
                        "publication_year": card.get("publication_year"),
                        "paper_url": card.get("paper_url", ""),
                        "display_title": card["title"],
                        "color": card.get("color", ""),
                        "course_transformation": card.get("course_transformation", ""),
                        "teachable_one_liner": card.get("teachable_one_liner", ""),
                        "review_status": card.get("review_decision", ""),
                        "comment_text": comment.get("comment", ""),
                        "comment_updated_at": comment.get("updated_at", ""),
                        "exclusion_type": "",
                        "export_eligible": is_card_export_eligible(card.get("review_decision") or ""),
                        "source_excluded_content_id": card.get("source_excluded_content_id"),
                        "promoted_from_excluded": bool(card.get("source_excluded_content_id")),
                        "created_at": card["created_at"],
                    }
                )
        if normalized_item_type in {"matrix", "matrix_items", "all"}:
            for item in self.list_matrix_items(run_id=run_id, topic=topic):
                if review_status and (item.get("review_decision") or "") != review_status:
                    continue
                comment = self.get_review_item_comment("matrix_item", item["id"]) or {}
                items.append(
                    {
                        "object_type": "matrix_item",
                        "object_id": item["id"],
                        "paper_id": item["paper_id"],
                        "run_id": item["run_id"],
                        "topic_name": item["topic_name"],
                        "paper_title": item["paper_title"],
                        "publication_year": item.get("publication_year"),
                        "paper_url": item.get("paper_url", ""),
                        "paper_access_status": item.get("paper_access_status", ""),
                        "paper_parse_status": item.get("paper_parse_status", ""),
                        "paper_content_basis": item.get("paper_content_basis", ""),
                        "paper_qa_available": bool(item.get("paper_qa_available")),
                        "paper_qa_status": item.get("paper_qa_status", ""),
                        "paper_qa_message": item.get("paper_qa_message", ""),
                        "paper_section_count": int(item.get("paper_section_count", 0) or 0),
                        "matrix_evidence_basis": item.get("matrix_evidence_basis", ""),
                        "display_title": item["summary"],
                        "color": item.get("verdict", ""),
                        "course_transformation": item.get("outcome_label", ""),
                        "teachable_one_liner": item.get("claim_text", ""),
                        "review_status": item.get("review_decision", ""),
                        "comment_text": comment.get("comment", ""),
                        "comment_updated_at": comment.get("updated_at", ""),
                        "exclusion_type": "",
                        "export_eligible": is_card_export_eligible(item.get("review_decision") or ""),
                        "verdict": item.get("verdict", ""),
                        "evidence_strength": item.get("evidence_strength", ""),
                        "dimension_label": item.get("dimension_label", ""),
                        "outcome_label": item.get("outcome_label", ""),
                        "created_at": item["created_at"],
                    }
                )
        if normalized_item_type in {"excluded", "both", "all"}:
            excluded_filters = {"run_id": run_id, "topic_id": None, "paper_id": None}
            excluded_rows = self.list_excluded_content(run_id=run_id)
            for item in excluded_rows:
                if topic and item["topic_name"].lower() != topic.lower():
                    continue
                latest_review = item.get("review") or {}
                if review_status and latest_review.get("decision", "") != review_status:
                    continue
                if exclusion_type and item["exclusion_type"] != exclusion_type:
                    continue
                comment = item.get("comment") or {}
                items.append(
                    {
                        "object_type": "excluded",
                        "object_id": item["id"],
                        "run_id": item["run_id"],
                        "topic_name": item["topic_name"],
                        "paper_title": item["paper_title"],
                        "publication_year": item.get("publication_year"),
                        "paper_url": item.get("paper_url", ""),
                        "display_title": item["label"],
                        "color": "",
                        "course_transformation": "",
                        "teachable_one_liner": "",
                        "review_status": latest_review.get("decision", ""),
                        "comment_text": comment.get("comment", ""),
                        "comment_updated_at": comment.get("updated_at", ""),
                        "exclusion_type": item["exclusion_type"],
                        "export_eligible": False,
                        "promoted_card_id": (item.get("promoted_card") or {}).get("id", ""),
                        "promoted_card_title": (item.get("promoted_card") or {}).get("title", ""),
                        "created_at": item["created_at"],
                    }
                )
        items.sort(key=lambda item: item["created_at"], reverse=True)
        return items

    def get_review_item(self, target_type: str, target_id: str) -> Optional[dict]:
        if target_type == "card":
            card = self.get_card(target_id)
            if not card:
                return None
            card["object_type"] = "card"
            card["object_id"] = card["id"]
            card["display_title"] = card["title"]
            card["export_eligible"] = is_card_export_eligible((card.get("review") or {}).get("decision", ""))
            if card.get("source_excluded_content_id"):
                card["source_excluded_item"] = self.get_excluded_content_summary(card["source_excluded_content_id"])
            return card
        if target_type == "excluded":
            item = self.get_excluded_content(target_id)
            if not item:
                return None
            item["object_type"] = "excluded"
            item["object_id"] = item["id"]
            item["display_title"] = item["label"]
            item["export_eligible"] = False
            return item
        if target_type == "matrix_item":
            item = self.get_matrix_item(target_id)
            if not item:
                return None
            item["object_type"] = "matrix_item"
            item["object_id"] = item["id"]
            item["display_title"] = item["summary"]
            item["export_eligible"] = is_card_export_eligible((item.get("review") or {}).get("decision", ""))
            return item
        return None

    def list_access_queue(self, run_id: Optional[str] = None) -> list[dict]:
        base_query = """
            SELECT access_queue.*, papers.title AS paper_title, papers.original_url,
                   papers.publication_year, papers.external_id AS paper_external_id,
                   (SELECT dr.asset_url FROM discovery_results dr
                    WHERE dr.paper_id = access_queue.paper_id
                      AND dr.asset_url IS NOT NULL AND dr.asset_url != ''
                    ORDER BY dr.created_at DESC LIMIT 1) AS best_asset_url
            FROM access_queue
            JOIN papers ON papers.id = access_queue.paper_id
        """
        if run_id:
            return self._fetchall(
                base_query + " WHERE access_queue.run_id = ? ORDER BY access_queue.created_at DESC",
                (run_id,),
            )
        return self._fetchall(base_query + " ORDER BY access_queue.created_at DESC")

    def list_cards_for_export(self, run_id: str, card_ids: list[str]) -> list[dict]:
        placeholders = ",".join("?" for _ in card_ids) or "''"
        params = [run_id] + card_ids
        rows = self._fetchall(
            f"""
            SELECT candidate_cards.*, papers.title AS paper_title, papers.original_url AS paper_url, topics.name AS topic_name
            FROM candidate_cards
            JOIN papers ON papers.id = candidate_cards.paper_id
            JOIN topics ON topics.id = candidate_cards.topic_id
            WHERE candidate_cards.run_id = ? AND candidate_cards.id IN ({placeholders})
            ORDER BY topics.name ASC, papers.title ASC, candidate_cards.created_at ASC
            """,
            tuple(params),
        )
        for row in rows:
            self._hydrate_card_row(row)
            row["excluded_content"] = self.list_excluded_content(
                paper_id=row["paper_id"],
                topic_id=row["topic_id"],
                run_id=row["run_id"],
            )
            row["judgement"] = self._fetchone(
                "SELECT * FROM judgements WHERE card_id = ? ORDER BY created_at DESC LIMIT 1",
                (row["id"],),
            )
            row["review"] = self.get_latest_review_decision("card", row["id"])
        return rows

    def list_matrix_items_for_export(self, run_id: str, matrix_item_ids: list[str]) -> list[dict]:
        placeholders = ",".join("?" for _ in matrix_item_ids) or "''"
        params = [run_id] + matrix_item_ids
        rows = self._fetchall(
            f"""
            SELECT evidence_matrix_items.*, papers.title AS paper_title, papers.original_url AS paper_url,
                   papers.publication_year AS publication_year, papers.access_status AS paper_access_status,
                   papers.parse_status AS paper_parse_status, topics.name AS topic_name
            FROM evidence_matrix_items
            JOIN papers ON papers.id = evidence_matrix_items.paper_id
            JOIN topics ON topics.id = evidence_matrix_items.topic_id
            WHERE evidence_matrix_items.run_id = ? AND evidence_matrix_items.id IN ({placeholders})
            ORDER BY evidence_matrix_items.dimension_label ASC, evidence_matrix_items.outcome_label ASC, evidence_matrix_items.created_at ASC
            """,
            tuple(params),
        )
        paper_qa_capability_cache: dict[str, dict[str, Any]] = {}
        for row in rows:
            self._hydrate_matrix_row(row, paper_qa_capability_cache=paper_qa_capability_cache)
            row["review"] = self.get_latest_review_decision("matrix_item", row["id"])
            row["comment"] = self.get_review_item_comment("matrix_item", row["id"])
            row["figures"] = self.get_figures_by_ids(row["paper_id"], row.get("figure_ids", []))
        return rows

    def build_neighbors(self, card_id: str, limit: int = 5) -> list[dict]:
        current = self.get_card(card_id)
        if not current:
            return []
        neighbors = []
        for candidate in self.list_cards(topic=current["topic_name"]):
            if candidate["id"] == card_id:
                continue
            similarity = cosine_similarity(current["embedding"], candidate["embedding"])
            relationship = classify_neighbor_relationship(similarity)
            neighbors.append(
                {
                    "id": candidate["id"],
                    "title": candidate["title"],
                    "paper_title": candidate["paper_title"],
                    "topic_name": candidate["topic_name"],
                    "similarity": round(similarity, 4),
                    "relationship": relationship,
                    "relationship_reason": neighbor_relationship_reason(relationship),
                    "review_decision": candidate.get("review_decision", ""),
                }
            )
        neighbors.sort(key=lambda item: item["similarity"], reverse=True)
        return neighbors[:limit]


class DiscoveryService:
    def __init__(
        self,
        providers: Optional[list[Any]] = None,
        strategy_builder: Optional[Any] = None,
    ):
        self.providers = providers or [
            OpenAlexDiscoveryProvider(),
            ArxivDiscoveryProvider(),
            CrossrefDiscoveryProvider(),
            SemanticScholarDiscoveryProvider(),
        ]
        self.strategy_builder = strategy_builder or build_topic_search_strategies

    def discover(self, topic: str) -> list[dict]:
        raw_results = []
        strategies = self.strategy_builder(topic)
        return self.discover_with_strategies(topic, strategies)

    def discover_with_strategies(self, topic: str, strategies: list[dict[str, Any]]) -> list[dict]:
        raw_results = []
        for strategy in strategies:
            allowlist = {
                str(item).strip().lower()
                for item in strategy.get("provider_allowlist", [])
                if str(item).strip()
            }
            for provider in self.providers:
                provider_name = str(getattr(provider, "provider_name", provider.__class__.__name__)).strip().lower()
                if allowlist and provider_name not in allowlist:
                    continue
                raw_results.extend(self._discover_with_provider(provider, topic, strategy))
        deduped: dict[str, dict] = {}
        for result in raw_results:
            dedupe_key = build_discovery_identity(
                result["title"],
                result.get("publication_year"),
                result.get("authors", []),
                result.get("ids", {}),
            )
            source = {
                "provider": result["provider"],
                "strategy_family": result.get("strategy_family", "core"),
                "strategy_type": result.get("strategy_type", "topic_query"),
                "strategy_order": int(result.get("strategy_order", 0)),
                "query_text": result.get("query_text", topic),
                "source_external_id": result.get("source_external_id", ""),
                "original_url": result.get("original_url", ""),
                "asset_url": result.get("asset_url", ""),
                "confidence": result.get("confidence", 0.0),
                "metadata": result.get("metadata", {}),
                "ids": result.get("ids", {}),
                "strategy_params": result.get("strategy_params", {}),
            }
            current = deduped.get(dedupe_key)
            if not current:
                deduped[dedupe_key] = {
                    "provider": result["provider"],
                    "title": result["title"],
                    "authors": result.get("authors", []),
                    "publication_year": result.get("publication_year"),
                    "external_id": dedupe_key,
                    "source_external_id": result.get("source_external_id", ""),
                    "original_url": result.get("original_url", ""),
                    "asset_url": result.get("asset_url", ""),
                    "confidence": result.get("confidence", 0.0),
                    "metadata": {
                        "identifiers": result.get("ids", {}),
                        "primary_provider": result["provider"],
                    },
                    "discovery_sources": [source],
                }
                continue
            current["discovery_sources"].append(source)
            if self._prefer_source(current, result):
                current.update(
                    {
                        "provider": result["provider"],
                        "title": result["title"],
                        "authors": result.get("authors", []),
                        "publication_year": result.get("publication_year"),
                        "source_external_id": result.get("source_external_id", ""),
                        "original_url": result.get("original_url", ""),
                        "asset_url": result.get("asset_url", ""),
                        "confidence": result.get("confidence", 0.0),
                        "metadata": {
                            "identifiers": result.get("ids", {}),
                            "primary_provider": result["provider"],
                        },
                    }
                )
        return sorted(
            deduped.values(),
            key=lambda item: (
                -float(item.get("confidence", 0.0)),
                -(item.get("publication_year") or 0),
                item.get("title", "").lower(),
            ),
        )

    def _discover_with_provider(self, provider: Any, topic: str, strategy: dict[str, Any]) -> list[dict]:
        try:
            results = provider.discover(topic, strategy)
        except TypeError:
            results = provider.discover(topic)
        normalized = []
        for result in results or []:
            if not isinstance(result, dict):
                continue
            normalized_result = {
                **result,
                "strategy_family": result.get("strategy_family", strategy.get("strategy_family", "core")),
                "strategy_type": result.get("strategy_type", strategy.get("strategy_type", "topic_query")),
                "strategy_order": int(result.get("strategy_order", strategy.get("strategy_order", 0))),
                "query_text": result.get("query_text", strategy.get("query_text", topic)),
                "strategy_params": result.get("strategy_params", strategy.get("params", {})),
            }
            normalized.append(normalized_result)
        return normalized

    def _prefer_source(self, current: dict, candidate: dict) -> bool:
        current_score = (
            1 if current.get("asset_url") else 0,
            len((current.get("metadata") or {}).get("identifiers", {})),
            float(current.get("confidence", 0.0)),
        )
        candidate_score = (
            1 if candidate.get("asset_url") else 0,
            len(candidate.get("ids", {})),
            float(candidate.get("confidence", 0.0)),
        )
        return candidate_score > current_score


class OpenAlexDiscoveryProvider:
    provider_name = "openalex"

    def discover(self, topic: str, strategy: Optional[dict[str, Any]] = None) -> list[dict]:
        strategy = strategy or {}
        query_text = str(strategy.get("query_text", topic)).strip() or topic
        query = urllib.parse.quote(query_text)
        result_limit = max(1, min(int((strategy.get("params") or {}).get("result_limit", 5) or 5), 15))
        url = f"https://api.openalex.org/works?search={query}&per-page={result_limit}"
        year_from = (strategy.get("params") or {}).get("year_from")
        if isinstance(year_from, int):
            filter_param = urllib.parse.quote(f"from_publication_date:{year_from}-01-01")
            url += f"&filter={filter_param}"
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        records = []
        for item in payload.get("results", []):
            pdf_url = (
                item.get("open_access", {}).get("oa_url")
                or item.get("primary_location", {}).get("pdf_url")
                or ""
            )
            landing_url = item.get("primary_location", {}).get("landing_page_url") or item.get("id", "")
            records.append(
                {
                    "provider": "openalex",
                    "strategy_family": strategy.get("strategy_family", "core"),
                    "strategy_type": strategy.get("strategy_type", "topic_query"),
                    "strategy_order": int(strategy.get("strategy_order", 0)),
                    "query_text": query_text,
                    "title": item.get("display_name", "Untitled"),
                    "authors": [author["author"]["display_name"] for author in item.get("authorships", [])[:5] if author.get("author")],
                    "publication_year": item.get("publication_year"),
                    "source_external_id": item.get("id", ""),
                    "original_url": landing_url,
                    "asset_url": pdf_url,
                    "confidence": 0.7,
                    "strategy_params": strategy.get("params", {}),
                    "ids": {
                        "doi": item.get("doi", ""),
                        "openalex": item.get("id", ""),
                    },
                    "metadata": item,
                }
            )
        return records


class ArxivDiscoveryProvider:
    provider_name = "arxiv"

    def discover(self, topic: str, strategy: Optional[dict[str, Any]] = None) -> list[dict]:
        strategy = strategy or {}
        query_text = str(strategy.get("query_text", topic)).strip() or topic
        query = urllib.parse.quote(query_text)
        sort_params = ""
        if str(strategy.get("strategy_type", "")) == "recent_window":
            sort_params = "&sortBy=submittedDate&sortOrder=descending"
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5{sort_params}"
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                feed = ET.fromstring(response.read().decode("utf-8"))
        except Exception:
            return []
        records = []
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in feed.findall("atom:entry", namespace):
            arxiv_id = entry.findtext("atom:id", default="", namespaces=namespace)
            pdf_url = ""
            for link in entry.findall("atom:link", namespace):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            records.append(
                {
                    "provider": "arxiv",
                    "strategy_family": strategy.get("strategy_family", "core"),
                    "strategy_type": strategy.get("strategy_type", "topic_query"),
                    "strategy_order": int(strategy.get("strategy_order", 0)),
                    "query_text": query_text,
                    "title": entry.findtext("atom:title", default="Untitled", namespaces=namespace).strip(),
                    "authors": [author.findtext("atom:name", default="", namespaces=namespace) for author in entry.findall("atom:author", namespace)],
                    "publication_year": int(entry.findtext("atom:published", default="1900", namespaces=namespace)[:4]),
                    "source_external_id": arxiv_id,
                    "original_url": arxiv_id,
                    "asset_url": pdf_url,
                    "confidence": 0.8,
                    "strategy_params": strategy.get("params", {}),
                    "ids": {"arxiv": arxiv_id},
                    "metadata": {"summary": entry.findtext("atom:summary", default="", namespaces=namespace)},
                }
            )
        return records


class CrossrefDiscoveryProvider:
    provider_name = "crossref"

    def discover(self, topic: str, strategy: Optional[dict[str, Any]] = None) -> list[dict]:
        strategy = strategy or {}
        query_text = str(strategy.get("query_text", topic)).strip() or topic
        query = urllib.parse.quote(query_text)
        result_limit = max(1, min(int((strategy.get("params") or {}).get("result_limit", 5) or 5), 15))
        url = f"https://api.crossref.org/works?query={query}&rows={result_limit}"
        year_from = (strategy.get("params") or {}).get("year_from")
        if isinstance(year_from, int):
            url += f"&filter={urllib.parse.quote(f'from-pub-date:{year_from}-01-01')}"
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        records = []
        for item in payload.get("message", {}).get("items", []):
            title = " ".join(item.get("title", [])).strip() or "Untitled"
            authors = [
                " ".join(part for part in [author.get("given", ""), author.get("family", "")] if part).strip()
                for author in item.get("author", [])[:5]
            ]
            authors = [author for author in authors if author]
            publication_year = None
            published_parts = item.get("published-print", {}).get("date-parts") or item.get("published-online", {}).get("date-parts") or []
            if published_parts and published_parts[0]:
                publication_year = int(published_parts[0][0])
            doi = item.get("DOI", "")
            records.append(
                {
                    "provider": "crossref",
                    "strategy_family": strategy.get("strategy_family", "core"),
                    "strategy_type": strategy.get("strategy_type", "topic_query"),
                    "strategy_order": int(strategy.get("strategy_order", 0)),
                    "query_text": query_text,
                    "title": title,
                    "authors": authors,
                    "publication_year": publication_year,
                    "source_external_id": doi or item.get("URL", ""),
                    "original_url": item.get("URL", ""),
                    "asset_url": "",
                    "confidence": 0.55,
                    "strategy_params": strategy.get("params", {}),
                    "ids": {"doi": doi},
                    "metadata": item,
                }
            )
        return records


class SemanticScholarDiscoveryProvider:
    provider_name = "semantic_scholar"

    def discover(self, topic: str, strategy: Optional[dict[str, Any]] = None) -> list[dict]:
        strategy = strategy or {}
        query_text = str(strategy.get("query_text", topic)).strip() or topic
        query = urllib.parse.quote(query_text)
        fields = urllib.parse.quote("title,year,authors,url,openAccessPdf,externalIds,abstract")
        result_limit = max(1, min(int((strategy.get("params") or {}).get("result_limit", 5) or 5), 15))
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={result_limit}&fields={fields}"
        request = urllib.request.Request(url, headers={"User-Agent": "paper2bullet/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []
        records = []
        for item in payload.get("data", []):
            external_ids = item.get("externalIds", {}) or {}
            records.append(
                {
                    "provider": "semantic_scholar",
                    "strategy_family": strategy.get("strategy_family", "core"),
                    "strategy_type": strategy.get("strategy_type", "topic_query"),
                    "strategy_order": int(strategy.get("strategy_order", 0)),
                    "query_text": query_text,
                    "title": item.get("title", "Untitled"),
                    "authors": [author.get("name", "") for author in item.get("authors", [])[:5] if author.get("name")],
                    "publication_year": item.get("year"),
                    "source_external_id": item.get("paperId", ""),
                    "original_url": item.get("url", ""),
                    "asset_url": (item.get("openAccessPdf") or {}).get("url", ""),
                    "confidence": 0.65,
                    "strategy_params": strategy.get("params", {}),
                    "ids": {
                        "doi": external_ids.get("DOI", ""),
                        "arxiv": external_ids.get("ArXiv", ""),
                        "semantic_scholar": item.get("paperId", ""),
                    },
                    "metadata": {"abstract": item.get("abstract", ""), "external_ids": external_ids},
                }
            )
        return records


class PdfParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._markitdown = self._build_markitdown()

    def _build_markitdown(self):
        if MarkItDown is None:
            return None
        try:
            return MarkItDown(enable_plugins=False)
        except TypeError:
            return MarkItDown()
        except Exception:
            return None

    def parse(self, paper: dict) -> dict:
        source_path = Path(paper["artifact_path"] or paper["local_path"])
        artifact_kind = self._detect_artifact_kind(source_path)
        if artifact_kind == "pdf":
            return self._parse_pdf(source_path)
        if artifact_kind == "html":
            return self._parse_html(source_path)
        raise ParseFailure("parse_failed", f"Unsupported or unrecognized artifact type: {source_path.name}")

    def _detect_artifact_kind(self, source_path: Path) -> str:
        head = source_path.read_bytes()[:4096]
        lowered_head = head.lower()
        if head.startswith(b"%PDF-"):
            return "pdf"
        if b"<html" in lowered_head or b"<!doctype html" in lowered_head:
            return "html"
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".html", ".htm"}:
            return "html"
        if lowered_head.lstrip().startswith(b"<"):
            return "html"
        return "unknown"

    def _normalize_pdf_text(self, text: str) -> str:
        replacements = {
            r"\(": "(",
            r"\)": ")",
            r"\\": "\\",
            r"\n": "\n",
            r"\r": "\r",
            r"\t": "\t",
            r"\b": " ",
            r"\f": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), text)
        return text

    def _markdown_to_readable_text(self, markdown_text: str) -> str:
        text = markdown_text.replace("\r\n", "\n")
        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda match: f"\n\nFigure: {match.group(1).strip() or match.group(2).strip()}\n\n", text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
        text = text.replace("```", "\n")
        text = text.replace("`", "")
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
        text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _figure_asset_dir(self, source_path: Path) -> Path:
        target = self.settings.figure_assets_dir / source_path.stem
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _build_figure_record(
        self,
        *,
        figure_label: str,
        caption: str,
        page_number: Optional[int] = None,
        linked_section_ids: Optional[list[str]] = None,
        asset_status: str = "metadata_only",
        asset_kind: str = "",
        asset_local_path: str = "",
        asset_source_url: str = "",
        mime_type: str = "",
        byte_size: int = 0,
        sha256: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        validation_error: str = "",
    ) -> dict[str, Any]:
        storage_path = asset_local_path or asset_source_url
        return {
            "id": new_id("figure"),
            "figure_label": figure_label,
            "caption": caption,
            "page_number": page_number,
            "storage_path": storage_path,
            "asset_status": asset_status,
            "asset_kind": asset_kind,
            "asset_local_path": asset_local_path,
            "asset_source_url": asset_source_url,
            "mime_type": mime_type,
            "byte_size": int(byte_size or 0),
            "sha256": sha256,
            "width": width,
            "height": height,
            "validation_error": validation_error,
            "linked_section_ids": list(linked_section_ids or []),
        }

    def _infer_html_base_url(self, html_text: str) -> str:
        patterns = [
            r"<base\b[^>]*href=['\"]([^'\"]+)['\"]",
            r"<link\b[^>]*rel=['\"]canonical['\"][^>]*href=['\"]([^'\"]+)['\"]",
            r"<meta\b[^>]*name=['\"]citation_fulltext_html_url['\"][^>]*content=['\"]([^'\"]+)['\"]",
            r"<meta\b[^>]*property=['\"]og:url['\"][^>]*content=['\"]([^'\"]+)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.I)
            if match:
                return match.group(1).strip()
        return ""

    def _write_figure_asset_bytes(
        self,
        *,
        source_path: Path,
        asset_bytes: bytes,
        target_name_hint: str,
        suffix: str,
    ) -> str:
        suffix = suffix if suffix.startswith(".") else f".{suffix.lstrip('.')}" if suffix else ".bin"
        target_hash = stable_hash(target_name_hint)[:16]
        destination = self._figure_asset_dir(source_path) / f"{slugify(target_name_hint)}-{target_hash}{suffix.lower()}"
        destination.write_bytes(asset_bytes)
        return str(destination)

    def _validate_local_image_asset(self, asset_path: Path) -> dict[str, Any]:
        if not asset_path.exists():
            return {"asset_status": "metadata_only", "validation_error": "asset file does not exist"}
        byte_size = asset_path.stat().st_size
        if byte_size <= 0:
            return {"asset_status": "metadata_only", "validation_error": "asset file is empty"}
        mime_type = mimetypes.guess_type(asset_path.name)[0] or ""
        width = None
        height = None
        if Image is not None:
            try:
                with Image.open(asset_path) as image:
                    width, height = image.size
                    mime_type = Image.MIME.get(image.format, mime_type) or mime_type
            except Exception as error:
                return {"asset_status": "metadata_only", "validation_error": f"image validation failed: {error}"}
        if width is not None and width <= 0:
            return {"asset_status": "metadata_only", "validation_error": "image width is invalid"}
        if height is not None and height <= 0:
            return {"asset_status": "metadata_only", "validation_error": "image height is invalid"}
        sha256 = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        return {
            "asset_status": "validated_local_asset",
            "asset_local_path": str(asset_path),
            "mime_type": mime_type,
            "byte_size": byte_size,
            "sha256": sha256,
            "width": width,
            "height": height,
            "validation_error": "",
        }

    def _pick_best_src_candidate(self, raw_src: str, raw_srcset: str) -> str:
        srcset = str(raw_srcset or "").strip()
        if srcset:
            candidates = []
            for item in srcset.split(","):
                token = item.strip().split()[0] if item.strip() else ""
                if token:
                    candidates.append(token)
            if candidates:
                return candidates[-1]
        return str(raw_src or "").strip()

    def _resolve_html_figure_target(self, target: str, source_path: Path, base_url: str) -> tuple[str, str]:
        target = str(target or "").strip()
        if not target:
            return "", ""
        if target.startswith("data:"):
            return target, "data_uri"
        if target.startswith("//"):
            return "https:" + target, "remote"
        if target.startswith(("http://", "https://")):
            return target, "remote"
        if target.startswith("/"):
            if base_url:
                return urllib.parse.urljoin(base_url, target), "remote"
            return target, "unresolved"
        candidate_path = (source_path.parent / urllib.parse.unquote(target)).resolve()
        if candidate_path.exists():
            return str(candidate_path), "local"
        if base_url:
            return urllib.parse.urljoin(base_url, target), "remote"
        return "", "unresolved"

    def _materialize_figure_asset(
        self,
        *,
        source_path: Path,
        raw_target: str,
        figure_label: str,
        base_url: str = "",
        asset_kind_hint: str = "",
    ) -> dict[str, Any]:
        resolved_target, target_kind = self._resolve_html_figure_target(raw_target, source_path, base_url)
        if target_kind == "data_uri" and resolved_target.startswith("data:"):
            header, encoded = resolved_target.split(",", 1)
            try:
                asset_bytes = base64.b64decode(encoded, validate=False)
            except Exception as error:
                return {
                    "asset_status": "metadata_only",
                    "asset_kind": "data_uri",
                    "asset_source_url": resolved_target[:80],
                    "validation_error": f"data URI decode failed: {error}",
                }
            mime_match = re.match(r"data:([^;]+)", header, flags=re.I)
            mime_type = mime_match.group(1).strip().lower() if mime_match else ""
            suffix = mimetypes.guess_extension(mime_type) or ".bin"
            asset_path = Path(
                self._write_figure_asset_bytes(
                    source_path=source_path,
                    asset_bytes=asset_bytes,
                    target_name_hint=f"{figure_label}-{mime_type or 'data-uri'}",
                    suffix=suffix,
                )
            )
            validated = self._validate_local_image_asset(asset_path)
            validated.update({"asset_kind": "data_uri", "asset_source_url": resolved_target[:80]})
            return validated
        if target_kind == "local" and resolved_target:
            original = Path(resolved_target)
            suffix = original.suffix or ".bin"
            copied_path = Path(
                self._write_figure_asset_bytes(
                    source_path=source_path,
                    asset_bytes=original.read_bytes(),
                    target_name_hint=f"{figure_label}-{original.name}",
                    suffix=suffix,
                )
            )
            validated = self._validate_local_image_asset(copied_path)
            validated.update({"asset_kind": "local_copy", "asset_source_url": ""})
            return validated
        if target_kind == "remote" and resolved_target.startswith(("http://", "https://")):
            request = urllib.request.Request(
                resolved_target,
                headers={"User-Agent": "paper2bullet/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.remote_asset_timeout_seconds) as response:
                    asset_bytes = response.read()
                    content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip()
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
                return {
                    "asset_status": "external_reference_only",
                    "asset_kind": asset_kind_hint or "remote_download",
                    "asset_source_url": resolved_target,
                    "validation_error": f"remote asset download failed: {error}",
                }
            suffix = Path(urllib.parse.urlparse(resolved_target).path).suffix or mimetypes.guess_extension(content_type) or ".bin"
            downloaded_path = Path(
                self._write_figure_asset_bytes(
                    source_path=source_path,
                    asset_bytes=asset_bytes,
                    target_name_hint=f"{figure_label}-{Path(urllib.parse.urlparse(resolved_target).path).name or 'remote'}",
                    suffix=suffix,
                )
            )
            validated = self._validate_local_image_asset(downloaded_path)
            validated.update(
                {
                    "asset_kind": asset_kind_hint or "remote_download",
                    "asset_source_url": resolved_target,
                    "mime_type": validated.get("mime_type") or content_type,
                }
            )
            return validated
        if resolved_target:
            return {
                "asset_status": "external_reference_only",
                "asset_kind": asset_kind_hint or "external_reference",
                "asset_source_url": resolved_target,
                "validation_error": "asset could not be materialized locally",
            }
        return {
            "asset_status": "metadata_only",
            "asset_kind": asset_kind_hint,
            "validation_error": "no resolvable asset target",
        }

    def _extract_caption_only_figures_from_sections(self, sections: list[dict]) -> list[dict]:
        figures = []
        seen_labels: set[str] = set()
        pattern = re.compile(r"^\s*(fig(?:ure)?|table)\s*\.?\s*(\d+[A-Za-z]?)\s*[:.\-]?\s*(.+)$", flags=re.I | re.S)
        inline_pattern = re.compile(r"(fig(?:ure)?|table)\s*\.?\s*(\d+[A-Za-z]?)\s*[:.\-]?\s*", flags=re.I)
        for section in sections:
            paragraph = str(section.get("paragraph_text", "")).strip()
            match = pattern.match(paragraph)
            extracted_matches = [match] if match else []
            if not extracted_matches:
                extracted_matches = list(inline_pattern.finditer(paragraph))
            for extracted in extracted_matches:
                prefix = "Figure" if extracted.group(1).lower().startswith("fig") else "Table"
                label = f"{prefix} {extracted.group(2)}"
                normalized_label = label.lower()
                if normalized_label in seen_labels:
                    continue
                if hasattr(extracted, "group") and extracted.re is inline_pattern:
                    raw_caption = paragraph[extracted.end():].strip()
                    if len(raw_caption) < 20:
                        continue
                    raw_caption = raw_caption[:260]
                    split_match = re.search(r"(?<=[.?!])\s+(?=[A-Z][a-z]+,)", raw_caption)
                    if split_match and split_match.start() >= 24:
                        raw_caption = raw_caption[: split_match.start()]
                    caption = re.sub(r"\s+", " ", raw_caption).strip()
                else:
                    caption = re.sub(r"\s+", " ", extracted.group(3)).strip()
                if len(caption) < 12:
                    continue
                seen_labels.add(normalized_label)
                figures.append(
                    self._build_figure_record(
                        figure_label=label,
                        caption=caption or label,
                        page_number=section.get("page_number"),
                        linked_section_ids=[section["id"]],
                        asset_status="metadata_only",
                        asset_kind="caption_only",
                    )
                )
        return figures

    def _extract_figures_from_html(self, html_text: str, source_path: Path) -> list[dict]:
        base_url = self._infer_html_base_url(html_text)
        if BeautifulSoup is None:
            figures = []
            for index, block in enumerate(re.finditer(r"<figure\b.*?>.*?</figure>", html_text, flags=re.I | re.S), start=1):
                snippet = block.group(0)
                img_match = re.search(r"<img\b[^>]*src=['\"]([^'\"]+)['\"]", snippet, flags=re.I)
                if not img_match:
                    continue
                figcaption_match = re.search(r"<figcaption\b[^>]*>(.*?)</figcaption>", snippet, flags=re.I | re.S)
                alt_match = re.search(r"<img\b[^>]*alt=['\"]([^'\"]*)['\"]", snippet, flags=re.I)
                caption_html = (figcaption_match.group(1) if figcaption_match else "") or (alt_match.group(1) if alt_match else "")
                caption = re.sub(r"<[^>]+>", " ", caption_html)
                caption = re.sub(r"\s+", " ", caption).strip() or f"Figure {index}"
                asset = self._materialize_figure_asset(
                    source_path=source_path,
                    raw_target=img_match.group(1),
                    figure_label=f"Figure {index}",
                    base_url=base_url,
                    asset_kind_hint="html_image",
                )
                figures.append(
                    self._build_figure_record(
                        figure_label=f"Figure {index}",
                        caption=caption,
                        linked_section_ids=[],
                        **asset,
                    )
                )
            return figures
        figures = []
        soup = BeautifulSoup(html_text, "html.parser")
        image_index = 0
        for image in soup.find_all("img"):
            parent_figure = image.find_parent("figure")
            metadata_blob = " ".join(
                [
                    " ".join(parent_figure.get("class", [])) if parent_figure else "",
                    str(parent_figure.get("id", "")) if parent_figure else "",
                    str(image.get("alt", "")),
                    str(image.get("title", "")),
                    str(image.get("src", "")),
                    str(image.get("srcset", "")),
                ]
            ).lower()
            if not parent_figure and not any(token in metadata_blob for token in ["fig", "figure"]):
                continue
            image_index += 1
            raw_target = self._pick_best_src_candidate(image.get("src", ""), image.get("srcset", ""))
            if not raw_target:
                continue
            caption_text = ""
            label = f"Figure {image_index}"
            if parent_figure:
                caption_node = parent_figure.find("figcaption")
                if caption_node:
                    caption_text = caption_node.get_text(" ", strip=True)
                    label_match = re.search(r"(fig(?:ure)?\.?\s*\d+[A-Za-z]?)", caption_text, flags=re.I)
                    if label_match:
                        normalized = re.sub(r"\s+", " ", label_match.group(1)).replace("Fig.", "Figure").replace("Fig", "Figure")
                        label = normalized.strip().rstrip(".")
            caption = caption_text or str(image.get("alt", "")).strip() or str(image.get("title", "")).strip() or label
            caption = re.sub(r"^\s*fig(?:ure)?\.?\s*\d+[A-Za-z]?\s*[:.\-]?\s*", "", caption, flags=re.I).strip() or label
            asset = self._materialize_figure_asset(
                source_path=source_path,
                raw_target=raw_target,
                figure_label=label,
                base_url=base_url,
                asset_kind_hint="html_image",
            )
            figures.append(
                self._build_figure_record(
                    figure_label=label,
                    caption=caption,
                    linked_section_ids=[],
                    **asset,
                )
            )
        return figures

    def _dedupe_figures(self, figures: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        by_key: dict[tuple[str, str], dict] = {}
        for figure in figures:
            label = str(figure.get("figure_label", "")).strip()
            caption = str(figure.get("caption", "")).strip()
            key = (
                re.sub(r"\s+", " ", label).lower(),
                re.sub(r"\s+", " ", caption).lower(),
            )
            existing = by_key.get(key)
            if not existing:
                figure["linked_section_ids"] = list(dict.fromkeys(figure.get("linked_section_ids", [])))
                deduped.append(figure)
                by_key[key] = figure
                continue
            existing_rank = {"validated_local_asset": 3, "external_reference_only": 2, "metadata_only": 1}.get(
                existing.get("asset_status", "metadata_only"),
                0,
            )
            new_rank = {"validated_local_asset": 3, "external_reference_only": 2, "metadata_only": 1}.get(
                figure.get("asset_status", "metadata_only"),
                0,
            )
            if new_rank > existing_rank:
                for field_name in [
                    "storage_path",
                    "asset_status",
                    "asset_kind",
                    "asset_local_path",
                    "asset_source_url",
                    "mime_type",
                    "byte_size",
                    "sha256",
                    "width",
                    "height",
                    "validation_error",
                    "page_number",
                ]:
                    existing[field_name] = figure.get(field_name, existing.get(field_name))
            existing["linked_section_ids"] = list(
                dict.fromkeys((existing.get("linked_section_ids", []) or []) + (figure.get("linked_section_ids", []) or []))
            )
        return deduped

    def _link_figures_to_sections(self, figures: list[dict], sections: list[dict]) -> list[dict]:
        for figure in figures:
            linked_ids = list(dict.fromkeys(figure.get("linked_section_ids", []) or []))
            label = str(figure.get("figure_label", "")).strip()
            caption = str(figure.get("caption", "")).strip()
            caption_excerpt = caption[:120]
            label_variants = {
                label.lower(),
                label.lower().replace("figure", "fig."),
                label.lower().replace("figure", "fig"),
                label.lower().replace(" ", ""),
            }
            for section in sections:
                paragraph = str(section.get("paragraph_text", "")).strip().lower()
                if not paragraph:
                    continue
                if any(variant and variant in paragraph for variant in label_variants):
                    linked_ids.append(section["id"])
                    continue
                if caption_excerpt and caption_excerpt.lower() in paragraph:
                    linked_ids.append(section["id"])
            figure["linked_section_ids"] = list(dict.fromkeys(linked_ids))
        return figures

    def _extract_figures_from_markdown(self, markdown_text: str, source_path: Path) -> list[dict]:
        figures = []
        for index, match in enumerate(re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", markdown_text), start=1):
            caption = match.group(1).strip() or Path(match.group(2).strip()).name or f"Figure {index}"
            target = match.group(2).strip()
            asset = self._materialize_figure_asset(
                source_path=source_path,
                raw_target=target,
                figure_label=f"Figure {index}",
                asset_kind_hint="markdown_image",
            )
            figures.append(
                self._build_figure_record(
                    figure_label=f"Figure {index}",
                    caption=caption,
                    linked_section_ids=[],
                    **asset,
                )
            )
        return figures

    def _extract_pdf_image_assets(self, source_path: Path) -> list[dict[str, Any]]:
        if fitz is None:
            return []
        try:
            document = fitz.open(source_path)
        except Exception:
            return []
        assets: list[dict[str, Any]] = []
        seen_xrefs: set[tuple[int, int]] = set()
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                    xref = int(image_info[0] or 0)
                    if xref <= 0 or (page_index, xref) in seen_xrefs:
                        continue
                    seen_xrefs.add((page_index, xref))
                    try:
                        extracted = document.extract_image(xref)
                    except Exception:
                        continue
                    image_bytes = extracted.get("image")
                    if not image_bytes:
                        continue
                    ext = str(extracted.get("ext") or "bin").strip().lower() or "bin"
                    asset_path = Path(
                        self._write_figure_asset_bytes(
                            source_path=source_path,
                            asset_bytes=image_bytes,
                            target_name_hint=f"pdf-page-{page_index + 1}-image-{image_index}",
                            suffix=f".{ext}",
                        )
                    )
                    validated = self._validate_local_image_asset(asset_path)
                    validated.update(
                        {
                            "asset_kind": "pdf_embedded",
                            "asset_source_url": f"pdf://{source_path.name}#page={page_index + 1}&xref={xref}",
                            "page_number": page_index + 1,
                        }
                    )
                    assets.append(validated)
        finally:
            document.close()
        return assets

    def _attach_pdf_assets_to_figures(self, figures: list[dict], source_path: Path) -> list[dict]:
        assets = self._extract_pdf_image_assets(source_path)
        if not assets:
            return figures
        assets_by_page: dict[int, list[dict[str, Any]]] = {}
        for asset in assets:
            page_number = int(asset.get("page_number") or 0)
            if page_number > 0:
                assets_by_page.setdefault(page_number, []).append(asset)
        unused_assets = list(assets)
        for figure in figures:
            if figure.get("asset_status") == "validated_local_asset":
                continue
            selected_asset = None
            page_number = int(figure.get("page_number") or 0)
            if page_number > 0:
                page_assets = assets_by_page.get(page_number, [])
                if page_assets:
                    selected_asset = page_assets.pop(0)
            if selected_asset is None and unused_assets:
                selected_asset = unused_assets.pop(0)
            elif selected_asset in unused_assets:
                unused_assets.remove(selected_asset)
            if not selected_asset:
                continue
            for field_name in [
                "asset_status",
                "asset_kind",
                "asset_local_path",
                "asset_source_url",
                "mime_type",
                "byte_size",
                "sha256",
                "width",
                "height",
                "validation_error",
            ]:
                figure[field_name] = selected_asset.get(field_name, figure.get(field_name))
            figure["storage_path"] = figure.get("asset_local_path") or figure.get("asset_source_url", "")
        return figures

    def _parse_pdf_with_markitdown(self, source_path: Path) -> Optional[dict]:
        if not self._markitdown:
            return None
        try:
            result = self._markitdown.convert(str(source_path))
        except Exception:
            return None

        markdown_text = getattr(result, "text_content", "") or ""
        if not markdown_text.strip():
            return None

        readable_text = self._markdown_to_readable_text(markdown_text)
        if not readable_text:
            return None
        self._validate_pdf_text(readable_text, source_path)

        paragraphs = split_paragraphs(readable_text)
        if not paragraphs:
            return None

        sections = [
            {
                "id": new_id("section"),
                "section_order": index,
                "section_title": "Markdown Extraction",
                "paragraph_text": paragraph,
                "page_number": None,
                "embedding": embedding_for_text(paragraph),
            }
            for index, paragraph in enumerate(paragraphs, start=1)
        ]
        sections = enrich_sections_with_structure(sections, "pdf_markitdown")
        figures = self._dedupe_figures(
            self._extract_figures_from_markdown(markdown_text, source_path)
            + self._extract_caption_only_figures_from_sections(sections)
        )
        figures = self._link_figures_to_sections(figures, sections)
        figures = self._attach_pdf_assets_to_figures(figures, source_path)
        return {"sections": sections, "figures": figures, "artifact_type": "pdf"}

    def _is_probably_readable_pdf_fragment(self, text: str) -> bool:
        candidate = re.sub(r"\s+", " ", text).strip()
        if len(candidate) < 20:
            return False

        lowered = candidate.lower()
        blocked_tokens = [
            "%pdf-",
            "endstream",
            "endobj",
            "xref",
            "trailer",
            "/type /page",
            " stream x",
            "jfif",
            "matlab handle graphics",
        ]
        if any(token in lowered for token in blocked_tokens):
            return False

        control_chars = sum(1 for char in candidate if ord(char) < 32 and char not in "\n\r\t")
        if control_chars / max(len(candidate), 1) > 0.01:
            return False

        readable_chars = sum(
            1 for char in candidate if char.isalpha() or char.isdigit() or char.isspace() or char in ".,;:!?-()[]/%'\""
        )
        return readable_chars / max(len(candidate), 1) >= 0.85

    def _validate_pdf_text(self, text: str, source_path: Path) -> None:
        cleaned = text.strip()
        if len(cleaned) < 40:
            raise ParseFailure("parse_failed", f"Could not extract enough readable PDF text from {source_path.name}")

        lowered = cleaned.lower()
        blocked_tokens = [
            "%pdf-",
            "endstream",
            "endobj",
            "xref",
            "trailer",
            "/type /page",
            " stream x",
            "jfif",
            "matlab handle graphics",
        ]
        if any(token in lowered for token in blocked_tokens):
            raise ParseFailure("quality_failed", f"PDF extraction for {source_path.name} contains PDF structure noise")

        control_chars = sum(1 for char in cleaned if ord(char) < 32 and char not in "\n\r\t")
        if control_chars / max(len(cleaned), 1) > 0.01:
            raise ParseFailure("quality_failed", f"PDF extraction for {source_path.name} contains too many control characters")

        readable_chars = sum(
            1 for char in cleaned if char.isalpha() or char.isdigit() or char.isspace() or char in ".,;:!?-()[]/%'\""
        )
        if readable_chars / max(len(cleaned), 1) < 0.85:
            raise ParseFailure("quality_failed", f"PDF extraction for {source_path.name} is not readable enough")

    def _parse_pdf(self, source_path: Path) -> dict:
        parsed_with_markitdown = self._parse_pdf_with_markitdown(source_path)
        if parsed_with_markitdown:
            return parsed_with_markitdown

        text_pages = self._extract_pdf_pages(source_path)
        if not text_pages:
            raise ParseFailure("parse_failed", f"Could not extract text from PDF: {source_path.name}")
        sections = []
        order = 0
        for page_number, page_text in enumerate(text_pages, start=1):
            for paragraph in split_paragraphs(page_text):
                order += 1
                sections.append(
                    {
                        "id": new_id("section"),
                        "section_order": order,
                        "section_title": f"Page {page_number}",
                        "paragraph_text": paragraph,
                        "page_number": page_number,
                        "embedding": embedding_for_text(paragraph),
                    }
                )
        sections = enrich_sections_with_structure(sections, "pdf_fallback")
        figures = self._link_figures_to_sections(self._extract_caption_only_figures_from_sections(sections), sections)
        figures = self._attach_pdf_assets_to_figures(figures, source_path)
        return {"sections": sections, "figures": figures, "artifact_type": "pdf"}

    def _parse_html(self, source_path: Path) -> dict:
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        figures = self._extract_figures_from_html(text, source_path)
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        paragraphs = split_paragraphs(text)
        sections = [
            {
                "id": new_id("section"),
                "section_order": index,
                "section_title": "HTML Snapshot",
                "paragraph_text": paragraph,
                "page_number": None,
                "embedding": embedding_for_text(paragraph),
            }
            for index, paragraph in enumerate(paragraphs, start=1)
        ]
        if not sections:
            sections = [
                {
                    "id": new_id("section"),
                    "section_order": index,
                    "section_title": "HTML Figure Caption",
                    "paragraph_text": figure.get("caption", "") or figure.get("figure_label", "Figure"),
                    "page_number": None,
                    "embedding": embedding_for_text(figure.get("caption", "") or figure.get("figure_label", "Figure")),
                }
                for index, figure in enumerate(figures, start=1)
                if str(figure.get("caption", "") or figure.get("figure_label", "")).strip()
            ]
        if not sections:
            raise ParseFailure("parse_failed", f"Could not extract readable HTML text from {source_path.name}")
        sections = enrich_sections_with_structure(sections, "html")
        figures = self._dedupe_figures(figures + self._extract_caption_only_figures_from_sections(sections))
        figures = self._link_figures_to_sections(figures, sections)
        return {"sections": sections, "figures": figures, "artifact_type": "html"}

    def _extract_pdf_pages(self, source_path: Path) -> list[str]:
        if fitz is not None:
            try:
                document = fitz.open(source_path)
            except Exception:
                document = None
            if document is not None:
                try:
                    pages = []
                    for page_index in range(document.page_count):
                        page = document.load_page(page_index)
                        page_text = page.get_text("text")
                        page_text = re.sub(r"\s+\n", "\n", page_text)
                        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
                        if page_text:
                            pages.append(page_text)
                    if pages:
                        combined = "\n\n".join(pages)
                        try:
                            self._validate_pdf_text(combined, source_path)
                            return pages
                        except ParseFailure:
                            # PyMuPDF sometimes truncates tiny synthetic PDFs; fall back to raw BT/ET extraction
                            # before declaring the document unreadable.
                            pass
                finally:
                    document.close()
        raw_text = source_path.read_bytes().decode("latin-1", errors="ignore")
        fragments = []

        for text_block in re.finditer(r"BT\b(.*?)\bET", raw_text, flags=re.S):
            block = text_block.group(1)

            for match in re.finditer(r"\((.*?)(?<!\\)\)\s*Tj", block, flags=re.S):
                fragments.append(self._normalize_pdf_text(match.group(1)))

            for array_match in re.finditer(r"\[(.*?)\]\s*TJ", block, flags=re.S):
                strings = re.findall(r"\((.*?)(?<!\\)\)", array_match.group(1), flags=re.S)
                if strings:
                    fragments.append("".join(self._normalize_pdf_text(item) for item in strings))

        normalized_fragments = []
        for fragment in fragments:
            fragment = re.sub(r"\s+", " ", fragment).strip()
            if self._is_probably_readable_pdf_fragment(fragment):
                normalized_fragments.append(fragment)

        if not normalized_fragments:
            return []

        page_text = "\n\n".join(normalized_fragments)
        self._validate_pdf_text(page_text, source_path)
        return [page_text]


class PaperPipeline:
    def __init__(self, settings: Settings, repository: Repository, card_engine: Optional[LLMCardEngine] = None):
        self.settings = settings
        self.repository = repository
        self.parser = PdfParser(settings)
        self.card_engine = card_engine or LLMCardEngine(settings)

    def acquire_remote_asset(self, paper: dict, asset_url: str) -> Optional[str]:
        if not asset_url:
            return None
        parsed = urllib.parse.urlparse(asset_url)
        destination_base = self.settings.artifacts_dir / paper["id"]
        temporary_path = destination_base.with_suffix(".download")
        try:
            req = urllib.request.Request(
                asset_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; paper2bullet/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=self.settings.remote_asset_timeout_seconds) as response, temporary_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            suffix = self._infer_downloaded_asset_suffix(temporary_path, parsed.path)
            destination = destination_base.with_suffix(suffix)
            if destination.exists():
                destination.unlink()
            temporary_path.replace(destination)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            temporary_path.unlink(missing_ok=True)
            return None
        return str(destination)

    def _unpaywall_get_pdf_url(self, doi: str) -> Optional[str]:
        """Query Unpaywall API for a direct open-access PDF URL given a DOI."""
        doi = doi.strip()
        if not doi:
            return None
        api_url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='')}?email=paper2bullet@local"
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "paper2bullet/1.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read())
            locations: list = []
            best = data.get("best_oa_location")
            if best:
                locations.append(best)
            locations += data.get("oa_locations") or []
            for loc in locations:
                pdf_url = loc.get("url_for_pdf")
                if pdf_url:
                    return str(pdf_url).strip()
        except Exception:
            pass
        return None

    def acquire_remote_asset_with_oa_fallback(self, paper: dict, asset_url: str) -> Optional[str]:
        """Try acquiring the asset from asset_url first; on failure, fall back to Unpaywall."""
        result = self.acquire_remote_asset(paper, asset_url)
        if result:
            return result
        # Extract DOI from external_id (format: "doi::10.xxxx/...")
        external_id = str(paper.get("external_id") or "")
        doi = ""
        if external_id.startswith("doi::"):
            doi = external_id[5:]
        elif external_id.startswith("doi:"):
            doi = external_id[4:]
        if not doi:
            return None
        oa_pdf_url = self._unpaywall_get_pdf_url(doi)
        if not oa_pdf_url:
            return None
        return self.acquire_remote_asset(paper, oa_pdf_url)

    def _infer_downloaded_asset_suffix(self, downloaded_path: Path, url_path: str) -> str:
        head = downloaded_path.read_bytes()[:4096]
        lowered = head.lower()
        if head.startswith(b"%PDF-"):
            return ".pdf"
        if b"<html" in lowered or b"<!doctype html" in lowered or lowered.lstrip().startswith(b"<"):
            return ".html"
        hinted_suffix = Path(url_path).suffix.lower()
        if hinted_suffix in {".pdf", ".html", ".htm"}:
            return hinted_suffix
        return ".bin"

    def ingest_local_pdf(self, path: str) -> str:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Local PDF path does not exist: {path}")
        if not source.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if source.suffix.lower() not in {".pdf", ".html", ".htm"}:
            raise ValueError(f"Unsupported local file type: {path}")
        destination = self.settings.artifacts_dir / f"{slugify(source.stem)}-{stable_hash(str(source.resolve()))[:10]}{source.suffix.lower()}"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return str(destination)

    def parse_and_store(self, paper: dict) -> int:
        try:
            parsed = self.parser.parse(paper)
        except ParseFailure as error:
            self.repository.persist_parse_result(
                paper_id=paper["id"],
                sections=[],
                figures=[],
                parse_status=error.status,
                ingestion_status="parse_failed",
                parse_failure_reason=error.reason,
                card_generation_status="blocked_parse_failed",
                card_generation_failure_reason="Card generation was skipped because paper parsing failed.",
            )
            return 0
        parsed_snapshot_path = self.settings.parsed_dir / f"{paper['id']}.json"
        parsed_snapshot_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        self.repository.persist_parse_result(
            paper_id=paper["id"],
            sections=parsed["sections"],
            figures=parsed["figures"],
            parse_status="parsed",
            ingestion_status="ready",
            parse_failure_reason="",
            card_generation_status="pending",
            card_generation_failure_reason="",
            artifact_path=paper["artifact_path"] or paper["local_path"],
        )
        return len(parsed["sections"])

    def build_cards(self, paper: dict, topic: dict, run_id: str, *, active_memory: Optional[dict[str, Any]] = None) -> int:
        sections = self.repository.get_sections(paper["id"])
        if not self.card_engine.is_enabled():
            self.repository.replace_generation_outputs_for_paper_topic(paper["id"], topic["id"], run_id, [], [])
            self.repository.update_paper(
                paper["id"],
                card_generation_status="llm_unavailable",
                card_generation_failure_reason="LLM-only card generation is enabled, but no LLM provider is configured.",
            )
            return 0
        try:
            generation_output = self.generate_outputs_for_sections(
                sections,
                topic,
                paper,
                run_id=run_id,
                active_memory=active_memory,
            )
        except LLMGenerationError as error:
            self.repository.replace_generation_outputs_for_paper_topic(paper["id"], topic["id"], run_id, [], [])
            self.repository.update_paper(
                paper["id"],
                card_generation_status="llm_failed",
                card_generation_failure_reason=str(error),
            )
            return 0
        cards = generation_output["cards"]
        excluded_items = generation_output["excluded_content"]
        if not cards:
            self.repository.replace_generation_outputs_for_paper_topic(
                paper["id"],
                topic["id"],
                run_id,
                [],
                excluded_items,
            )
            self.repository.update_paper(
                paper["id"],
                card_generation_status="no_cards",
                card_generation_failure_reason="LLM completed successfully but did not return any valid candidate cards.",
            )
            return 0
        self.repository.replace_generation_outputs_for_paper_topic(
            paper["id"],
            topic["id"],
            run_id,
            cards,
            excluded_items,
        )
        self.repository.update_paper(
            paper["id"],
            card_generation_status="generated",
            card_generation_failure_reason="",
        )
        return len(cards)

    def build_matrix_items(
        self,
        paper: dict,
        topic: dict,
        run_id: str,
        *,
        claim_plan: dict[str, Any],
        active_memory: Optional[dict[str, Any]] = None,
    ) -> int:
        sections = self.repository.get_sections(paper["id"])
        abstract_only_fallback = False
        if not sections:
            sections = self._build_claim_evidence_abstract_fallback_sections(paper)
            abstract_only_fallback = bool(sections)
        if not sections:
            self.repository.replace_matrix_items_for_paper_topic(paper["id"], topic["id"], run_id, [])
            self.repository.update_paper(
                paper["id"],
                card_generation_status="matrix_no_items",
                card_generation_failure_reason="No parsed sections or abstract fallback evidence were available for claim evidence generation.",
            )
            return 0
        if not self.card_engine.is_enabled():
            self.repository.replace_matrix_items_for_paper_topic(paper["id"], topic["id"], run_id, [])
            self.repository.update_paper(
                paper["id"],
                card_generation_status="matrix_llm_unavailable",
                card_generation_failure_reason="LLM-only claim evidence generation is enabled, but no LLM provider is configured.",
            )
            return 0
        try:
            generation_output = self.generate_matrix_items_for_sections(
                sections,
                topic,
                paper,
                claim_plan=claim_plan,
                active_memory=active_memory,
                abstract_only_fallback=abstract_only_fallback,
            )
        except LLMGenerationError as error:
            self.repository.replace_matrix_items_for_paper_topic(paper["id"], topic["id"], run_id, [])
            self.repository.update_paper(
                paper["id"],
                card_generation_status="matrix_llm_failed",
                card_generation_failure_reason=str(error),
            )
            return 0
        items = generation_output["items"]
        self.repository.replace_matrix_items_for_paper_topic(paper["id"], topic["id"], run_id, items)
        self.repository.update_paper(
            paper["id"],
            card_generation_status=(
                "matrix_generated_abstract_only"
                if items and abstract_only_fallback
                else "matrix_generated"
                if items
                else "matrix_no_items_abstract_only"
                if abstract_only_fallback
                else "matrix_no_items"
            ),
            card_generation_failure_reason="" if items else "LLM completed successfully but did not return any valid evidence matrix items.",
        )
        return len(items)

    def generate_outputs_for_sections(
        self,
        sections: list[dict],
        topic: dict,
        paper: dict,
        run_id: str = "",
        *,
        active_memory: Optional[dict[str, Any]] = None,
    ) -> dict:
        return self._build_cards_with_llm(sections, topic, paper, run_id=run_id, active_memory=active_memory)

    def generate_matrix_items_for_sections(
        self,
        sections: list[dict],
        topic: dict,
        paper: dict,
        *,
        claim_plan: dict[str, Any],
        active_memory: Optional[dict[str, Any]] = None,
        abstract_only_fallback: bool = False,
    ) -> dict[str, list[dict]]:
        figures = self.repository.get_figures(paper["id"])
        topic_entry = self._resolve_claim_plan_topic_entry(claim_plan, topic["name"])
        if not self._paper_is_direct_claim_evidence_match(paper, sections, topic_entry, claim_plan):
            return {"items": []}
        query_anchor = str(topic_entry.get("query_anchor", "") or topic["name"]).strip()
        evidence_packet = self._build_evidence_packet(sections, figures, query_anchor)
        generated = self.card_engine.generate_matrix_items(
            claim_text=str(claim_plan.get("claim", "") or claim_plan.get("research_brief", "")).strip(),
            topic_name=topic["name"],
            paper_title=paper["title"],
            dimension=topic_entry,
            sections=evidence_packet["prompt_sections"],
            figures=evidence_packet["figure_candidates"],
            active_memory=active_memory,
            evidence_policy={
                **(claim_plan.get("evidence_policy", {}) or {}),
                "abstract_only_fallback": abstract_only_fallback,
            },
        )
        generated_items = generated.get("items", [])
        if abstract_only_fallback and generated_items:
            generated_items = self._downgrade_abstract_only_matrix_items(generated_items)
        return {
            "items": [self._finalize_matrix_item(item, topic_entry, claim_plan) for item in generated_items],
        }

    def _build_claim_evidence_abstract_fallback_sections(self, paper: dict) -> list[dict]:
        best_abstract = self._extract_best_metadata_abstract_for_paper(paper["id"])
        if len(best_abstract) < 120:
            return []
        return [
            {
                "id": f"abstract_meta::{paper['id']}",
                "paper_id": paper["id"],
                "section_order": 0,
                "section_title": "Abstract (discovery metadata)",
                "paragraph_text": best_abstract,
                "page_number": None,
                "section_kind": "abstract",
                "section_label": "metadata_abstract",
                "is_front_matter": False,
                "is_abstract": True,
                "is_body": False,
                "body_role": "",
                "has_figure_reference": False,
                "source_format": "metadata_abstract",
                "selection_score": 0.62,
                "selection_reason_json": {"reason": "claim_evidence_abstract_fallback"},
            }
        ]

    def _extract_best_metadata_abstract_for_paper(self, paper_id: str) -> str:
        paper_sources = self.repository.list_paper_sources(paper_id)
        abstract_candidates: list[str] = []
        for source in paper_sources:
            metadata = source.get("metadata", {}) if isinstance(source.get("metadata"), dict) else {}
            abstract_text = _extract_source_metadata_abstract(metadata)
            if abstract_text:
                abstract_candidates.append(abstract_text)
        if not abstract_candidates:
            return ""
        return max(abstract_candidates, key=len).strip()

    def _downgrade_abstract_only_matrix_items(self, items: list[dict]) -> list[dict]:
        downgraded: list[dict] = []
        for item in items[:1]:
            adjusted = dict(item)
            adjusted["evidence_strength"] = "weak"
            limitation = str(adjusted.get("limitation_text", "")).strip()
            abstract_note = "仅基于论文摘要/发现元数据生成，尚未核对全文，因此只能作为弱证据使用。"
            if abstract_note not in limitation:
                adjusted["limitation_text"] = f"{limitation} {abstract_note}".strip()
            downgraded.append(adjusted)
        return downgraded

    def _paper_is_direct_claim_evidence_match(
        self,
        paper: dict,
        sections: list[dict],
        topic_entry: dict[str, Any],
        claim_plan: dict[str, Any],
    ) -> bool:
        context = infer_claim_evidence_context(
            str(topic_entry.get("query_anchor", "") or topic_entry.get("topic_name", "")).strip(),
            outcome_terms=topic_entry.get("outcome_terms", []),
            claim_text=str(claim_plan.get("claim", "")).strip(),
            research_brief=str(claim_plan.get("research_brief", "")).strip(),
        )
        if not context["requires_workplace_context"]:
            return True
        title_text = _joined_lower_text(paper.get("title", ""))
        sampled_sections = sections[:8]
        section_text = _joined_lower_text(
            " ".join(section.get("section_title", "") for section in sampled_sections),
            " ".join(section.get("paragraph_text", "")[:320] for section in sampled_sections),
        )
        searchable_text = _joined_lower_text(title_text, section_text)
        direct_hits = sum(1 for token in WORKPLACE_DIRECT_EVIDENCE_TOKENS if token in searchable_text)
        offdomain_hits = sum(1 for token in OFFDOMAIN_CLAIM_EVIDENCE_TITLE_TOKENS if token in title_text)
        if direct_hits > 0:
            return True
        if offdomain_hits > 0:
            return False
        return False

    def _extract_candidates_with_optional_memory(self, **kwargs) -> dict[str, Any]:
        active_memory = kwargs.pop("active_memory", None)
        try:
            return self.card_engine.extract_candidates(active_memory=active_memory, **kwargs)
        except TypeError as error:
            if "active_memory" not in str(error):
                raise
            return self.card_engine.extract_candidates(**kwargs)

    def _judge_candidates_with_optional_memory(self, **kwargs) -> dict[str, Any]:
        active_memory = kwargs.pop("active_memory", None)
        try:
            return self.card_engine.judge_candidates(active_memory=active_memory, **kwargs)
        except TypeError as error:
            if "active_memory" not in str(error):
                raise
            return self.card_engine.judge_candidates(**kwargs)

    def _build_paper_understanding(
        self,
        *,
        sections: list[dict],
        figures: list[dict],
        topic_name: str,
        paper_title: str,
    ) -> dict[str, Any]:
        packet = self._build_evidence_packet(sections, figures, topic_name)
        body_candidates = [item for item in sections if item.get("is_body")]
        body_candidates.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
        if not body_candidates:
            body_candidates = sorted(sections, key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
        understanding_sections = body_candidates[:28]
        intro_context = [item for item in sections if item.get("body_role") == "introduction"][:6]
        abstract_context = [item for item in sections if item.get("is_abstract")][:4]
        merged_sections = []
        seen_ids: set[str] = set()
        for section in intro_context + abstract_context + understanding_sections:
            if section["id"] in seen_ids:
                continue
            merged_sections.append(section)
            seen_ids.add(section["id"])
        if self.card_engine.is_enabled() and hasattr(self.card_engine, "build_paper_understanding"):
            try:
                llm_understanding = self.card_engine.build_paper_understanding(
                    topic_name=topic_name,
                    paper_title=paper_title,
                    sections=merged_sections,
                    figures=figures,
                )
                if llm_understanding and (
                    llm_understanding.get("global_contribution_objects")
                    or str(llm_understanding.get("paper_relevance_verdict", "")).strip().lower()
                    in {"borderline_reject", "off_topic_hard"}
                    or str(llm_understanding.get("paper_relevance_reason", "")).strip()
                ):
                    llm_understanding["version"] = "understanding-v2-llm"
                    llm_understanding["topic_name"] = topic_name
                    llm_understanding["paper_title"] = paper_title
                    llm_understanding["selection_overview"] = self._build_selection_overview(packet["selection_diagnostics"])
                    for item in llm_understanding["global_contribution_objects"]:
                        item["label"] = self._normalize_object_label(item.get("label", ""), merged_sections, item.get("evidence_section_ids", []))
                    return llm_understanding
            except Exception:
                pass
        objects = []
        for index, section in enumerate(body_candidates[:5], start=1):
            object_id = f"obj_{index}"
            role = str(section.get("body_role", "")).lower()
            if index == 1:
                level_hint = "overall"
            elif role in {"methods", "discussion", "introduction"}:
                level_hint = "local"
            else:
                level_hint = "detail"
            linked_figures = [
                figure["id"]
                for figure in figures
                if section["id"] in set(figure.get("linked_section_ids", []))
            ]
            objects.append(
                {
                    "id": object_id,
                    "label": self._normalize_object_label(
                        section.get("section_label") or section.get("section_title") or section.get("paragraph_text", "")[:80],
                        sections,
                        [section["id"]],
                    ),
                    "object_type": section.get("section_kind", "other"),
                    "level_hint": level_hint,
                    "evidence_section_ids": [section["id"]],
                    "evidence_figure_ids": linked_figures,
                    "summary": section.get("paragraph_text", "")[:220],
                    "importance_score": round(float(section.get("selection_score", 0.0)), 4),
                }
            )
        if not objects:
            return {
                "version": "understanding-v1",
                "topic_name": topic_name,
                "paper_title": paper_title,
                "paper_relevance_verdict": "borderline_reject",
                "paper_relevance_reason": "Fallback understanding could not name a concrete course object from this paper.",
                "relevance_failure_type": "cannot_name_course_object",
                "global_contribution_objects": [],
                "contribution_graph": [],
                "evidence_index": {},
                "candidate_level_hints": {},
                "selection_overview": self._build_selection_overview(packet["selection_diagnostics"]),
            }
        graph = []
        for item in objects[1:]:
            graph.append({"from": objects[0]["id"], "to": item["id"], "relation": "supports"})
        evidence_index = {
            item["id"]: {
                "section_ids": item["evidence_section_ids"],
                "figure_ids": item["evidence_figure_ids"],
            }
            for item in objects
        }
        level_hints = {item["id"]: item["level_hint"] for item in objects}
        return {
            "version": "understanding-v1",
            "topic_name": topic_name,
            "paper_title": paper_title,
            "paper_relevance_verdict": "on_topic",
            "paper_relevance_reason": "",
            "relevance_failure_type": "",
            "global_contribution_objects": objects,
            "contribution_graph": graph,
            "evidence_index": evidence_index,
            "candidate_level_hints": level_hints,
            "selection_overview": self._build_selection_overview(packet["selection_diagnostics"]),
        }

    def _build_card_plan(
        self,
        *,
        understanding: dict[str, Any],
        topic_name: str,
    ) -> dict[str, Any]:
        if self.card_engine.is_enabled() and hasattr(self.card_engine, "build_card_plan"):
            try:
                active_calibration_set = self.repository.get_active_calibration_set()
                llm_plan = self.card_engine.build_card_plan(
                    topic_name=topic_name,
                    paper_title=understanding.get("paper_title", ""),
                    understanding=understanding,
                    max_cards=3,
                    calibration_examples=(active_calibration_set or {}).get("examples", []),
                    calibration_set_name=(active_calibration_set or {}).get("name", ""),
                )
                if llm_plan and (
                    "planned_cards" in llm_plan
                    or str(llm_plan.get("paper_relevance_verdict", "")).strip().lower()
                    in {"borderline_reject", "off_topic_hard"}
                    or str(llm_plan.get("paper_relevance_reason", "")).strip()
                ):
                    llm_plan["version"] = CARD_PLAN_PROMPT_VERSION
                    llm_plan["topic_name"] = topic_name
                    return llm_plan
            except Exception:
                pass
        paper_relevance_verdict = str(understanding.get("paper_relevance_verdict", "on_topic")).strip().lower() or "on_topic"
        paper_relevance_reason = str(understanding.get("paper_relevance_reason", "")).strip()
        relevance_failure_type = str(understanding.get("relevance_failure_type", "")).strip().lower()
        if paper_relevance_verdict != "on_topic":
            return {
                "version": "card-plan-v1",
                "topic_name": topic_name,
                "paper_relevance_verdict": paper_relevance_verdict,
                "paper_relevance_reason": paper_relevance_reason,
                "relevance_failure_type": relevance_failure_type,
                "planned_cards": [],
                "coverage_report": {"produce": 0, "exclude": 0, "overall": 0, "local": 0, "detail": 0},
            }
        objects = list(understanding.get("global_contribution_objects", []))
        objects.sort(key=lambda item: float(item.get("importance_score", 0.0)), reverse=True)
        planned_cards = []
        for obj in objects:
            evidence_ids = [str(s) for s in (obj.get("evidence_section_ids") or [])]
            level = str(obj.get("level_hint", "detail")).strip().lower()
            if level not in {"overall", "local", "detail"}:
                level = "detail"
            disposition = "produce" if evidence_ids else "exclude"
            planned_cards.append(
                {
                    "plan_id": f"plan_{obj['id']}",
                    "level": level,
                    "target_object_id": obj["id"],
                    "target_object_label": obj.get("label", ""),
                    "why_valuable_for_course": f"{topic_name} 课程中可讲的对象：{obj.get('label', '')}",
                    "must_have_evidence_ids": evidence_ids[:2],
                    "optional_supporting_ids": evidence_ids[2:4],
                    "disposition": disposition,
                    "disposition_reason": "" if disposition == "produce" else "Missing evidence anchor.",
                }
            )
        return {
            "version": "card-plan-v1",
            "topic_name": topic_name,
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

    def _build_selection_overview(self, diagnostics_by_section_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
        ranked = sorted(
            diagnostics_by_section_id.items(),
            key=lambda item: float((item[1] or {}).get("score", 0.0)),
            reverse=True,
        )
        top_sections = []
        for section_id, item in ranked[:20]:
            top_sections.append(
                {
                    "section_id": section_id,
                    "score": round(float((item or {}).get("score", 0.0)), 4),
                    "section_kind": (item or {}).get("section_kind", "other"),
                    "body_role": (item or {}).get("body_role", ""),
                    "matched_topic_tokens": (item or {}).get("matched_topic_tokens", []),
                }
            )
        return {
            "total_scored_sections": len(diagnostics_by_section_id),
            "top_sections": top_sections,
        }

    def _normalize_object_label(self, label: str, sections: list[dict], evidence_section_ids: list[str]) -> str:
        candidate = str(label or "").strip()
        if candidate and candidate.lower() not in {"markdown extraction", "html snapshot"} and not re.fullmatch(r"page\s+\d+", candidate.lower()):
            return candidate
        section_map = {section["id"]: section for section in sections}
        for section_id in evidence_section_ids:
            section = section_map.get(section_id)
            if not section:
                continue
            text = str(section.get("paragraph_text", "")).strip()
            if not text:
                continue
            snippet = re.split(r"[.;:!?]", text)[0].strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > 96:
                snippet = snippet[:96].rstrip() + "..."
            if snippet:
                return snippet
        return candidate or "Unnamed contribution object"

    def _assemble_plan_driven_packet(
        self,
        *,
        sections: list[dict],
        figures: list[dict],
        topic_name: str,
        card_plan: dict[str, Any],
    ) -> dict[str, Any]:
        base_packet = self._build_evidence_packet(sections, figures, topic_name)
        section_map = {section["id"]: section for section in sections}
        figure_map = {figure["id"]: figure for figure in figures}
        planned_ids = []
        planned_figure_ids = []
        for item in card_plan.get("planned_cards", []):
            if item.get("disposition") != "produce":
                continue
            for section_id in item.get("must_have_evidence_ids", []) + item.get("optional_supporting_ids", []):
                if section_id in section_map and section_id not in planned_ids:
                    planned_ids.append(section_id)
            for figure_id in item.get("must_have_figure_ids", []) + item.get("optional_supporting_figure_ids", []):
                if figure_id in figure_map and figure_id not in planned_figure_ids:
                    planned_figure_ids.append(figure_id)
        if not planned_ids:
            if planned_figure_ids:
                base_packet["figure_candidates"] = [figure_map[figure_id] for figure_id in planned_figure_ids]
            return base_packet
        planned_sections = [section_map[section_id] for section_id in planned_ids]
        for section in planned_sections:
            section["role_hint"] = "primary"
        context_sections = [item for item in sections if item.get("is_abstract")][:2]
        prompt_sections = context_sections + [item for item in planned_sections if item["id"] not in {ctx["id"] for ctx in context_sections}]
        selected_ids = {item["id"] for item in prompt_sections}
        figure_candidates = [
            figure for figure in figures
            if set(figure.get("linked_section_ids", [])).intersection(selected_ids)
        ]
        for figure_id in planned_figure_ids:
            figure = figure_map.get(figure_id)
            if figure and figure not in figure_candidates:
                figure_candidates.append(figure)
        if not figure_candidates:
            figure_candidates = [figure_map[figure_id] for figure_id in planned_figure_ids] or figures[:4]
        base_packet["prompt_sections"] = prompt_sections
        base_packet["figure_candidates"] = figure_candidates
        return base_packet

    def _recover_candidates_with_expanded_context(
        self,
        *,
        topic_name: str,
        paper_title: str,
        sections: list[dict],
        figures: list[dict],
        planned_cards: list[dict],
        planning_context: dict[str, Any],
        calibration_examples: list[dict],
        calibration_set_name: str,
        active_memory: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        body_sections = [section for section in sections if section.get("is_body")]
        body_sections.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
        expanded_sections = (body_sections[:10] or sections[:10]) + [item for item in sections if item.get("is_abstract")][:2]
        deduped = []
        seen: set[str] = set()
        for section in expanded_sections:
            if section["id"] in seen:
                continue
            deduped.append(section)
            seen.add(section["id"])
        return self._extract_candidates_with_optional_memory(
            topic_name=topic_name,
            paper_title=paper_title,
            sections=deduped,
            figures=figures,
            planned_cards=planned_cards,
            planning_context=planning_context,
            calibration_examples=calibration_examples,
            calibration_set_name=calibration_set_name,
            active_memory=active_memory,
        )

    def _align_cards_to_plan(self, cards: list[dict], card_plan: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        planned_produce = [item for item in card_plan.get("planned_cards", []) if item.get("disposition") == "produce"]
        if not planned_produce:
            paper_relevance_verdict = str(card_plan.get("paper_relevance_verdict", "on_topic")).strip().lower() or "on_topic"
            paper_relevance_reason = str(card_plan.get("paper_relevance_reason", "")).strip()
            relevance_failure_type = str(card_plan.get("relevance_failure_type", "")).strip().lower() or "other"
            excluded: list[dict] = []
            for card in cards:
                excluded.append(
                    {
                        "label": card.get("title", "未命中计划的候选"),
                        "exclusion_type": relevance_failure_type if paper_relevance_verdict != "on_topic" else "other",
                        "reason": (
                            paper_relevance_reason
                            if paper_relevance_verdict != "on_topic"
                            else "Card plan produced zero slots, so extraction output cannot survive without an approved plan slot."
                        ),
                        "section_ids": card.get("primary_section_ids", []) or [item.get("section_id") for item in card.get("evidence", []) if item.get("section_id")],
                    }
                )
            return [], excluded
        remaining = sorted(
            cards,
            key=lambda item: (
                {"green": 3, "yellow": 2, "red": 1}.get((item.get("judgement") or {}).get("color", "yellow"), 2),
                {"strong": 3, "medium": 2, "weak": 1}.get(item.get("evidence_level", "medium"), 2),
                len(item.get("primary_section_ids", [])),
            ),
            reverse=True,
        )
        kept: list[dict] = []
        used_ids: set[str] = set()
        for plan_item in planned_produce:
            must_have = set(plan_item.get("must_have_evidence_ids", []))
            must_have_figures = set(plan_item.get("must_have_figure_ids", []))
            best_card = None
            best_score = -1.0
            for card in remaining:
                if card.get("title") in used_ids:
                    continue
                source_plan_id = str(card.get("source_plan_id", "")).strip()
                if source_plan_id and source_plan_id != str(plan_item.get("plan_id", "")).strip():
                    continue
                card_primary = set(
                    card.get("primary_section_ids")
                    or [item.get("section_id") for item in card.get("evidence", []) if item.get("section_id")]
                )
                card_figures = set(card.get("figure_ids", []))
                overlap = len(must_have.intersection(card_primary))
                figure_overlap = len(must_have_figures.intersection(card_figures))
                if must_have and overlap == 0 and not (must_have_figures and figure_overlap > 0):
                    continue
                if must_have_figures and figure_overlap == 0 and not must_have:
                    continue
                object_match_score = compute_plan_object_match_score(plan_item, card)
                exact_source_plan_bonus = 12.0 if source_plan_id == str(plan_item.get("plan_id", "")).strip() else 0.0
                if source_plan_id == str(plan_item.get("plan_id", "")).strip() and object_match_score <= 0:
                    continue
                score = overlap * 10
                score += figure_overlap * 8
                score += object_match_score * 6
                score += exact_source_plan_bonus
                score += {"green": 3, "yellow": 2, "red": 1}.get((card.get("judgement") or {}).get("color", "yellow"), 2)
                score += {"strong": 3, "medium": 2, "weak": 1}.get(card.get("evidence_level", "medium"), 2)
                if score > best_score:
                    best_score = score
                    best_card = card
            if best_card:
                if must_have_figures and not best_card.get("figure_ids"):
                    best_card["figure_ids"] = list(sorted(must_have_figures))
                best_card["planned_level"] = plan_item.get("level", "")
                best_card["plan_id"] = plan_item.get("plan_id", "")
                best_card["plan_target_object_id"] = plan_item.get("target_object_id", "")
                best_card["plan_target_object_label"] = plan_item.get("target_object_label", "")
                kept.append(best_card)
                used_ids.add(best_card.get("title", ""))
        excluded = []
        for card in remaining:
            if card.get("title", "") in used_ids:
                continue
            excluded.append(
                {
                    "label": card.get("title", "计划外候选"),
                    "exclusion_type": "replaced_by_stronger_card",
                    "reason": "Card plan alignment selected stronger matches for planned objects.",
                    "section_ids": card.get("primary_section_ids", []) or [item.get("section_id") for item in card.get("evidence", []) if item.get("section_id")],
                }
            )
        return kept, excluded

    def _build_evidence_packet(self, sections: list[dict], figures: list[dict], topic_name: str) -> dict[str, Any]:
        topic_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", topic_name.lower())
            if len(token) >= 3
        }
        diagnostics_by_section_id: dict[str, dict[str, Any]] = {}
        scored_sections = []
        for section in sections:
            if not section.get("section_kind"):
                section.update(
                    classify_section_metadata(
                        section_title=section.get("section_title", ""),
                        paragraph_text=section.get("paragraph_text", ""),
                        section_order=int(section.get("section_order", 0) or 0),
                        total_sections=len(sections),
                        source_format=section.get("source_format", "legacy"),
                    )
                )
            text = str(section.get("paragraph_text", "")).lower()
            title = str(section.get("section_title", "")).lower()
            matched_tokens = sorted(token for token in topic_tokens if token in text or token in title)
            topic_relevance = min(1.0, len(matched_tokens) / max(len(topic_tokens), 1))
            body_weight = 1.0 if section.get("is_body") else (0.45 if section.get("is_abstract") else 0.2)
            role_weight = {
                "results": 1.0,
                "methods": 0.92,
                "discussion": 0.82,
                "conclusion": 0.7,
                "introduction": 0.65,
            }.get(str(section.get("body_role", "")).lower(), 0.45)
            evidence_density = min(1.0, len(text) / 900.0)
            figure_bonus = 0.2 if section.get("has_figure_reference") else 0.0
            mechanism_bonus = 0.2 if re.search(r"\b(model|method|mechanism|framework|result|failure|ablation)\b", text) else 0.0
            novelty_bonus = 0.15 if re.search(r"\b(we propose|we introduce|novel|first|outperform)\b", text) else 0.0
            score = round(
                (0.32 * topic_relevance)
                + (0.24 * body_weight)
                + (0.16 * role_weight)
                + (0.14 * evidence_density)
                + figure_bonus
                + mechanism_bonus
                + novelty_bonus,
                6,
            )
            reason = {
                "score": score,
                "topic_relevance": round(topic_relevance, 4),
                "body_weight": round(body_weight, 4),
                "role_weight": round(role_weight, 4),
                "evidence_density": round(evidence_density, 4),
                "figure_bonus": round(figure_bonus, 4),
                "mechanism_bonus": round(mechanism_bonus, 4),
                "novelty_bonus": round(novelty_bonus, 4),
                "matched_topic_tokens": matched_tokens,
                "section_kind": section.get("section_kind", "other"),
                "body_role": section.get("body_role", ""),
            }
            section["selection_score"] = score
            section["selection_reason"] = reason
            diagnostics_by_section_id[section["id"]] = reason
            scored_sections.append(section)

        scored_sections.sort(
            key=lambda item: (
                float(item.get("selection_score", 0.0)),
                int(item.get("is_body", False)),
                -1 * int(item.get("section_order", 0)),
            ),
            reverse=True,
        )

        primary_candidate_sections = [item for item in scored_sections if item.get("is_body")][:6]
        if not primary_candidate_sections:
            primary_candidate_sections = [item for item in scored_sections if item.get("is_abstract")][:2]
        context_sections = [item for item in sections if item.get("is_abstract") or item.get("body_role") == "introduction"][:3]
        supporting_sections = [
            item for item in scored_sections
            if item["id"] not in {s["id"] for s in primary_candidate_sections}
        ][:4]

        selected_ids: list[str] = []
        prompt_sections: list[dict] = []
        for group_name, group in (
            ("context", context_sections),
            ("primary", primary_candidate_sections),
            ("supporting", supporting_sections),
        ):
            for section in group:
                if section["id"] in selected_ids:
                    continue
                selected_ids.append(section["id"])
                section["role_hint"] = group_name
                prompt_sections.append(section)

        selected_id_set = set(selected_ids)
        scored_figures: list[tuple[float, dict[str, Any]]] = []
        for figure in figures:
            linked_ids = set(figure.get("linked_section_ids", []))
            caption = str(figure.get("caption", "")).lower()
            matched_tokens = [token for token in topic_tokens if token in caption]
            score = (3.0 * len(linked_ids.intersection(selected_id_set))) + len(matched_tokens)
            asset_status = str(figure.get("asset_status", "")).strip()
            if asset_status == "validated_local_asset":
                score += 1.0
            elif asset_status == "external_reference_only":
                score += 0.25
            if caption:
                score += 0.25
            if score > 0:
                scored_figures.append((score, figure))
        if not scored_figures:
            scored_figures = [
                (
                    len([token for token in topic_tokens if token in str(figure.get("caption", "")).lower()])
                    + (1.0 if figure.get("asset_status", "") == "validated_local_asset" else 0.25 if figure.get("asset_status", "") == "external_reference_only" else 0.0),
                    figure,
                )
                for figure in figures[:4]
            ]
        scored_figures.sort(key=lambda item: item[0], reverse=True)
        figure_candidates = [item[1] for item in scored_figures[:4]]

        return {
            "context_sections": context_sections,
            "primary_candidate_sections": primary_candidate_sections,
            "supporting_sections": supporting_sections,
            "prompt_sections": prompt_sections,
            "figure_candidates": figure_candidates,
            "selection_diagnostics": diagnostics_by_section_id,
        }

    def _gate_extracted_candidates(
        self,
        extracted_cards: list[dict],
        sections: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        section_map = {section["id"]: section for section in sections}
        body_exists = any(section.get("is_body") for section in sections)
        kept: list[dict] = []
        excluded: list[dict] = []
        for candidate in extracted_cards:
            primary_ids = list(candidate.get("primary_section_ids") or [])
            if not primary_ids:
                primary_ids = [item["section_id"] for item in candidate.get("evidence", [])[:1]]
            supporting_ids = list(candidate.get("supporting_section_ids") or [])
            evidence_ids = primary_ids + [sid for sid in supporting_ids if sid not in primary_ids]
            primary_sections = [section_map[sid] for sid in primary_ids if sid in section_map]
            evidence_sections = [section_map[sid] for sid in evidence_ids if sid in section_map]
            fail_reasons: list[str] = []
            if not str(candidate.get("paper_specific_object", "")).strip():
                # Backward-compatible fallback for old extraction contract.
                candidate["paper_specific_object"] = str(candidate.get("title", "")).strip()
            if not str(candidate.get("paper_specific_object", "")).strip():
                fail_reasons.append("paper_specific_object_missing_gate")
            if body_exists and primary_sections and all(section.get("is_abstract") for section in primary_sections):
                fail_reasons.append("abstract_dominant_evidence_gate")
            if body_exists and primary_sections and all(section.get("is_front_matter") for section in primary_sections):
                fail_reasons.append("front_matter_primary_evidence_gate")
            if body_exists and evidence_sections and not any(section.get("is_body") for section in evidence_sections):
                fail_reasons.append("weak_body_grounding_gate")

            if fail_reasons:
                excluded.append(
                    {
                        "label": candidate.get("title", "被门禁拒绝的候选"),
                        "exclusion_type": "insufficient_evidence",
                        "reason": f"Structured grounding gate failed: {', '.join(fail_reasons)}",
                        "section_ids": evidence_ids or [item["section_id"] for item in candidate.get("evidence", [])],
                    }
                )
                continue
            candidate["primary_section_ids"] = primary_ids
            candidate["supporting_section_ids"] = supporting_ids
            kept.append(candidate)
        return kept, excluded

    def _suppress_same_paper_duplicates(self, cards: list[dict]) -> tuple[list[dict], list[dict]]:
        if len(cards) <= 1:
            for card in cards:
                card["duplicate_disposition"] = "kept"
                card["duplicate_rank"] = 1
            return cards, []

        clusters: dict[str, list[dict]] = {}
        for card in cards:
            primary_ids = tuple(sorted(card.get("primary_section_ids", [])))
            signature = (
                card.get("possible_duplicate_signature", "").strip()
                or f"{'|'.join(primary_ids)}::{card.get('paper_specific_object', '').strip().lower()}::{card.get('claim_type', '').strip().lower()}"
            )
            clusters.setdefault(signature or card["title"], []).append(card)

        kept: list[dict] = []
        excluded: list[dict] = []
        for cluster_index, members in enumerate(clusters.values(), start=1):
            members.sort(
                key=lambda item: (
                    {"strong": 3, "medium": 2, "weak": 1}.get(item.get("evidence_level", "medium"), 2),
                    {"green": 3, "yellow": 2, "red": 1}.get((item.get("judgement") or {}).get("color", "yellow"), 2),
                    len(item.get("primary_section_ids", [])),
                    len(item.get("evidence", [])),
                ),
                reverse=True,
            )
            cluster_id = f"dup_cluster_{cluster_index}"
            for rank, card in enumerate(members, start=1):
                card["duplicate_cluster_id"] = cluster_id
                card["duplicate_rank"] = rank
                if rank == 1:
                    card["duplicate_disposition"] = "kept"
                    kept.append(card)
                else:
                    card["duplicate_disposition"] = "suppressed_variant"
                    excluded.append(
                        {
                            "label": card.get("title", "重复候选"),
                            "exclusion_type": "replaced_by_stronger_card",
                            "reason": f"Suppressed by same-paper duplicate governance; replaced by cluster representative {members[0]['title']}.",
                            "section_ids": card.get("primary_section_ids", []) or [item["section_id"] for item in card.get("evidence", [])],
                        }
                    )
        return kept, excluded

    def _gate_judged_cards_for_concept_alignment(self, cards: list[dict]) -> tuple[list[dict], list[dict]]:
        kept: list[dict] = []
        excluded: list[dict] = []
        for card in cards:
            fail_reasons: list[str] = []
            has_belief_gap = has_concept_belief_gap_signal(
                card.get("title", ""),
                card.get("teachable_one_liner", ""),
                (card.get("judgement") or {}).get("reason", ""),
            )
            has_direct_transfer = has_direct_transfer_signal(
                card.get("title", ""),
                card.get("teachable_one_liner", ""),
                card.get("course_transformation", ""),
                card.get("paper_specific_object", ""),
                (card.get("judgement") or {}).get("reason", ""),
            )
            has_source_fidelity = has_source_object_fidelity_signal(
                card.get("course_transformation", ""),
                card.get("title", ""),
                card.get("paper_specific_object", ""),
            )
            if not has_named_course_object_signal(card.get("course_transformation", "")):
                fail_reasons.append("course_object_naming_missing_gate")
            if not (has_belief_gap or has_direct_transfer):
                fail_reasons.append("belief_gap_or_direct_transfer_missing_gate")
            if not (has_source_fidelity or has_direct_transfer):
                fail_reasons.append("source_object_fidelity_missing_gate")
            if fail_reasons:
                excluded.append(
                    {
                        "label": card.get("title", "概念对齐未通过候选"),
                        "exclusion_type": "weak_transfer",
                        "reason": f"CONCEPT alignment gate failed: {', '.join(fail_reasons)}",
                        "section_ids": card.get("primary_section_ids", []) or [item.get("section_id") for item in card.get("evidence", []) if item.get("section_id")],
                    }
                )
                continue
            kept.append(card)
        return kept, excluded

    def _build_cards_with_llm(
        self,
        sections: list[dict],
        topic: dict,
        paper: dict,
        run_id: str = "",
        precomputed_understanding: Optional[dict[str, Any]] = None,
        precomputed_card_plan: Optional[dict[str, Any]] = None,
        persist_records: bool = True,
        active_memory: Optional[dict[str, Any]] = None,
    ) -> dict:
        active_calibration_set = self.repository.get_active_calibration_set()
        figures = self.repository.get_figures(paper["id"])
        understanding = precomputed_understanding or self._build_paper_understanding(
            sections=sections,
            figures=figures,
            topic_name=topic["name"],
            paper_title=paper["title"],
        )
        if run_id and persist_records:
            self.repository.create_paper_understanding_record(
                paper_id=paper["id"],
                topic_id=topic["id"],
                run_id=run_id,
                version=understanding.get("version", "understanding-v1"),
                understanding=understanding,
            )
        card_plan = precomputed_card_plan or self._build_card_plan(
            understanding=understanding,
            topic_name=topic["name"],
        )
        if run_id and persist_records:
            self.repository.create_card_plan(
                paper_id=paper["id"],
                topic_id=topic["id"],
                run_id=run_id,
                version=card_plan.get("version", "card-plan-v1"),
                plan=card_plan,
            )
        planned_card_slots = [
            item
            for item in card_plan.get("planned_cards", [])
            if isinstance(item, dict) and item.get("disposition") == "produce"
        ]
        evidence_packet = self._assemble_plan_driven_packet(
            sections=sections,
            figures=figures,
            topic_name=topic["name"],
            card_plan=card_plan,
        )
        self.repository.update_section_selection_diagnostics(paper["id"], evidence_packet["selection_diagnostics"])
        extracted_output = self._extract_candidates_with_optional_memory(
            topic_name=topic["name"],
            paper_title=paper["title"],
            sections=evidence_packet["prompt_sections"],
            figures=evidence_packet["figure_candidates"],
            planned_cards=planned_card_slots,
            planning_context=card_plan,
            calibration_examples=(active_calibration_set or {}).get("examples", []),
            calibration_set_name=(active_calibration_set or {}).get("name", ""),
            active_memory=active_memory,
        )
        gated_cards, gated_excluded = self._gate_extracted_candidates(extracted_output["cards"], sections)
        recovery_excluded: list[dict] = []
        if not gated_cards and extracted_output["cards"]:
            recovered_extraction = self._recover_candidates_with_expanded_context(
                topic_name=topic["name"],
                paper_title=paper["title"],
                sections=sections,
                figures=evidence_packet["figure_candidates"],
                planned_cards=planned_card_slots,
                planning_context=card_plan,
                calibration_examples=(active_calibration_set or {}).get("examples", []),
                calibration_set_name=(active_calibration_set or {}).get("name", ""),
                active_memory=active_memory,
            )
            recovered_cards, recovered_excluded = self._gate_extracted_candidates(recovered_extraction["cards"], sections)
            if recovered_cards:
                gated_cards = recovered_cards
                extracted_output = recovered_extraction
            recovery_excluded = recovered_excluded
        judged_output = self._judge_candidates_with_optional_memory(
            topic_name=topic["name"],
            paper_title=paper["title"],
            extracted_cards=gated_cards,
            figures=evidence_packet["figure_candidates"],
            calibration_examples=(active_calibration_set or {}).get("examples", []),
            calibration_set_name=(active_calibration_set or {}).get("name", ""),
            active_memory=active_memory,
        )
        concept_kept_cards, concept_excluded = self._gate_judged_cards_for_concept_alignment(judged_output["cards"])
        plan_aligned_cards, plan_excluded = self._align_cards_to_plan(concept_kept_cards, card_plan)
        deduped_cards, duplicate_excluded = self._suppress_same_paper_duplicates(plan_aligned_cards)
        return {
            "cards": [self._finalize_card(card, topic["name"]) for card in deduped_cards],
            "excluded_content": [
                self._finalize_excluded_content(item)
                for item in (extracted_output["excluded_content"] + gated_excluded + recovery_excluded + concept_excluded + plan_excluded + duplicate_excluded)
            ],
        }

    def _finalize_card(self, card: dict, topic_name: str) -> dict:
        evidence = card["evidence"]
        quote_first_blocks = build_quote_first_blocks(card)
        quote_first_markdown = render_quote_first_markdown(card)
        embedding_source = " ".join(
            [
                topic_name,
                card["title"],
                card["course_transformation"],
                card["teachable_one_liner"],
            ]
            + [item["quote"] for item in evidence]
            + [item.get("analysis", "") for item in evidence]
        )
        return {
            "id": new_id("card"),
            "title": card["title"],
            "granularity_level": card["granularity_level"],
            "course_transformation": card["course_transformation"],
            "teachable_one_liner": card["teachable_one_liner"],
            "draft_body": card["draft_body"],
            "body_format": "quote_first_interleaved_analysis",
            "quote_first_blocks": quote_first_blocks,
            "quote_first_markdown": quote_first_markdown,
            "evidence": evidence,
            "figure_ids": card.get("figure_ids", []),
            "status": card.get("status", "candidate"),
            "embedding": embedding_for_text(embedding_source),
            "primary_section_ids": card.get("primary_section_ids", []),
            "supporting_section_ids": card.get("supporting_section_ids", []),
            "paper_specific_object": card.get("paper_specific_object", ""),
            "claim_type": card.get("claim_type", ""),
            "evidence_level": card.get("evidence_level", ""),
            "body_grounding_reason": card.get("body_grounding_reason", ""),
            "grounding_quality": card.get("grounding_quality", ""),
            "duplicate_cluster_id": card.get("duplicate_cluster_id", ""),
            "duplicate_rank": int(card.get("duplicate_rank", 0) or 0),
            "duplicate_disposition": card.get("duplicate_disposition", ""),
            "planned_level": card.get("planned_level", ""),
            "plan_id": card.get("plan_id", ""),
            "plan_target_object_id": card.get("plan_target_object_id", ""),
            "plan_target_object_label": card.get("plan_target_object_label", ""),
            "created_at": utc_now(),
            "judgement": card["judgement"],
        }

    def _finalize_excluded_content(self, item: dict) -> dict:
        return {
            "id": new_id("excluded"),
            "label": item["label"],
            "exclusion_type": item["exclusion_type"],
            "reason": item["reason"],
            "section_ids": item["section_ids"],
            "created_at": utc_now(),
        }

    def _resolve_claim_plan_topic_entry(self, claim_plan: dict[str, Any], topic_name: str) -> dict[str, Any]:
        search_topics = claim_plan.get("search_topics", [])
        for entry in search_topics if isinstance(search_topics, list) else []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("topic_name", "")).strip().lower() == str(topic_name or "").strip().lower():
                normalized = dict(entry)
                normalized.setdefault("dimension_key", str(entry.get("dimension_key", "") or slugify(topic_name)).strip())
                normalized.setdefault("dimension_label", str(entry.get("dimension_label", "") or topic_name).strip())
                normalized.setdefault("query_anchor", str(entry.get("query_anchor", "") or topic_name).strip())
                normalized.setdefault(
                    "outcome_terms",
                    [str(item).strip() for item in claim_plan.get("outcomes", []) if str(item).strip()],
                )
                return normalized
        return {
            "topic_name": topic_name,
            "dimension_key": slugify(topic_name),
            "dimension_label": topic_name,
            "query_anchor": topic_name,
            "outcome_terms": [str(item).strip() for item in claim_plan.get("outcomes", []) if str(item).strip()],
        }

    def _finalize_matrix_item(self, item: dict, topic_entry: dict[str, Any], claim_plan: dict[str, Any]) -> dict[str, Any]:
        evidence = item.get("evidence", [])
        primary_section_ids = [
            str(entry.get("section_id", "")).strip()
            for entry in evidence
            if str(entry.get("section_id", "")).strip()
        ]
        citation_text = str(item.get("citation_text", "")).strip()
        if not citation_text:
            citation_text = f"{topic_entry.get('dimension_label', '')} | {topic_entry.get('query_anchor', '')}".strip(" |")
        return {
            "id": new_id("matrix"),
            "dimension_key": str(item.get("dimension_key", "") or topic_entry.get("dimension_key", "")).strip(),
            "dimension_label": str(item.get("dimension_label", "") or topic_entry.get("dimension_label", "")).strip(),
            "outcome_key": str(item.get("outcome_key", "") or slugify(item.get("outcome_label", "")) or slugify((topic_entry.get("outcome_terms") or ["outcome"])[0])).strip(),
            "outcome_label": str(item.get("outcome_label", "") or ((topic_entry.get("outcome_terms") or ["Outcome"])[0])).strip(),
            "claim_text": str(item.get("claim_text", "") or claim_plan.get("claim", "")).strip(),
            "verdict": str(item.get("verdict", "")).strip() or "context_only",
            "evidence_strength": str(item.get("evidence_strength", "")).strip() or "medium",
            "summary": str(item.get("summary", "")).strip(),
            "limitation_text": str(item.get("limitation_text", "")).strip(),
            "citation_text": citation_text,
            "evidence": evidence,
            "figure_ids": [str(figure_id).strip() for figure_id in item.get("figure_ids", []) if str(figure_id).strip()],
            "supporting_section_ids": list(dict.fromkeys(primary_section_ids)),
            "created_at": utc_now(),
        }

    def validate_single_paper_flow(self, *, paper: dict, topic: dict, run_id: str) -> dict[str, Any]:
        sections = self.repository.get_sections(paper["id"])
        if not sections:
            raise ValueError("Paper has no parsed sections; parse must succeed before single-paper validation.")
        figures = self.repository.get_figures(paper["id"])
        llm_trace_events: list[dict[str, Any]] = []
        if hasattr(self.card_engine, "set_trace_sink"):
            self.card_engine.set_trace_sink(lambda event: llm_trace_events.append(event))
        understanding = self._build_paper_understanding(
            sections=sections,
            figures=figures,
            topic_name=topic["name"],
            paper_title=paper["title"],
        )
        understanding_record = self.repository.create_paper_understanding_record(
            paper_id=paper["id"],
            topic_id=topic["id"],
            run_id=run_id,
            version=understanding.get("version", "understanding-v1"),
            understanding=understanding,
        )
        card_plan = self._build_card_plan(
            understanding=understanding,
            topic_name=topic["name"],
        )
        card_plan_record = self.repository.create_card_plan(
            paper_id=paper["id"],
            topic_id=topic["id"],
            run_id=run_id,
            version=card_plan.get("version", "card-plan-v1"),
            plan=card_plan,
        )
        try:
            generation_output = self._build_cards_with_llm(
                sections,
                topic,
                paper,
                run_id=run_id,
                precomputed_understanding=understanding,
                precomputed_card_plan=card_plan,
                persist_records=False,
            )
        finally:
            if hasattr(self.card_engine, "set_trace_sink"):
                self.card_engine.set_trace_sink(None)

        validation_dir = self.settings.data_dir / "validation" / f"{run_id}_{paper['id']}_{topic['id']}"
        validation_dir.mkdir(parents=True, exist_ok=True)
        understanding_path = validation_dir / "paper_understanding.json"
        card_plan_path = validation_dir / "card_plan.json"
        cards_path = validation_dir / "final_cards.json"
        excluded_path = validation_dir / "excluded_content.json"
        llm_trace_path = validation_dir / "llm_step_traces.json"
        report_path = validation_dir / "single_paper_validation_report.md"
        understanding_path.write_text(json.dumps(understanding, ensure_ascii=False, indent=2), encoding="utf-8")
        card_plan_path.write_text(json.dumps(card_plan, ensure_ascii=False, indent=2), encoding="utf-8")
        cards_path.write_text(json.dumps(generation_output["cards"], ensure_ascii=False, indent=2), encoding="utf-8")
        excluded_path.write_text(json.dumps(generation_output["excluded_content"], ensure_ascii=False, indent=2), encoding="utf-8")
        llm_trace_path.write_text(json.dumps(llm_trace_events, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(
            self._build_single_paper_validation_report(
                paper=paper,
                topic=topic,
                run_id=run_id,
                understanding=understanding,
                card_plan=card_plan,
                generation_output=generation_output,
                understanding_record=understanding_record,
                card_plan_record=card_plan_record,
            ),
            encoding="utf-8",
        )
        return {
            "paper_id": paper["id"],
            "topic_id": topic["id"],
            "run_id": run_id,
            "understanding_record_id": understanding_record["id"],
            "card_plan_id": card_plan_record["id"],
            "card_count": len(generation_output["cards"]),
            "excluded_count": len(generation_output["excluded_content"]),
            "artifacts": {
                "paper_understanding": str(understanding_path),
                "card_plan": str(card_plan_path),
                "final_cards": str(cards_path),
                "excluded_content": str(excluded_path),
                "llm_step_traces": str(llm_trace_path),
                "report": str(report_path),
            },
        }

    def _build_single_paper_validation_report(
        self,
        *,
        paper: dict,
        topic: dict,
        run_id: str,
        understanding: dict[str, Any],
        card_plan: dict[str, Any],
        generation_output: dict[str, Any],
        understanding_record: dict,
        card_plan_record: dict,
    ) -> str:
        lines = [
            "# Single Paper Validation Report",
            "",
            f"- run_id: `{run_id}`",
            f"- topic: `{topic['name']}`",
            f"- paper: `{paper['title']}`",
            f"- understanding_record_id: `{understanding_record['id']}`",
            f"- card_plan_id: `{card_plan_record['id']}`",
            "",
            "## Contribution Objects",
        ]
        objects = understanding.get("global_contribution_objects", [])
        if not objects:
            lines.append("- (none)")
        for item in objects:
            lines.append(
                f"- `{item.get('id', '')}` | {item.get('level_hint', '')} | {item.get('object_type', '')} | {item.get('label', '')}"
            )
            lines.append(f"  - evidence_sections: {item.get('evidence_section_ids', [])}")
            lines.append(f"  - evidence_figures: {item.get('evidence_figure_ids', [])}")
            if item.get("evidence_figure_ids"):
                lines.append("  - figure_note: object has figure-backed evidence")

        lines.append("")
        lines.append("## Card Plan")
        planned_cards = card_plan.get("planned_cards", [])
        if not planned_cards:
            lines.append("- (none)")
        for item in planned_cards:
            lines.append(
                f"- `{item.get('plan_id', '')}` | level={item.get('level', '')} | disposition={item.get('disposition', '')} | object={item.get('target_object_id', '')}"
            )
            lines.append(f"  - must_have_evidence_ids: {item.get('must_have_evidence_ids', [])}")
            lines.append(f"  - optional_supporting_ids: {item.get('optional_supporting_ids', [])}")
            lines.append(f"  - must_have_figure_ids: {item.get('must_have_figure_ids', [])}")
            lines.append(f"  - optional_supporting_figure_ids: {item.get('optional_supporting_figure_ids', [])}")
            if item.get("disposition_reason"):
                lines.append(f"  - reason: {item.get('disposition_reason', '')}")

        lines.append("")
        lines.append("## Final Cards")
        cards = generation_output.get("cards", [])
        if not cards:
            lines.append("- (no cards)")
        for item in cards:
            lines.append(f"- {item.get('title', '')}")
            lines.append(f"  - level: {item.get('planned_level', '')}")
            lines.append(f"  - course_transformation: {item.get('course_transformation', '')}")
            lines.append(f"  - paper_specific_object: {item.get('paper_specific_object', '')}")
            lines.append(f"  - primary_section_ids: {item.get('primary_section_ids', [])}")
            lines.append(f"  - figure_ids: {item.get('figure_ids', [])}")
            lines.append(f"  - judgement: {(item.get('judgement') or {}).get('color', '')} | {(item.get('judgement') or {}).get('reason', '')}")

        lines.append("")
        lines.append("## Excluded Content")
        excluded = generation_output.get("excluded_content", [])
        if not excluded:
            lines.append("- (none)")
        for item in excluded:
            lines.append(
                f"- {item.get('label', '')} | type={item.get('exclusion_type', '')} | reason={item.get('reason', '')}"
            )

        return "\n".join(lines) + "\n"


class ReviewService:
    def __init__(self, settings: Settings, repository: Repository, card_engine: Optional[LLMCardEngine] = None):
        self.settings = settings
        self.repository = repository
        self.card_engine = card_engine or LLMCardEngine(settings)
        self.pipeline = PaperPipeline(settings, repository, self.card_engine)

    def review_item(self, target_type: str, target_id: str, reviewer: str, decision: str, note: str) -> dict:
        item = self.repository.get_review_item(target_type, target_id)
        if not item:
            raise LookupError("Review item not found")
        allowed_decisions = allowed_review_decisions(target_type)
        if decision not in allowed_decisions:
            choices = ", ".join(sorted(allowed_decisions)) or "(none)"
            raise ValueError(f"Unsupported decision for {target_type}: {decision}. Allowed values: {choices}")
        self.repository.create_review_decision(target_type, target_id, reviewer, decision, note)
        return self.repository.get_review_item(target_type, target_id) or item

    def save_comment(self, target_type: str, target_id: str, reviewer: str, comment: str) -> dict:
        item = self.repository.get_review_item(target_type, target_id)
        if not item:
            raise LookupError("Review item not found")
        self.repository.upsert_review_item_comment(target_type, target_id, reviewer, comment)
        return self.repository.get_review_item(target_type, target_id) or item

    def promote_excluded_item(self, excluded_content_id: str, reviewer: str, note: str) -> dict:
        excluded_item = self.repository.get_excluded_content(excluded_content_id)
        if not excluded_item:
            raise LookupError("Excluded content not found")
        existing_card = self.repository.get_promoted_card_summary(excluded_content_id)
        if existing_card:
            raise ValueError(f"Excluded item has already been promoted to {existing_card['id']}")
        if not self.card_engine.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")

        paper = self.repository.get_paper(excluded_item["paper_id"])
        topic = self.repository.get_topic(excluded_item["topic_id"])
        if not paper or not topic:
            raise LookupError("Source paper or topic for excluded content was not found")

        section_id_set = set(excluded_item["section_ids"])
        source_sections = [
            section
            for section in self.repository.get_sections(excluded_item["paper_id"])
            if section["id"] in section_id_set
        ]
        if not source_sections:
            raise ValueError("Excluded item does not have any source sections to reconsider")

        generation_output = self.pipeline.generate_outputs_for_sections(source_sections, topic, paper)
        promoted_cards = generation_output["cards"]
        if len(promoted_cards) != 1:
            raise LLMGenerationError(
                f"Excluded reconsideration must yield exactly one candidate card, got {len(promoted_cards)}"
            )

        promoted_card = self.repository.create_promoted_candidate_card(excluded_content_id, promoted_cards[0])
        promote_note = note.strip()
        if promote_note:
            promote_note = f"{promote_note}\nPromoted to candidate review as {promoted_card['id']}."
        else:
            promote_note = f"Promoted to candidate review as {promoted_card['id']}."
        self.repository.create_review_decision("excluded", excluded_content_id, reviewer, "reopened", promote_note)
        return {
            "excluded_item": self.repository.get_review_item("excluded", excluded_content_id) or excluded_item,
            "card": self.repository.get_review_item("card", promoted_card["id"]) or promoted_card,
        }


class PreferenceMemoryStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.active_path = self.settings.preference_memory_dir / "active_memory.json"

    def get_active_memory(self) -> Optional[dict[str, Any]]:
        if not self.active_path.exists():
            return None
        try:
            payload = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def activate_memory(self, memory_draft: dict[str, Any], reviewer: str) -> dict[str, Any]:
        activated = {
            "id": new_id("memory"),
            "scope": str(memory_draft.get("scope", "project")).strip() or "project",
            "mode": str(memory_draft.get("mode", "")).strip(),
            "summary": str(memory_draft.get("summary", "")).strip(),
            "prefer": [str(item).strip() for item in memory_draft.get("prefer", []) if str(item).strip()],
            "avoid": [str(item).strip() for item in memory_draft.get("avoid", []) if str(item).strip()],
            "review_signals": [str(item).strip() for item in memory_draft.get("review_signals", []) if str(item).strip()],
            "source_run_id": str(memory_draft.get("source_run_id", "")).strip(),
            "source_task_type": normalize_task_type(memory_draft.get("source_task_type", ""), default="aha_exploration"),
            "status": "active",
            "reviewer": reviewer,
            "created_at": utc_now(),
            "activated_at": utc_now(),
        }
        history_path = self.settings.preference_memory_dir / f"{activated['id']}.json"
        history_path.write_text(json.dumps(activated, ensure_ascii=False, indent=2), encoding="utf-8")
        self.active_path.write_text(json.dumps(activated, ensure_ascii=False, indent=2), encoding="utf-8")
        return activated


class ResearchPlanningService:
    def __init__(self, settings: Settings, llm_engine: Optional[LLMCardEngine] = None, memory_store: Optional[PreferenceMemoryStore] = None):
        self.settings = settings
        self.llm_engine = llm_engine or LLMCardEngine(settings)
        self.memory_store = memory_store or PreferenceMemoryStore(settings)

    def draft_plan(
        self,
        research_brief: str,
        *,
        requested_task_type: str = "auto",
        max_terms: int = 6,
        use_active_memory: bool = True,
        also_generate_aha_cards: bool = False,
    ) -> dict[str, Any]:
        normalized_brief = str(research_brief or "").strip()
        if not normalized_brief:
            raise ValueError("research_brief is required")
        requested = str(requested_task_type or "auto").strip().lower()
        active_memory = self.memory_store.get_active_memory() if use_active_memory else None
        if self.llm_engine.is_enabled() and hasattr(self.llm_engine, "draft_research_plan"):
            raw_plan = self.llm_engine.draft_research_plan(
                normalized_brief,
                requested_task_type=requested,
                max_terms=max_terms,
                active_memory=active_memory,
                also_generate_aha_cards=also_generate_aha_cards,
            )
        else:
            raw_plan = self._fallback_draft_plan(
                normalized_brief,
                requested_task_type=requested,
                max_terms=max_terms,
                active_memory=active_memory,
                also_generate_aha_cards=also_generate_aha_cards,
            )
        return self._normalize_draft_plan(
            raw_plan,
            research_brief=normalized_brief,
            requested_task_type=requested,
            max_terms=max_terms,
            active_memory=active_memory,
            also_generate_aha_cards=also_generate_aha_cards,
        )

    def _infer_task_type(self, research_brief: str) -> str:
        lowered = str(research_brief or "").lower()
        claim_markers = [
            "prove",
            "proof",
            "support",
            "evidence",
            "matrix",
            "claim",
            "dimension",
            "命题",
            "证明",
            "支持",
            "证据",
            "维度",
            "反证",
            "领导力",
            "绩效",
        ]
        if any(marker in lowered for marker in claim_markers):
            return "claim_evidence"
        return "aha_exploration"

    def _fallback_topics(self, research_brief: str, max_terms: int) -> list[str]:
        candidates: list[str] = []
        for raw in re.split(r"[\n,;，。！？/]+", research_brief):
            cleaned = str(raw or "").strip().strip("-*")
            cleaned = re.sub(r"\s+", " ", cleaned)
            if not cleaned:
                continue
            word_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", cleaned))
            if word_count > 8:
                continue
            candidates.append(cleaned)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(item)
            if len(deduped) >= max_terms:
                break
        if deduped:
            return deduped
        fallback_tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", research_brief)
        return fallback_tokens[:max_terms] or ["research topic"]

    def _fallback_draft_plan(
        self,
        research_brief: str,
        *,
        requested_task_type: str,
        max_terms: int,
        active_memory: Optional[dict[str, Any]],
        also_generate_aha_cards: bool,
    ) -> dict[str, Any]:
        task_type = normalize_task_type(
            requested_task_type if requested_task_type != "auto" else self._infer_task_type(research_brief)
        )
        topics = self._fallback_topics(research_brief, max_terms)
        if task_type == "claim_evidence":
            dimension_entries = []
            for index, topic_name in enumerate(topics[:max_terms], start=1):
                dimension_entries.append(
                    {
                        "topic_name": topic_name,
                        "dimension_key": slugify(topic_name),
                        "dimension_label": topic_name,
                        "query_anchor": topic_name,
                        "outcome_terms": ["leadership effectiveness", "team performance"],
                    }
                )
            return {
                "task_type": task_type,
                "suggested_task_type": task_type,
                "claim": research_brief,
                "search_topics": dimension_entries,
                "outcomes": ["leadership effectiveness", "team performance"],
                "evidence_policy": {
                    "surface_contradictions": True,
                    "minimum_supporting_papers_per_dimension": 3,
                },
                "also_generate_aha_cards": also_generate_aha_cards,
                "summary": "Fallback claim-evidence plan generated without an LLM.",
                "active_memory_snapshot": active_memory or {},
            }
        return {
            "task_type": task_type,
            "suggested_task_type": task_type,
            "recommended_topics": [{"topic_name": topic_name, "query_anchor": topic_name} for topic_name in topics],
            "summary": "Fallback aha plan generated without an LLM.",
            "active_memory_snapshot": active_memory or {},
        }

    def _normalize_draft_plan(
        self,
        raw_plan: dict[str, Any],
        *,
        research_brief: str,
        requested_task_type: str,
        max_terms: int,
        active_memory: Optional[dict[str, Any]],
        also_generate_aha_cards: bool,
    ) -> dict[str, Any]:
        task_type = normalize_task_type(
            raw_plan.get("task_type") or raw_plan.get("suggested_task_type") or (
                requested_task_type if requested_task_type != "auto" else self._infer_task_type(research_brief)
            )
        )
        if task_type == "claim_evidence":
            normalized_topics = []
            seen: set[str] = set()
            for entry in raw_plan.get("search_topics", []) if isinstance(raw_plan.get("search_topics", []), list) else []:
                if not isinstance(entry, dict):
                    entry = {"topic_name": str(entry or "").strip(), "query_anchor": str(entry or "").strip()}
                topic_name = str(entry.get("topic_name", "") or entry.get("dimension_label", "") or entry.get("query_anchor", "")).strip()
                if not topic_name:
                    continue
                lowered = topic_name.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                query_anchor = str(entry.get("query_anchor", "") or topic_name).strip()
                outcome_terms = [str(item).strip() for item in entry.get("outcome_terms", []) if str(item).strip()]
                if not outcome_terms:
                    outcome_terms = [str(item).strip() for item in raw_plan.get("outcomes", []) if str(item).strip()]
                normalized_topics.append(
                    {
                        "topic_name": topic_name,
                        "dimension_key": str(entry.get("dimension_key", "") or slugify(topic_name)).strip(),
                        "dimension_label": str(entry.get("dimension_label", "") or topic_name).strip(),
                        "query_anchor": query_anchor,
                        "outcome_terms": outcome_terms[:4],
                    }
                )
                if len(normalized_topics) >= max_terms:
                    break
            if not normalized_topics:
                normalized_topics = self._fallback_draft_plan(
                    research_brief,
                    requested_task_type="claim_evidence",
                    max_terms=max_terms,
                    active_memory=active_memory,
                    also_generate_aha_cards=also_generate_aha_cards,
                )["search_topics"]
            return {
                "task_type": "claim_evidence",
                "suggested_task_type": "claim_evidence",
                "research_brief": research_brief,
                "claim": str(raw_plan.get("claim", "") or research_brief).strip(),
                "search_topics": normalized_topics,
                "outcomes": [str(item).strip() for item in raw_plan.get("outcomes", []) if str(item).strip()],
                "evidence_policy": {
                    "surface_contradictions": bool((raw_plan.get("evidence_policy", {}) or {}).get("surface_contradictions", True)),
                    "minimum_supporting_papers_per_dimension": int((raw_plan.get("evidence_policy", {}) or {}).get("minimum_supporting_papers_per_dimension", 3) or 3),
                },
                "also_generate_aha_cards": bool(raw_plan.get("also_generate_aha_cards", also_generate_aha_cards)),
                "confirmed_topics_text": "\n".join(item["topic_name"] for item in normalized_topics),
                "summary": str(raw_plan.get("summary", "")).strip(),
                "active_memory_snapshot": active_memory or {},
            }
        normalized_topics = []
        seen_topics: set[str] = set()
        raw_topics = raw_plan.get("recommended_topics", [])
        if isinstance(raw_topics, list):
            for entry in raw_topics:
                if isinstance(entry, dict):
                    topic_name = str(entry.get("topic_name", "") or entry.get("topic", "")).strip()
                else:
                    topic_name = str(entry or "").strip()
                if not topic_name:
                    continue
                lowered = topic_name.lower()
                if lowered in seen_topics:
                    continue
                seen_topics.add(lowered)
                normalized_topics.append({"topic_name": topic_name, "query_anchor": topic_name})
                if len(normalized_topics) >= max_terms:
                    break
        if not normalized_topics:
            normalized_topics = [{"topic_name": item, "query_anchor": item} for item in self._fallback_topics(research_brief, max_terms)]
        return {
            "task_type": "aha_exploration",
            "suggested_task_type": "aha_exploration",
            "research_brief": research_brief,
            "recommended_topics": normalized_topics,
            "confirmed_topics_text": "\n".join(item["topic_name"] for item in normalized_topics),
            "summary": str(raw_plan.get("summary", "")).strip(),
            "active_memory_snapshot": active_memory or {},
        }


class PreferenceMemoryService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        llm_engine: Optional[LLMCardEngine] = None,
        memory_store: Optional[PreferenceMemoryStore] = None,
    ):
        self.settings = settings
        self.repository = repository
        self.llm_engine = llm_engine or LLMCardEngine(settings)
        self.memory_store = memory_store or PreferenceMemoryStore(settings)

    def draft_memory(self, *, task_type: str = "", run_id: str = "", reviewer: str = "internal") -> dict[str, Any]:
        normalized_task_type = normalize_task_type(task_type, default="")
        review_items = self.repository.list_review_items(
            run_id=run_id or None,
            item_type="all",
            review_status="",
        )
        latest_items = [item for item in review_items if (not normalized_task_type or self._item_matches_task_type(item, normalized_task_type))]
        active_memory = self.memory_store.get_active_memory()
        if self.llm_engine.is_enabled() and hasattr(self.llm_engine, "distill_preference_memory"):
            raw_memory = self.llm_engine.distill_preference_memory(
                latest_items,
                task_type=normalized_task_type or "",
                active_memory=active_memory,
            )
        else:
            raw_memory = self._fallback_memory(latest_items, normalized_task_type)
        return {
            "scope": str(raw_memory.get("scope", "project")).strip() or "project",
            "mode": str(raw_memory.get("mode", normalized_task_type or "mixed")).strip(),
            "summary": str(raw_memory.get("summary", "")).strip(),
            "prefer": [str(item).strip() for item in raw_memory.get("prefer", []) if str(item).strip()],
            "avoid": [str(item).strip() for item in raw_memory.get("avoid", []) if str(item).strip()],
            "review_signals": [str(item).strip() for item in raw_memory.get("review_signals", []) if str(item).strip()],
            "source_run_id": run_id,
            "source_task_type": normalized_task_type or "mixed",
            "reviewer": reviewer,
            "active_memory_snapshot": active_memory or {},
            "signal_count": len(latest_items),
        }

    def activate_memory(self, memory_draft: dict[str, Any], reviewer: str) -> dict[str, Any]:
        if not isinstance(memory_draft, dict) or not memory_draft:
            raise ValueError("memory_draft is required")
        return self.memory_store.activate_memory(memory_draft, reviewer)

    def _item_matches_task_type(self, item: dict[str, Any], task_type: str) -> bool:
        if not task_type:
            return True
        object_type = str(item.get("object_type", "")).strip()
        if task_type == "claim_evidence":
            return object_type == "matrix_item"
        return object_type in {"card", "excluded"}

    def _fallback_memory(self, items: list[dict], task_type: str) -> dict[str, Any]:
        accepted = [item for item in items if item.get("review_status") == "accepted"]
        rejected = [item for item in items if item.get("review_status") == "rejected"]
        manual = [item for item in items if item.get("review_status") == "needs_manual_check"]
        context_only_rejects = [
            item for item in rejected
            if item.get("object_type") == "matrix_item" and str(item.get("verdict", "")).strip() == "context_only"
        ]
        mixed_accepts = [
            item for item in accepted
            if item.get("object_type") == "matrix_item" and str(item.get("verdict", "")).strip() == "mixed"
        ]
        prefer = []
        avoid = []
        signals = [
            f"accepted={len(accepted)}",
            f"rejected={len(rejected)}",
            f"needs_manual_check={len(manual)}",
        ]
        if context_only_rejects:
            avoid.append("Avoid context-only evidence items unless they establish a strong direct outcome link.")
            signals.append(f"context_only_rejects={len(context_only_rejects)}")
        if mixed_accepts:
            prefer.append("Prefer mixed-evidence items when limitations are explicit and evidence remains decision-useful.")
            signals.append(f"mixed_accepts={len(mixed_accepts)}")
        if not prefer:
            prefer.append("Prefer direct empirical evidence with short transfer distance to the reporting claim.")
        if not avoid:
            avoid.append("Avoid weakly grounded summaries, taxonomy recaps, and items whose course or reporting use is still vague.")
        return {
            "scope": "project",
            "mode": task_type or "mixed",
            "summary": "Fallback preference memory distilled from explicit review decisions.",
            "prefer": prefer,
            "avoid": avoid,
            "review_signals": signals,
        }


class PaperQANotReadyError(ValueError):
    def __init__(self, capability: dict[str, Any]):
        self.capability = capability
        super().__init__(capability.get("qa_message", "Paper QA is not ready"))

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": "paper_qa_not_ready",
            "status": self.capability.get("qa_status", ""),
            "qa_status": self.capability.get("qa_status", ""),
            "message": self.capability.get("qa_message", ""),
            "qa_message": self.capability.get("qa_message", ""),
            "paper_content_basis": self.capability.get("paper_content_basis", ""),
            "access_status": self.capability.get("access_status", ""),
            "parse_status": self.capability.get("parse_status", ""),
            "section_count": int(self.capability.get("section_count", 0) or 0),
            "has_abstract_backed_matrix_items": bool(self.capability.get("has_abstract_backed_matrix_items", False)),
            "paper_id": self.capability.get("paper_id", ""),
            "paper_title": self.capability.get("paper_title", ""),
        }


class PaperQAService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        llm_engine: Optional[LLMCardEngine] = None,
        memory_store: Optional[PreferenceMemoryStore] = None,
    ):
        self.settings = settings
        self.repository = repository
        self.llm_engine = llm_engine or LLMCardEngine(settings)
        self.memory_store = memory_store or PreferenceMemoryStore(settings)

    def answer_question(self, paper_id: str, question: str, *, max_sections: int = 6) -> dict[str, Any]:
        paper = self.repository.get_paper(paper_id)
        if not paper:
            raise LookupError("Paper not found")
        capability = self.repository.get_paper_qa_capability(paper_id)
        if not capability:
            raise LookupError("Paper not found")
        if not capability["qa_available"]:
            raise PaperQANotReadyError(capability)
        normalized_question = str(question or "").strip()
        if not normalized_question:
            raise ValueError("question is required")
        sections = self.repository.get_sections(paper_id)
        if not sections:
            raise PaperQANotReadyError(self.repository.get_paper_qa_capability(paper_id) or capability)
        if not self.llm_engine.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")
        scored_sections = self._rank_sections(normalized_question, sections, limit=max_sections)
        selected_ids = {item["id"] for item in scored_sections}
        figures = [
            figure
            for figure in self.repository.get_figures(paper_id)
            if set(figure.get("linked_section_ids", [])).intersection(selected_ids)
        ][:4]
        answer = self.llm_engine.answer_paper_question(
            paper_title=capability.get("paper_title", paper["title"]),
            question=normalized_question,
            sections=scored_sections,
            figures=figures,
            active_memory=self.memory_store.get_active_memory(),
        )
        answer["paper_id"] = paper_id
        answer["paper_title"] = capability.get("paper_title", paper["title"])
        answer["question"] = normalized_question
        return answer

    def _rank_sections(self, question: str, sections: list[dict], *, limit: int) -> list[dict]:
        question_embedding = embedding_for_text(question)
        question_tokens = {
            token
            for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", question.lower())
            if len(token) >= 2
        }
        scored = []
        for section in sections:
            section_embedding = section.get("embedding") or embedding_for_text(section.get("paragraph_text", ""))
            cosine = cosine_similarity(question_embedding, section_embedding)
            text = f"{section.get('section_title', '')} {section.get('paragraph_text', '')}".lower()
            lexical_overlap = sum(1 for token in question_tokens if token in text)
            score = cosine + (0.12 * lexical_overlap) + (0.08 if section.get("is_body") else 0.0)
            scored.append((score, section))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen: set[str] = set()
        for _, section in scored:
            if section["id"] in seen:
                continue
            seen.add(section["id"])
            selected.append(section)
            if len(selected) >= limit:
                break
        return selected


class EvaluationService:
    def __init__(self, settings: Settings, repository: Repository, card_engine: Optional[LLMCardEngine] = None):
        self.settings = settings
        self.repository = repository
        self.card_engine = card_engine or LLMCardEngine(settings)

    def run_calibration_set(self, calibration_set_id: str) -> dict:
        calibration_set = self.repository.get_calibration_set(calibration_set_id)
        if not calibration_set:
            raise ValueError("Calibration set not found.")
        if not self.card_engine.is_enabled():
            raise LLMGenerationError("LLM provider is not enabled")

        evaluation_run = self.repository.create_evaluation_run(
            calibration_set=calibration_set,
            llm_mode=self.settings.llm_mode,
            model_name=self.card_engine.client.model if self.card_engine.client else "",
            extraction_prompt_version=EXTRACTION_PROMPT_VERSION,
            judgement_prompt_version=JUDGEMENT_PROMPT_VERSION,
            rubric_version=CARD_RUBRIC_VERSION,
        )
        results = []
        for example in calibration_set["examples"]:
            results.append(self._evaluate_example(evaluation_run["id"], calibration_set, example))

        summary = self._build_evaluation_summary(calibration_set, results)
        status = "completed" if summary["failed_examples"] == 0 else "completed_with_regressions"
        return self.repository.finalize_evaluation_run(evaluation_run["id"], status, summary) or evaluation_run

    def _evaluate_example(self, evaluation_run_id: str, calibration_set: dict, example: dict) -> dict:
        section_id = f"eval_section_{example['id']}"
        sections = [
            {
                "id": section_id,
                "section_title": "Calibration Example",
                "page_number": 1,
                "paragraph_text": example["source_text"],
            }
        ]
        extraction_output = self.card_engine.extract_candidates(
            topic_name=example["topic_name"],
            paper_title=example["title"],
            sections=sections,
            figures=[],
            calibration_examples=calibration_set["examples"],
            calibration_set_name=calibration_set["name"],
        )
        judgement_output = self.card_engine.judge_candidates(
            topic_name=example["topic_name"],
            paper_title=example["title"],
            extracted_cards=extraction_output["cards"],
            figures=[],
            calibration_examples=calibration_set["examples"],
            calibration_set_name=calibration_set["name"],
        )
        expected = {
            "expected_card_count": len(example["expected_cards"]),
            "expected_exclusion_count": len(example["expected_exclusions"]),
            "expected_cards": example["expected_cards"],
            "expected_exclusions": example["expected_exclusions"],
        }
        actual = {
            "judged_card_count": len(judgement_output["cards"]),
            "excluded_count": len(extraction_output["excluded_content"]),
            "card_titles": [card["title"] for card in judgement_output["cards"]],
            "card_colors": [card["judgement"]["color"] for card in judgement_output["cards"]],
            "course_transformations": [card.get("course_transformation", "") for card in judgement_output["cards"]],
            "paper_specific_objects": [card.get("paper_specific_object", "") for card in judgement_output["cards"]],
            "figure_attachment_count": sum(len(card.get("figure_ids", [])) for card in judgement_output["cards"]),
            "excluded_labels": [item["label"] for item in extraction_output["excluded_content"]],
            "excluded_types": [item["exclusion_type"] for item in extraction_output["excluded_content"]],
        }
        verdict, regression_type, reason = self._compare_expected_and_actual(example, actual)
        return self.repository.create_evaluation_result(
            evaluation_run_id=evaluation_run_id,
            calibration_example=example,
            extraction_output=extraction_output,
            judgement_output=judgement_output,
            expected=expected,
            actual=actual,
            verdict=verdict,
            regression_type=regression_type,
            reason=reason,
        )

    def _compare_expected_and_actual(self, example: dict, actual: dict) -> tuple[str, str, str]:
        example_type = example["example_type"]
        judged_card_count = actual["judged_card_count"]
        excluded_count = actual["excluded_count"]
        colors = actual["card_colors"]
        expected_card_count = len(example["expected_cards"])
        expected_exclusion_count = len(example["expected_exclusions"])
        tags = {str(tag).strip().lower() for tag in example.get("tags", []) if str(tag).strip()}

        if judged_card_count > 0:
            course_transformations = actual.get("course_transformations", [])
            paper_specific_objects = actual.get("paper_specific_objects", [])
            has_fidelity = any(
                has_source_object_fidelity_signal(course, title, paper_object)
                for course, title, paper_object in zip(
                    course_transformations,
                    actual.get("card_titles", []),
                    paper_specific_objects,
                )
            )
            if "principle-drift-negative" in tags and judged_card_count > 0:
                return ("failed", "principle_drift", "Negative example incorrectly produced a card despite principle-drift guardrails.")
            if "audience-mismatch" in tags and judged_card_count > 0:
                return ("failed", "audience_mismatch", "Negative example incorrectly produced a card despite audience-fit guardrails.")
            if "visual-evidence-required" in tags and int(actual.get("figure_attachment_count", 0)) <= 0:
                return ("failed", "missing_figure_support", "Card passed without attaching required figure evidence.")
            if not has_fidelity and ("direct-transfer" in tags or "nontechnical-audience" in tags):
                return ("failed", "principle_drift", "Card drifted away from the source object for a direct-transfer / non-technical example.")

        if example_type == "positive":
            if judged_card_count <= 0:
                return ("failed", "missed_aha", "Positive example failed to produce any judged card.")
            if "red" in colors and all(color == "red" for color in colors):
                return ("failed", "missed_aha", "Positive example was judged as red instead of a usable aha candidate.")
            return ("passed", "none", "Positive example produced a judged candidate card.")

        if example_type == "negative":
            if judged_card_count > 0:
                if "weak_transfer" in example["tags"] or any(
                    item.get("exclusion_type") == "weak_transfer" for item in example["expected_exclusions"]
                ):
                    return ("failed", "weak_transfer_drift", "Negative example incorrectly produced a card despite weak-transfer guardrails.")
                return ("failed", "summary_drift", "Negative example incorrectly produced a learner-facing card.")
            if expected_exclusion_count > 0 and excluded_count <= 0:
                return ("failed", "summary_drift", "Negative example produced no card but also failed to surface the expected excluded content.")
            return ("passed", "none", "Negative example stayed out of the card set.")

        if judged_card_count > 0 and "yellow" in colors:
            return ("passed", "none", "Boundary example remained in the yellow review zone.")
        if judged_card_count == 0 and expected_card_count == 0:
            return ("passed", "none", "Boundary example produced no card and did not violate expectations.")
        return ("failed", "boundary_mismatch", "Boundary example did not remain in a yellow-zone or equivalent borderline outcome.")

    def _build_evaluation_summary(self, calibration_set: dict, results: list[dict]) -> dict[str, Any]:
        summary = {
            "calibration_set_id": calibration_set["id"],
            "calibration_set_name": calibration_set["name"],
            "total_examples": len(results),
            "passed_examples": 0,
            "failed_examples": 0,
            "positive_examples": 0,
            "negative_examples": 0,
            "boundary_examples": 0,
            "summary_drift_count": 0,
            "weak_transfer_drift_count": 0,
            "principle_drift_count": 0,
            "audience_mismatch_count": 0,
            "missing_figure_support_count": 0,
            "missed_aha_count": 0,
            "boundary_mismatch_count": 0,
            "abstract_only_evidence_failures": 0,
            "framing_only_card_failures": 0,
            "same_evidence_duplicate_split_failures": 0,
            "paper_specific_object_missing_failures": 0,
            "body_evidence_ignored_failures": 0,
        }
        for result in results:
            summary[f"{result['example_type']}_examples"] += 1
            if result["verdict"] == "passed":
                summary["passed_examples"] += 1
            else:
                summary["failed_examples"] += 1
            if result["regression_type"] == "summary_drift":
                summary["summary_drift_count"] += 1
            elif result["regression_type"] == "weak_transfer_drift":
                summary["weak_transfer_drift_count"] += 1
            elif result["regression_type"] == "principle_drift":
                summary["principle_drift_count"] += 1
            elif result["regression_type"] == "audience_mismatch":
                summary["audience_mismatch_count"] += 1
            elif result["regression_type"] == "missing_figure_support":
                summary["missing_figure_support_count"] += 1
            elif result["regression_type"] == "missed_aha":
                summary["missed_aha_count"] += 1
            elif result["regression_type"] == "boundary_mismatch":
                summary["boundary_mismatch_count"] += 1
            actual = result.get("actual", {}) if isinstance(result, dict) else {}
            card_titles = " ".join(actual.get("card_titles", [])) if isinstance(actual.get("card_titles", []), list) else ""
            if "abstract" in card_titles.lower():
                summary["abstract_only_evidence_failures"] += 1
            if result.get("regression_type") == "summary_drift":
                summary["framing_only_card_failures"] += 1
            if result.get("regression_type") == "principle_drift":
                summary["framing_only_card_failures"] += 1
            if result.get("regression_type") == "boundary_mismatch":
                summary["same_evidence_duplicate_split_failures"] += 1
            if result.get("regression_type") == "missed_aha":
                summary["paper_specific_object_missing_failures"] += 1
            if result.get("regression_type") in {"weak_transfer_drift", "summary_drift"}:
                summary["body_evidence_ignored_failures"] += 1
        return summary


class AccessQueueService:
    def __init__(self, settings: Settings, repository: Repository, coordinator: RunCoordinator):
        self.settings = settings
        self.repository = repository
        self.coordinator = coordinator
        self.pipeline = coordinator.pipeline

    def reactivate_item(self, queue_item_id: str, local_path: str, reviewer: str) -> dict:
        queue_item = self.repository.get_access_queue_item(queue_item_id)
        if not queue_item:
            raise LookupError("Access queue item not found")
        if queue_item["status"] != "open":
            raise ValueError(f"Access queue item {queue_item_id} is not open")
        paper = self.repository.get_paper(queue_item["paper_id"])
        if not paper:
            raise LookupError("Paper not found for access queue item")

        artifact_path = self.pipeline.ingest_local_pdf(local_path)
        self.repository.update_paper(
            paper["id"],
            local_path=local_path,
            artifact_path=artifact_path,
            access_status="open_fulltext",
            ingestion_status="artifact_ready",
            parse_status="pending",
            parse_failure_reason="",
            card_generation_status="pending",
            card_generation_failure_reason="",
        )
        self.repository.update_access_queue_item(queue_item_id, status="reactivated", owner=reviewer)

        refreshed_paper = self.repository.get_paper(paper["id"])
        if not refreshed_paper:
            raise LookupError("Paper disappeared during reactivation")
        self.pipeline.parse_and_store(refreshed_paper)
        refreshed_paper = self.repository.get_paper(paper["id"])
        topic_runs = self.repository.list_topic_runs_for_paper_run(paper["id"], queue_item["run_id"])
        task_context = self.coordinator._get_run_task_context(queue_item["run_id"])
        processed_topics = []

        for topic_run in topic_runs:
            stats = topic_run["stats"]
            self.coordinator._mark_topic_progress(topic_run["id"], stats, stage="parsing")
            if refreshed_paper and refreshed_paper["parse_status"] == "parsed":
                output_counts = self.coordinator._build_task_outputs_for_paper(
                    refreshed_paper,
                    {"id": topic_run["topic_id"], "name": topic_run["topic_name"]},
                    queue_item["run_id"],
                    task_context=task_context,
                )
                stats["parsed_papers"] = stats.get("parsed_papers", 0) + 1
                stats["card_generation_attempts"] = stats.get("card_generation_attempts", 0) + 1
                stats["cards"] = int(stats.get("cards", 0) or 0) + int(output_counts.get("cards", 0) or 0)
                stats["matrix_items"] = int(stats.get("matrix_items", 0) or 0) + int(output_counts.get("matrix_items", 0) or 0)
            else:
                self.repository.replace_cards_for_paper_topic(paper["id"], topic_run["topic_id"], queue_item["run_id"], [])
                self.repository.replace_matrix_items_for_paper_topic(paper["id"], topic_run["topic_id"], queue_item["run_id"], [])
            stats["accessible"] = sum(
                1
                for item in self.repository.list_papers_for_topic_run(queue_item["run_id"], topic_run["topic_id"])
                if item["access_status"] == "open_fulltext"
            )
            stats["parsed_papers"] = sum(
                1
                for item in self.repository.list_papers_for_topic_run(queue_item["run_id"], topic_run["topic_id"])
                if item["parse_status"] == "parsed"
            )
            stats["cards"] = len(self.repository.list_cards(run_id=queue_item["run_id"], topic=topic_run["topic_name"]))
            stats["matrix_items"] = len(self.repository.list_matrix_items(run_id=queue_item["run_id"], topic=topic_run["topic_name"]))
            stats["queued_for_access"] = self.repository.count_open_access_queue_for_topic(queue_item["run_id"], topic_run["topic_id"])
            stats = self.coordinator._build_topic_run_metrics(
                queue_item["run_id"],
                {"id": topic_run["topic_id"], "name": topic_run["topic_name"]},
                topic_run,
                stats,
            )
            if task_context["task_type"] == "aha_exploration":
                stats = self.coordinator._attach_topic_stop_decision(
                    {"id": topic_run["topic_id"], "name": topic_run["topic_name"]},
                    stats,
                )
            stats["current_stage"] = "completed"
            stats["last_progress_at"] = utc_now()
            self.repository.update_topic_run(topic_run["id"], "completed", stats=stats)
            processed_topics.append(self.repository.get_topic_run(topic_run["id"]))

        self.coordinator._refresh_run_status_from_topics(queue_item["run_id"])
        return {
            "queue_item": self.repository.get_access_queue_item(queue_item_id),
            "paper": self.repository.get_paper(paper["id"]),
            "topic_runs": [item for item in processed_topics if item],
        }

    def auto_download_item(self, queue_item_id: str, reviewer: str) -> dict:
        queue_item = self.repository.get_access_queue_item(queue_item_id)
        if not queue_item:
            raise LookupError("Access queue item not found")
        if queue_item["status"] != "open":
            raise ValueError(f"Access queue item {queue_item_id} is not open")
        paper = self.repository.get_paper(queue_item["paper_id"])
        if not paper:
            raise LookupError("Paper not found for access queue item")

        # Try best_asset_url from discovery_results first, then Unpaywall
        enriched = self.repository.list_access_queue(queue_item["run_id"])
        best_asset_url = next(
            (item.get("best_asset_url") for item in enriched if item["id"] == queue_item_id),
            None,
        ) or ""
        artifact_path = self.pipeline.acquire_remote_asset(paper, best_asset_url) if best_asset_url else None
        if not artifact_path:
            artifact_path = self.pipeline.acquire_remote_asset_with_oa_fallback(paper, "")
        if not artifact_path:
            return {"success": False, "message": "Could not download PDF automatically. Please reactivate manually."}

        self.repository.update_paper(
            paper["id"],
            artifact_path=artifact_path,
            access_status="open_fulltext",
            ingestion_status="artifact_ready",
            parse_status="pending",
            parse_failure_reason="",
            card_generation_status="pending",
            card_generation_failure_reason="",
        )
        self.repository.update_access_queue_item(queue_item_id, status="reactivated", owner=reviewer)

        refreshed_paper = self.repository.get_paper(paper["id"])
        if not refreshed_paper:
            raise LookupError("Paper disappeared during auto-download reactivation")
        self.pipeline.parse_and_store(refreshed_paper)
        refreshed_paper = self.repository.get_paper(paper["id"])
        topic_runs = self.repository.list_topic_runs_for_paper_run(paper["id"], queue_item["run_id"])
        task_context = self.coordinator._get_run_task_context(queue_item["run_id"])
        processed_topics = []

        for topic_run in topic_runs:
            stats = topic_run["stats"]
            self.coordinator._mark_topic_progress(topic_run["id"], stats, stage="parsing")
            if refreshed_paper and refreshed_paper["parse_status"] == "parsed":
                output_counts = self.coordinator._build_task_outputs_for_paper(
                    refreshed_paper,
                    {"id": topic_run["topic_id"], "name": topic_run["topic_name"]},
                    queue_item["run_id"],
                    task_context=task_context,
                )
                stats["parsed_papers"] = stats.get("parsed_papers", 0) + 1
                stats["cards"] = int(stats.get("cards", 0) or 0) + int(output_counts.get("cards", 0) or 0)
                stats["matrix_items"] = int(stats.get("matrix_items", 0) or 0) + int(output_counts.get("matrix_items", 0) or 0)
            else:
                self.repository.replace_cards_for_paper_topic(paper["id"], topic_run["topic_id"], queue_item["run_id"], [])
                self.repository.replace_matrix_items_for_paper_topic(paper["id"], topic_run["topic_id"], queue_item["run_id"], [])
            stats["accessible"] = sum(
                1 for item in self.repository.list_papers_for_topic_run(queue_item["run_id"], topic_run["topic_id"])
                if item["access_status"] == "open_fulltext"
            )
            stats["queued_for_access"] = self.repository.count_open_access_queue_for_topic(queue_item["run_id"], topic_run["topic_id"])
            stats = self.coordinator._build_topic_run_metrics(
                queue_item["run_id"],
                {"id": topic_run["topic_id"], "name": topic_run["topic_name"]},
                topic_run,
                stats,
            )
            if task_context["task_type"] == "aha_exploration":
                stats = self.coordinator._attach_topic_stop_decision(
                    {"id": topic_run["topic_id"], "name": topic_run["topic_name"]},
                    stats,
                )
            stats["current_stage"] = "completed"
            stats["last_progress_at"] = utc_now()
            self.repository.update_topic_run(topic_run["id"], "completed", stats=stats)
            processed_topics.append(self.repository.get_topic_run(topic_run["id"]))

        self.coordinator._refresh_run_status_from_topics(queue_item["run_id"])
        return {
            "success": True,
            "artifact_path": artifact_path,
            "queue_item": self.repository.get_access_queue_item(queue_item_id),
            "paper": self.repository.get_paper(paper["id"]),
            "topic_runs": [item for item in processed_topics if item],
        }


class ExportService:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings = settings
        self.repository = repository

    def export_google_doc_package(self, run_id: str, card_ids: list[str], document_title: str, existing_google_doc_id: str = "") -> dict:
        selection = self._resolve_export_selection(run_id, card_ids)
        cards = selection["cards"]
        grouped: dict[str, dict[str, list[dict]]] = {}
        for card in cards:
            grouped.setdefault(card["topic_name"], {}).setdefault(card["paper_title"], []).append(card)

        markdown_lines = [f"# {document_title}", ""]
        requests = []
        current_index = 1
        markdown_lines.append(f"Run 编号：`{run_id}`")
        markdown_lines.append("")
        for topic_name, papers in grouped.items():
            markdown_lines.append(f"## 主题：{topic_name}")
            markdown_lines.append("")
            requests.extend(self._insert_text(current_index, f"{topic_name}\n"))
            current_index += len(topic_name) + 1
            for paper_title, paper_cards in papers.items():
                markdown_lines.append(f"### {paper_title}")
                markdown_lines.append("")
                paper_url = paper_cards[0].get("paper_url", "") if paper_cards else ""
                if paper_url:
                    markdown_lines.append(f"论文链接：{paper_url}")
                    markdown_lines.append("")
                requests.extend(self._insert_text(current_index, f"{paper_title}\n"))
                current_index += len(paper_title) + 1
                if paper_url:
                    paper_link_line = f"论文链接：{paper_url}\n"
                    requests.extend(self._insert_text(current_index, paper_link_line))
                    current_index += len(paper_link_line)
                for card in paper_cards:
                    color = (card["judgement"] or {}).get("color", "yellow").upper()
                    quote_first_markdown = render_quote_first_markdown(card)
                    markdown_lines.append(f"- [{color}] **{card['title']}**")
                    markdown_lines.append(f"  - 它在课程里变成什么：{card['course_transformation']}")
                    markdown_lines.append(f"  - 可直接讲的一句话：{card.get('teachable_one_liner', '')}")
                    markdown_lines.append(f"  - 判断理由：{(card['judgement'] or {}).get('reason', '')}")
                    markdown_lines.append(f"  - {quote_first_markdown.replace(chr(10), chr(10) + '    ')}")
                    markdown_lines.append("")
                    block = (
                        f"[{color}] {card['title']}\n"
                        f"它在课程里变成什么：{card['course_transformation']}\n"
                        f"可直接讲的一句话：{card.get('teachable_one_liner', '')}\n"
                        f"判断理由：{(card['judgement'] or {}).get('reason', '')}\n"
                    )
                    block += f"{quote_first_markdown}\n"
                    block += "\n"
                    requests.extend(self._insert_text(current_index, block))
                    current_index += len(block)

        export_payload = {
            "document_title": document_title,
            "existing_google_doc_id": existing_google_doc_id,
            "requested_card_ids": selection["requested_card_ids"],
            "resolved_card_ids": [card["id"] for card in cards],
            "selection_snapshot": selection["selection_snapshot"],
            "review_snapshot": selection["review_snapshot"],
            "requests": requests,
            "card_count": len(cards),
        }

        export_id = new_id("exportdoc")
        artifact_base = self.settings.exports_dir / export_id
        markdown_path = artifact_base.with_suffix(".md")
        json_path = artifact_base.with_suffix(".json")
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        json_path.write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        google_doc_id = ""
        export_status = "artifact_only"
        export_mode = "artifact_only"
        error_message = ""
        if self.settings.google_docs_mode == "gws":
            gws_result = self._try_gws_export(document_title, existing_google_doc_id, export_payload)
            google_doc_id = gws_result["google_doc_id"]
            export_status = gws_result["export_status"]
            export_mode = gws_result["export_mode"]
            error_message = gws_result["error_message"]

        record = self.repository.create_export(
            run_id=run_id,
            destination_type="google_docs",
            export_mode=export_mode,
            google_doc_id=google_doc_id,
            export_status=export_status,
            error_message=error_message,
            artifact_path=str(markdown_path),
            request_payload=export_payload,
        )
        record["markdown_path"] = str(markdown_path)
        record["json_path"] = str(json_path)
        return record

    def export_matrix_google_doc_package(
        self,
        run_id: str,
        matrix_item_ids: list[str],
        document_title: str,
        existing_google_doc_id: str = "",
    ) -> dict:
        selection = self._resolve_matrix_export_selection(run_id, matrix_item_ids)
        matrix_items = selection["matrix_items"]
        grouped: dict[str, dict[str, list[dict]]] = {}
        for item in matrix_items:
            grouped.setdefault(item["dimension_label"], {}).setdefault(item["outcome_label"], []).append(item)

        markdown_lines = [f"# {document_title}", ""]
        requests = []
        current_index = 1
        markdown_lines.append(f"Run 编号：`{run_id}`")
        markdown_lines.append("")
        for dimension_label, outcomes in grouped.items():
            markdown_lines.append(f"## 维度：{dimension_label}")
            markdown_lines.append("")
            requests.extend(self._insert_text(current_index, f"{dimension_label}\n"))
            current_index += len(dimension_label) + 1
            for outcome_label, items in outcomes.items():
                markdown_lines.append(f"### 结果：{outcome_label}")
                markdown_lines.append("")
                requests.extend(self._insert_text(current_index, f"{outcome_label}\n"))
                current_index += len(outcome_label) + 1
                for item in items:
                    markdown_lines.append(f"- [{item['verdict']}] **{item['summary']}**")
                    markdown_lines.append(f"  - 命题：{item['claim_text']}")
                    markdown_lines.append(f"  - 证据强度：{item['evidence_strength']}")
                    markdown_lines.append(f"  - 限制：{item.get('limitation_text', '')}")
                    markdown_lines.append(f"  - 引文：{item.get('citation_text', '')}")
                    markdown_lines.append(f"  - 论文：{item.get('paper_title', '')}")
                    markdown_lines.append("")
                    block = (
                        f"[{item['verdict']}] {item['summary']}\n"
                        f"命题：{item['claim_text']}\n"
                        f"证据强度：{item['evidence_strength']}\n"
                        f"限制：{item.get('limitation_text', '')}\n"
                        f"引文：{item.get('citation_text', '')}\n"
                        f"论文：{item.get('paper_title', '')}\n\n"
                    )
                    requests.extend(self._insert_text(current_index, block))
                    current_index += len(block)

        export_payload = {
            "document_title": document_title,
            "existing_google_doc_id": existing_google_doc_id,
            "requested_matrix_item_ids": selection["requested_matrix_item_ids"],
            "resolved_matrix_item_ids": [item["id"] for item in matrix_items],
            "selection_snapshot": selection["selection_snapshot"],
            "review_snapshot": selection["review_snapshot"],
            "requests": requests,
            "matrix_item_count": len(matrix_items),
        }

        export_id = new_id("exportdoc")
        artifact_base = self.settings.exports_dir / export_id
        markdown_path = artifact_base.with_suffix(".md")
        json_path = artifact_base.with_suffix(".json")
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        json_path.write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        google_doc_id = ""
        export_status = "artifact_only"
        export_mode = "artifact_only"
        error_message = ""
        if self.settings.google_docs_mode == "gws":
            gws_result = self._try_gws_export(document_title, existing_google_doc_id, export_payload)
            google_doc_id = gws_result["google_doc_id"]
            export_status = gws_result["export_status"]
            export_mode = gws_result["export_mode"]
            error_message = gws_result["error_message"]

        record = self.repository.create_export(
            run_id=run_id,
            destination_type="google_docs",
            export_mode=export_mode,
            google_doc_id=google_doc_id,
            export_status=export_status,
            error_message=error_message,
            artifact_path=str(markdown_path),
            request_payload=export_payload,
        )
        record["markdown_path"] = str(markdown_path)
        record["json_path"] = str(json_path)
        return record

    def _resolve_export_selection(self, run_id: str, card_ids: list[str]) -> dict:
        run = self.repository.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        normalized_card_ids = []
        seen = set()
        for raw_card_id in card_ids:
            card_id = raw_card_id.strip()
            if not card_id or card_id in seen:
                continue
            seen.add(card_id)
            normalized_card_ids.append(card_id)
        if not normalized_card_ids:
            raise ValueError("At least one card must be selected for export")

        cards = []
        selection_snapshot = []
        review_snapshot = []
        for card_id in normalized_card_ids:
            card = self.repository.get_card(card_id)
            if not card:
                raise ValueError(f"Selected card does not exist: {card_id}")

            review = card.get("review") or {}
            review_decision = review.get("decision", "")
            selection_snapshot.append(
                {
                    "card_id": card["id"],
                    "run_id": card["run_id"],
                    "requested_run_id": run_id,
                    "review_status": review_decision,
                    "export_eligible": is_card_export_eligible(review_decision) and card["run_id"] == run_id,
                    "source_excluded_content_id": card.get("source_excluded_content_id"),
                }
            )
            review_snapshot.append(
                {
                    "card_id": card["id"],
                    "decision": review_decision,
                    "reviewer": review.get("reviewer", ""),
                    "note": review.get("note", ""),
                    "created_at": review.get("created_at", ""),
                }
            )

            if card["run_id"] != run_id:
                raise ValueError(f"Selected card {card_id} belongs to run {card['run_id']}, not {run_id}")
            if not review_decision:
                raise ValueError(f"Selected card {card_id} has not been reviewed as accepted")
            if not is_card_export_eligible(review_decision):
                raise ValueError(f"Selected card {card_id} is not export eligible because its review decision is {review_decision}")
            cards.append(card)

        cards.sort(key=lambda card: (card["topic_name"], card["paper_title"], card["created_at"], card["id"]))
        return {
            "cards": cards,
            "requested_card_ids": normalized_card_ids,
            "selection_snapshot": selection_snapshot,
            "review_snapshot": review_snapshot,
        }

    def _resolve_matrix_export_selection(self, run_id: str, matrix_item_ids: list[str]) -> dict:
        run = self.repository.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")
        normalized_ids = []
        seen = set()
        for raw_item_id in matrix_item_ids:
            item_id = str(raw_item_id or "").strip()
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            normalized_ids.append(item_id)
        if not normalized_ids:
            raise ValueError("At least one matrix item must be selected for export")

        matrix_items = []
        selection_snapshot = []
        review_snapshot = []
        for item_id in normalized_ids:
            item = self.repository.get_matrix_item(item_id)
            if not item:
                raise ValueError(f"Selected matrix item does not exist: {item_id}")
            review = item.get("review") or {}
            review_decision = review.get("decision", "")
            selection_snapshot.append(
                {
                    "matrix_item_id": item["id"],
                    "run_id": item["run_id"],
                    "requested_run_id": run_id,
                    "review_status": review_decision,
                    "export_eligible": is_card_export_eligible(review_decision) and item["run_id"] == run_id,
                    "dimension_label": item.get("dimension_label", ""),
                    "outcome_label": item.get("outcome_label", ""),
                }
            )
            review_snapshot.append(
                {
                    "matrix_item_id": item["id"],
                    "decision": review_decision,
                    "reviewer": review.get("reviewer", ""),
                    "note": review.get("note", ""),
                    "created_at": review.get("created_at", ""),
                }
            )
            if item["run_id"] != run_id:
                raise ValueError(f"Selected matrix item {item_id} belongs to run {item['run_id']}, not {run_id}")
            if not review_decision:
                raise ValueError(f"Selected matrix item {item_id} has not been reviewed as accepted")
            if not is_card_export_eligible(review_decision):
                raise ValueError(
                    f"Selected matrix item {item_id} is not export eligible because its review decision is {review_decision}"
                )
            matrix_items.append(item)
        matrix_items.sort(
            key=lambda item: (
                item.get("dimension_label", ""),
                item.get("outcome_label", ""),
                item.get("created_at", ""),
                item["id"],
            )
        )
        return {
            "matrix_items": matrix_items,
            "requested_matrix_item_ids": normalized_ids,
            "selection_snapshot": selection_snapshot,
            "review_snapshot": review_snapshot,
        }

    def _insert_text(self, index: int, text: str) -> list[dict]:
        return [{"insertText": {"location": {"index": index}, "text": text}}]

    def _offset_google_doc_requests(self, requests: list[dict], base_index: int) -> list[dict]:
        if base_index <= 1:
            return requests
        shifted = []
        offset = base_index - 1
        for request in requests:
            cloned = json.loads(json.dumps(request))
            insert_text = cloned.get("insertText")
            if insert_text and isinstance(insert_text.get("location"), dict):
                insert_text["location"]["index"] = int(insert_text["location"].get("index", 1)) + offset
            shifted.append(cloned)
        return shifted

    def _get_google_doc_append_index(self, document_id: str) -> int:
        result = subprocess.run(
            [
                "gws",
                "docs",
                "documents",
                "get",
                "--params",
                json.dumps({"documentId": document_id}),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout or "{}")
        content = payload.get("body", {}).get("content", [])
        if not content:
            return 1
        end_index = int(content[-1].get("endIndex", 1))
        return max(1, end_index - 1)

    def _try_gws_export(self, document_title: str, existing_google_doc_id: str, payload: dict) -> dict[str, str]:
        export_mode = "append" if existing_google_doc_id else "create"
        document_id = existing_google_doc_id
        try:
            if existing_google_doc_id:
                insert_index = self._get_google_doc_append_index(document_id)
            else:
                result = subprocess.run(
                    ["gws", "docs", "documents", "create", "--json", json.dumps({"title": document_title})],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                created = json.loads(result.stdout or "{}")
                document_id = created.get("documentId", "")
                insert_index = 1
            if not document_id:
                return {
                    "google_doc_id": "",
                    "export_status": "export_failed",
                    "export_mode": export_mode,
                    "error_message": "Google Docs export did not return a document id.",
                }
            shifted_requests = self._offset_google_doc_requests(payload["requests"], insert_index)
            subprocess.run(
                [
                    "gws",
                    "docs",
                    "documents",
                    "batchUpdate",
                    "--params",
                    json.dumps({"documentId": document_id}),
                    "--json",
                    json.dumps({"requests": shifted_requests}, ensure_ascii=False),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return {
                "google_doc_id": document_id,
                "export_status": "exported",
                "export_mode": export_mode,
                "error_message": "",
            }
        except Exception as error:
            error_message = str(error)
            stderr = getattr(error, "stderr", "")
            stdout = getattr(error, "stdout", "")
            if stderr:
                error_message = str(stderr).strip()
            elif stdout:
                error_message = str(stdout).strip()
            return {
                "google_doc_id": document_id,
                "export_status": "export_failed",
                "export_mode": export_mode,
                "error_message": error_message,
            }


class RunCoordinator:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings = settings
        self.repository = repository
        self.discovery = DiscoveryService()
        self.pipeline = PaperPipeline(settings, repository)
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)

    def create_run(self, topics_text: str, metadata: dict, local_pdfs: list[dict]) -> dict:
        run_metadata = dict(metadata or {})
        task_type = normalize_task_type(run_metadata.get("task_type", "aha_exploration"))
        run_metadata["task_type"] = task_type
        confirmed_plan = run_metadata.get("confirmed_plan") or {}
        topics = normalize_topics(topics_text)
        if not topics and isinstance(confirmed_plan, dict):
            topics = derive_topics_from_confirmed_plan(task_type, confirmed_plan)
        run = self.repository.create_run(topics_text="\n".join(topics), metadata=run_metadata)
        if not topics and not local_pdfs:
            self.repository.update_run_status(run["id"], "failed")
            raise ValueError("At least one topic or one local PDF is required.")

        topic_records = [self.repository.create_or_get_topic(topic_name) for topic_name in topics]
        if not topic_records and local_pdfs:
            topic_records = [self.repository.create_or_get_topic("manual-import")]
        topic_runs = [self.repository.create_topic_run(run["id"], topic["id"]) for topic in topic_records]

        local_mapping = self._ingest_local_pdfs(run["id"], topic_records, local_pdfs)
        self.repository.update_run_status(run["id"], "running")

        if task_type == "claim_evidence":
            threading.Thread(
                target=self._process_run_serially,
                args=(run["id"], topic_records, topic_runs, local_mapping),
                daemon=True,
            ).start()
        else:
            futures = []
            for topic, topic_run in zip(topic_records, topic_runs):
                futures.append(
                    self.executor.submit(
                        self._process_topic_run,
                        run["id"],
                        topic,
                        topic_run,
                        local_mapping.get(topic["name"].lower(), []),
                    )
                )

            threading.Thread(target=self._finalize_run, args=(run["id"], futures), daemon=True).start()
        return run

    def retry_topic_run(self, topic_run_id: str) -> dict:
        topic_run = self.repository.get_topic_run(topic_run_id)
        if not topic_run:
            raise LookupError("Topic run not found")
        if topic_run["status"] == "running":
            raise ValueError("Topic run is already active")
        topic = self.repository.get_topic(topic_run["topic_id"])
        if not topic:
            raise LookupError("Topic not found for topic run")
        local_papers = self.repository.list_local_papers_for_topic_run(topic_run["run_id"], topic_run["topic_id"])
        self.repository.update_run_status(topic_run["run_id"], "running")
        task_context = self._get_run_task_context(topic_run["run_id"])
        if normalize_task_type(task_context.get("task_type", "aha_exploration")) == "claim_evidence":
            threading.Thread(
                target=self._retry_topic_run_serially,
                args=(topic_run["run_id"], topic, topic_run, local_papers),
                daemon=True,
            ).start()
        else:
            future = self.executor.submit(self._process_topic_run, topic_run["run_id"], topic, topic_run, local_papers)
            threading.Thread(target=self._finalize_run, args=(topic_run["run_id"], [future]), daemon=True).start()
        refreshed = self.repository.get_topic_run(topic_run_id)
        if not refreshed:
            raise LookupError("Topic run disappeared during retry")
        return refreshed

    def _process_run_serially(
        self,
        run_id: str,
        topics: list[dict[str, Any]],
        topic_runs: list[dict[str, Any]],
        local_mapping: dict[str, list[dict[str, Any]]],
    ) -> None:
        for topic, topic_run in zip(topics, topic_runs):
            self._process_topic_run(run_id, topic, topic_run, local_mapping.get(topic["name"].lower(), []))
        self._finalize_run(run_id, [])

    def _retry_topic_run_serially(
        self,
        run_id: str,
        topic: dict[str, Any],
        topic_run: dict[str, Any],
        local_papers: list[dict[str, Any]],
    ) -> None:
        self._process_topic_run(run_id, topic, topic_run, local_papers)
        self._finalize_run(run_id, [])

    def _mark_topic_progress(
        self,
        topic_run_id: str,
        stats: dict[str, Any],
        *,
        stage: str = "",
        note: str = "",
        status: str = "running",
    ) -> None:
        now = utc_now()
        if stage and stats.get("current_stage") != stage:
            stats["current_stage"] = stage
            stats["stage_started_at"] = now
        elif stage and not stats.get("stage_started_at"):
            stats["stage_started_at"] = now
        stats["last_progress_at"] = now
        if note:
            stats.setdefault("processing_warnings", []).append(note)
        self.repository.update_topic_run(topic_run_id, status, stats=stats, started=(status == "running"))

    def _run_discovery_with_budget(self, topic_name: str, *, task_context: Optional[dict[str, Any]] = None) -> list[dict]:
        discovery_executor = ThreadPoolExecutor(max_workers=1)
        strategies = self._build_run_topic_discovery_strategies(topic_name, task_context or {})
        provider_count = max(1, len(self.discovery.providers))
        strategy_count = max(1, len(strategies))
        provider_timeout_budget = strategy_count * provider_count * 8
        # Respect the configured timeout as the hard cap when it is set explicitly.
        timeout_budget = int(self.settings.discovery_timeout_seconds or 0) or (provider_timeout_budget + 10)
        if strategies:
            future = discovery_executor.submit(self.discovery.discover_with_strategies, topic_name, strategies)
        else:
            future = discovery_executor.submit(self.discovery.discover, topic_name)
        try:
            discovered = future.result(timeout=timeout_budget)
            if normalize_task_type((task_context or {}).get("task_type", "aha_exploration")) == "claim_evidence":
                return self._rerank_claim_evidence_discovery_candidates(topic_name, discovered, task_context or {})
            return discovered
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"Discovery timed out after {timeout_budget}s for topic '{topic_name}'"
            ) from None
        finally:
            discovery_executor.shutdown(wait=False, cancel_futures=True)

    def _parse_paper(self, paper: dict) -> tuple[str, bool]:
        current = self.repository.get_paper(paper["id"])
        if not current:
            return paper["id"], False
        self.pipeline.parse_and_store(current)
        refreshed = self.repository.get_paper(paper["id"])
        parsed_ok = bool(refreshed and refreshed["parse_status"] == "parsed")
        if not parsed_ok:
            self.repository.replace_cards_for_paper_topic(paper["id"], paper["topic_id"], paper["run_id"], [])
        return paper["id"], parsed_ok

    def _build_cards_for_paper(self, paper: dict, topic: dict, run_id: str, *, active_memory: Optional[dict[str, Any]] = None) -> int:
        current = self.repository.get_paper(paper["id"])
        if not current or current["parse_status"] != "parsed":
            self.repository.replace_cards_for_paper_topic(current["id"] if current else paper["id"], topic["id"], run_id, [])
            return 0
        primary_topic = self._resolve_primary_topic_for_paper_run(paper["id"], run_id)
        if primary_topic and primary_topic["id"] != topic["id"]:
            self.repository.replace_cards_for_paper_topic(current["id"], topic["id"], run_id, [])
            return 0
        return self.pipeline.build_cards(current, topic, run_id, active_memory=active_memory)

    def _build_task_outputs_for_paper(
        self,
        paper: dict,
        topic: dict,
        run_id: str,
        *,
        task_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        current = self.repository.get_paper(paper["id"])
        if not current:
            self.repository.replace_cards_for_paper_topic(paper["id"], topic["id"], run_id, [])
            self.repository.replace_matrix_items_for_paper_topic(paper["id"], topic["id"], run_id, [])
            return {"cards": 0, "matrix_items": 0}
        resolved_context = task_context or {}
        active_memory = resolved_context.get("active_memory_snapshot") if isinstance(resolved_context.get("active_memory_snapshot"), dict) else None
        task_type = normalize_task_type(resolved_context.get("task_type", "aha_exploration"))
        if task_type == "claim_evidence":
            matrix_count = self.pipeline.build_matrix_items(
                current,
                topic,
                run_id,
                claim_plan=resolved_context.get("confirmed_plan", {}),
                active_memory=active_memory,
            )
            card_count = 0
            if resolved_context.get("also_generate_aha_cards"):
                card_count = self.pipeline.build_cards(current, topic, run_id, active_memory=active_memory)
            else:
                self.repository.replace_cards_for_paper_topic(current["id"], topic["id"], run_id, [])
            return {"cards": card_count, "matrix_items": matrix_count}
        primary_topic = self._resolve_primary_topic_for_paper_run(paper["id"], run_id)
        if primary_topic and primary_topic["id"] != topic["id"]:
            self.repository.replace_cards_for_paper_topic(current["id"], topic["id"], run_id, [])
            self.repository.replace_matrix_items_for_paper_topic(current["id"], topic["id"], run_id, [])
            return {"cards": 0, "matrix_items": 0}
        if current["parse_status"] != "parsed":
            self.repository.replace_cards_for_paper_topic(current["id"], topic["id"], run_id, [])
            self.repository.replace_matrix_items_for_paper_topic(current["id"], topic["id"], run_id, [])
            return {"cards": 0, "matrix_items": 0}
        self.repository.replace_matrix_items_for_paper_topic(current["id"], topic["id"], run_id, [])
        return {
            "cards": self.pipeline.build_cards(current, topic, run_id, active_memory=active_memory),
            "matrix_items": 0,
        }

    def _build_run_topic_priority(self, run_id: str) -> dict[str, int]:
        run = self.repository.get_run(run_id)
        if not run:
            return {}
        priority = {name.lower(): index for index, name in enumerate(normalize_topics(run.get("topics_text", "")), start=1)}
        if "manual-import" not in priority:
            priority["manual-import"] = max(priority.values(), default=0) + 1
        return priority

    def _get_run_task_context(self, run_id: str) -> dict[str, Any]:
        run = self.repository.get_run(run_id) or {}
        metadata = dict(run.get("metadata", {}) or {})
        task_type = normalize_task_type(metadata.get("task_type", "aha_exploration"))
        confirmed_plan = metadata.get("confirmed_plan", {}) if isinstance(metadata.get("confirmed_plan", {}), dict) else {}
        active_memory = metadata.get("active_memory_snapshot", {}) if isinstance(metadata.get("active_memory_snapshot", {}), dict) else {}
        return {
            "task_type": task_type,
            "confirmed_plan": confirmed_plan,
            "active_memory_snapshot": active_memory,
            "also_generate_aha_cards": bool(confirmed_plan.get("also_generate_aha_cards", False)),
            "research_brief": str(metadata.get("task_brief", "") or metadata.get("research_brief", "")).strip(),
            "local_only": bool(metadata.get("local_only", False)),
        }

    def _build_run_topic_discovery_strategies(self, topic_name: str, task_context: dict[str, Any]) -> list[dict[str, Any]]:
        if normalize_task_type(task_context.get("task_type", "aha_exploration")) != "claim_evidence":
            return []
        topic_entry = self.pipeline._resolve_claim_plan_topic_entry(task_context.get("confirmed_plan", {}), topic_name)
        return build_claim_evidence_search_strategies(
            str(topic_entry.get("query_anchor", "") or topic_name).strip(),
            outcome_terms=topic_entry.get("outcome_terms", []),
            dimension_key=str(topic_entry.get("dimension_key", "")).strip(),
            claim_text=str((task_context.get("confirmed_plan", {}) or {}).get("claim", "")).strip(),
            research_brief=str(task_context.get("research_brief", "")).strip(),
        )

    def _score_claim_evidence_text_candidate(
        self,
        *,
        title: str,
        abstract_text: str,
        asset_url: str,
        publication_year: Optional[int],
        confidence: float,
        topic_entry: dict[str, Any],
        task_context: dict[str, Any],
    ) -> float:
        cleaned_title = _clean_metadata_abstract(title)
        cleaned_abstract = _clean_metadata_abstract(abstract_text)
        title_lower = cleaned_title.lower()
        abstract_lower = cleaned_abstract.lower()
        combined = _joined_lower_text(cleaned_title, cleaned_abstract)
        if _title_has_claim_evidence_noise(cleaned_title):
            return -100.0
        score = max(float(confidence or 0.0), 0.0) * 3.0
        if asset_url:
            score += 4.0
        if cleaned_abstract:
            score += 2.0
        if any(token in combined for token in WORKPLACE_DIRECT_EVIDENCE_TOKENS):
            score += 6.0
        else:
            score -= 4.0
        if any(token in title_lower for token in OFFDOMAIN_CLAIM_EVIDENCE_TITLE_TOKENS):
            score -= 15.0

        dimension_key = str(topic_entry.get("dimension_key", "")).strip().lower()
        dimension_anchor = str(topic_entry.get("query_anchor", "") or "").strip()
        dimension_expansions = WORKPLACE_DIMENSION_QUERY_EXPANSIONS.get(dimension_key, ())
        outcome_terms = [str(item or "").strip() for item in topic_entry.get("outcome_terms", []) if str(item or "").strip()]

        for phrase in [dimension_anchor, *dimension_expansions, *outcome_terms]:
            normalized_phrase = _clean_metadata_abstract(phrase).lower()
            if not normalized_phrase:
                continue
            if normalized_phrase in combined:
                score += 3.5

        keywords = _keywordize_claim_text(
            dimension_anchor,
            " ".join(dimension_expansions),
            " ".join(outcome_terms),
            task_context.get("research_brief", ""),
            (task_context.get("confirmed_plan", {}) or {}).get("claim", ""),
        )
        title_hits = sum(1 for keyword in keywords if keyword in title_lower)
        abstract_hits = sum(1 for keyword in keywords if keyword in abstract_lower)
        score += title_hits * 1.25
        score += abstract_hits * 0.4

        if publication_year:
            score += max(min(int(publication_year) - 2000, 25), 0) * 0.04
        return score

    def _rerank_claim_evidence_discovery_candidates(
        self,
        topic_name: str,
        candidates: list[dict[str, Any]],
        task_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        topic_entry = self.pipeline._resolve_claim_plan_topic_entry(task_context.get("confirmed_plan", {}), topic_name)
        target_items = int(
            ((task_context.get("confirmed_plan", {}) or {}).get("evidence_policy", {}) or {}).get(
                "minimum_supporting_papers_per_dimension",
                3,
            )
            or 3
        )
        candidate_cap = max(CLAIM_EVIDENCE_DISCOVERY_CANDIDATE_CAP, target_items * 8)
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            if _title_has_claim_evidence_noise(str(candidate.get("title", "")).strip()):
                continue
            abstract_text = _extract_candidate_metadata_abstract(candidate)
            claim_score = self._score_claim_evidence_text_candidate(
                title=str(candidate.get("title", "")).strip(),
                abstract_text=abstract_text,
                asset_url=str(candidate.get("asset_url", "")).strip(),
                publication_year=candidate.get("publication_year"),
                confidence=float(candidate.get("confidence", 0.0) or 0.0),
                topic_entry=topic_entry,
                task_context=task_context,
            )
            enriched = dict(candidate)
            enriched["claim_candidate_score"] = round(claim_score, 4)
            scored.append(enriched)
        scored.sort(
            key=lambda item: (
                -float(item.get("claim_candidate_score", 0.0)),
                -(1 if item.get("asset_url") else 0),
                -(item.get("publication_year") or 0),
                str(item.get("title", "")).lower(),
            )
        )
        return scored[:candidate_cap]

    def _matrix_item_is_abstract_only(self, item: dict[str, Any]) -> bool:
        supporting_ids = list(item.get("supporting_section_ids", []) or [])
        return bool(supporting_ids and all(str(section_id).startswith("abstract_meta::") for section_id in supporting_ids))

    def _claim_evidence_target_item_count(self, task_context: dict[str, Any]) -> int:
        return int(
            ((task_context.get("confirmed_plan", {}) or {}).get("evidence_policy", {}) or {}).get(
                "minimum_supporting_papers_per_dimension",
                3,
            )
            or 3
        )

    def _list_topic_matrix_items(self, run_id: str, topic_name: str) -> list[dict[str, Any]]:
        return self.repository.list_matrix_items(run_id=run_id, topic=topic_name)

    def _rank_claim_evidence_abstract_fallback_papers(
        self,
        fallback_papers: list[dict[str, Any]],
        topic: dict[str, Any],
        task_context: dict[str, Any],
    ) -> list[tuple[float, dict[str, Any]]]:
        unique_papers = {paper["id"]: paper for paper in fallback_papers}
        if not unique_papers:
            return []
        topic_entry = self.pipeline._resolve_claim_plan_topic_entry(task_context.get("confirmed_plan", {}), topic["name"])
        ranked = []
        for paper in unique_papers.values():
            if _title_has_claim_evidence_noise(str(paper.get("title", "")).strip()):
                continue
            abstract_text = self.pipeline._extract_best_metadata_abstract_for_paper(paper["id"])
            if not abstract_text:
                continue
            score = self._score_claim_evidence_text_candidate(
                title=str(paper.get("title", "")).strip(),
                abstract_text=abstract_text,
                asset_url="",
                publication_year=paper.get("publication_year"),
                confidence=0.25,
                topic_entry=topic_entry,
                task_context=task_context,
            )
            if score <= -50:
                continue
            ranked.append((score, paper))
        ranked.sort(key=lambda item: (-item[0], -(item[1].get("publication_year") or 0), str(item[1].get("title", "")).lower()))
        return ranked

    def _resolve_primary_topic_from_routes(self, run_id: str, topic_routes: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not topic_routes:
            return None
        priority = self._build_run_topic_priority(run_id)
        unique_routes: dict[str, dict[str, Any]] = {}
        for route in topic_routes:
            unique_routes.setdefault(route["topic_id"], route)
        ordered = sorted(
            unique_routes.values(),
            key=lambda item: (
                priority.get(str(item.get("topic_name", "")).lower(), 10_000),
                str(item.get("topic_name", "")).lower(),
                str(item.get("topic_id", "")),
            ),
        )
        if not ordered:
            return None
        chosen = ordered[0]
        return {
            "id": chosen["topic_id"],
            "name": chosen.get("topic_name", ""),
            "topic_count": len(unique_routes),
        }

    def _resolve_primary_topic_for_paper_run(self, paper_id: str, run_id: str) -> Optional[dict[str, Any]]:
        return self._resolve_primary_topic_from_routes(
            run_id,
            self.repository.list_topic_runs_for_paper_run(paper_id, run_id),
        )

    def _refresh_run_status_from_topics(self, run_id: str) -> None:
        topic_runs = self.repository.list_topic_runs(run_id)
        statuses = [item["status"] for item in topic_runs]
        if not statuses:
            self.repository.update_run_status(run_id, "failed")
            return
        if any(status == "running" for status in statuses):
            self.repository.update_run_status(run_id, "running")
        elif all(status == "completed" for status in statuses):
            self.repository.update_run_status(run_id, "completed")
        elif "completed" in statuses:
            self.repository.update_run_status(run_id, "partial_failed")
        else:
            self.repository.update_run_status(run_id, "failed")

    def _ingest_local_pdfs(self, run_id: str, topics: list[dict], local_pdfs: list[dict]) -> dict[str, list[dict]]:
        topic_by_name = {topic["name"].lower(): topic for topic in topics}
        local_mapping: dict[str, list[dict]] = {topic["name"].lower(): [] for topic in topics}
        for item in local_pdfs:
            artifact_path = self.pipeline.ingest_local_pdf(item["path"])
            source_file = Path(item["path"])
            paper = self.repository.create_or_get_paper(
                title=source_file.stem,
                authors=[],
                publication_year=None,
                external_id=f"local::{source_file.resolve()}",
                source_type="local",
                local_path=item["path"],
                original_url="",
                access_status="open_fulltext",
                ingestion_status="artifact_ready",
                parse_status="pending",
                artifact_path=artifact_path,
            )
            target_topics = item.get("topics") or [topic["name"] for topic in topics] or ["manual-import"]
            for topic_name in target_topics:
                topic = topic_by_name.get(topic_name.lower()) or self.repository.create_or_get_topic(topic_name)
                topic_by_name[topic["name"].lower()] = topic
                local_mapping.setdefault(topic["name"].lower(), []).append(paper)
                self.repository.link_paper_to_topic(paper["id"], topic["id"], run_id, "local_pdf")
        return local_mapping

    def _create_discovery_strategy_records(
        self,
        run_id: str,
        topic: dict,
        topic_run: dict,
        discovered: list[dict],
    ) -> dict[tuple[str, str, str, int, str], dict]:
        strategy_counts: dict[tuple[str, str, str, int, str], int] = {}
        strategy_params: dict[tuple[str, str, str, int, str], dict] = {}
        for candidate in discovered:
            for source in candidate.get("discovery_sources", []):
                key = (
                    str(source.get("provider", "")).strip(),
                    str(source.get("strategy_type", "topic_query")).strip(),
                    str(source.get("query_text", topic["name"])).strip(),
                    int(source.get("strategy_order", 0)),
                    str(source.get("strategy_family", "core")).strip(),
                )
                strategy_counts[key] = strategy_counts.get(key, 0) + 1
                strategy_params.setdefault(key, source.get("strategy_params", {}))
        strategy_records = {}
        ordered_items = sorted(strategy_counts.items(), key=lambda item: (item[0][3], item[0][0], item[0][1], item[0][2]))
        for (provider, strategy_type, query_text, strategy_order, strategy_family), result_count in ordered_items:
            strategy_records[(provider, strategy_type, query_text, strategy_order, strategy_family)] = self.repository.create_discovery_strategy(
                run_id=run_id,
                topic_run_id=topic_run["id"],
                topic_id=topic["id"],
                provider=provider,
                strategy_family=strategy_family,
                strategy_type=strategy_type,
                strategy_order=strategy_order,
                query_text=query_text,
                result_count=result_count,
                metadata={
                    "topic_name": topic["name"],
                    "strategy_params": strategy_params.get((provider, strategy_type, query_text, strategy_order, strategy_family), {}),
                },
            )
        return strategy_records

    def _build_topic_run_metrics(self, run_id: str, topic: dict, topic_run: dict, stats: dict[str, Any]) -> dict[str, Any]:
        task_context = self._get_run_task_context(run_id)
        cards = self.repository.list_cards(run_id=run_id, topic=topic["name"])
        matrix_items = self.repository.list_matrix_items(run_id=run_id, topic=topic["name"])
        strategies = self.repository.list_discovery_strategies(topic_run_id=topic_run["id"])
        discovery_results = self.repository.list_discovery_results(topic_run_id=topic_run["id"])
        papers = {paper["id"]: paper for paper in self.repository.list_papers_for_topic_run(run_id, topic["id"])}
        if task_context["task_type"] == "claim_evidence":
            verdict_counts: dict[str, int] = {}
            dimension_counts: dict[str, int] = {}
            for item in matrix_items:
                verdict = str(item.get("verdict", "")).strip() or "unknown"
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                dimension = str(item.get("dimension_label", "") or item.get("dimension_key", "")).strip() or topic["name"]
                dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
            stats["claim_evidence_metrics"] = {
                "matrix_item_count": len(matrix_items),
                "verdict_counts": verdict_counts,
                "dimension_counts": dimension_counts,
            }
        cross_topic_resurfaced_papers = 0
        suppressed_cross_topic_papers = 0
        primary_topic_papers = 0
        max_topic_fanout = 0
        for paper_id in papers:
            topic_routes = self.repository.list_topic_runs_for_paper_run(paper_id, run_id)
            primary_topic = self._resolve_primary_topic_from_routes(run_id, topic_routes)
            topic_count = len({item["topic_id"] for item in topic_routes})
            max_topic_fanout = max(max_topic_fanout, topic_count)
            if topic_count >= 2:
                cross_topic_resurfaced_papers += 1
            if primary_topic and primary_topic["id"] == topic["id"]:
                primary_topic_papers += 1
            elif topic_count >= 2:
                suppressed_cross_topic_papers += 1

        near_duplicate_cards = 0
        same_pattern_cards = 0
        novel_cards = 0
        for card in cards:
            neighbors = self.repository.build_neighbors(card["id"], limit=1)
            if not neighbors:
                novel_cards += 1
                continue
            relationship = neighbors[0]["relationship"]
            if relationship == "near_duplicate":
                near_duplicate_cards += 1
            elif relationship == "same_pattern":
                same_pattern_cards += 1
            else:
                novel_cards += 1

        semantic_duplicate_cards = near_duplicate_cards + same_pattern_cards
        aha_class_clusters = cluster_cards_into_aha_classes(cards)
        card_to_aha_class_id: dict[str, str] = {}
        reportable_aha_class_count = 0
        for cluster in aha_class_clusters:
            representative = cluster["representative"]
            representative_color = str((representative.get("judgement") or {}).get("color", "yellow")).strip().lower()
            if representative_color in {"green", "yellow"}:
                reportable_aha_class_count += 1
            for member in cluster["members"]:
                card_to_aha_class_id[member["id"]] = cluster["aha_class_id"]
        independent_aha_class_count = len(aha_class_clusters)
        aha_class_duplication_ratio = round(
            max(len(cards) - independent_aha_class_count, 0) / max(len(cards), 1),
            4,
        )
        strategy_comparison = []
        seen_paper_ids: set[str] = set()
        seen_card_ids: set[str] = set()
        seen_aha_class_ids: set[str] = set()
        ordered_strategies = sorted(
            strategies,
            key=lambda item: (
                int(item.get("strategy_order", 0)),
                str(item.get("provider", "")),
                str(item.get("strategy_type", "")),
                str(item.get("query_text", "")),
            ),
        )
        for strategy in ordered_strategies:
            canonical_results = [
                item
                for item in discovery_results
                if item["strategy_id"] == strategy["id"] and item["dedupe_status"] == "canonical"
            ]
            canonical_paper_ids = {item["paper_id"] for item in canonical_results}
            accessible_papers = sum(
                1 for paper_id in canonical_paper_ids if (papers.get(paper_id) or {}).get("access_status") == "open_fulltext"
            )
            strategy_card_ids = {card["id"] for card in cards if card["paper_id"] in canonical_paper_ids}
            strategy_aha_class_ids = {
                card_to_aha_class_id[card_id]
                for card_id in strategy_card_ids
                if card_id in card_to_aha_class_id
            }
            yielded_cards = len(strategy_card_ids)
            incremental_new_papers = len(canonical_paper_ids - seen_paper_ids)
            incremental_new_cards = len(strategy_card_ids - seen_card_ids)
            incremental_new_aha_classes = len(strategy_aha_class_ids - seen_aha_class_ids)
            seen_paper_ids.update(canonical_paper_ids)
            seen_card_ids.update(strategy_card_ids)
            seen_aha_class_ids.update(strategy_aha_class_ids)
            strategy_comparison.append(
                {
                    "strategy_id": strategy["id"],
                    "provider": strategy["provider"],
                    "strategy_family": strategy.get("strategy_family", "core"),
                    "strategy_type": strategy["strategy_type"],
                    "strategy_order": int(strategy.get("strategy_order", 0)),
                    "query_text": strategy["query_text"],
                    "raw_hits": strategy["result_count"],
                    "canonical_candidates": len(canonical_paper_ids),
                    "accessible_papers": accessible_papers,
                    "yielded_cards": yielded_cards,
                    "yielded_aha_classes": len(strategy_aha_class_ids),
                    "incremental_new_papers": incremental_new_papers,
                    "incremental_new_cards": incremental_new_cards,
                    "incremental_new_aha_classes": incremental_new_aha_classes,
                }
            )
        comparison_tail = strategy_comparison[-3:] if len(strategy_comparison) >= 3 else strategy_comparison
        flattening_likely = (
            bool(comparison_tail)
            and all(item["incremental_new_aha_classes"] == 0 for item in comparison_tail)
            and (aha_class_duplication_ratio > 0)
        )

        stats["saturation_metrics"] = {
            "card_count": len(cards),
            "novel_cards": novel_cards,
            "same_pattern_cards": same_pattern_cards,
            "near_duplicate_cards": near_duplicate_cards,
            "semantic_duplicate_cards": semantic_duplicate_cards,
            "semantic_duplication_ratio": round(semantic_duplicate_cards / max(len(cards), 1), 4),
            "independent_aha_class_count": independent_aha_class_count,
            "reportable_aha_class_count": reportable_aha_class_count,
            "aha_class_duplication_ratio": aha_class_duplication_ratio,
            "cross_topic_resurfaced_papers": cross_topic_resurfaced_papers,
            "suppressed_cross_topic_papers": suppressed_cross_topic_papers,
            "primary_topic_papers": primary_topic_papers,
            "max_topic_fanout": max_topic_fanout,
            "search_strategy_comparison": strategy_comparison,
            "flattening_signal": {
                "tail_size": len(comparison_tail),
                "tail_incremental_new_cards": [item["incremental_new_cards"] for item in comparison_tail],
                "tail_incremental_new_aha_classes": [item["incremental_new_aha_classes"] for item in comparison_tail],
                "likely_flattening": flattening_likely,
            },
            "aha_class_shadow_summary": [
                {
                    "aha_class_id": cluster["aha_class_id"],
                    "representative_card_id": cluster["representative"]["id"],
                    "representative_title": cluster["representative"]["title"],
                    "member_card_ids": [member["id"] for member in cluster["members"]],
                    "member_paper_ids": sorted({member["paper_id"] for member in cluster["members"]}),
                }
                for cluster in aha_class_clusters
            ],
        }
        return stats

    def _attach_topic_stop_decision(self, topic: dict, stats: dict[str, Any]) -> dict[str, Any]:
        saturation_metrics = stats.get("saturation_metrics", {})
        if not isinstance(saturation_metrics, dict):
            return stats
        previous_snapshots = self.repository.list_topic_saturation_snapshots(topic=topic["name"], limit=10)
        evaluate_saturation_stop(
            current_metrics=saturation_metrics,
            previous_snapshots=previous_snapshots,
            policy=default_saturation_stop_policy(),
        )
        stats["saturation_stop"] = saturation_metrics.get("stop_decision", {})
        return stats

    def _process_accessible_papers(
        self,
        accessible_papers: list[dict],
        topic: dict,
        run_id: str,
        topic_run: dict,
        stats: dict[str, Any],
        task_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        unique_papers = {paper["id"]: paper for paper in accessible_papers}
        stats["accessible"] = len(unique_papers)
        if not unique_papers:
            return stats
        task_type = normalize_task_type((task_context or {}).get("task_type", "aha_exploration"))
        paper_worker_cap = 1 if task_type == "claim_evidence" else self.settings.max_workers
        paper_worker_count = max(1, min(paper_worker_cap, len(unique_papers)))

        self._mark_topic_progress(topic_run["id"], stats, stage="parsing")
        parsed_papers: list[dict] = []
        with ThreadPoolExecutor(max_workers=paper_worker_count) as paper_executor:
            futures = [
                paper_executor.submit(
                    self._parse_paper,
                    {
                        **paper,
                        "topic_id": topic["id"],
                        "run_id": run_id,
                    },
                )
                for paper in unique_papers.values()
            ]
            for future in as_completed(futures):
                try:
                    paper_id, parsed_ok = future.result()
                    if parsed_ok:
                        stats["parsed_papers"] += 1
                        refreshed = self.repository.get_paper(paper_id)
                        if refreshed:
                            parsed_papers.append(refreshed)
                except Exception as error:
                    stats["paper_processing_errors"] += 1
                    stats["processing_warnings"].append(f"paper_process_failed:{error}")
                    append_failure_log(
                        stats,
                        stage="parsing",
                        code="paper_parse_failed",
                        message=str(error),
                    )
                self._mark_topic_progress(topic_run["id"], stats, stage="parsing")

        if not parsed_papers:
            return stats

        generation_stage = "matrix_generation" if task_type == "claim_evidence" else "card_generation"
        self._mark_topic_progress(topic_run["id"], stats, stage=generation_stage)
        generation_worker_cap = 1 if task_type == "claim_evidence" else self.settings.max_workers
        with ThreadPoolExecutor(max_workers=max(1, min(generation_worker_cap, len(parsed_papers)))) as paper_executor:
            futures = [
                paper_executor.submit(
                    self._build_task_outputs_for_paper,
                    paper,
                    topic,
                    run_id,
                    task_context=task_context,
                )
                for paper in parsed_papers
            ]
            for future in as_completed(futures):
                try:
                    stats["card_generation_attempts"] += 1
                    output_counts = future.result()
                    stats["cards"] += int(output_counts.get("cards", 0) or 0)
                    stats["matrix_items"] += int(output_counts.get("matrix_items", 0) or 0)
                except Exception as error:
                    stats["paper_processing_errors"] += 1
                    stats["processing_warnings"].append(f"{generation_stage}_failed:{error}")
                    append_failure_log(
                        stats,
                        stage=generation_stage,
                        code=f"{generation_stage}_failed",
                        message=str(error),
                    )
                self._mark_topic_progress(topic_run["id"], stats, stage=generation_stage)
        return stats

    def _process_claim_evidence_abstract_fallback_papers(
        self,
        fallback_papers: list[dict],
        topic: dict,
        run_id: str,
        topic_run: dict,
        stats: dict[str, Any],
        *,
        task_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        resolved_context = task_context or {}
        ranked_papers = self._rank_claim_evidence_abstract_fallback_papers(
            fallback_papers,
            topic,
            resolved_context,
        )
        if not ranked_papers:
            return stats
        target_items = self._claim_evidence_target_item_count(resolved_context)
        existing_items = self._list_topic_matrix_items(run_id, topic["name"])
        current_total = len(existing_items)
        if current_total >= target_items:
            return stats
        current_abstract_only = sum(1 for item in existing_items if self._matrix_item_is_abstract_only(item))
        fulltext_items = [item for item in existing_items if not self._matrix_item_is_abstract_only(item)]
        abstract_allowance = max(0, target_items - current_total)
        if not fulltext_items:
            abstract_allowance = max(abstract_allowance, CLAIM_EVIDENCE_MAX_ABSTRACT_FALLBACK_ITEMS_NO_FULLTEXT)
        else:
            abstract_allowance = max(abstract_allowance, CLAIM_EVIDENCE_MAX_ABSTRACT_FALLBACK_ITEMS_WITH_FULLTEXT)
        attempt_budget = min(len(ranked_papers), max(abstract_allowance * 3, 3), CLAIM_EVIDENCE_MAX_ABSTRACT_FALLBACK_PAPER_ATTEMPTS)
        if attempt_budget <= 0:
            return stats
        self._mark_topic_progress(topic_run["id"], stats, stage="matrix_generation")
        attempted: set[str] = set()
        for _, paper in ranked_papers[:attempt_budget]:
            if paper["id"] in attempted:
                continue
            if current_total >= target_items:
                break
            if current_abstract_only >= abstract_allowance and current_total > 0:
                break
            attempted.add(paper["id"])
            try:
                stats["card_generation_attempts"] += 1
                before_ids = {item["id"] for item in self._list_topic_matrix_items(run_id, topic["name"])}
                output_counts = self._build_task_outputs_for_paper(
                    paper,
                    topic,
                    run_id,
                    task_context=resolved_context,
                )
                stats["cards"] += int(output_counts.get("cards", 0) or 0)
                stats["matrix_items"] += int(output_counts.get("matrix_items", 0) or 0)
                after_items = self._list_topic_matrix_items(run_id, topic["name"])
                current_total = len(after_items)
                new_items = [item for item in after_items if item["id"] not in before_ids]
                current_abstract_only += sum(1 for item in new_items if self._matrix_item_is_abstract_only(item))
            except Exception as error:
                stats["paper_processing_errors"] += 1
                stats["processing_warnings"].append(f"matrix_generation_failed:{error}")
                append_failure_log(
                    stats,
                    stage="matrix_generation",
                    code="matrix_generation_failed",
                    message=str(error),
                )
            self._mark_topic_progress(topic_run["id"], stats, stage="matrix_generation")
        return stats

    def _process_topic_run(self, run_id: str, topic: dict, topic_run: dict, local_papers: list[dict]) -> None:
        stats = initial_topic_run_stats()
        task_context = self._get_run_task_context(run_id)
        try:
            self._mark_topic_progress(topic_run["id"], stats, stage="discovery")
            if task_context.get("local_only"):
                discovered = []
            else:
                discovered = self._run_discovery_with_budget(topic["name"], task_context=task_context)
            raw_discovered = sum(len(item.get("discovery_sources", [])) or 1 for item in discovered)
            strategy_records = self._create_discovery_strategy_records(run_id, topic, topic_run, discovered)
            stats["discovered"] = len(discovered)
            stats["discovered_raw"] = raw_discovered
            stats["deduped_candidates"] = len(discovered)
            stats["duplicate_candidates_collapsed"] = max(raw_discovered - len(discovered), 0)
            stats["discovery_strategy_count"] = len(strategy_records)
            provider_summary: dict[str, dict[str, int]] = {}
            for item in discovered:
                for source in item.get("discovery_sources", []):
                    provider = str(source.get("provider", "")).strip() or "unknown"
                    summary = provider_summary.setdefault(provider, {"raw_hits": 0, "deduped_candidates": 0})
                    summary["raw_hits"] += 1
                provider = str(item.get("provider", "")).strip() or "unknown"
                summary = provider_summary.setdefault(provider, {"raw_hits": 0, "deduped_candidates": 0})
                summary["deduped_candidates"] += 1
            stats["provider_summary"] = provider_summary
            accessible_papers = list(local_papers)
            abstract_fallback_papers: list[dict] = []
            self._mark_topic_progress(topic_run["id"], stats, stage="acquisition")
            for item in discovered:
                try:
                    paper = self.repository.create_or_get_paper(
                        title=item["title"],
                        authors=item["authors"],
                        publication_year=item["publication_year"],
                        external_id=item["external_id"],
                        source_type=item["provider"],
                        original_url=item["original_url"],
                        access_status="metadata_only",
                        ingestion_status="discovered",
                        parse_status="pending",
                    )
                    primary_source_id = str(item.get("source_external_id", "")).strip()
                    for source in item.get("discovery_sources", []):
                        self.repository.add_paper_source(
                            paper["id"],
                            str(source.get("provider", item["provider"])).strip() or item["provider"],
                            float(source.get("confidence", item["confidence"])),
                            {
                                "source_external_id": source.get("source_external_id", ""),
                                "query_text": source.get("query_text", topic["name"]),
                                "strategy_family": source.get("strategy_family", "core"),
                                "strategy_type": source.get("strategy_type", "topic_query"),
                                "strategy_order": int(source.get("strategy_order", 0)),
                                "strategy_params": source.get("strategy_params", {}),
                                "identifiers": source.get("ids", {}),
                                "source_metadata": source.get("metadata", {}),
                            },
                        )
                        strategy_key = (
                            str(source.get("provider", "")).strip(),
                            str(source.get("strategy_type", "topic_query")).strip(),
                            str(source.get("query_text", topic["name"])).strip(),
                            int(source.get("strategy_order", 0)),
                            str(source.get("strategy_family", "core")).strip(),
                        )
                        strategy_record = strategy_records.get(strategy_key)
                        if strategy_record:
                            self.repository.create_discovery_result(
                                run_id=run_id,
                                topic_run_id=topic_run["id"],
                                strategy_id=strategy_record["id"],
                                dedupe_key=item["external_id"],
                                provider=str(source.get("provider", item["provider"])).strip() or item["provider"],
                                source_external_id=str(source.get("source_external_id", "")).strip(),
                                paper_title=item["title"],
                                authors=item["authors"],
                                publication_year=item["publication_year"],
                                original_url=str(source.get("original_url", item["original_url"])).strip(),
                                asset_url=str(source.get("asset_url", item["asset_url"])).strip(),
                                confidence=float(source.get("confidence", item["confidence"])),
                                dedupe_status="canonical" if str(source.get("source_external_id", "")).strip() == primary_source_id else "duplicate_source",
                                paper_id=paper["id"],
                                metadata={
                                    "topic_name": topic["name"],
                                    "strategy_family": source.get("strategy_family", "core"),
                                    "strategy_type": source.get("strategy_type", "topic_query"),
                                    "strategy_order": int(source.get("strategy_order", 0)),
                                    "strategy_params": source.get("strategy_params", {}),
                                    "identifiers": source.get("ids", {}),
                                    "source_metadata": source.get("metadata", {}),
                                },
                            )
                    self.repository.link_paper_to_topic(paper["id"], topic["id"], run_id, "search")
                    try:
                        artifact_path = self.pipeline.acquire_remote_asset_with_oa_fallback(paper, item.get("asset_url", ""))
                    except Exception as error:
                        stats["acquisition_errors"] += 1
                        stats["processing_warnings"].append(f"asset_acquire_failed:{paper['id']}:{error}")
                        append_failure_log(
                            stats,
                            stage="acquisition",
                            code="asset_acquire_failed",
                            message=str(error),
                        )
                        artifact_path = None
                    if artifact_path:
                        self.repository.update_paper(
                            paper["id"],
                            access_status="open_fulltext",
                            ingestion_status="artifact_ready",
                            artifact_path=artifact_path,
                        )
                        paper["artifact_path"] = artifact_path
                        paper["access_status"] = "open_fulltext"
                        accessible_papers.append(paper)
                    else:
                        self.repository.update_paper(paper["id"], access_status="manual_needed", ingestion_status="queued")
                        stats["queued_for_access"] += 1
                        self.repository.create_access_queue_item(
                            paper_id=paper["id"],
                            run_id=run_id,
                            reason="Relevant paper discovered but full text could not be acquired automatically.",
                        )
                        if task_context["task_type"] == "claim_evidence":
                            refreshed_paper = self.repository.get_paper(paper["id"]) or paper
                            if self.pipeline._build_claim_evidence_abstract_fallback_sections(refreshed_paper):
                                abstract_fallback_papers.append(refreshed_paper)
                    self._mark_topic_progress(topic_run["id"], stats, stage="acquisition")
                except Exception as error:
                    stats["processing_warnings"].append(
                        f"discovery_item_failed:{item.get('external_id', item.get('title', 'unknown'))}:{error}"
                    )
                    append_failure_log(
                        stats,
                        stage="acquisition",
                        code="discovery_item_failed",
                        message=str(error),
                    )
                    self._mark_topic_progress(topic_run["id"], stats, stage="acquisition")
                    continue

            stats = self._process_accessible_papers(
                accessible_papers,
                topic,
                run_id,
                topic_run,
                stats,
                task_context=task_context,
            )
            if task_context["task_type"] == "claim_evidence" and abstract_fallback_papers:
                stats = self._process_claim_evidence_abstract_fallback_papers(
                    abstract_fallback_papers,
                    topic,
                    run_id,
                    topic_run,
                    stats,
                    task_context=task_context,
                )
            stats = self._build_topic_run_metrics(run_id, topic, topic_run, stats)
            if task_context["task_type"] == "aha_exploration":
                stats = self._attach_topic_stop_decision(topic, stats)
            self._mark_topic_progress(topic_run["id"], stats, stage="review_ready")
            self.repository.create_topic_saturation_snapshot(
                run_id=run_id,
                topic_run_id=topic_run["id"],
                topic_id=topic["id"],
                saturation_metrics=stats.get("saturation_metrics", {}),
            )
            stats["current_stage"] = "completed"
            stats["stage_started_at"] = stats.get("stage_started_at") or utc_now()
            stats["last_progress_at"] = utc_now()
            self.repository.update_topic_run(topic_run["id"], "completed", stats=stats)
        except Exception as error:
            stats["error"] = str(error)
            append_failure_log(stats, stage=stats.get("current_stage", "unknown"), code="topic_run_failed", message=str(error))
            stats["current_stage"] = "failed"
            stats["last_progress_at"] = utc_now()
            self.repository.update_topic_run(topic_run["id"], "failed", stats=stats)

    def _finalize_run(self, run_id: str, futures: Iterable) -> None:
        for future in futures:
            try:
                future.result()
            except Exception:
                continue
        try:
            self._refresh_run_status_from_topics(run_id)
        except sqlite3.OperationalError as error:
            if "unable to open database file" not in str(error).lower():
                raise
