"""
This module implements the Phase 0 workflow for the Paper to Bullet application.
Main classes: `Repository`, `PaperPipeline`, and `RunCoordinator`.
Data structures: runs, topic jobs, papers, sections, candidate cards, judgements, access queue items, and export artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional
from uuid import uuid4

from .config import Settings
from .db import db_cursor
from .llm import (
    CARD_RUBRIC_VERSION,
    EXTRACTION_PROMPT_VERSION,
    JUDGEMENT_PROMPT_VERSION,
    LLMCardEngine,
    LLMGenerationError,
)

try:
    from markitdown import MarkItDown
except ImportError:  # pragma: no cover - optional dependency
    MarkItDown = None


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
        "discovered_raw": 0,
        "deduped_candidates": 0,
        "duplicate_candidates_collapsed": 0,
        "discovery_strategy_count": 0,
        "queued_for_access": 0,
        "provider_summary": {},
    }


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


CARD_REVIEW_DECISIONS = frozenset({"accepted", "rejected", "keep_for_later", "needs_manual_check"})
EXCLUDED_REVIEW_DECISIONS = frozenset({"accepted", "reopened", "needs_manual_check"})


def allowed_review_decisions(target_type: str) -> frozenset[str]:
    if target_type == "card":
        return CARD_REVIEW_DECISIONS
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
        with db_cursor(self.settings.db_path) as connection:
            return connection.execute(query, params).fetchone()

    def _fetchall(self, query: str, params: tuple = ()) -> list[dict]:
        with db_cursor(self.settings.db_path) as connection:
            return connection.execute(query, params).fetchall()

    def _hydrate_evidence(self, evidence: list[dict]) -> list[dict]:
        hydrated = []
        for item in evidence:
            quote_zh = str(item.get("quote_zh", "")).strip() or str(item.get("analysis", "")).strip()
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
        row["paper_url"] = row.get("paper_url", "")
        row["source_excluded_content_id"] = row.get("source_excluded_content_id")
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
                source_excluded_content_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                card["created_at"],
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
            row["example_count"] = self._fetchone(
                "SELECT COUNT(*) AS count FROM calibration_examples WHERE calibration_set_id = ?",
                (row["id"],),
            )["count"]
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
        strategy_type: str,
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
            "strategy_type": strategy_type,
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
                    id, run_id, topic_run_id, topic_id, provider, strategy_type, query_text,
                    status, result_count, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["run_id"],
                    record["topic_run_id"],
                    record["topic_id"],
                    record["provider"],
                    record["strategy_type"],
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
            ORDER BY discovery_strategies.created_at ASC, discovery_strategies.id ASC
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
            SELECT discovery_results.*, discovery_strategies.query_text, discovery_strategies.strategy_type
            FROM discovery_results
            JOIN discovery_strategies ON discovery_strategies.id = discovery_results.strategy_id
            {where_clause}
            ORDER BY discovery_results.created_at ASC, discovery_results.id ASC
            """,
            tuple(params),
        )
        for row in rows:
            row["authors"] = json.loads(row["authors_json"])
            row["metadata"] = json.loads(row["metadata_json"])
        return rows

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

    def add_paper_source(self, paper_id: str, provider: str, confidence: float, metadata: dict) -> None:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO paper_sources(id, paper_id, provider, confidence, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_id("psrc"), paper_id, provider, confidence, json.dumps(metadata, ensure_ascii=False), utc_now()),
            )

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

    def replace_sections(self, paper_id: str, sections: list[dict]) -> None:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute("DELETE FROM paper_sections WHERE paper_id = ?", (paper_id,))
            for section in sections:
                connection.execute(
                    """
                    INSERT INTO paper_sections(
                        id, paper_id, section_order, section_title, paragraph_text, page_number, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        section["id"],
                        paper_id,
                        section["section_order"],
                        section["section_title"],
                        section["paragraph_text"],
                        section["page_number"],
                        json.dumps(section["embedding"]),
                    ),
                )

    def replace_figures(self, paper_id: str, figures: list[dict]) -> None:
        with db_cursor(self.settings.db_path) as connection:
            connection.execute("DELETE FROM figures WHERE paper_id = ?", (paper_id,))
            for figure in figures:
                connection.execute(
                    """
                    INSERT INTO figures(id, paper_id, figure_label, caption, storage_path, linked_section_ids_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        figure["id"],
                        paper_id,
                        figure["figure_label"],
                        figure["caption"],
                        figure["storage_path"],
                        json.dumps(figure["linked_section_ids"]),
                    ),
                )

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

    def create_export(self, run_id: str, destination_type: str, google_doc_id: str, export_status: str, artifact_path: str, request_payload: dict) -> dict:
        record = {
            "id": new_id("export"),
            "run_id": run_id,
            "destination_type": destination_type,
            "google_doc_id": google_doc_id,
            "export_status": export_status,
            "artifact_path": artifact_path,
            "request_json": json.dumps(request_payload, ensure_ascii=False),
            "created_at": utc_now(),
            "completed_at": utc_now(),
        }
        with db_cursor(self.settings.db_path) as connection:
            connection.execute(
                """
                INSERT INTO exports(id, run_id, destination_type, google_doc_id, export_status, artifact_path, request_json, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["run_id"],
                    record["destination_type"],
                    record["google_doc_id"],
                    record["export_status"],
                    record["artifact_path"],
                    record["request_json"],
                    record["created_at"],
                    record["completed_at"],
                ),
            )
        return record

    def get_run(self, run_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))

    def list_runs(self) -> list[dict]:
        return self._fetchall("SELECT * FROM runs ORDER BY created_at DESC")

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
            row["stats"] = json.loads(row["stats_json"])
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
            row["stats"] = json.loads(row["stats_json"])
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

    def get_sections(self, paper_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM paper_sections WHERE paper_id = ? ORDER BY section_order ASC",
            (paper_id,),
        )
        for row in rows:
            row["embedding"] = json.loads(row["embedding_json"])
        return rows

    def get_figures(self, paper_id: str) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM figures WHERE paper_id = ? ORDER BY id ASC",
            (paper_id,),
        )
        for row in rows:
            row["linked_section_ids"] = json.loads(row["linked_section_ids_json"])
        return rows

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
        row["excluded_content"] = self.list_excluded_content(
            paper_id=row["paper_id"],
            topic_id=row["topic_id"],
            run_id=row["run_id"],
        )
        return row

    def list_cards(self, run_id: Optional[str] = None, topic: str = "") -> list[dict]:
        params: list[str] = []
        filters = []
        if run_id:
            filters.append("candidate_cards.run_id = ?")
            params.append(run_id)
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

    def get_excluded_content_summary(self, excluded_content_id: str) -> Optional[dict]:
        row = self._fetchone(
            """
            SELECT paper_excluded_content.*, papers.title AS paper_title, papers.original_url AS paper_url, topics.name AS topic_name
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
            SELECT paper_excluded_content.*, papers.title AS paper_title, papers.original_url AS paper_url, topics.name AS topic_name
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
        if normalized_item_type in {"cards", "both"}:
            for card in self.list_cards(run_id=run_id, topic=topic):
                if review_status and (card.get("review_decision") or "") != review_status:
                    continue
                items.append(
                    {
                        "object_type": "card",
                        "object_id": card["id"],
                        "run_id": card["run_id"],
                        "topic_name": card["topic_name"],
                        "paper_title": card["paper_title"],
                        "paper_url": card.get("paper_url", ""),
                        "display_title": card["title"],
                        "color": card.get("color", ""),
                        "course_transformation": card.get("course_transformation", ""),
                        "teachable_one_liner": card.get("teachable_one_liner", ""),
                        "review_status": card.get("review_decision", ""),
                        "exclusion_type": "",
                        "export_eligible": is_card_export_eligible(card.get("review_decision") or ""),
                        "source_excluded_content_id": card.get("source_excluded_content_id"),
                        "promoted_from_excluded": bool(card.get("source_excluded_content_id")),
                        "created_at": card["created_at"],
                    }
                )
        if normalized_item_type in {"excluded", "both"}:
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
                items.append(
                    {
                        "object_type": "excluded",
                        "object_id": item["id"],
                        "run_id": item["run_id"],
                        "topic_name": item["topic_name"],
                        "paper_title": item["paper_title"],
                        "paper_url": item.get("paper_url", ""),
                        "display_title": item["label"],
                        "color": "",
                        "course_transformation": "",
                        "teachable_one_liner": "",
                        "review_status": latest_review.get("decision", ""),
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
        return None

    def list_access_queue(self, run_id: Optional[str] = None) -> list[dict]:
        if run_id:
            return self._fetchall(
                """
                SELECT access_queue.*, papers.title AS paper_title, papers.original_url
                FROM access_queue
                JOIN papers ON papers.id = access_queue.paper_id
                WHERE access_queue.run_id = ?
                ORDER BY access_queue.created_at DESC
                """,
                (run_id,),
            )
        return self._fetchall(
            """
            SELECT access_queue.*, papers.title AS paper_title, papers.original_url
            FROM access_queue
            JOIN papers ON papers.id = access_queue.paper_id
            ORDER BY access_queue.created_at DESC
            """
        )

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
    def __init__(self, providers: Optional[list[Any]] = None):
        self.providers = providers or [
            OpenAlexDiscoveryProvider(),
            ArxivDiscoveryProvider(),
            CrossrefDiscoveryProvider(),
            SemanticScholarDiscoveryProvider(),
        ]

    def discover(self, topic: str) -> list[dict]:
        raw_results = []
        for provider in self.providers:
            raw_results.extend(provider.discover(topic))
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
                "strategy_type": result.get("strategy_type", "topic_query"),
                "query_text": result.get("query_text", topic),
                "source_external_id": result.get("source_external_id", ""),
                "original_url": result.get("original_url", ""),
                "asset_url": result.get("asset_url", ""),
                "confidence": result.get("confidence", 0.0),
                "metadata": result.get("metadata", {}),
                "ids": result.get("ids", {}),
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

    def discover(self, topic: str) -> list[dict]:
        query = urllib.parse.quote(topic)
        url = f"https://api.openalex.org/works?search={query}&per-page=5"
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
                    "strategy_type": "topic_query",
                    "query_text": topic,
                    "title": item.get("display_name", "Untitled"),
                    "authors": [author["author"]["display_name"] for author in item.get("authorships", [])[:5] if author.get("author")],
                    "publication_year": item.get("publication_year"),
                    "source_external_id": item.get("id", ""),
                    "original_url": landing_url,
                    "asset_url": pdf_url,
                    "confidence": 0.7,
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

    def discover(self, topic: str) -> list[dict]:
        query = urllib.parse.quote(topic)
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
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
                    "strategy_type": "topic_query",
                    "query_text": topic,
                    "title": entry.findtext("atom:title", default="Untitled", namespaces=namespace).strip(),
                    "authors": [author.findtext("atom:name", default="", namespaces=namespace) for author in entry.findall("atom:author", namespace)],
                    "publication_year": int(entry.findtext("atom:published", default="1900", namespaces=namespace)[:4]),
                    "source_external_id": arxiv_id,
                    "original_url": arxiv_id,
                    "asset_url": pdf_url,
                    "confidence": 0.8,
                    "ids": {"arxiv": arxiv_id},
                    "metadata": {"summary": entry.findtext("atom:summary", default="", namespaces=namespace)},
                }
            )
        return records


class CrossrefDiscoveryProvider:
    provider_name = "crossref"

    def discover(self, topic: str) -> list[dict]:
        query = urllib.parse.quote(topic)
        url = f"https://api.crossref.org/works?query={query}&rows=5"
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
                    "strategy_type": "topic_query",
                    "query_text": topic,
                    "title": title,
                    "authors": authors,
                    "publication_year": publication_year,
                    "source_external_id": doi or item.get("URL", ""),
                    "original_url": item.get("URL", ""),
                    "asset_url": "",
                    "confidence": 0.55,
                    "ids": {"doi": doi},
                    "metadata": item,
                }
            )
        return records


class SemanticScholarDiscoveryProvider:
    provider_name = "semantic_scholar"

    def discover(self, topic: str) -> list[dict]:
        query = urllib.parse.quote(topic)
        fields = urllib.parse.quote("title,year,authors,url,openAccessPdf,externalIds,abstract")
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields={fields}"
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
                    "strategy_type": "topic_query",
                    "query_text": topic,
                    "title": item.get("title", "Untitled"),
                    "authors": [author.get("name", "") for author in item.get("authors", [])[:5] if author.get("name")],
                    "publication_year": item.get("year"),
                    "source_external_id": item.get("paperId", ""),
                    "original_url": item.get("url", ""),
                    "asset_url": (item.get("openAccessPdf") or {}).get("url", ""),
                    "confidence": 0.65,
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

    def _extract_figures_from_markdown(self, markdown_text: str, source_path: Path) -> list[dict]:
        figures = []
        for index, match in enumerate(re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", markdown_text), start=1):
            caption = match.group(1).strip() or Path(match.group(2).strip()).name or f"Figure {index}"
            target = match.group(2).strip()
            storage_path = ""
            candidate_path = (source_path.parent / target).resolve()
            if target and not target.startswith(("http://", "https://", "data:")) and candidate_path.exists():
                storage_path = str(candidate_path)
            figures.append(
                {
                    "id": new_id("figure"),
                    "figure_label": f"Figure {index}",
                    "caption": caption,
                    "storage_path": storage_path,
                    "linked_section_ids": [],
                }
            )
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
        figures = self._extract_figures_from_markdown(markdown_text, source_path)
        if sections:
            linked_ids = [section["id"] for section in sections]
            for figure in figures:
                figure["linked_section_ids"] = linked_ids
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
        return {"sections": sections, "figures": [], "artifact_type": "pdf"}

    def _parse_html(self, source_path: Path) -> dict:
        text = source_path.read_text(encoding="utf-8", errors="ignore")
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
            raise ParseFailure("parse_failed", f"Could not extract readable HTML text from {source_path.name}")
        return {"sections": sections, "figures": [], "artifact_type": "html"}

    def _extract_pdf_pages(self, source_path: Path) -> list[str]:
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
            with urllib.request.urlopen(asset_url, timeout=15) as response, temporary_path.open("wb") as handle:
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
            self.repository.replace_sections(paper["id"], [])
            self.repository.replace_figures(paper["id"], [])
            self.repository.update_paper(
                paper["id"],
                parse_status=error.status,
                ingestion_status="parse_failed",
                parse_failure_reason=error.reason,
                card_generation_status="blocked_parse_failed",
                card_generation_failure_reason="Card generation was skipped because paper parsing failed.",
            )
            return 0
        parsed_snapshot_path = self.settings.parsed_dir / f"{paper['id']}.json"
        parsed_snapshot_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        self.repository.replace_sections(paper["id"], parsed["sections"])
        self.repository.replace_figures(paper["id"], parsed["figures"])
        self.repository.update_paper(
            paper["id"],
            parse_status="parsed",
            ingestion_status="ready",
            parse_failure_reason="",
            card_generation_status="pending",
            card_generation_failure_reason="",
            artifact_path=paper["artifact_path"] or paper["local_path"],
        )
        return len(parsed["sections"])

    def build_cards(self, paper: dict, topic: dict, run_id: str) -> int:
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
            generation_output = self.generate_outputs_for_sections(sections, topic, paper)
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

    def generate_outputs_for_sections(self, sections: list[dict], topic: dict, paper: dict) -> dict:
        return self._build_cards_with_llm(sections, topic, paper)

    def _build_cards_with_llm(self, sections: list[dict], topic: dict, paper: dict) -> dict:
        active_calibration_set = self.repository.get_active_calibration_set()
        figures = self.repository.get_figures(paper["id"])
        extracted_output = self.card_engine.extract_candidates(
            topic_name=topic["name"],
            paper_title=paper["title"],
            sections=sections,
            figures=figures,
            calibration_examples=(active_calibration_set or {}).get("examples", []),
            calibration_set_name=(active_calibration_set or {}).get("name", ""),
        )
        judged_output = self.card_engine.judge_candidates(
            topic_name=topic["name"],
            paper_title=paper["title"],
            extracted_cards=extracted_output["cards"],
            calibration_examples=(active_calibration_set or {}).get("examples", []),
            calibration_set_name=(active_calibration_set or {}).get("name", ""),
        )
        return {
            "cards": [self._finalize_card(card, topic["name"]) for card in judged_output["cards"]],
            "excluded_content": [
                self._finalize_excluded_content(item)
                for item in extracted_output["excluded_content"]
            ],
        }

    def _finalize_card(self, card: dict, topic_name: str) -> dict:
        evidence = card["evidence"]
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
            "evidence": evidence,
            "figure_ids": card.get("figure_ids", []),
            "status": card.get("status", "candidate"),
            "embedding": embedding_for_text(embedding_source),
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
            "missed_aha_count": 0,
            "boundary_mismatch_count": 0,
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
            elif result["regression_type"] == "missed_aha":
                summary["missed_aha_count"] += 1
            elif result["regression_type"] == "boundary_mismatch":
                summary["boundary_mismatch_count"] += 1
        return summary


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
                    markdown_lines.append(f"- [{color}] **{card['title']}**")
                    markdown_lines.append(f"  - 它在课程里变成什么：{card['course_transformation']}")
                    markdown_lines.append(f"  - 可直接讲的一句话：{card.get('teachable_one_liner', '')}")
                    markdown_lines.append(f"  - 判断理由：{(card['judgement'] or {}).get('reason', '')}")
                    for evidence in card["evidence"]:
                        markdown_lines.append(f"  - 证据原文（EN）：{evidence['quote']}")
                        markdown_lines.append(f"  - 证据译文（ZH）：{evidence.get('quote_zh', '')}")
                        if evidence.get("analysis"):
                            markdown_lines.append(f"    - 为什么值得教：{evidence['analysis']}")
                    markdown_lines.append("")
                    block = (
                        f"[{color}] {card['title']}\n"
                        f"它在课程里变成什么：{card['course_transformation']}\n"
                        f"可直接讲的一句话：{card.get('teachable_one_liner', '')}\n"
                        f"判断理由：{(card['judgement'] or {}).get('reason', '')}\n"
                    )
                    for evidence in card["evidence"]:
                        block += f"证据原文（EN）：{evidence['quote']}\n"
                        block += f"证据译文（ZH）：{evidence.get('quote_zh', '')}\n"
                        if evidence.get("analysis"):
                            block += f"为什么值得教：{evidence['analysis']}\n"
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
        if self.settings.google_docs_mode == "gws":
            google_doc_id = self._try_gws_export(document_title, existing_google_doc_id, export_payload)
            export_status = "exported" if google_doc_id else "artifact_only"

        record = self.repository.create_export(
            run_id=run_id,
            destination_type="google_docs",
            google_doc_id=google_doc_id,
            export_status=export_status,
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

    def _insert_text(self, index: int, text: str) -> list[dict]:
        return [{"insertText": {"location": {"index": index}, "text": text}}]

    def _try_gws_export(self, document_title: str, existing_google_doc_id: str, payload: dict) -> str:
        try:
            if existing_google_doc_id:
                document_id = existing_google_doc_id
            else:
                result = subprocess.run(
                    ["gws", "docs", "documents", "create", "--json", json.dumps({"title": document_title})],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                created = json.loads(result.stdout or "{}")
                document_id = created.get("documentId", "")
            if not document_id:
                return ""
            subprocess.run(
                [
                    "gws",
                    "docs",
                    "documents",
                    "batchUpdate",
                    "--params",
                    json.dumps({"documentId": document_id}),
                    "--json",
                    json.dumps({"requests": payload["requests"]}, ensure_ascii=False),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return document_id
        except Exception:
            return ""


class RunCoordinator:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings = settings
        self.repository = repository
        self.discovery = DiscoveryService()
        self.pipeline = PaperPipeline(settings, repository)
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)

    def create_run(self, topics_text: str, metadata: dict, local_pdfs: list[dict]) -> dict:
        topics = normalize_topics(topics_text)
        run = self.repository.create_run(topics_text="\n".join(topics), metadata=metadata)
        if not topics and not local_pdfs:
            self.repository.update_run_status(run["id"], "failed")
            raise ValueError("At least one topic or one local PDF is required.")

        topic_records = [self.repository.create_or_get_topic(topic_name) for topic_name in topics]
        if not topic_records and local_pdfs:
            topic_records = [self.repository.create_or_get_topic("manual-import")]
        topic_runs = [self.repository.create_topic_run(run["id"], topic["id"]) for topic in topic_records]

        local_mapping = self._ingest_local_pdfs(run["id"], topic_records, local_pdfs)
        self.repository.update_run_status(run["id"], "running")

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

        self.executor.submit(self._finalize_run, run["id"], futures)
        return run

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

    def _create_discovery_strategy_records(self, run_id: str, topic: dict, topic_run: dict, discovered: list[dict]) -> dict[tuple[str, str, str], dict]:
        strategy_counts: dict[tuple[str, str, str], int] = {}
        for candidate in discovered:
            for source in candidate.get("discovery_sources", []):
                key = (
                    str(source.get("provider", "")).strip(),
                    str(source.get("strategy_type", "topic_query")).strip(),
                    str(source.get("query_text", topic["name"])).strip(),
                )
                strategy_counts[key] = strategy_counts.get(key, 0) + 1
        strategy_records = {}
        for (provider, strategy_type, query_text), result_count in strategy_counts.items():
            strategy_records[(provider, strategy_type, query_text)] = self.repository.create_discovery_strategy(
                run_id=run_id,
                topic_run_id=topic_run["id"],
                topic_id=topic["id"],
                provider=provider,
                strategy_type=strategy_type,
                query_text=query_text,
                result_count=result_count,
                metadata={"topic_name": topic["name"]},
            )
        return strategy_records

    def _build_topic_run_metrics(self, run_id: str, topic: dict, topic_run: dict, stats: dict[str, Any]) -> dict[str, Any]:
        cards = self.repository.list_cards(run_id=run_id, topic=topic["name"])
        strategies = self.repository.list_discovery_strategies(topic_run_id=topic_run["id"])
        discovery_results = self.repository.list_discovery_results(topic_run_id=topic_run["id"])
        papers = {paper["id"]: paper for paper in self.repository.list_papers_for_topic_run(run_id, topic["id"])}

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
        strategy_comparison = []
        for strategy in strategies:
            canonical_results = [
                item
                for item in discovery_results
                if item["strategy_id"] == strategy["id"] and item["dedupe_status"] == "canonical"
            ]
            canonical_paper_ids = {item["paper_id"] for item in canonical_results}
            accessible_papers = sum(
                1 for paper_id in canonical_paper_ids if (papers.get(paper_id) or {}).get("access_status") == "open_fulltext"
            )
            yielded_cards = sum(1 for card in cards if card["paper_id"] in canonical_paper_ids)
            strategy_comparison.append(
                {
                    "strategy_id": strategy["id"],
                    "provider": strategy["provider"],
                    "strategy_type": strategy["strategy_type"],
                    "query_text": strategy["query_text"],
                    "raw_hits": strategy["result_count"],
                    "canonical_candidates": len(canonical_paper_ids),
                    "accessible_papers": accessible_papers,
                    "yielded_cards": yielded_cards,
                }
            )

        stats["saturation_metrics"] = {
            "card_count": len(cards),
            "novel_cards": novel_cards,
            "same_pattern_cards": same_pattern_cards,
            "near_duplicate_cards": near_duplicate_cards,
            "semantic_duplicate_cards": semantic_duplicate_cards,
            "semantic_duplication_ratio": round(semantic_duplicate_cards / max(len(cards), 1), 4),
            "search_strategy_comparison": strategy_comparison,
        }
        return stats

    def _process_topic_run(self, run_id: str, topic: dict, topic_run: dict, local_papers: list[dict]) -> None:
        self.repository.update_topic_run(topic_run["id"], "running", started=True)
        stats = initial_topic_run_stats()
        try:
            discovered = self.discovery.discover(topic["name"])
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
            for item in discovered:
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
                            "strategy_type": source.get("strategy_type", "topic_query"),
                            "identifiers": source.get("ids", {}),
                            "source_metadata": source.get("metadata", {}),
                        },
                    )
                    strategy_key = (
                        str(source.get("provider", "")).strip(),
                        str(source.get("strategy_type", "topic_query")).strip(),
                        str(source.get("query_text", topic["name"])).strip(),
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
                                "identifiers": source.get("ids", {}),
                                "source_metadata": source.get("metadata", {}),
                            },
                        )
                self.repository.link_paper_to_topic(paper["id"], topic["id"], run_id, "search")
                artifact_path = self.pipeline.acquire_remote_asset(paper, item.get("asset_url", ""))
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

            unique_papers = {paper["id"]: paper for paper in accessible_papers}
            stats["accessible"] = len(unique_papers)
            futures = [self.executor.submit(self._process_paper, paper, topic, run_id) for paper in unique_papers.values()]
            for future in as_completed(futures):
                stats["cards"] += future.result()
            stats = self._build_topic_run_metrics(run_id, topic, topic_run, stats)
            self.repository.update_topic_run(topic_run["id"], "completed", stats=stats)
        except Exception as error:
            stats["error"] = str(error)
            self.repository.update_topic_run(topic_run["id"], "failed", stats=stats)

    def _process_paper(self, paper: dict, topic: dict, run_id: str) -> int:
        current = self.repository._fetchone("SELECT * FROM papers WHERE id = ?", (paper["id"],))
        if not current:
            return 0
        self.pipeline.parse_and_store(current)
        current = self.repository._fetchone("SELECT * FROM papers WHERE id = ?", (paper["id"],))
        if not current or current["parse_status"] != "parsed":
            self.repository.replace_cards_for_paper_topic(current["id"] if current else paper["id"], topic["id"], run_id, [])
            return 0
        return self.pipeline.build_cards(current, topic, run_id)

    def _finalize_run(self, run_id: str, futures: Iterable) -> None:
        statuses = []
        for future in futures:
            try:
                future.result()
                statuses.append("completed")
            except Exception:
                statuses.append("failed")
        if statuses and all(status == "completed" for status in statuses):
            self.repository.update_run_status(run_id, "completed")
        elif "completed" in statuses:
            self.repository.update_run_status(run_id, "partial_failed")
        else:
            self.repository.update_run_status(run_id, "failed")
