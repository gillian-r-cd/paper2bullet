"""
This module manages SQLite persistence for the Paper to Bullet application.
Main functions: schema initialization, connection creation, and row conversion helpers.
Data structures: SQL tables for runs, papers, evidence sections, cards, judgements, reviews, and exports.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS = 30
DEFAULT_SQLITE_JOURNAL_MODE = "WAL"
_DB_RUNTIME_CONFIGS: dict[str, dict[str, str | int]] = {}


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    topics_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    activated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration_examples (
    id TEXT PRIMARY KEY,
    calibration_set_id TEXT NOT NULL,
    example_type TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    audience TEXT NOT NULL,
    title TEXT NOT NULL,
    source_text TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    expected_cards_json TEXT NOT NULL,
    expected_exclusions_json TEXT NOT NULL,
    rationale TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (calibration_set_id) REFERENCES calibration_sets(id)
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id TEXT PRIMARY KEY,
    prompt_stage TEXT NOT NULL,
    version TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    details_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    activated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rubric_versions (
    id TEXT PRIMARY KEY,
    rubric_name TEXT NOT NULL,
    version TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    details_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    activated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_runs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    stats_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS discovery_strategies (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    topic_run_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    strategy_order INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    status TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (topic_run_id) REFERENCES topic_runs(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS discovery_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    topic_run_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    source_external_id TEXT NOT NULL,
    paper_title TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    publication_year INTEGER,
    original_url TEXT NOT NULL,
    asset_url TEXT NOT NULL,
    confidence REAL NOT NULL,
    dedupe_status TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (topic_run_id) REFERENCES topic_runs(id),
    FOREIGN KEY (strategy_id) REFERENCES discovery_strategies(id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS topic_saturation_snapshots (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    topic_run_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    card_count INTEGER NOT NULL,
    near_duplicate_cards INTEGER NOT NULL,
    same_pattern_cards INTEGER NOT NULL,
    novel_cards INTEGER NOT NULL,
    semantic_duplication_ratio REAL NOT NULL,
    likely_flattening INTEGER NOT NULL,
    stop_decision TEXT NOT NULL,
    stop_reason TEXT NOT NULL,
    stop_policy_json TEXT NOT NULL,
    tail_incremental_new_cards_json TEXT NOT NULL,
    search_strategy_comparison_json TEXT NOT NULL,
    saturation_metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (topic_run_id) REFERENCES topic_runs(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    publication_year INTEGER,
    external_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    local_path TEXT NOT NULL,
    original_url TEXT NOT NULL,
    access_status TEXT NOT NULL,
    ingestion_status TEXT NOT NULL,
    parse_status TEXT NOT NULL,
    parse_failure_reason TEXT NOT NULL DEFAULT '',
    card_generation_status TEXT NOT NULL DEFAULT 'pending',
    card_generation_failure_reason TEXT NOT NULL DEFAULT '',
    artifact_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_sources (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    confidence REAL NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    PRIMARY KEY (paper_id, topic_id, run_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS paper_sections (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    section_order INTEGER NOT NULL,
    section_title TEXT NOT NULL,
    paragraph_text TEXT NOT NULL,
    page_number INTEGER,
    section_kind TEXT NOT NULL DEFAULT 'other',
    section_label TEXT NOT NULL DEFAULT '',
    is_front_matter INTEGER NOT NULL DEFAULT 0,
    is_abstract INTEGER NOT NULL DEFAULT 0,
    is_body INTEGER NOT NULL DEFAULT 0,
    body_role TEXT NOT NULL DEFAULT '',
    has_figure_reference INTEGER NOT NULL DEFAULT 0,
    source_format TEXT NOT NULL DEFAULT '',
    selection_score REAL NOT NULL DEFAULT 0,
    selection_reason_json TEXT NOT NULL DEFAULT '{}',
    embedding_json TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS figures (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    figure_label TEXT NOT NULL,
    caption TEXT NOT NULL,
    page_number INTEGER,
    storage_path TEXT NOT NULL,
    asset_status TEXT NOT NULL DEFAULT 'metadata_only',
    asset_kind TEXT NOT NULL DEFAULT '',
    asset_local_path TEXT NOT NULL DEFAULT '',
    asset_source_url TEXT NOT NULL DEFAULT '',
    mime_type TEXT NOT NULL DEFAULT '',
    byte_size INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL DEFAULT '',
    width INTEGER,
    height INTEGER,
    validation_error TEXT NOT NULL DEFAULT '',
    linked_section_ids_json TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS candidate_cards (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    title TEXT NOT NULL,
    granularity_level TEXT NOT NULL,
    course_transformation TEXT NOT NULL,
    teachable_one_liner TEXT NOT NULL DEFAULT '',
    draft_body TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    figure_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    source_excluded_content_id TEXT,
    primary_section_ids_json TEXT NOT NULL DEFAULT '[]',
    supporting_section_ids_json TEXT NOT NULL DEFAULT '[]',
    paper_specific_object TEXT NOT NULL DEFAULT '',
    claim_type TEXT NOT NULL DEFAULT '',
    evidence_level TEXT NOT NULL DEFAULT '',
    body_grounding_reason TEXT NOT NULL DEFAULT '',
    grounding_quality TEXT NOT NULL DEFAULT '',
    duplicate_cluster_id TEXT NOT NULL DEFAULT '',
    duplicate_rank INTEGER NOT NULL DEFAULT 0,
    duplicate_disposition TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (source_excluded_content_id) REFERENCES paper_excluded_content(id)
);

CREATE TABLE IF NOT EXISTS judgements (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL,
    color TEXT NOT NULL,
    reason TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (card_id) REFERENCES candidate_cards(id)
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id TEXT PRIMARY KEY,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    card_id TEXT,
    reviewer TEXT NOT NULL,
    decision TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (card_id) REFERENCES candidate_cards(id)
);

CREATE TABLE IF NOT EXISTS paper_excluded_content (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    label TEXT NOT NULL,
    exclusion_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    section_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS access_queue (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    priority TEXT NOT NULL,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS exports (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    destination_type TEXT NOT NULL,
    export_mode TEXT NOT NULL,
    google_doc_id TEXT NOT NULL,
    export_status TEXT NOT NULL,
    error_message TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    calibration_set_id TEXT NOT NULL,
    calibration_set_name TEXT NOT NULL,
    llm_mode TEXT NOT NULL,
    model_name TEXT NOT NULL,
    extraction_prompt_version TEXT NOT NULL,
    judgement_prompt_version TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    status TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    FOREIGN KEY (calibration_set_id) REFERENCES calibration_sets(id)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id TEXT PRIMARY KEY,
    evaluation_run_id TEXT NOT NULL,
    calibration_example_id TEXT NOT NULL,
    example_type TEXT NOT NULL,
    title TEXT NOT NULL,
    source_text TEXT NOT NULL,
    extraction_json TEXT NOT NULL,
    judgement_json TEXT NOT NULL,
    expected_json TEXT NOT NULL,
    actual_json TEXT NOT NULL,
    verdict TEXT NOT NULL,
    regression_type TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (evaluation_run_id) REFERENCES evaluation_runs(id),
    FOREIGN KEY (calibration_example_id) REFERENCES calibration_examples(id)
);

CREATE TABLE IF NOT EXISTS paper_understanding_records (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    version TEXT NOT NULL,
    understanding_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS card_plans (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    version TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict:
    return {description[0]: row[index] for index, description in enumerate(cursor.description)}


def normalize_db_key(db_path: Path) -> str:
    return str(Path(db_path).resolve())


def configure_db_runtime(db_path: Path, *, busy_timeout_seconds: int, journal_mode: str) -> None:
    normalized_mode = (journal_mode or DEFAULT_SQLITE_JOURNAL_MODE).strip().upper() or DEFAULT_SQLITE_JOURNAL_MODE
    if not normalized_mode.isalpha():
        normalized_mode = DEFAULT_SQLITE_JOURNAL_MODE
    _DB_RUNTIME_CONFIGS[normalize_db_key(db_path)] = {
        "busy_timeout_seconds": max(1, int(busy_timeout_seconds)),
        "journal_mode": normalized_mode,
    }


def get_db_runtime_config(db_path: Path) -> dict[str, str | int]:
    return _DB_RUNTIME_CONFIGS.get(
        normalize_db_key(db_path),
        {
            "busy_timeout_seconds": DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS,
            "journal_mode": DEFAULT_SQLITE_JOURNAL_MODE,
        },
    )


def apply_connection_pragmas(connection: sqlite3.Connection, *, busy_timeout_seconds: int, journal_mode: str) -> None:
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute(f"PRAGMA busy_timeout = {max(1, busy_timeout_seconds) * 1000};")
    try:
        connection.execute(f"PRAGMA journal_mode = {journal_mode};")
    except sqlite3.OperationalError:
        pass
    if journal_mode == "WAL":
        connection.execute("PRAGMA synchronous = NORMAL;")


def get_connection(db_path: Path) -> sqlite3.Connection:
    runtime = get_db_runtime_config(db_path)
    busy_timeout_seconds = int(runtime["busy_timeout_seconds"])
    journal_mode = str(runtime["journal_mode"])
    connection = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        timeout=float(max(5, busy_timeout_seconds)),
    )
    connection.row_factory = dict_factory
    apply_connection_pragmas(connection, busy_timeout_seconds=busy_timeout_seconds, journal_mode=journal_mode)
    return connection


def init_db(db_path: Path, *, busy_timeout_seconds: int | None = None, journal_mode: str | None = None) -> None:
    if busy_timeout_seconds is not None or journal_mode is not None:
        configure_db_runtime(
            db_path,
            busy_timeout_seconds=busy_timeout_seconds or DEFAULT_SQLITE_BUSY_TIMEOUT_SECONDS,
            journal_mode=journal_mode or DEFAULT_SQLITE_JOURNAL_MODE,
        )
    connection = get_connection(db_path)
    try:
        connection.executescript(SCHEMA)
        ensure_migrations(connection)
        connection.commit()
    finally:
        connection.close()


def ensure_migrations(connection: sqlite3.Connection) -> None:
    ensure_column(connection, "papers", "parse_failure_reason", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "papers", "card_generation_status", "TEXT NOT NULL DEFAULT 'pending'")
    ensure_column(connection, "papers", "card_generation_failure_reason", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "teachable_one_liner", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "source_excluded_content_id", "TEXT")
    ensure_column(connection, "discovery_strategies", "strategy_family", "TEXT NOT NULL DEFAULT 'core'")
    ensure_column(connection, "discovery_strategies", "strategy_order", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(connection, "topic_saturation_snapshots", "stop_decision", "TEXT NOT NULL DEFAULT 'insufficient_history'")
    ensure_column(connection, "topic_saturation_snapshots", "stop_reason", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "topic_saturation_snapshots", "stop_policy_json", "TEXT NOT NULL DEFAULT '{}'")
    ensure_column(connection, "exports", "export_mode", "TEXT NOT NULL DEFAULT 'artifact_only'")
    ensure_column(connection, "exports", "error_message", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "paper_sections", "section_kind", "TEXT NOT NULL DEFAULT 'other'")
    ensure_column(connection, "paper_sections", "section_label", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "paper_sections", "is_front_matter", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(connection, "paper_sections", "is_abstract", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(connection, "paper_sections", "is_body", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(connection, "paper_sections", "body_role", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "paper_sections", "has_figure_reference", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(connection, "paper_sections", "source_format", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "paper_sections", "selection_score", "REAL NOT NULL DEFAULT 0")
    ensure_column(connection, "paper_sections", "selection_reason_json", "TEXT NOT NULL DEFAULT '{}'")
    ensure_column(connection, "figures", "page_number", "INTEGER")
    ensure_column(connection, "figures", "asset_status", "TEXT NOT NULL DEFAULT 'metadata_only'")
    ensure_column(connection, "figures", "asset_kind", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "figures", "asset_local_path", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "figures", "asset_source_url", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "figures", "mime_type", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "figures", "byte_size", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(connection, "figures", "sha256", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "figures", "width", "INTEGER")
    ensure_column(connection, "figures", "height", "INTEGER")
    ensure_column(connection, "figures", "validation_error", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "primary_section_ids_json", "TEXT NOT NULL DEFAULT '[]'")
    ensure_column(connection, "candidate_cards", "supporting_section_ids_json", "TEXT NOT NULL DEFAULT '[]'")
    ensure_column(connection, "candidate_cards", "paper_specific_object", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "claim_type", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "evidence_level", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "body_grounding_reason", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "grounding_quality", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "duplicate_cluster_id", "TEXT NOT NULL DEFAULT ''")
    ensure_column(connection, "candidate_cards", "duplicate_rank", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(connection, "candidate_cards", "duplicate_disposition", "TEXT NOT NULL DEFAULT ''")
    ensure_promoted_linkage_index(connection)
    ensure_review_decisions_target_schema(connection)
    ensure_discovery_indexes(connection)


def ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if not table_exists:
        return
    existing_columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in existing_columns:
        return
    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def ensure_review_decisions_target_schema(connection: sqlite3.Connection) -> None:
    existing_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(review_decisions)").fetchall()
    }
    if {"target_type", "target_id"}.issubset(existing_columns):
        connection.execute(
            """
            UPDATE review_decisions
            SET target_type = COALESCE(NULLIF(target_type, ''), 'card'),
                target_id = COALESCE(NULLIF(target_id, ''), card_id)
            """
        )
        return

    connection.execute("PRAGMA foreign_keys = OFF;")
    try:
        connection.execute("ALTER TABLE review_decisions RENAME TO review_decisions_legacy")
        connection.execute(
            """
            CREATE TABLE review_decisions (
                id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                card_id TEXT,
                reviewer TEXT NOT NULL,
                decision TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (card_id) REFERENCES candidate_cards(id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO review_decisions(id, target_type, target_id, card_id, reviewer, decision, note, created_at)
            SELECT id, 'card', card_id, card_id, reviewer, decision, note, created_at
            FROM review_decisions_legacy
            """
        )
        connection.execute("DROP TABLE review_decisions_legacy")
    finally:
        connection.execute("PRAGMA foreign_keys = ON;")


def ensure_promoted_linkage_index(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_cards_source_excluded_content_id
        ON candidate_cards(source_excluded_content_id)
        WHERE source_excluded_content_id IS NOT NULL
        """
    )


def ensure_discovery_indexes(connection: sqlite3.Connection) -> None:
    existing_tables = {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    if "discovery_strategies" in existing_tables:
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_discovery_strategies_topic_run_id
            ON discovery_strategies(topic_run_id, created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_discovery_strategies_strategy_order
            ON discovery_strategies(topic_run_id, strategy_order, provider)
            """
        )
    if "discovery_results" in existing_tables:
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_discovery_results_topic_run_id
            ON discovery_results(topic_run_id, created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_discovery_results_dedupe_key
            ON discovery_results(dedupe_key)
            """
        )
    if "topic_saturation_snapshots" in existing_tables:
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_topic_saturation_snapshots_topic_id
            ON topic_saturation_snapshots(topic_id, created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_topic_saturation_snapshots_topic_run_id
            ON topic_saturation_snapshots(topic_run_id)
            """
        )


@contextmanager
def db_cursor(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    connection = get_connection(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
