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
    embedding_json TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS figures (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    figure_label TEXT NOT NULL,
    caption TEXT NOT NULL,
    storage_path TEXT NOT NULL,
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
    created_at TEXT NOT NULL,
    FOREIGN KEY (paper_id) REFERENCES papers(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
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
    card_id TEXT NOT NULL,
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
    google_doc_id TEXT NOT NULL,
    export_status TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict:
    return {description[0]: row[index] for index, description in enumerate(cursor.description)}


def get_connection(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.row_factory = dict_factory
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def init_db(db_path: Path) -> None:
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


def ensure_column(connection: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    existing_columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in existing_columns:
        return
    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


@contextmanager
def db_cursor(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    connection = get_connection(db_path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
