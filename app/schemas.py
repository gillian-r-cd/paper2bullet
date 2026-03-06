"""
This module defines request and response schemas for the Paper to Bullet API.
Main models: run intake payloads, review updates, export requests, and API summaries.
Data structures: Pydantic models used by FastAPI endpoints.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LocalPdfInput(BaseModel):
    path: str
    topics: List[str] = Field(default_factory=list)


class RunCreateRequest(BaseModel):
    topics_text: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
    local_pdfs: List[LocalPdfInput] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    reviewer: str = Field(default="internal")
    decision: str
    note: str = Field(default="")


class ExportRequest(BaseModel):
    run_id: str
    card_ids: List[str]
    document_title: str
    existing_google_doc_id: str = Field(default="")


class CalibrationExampleInput(BaseModel):
    example_type: str
    topic_name: str
    audience: str = Field(default="")
    title: str
    source_text: str
    evidence: List[dict] = Field(default_factory=list)
    expected_cards: List[dict] = Field(default_factory=list)
    expected_exclusions: List[dict] = Field(default_factory=list)
    rationale: str = Field(default="")
    tags: List[str] = Field(default_factory=list)


class CalibrationSetImportRequest(BaseModel):
    name: str
    description: str = Field(default="")
    metadata: dict = Field(default_factory=dict)
    examples: List[CalibrationExampleInput] = Field(default_factory=list)


class RunSummary(BaseModel):
    id: str
    created_at: str
    status: str
    topics_text: str


class TopicRunSummary(BaseModel):
    id: str
    run_id: str
    topic_name: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    stats: dict


class CardSummary(BaseModel):
    id: str
    run_id: str
    topic_name: str
    paper_title: str
    title: str
    color: str
    course_transformation: str
    teachable_one_liner: str
    review_decision: str
    status: str
