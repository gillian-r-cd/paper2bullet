"""
This module defines request and response schemas for the Paper to Bullet API.
Main models: run intake payloads, review updates, export requests, and API summaries.
Data structures: Pydantic models used by FastAPI endpoints.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

TaskType = Literal["aha_exploration", "claim_evidence"]


class LocalPdfInput(BaseModel):
    path: str
    topics: List[str] = Field(default_factory=list)


class RunCreateRequest(BaseModel):
    topics_text: str = Field(default="")
    research_brief: str = Field(default="")
    task_type: TaskType = Field(default="aha_exploration")
    confirmed_plan: dict = Field(default_factory=dict)
    use_active_memory: bool = Field(default=True)
    metadata: dict = Field(default_factory=dict)
    local_pdfs: List[LocalPdfInput] = Field(default_factory=list)


class SearchTermRecommendationRequest(BaseModel):
    research_goal: str = Field(default="")
    max_terms: int = Field(default=6, ge=1, le=12)


class ResearchPlanDraftRequest(BaseModel):
    research_brief: str = Field(default="")
    task_type: Literal["auto", "aha_exploration", "claim_evidence"] = Field(default="auto")
    max_terms: int = Field(default=6, ge=1, le=12)
    use_active_memory: bool = Field(default=True)
    also_generate_aha_cards: bool = Field(default=False)


class ReviewRequest(BaseModel):
    reviewer: str = Field(default="internal")
    decision: Literal["accepted", "rejected", "keep_for_later", "needs_manual_check", "reopened"]
    note: str = Field(default="")


class PromoteExcludedRequest(BaseModel):
    reviewer: str = Field(default="internal")
    note: str = Field(default="")


class ReviewCommentRequest(BaseModel):
    reviewer: str = Field(default="internal")
    comment: str = Field(default="")


class ExportRequest(BaseModel):
    run_id: str
    export_kind: Literal["cards", "matrix_items"] = Field(default="cards")
    card_ids: List[str] = Field(default_factory=list)
    matrix_item_ids: List[str] = Field(default_factory=list)
    document_title: str
    existing_google_doc_id: str = Field(default="")


class AccessQueueReactivateRequest(BaseModel):
    local_path: str
    reviewer: str = Field(default="internal")


class SinglePaperValidationRequest(BaseModel):
    topic_id: str
    run_id: str


class PaperQuestionRequest(BaseModel):
    question: str = Field(default="")
    max_sections: int = Field(default=6, ge=1, le=12)


class MemoryDraftRequest(BaseModel):
    task_type: Literal["", "aha_exploration", "claim_evidence"] = Field(default="")
    run_id: str = Field(default="")
    reviewer: str = Field(default="internal")


class MemoryActivateRequest(BaseModel):
    reviewer: str = Field(default="internal")
    memory_draft: dict = Field(default_factory=dict)


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


class EvaluationRunRequest(BaseModel):
    calibration_set_id: str = Field(default="")


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
