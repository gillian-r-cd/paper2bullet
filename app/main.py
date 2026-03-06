"""
This module exposes the FastAPI application for the Paper to Bullet Phase 0 workflow.
Main functions: `create_app()` and HTTP endpoints for runs, cards, reviews, access queue, and exports.
Data structures: app state wiring settings, repository, coordinator, and export service.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .config import Settings, get_settings
from .db import init_db
from .schemas import CalibrationSetImportRequest, ExportRequest, ReviewRequest, RunCreateRequest
from .llm import LLMCardEngine, LLMGenerationError
from .services import ExportService, Repository, RunCoordinator


def build_index_html(static_path: Path) -> str:
    return static_path.read_text(encoding="utf-8")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    app_settings = settings or get_settings()
    app_settings.ensure_directories()
    init_db(app_settings.db_path)

    repository = Repository(app_settings)
    coordinator = RunCoordinator(app_settings, repository)
    exporter = ExportService(app_settings, repository)
    llm_engine = LLMCardEngine(app_settings)

    app = FastAPI(title=app_settings.app_name)
    index_html = build_index_html(Path(__file__).parent / "static" / "index.html")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return index_html

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "data_dir": str(app_settings.data_dir), "db_path": str(app_settings.db_path)}

    @app.post("/api/runs")
    def create_run(payload: RunCreateRequest) -> dict:
        try:
            run = coordinator.create_run(
                topics_text=payload.topics_text,
                metadata=payload.metadata,
                local_pdfs=[item.model_dump() for item in payload.local_pdfs],
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"run": run}

    @app.get("/api/runs")
    def list_runs() -> dict:
        return {
            "runs": repository.list_runs(),
            "topic_runs": repository.list_topic_runs(),
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run = repository.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run": run,
            "topic_runs": repository.list_topic_runs(run_id),
            "access_queue": repository.list_access_queue(run_id),
        }

    @app.get("/api/calibration/sets")
    def list_calibration_sets() -> dict:
        return {
            "active_set": repository.get_active_calibration_set(),
            "sets": repository.list_calibration_sets(),
        }

    @app.get("/api/calibration/sets/{calibration_set_id}")
    def get_calibration_set(calibration_set_id: str) -> dict:
        calibration_set = repository.get_calibration_set(calibration_set_id)
        if not calibration_set:
            raise HTTPException(status_code=404, detail="Calibration set not found")
        return {"calibration_set": calibration_set}

    @app.post("/api/calibration/sets/import")
    def import_calibration_set(payload: CalibrationSetImportRequest) -> dict:
        try:
            calibration_set = repository.import_calibration_set(
                name=payload.name,
                description=payload.description,
                metadata=payload.metadata,
                examples=[item.model_dump() for item in payload.examples],
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"calibration_set": calibration_set}

    @app.post("/api/calibration/sets/{calibration_set_id}/activate")
    def activate_calibration_set(calibration_set_id: str) -> dict:
        calibration_set = repository.activate_calibration_set(calibration_set_id)
        if not calibration_set:
            raise HTTPException(status_code=404, detail="Calibration set not found")
        return {"calibration_set": calibration_set}

    @app.get("/api/cards")
    def list_cards(run_id: str = Query(default=""), topic: str = Query(default="")) -> dict:
        cards = repository.list_cards(run_id=run_id or None, topic=topic)
        return {"cards": cards}

    @app.get("/api/cards/{card_id}")
    def get_card(card_id: str) -> dict:
        card = repository.get_card(card_id)
        if not card:
            raise HTTPException(status_code=404, detail="Card not found")
        card["neighbors"] = repository.build_neighbors(card_id)
        if card.get("judgement") and isinstance(card["judgement"].get("created_at"), str):
            pass
        return {"card": card}

    @app.post("/api/cards/{card_id}/review")
    def review_card(card_id: str, payload: ReviewRequest) -> dict:
        card = repository.get_card(card_id)
        if not card:
            raise HTTPException(status_code=404, detail="Card not found")
        repository.create_review_decision(card_id, payload.reviewer, payload.decision, payload.note)
        return {"status": "ok", "card_id": card_id}

    @app.get("/api/access-queue")
    def list_access_queue(run_id: str = Query(default="")) -> dict:
        return {"items": repository.list_access_queue(run_id or None)}

    @app.post("/api/exports/google-doc")
    def export_google_doc(payload: ExportRequest) -> dict:
        if not payload.card_ids:
            raise HTTPException(status_code=400, detail="At least one card must be selected for export")
        export_record = exporter.export_google_doc_package(
            run_id=payload.run_id,
            card_ids=payload.card_ids,
            document_title=payload.document_title,
            existing_google_doc_id=payload.existing_google_doc_id,
        )
        return {"export": export_record}

    @app.get("/api/debug/state")
    def debug_state() -> dict:
        return {
            "settings": {
                "data_dir": str(app_settings.data_dir),
                "db_path": str(app_settings.db_path),
                "max_workers": app_settings.max_workers,
                "google_docs_mode": app_settings.google_docs_mode,
                "llm_mode": app_settings.llm_mode,
                "llm_model": app_settings.llm_model,
            },
            "runs": repository.list_runs(),
            "topic_runs": repository.list_topic_runs(),
            "cards": repository.list_cards(),
            "access_queue": repository.list_access_queue(),
            "calibration_sets": repository.list_calibration_sets(),
            "active_calibration_set": repository.get_active_calibration_set(),
        }

    @app.post("/api/llm/smoke")
    def llm_smoke() -> dict:
        try:
            return llm_engine.smoke_test()
        except LLMGenerationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
