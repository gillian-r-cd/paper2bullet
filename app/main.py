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
from .schemas import CalibrationSetImportRequest, EvaluationRunRequest, ExportRequest, PromoteExcludedRequest, ReviewRequest, RunCreateRequest
from .llm import LLMCardEngine, LLMGenerationError
from .services import EvaluationService, ExportService, Repository, ReviewService, RunCoordinator


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
    reviewer = ReviewService(app_settings, repository, llm_engine)
    evaluator = EvaluationService(app_settings, repository, llm_engine)

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
            "discovery_strategies": repository.list_discovery_strategies(run_id=run_id),
            "discovery_results": repository.list_discovery_results(run_id=run_id),
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

    @app.post("/api/evaluations/runs")
    def create_evaluation_run(payload: EvaluationRunRequest) -> dict:
        calibration_set_id = payload.calibration_set_id.strip()
        if not calibration_set_id:
            active_set = repository.get_active_calibration_set()
            if not active_set:
                raise HTTPException(status_code=400, detail="No calibration set id provided and no active calibration set is available")
            calibration_set_id = active_set["id"]
        try:
            evaluation_run = evaluator.run_calibration_set(calibration_set_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except LLMGenerationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"evaluation_run": evaluation_run}

    @app.get("/api/evaluations/runs")
    def list_evaluation_runs(limit: int = Query(default=10, ge=1, le=50)) -> dict:
        return {"evaluation_runs": repository.list_evaluation_runs(limit=limit)}

    @app.get("/api/evaluations/runs/{evaluation_run_id}")
    def get_evaluation_run(evaluation_run_id: str) -> dict:
        evaluation_run = repository.get_evaluation_run(evaluation_run_id)
        if not evaluation_run:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        return {"evaluation_run": evaluation_run}

    @app.get("/api/cards")
    def list_cards(run_id: str = Query(default=""), topic: str = Query(default="")) -> dict:
        cards = repository.list_cards(run_id=run_id or None, topic=topic)
        return {"cards": cards}

    @app.get("/api/review-items")
    def list_review_items(
        run_id: str = Query(default=""),
        topic: str = Query(default=""),
        item_type: str = Query(default="cards"),
        review_status: str = Query(default=""),
        exclusion_type: str = Query(default=""),
    ) -> dict:
        return {
            "items": repository.list_review_items(
                run_id=run_id or None,
                topic=topic,
                item_type=item_type,
                review_status=review_status,
                exclusion_type=exclusion_type,
            )
        }

    @app.get("/api/review-items/{target_type}/{target_id}")
    def get_review_item(target_type: str, target_id: str) -> dict:
        item = repository.get_review_item(target_type, target_id)
        if not item:
            raise HTTPException(status_code=404, detail="Review item not found")
        if target_type == "card":
            item["neighbors"] = repository.build_neighbors(target_id)
        return {"item": item}

    @app.post("/api/review-items/{target_type}/{target_id}/review")
    def review_item(target_type: str, target_id: str, payload: ReviewRequest) -> dict:
        try:
            reviewer.review_item(target_type, target_id, payload.reviewer, payload.decision, payload.note)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "ok", "target_type": target_type, "target_id": target_id}

    @app.post("/api/review-items/excluded/{target_id}/promote")
    def promote_excluded_item(target_id: str, payload: PromoteExcludedRequest) -> dict:
        try:
            promoted = reviewer.promote_excluded_item(target_id, payload.reviewer, payload.note)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, LLMGenerationError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "ok", **promoted}

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
        try:
            reviewer.review_item("card", card_id, payload.reviewer, payload.decision, payload.note)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "ok", "card_id": card_id}

    @app.get("/api/access-queue")
    def list_access_queue(run_id: str = Query(default="")) -> dict:
        return {"items": repository.list_access_queue(run_id or None)}

    @app.post("/api/exports/google-doc")
    def export_google_doc(payload: ExportRequest) -> dict:
        if not payload.card_ids:
            raise HTTPException(status_code=400, detail="At least one card must be selected for export")
        try:
            export_record = exporter.export_google_doc_package(
                run_id=payload.run_id,
                card_ids=payload.card_ids,
                document_title=payload.document_title,
                existing_google_doc_id=payload.existing_google_doc_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
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
            "discovery_strategies": repository.list_discovery_strategies(),
            "discovery_results": repository.list_discovery_results(),
            "calibration_sets": repository.list_calibration_sets(),
            "active_calibration_set": repository.get_active_calibration_set(),
            "latest_evaluation_run": repository.get_latest_evaluation_run(),
        }

    @app.post("/api/llm/smoke")
    def llm_smoke() -> dict:
        try:
            return llm_engine.smoke_test()
        except LLMGenerationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
