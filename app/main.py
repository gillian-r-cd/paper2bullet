"""
This module exposes the FastAPI application for the Paper to Bullet Phase 0 workflow.
Main functions: `create_app()` and HTTP endpoints for runs, cards, reviews, access queue, and exports.
Data structures: app state wiring settings, repository, coordinator, and export service.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .config import Settings, get_settings
from .db import init_db
from .schemas import (
    AccessQueueReactivateRequest,
    CalibrationSetImportRequest,
    EvaluationRunRequest,
    ExportRequest,
    MemoryActivateRequest,
    MemoryDraftRequest,
    PaperQuestionRequest,
    PromoteExcludedRequest,
    ResearchPlanDraftRequest,
    ReviewCommentRequest,
    ReviewRequest,
    RunCreateRequest,
    SearchTermRecommendationRequest,
    SinglePaperValidationRequest,
)
from .llm import LLMCardEngine, LLMGenerationError
from .services import (
    AccessQueueService,
    EvaluationService,
    ExportService,
    PaperQAService,
    PreferenceMemoryService,
    PreferenceMemoryStore,
    Repository,
    ResearchPlanningService,
    ReviewService,
    RunCoordinator,
)


def build_index_html(static_path: Path) -> str:
    return static_path.read_text(encoding="utf-8")


def build_csv_response(filename: str, fieldnames: list[str], rows: list[dict]) -> Response:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    app_settings = settings or get_settings()
    app_settings.ensure_directories()
    init_db(
        app_settings.db_path,
        busy_timeout_seconds=app_settings.sqlite_busy_timeout_seconds,
        journal_mode=app_settings.sqlite_journal_mode,
    )

    repository = Repository(app_settings)
    repository.backfill_missing_publication_years()
    repository.sync_governance_records()
    coordinator = RunCoordinator(app_settings, repository)
    exporter = ExportService(app_settings, repository)
    llm_engine = LLMCardEngine(app_settings)
    memory_store = PreferenceMemoryStore(app_settings)
    planner = ResearchPlanningService(app_settings, llm_engine, memory_store)
    reviewer = ReviewService(app_settings, repository, llm_engine)
    memory_service = PreferenceMemoryService(app_settings, repository, llm_engine, memory_store)
    paper_qa_service = PaperQAService(app_settings, repository, llm_engine, memory_store)
    evaluator = EvaluationService(app_settings, repository, llm_engine)
    access_queue_service = AccessQueueService(app_settings, repository, coordinator)

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
            metadata = dict(payload.metadata or {})
            metadata["task_type"] = payload.task_type
            if payload.research_brief.strip():
                metadata["research_brief"] = payload.research_brief.strip()
                metadata["task_brief"] = payload.research_brief.strip()
            if payload.confirmed_plan:
                metadata["confirmed_plan"] = payload.confirmed_plan
            if payload.use_active_memory:
                active_memory = memory_store.get_active_memory()
                if active_memory:
                    metadata["active_memory_snapshot"] = active_memory
            run = coordinator.create_run(
                topics_text=payload.topics_text,
                metadata=metadata,
                local_pdfs=[item.model_dump() for item in payload.local_pdfs],
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"run": run}

    @app.post("/api/discovery/recommend-search-terms")
    def recommend_search_terms(payload: SearchTermRecommendationRequest) -> dict:
        try:
            recommendation = llm_engine.recommend_search_terms(
                payload.research_goal,
                max_terms=payload.max_terms,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except LLMGenerationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"recommendation": recommendation}

    @app.post("/api/research-plans/draft")
    def draft_research_plan(payload: ResearchPlanDraftRequest) -> dict:
        try:
            plan = planner.draft_plan(
                payload.research_brief,
                requested_task_type=payload.task_type,
                max_terms=payload.max_terms,
                use_active_memory=payload.use_active_memory,
                also_generate_aha_cards=payload.also_generate_aha_cards,
            )
        except (ValueError, LLMGenerationError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"plan": plan}

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
            "matrix_items": repository.list_matrix_items(run_id=run_id),
        }

    @app.get("/api/saturation/topics")
    def list_saturation_topics(topic: str = Query(default=""), history_limit: int = Query(default=5, ge=1, le=20)) -> dict:
        return {
            "trends": repository.list_topic_saturation_trends(topic=topic, history_limit=history_limit),
            "snapshots": repository.list_topic_saturation_snapshots(topic=topic, limit=50),
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

    @app.get("/api/calibration/workflow")
    def get_calibration_workflow() -> dict:
        return repository.get_calibration_workflow_status()

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
    def list_cards(
        run_id: str = Query(default=""),
        topic: str = Query(default=""),
        paper_id: str = Query(default=""),
        topic_id: str = Query(default=""),
    ) -> dict:
        cards = repository.list_cards(
            run_id=run_id or None,
            topic=topic,
            paper_id=paper_id or None,
            topic_id=topic_id or None,
        )
        return {"cards": cards}

    @app.get("/api/matrix-items")
    def list_matrix_items(
        run_id: str = Query(default=""),
        topic: str = Query(default=""),
        paper_id: str = Query(default=""),
        topic_id: str = Query(default=""),
    ) -> dict:
        items = repository.list_matrix_items(
            run_id=run_id or None,
            topic=topic,
            paper_id=paper_id or None,
            topic_id=topic_id or None,
        )
        return {"matrix_items": items}

    @app.get("/api/matrix-items/{matrix_item_id}")
    def get_matrix_item(matrix_item_id: str) -> dict:
        item = repository.get_matrix_item(matrix_item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Matrix item not found")
        return {"matrix_item": item}

    @app.post("/api/papers/{paper_id}/validate-single")
    def validate_single_paper_flow(paper_id: str, payload: SinglePaperValidationRequest) -> dict:
        paper = repository.get_paper(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        topic = repository.get_topic(payload.topic_id)
        if not topic:
            raise HTTPException(status_code=404, detail="Topic not found")
        run = repository.get_run(payload.run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            result = coordinator.pipeline.validate_single_paper_flow(
                paper=paper,
                topic=topic,
                run_id=payload.run_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except LLMGenerationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"validation": result}

    @app.get("/api/papers/{paper_id}/understanding")
    def get_latest_paper_understanding(paper_id: str, topic_id: str = Query(default=""), run_id: str = Query(default="")) -> dict:
        if not topic_id or not run_id:
            raise HTTPException(status_code=400, detail="topic_id and run_id are required")
        record = repository.get_latest_paper_understanding(paper_id, topic_id, run_id)
        if not record:
            raise HTTPException(status_code=404, detail="Paper understanding record not found")
        return {"understanding_record": record}

    @app.get("/api/papers/{paper_id}/card-plan")
    def get_latest_card_plan(paper_id: str, topic_id: str = Query(default=""), run_id: str = Query(default="")) -> dict:
        if not topic_id or not run_id:
            raise HTTPException(status_code=400, detail="topic_id and run_id are required")
        record = repository.get_latest_card_plan(paper_id, topic_id, run_id)
        if not record:
            raise HTTPException(status_code=404, detail="Card plan record not found")
        return {"card_plan": record}

    @app.post("/api/papers/{paper_id}/qa")
    def answer_paper_question(paper_id: str, payload: PaperQuestionRequest) -> dict:
        try:
            answer = paper_qa_service.answer_question(
                paper_id,
                payload.question,
                max_sections=payload.max_sections,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, LLMGenerationError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"answer": answer}

    @app.get("/api/quality/metrics")
    def get_quality_metrics(run_id: str = Query(default="")) -> dict:
        return {"metrics": repository.get_quality_metrics(run_id=run_id or None)}

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

    @app.get("/api/review-items/export.csv")
    def export_review_items_csv(
        run_id: str = Query(default=""),
        topic: str = Query(default=""),
        item_type: str = Query(default="cards"),
        review_status: str = Query(default=""),
        exclusion_type: str = Query(default=""),
    ) -> Response:
        rows = repository.list_review_items(
            run_id=run_id or None,
            topic=topic,
            item_type=item_type,
            review_status=review_status,
            exclusion_type=exclusion_type,
        )
        return build_csv_response(
            "review_items.csv",
            [
                "object_type",
                "object_id",
                "run_id",
                "topic_name",
                "paper_title",
                "publication_year",
                "paper_url",
                "display_title",
                "color",
                "course_transformation",
                "teachable_one_liner",
                "review_status",
                "comment_text",
                "comment_updated_at",
                "exclusion_type",
                "dimension_label",
                "outcome_label",
                "verdict",
                "evidence_strength",
                "export_eligible",
                "promoted_card_id",
                "promoted_card_title",
                "created_at",
            ],
            rows,
        )

    @app.get("/api/review-items/{target_type}/{target_id}")
    def get_review_item(target_type: str, target_id: str) -> dict:
        item = repository.get_review_item(target_type, target_id)
        if not item:
            raise HTTPException(status_code=404, detail="Review item not found")
        if target_type == "card":
            item["neighbors"] = repository.build_neighbors(target_id)
        return {"item": item}

    @app.get("/api/figures/{figure_id}")
    def get_figure(figure_id: str) -> dict:
        figure = repository.get_figure(figure_id)
        if not figure:
            raise HTTPException(status_code=404, detail="Figure not found")
        return {"figure": figure}

    @app.get("/api/figures/{figure_id}/asset")
    def get_figure_asset(figure_id: str):
        figure = repository.get_figure(figure_id)
        if not figure:
            raise HTTPException(status_code=404, detail="Figure not found")
        asset_status = str(figure.get("asset_status", "")).strip()
        asset_local_path = str(figure.get("asset_local_path", "")).strip()
        asset_source_url = str(figure.get("asset_source_url", "")).strip()
        if asset_status == "validated_local_asset" and asset_local_path:
            asset_path = Path(asset_local_path)
            if not asset_path.exists():
                raise HTTPException(status_code=404, detail="Validated figure asset path does not exist")
            media_type = str(figure.get("mime_type", "")).strip() or None
            return FileResponse(asset_path, media_type=media_type)
        if asset_status == "external_reference_only" and asset_source_url.startswith(("http://", "https://")):
            return RedirectResponse(asset_source_url)
        raise HTTPException(
            status_code=404,
            detail=f"Figure asset is not available as a validated local file (status={asset_status or 'unknown'})",
        )

    @app.post("/api/review-items/{target_type}/{target_id}/review")
    def review_item(target_type: str, target_id: str, payload: ReviewRequest) -> dict:
        try:
            reviewer.review_item(target_type, target_id, payload.reviewer, payload.decision, payload.note)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "ok", "target_type": target_type, "target_id": target_id}

    @app.post("/api/review-items/{target_type}/{target_id}/comment")
    def save_review_item_comment(target_type: str, target_id: str, payload: ReviewCommentRequest) -> dict:
        try:
            item = reviewer.save_comment(target_type, target_id, payload.reviewer, payload.comment)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"status": "ok", "item": item}

    @app.post("/api/review-items/excluded/{target_id}/promote")
    def promote_excluded_item(target_id: str, payload: PromoteExcludedRequest) -> dict:
        try:
            promoted = reviewer.promote_excluded_item(target_id, payload.reviewer, payload.note)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, LLMGenerationError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "ok", **promoted}

    @app.get("/api/memory/active")
    def get_active_memory() -> dict:
        return {"active_memory": memory_store.get_active_memory()}

    @app.post("/api/memory/draft")
    def draft_memory(payload: MemoryDraftRequest) -> dict:
        try:
            memory_draft = memory_service.draft_memory(
                task_type=payload.task_type,
                run_id=payload.run_id,
                reviewer=payload.reviewer,
            )
        except (ValueError, LLMGenerationError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"memory_draft": memory_draft}

    @app.post("/api/memory/activate")
    def activate_memory(payload: MemoryActivateRequest) -> dict:
        try:
            activated = memory_service.activate_memory(payload.memory_draft, payload.reviewer)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"active_memory": activated}

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

    @app.get("/api/access-queue/export.csv")
    def export_access_queue_csv(run_id: str = Query(default="")) -> Response:
        rows = repository.list_access_queue(run_id or None)
        return build_csv_response(
            "access_queue.csv",
            [
                "id",
                "paper_id",
                "run_id",
                "paper_title",
                "publication_year",
                "reason",
                "priority",
                "owner",
                "status",
                "original_url",
                "created_at",
            ],
            rows,
        )

    @app.post("/api/access-queue/{queue_item_id}/reactivate")
    def reactivate_access_queue_item(queue_item_id: str, payload: AccessQueueReactivateRequest) -> dict:
        try:
            result = access_queue_service.reactivate_item(queue_item_id, payload.local_path, payload.reviewer)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return result

    @app.post("/api/topic-runs/{topic_run_id}/retry")
    def retry_topic_run(topic_run_id: str) -> dict:
        try:
            topic_run = coordinator.retry_topic_run(topic_run_id)
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"topic_run": topic_run}

    @app.post("/api/exports/google-doc")
    def export_google_doc(payload: ExportRequest) -> dict:
        try:
            if payload.export_kind == "matrix_items":
                if not payload.matrix_item_ids:
                    raise HTTPException(status_code=400, detail="At least one matrix item must be selected for export")
                export_record = exporter.export_matrix_google_doc_package(
                    run_id=payload.run_id,
                    matrix_item_ids=payload.matrix_item_ids,
                    document_title=payload.document_title,
                    existing_google_doc_id=payload.existing_google_doc_id,
                )
            else:
                if not payload.card_ids:
                    raise HTTPException(status_code=400, detail="At least one card must be selected for export")
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
                "sqlite_busy_timeout_seconds": app_settings.sqlite_busy_timeout_seconds,
                "sqlite_journal_mode": app_settings.sqlite_journal_mode,
                "discovery_timeout_seconds": app_settings.discovery_timeout_seconds,
                "remote_asset_timeout_seconds": app_settings.remote_asset_timeout_seconds,
                "stalled_after_seconds": app_settings.stalled_after_seconds,
                "google_docs_mode": app_settings.google_docs_mode,
                "llm_mode": app_settings.llm_mode,
                "llm_model": app_settings.llm_model,
            },
            "runs": repository.list_runs(),
            "topic_runs": repository.list_topic_runs(),
            "cards": repository.list_cards(),
            "matrix_items": repository.list_matrix_items(),
            "access_queue": repository.list_access_queue(),
            "discovery_strategies": repository.list_discovery_strategies(),
            "discovery_results": repository.list_discovery_results(),
            "saturation_trends": repository.list_topic_saturation_trends(history_limit=3),
            "calibration_sets": repository.list_calibration_sets(),
            "active_calibration_set": repository.get_active_calibration_set(),
            "active_memory": memory_store.get_active_memory(),
            "prompt_versions": repository.list_prompt_versions(),
            "rubric_versions": repository.list_rubric_versions(),
            "calibration_workflow": repository.get_calibration_workflow_status(),
            "latest_evaluation_run": repository.get_latest_evaluation_run(),
            "quality_metrics": repository.get_quality_metrics(),
        }

    @app.post("/api/llm/smoke")
    def llm_smoke() -> dict:
        try:
            return llm_engine.smoke_test()
        except LLMGenerationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
