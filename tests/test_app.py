"""
This file contains Phase 0 integration tests for the Paper to Bullet application.
Main tests: local PDF intake, card generation, review actions, and export artifact creation.
Data structures: temporary app settings, a minimal PDF fixture, and API-level assertions.
"""
from __future__ import annotations

import base64
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

import app.services as services_module
from app.config import Settings
from app.llm import (
    CARD_RUBRIC_VERSION,
    EXTRACTION_PROMPT_VERSION,
    JUDGEMENT_PROMPT_VERSION,
    AnthropicLLMClient,
    GeminiLLMClient,
    LLMGenerationError,
    LLMCardEngine,
    OpenAICompatibleLLMClient,
    get_prompt_version_records,
)
from app.main import create_app
from app.db import ensure_migrations, get_connection, init_db
from app.services import EvaluationService, PaperPipeline, PdfParser, Repository, split_paragraphs

ORIGINAL_DISCOVERY_DISCOVER = services_module.DiscoveryService.discover


def build_minimal_pdf_bytes(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 50 100 Td ({escaped}) Tj ET"
    objects = []
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        "3 0 obj\n"
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        "endobj\n"
    )
    objects.append(f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream}\nendstream\nendobj\n")
    objects.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    result = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(result.encode("latin-1")))
        result += obj
    xref_start = len(result.encode("latin-1"))
    result += f"xref\n0 {len(objects) + 1}\n"
    result += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        result += f"{offset:010d} 00000 n \n"
    result += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    return result.encode("latin-1")


def build_corrupted_pdf_bytes() -> bytes:
    return (
        b"%PDF-1.5\n"
        b"132 0 obj\n<< /Type /Page >>\nstream\n"
        b"x\x9c\x03\x00\x00\x00\x00\x01 endstream\n"
        b"endobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 >>\n%%EOF\n"
    )


def build_pdf_with_non_text_tj_noise(valid_text: str) -> bytes:
    escaped = valid_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    text_stream = f"BT /F1 18 Tf 50 100 Td ({escaped}) Tj ET"
    noisy_stream = (
        "stream\n"
        "(JFIF) Tj\n"
        "(MATLAB Handle Graphics) Tj\n"
        "endstream\n"
    )
    objects = []
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        "3 0 obj\n"
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        "endobj\n"
    )
    objects.append(f"4 0 obj\n<< /Length {len(text_stream)} >>\nstream\n{text_stream}\nendstream\nendobj\n")
    objects.append("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(f"6 0 obj\n<< /Length {len(noisy_stream)} >>\n{noisy_stream}endobj\n")

    result = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(result.encode("latin-1")))
        result += obj
    xref_start = len(result.encode("latin-1"))
    result += f"xref\n0 {len(objects) + 1}\n"
    result += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        result += f"{offset:010d} 00000 n \n"
    result += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    return result.encode("latin-1")


def build_png_bytes(size: tuple[int, int] = (24, 16), color: tuple[int, int, int] = (60, 120, 220)) -> bytes:
    from PIL import Image

    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_pdf_with_embedded_png(image_bytes: bytes, caption_text: str) -> bytes:
    import fitz

    document = fitz.open()
    try:
        page = document.new_page(width=400, height=300)
        page.insert_image(fitz.Rect(50, 40, 250, 180), stream=image_bytes)
        page.insert_text((50, 230), caption_text, fontsize=12)
        return document.tobytes()
    finally:
        document.close()


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class PhaseZeroWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.settings = Settings(
            data_dir=base / "data",
            db_path=base / "data" / "paper2bullet.sqlite3",
            max_workers=4,
            google_docs_mode="artifact_only",
            llm_mode="disabled",
        )
        self.discovery_patcher = patch.object(services_module.DiscoveryService, "discover", return_value=[])
        self.discovery_patcher.start()
        self.client = TestClient(create_app(self.settings))
        init_db(self.settings.db_path)
        self.repository = Repository(self.settings)

    def tearDown(self) -> None:
        self.discovery_patcher.stop()
        self.temp_dir.cleanup()

    def _create_excluded_review_fixture(self) -> dict:
        run = self.repository.create_run("Context Engineering", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("Context Engineering")
        paper = self.repository.create_or_get_paper(
            title="Standards in Practice",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::excluded::fixture",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": "section_excluded_1",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "A retired standard can remain in everyday use after a formal replacement appears.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )
        self.repository.replace_generation_outputs_for_paper_topic(
            paper["id"],
            topic["id"],
            run["id"],
            [],
            [
                {
                    "id": "excluded_promote",
                    "label": "分类法回顾",
                    "exclusion_type": "summary",
                    "reason": "这只是分类梳理，不足以形成学员 aha。",
                    "section_ids": ["section_excluded_1"],
                    "created_at": "2026-03-06T00:00:01+00:00",
                }
            ],
        )
        return {"run": run, "topic": topic, "paper": paper}

    def _create_export_card_fixture(
        self,
        *,
        card_id: str,
        review_decision: str | None,
        run: dict | None = None,
        original_url: str = "",
        figure_ids: list[str] | None = None,
    ) -> dict:
        figure_ids = figure_ids or []
        run = run or self.repository.create_run("Export Topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("Export Topic")
        paper = self.repository.create_or_get_paper(
            title=f"Export Paper {card_id}",
            authors=["Test Author"],
            publication_year=2026,
            external_id=f"paper::export::{card_id}",
            source_type="local",
            local_path="",
            original_url=original_url,
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": f"section_{card_id}",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "The coordinator only improves quality when it explicitly checks conflicts between agents.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )
        self.repository.replace_generation_outputs_for_paper_topic(
            paper["id"],
            topic["id"],
            run["id"],
            [
                {
                    "id": card_id,
                    "title": f"{card_id} 标题",
                    "granularity_level": "detail",
                    "course_transformation": "多智能体协作：冲突校验案例",
                    "teachable_one_liner": "显式冲突校验，比盲目加 agent 更重要。",
                    "draft_body": "这条卡片说明编排步骤里的冲突检查比 agent 数量更决定质量。",
                    "evidence": [
                        {
                            "section_id": f"section_{card_id}",
                            "quote": "The coordinator only improves quality when it explicitly checks conflicts between agents.",
                            "page_number": 1,
                            "analysis": "这段证据说明真正有效的是冲突检查，而不是堆更多角色。",
                        }
                    ],
                    "figure_ids": figure_ids,
                    "status": "candidate",
                    "embedding": [0.0] * 64,
                    "created_at": "2026-03-06T00:00:02+00:00",
                    "judgement": {
                        "color": "green",
                        "reason": "这条结论有明确证据，也能直接转成课程上的方法提醒。",
                        "model_version": "stub-model",
                        "prompt_version": JUDGEMENT_PROMPT_VERSION,
                        "rubric_version": CARD_RUBRIC_VERSION,
                    },
                }
            ],
            [],
        )
        if review_decision:
            self.repository.create_review_decision("card", card_id, "tester", review_decision, f"review:{review_decision}")
        return {"run": run, "topic": topic, "paper": paper, "card_id": card_id}

    def test_local_pdf_to_export_flow(self) -> None:
        source_pdf = Path(self.temp_dir.name) / "adaptive-selling.pdf"
        source_pdf.write_bytes(
            build_minimal_pdf_bytes(
                "We find that adaptive selling beats rigid scripts by 24 percent in perceived customer fit."
            )
        )
        test_case = self

        class StubClient:
            model = "stub-api-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                prompt = json.loads(user_prompt)
                if prompt["stage"] == "candidate_extraction":
                    section_id = prompt["sections"][0]["section_id"]
                    test_case.assertEqual(prompt["output_language"], "zh-CN")
                    return {
                        "cards": [
                            {
                                "title": "适应性销售会明显提升客户感知匹配度",
                                "section_ids": [section_id],
                                "granularity_level": "detail",
                                "draft_body": "当客户匹配度重要时，适应性销售比僵硬话术更值得教。",
                                "evidence_analysis": [
                                    {
                                        "section_id": section_id,
                                        "analysis": "这给出了一个可教学的对比：适应性销售不是空泛技巧，而是会改变客户感知结果。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "适应性销售会明显提升客户感知匹配度",
                            "course_transformation": "适应性销售：可直接讲给学员的证据型论点",
                            "teachable_one_liner": "不是所有销售都该背统一话术，真正拉开差距的是根据客户调整表达。",
                            "draft_body": "当客户匹配度重要时，适应性销售比僵硬话术更值得教。",
                            "evidence_localization": [
                                {
                                    "section_id": prompt["candidates"][0]["evidence"][0]["section_id"],
                                    "quote_zh": "研究发现，当客户感知匹配度很重要时，适应性销售比僵硬统一的话术更有效；这种差异不只是沟通风格不同，而是会直接改变客户对销售互动是否贴合自身需求的判断。",
                                }
                            ],
                            "judgement": {
                                "color": "green",
                                "reason": "这条发现既有明确证据，又能直接转成课程中的销售判断原则。",
                            },
                        }
                    ]
                }

        with patch.object(services_module.LLMCardEngine, "_build_client", return_value=StubClient()):
            client = TestClient(create_app(self.settings))

            create_response = client.post(
                "/api/runs",
                json={
                    "topics_text": "adaptive selling\nsales coaching",
                    "metadata": {"operator": "test"},
                    "local_pdfs": [{"path": str(source_pdf), "topics": ["adaptive selling"]}],
                },
            )
            self.assertEqual(create_response.status_code, 200, create_response.text)
            run_id = create_response.json()["run"]["id"]

            for _ in range(30):
                run_response = client.get(f"/api/runs/{run_id}")
                self.assertEqual(run_response.status_code, 200, run_response.text)
                if run_response.json()["run"]["status"] in {"completed", "partial_failed", "failed"}:
                    break
                time.sleep(0.2)

            cards_response = client.get("/api/cards", params={"run_id": run_id})
            self.assertEqual(cards_response.status_code, 200, cards_response.text)
            cards = cards_response.json()["cards"]
            self.assertGreaterEqual(len(cards), 1)

            card_id = cards[0]["id"]
            detail_response = client.get(f"/api/cards/{card_id}")
            self.assertEqual(detail_response.status_code, 200, detail_response.text)
            self.assertIn("neighbors", detail_response.json()["card"])
            self.assertIn("paper_url", detail_response.json()["card"])
            self.assertEqual(
                detail_response.json()["card"]["evidence"][0]["quote_zh"],
                "研究发现，当客户感知匹配度很重要时，适应性销售比僵硬统一的话术更有效；这种差异不只是沟通风格不同，而是会直接改变客户对销售互动是否贴合自身需求的判断。",
            )

            review_response = client.post(
                f"/api/cards/{card_id}/review",
                json={"reviewer": "tester", "decision": "accepted", "note": "Looks good."},
            )
            self.assertEqual(review_response.status_code, 200, review_response.text)

            export_response = client.post(
                "/api/exports/google-doc",
                json={
                    "run_id": run_id,
                    "card_ids": [card_id],
                    "document_title": "Boss Report - Adaptive Selling",
                    "existing_google_doc_id": "",
                },
            )
            self.assertEqual(export_response.status_code, 200, export_response.text)
            export_payload = export_response.json()["export"]
            self.assertEqual(export_payload["export_status"], "artifact_only")
            self.assertTrue(Path(export_payload["artifact_path"]).exists())
            self.assertTrue(Path(export_payload["json_path"]).exists())
            markdown_text = Path(export_payload["artifact_path"]).read_text(encoding="utf-8")
            export_json = json.loads(Path(export_payload["json_path"]).read_text(encoding="utf-8"))
            self.assertIn("它在课程里变成什么", markdown_text)
            self.assertIn("可直接讲的一句话", markdown_text)
            self.assertIn("原文（穿插分析）：", markdown_text)
            self.assertIn("*→", markdown_text)
            self.assertEqual(export_json["requested_card_ids"], [card_id])
            self.assertEqual(export_json["selection_snapshot"][0]["card_id"], card_id)
            self.assertEqual(export_json["selection_snapshot"][0]["review_status"], "accepted")
            self.assertEqual(export_json["review_snapshot"][0]["decision"], "accepted")

    def test_homepage_prefills_safe_metadata_default(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('<textarea id="metadata" rows="4">{}</textarea>', response.text)
        self.assertIn('id="filter-item-type"', response.text)
        self.assertIn("/api/review-items", response.text)
        self.assertIn('id="refresh-calibration-status"', response.text)
        self.assertIn('id="refresh-saturation-trends"', response.text)
        self.assertIn('id="saturation-trends"', response.text)
        self.assertIn("Promote", response.text)
        self.assertIn('id="export-card-picker"', response.text)
        self.assertIn('<select id="export-run-id">', response.text)
        self.assertNotIn('id="review-item-detail"', response.text)
        self.assertIn("/api/topic-runs/", response.text)

    def test_can_import_and_activate_calibration_set_via_api(self) -> None:
        payload = {
            "name": "core-aha-boundaries-v1",
            "description": "Initial positive, negative, and boundary examples.",
            "metadata": {"owner": "test"},
            "examples": [
                {
                    "example_type": "positive",
                    "topic_name": "Context Engineering",
                    "audience": "AI literacy learners",
                    "title": "Legacy standards can outlive their official replacement in practice",
                    "source_text": "Practitioners still use retired standards even after successors are published.",
                    "evidence": [{"quote": "A retired standard remains widely referenced."}],
                    "expected_cards": [{"title": "Legacy-standard inertia is real"}],
                    "expected_exclusions": [],
                    "rationale": "This creates a practical learner-facing insight.",
                    "tags": ["positive", "weak-transfer-guard"],
                },
                {
                    "example_type": "negative",
                    "topic_name": "Context Engineering",
                    "audience": "AI literacy learners",
                    "title": "Background theory taxonomy",
                    "source_text": "The paper surveys several classification schemes.",
                    "evidence": [{"quote": "Several prior taxonomies are reviewed."}],
                    "expected_cards": [],
                    "expected_exclusions": [{"label": "Taxonomy recap", "exclusion_type": "summary"}],
                    "rationale": "Useful background, but not an aha moment.",
                    "tags": ["negative", "summary"],
                },
            ],
        }

        import_response = self.client.post("/api/calibration/sets/import", json=payload)
        self.assertEqual(import_response.status_code, 200, import_response.text)
        calibration_set = import_response.json()["calibration_set"]
        self.assertEqual(calibration_set["name"], "core-aha-boundaries-v1")
        self.assertEqual(len(calibration_set["examples"]), 2)

        sets_response = self.client.get("/api/calibration/sets")
        self.assertEqual(sets_response.status_code, 200, sets_response.text)
        self.assertEqual(sets_response.json()["active_set"], None)
        self.assertEqual(len(sets_response.json()["sets"]), 1)
        self.assertEqual(sets_response.json()["sets"][0]["example_count"], 2)

        activate_response = self.client.post(f"/api/calibration/sets/{calibration_set['id']}/activate")
        self.assertEqual(activate_response.status_code, 200, activate_response.text)
        self.assertEqual(activate_response.json()["calibration_set"]["status"], "active")

        active_response = self.client.get("/api/calibration/sets")
        self.assertEqual(active_response.status_code, 200, active_response.text)
        self.assertIsNotNone(active_response.json()["active_set"])
        self.assertEqual(active_response.json()["active_set"]["id"], calibration_set["id"])

    def test_calibration_workflow_exposes_active_versions_and_boundary_failures(self) -> None:
        calibration_set = self.repository.import_calibration_set(
            name="governance-pack",
            description="Boundary-oriented governance examples.",
            metadata={"owner": "ops"},
            examples=[
                {
                    "example_type": "positive",
                    "topic_name": "AI literacy",
                    "audience": "teachers",
                    "title": "Positive Example",
                    "source_text": "A positive calibration example with a clear teaching insight.",
                    "evidence": [],
                    "expected_cards": [],
                    "expected_exclusions": [],
                    "rationale": "",
                    "tags": ["positive"],
                },
                {
                    "example_type": "boundary",
                    "topic_name": "AI literacy",
                    "audience": "teachers",
                    "title": "Boundary Example",
                    "source_text": "A boundary calibration example that may pass or fail depending on rubric details.",
                    "evidence": [],
                    "expected_cards": [],
                    "expected_exclusions": [],
                    "rationale": "Needs careful judgment.",
                    "tags": ["boundary", "review"],
                },
            ],
        )
        self.repository.activate_calibration_set(calibration_set["id"])
        evaluation_run = self.repository.create_evaluation_run(
            calibration_set=calibration_set,
            llm_mode="disabled",
            model_name="",
            extraction_prompt_version=EXTRACTION_PROMPT_VERSION,
            judgement_prompt_version=JUDGEMENT_PROMPT_VERSION,
            rubric_version=CARD_RUBRIC_VERSION,
        )
        boundary_example = calibration_set["examples"][1]
        self.repository.create_evaluation_result(
            evaluation_run_id=evaluation_run["id"],
            calibration_example=boundary_example,
            extraction_output={},
            judgement_output={},
            expected={},
            actual={},
            verdict="failed",
            regression_type="boundary_miss",
            reason="Boundary example regressed.",
        )
        self.repository.finalize_evaluation_run(
            evaluation_run["id"],
            "completed",
            {"passed": 0, "failed": 1},
        )

        response = self.client.get("/api/calibration/workflow")
        self.assertEqual(response.status_code, 200, response.text)
        workflow = response.json()
        self.assertEqual(workflow["active_calibration_set"]["id"], calibration_set["id"])
        self.assertEqual(workflow["example_counts"]["positive"], 1)
        self.assertEqual(workflow["example_counts"]["boundary"], 1)
        self.assertGreaterEqual(len(workflow["active_prompt_versions"]), 4)
        self.assertTrue(any(item["version"] == EXTRACTION_PROMPT_VERSION for item in workflow["active_prompt_versions"]))
        self.assertEqual(workflow["active_rubric_versions"][0]["version"], CARD_RUBRIC_VERSION)
        self.assertEqual(len(workflow["failed_examples"]), 1)
        self.assertEqual(workflow["failed_boundary_examples"][0]["title"], "Boundary Example")

    def test_debug_state_exposes_governance_records(self) -> None:
        response = self.client.get("/api/debug/state")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(any(item["version"] == EXTRACTION_PROMPT_VERSION for item in payload["prompt_versions"]))
        self.assertTrue(any(item["version"] == CARD_RUBRIC_VERSION for item in payload["rubric_versions"]))

    def test_calibration_import_rejects_invalid_example_type(self) -> None:
        response = self.client.post(
            "/api/calibration/sets/import",
            json={
                "name": "bad-set",
                "description": "",
                "metadata": {},
                "examples": [
                    {
                        "example_type": "maybe",
                        "topic_name": "LLM agent",
                        "title": "Bad example type",
                        "source_text": "invalid",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("example_type", response.text)

    def test_evaluation_service_persists_run_and_results(self) -> None:
        calibration_set = self.repository.import_calibration_set(
            name="evaluation-core-v1",
            description="Examples for evaluation service tests.",
            metadata={},
            examples=[
                {
                    "example_type": "positive",
                    "topic_name": "Context Engineering",
                    "audience": "operators",
                    "title": "Legacy standards persist",
                    "source_text": "Old standards can remain active in real workflows after an official replacement appears.",
                    "evidence": [{"quote": "Teams still rely on a retired standard."}],
                    "expected_cards": [{"title": "遗留标准惯性"}],
                    "expected_exclusions": [],
                    "rationale": "This should become a learner-facing card.",
                    "tags": ["positive"],
                },
                {
                    "example_type": "negative",
                    "topic_name": "Context Engineering",
                    "audience": "operators",
                    "title": "Taxonomy recap",
                    "source_text": "The paper reviews several existing taxonomies without adding a sharp practical insight.",
                    "evidence": [{"quote": "Several prior taxonomies are summarized."}],
                    "expected_cards": [],
                    "expected_exclusions": [{"label": "分类法回顾", "exclusion_type": "summary"}],
                    "rationale": "This should stay out of the card set.",
                    "tags": ["summary"],
                },
            ],
        )

        class StubClient:
            model = "stub-eval-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                if payload["stage"] == "candidate_extraction":
                    if payload["paper_title"] == "Legacy standards persist":
                        section_id = payload["sections"][0]["section_id"]
                        return {
                            "cards": [
                                {
                                    "title": "旧标准会在正式替换后继续支配真实流程",
                                    "section_ids": [section_id],
                                    "granularity_level": "detail",
                                    "draft_body": "正式替换和现场迁移之间往往隔着很长的惯性期。",
                                    "evidence_analysis": [
                                        {
                                            "section_id": section_id,
                                            "analysis": "这段证据说明制度变了，但流程现场不会同步立刻变。",
                                        }
                                    ],
                                }
                            ],
                            "excluded_content": [],
                        }
                    section_id = payload["sections"][0]["section_id"]
                    return {
                        "cards": [],
                        "excluded_content": [
                            {
                                "label": "分类法回顾",
                                "section_ids": [section_id],
                                "exclusion_type": "summary",
                                "reason": "这只是分类整理，不足以形成学员认知跃迁。",
                            }
                        ],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "旧标准会在正式替换后继续支配真实流程",
                            "course_transformation": "遗留标准迁移：流程惯性案例",
                            "teachable_one_liner": "制度文件换了，不等于一线流程第二天就真的换了。",
                            "draft_body": "正式替换和现场迁移之间往往隔着很长的惯性期。",
                            "evidence_localization": [
                                {
                                    "section_id": payload["candidates"][0]["evidence"][0]["section_id"],
                                    "quote_zh": "一项已经退役的标准，在正式替代方案出现之后，仍然可能继续留在日常实际使用中；这说明制度文本层面的替换和组织现场层面的迁移并不会同步发生。",
                                }
                            ],
                            "judgement": {
                                "color": "yellow",
                                "reason": "这是一张很适合课堂讨论的边界型现实惯性案例。",
                            },
                        }
                    ]
                }

        evaluator = EvaluationService(self.settings, self.repository, card_engine=LLMCardEngine(self.settings, client=StubClient()))
        evaluation_run = evaluator.run_calibration_set(calibration_set["id"])

        self.assertEqual(evaluation_run["status"], "completed")
        self.assertEqual(evaluation_run["summary"]["total_examples"], 2)
        self.assertEqual(evaluation_run["summary"]["passed_examples"], 2)
        self.assertEqual(len(evaluation_run["results"]), 2)
        self.assertEqual(evaluation_run["results"][0]["actual"]["judged_card_count"], 1)
        self.assertEqual(evaluation_run["results"][1]["actual"]["excluded_count"], 1)

    def test_evaluation_api_uses_active_calibration_set_when_not_specified(self) -> None:
        calibration_set = self.repository.import_calibration_set(
            name="evaluation-api-v1",
            description="Examples for evaluation API tests.",
            metadata={},
            examples=[
                {
                    "example_type": "positive",
                    "topic_name": "Context Engineering",
                    "audience": "operators",
                    "title": "Legacy standard inertia",
                    "source_text": "A retired standard can remain in everyday use after a formal replacement appears.",
                    "evidence": [{"quote": "The old standard remains widely referenced."}],
                    "expected_cards": [{"title": "遗留标准惯性"}],
                    "expected_exclusions": [],
                    "rationale": "Useful learner-facing pattern.",
                    "tags": ["positive"],
                }
            ],
        )
        self.repository.activate_calibration_set(calibration_set["id"])

        class StubClient:
            model = "stub-eval-api-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                if payload["stage"] == "candidate_extraction":
                    section_id = payload["sections"][0]["section_id"]
                    return {
                        "cards": [
                            {
                                "title": "旧标准退役后仍会长期留在实际流程里",
                                "section_ids": [section_id],
                                "granularity_level": "detail",
                                "draft_body": "这暴露了制度更新和实际执行之间的惯性差。",
                                "evidence_analysis": [
                                    {
                                        "section_id": section_id,
                                        "analysis": "这条证据能直接拿来讲流程迁移为什么总比制度文件慢。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "旧标准退役后仍会长期留在实际流程里",
                            "course_transformation": "遗留标准迁移：现实阻力案例",
                            "teachable_one_liner": "官方替换完成，不代表现场替换完成。",
                            "draft_body": "这暴露了制度更新和实际执行之间的惯性差。",
                            "evidence_localization": [
                                {
                                    "section_id": payload["candidates"][0]["evidence"][0]["section_id"],
                                    "quote_zh": "一项已经退役的标准，在正式替代方案出现之后，仍然可能继续存在于日常工作流程中；也就是说，制度上的更新并不会自动带来现场执行层面的同步更新。",
                                }
                            ],
                            "judgement": {
                                "color": "yellow",
                                "reason": "这是一条边界型但很适合教学的现实惯性洞见。",
                            },
                        }
                    ]
                }

        with patch.object(services_module.LLMCardEngine, "_build_client", return_value=StubClient()):
            client = TestClient(create_app(self.settings))
            response = client.post("/api/evaluations/runs", json={})

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()["evaluation_run"]
        self.assertEqual(payload["calibration_set_name"], "evaluation-api-v1")
        self.assertEqual(payload["summary"]["passed_examples"], 1)
        debug_state = self.client.get("/api/debug/state")
        self.assertEqual(debug_state.status_code, 200, debug_state.text)
        self.assertEqual(debug_state.json()["latest_evaluation_run"]["id"], payload["id"])

    def test_evaluate_calibration_set_script_reports_provider_error_without_live_llm(self) -> None:
        calibration_set = self.repository.import_calibration_set(
            name="evaluation-script-v1",
            description="Script invocation test.",
            metadata={},
            examples=[
                {
                    "example_type": "positive",
                    "topic_name": "Context Engineering",
                    "audience": "operators",
                    "title": "Legacy standard inertia",
                    "source_text": "A retired standard can remain in everyday use after a formal replacement appears.",
                    "evidence": [{"quote": "The old standard remains widely referenced."}],
                    "expected_cards": [{"title": "遗留标准惯性"}],
                    "expected_exclusions": [],
                    "rationale": "Useful learner-facing pattern.",
                    "tags": ["positive"],
                }
            ],
        )
        self.repository.activate_calibration_set(calibration_set["id"])

        env = os.environ.copy()
        env["P2B_DATA_DIR"] = str(self.settings.data_dir)
        env["P2B_DB_PATH"] = str(self.settings.db_path)
        env["P2B_LLM_MODE"] = "disabled"
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_calibration_set.py", "--use-active"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("LLM provider is not enabled", result.stdout)

    def test_review_decisions_migration_backfills_target_columns(self) -> None:
        legacy_db_path = Path(self.temp_dir.name) / "legacy-review.sqlite3"
        connection = get_connection(legacy_db_path)
        try:
            connection.executescript(
                """
                CREATE TABLE papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL
                );
                CREATE TABLE candidate_cards (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    granularity_level TEXT NOT NULL,
                    course_transformation TEXT NOT NULL,
                    draft_body TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    figure_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE review_decisions (
                    id TEXT PRIMARY KEY,
                    card_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                INSERT INTO review_decisions(id, card_id, reviewer, decision, note, created_at)
                VALUES ('review_1', 'card_123', 'tester', 'accepted', 'legacy row', '2026-03-06T00:00:00+00:00');
                """
            )
            ensure_migrations(connection)
            row = connection.execute("SELECT * FROM review_decisions WHERE id = 'review_1'").fetchone()
        finally:
            connection.close()

        self.assertEqual(row["target_type"], "card")
        self.assertEqual(row["target_id"], "card_123")
        self.assertEqual(row["card_id"], "card_123")

    def test_review_items_api_supports_card_and_excluded_targets(self) -> None:
        run = self.repository.create_run("Context Engineering", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("Context Engineering")
        paper = self.repository.create_or_get_paper(
            title="Standards in Practice",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::review-items",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": "section_review",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "A retired standard can remain in everyday use after a formal replacement appears.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )
        self.repository.replace_generation_outputs_for_paper_topic(
            paper["id"],
            topic["id"],
            run["id"],
            [
                {
                    "id": "card_review",
                    "title": "退役标准仍会留在真实流程里",
                    "granularity_level": "detail",
                    "course_transformation": "遗留标准迁移：现实阻力案例",
                    "teachable_one_liner": "制度文件更新了，不代表流程现场就同步更新了。",
                    "draft_body": "这条卡片强调的是流程惯性。",
                    "evidence": [
                        {
                            "section_id": "section_review",
                            "quote": "A retired standard can remain in everyday use after a formal replacement appears.",
                            "page_number": 1,
                            "analysis": "这条证据解释了为什么正式替换和实际执行会脱节。",
                        }
                    ],
                    "figure_ids": [],
                    "status": "candidate",
                    "embedding": [0.0] * 64,
                    "created_at": "2026-03-06T00:00:00+00:00",
                    "judgement": {
                        "color": "yellow",
                        "reason": "边界型但有教学价值。",
                        "model_version": "stub-model",
                        "prompt_version": JUDGEMENT_PROMPT_VERSION,
                        "rubric_version": CARD_RUBRIC_VERSION,
                    },
                }
            ],
            [
                {
                    "id": "excluded_review",
                    "label": "分类法回顾",
                    "exclusion_type": "summary",
                    "reason": "这只是分类梳理，不足以形成学员 aha。",
                    "section_ids": ["section_review"],
                    "created_at": "2026-03-06T00:00:01+00:00",
                }
            ],
        )
        self.repository.create_review_decision("card", "card_review", "tester", "accepted", "通过")
        self.repository.create_review_decision("excluded", "excluded_review", "tester", "reopened", "重新检查")

        list_response = self.client.get(
            "/api/review-items",
            params={"run_id": run["id"], "item_type": "both"},
        )
        self.assertEqual(list_response.status_code, 200, list_response.text)
        items = list_response.json()["items"]
        self.assertEqual({item["object_type"] for item in items}, {"card", "excluded"})

        excluded_detail = self.client.get("/api/review-items/excluded/excluded_review")
        self.assertEqual(excluded_detail.status_code, 200, excluded_detail.text)
        excluded_item = excluded_detail.json()["item"]
        self.assertEqual(excluded_item["review"]["decision"], "reopened")
        self.assertEqual(excluded_item["evidence_sections"][0]["id"], "section_review")

        review_response = self.client.post(
            "/api/review-items/excluded/excluded_review/review",
            json={"reviewer": "tester", "decision": "accepted", "note": "确认继续排除"},
        )
        self.assertEqual(review_response.status_code, 200, review_response.text)
        refreshed_excluded = self.client.get("/api/review-items/excluded/excluded_review").json()["item"]
        self.assertEqual(refreshed_excluded["review"]["decision"], "accepted")

    def test_card_review_rejects_excluded_only_decision(self) -> None:
        fixture = self._create_export_card_fixture(card_id="card_review_enum", review_decision=None)
        response = self.client.post(
            f"/api/cards/{fixture['card_id']}/review",
            json={"reviewer": "tester", "decision": "reopened", "note": ""},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Unsupported decision for card", response.text)

    def test_excluded_review_rejects_card_only_decision(self) -> None:
        self._create_excluded_review_fixture()
        response = self.client.post(
            "/api/review-items/excluded/excluded_promote/review",
            json={"reviewer": "tester", "decision": "rejected", "note": ""},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Unsupported decision for excluded", response.text)

    def test_review_request_rejects_unknown_decision_value(self) -> None:
        fixture = self._create_export_card_fixture(card_id="card_bad_review", review_decision=None)
        response = self.client.post(
            f"/api/cards/{fixture['card_id']}/review",
            json={"reviewer": "tester", "decision": "maybe", "note": ""},
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_card_and_review_items_expose_paper_link(self) -> None:
        fixture = self._create_export_card_fixture(
            card_id="card_paper_link",
            review_decision="accepted",
            original_url="https://example.com/paper-link",
        )
        card_response = self.client.get(f"/api/cards/{fixture['card_id']}")
        self.assertEqual(card_response.status_code, 200, card_response.text)
        self.assertEqual(card_response.json()["card"]["paper_url"], "https://example.com/paper-link")

        review_items = self.client.get(
            "/api/review-items",
            params={"run_id": fixture["run"]["id"], "item_type": "cards"},
        )
        self.assertEqual(review_items.status_code, 200, review_items.text)
        self.assertEqual(review_items.json()["items"][0]["paper_url"], "https://example.com/paper-link")

    def test_cards_api_supports_paper_and_topic_id_filters(self) -> None:
        run = self.repository.create_run("Validation Topic A\nValidation Topic B", {"operator": "tester"})
        topic_a = self.repository.create_or_get_topic("Validation Topic A")
        topic_b = self.repository.create_or_get_topic("Validation Topic B")
        paper_a = self.repository.create_or_get_paper(
            title="Validation Paper A",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::validation-filter-a",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        paper_b = self.repository.create_or_get_paper(
            title="Validation Paper B",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::validation-filter-b",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper_a["id"], topic_a["id"], run["id"], "local_pdf")
        self.repository.link_paper_to_topic(paper_b["id"], topic_b["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper_a["id"],
            [
                {
                    "id": "section_validation_filter_a",
                    "section_order": 1,
                    "section_title": "Results",
                    "paragraph_text": "Topic A result.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )
        self.repository.replace_sections(
            paper_b["id"],
            [
                {
                    "id": "section_validation_filter_b",
                    "section_order": 1,
                    "section_title": "Results",
                    "paragraph_text": "Topic B result.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )
        self.repository.replace_generation_outputs_for_paper_topic(
            paper_a["id"],
            topic_a["id"],
            run["id"],
            [
                {
                    "id": "card_validation_filter_a",
                    "title": "Validation filter A",
                    "granularity_level": "detail",
                    "course_transformation": "A",
                    "teachable_one_liner": "A",
                    "draft_body": "A",
                    "evidence": [
                        {
                            "section_id": "section_validation_filter_a",
                            "quote": "Topic A result.",
                            "page_number": 1,
                            "analysis": "A",
                        }
                    ],
                    "figure_ids": [],
                    "status": "candidate",
                    "embedding": [0.0] * 64,
                    "created_at": "2026-03-06T00:00:03+00:00",
                    "judgement": {
                        "color": "green",
                        "reason": "A",
                        "model_version": "stub-model",
                        "prompt_version": JUDGEMENT_PROMPT_VERSION,
                        "rubric_version": CARD_RUBRIC_VERSION,
                    },
                }
            ],
            [],
        )
        self.repository.replace_generation_outputs_for_paper_topic(
            paper_b["id"],
            topic_b["id"],
            run["id"],
            [
                {
                    "id": "card_validation_filter_b",
                    "title": "Validation filter B",
                    "granularity_level": "detail",
                    "course_transformation": "B",
                    "teachable_one_liner": "B",
                    "draft_body": "B",
                    "evidence": [
                        {
                            "section_id": "section_validation_filter_b",
                            "quote": "Topic B result.",
                            "page_number": 1,
                            "analysis": "B",
                        }
                    ],
                    "figure_ids": [],
                    "status": "candidate",
                    "embedding": [0.0] * 64,
                    "created_at": "2026-03-06T00:00:04+00:00",
                    "judgement": {
                        "color": "yellow",
                        "reason": "B",
                        "model_version": "stub-model",
                        "prompt_version": JUDGEMENT_PROMPT_VERSION,
                        "rubric_version": CARD_RUBRIC_VERSION,
                    },
                }
            ],
            [],
        )

        response = self.client.get(
            "/api/cards",
            params={"run_id": run["id"], "paper_id": paper_a["id"], "topic_id": topic_a["id"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        cards = response.json()["cards"]
        self.assertEqual([card["id"] for card in cards], ["card_validation_filter_a"])

    def test_promote_excluded_item_creates_linked_candidate_card(self) -> None:
        self._create_excluded_review_fixture()
        test_case = self

        class StubClient:
            model = "stub-promote-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                prompt = json.loads(user_prompt)
                if prompt["stage"] == "candidate_extraction":
                    section_id = prompt["sections"][0]["section_id"]
                    test_case.assertEqual(prompt["output_language"], "zh-CN")
                    return {
                        "cards": [
                            {
                                "title": "旧标准退役后仍会长期留在实际流程里",
                                "section_ids": [section_id],
                                "granularity_level": "detail",
                                "draft_body": "这条内容指出制度替换和现场替换之间总有惯性差。",
                                "evidence_analysis": [
                                    {
                                        "section_id": section_id,
                                        "analysis": "这段证据能直接解释为什么正式标准更新后，业务现场仍会滞后。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "旧标准退役后仍会长期留在实际流程里",
                            "course_transformation": "遗留标准迁移：现实阻力案例",
                            "teachable_one_liner": "官方替换完成，不代表现场替换完成。",
                            "draft_body": "这暴露了制度更新和实际执行之间的惯性差。",
                            "evidence_localization": [
                                {
                                    "section_id": prompt["candidates"][0]["evidence"][0]["section_id"],
                                    "quote_zh": "一项已经退役的标准，在正式替代方案出现之后，仍然可能继续存在于日常工作流程中；这恰好说明流程现场的替换节奏通常会慢于制度文本的替换节奏。",
                                }
                            ],
                            "judgement": {
                                "color": "yellow",
                                "reason": "这是一条边界型但非常适合教学的现实惯性洞见。",
                            },
                        }
                    ]
                }

        with patch.object(services_module.LLMCardEngine, "_build_client", return_value=StubClient()):
            client = TestClient(create_app(self.settings))
            response = client.post(
                "/api/review-items/excluded/excluded_promote/promote",
                json={"reviewer": "tester", "note": "重新抽取为候选卡片"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        promoted_card = payload["card"]
        excluded_item = payload["excluded_item"]
        self.assertEqual(excluded_item["review"]["decision"], "reopened")
        self.assertEqual(excluded_item["promoted_card"]["id"], promoted_card["id"])
        self.assertEqual(promoted_card["source_excluded_item"]["id"], "excluded_promote")

        stored_card = self.repository.get_card(promoted_card["id"])
        self.assertEqual(stored_card["source_excluded_content_id"], "excluded_promote")
        self.assertEqual(self.repository.get_promoted_card_summary("excluded_promote")["id"], promoted_card["id"])

        review_items = client.get("/api/review-items", params={"run_id": stored_card["run_id"], "item_type": "both"}).json()["items"]
        promoted_review_row = next(item for item in review_items if item["object_id"] == promoted_card["id"])
        excluded_review_row = next(item for item in review_items if item["object_id"] == "excluded_promote")
        self.assertTrue(promoted_review_row["promoted_from_excluded"])
        self.assertEqual(excluded_review_row["promoted_card_id"], promoted_card["id"])

    def test_promote_excluded_item_rejects_already_promoted_item(self) -> None:
        self._create_excluded_review_fixture()
        self.repository.create_promoted_candidate_card(
            "excluded_promote",
            {
                "id": "card_promoted_existing",
                "title": "已存在的 promoted 卡片",
                "granularity_level": "detail",
                "course_transformation": "已有人审复用",
                "teachable_one_liner": "这条卡片已经存在，不应再次 promote。",
                "draft_body": "同一个 excluded item 只能链接一个 promoted candidate。",
                "evidence": [
                    {
                        "section_id": "section_excluded_1",
                        "quote": "A retired standard can remain in everyday use after a formal replacement appears.",
                        "page_number": 1,
                        "analysis": "这说明标准更新和实际执行会脱节。",
                    }
                ],
                "figure_ids": [],
                "status": "candidate",
                "embedding": [0.0] * 64,
                "created_at": "2026-03-06T00:00:02+00:00",
                "judgement": {
                    "color": "yellow",
                    "reason": "已有候选卡片。",
                    "model_version": "stub-model",
                    "prompt_version": JUDGEMENT_PROMPT_VERSION,
                    "rubric_version": CARD_RUBRIC_VERSION,
                },
            },
        )

        class StubClient:
            model = "stub-promote-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                return {"cards": [], "excluded_content": []}

        with patch.object(services_module.LLMCardEngine, "_build_client", return_value=StubClient()):
            client = TestClient(create_app(self.settings))
            response = client.post(
                "/api/review-items/excluded/excluded_promote/promote",
                json={"reviewer": "tester", "note": ""},
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("already been promoted", response.text)

    def test_promote_excluded_item_returns_not_found(self) -> None:
        response = self.client.post(
            "/api/review-items/excluded/excluded_missing/promote",
            json={"reviewer": "tester", "note": ""},
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_promote_excluded_item_reports_disabled_llm(self) -> None:
        self._create_excluded_review_fixture()
        response = self.client.post(
            "/api/review-items/excluded/excluded_promote/promote",
            json={"reviewer": "tester", "note": ""},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("LLM provider is not enabled", response.text)

    def test_promote_excluded_item_reports_llm_failure(self) -> None:
        self._create_excluded_review_fixture()

        class StubClient:
            model = "stub-promote-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                raise LLMGenerationError("promote llm failed")

        with patch.object(services_module.LLMCardEngine, "_build_client", return_value=StubClient()):
            client = TestClient(create_app(self.settings))
            response = client.post(
                "/api/review-items/excluded/excluded_promote/promote",
                json={"reviewer": "tester", "note": ""},
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("promote llm failed", response.text)

    def test_export_google_doc_rejects_unreviewed_card_selection(self) -> None:
        fixture = self._create_export_card_fixture(card_id="card_unreviewed", review_decision=None)
        response = self.client.post(
            "/api/exports/google-doc",
            json={
                "run_id": fixture["run"]["id"],
                "card_ids": [fixture["card_id"]],
                "document_title": "Boss Report - Export Validation",
                "existing_google_doc_id": "",
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("has not been reviewed as accepted", response.text)

    def test_export_google_doc_rejects_non_accepted_review_statuses(self) -> None:
        for decision in ["rejected", "keep_for_later", "needs_manual_check"]:
            with self.subTest(decision=decision):
                fixture = self._create_export_card_fixture(card_id=f"card_{decision}", review_decision=decision)
                response = self.client.post(
                    "/api/exports/google-doc",
                    json={
                        "run_id": fixture["run"]["id"],
                        "card_ids": [fixture["card_id"]],
                        "document_title": "Boss Report - Export Validation",
                        "existing_google_doc_id": "",
                    },
                )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(f"review decision is {decision}", response.text)

    def test_export_google_doc_fails_closed_for_mixed_valid_and_invalid_selection(self) -> None:
        run = self.repository.create_run("Export Topic", {"operator": "tester"})
        accepted = self._create_export_card_fixture(card_id="card_valid", review_decision="accepted", run=run)
        rejected = self._create_export_card_fixture(card_id="card_invalid", review_decision="rejected", run=run)
        response = self.client.post(
            "/api/exports/google-doc",
            json={
                "run_id": run["id"],
                "card_ids": [accepted["card_id"], rejected["card_id"]],
                "document_title": "Boss Report - Export Validation",
                "existing_google_doc_id": "",
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("card_invalid", response.text)
        self.assertEqual(self.repository._fetchall("SELECT * FROM exports"), [])

    def test_export_google_doc_rejects_foreign_run_card_selection(self) -> None:
        source = self._create_export_card_fixture(card_id="card_foreign", review_decision="accepted")
        foreign_run = self.repository.create_run("Another Topic", {"operator": "tester"})
        response = self.client.post(
            "/api/exports/google-doc",
            json={
                "run_id": foreign_run["id"],
                "card_ids": [source["card_id"]],
                "document_title": "Boss Report - Export Validation",
                "existing_google_doc_id": "",
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("belongs to run", response.text)

    def test_export_google_doc_rejects_nonexistent_card_selection(self) -> None:
        run = self.repository.create_run("Export Topic", {"operator": "tester"})
        response = self.client.post(
            "/api/exports/google-doc",
            json={
                "run_id": run["id"],
                "card_ids": ["card_missing"],
                "document_title": "Boss Report - Export Validation",
                "existing_google_doc_id": "",
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("does not exist", response.text)

    def test_export_google_doc_uses_gws_create_mode_when_enabled(self) -> None:
        fixture = self._create_export_card_fixture(card_id="card_gws_create", review_decision="accepted")
        exporter = services_module.ExportService(
            Settings(
                data_dir=self.settings.data_dir,
                db_path=self.settings.db_path,
                google_docs_mode="gws",
                llm_mode="disabled",
            ),
            self.repository,
        )

        calls: list[list[str]] = []

        def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
            calls.append(command)
            if command[3] == "create":
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"documentId": "doc_created"}), stderr="")
            if command[3] == "batchUpdate":
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")
            raise AssertionError(f"Unexpected command: {command}")

        with patch("subprocess.run", side_effect=fake_run):
            record = exporter.export_google_doc_package(
                run_id=fixture["run"]["id"],
                card_ids=[fixture["card_id"]],
                document_title="Boss Report - Live Create",
                existing_google_doc_id="",
            )

        self.assertEqual(record["export_status"], "exported")
        self.assertEqual(record["export_mode"], "create")
        self.assertEqual(record["google_doc_id"], "doc_created")
        self.assertEqual(record["error_message"], "")
        self.assertEqual([command[3] for command in calls], ["create", "batchUpdate"])

    def test_export_google_doc_uses_append_mode_for_existing_doc(self) -> None:
        fixture = self._create_export_card_fixture(card_id="card_gws_append", review_decision="accepted")
        exporter = services_module.ExportService(
            Settings(
                data_dir=self.settings.data_dir,
                db_path=self.settings.db_path,
                google_docs_mode="gws",
                llm_mode="disabled",
            ),
            self.repository,
        )

        batch_payloads: list[dict] = []

        def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
            if command[3] == "get":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"body": {"content": [{"endIndex": 1}, {"endIndex": 25}]}}),
                    stderr="",
                )
            if command[3] == "batchUpdate":
                batch_payloads.append(json.loads(command[-1]))
                return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")
            if command[3] == "create":
                raise AssertionError("Append mode should not create a new document")
            raise AssertionError(f"Unexpected command: {command}")

        with patch("subprocess.run", side_effect=fake_run):
            record = exporter.export_google_doc_package(
                run_id=fixture["run"]["id"],
                card_ids=[fixture["card_id"]],
                document_title="Boss Report - Live Append",
                existing_google_doc_id="doc_existing",
            )

        self.assertEqual(record["export_status"], "exported")
        self.assertEqual(record["export_mode"], "append")
        self.assertEqual(record["google_doc_id"], "doc_existing")
        self.assertEqual(len(batch_payloads), 1)
        first_index = batch_payloads[0]["requests"][0]["insertText"]["location"]["index"]
        self.assertGreaterEqual(first_index, 24)

    def test_export_google_doc_records_gws_failure_result(self) -> None:
        fixture = self._create_export_card_fixture(card_id="card_gws_failed", review_decision="accepted")
        exporter = services_module.ExportService(
            Settings(
                data_dir=self.settings.data_dir,
                db_path=self.settings.db_path,
                google_docs_mode="gws",
                llm_mode="disabled",
            ),
            self.repository,
        )

        def fake_run(command: list[str], check: bool, capture_output: bool, text: bool):
            if command[3] == "create":
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"documentId": "doc_failed"}), stderr="")
            if command[3] == "batchUpdate":
                raise subprocess.CalledProcessError(1, command, stderr="gws batch update failed")
            raise AssertionError(f"Unexpected command: {command}")

        with patch("subprocess.run", side_effect=fake_run):
            record = exporter.export_google_doc_package(
                run_id=fixture["run"]["id"],
                card_ids=[fixture["card_id"]],
                document_title="Boss Report - Live Failure",
                existing_google_doc_id="",
            )

        self.assertEqual(record["export_status"], "export_failed")
        self.assertEqual(record["export_mode"], "create")
        self.assertEqual(record["google_doc_id"], "doc_failed")
        self.assertIn("gws batch update failed", record["error_message"])

    def test_access_queue_reactivation_reprocesses_paper_into_main_pipeline(self) -> None:
        run = self.repository.create_run("reactivation topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("reactivation topic")
        topic_run = self.repository.create_topic_run(run["id"], topic["id"])
        paper = self.repository.create_or_get_paper(
            title="Queued Paper",
            authors=["Ada Researcher"],
            publication_year=2026,
            external_id="paper::queue::1",
            source_type="semantic_scholar",
            local_path="",
            original_url="https://example.com/queued-paper",
            access_status="manual_needed",
            ingestion_status="queued",
            parse_status="pending",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "search")
        self.repository.create_access_queue_item(paper["id"], run["id"], "Need full text")

        queue_item = self.repository.list_access_queue(run["id"])[0]
        source_pdf = Path(self.temp_dir.name) / "reactivation-source.pdf"
        source_pdf.write_bytes(
            build_minimal_pdf_bytes(
                "Reactivated evidence body for testing. This full text is deliberately long enough to pass PDF readability validation."
            )
        )

        response = self.client.post(
            f"/api/access-queue/{queue_item['id']}/reactivate",
            json={"local_path": str(source_pdf), "reviewer": "tester"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["queue_item"]["status"], "reactivated")
        self.assertEqual(payload["paper"]["access_status"], "open_fulltext")
        self.assertEqual(payload["paper"]["parse_status"], "parsed")
        self.assertEqual(len(payload["topic_runs"]), 1)
        self.assertEqual(payload["topic_runs"][0]["status"], "completed")
        self.assertEqual(payload["topic_runs"][0]["current_stage"], "completed")
        refreshed_queue = self.repository.get_access_queue_item(queue_item["id"])
        self.assertEqual(refreshed_queue["status"], "reactivated")
        refreshed_topic_run = self.repository.get_topic_run(topic_run["id"])
        self.assertEqual(refreshed_topic_run["status"], "completed")
        self.assertEqual(refreshed_topic_run["stats"]["queued_for_access"], 0)

    def test_access_queue_reactivation_rejects_missing_local_file(self) -> None:
        run = self.repository.create_run("reactivation topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("reactivation topic")
        paper = self.repository.create_or_get_paper(
            title="Queued Paper Missing",
            authors=["Ada Researcher"],
            publication_year=2026,
            external_id="paper::queue::missing",
            source_type="semantic_scholar",
            local_path="",
            original_url="https://example.com/queued-paper-missing",
            access_status="manual_needed",
            ingestion_status="queued",
            parse_status="pending",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "search")
        self.repository.create_access_queue_item(paper["id"], run["id"], "Need full text")
        queue_item = self.repository.list_access_queue(run["id"])[0]

        response = self.client.post(
            f"/api/access-queue/{queue_item['id']}/reactivate",
            json={"local_path": str(Path(self.temp_dir.name) / "missing-file.pdf"), "reviewer": "tester"},
        )
        self.assertEqual(response.status_code, 400, response.text)
        refreshed_queue = self.repository.get_access_queue_item(queue_item["id"])
        self.assertEqual(refreshed_queue["status"], "open")

    def test_sqlite_write_waits_for_transient_lock_and_then_succeeds(self) -> None:
        run = self.repository.create_run("Lock Topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("Lock Topic")
        topic_run = self.repository.create_topic_run(run["id"], topic["id"])
        errors: list[Exception] = []

        lock_connection = get_connection(self.settings.db_path)
        lock_connection.execute("BEGIN IMMEDIATE")
        lock_connection.execute("UPDATE runs SET status = status WHERE id = ?", (run["id"],))

        def delayed_write() -> None:
            try:
                self.repository.update_topic_run(topic_run["id"], "running", stats={"discovered": 1})
            except Exception as error:  # pragma: no cover - assertion captures unexpected failure
                errors.append(error)

        worker = threading.Thread(target=delayed_write)
        started_at = time.monotonic()
        worker.start()
        time.sleep(6.0)
        lock_connection.commit()
        lock_connection.close()
        worker.join(timeout=10)

        self.assertFalse(errors, errors[0] if errors else "")
        self.assertFalse(worker.is_alive())
        self.assertGreaterEqual(time.monotonic() - started_at, 6.0)
        refreshed_topic_run = self.repository.get_topic_run(topic_run["id"])
        self.assertEqual(refreshed_topic_run["stats"]["discovered"], 1)

    def test_parse_and_store_uses_atomic_parse_persistence_path(self) -> None:
        paper = self.repository.create_or_get_paper(
            title="Atomic Parse Paper",
            authors=["Ada Researcher"],
            publication_year=2026,
            external_id="paper::atomic::1",
            source_type="local",
            local_path="",
            original_url="https://example.com/atomic-parse-paper",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="pending",
            artifact_path="artifact.pdf",
        )
        pipeline = PaperPipeline(self.settings, self.repository)
        parsed_payload = {
            "sections": [
                {
                    "id": "section_atomic_1",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "Atomic parsing should persist through one repository write path.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
            "figures": [
                {
                    "id": "figure_atomic_1",
                    "figure_label": "Figure 1",
                    "caption": "Atomic figure caption",
                    "storage_path": "",
                    "linked_section_ids": ["section_atomic_1"],
                }
            ],
        }

        with patch.object(pipeline.parser, "parse", return_value=parsed_payload), \
            patch.object(self.repository, "persist_parse_result", wraps=self.repository.persist_parse_result) as persist_result, \
            patch.object(self.repository, "replace_sections", side_effect=AssertionError("replace_sections should not be called")), \
            patch.object(self.repository, "replace_figures", side_effect=AssertionError("replace_figures should not be called")), \
            patch.object(self.repository, "update_paper", side_effect=AssertionError("update_paper should not be called directly")):
            stored_count = pipeline.parse_and_store(paper)

        self.assertEqual(stored_count, 1)
        self.assertEqual(persist_result.call_count, 1)
        refreshed_paper = self.repository.get_paper(paper["id"])
        self.assertEqual(refreshed_paper["parse_status"], "parsed")
        self.assertEqual(len(self.repository.get_sections(paper["id"])), 1)
        self.assertEqual(len(self.repository.get_figures(paper["id"])), 1)

    def test_parser_treats_pdf_magic_signature_as_pdf_even_without_pdf_suffix(self) -> None:
        source_pdf = Path(self.temp_dir.name) / "paper.03314"
        source_pdf.write_bytes(build_minimal_pdf_bytes("We find that agent handoffs reduce response drift."))

        parser = PdfParser(self.settings)
        parsed = parser.parse({"artifact_path": str(source_pdf), "local_path": ""})

        self.assertEqual(parsed["artifact_type"], "pdf")
        self.assertGreaterEqual(len(parsed["sections"]), 1)
        self.assertIn("agent handoffs reduce response drift", parsed["sections"][0]["paragraph_text"])

    def test_corrupted_pdf_is_blocked_before_card_generation(self) -> None:
        source_pdf = Path(self.temp_dir.name) / "corrupted.pdf"
        source_pdf.write_bytes(build_corrupted_pdf_bytes())

        create_response = self.client.post(
            "/api/runs",
            json={
                "topics_text": "LLM agent",
                "metadata": {},
                "local_pdfs": [{"path": str(source_pdf), "topics": ["LLM agent"]}],
            },
        )
        self.assertEqual(create_response.status_code, 200, create_response.text)
        run_id = create_response.json()["run"]["id"]

        for _ in range(30):
            run_response = self.client.get(f"/api/runs/{run_id}")
            self.assertEqual(run_response.status_code, 200, run_response.text)
            if run_response.json()["run"]["status"] in {"completed", "partial_failed", "failed"}:
                break
            time.sleep(0.2)

        cards_response = self.client.get("/api/cards", params={"run_id": run_id})
        self.assertEqual(cards_response.status_code, 200, cards_response.text)
        self.assertEqual(cards_response.json()["cards"], [])

        papers = self.repository._fetchall("SELECT * FROM papers WHERE local_path = ?", (str(source_pdf),))
        self.assertEqual(len(papers), 1)
        self.assertIn(papers[0]["parse_status"], {"parse_failed", "quality_failed"})
        self.assertNotEqual(papers[0]["parse_failure_reason"], "")

    def test_parser_ignores_non_text_stream_noise_outside_bt_et_blocks(self) -> None:
        source_pdf = Path(self.temp_dir.name) / "noisy-valid.pdf"
        source_pdf.write_bytes(
            build_pdf_with_non_text_tj_noise(
                "Verifier agents reduced contradiction rate by 23 percent in the reported experiment."
            )
        )

        parser = PdfParser(self.settings)
        parsed = parser.parse({"artifact_path": str(source_pdf), "local_path": ""})

        self.assertEqual(parsed["artifact_type"], "pdf")
        self.assertEqual(len(parsed["sections"]), 1)
        self.assertIn("Verifier agents reduced contradiction rate", parsed["sections"][0]["paragraph_text"])
        self.assertNotIn("JFIF", parsed["sections"][0]["paragraph_text"])

    def test_parser_uses_markitdown_when_available(self) -> None:
        source_pdf = Path(self.temp_dir.name) / "markitdown.pdf"
        source_pdf.write_bytes(build_corrupted_pdf_bytes())

        class StubResult:
            text_content = (
                "# Multi-Agent Collaboration\n\n"
                "Verifier agents reduced contradiction rate by 23 percent in the experiment.\n\n"
                "![Agent architecture](figure-1.png)\n\n"
                "The coordinator agent only improves quality when it explicitly checks conflicts."
            )

        class StubMarkItDown:
            def __init__(self, enable_plugins: bool = False):
                self.enable_plugins = enable_plugins

            def convert(self, source: str) -> StubResult:
                return StubResult()

        with patch.object(services_module, "MarkItDown", StubMarkItDown):
            parser = PdfParser(self.settings)
            parsed = parser.parse({"artifact_path": str(source_pdf), "local_path": ""})

        self.assertEqual(parsed["artifact_type"], "pdf")
        self.assertGreaterEqual(len(parsed["sections"]), 1)
        self.assertIn("Verifier agents reduced contradiction rate", parsed["sections"][0]["paragraph_text"])
        self.assertEqual(len(parsed["figures"]), 1)
        self.assertEqual(parsed["figures"][0]["caption"], "Agent architecture")

    def test_parser_extracts_caption_only_figures_from_pdf_text(self) -> None:
        source_pdf = Path(self.temp_dir.name) / "caption-only.pdf"
        source_pdf.write_bytes(build_minimal_pdf_bytes("Fig. 1 Overview of the verifier coordination loop."))

        parser = PdfParser(self.settings)
        parsed = parser.parse({"artifact_path": str(source_pdf), "local_path": ""})

        self.assertEqual(parsed["artifact_type"], "pdf")
        self.assertEqual(len(parsed["figures"]), 1)
        self.assertEqual(parsed["figures"][0]["figure_label"], "Figure 1")
        self.assertIn("Overview of the verifier coordination loop", parsed["figures"][0]["caption"])
        self.assertTrue(parsed["figures"][0]["linked_section_ids"])

    def test_parser_materializes_html_data_uri_figure_assets(self) -> None:
        png_bytes = build_png_bytes()
        encoded = base64.b64encode(png_bytes).decode("ascii")
        source_html = Path(self.temp_dir.name) / "figure-data-uri.html"
        source_html.write_text(
            (
                "<html><head><meta property='og:url' content='https://example.com/paper'></head><body>"
                "<figure><img src='data:image/png;base64,"
                + encoded
                + "' alt='Verifier workflow'><figcaption>Figure 1. Verifier workflow</figcaption></figure>"
                "</body></html>"
            ),
            encoding="utf-8",
        )

        parser = PdfParser(self.settings)
        parsed = parser.parse({"artifact_path": str(source_html), "local_path": ""})

        self.assertEqual(parsed["artifact_type"], "html")
        self.assertEqual(len(parsed["figures"]), 1)
        figure = parsed["figures"][0]
        self.assertEqual(figure["asset_status"], "validated_local_asset")
        self.assertTrue(Path(figure["asset_local_path"]).exists())
        self.assertEqual(figure["mime_type"], "image/png")
        self.assertGreater(figure["byte_size"], 0)

    def test_parser_extracts_validated_pdf_image_assets(self) -> None:
        if getattr(services_module, "fitz", None) is None:
            self.skipTest("PyMuPDF not installed")
        source_pdf = Path(self.temp_dir.name) / "embedded-figure.pdf"
        source_pdf.write_bytes(
            build_pdf_with_embedded_png(
                build_png_bytes(size=(48, 32)),
                "Fig. 1 Embedded verifier coordination asset.",
            )
        )

        class StubResult:
            text_content = (
                "# Embedded Figure\n\n"
                "Fig. 1 Embedded verifier coordination asset.\n\n"
                "Verifier agents reduce contradictions when they explicitly check conflicts."
            )

        class StubMarkItDown:
            def __init__(self, enable_plugins: bool = False):
                self.enable_plugins = enable_plugins

            def convert(self, source: str) -> StubResult:
                return StubResult()

        with patch.object(services_module, "MarkItDown", StubMarkItDown):
            parser = PdfParser(self.settings)
            parsed = parser.parse({"artifact_path": str(source_pdf), "local_path": ""})

        self.assertEqual(parsed["artifact_type"], "pdf")
        self.assertEqual(len(parsed["figures"]), 1)
        figure = parsed["figures"][0]
        self.assertEqual(figure["asset_status"], "validated_local_asset")
        self.assertTrue(Path(figure["asset_local_path"]).exists())
        self.assertGreater(figure["width"], 0)
        self.assertGreater(figure["height"], 0)
        self.assertEqual(figure["asset_kind"], "pdf_embedded")

    def test_split_paragraphs_dehyphenates_words_and_splits_before_section_headers(self) -> None:
        text = (
            "Abstract Recent generative models have achieved remark- able progress in image editing. "
            "However, ex- isting systems remain text-guided. Introduction "
            "We show coding-oriented interaction patterns."
        )
        paragraphs = split_paragraphs(text)
        self.assertIn("remarkable progress", paragraphs[0])
        self.assertIn("existing systems remain", paragraphs[0])
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(paragraphs[1].startswith("Introduction"))

    def test_pipeline_uses_llm_card_engine_when_available(self) -> None:
        run = self.repository.create_run("LLM agent", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("LLM agent")
        paper = self.repository.create_or_get_paper(
            title="LLM Agents in Practice",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::llm-agents-practice",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": "section_alpha",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "Multi-agent coordination improves planning reliability by exposing intermediate decisions.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )

        class StubCardEngine:
            def is_enabled(self) -> bool:
                return True

            def extract_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                sections: list[dict],
                figures: list[dict] | None = None,
                planned_cards: list[dict] | None = None,
                planning_context: dict | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                return {
                    "cards": [
                        {
                            "title": "显式协调能提升多智能体规划可靠性",
                            "granularity_level": "subpattern",
                            "draft_body": "这张卡强调的不是多智能体本身，而是显式协调带来的稳定性提升。",
                            "evidence": [
                                {
                                    "section_id": sections[0]["id"],
                                    "quote": sections[0]["paragraph_text"],
                                    "page_number": sections[0]["page_number"],
                                    "analysis": "这段证据把显式协调和实际可靠性收益直接连了起来，所以很适合变成课程里的方法卡。",
                                }
                            ],
                            "figure_ids": [],
                            "status": "candidate",
                        }
                    ],
                    "excluded_content": [
                        {
                            "label": "多智能体的一般背景介绍",
                            "exclusion_type": "background",
                            "reason": "背景介绍有帮助，但不是这篇里最值得教给学员的认知跃迁。",
                            "section_ids": [sections[0]["id"]],
                        }
                    ],
                }

            def judge_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                extracted_cards: list[dict],
                figures: list[dict] | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                return {
                    "cards": [
                        {
                            "title": extracted_cards[0]["title"],
                            "granularity_level": extracted_cards[0]["granularity_level"],
                            "course_transformation": f"{topic_name}：显式协调检查法",
                            "teachable_one_liner": "如果你要多智能体规划更稳，就别只分工，还要让它们彼此解释并互相检查。",
                            "draft_body": extracted_cards[0]["draft_body"],
                            "evidence": extracted_cards[0]["evidence"],
                            "figure_ids": [],
                            "status": "candidate",
                            "judgement": {
                                "color": "green",
                                "reason": "这是可以直接讲、直接用的高可操作方法模式。",
                                "model_version": "stub-llm-model",
                                "prompt_version": JUDGEMENT_PROMPT_VERSION,
                                "rubric_version": CARD_RUBRIC_VERSION,
                            },
                        }
                    ]
                }

        pipeline = PaperPipeline(self.settings, self.repository, card_engine=StubCardEngine())
        created_count = pipeline.build_cards(paper, topic, run["id"])

        self.assertEqual(created_count, 1)
        cards = self.repository.list_cards(run_id=run["id"])
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], "显式协调能提升多智能体规划可靠性")
        card_detail = self.repository.get_card(cards[0]["id"])
        self.assertEqual(card_detail["judgement"]["model_version"], "stub-llm-model")
        self.assertEqual(card_detail["teachable_one_liner"], "如果你要多智能体规划更稳，就别只分工，还要让它们彼此解释并互相检查。")
        self.assertEqual(len(card_detail["excluded_content"]), 1)
        self.assertEqual(card_detail["excluded_content"][0]["exclusion_type"], "background")

    def test_section_structure_classification_marks_abstract_vs_body(self) -> None:
        sections = services_module.enrich_sections_with_structure(
            [
                {
                    "id": "sec_1",
                    "section_order": 1,
                    "section_title": "Abstract",
                    "paragraph_text": "Abstract We introduce a new framework and summarize main findings.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                },
                {
                    "id": "sec_2",
                    "section_order": 2,
                    "section_title": "Results",
                    "paragraph_text": "Results show the proposed method improves reliability by 23 percent.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                },
            ],
            "pdf_markitdown",
        )
        self.assertTrue(sections[0]["is_abstract"])
        self.assertFalse(sections[0]["is_body"])
        self.assertEqual(sections[1]["section_kind"], "results")
        self.assertTrue(sections[1]["is_body"])

    def test_evidence_packet_prefers_body_sections_for_primary(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        sections = services_module.enrich_sections_with_structure(
            [
                {
                    "id": "sec_abs",
                    "section_order": 1,
                    "section_title": "Abstract",
                    "paragraph_text": "Abstract discusses verifier pattern at high level.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                },
                {
                    "id": "sec_body",
                    "section_order": 2,
                    "section_title": "Results",
                    "paragraph_text": "Results: verifier checks reduce contradiction by 23 percent in agent workflows.",
                    "page_number": 2,
                    "embedding": [0.0] * 64,
                },
            ],
            "pdf_markitdown",
        )
        packet = pipeline._build_evidence_packet(sections, [], "verifier workflow")
        primary_ids = [item["id"] for item in packet["primary_candidate_sections"]]
        self.assertIn("sec_body", primary_ids)
        self.assertNotEqual(primary_ids[0], "sec_abs")

    def test_grounding_gate_rejects_abstract_only_primary_when_body_exists(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        sections = services_module.enrich_sections_with_structure(
            [
                {
                    "id": "sec_abs",
                    "section_order": 1,
                    "section_title": "Abstract",
                    "paragraph_text": "Abstract explains the motivation.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                },
                {
                    "id": "sec_res",
                    "section_order": 2,
                    "section_title": "Results",
                    "paragraph_text": "Results show clear measured gains.",
                    "page_number": 2,
                    "embedding": [0.0] * 64,
                },
            ],
            "pdf_markitdown",
        )
        kept, excluded = pipeline._gate_extracted_candidates(
            [
                {
                    "title": "摘要型候选",
                    "primary_section_ids": ["sec_abs"],
                    "supporting_section_ids": [],
                    "paper_specific_object": "摘要型候选",
                    "evidence": [{"section_id": "sec_abs", "quote": "Abstract explains the motivation.", "analysis": "summary", "page_number": 1}],
                }
            ],
            sections,
        )
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(excluded), 1)
        self.assertIn("abstract_dominant_evidence_gate", excluded[0]["reason"])

    def test_duplicate_governance_suppresses_same_evidence_variants(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        kept, excluded = pipeline._suppress_same_paper_duplicates(
            [
                {
                    "title": "候选A",
                    "primary_section_ids": ["sec_1"],
                    "paper_specific_object": "verifier check",
                    "claim_type": "method",
                    "evidence_level": "strong",
                    "judgement": {"color": "green"},
                    "evidence": [{"section_id": "sec_1"}],
                },
                {
                    "title": "候选B",
                    "primary_section_ids": ["sec_1"],
                    "paper_specific_object": "verifier check",
                    "claim_type": "method",
                    "evidence_level": "weak",
                    "judgement": {"color": "yellow"},
                    "evidence": [{"section_id": "sec_1"}],
                },
            ]
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(kept[0]["duplicate_disposition"], "kept")
        self.assertIn("replaced by cluster representative", excluded[0]["reason"])

    def test_quality_metrics_endpoint_reports_grounding_and_duplicate_rates(self) -> None:
        run = self.repository.create_run("quality metrics", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("quality metrics")
        paper = self.repository.create_or_get_paper(
            title="Quality Metrics Paper",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::quality-metrics",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            services_module.enrich_sections_with_structure(
                [
                    {
                        "id": "sec_q_abs",
                        "section_order": 1,
                        "section_title": "Abstract",
                        "paragraph_text": "Abstract overview.",
                        "page_number": 1,
                        "embedding": [0.0] * 64,
                    },
                    {
                        "id": "sec_q_body",
                        "section_order": 2,
                        "section_title": "Results",
                        "paragraph_text": "Measured result with concrete mechanism evidence.",
                        "page_number": 2,
                        "embedding": [0.0] * 64,
                    },
                ],
                "pdf_markitdown",
            ),
        )
        self.repository.replace_generation_outputs_for_paper_topic(
            paper["id"],
            topic["id"],
            run["id"],
            [
                {
                    "id": "card_quality_1",
                    "title": "正文证据卡",
                    "granularity_level": "detail",
                    "course_transformation": "课程对象A",
                    "teachable_one_liner": "一句话A",
                    "draft_body": "说明A",
                    "evidence": [{"section_id": "sec_q_body", "quote": "Measured result", "quote_zh": "量化结果", "page_number": 2, "analysis": "直接证据"}],
                    "figure_ids": [],
                    "status": "candidate",
                    "embedding": [0.0] * 64,
                    "primary_section_ids": ["sec_q_body"],
                    "supporting_section_ids": ["sec_q_abs"],
                    "paper_specific_object": "mechanism A",
                    "claim_type": "result",
                    "evidence_level": "strong",
                    "body_grounding_reason": "正文结果段直接支撑",
                    "grounding_quality": "strong",
                    "duplicate_cluster_id": "dup_cluster_1",
                    "duplicate_rank": 1,
                    "duplicate_disposition": "kept",
                    "created_at": "2026-03-07T00:00:01+00:00",
                    "judgement": {
                        "color": "green",
                        "reason": "有效",
                        "model_version": "stub",
                        "prompt_version": JUDGEMENT_PROMPT_VERSION,
                        "rubric_version": CARD_RUBRIC_VERSION,
                    },
                }
            ],
            [],
        )
        self.repository.create_review_decision("card", "card_quality_1", "tester", "accepted", "ok")
        response = self.client.get("/api/quality/metrics", params={"run_id": run["id"]})
        self.assertEqual(response.status_code, 200, response.text)
        metrics = response.json()["metrics"]
        self.assertEqual(metrics["total_cards"], 1)
        self.assertGreaterEqual(metrics["body_grounded_card_rate"], 1.0)
        self.assertGreaterEqual(metrics["paper_specific_object_presence_rate"], 1.0)
        self.assertEqual(metrics["accepted_cards"], 1)

    def test_review_item_detail_includes_grounding_diagnostics(self) -> None:
        fixture = self._create_export_card_fixture(card_id="card_diag", review_decision="accepted")
        response = self.client.get(f"/api/review-items/card/{fixture['card_id']}")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["item"]
        self.assertIn("grounding_diagnostics", item)
        diagnostics = item["grounding_diagnostics"]
        self.assertIn("section_type_mix", diagnostics)
        self.assertIn("same_paper_siblings", diagnostics)

    def test_review_item_detail_includes_linked_figures(self) -> None:
        fixture = self._create_export_card_fixture(
            card_id="card_fig",
            review_decision="accepted",
            figure_ids=["figure_demo_1"],
        )
        self.repository.replace_figures(
            fixture["paper"]["id"],
            [
                {
                    "id": "figure_demo_1",
                    "figure_label": "Figure 1",
                    "caption": "Verifier coordination loop",
                    "storage_path": "",
                    "linked_section_ids": [f"section_{fixture['card_id']}"],
                }
            ],
        )
        response = self.client.get(f"/api/review-items/card/{fixture['card_id']}")
        self.assertEqual(response.status_code, 200, response.text)
        item = response.json()["item"]
        self.assertEqual(len(item["figures"]), 1)
        self.assertEqual(item["figures"][0]["caption"], "Verifier coordination loop")

    def test_figure_asset_endpoint_serves_validated_local_asset(self) -> None:
        run = self.repository.create_run("figure asset", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("figure asset")
        paper = self.repository.create_or_get_paper(
            title="Figure Asset Paper",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::figure-asset",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        asset_path = Path(self.temp_dir.name) / "figure-asset.png"
        asset_path.write_bytes(build_png_bytes())
        self.repository.replace_figures(
            paper["id"],
            [
                {
                    "id": "figure_asset_1",
                    "figure_label": "Figure 1",
                    "caption": "Validated verifier asset",
                    "storage_path": str(asset_path),
                    "asset_status": "validated_local_asset",
                    "asset_kind": "local_copy",
                    "asset_local_path": str(asset_path),
                    "mime_type": "image/png",
                    "byte_size": asset_path.stat().st_size,
                    "width": 24,
                    "height": 16,
                    "linked_section_ids": [],
                }
            ],
        )

        response = self.client.get("/api/figures/figure_asset_1/asset")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertGreater(len(response.content), 0)

    def test_repository_persists_understanding_and_card_plan_records(self) -> None:
        run = self.repository.create_run("understanding topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("understanding topic")
        paper = self.repository.create_or_get_paper(
            title="Understanding Test Paper",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::understanding-record",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        understanding = self.repository.create_paper_understanding_record(
            paper_id=paper["id"],
            topic_id=topic["id"],
            run_id=run["id"],
            version="understanding-v1",
            understanding={
                "global_contribution_objects": [{"id": "obj_1", "label": "Verifier conflict resolver"}],
                "contribution_graph": [],
                "evidence_index": {"obj_1": {"section_ids": ["sec_1"], "figure_ids": []}},
                "candidate_level_hints": {"obj_1": "overall"},
            },
        )
        plan = self.repository.create_card_plan(
            paper_id=paper["id"],
            topic_id=topic["id"],
            run_id=run["id"],
            version="card-plan-v1",
            plan={
                "planned_cards": [{"id": "plan_1", "level": "overall", "target_object_id": "obj_1", "disposition": "produce"}],
                "coverage_report": {"produce": 1, "exclude": 0},
            },
        )
        latest_understanding = self.repository.get_latest_paper_understanding(paper["id"], topic["id"], run["id"])
        latest_plan = self.repository.get_latest_card_plan(paper["id"], topic["id"], run["id"])

        self.assertEqual(understanding["version"], "understanding-v1")
        self.assertEqual(plan["version"], "card-plan-v1")
        self.assertEqual(latest_understanding["understanding"]["global_contribution_objects"][0]["id"], "obj_1")
        self.assertEqual(latest_plan["plan"]["planned_cards"][0]["level"], "overall")

    def test_pipeline_single_paper_validation_writes_required_artifacts(self) -> None:
        run = self.repository.create_run("single paper validation", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("single paper validation")
        paper = self.repository.create_or_get_paper(
            title="Single Paper Validation",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::single-validation",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            services_module.enrich_sections_with_structure(
                [
                    {
                        "id": "section_sv_1",
                        "section_order": 1,
                        "section_title": "Abstract",
                        "paragraph_text": "Abstract: this paper studies verifier-based workflows.",
                        "page_number": 1,
                        "embedding": [0.0] * 64,
                    },
                    {
                        "id": "section_sv_2",
                        "section_order": 2,
                        "section_title": "Results",
                        "paragraph_text": "Results: verifier checks reduce contradiction rate by 23 percent.",
                        "page_number": 2,
                        "embedding": [0.0] * 64,
                    },
                ],
                "pdf_markitdown",
            ),
        )

        class StubCardEngine:
            def is_enabled(self) -> bool:
                return True

            def extract_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                sections: list[dict],
                figures: list[dict] | None = None,
                planned_cards: list[dict] | None = None,
                planning_context: dict | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                section_id = sections[-1]["id"]
                return {
                    "cards": [
                        {
                            "title": "验证者检查能显著降低冲突率",
                            "primary_section_ids": [section_id],
                            "supporting_section_ids": [],
                            "granularity_level": "detail",
                            "claim_type": "result",
                            "paper_specific_object": "verifier checks",
                            "body_grounding_reason": "结果段给出量化提升",
                            "evidence_level": "strong",
                            "possible_duplicate_signature": "verifier-check-result",
                            "draft_body": "验证者检查是多智能体流程里的高收益步骤。",
                            "evidence": [
                                {
                                    "section_id": section_id,
                                    "quote": "Results: verifier checks reduce contradiction rate by 23 percent.",
                                    "quote_zh": "",
                                    "page_number": 2,
                                    "analysis": "有量化结果，适合课程使用。",
                                }
                            ],
                            "figure_ids": [],
                            "status": "candidate",
                        }
                    ],
                    "excluded_content": [],
                }

            def judge_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                extracted_cards: list[dict],
                figures: list[dict] | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                card = extracted_cards[0]
                return {
                    "cards": [
                        {
                            **card,
                            "course_transformation": "验证者冲突检查模板：显式矛盾发现流程",
                            "teachable_one_liner": "多智能体流程不是只分工，而是要显式做冲突检查，不然就会出错。",
                            "judgement": {
                                "color": "green",
                                "reason": "具备可行动性且有量化证据。",
                                "model_version": "stub-model",
                                "prompt_version": JUDGEMENT_PROMPT_VERSION,
                                "rubric_version": CARD_RUBRIC_VERSION,
                            },
                        }
                    ]
                }

        pipeline = PaperPipeline(self.settings, self.repository, card_engine=StubCardEngine())
        result = pipeline.validate_single_paper_flow(paper=paper, topic=topic, run_id=run["id"])

        self.assertEqual(result["card_count"], 1)
        self.assertEqual(result["excluded_count"], 0)
        for key in ("paper_understanding", "card_plan", "final_cards", "excluded_content", "report"):
            self.assertTrue(Path(result["artifacts"][key]).exists(), key)
        card_plan = json.loads(Path(result["artifacts"]["card_plan"]).read_text(encoding="utf-8"))
        final_cards = json.loads(Path(result["artifacts"]["final_cards"]).read_text(encoding="utf-8"))
        produced_plan_ids = {
            item["plan_id"]
            for item in card_plan.get("planned_cards", [])
            if item.get("disposition") == "produce" and item.get("plan_id")
        }
        for card in final_cards:
            self.assertIn(card.get("plan_id", ""), produced_plan_ids)

    def test_align_cards_to_plan_requires_must_have_evidence_overlap(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        cards = [
            {
                "title": "Card A",
                "primary_section_ids": ["section_a"],
                "evidence": [{"section_id": "section_a"}],
                "judgement": {"color": "green"},
                "evidence_level": "strong",
            },
            {
                "title": "Card B",
                "primary_section_ids": ["section_b"],
                "evidence": [{"section_id": "section_b"}],
                "judgement": {"color": "green"},
                "evidence_level": "strong",
            },
        ]
        card_plan = {
            "planned_cards": [
                {
                    "plan_id": "plan_obj_x",
                    "level": "overall",
                    "target_object_id": "obj_x",
                    "target_object_label": "Object X",
                    "must_have_evidence_ids": ["section_x"],
                    "optional_supporting_ids": [],
                    "disposition": "produce",
                },
                {
                    "plan_id": "plan_obj_b",
                    "level": "local",
                    "target_object_id": "obj_b",
                    "target_object_label": "Object B",
                    "must_have_evidence_ids": ["section_b"],
                    "optional_supporting_ids": [],
                    "disposition": "produce",
                },
            ]
        }
        kept, excluded = pipeline._align_cards_to_plan(cards, card_plan)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["title"], "Card B")
        self.assertEqual(kept[0]["plan_id"], "plan_obj_b")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["label"], "Card A")

    def test_align_cards_to_plan_rejects_cards_when_planner_produces_zero_slots(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        cards = [
            {
                "title": "Card A",
                "primary_section_ids": ["section_a"],
                "evidence": [{"section_id": "section_a"}],
                "granularity_level": "subpattern",
                "judgement": {"color": "green"},
            },
            {
                "title": "Card B",
                "primary_section_ids": ["section_b"],
                "evidence": [{"section_id": "section_b"}],
                "granularity_level": "framework",
                "judgement": {"color": "red"},
            },
        ]
        card_plan = {
            "planned_cards": [
                {
                    "plan_id": "plan_obj_a",
                    "level": "detail",
                    "target_object_id": "obj_a",
                    "target_object_label": "Object A",
                    "must_have_evidence_ids": ["section_a"],
                    "optional_supporting_ids": [],
                    "disposition": "exclude",
                }
            ]
        }
        kept, excluded = pipeline._align_cards_to_plan(cards, card_plan)
        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 2)
        self.assertEqual(excluded[0]["label"], "Card A")
        self.assertIn("zero slots", excluded[0]["reason"])
        self.assertEqual(excluded[1]["label"], "Card B")

    def test_align_cards_to_plan_prefers_matching_planned_object_with_same_evidence(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        cards = [
            {
                "title": "Self-consistency CoT：多次生成不同推理路径，用最高频答案做最终输出",
                "paper_specific_object": "Self-consistency CoT：多条推理链 + 最高频答案作为最终结果",
                "course_transformation": "方法卡：多路径生成后做频次投票",
                "teachable_one_liner": "不要只采样一次，改成多次生成后投票。",
                "primary_section_ids": ["section_shared"],
                "evidence": [{"section_id": "section_shared"}],
                "judgement": {"color": "yellow"},
                "evidence_level": "strong",
                "source_plan_id": "plan_obj_self_consistency",
            },
            {
                "title": "DECKARD：Dreaming 与 Awake 两阶段",
                "paper_specific_object": "DECKARD 两阶段流程",
                "course_transformation": "流程卡：先拆子目标再校验假设",
                "teachable_one_liner": "把拆解和校验拆开。",
                "primary_section_ids": ["section_shared"],
                "evidence": [{"section_id": "section_shared"}],
                "judgement": {"color": "green"},
                "evidence_level": "strong",
            },
        ]
        card_plan = {
            "planned_cards": [
                {
                    "plan_id": "plan_obj_self_consistency",
                    "level": "detail",
                    "target_object_id": "obj_self_consistency",
                    "target_object_label": "Self-consistency CoT：多次生成不同推理路径，用最高频答案做最终输出",
                    "why_valuable_for_course": "把单次推理改成多次采样后投票。",
                    "must_have_evidence_ids": ["section_shared"],
                    "optional_supporting_ids": [],
                    "disposition": "produce",
                }
            ]
        }
        kept, excluded = pipeline._align_cards_to_plan(cards, card_plan)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["title"], "Self-consistency CoT：多次生成不同推理路径，用最高频答案做最终输出")
        self.assertEqual(kept[0]["plan_id"], "plan_obj_self_consistency")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["label"], "DECKARD：Dreaming 与 Awake 两阶段")

    def test_finalize_card_adds_quote_first_rendering_fields(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        finalized = pipeline._finalize_card(
            {
                "title": "中央规划器架构",
                "granularity_level": "subpattern",
                "course_transformation": "模式卡：中央规划器架构选择",
                "teachable_one_liner": "把协商改成中央规划。",
                "draft_body": "这是一个很短的过桥说明。",
                "evidence": [
                    {
                        "section_id": "section_a",
                        "quote": "A single LLM acts as the central planner.",
                        "quote_zh": "单个LLM充当中央规划器。",
                        "analysis": "这句定义了架构核心。",
                    },
                    {
                        "section_id": "section_b",
                        "quote": "It reduces context demand and improves scalability.",
                        "quote_zh": "它减少上下文需求并提升可扩展性。",
                        "analysis": "这句说明为什么值得教。",
                    },
                ],
                "figure_ids": [],
                "status": "candidate",
                "primary_section_ids": ["section_a"],
                "supporting_section_ids": ["section_b"],
                "paper_specific_object": "central planner",
                "claim_type": "method",
                "evidence_level": "strong",
                "body_grounding_reason": "正文直接描述机制和收益。",
                "grounding_quality": "strong",
                "duplicate_cluster_id": "",
                "duplicate_rank": 1,
                "duplicate_disposition": "kept",
                "planned_level": "local",
                "plan_id": "plan_obj_1",
                "plan_target_object_id": "obj_1",
                "plan_target_object_label": "中央规划器",
                "judgement": {"color": "green", "reason": "可直接讲。"},
            },
            "agentic workflow",
        )

        self.assertEqual(finalized["body_format"], "quote_first_interleaved_analysis")
        self.assertEqual(len(finalized["quote_first_blocks"]), 2)
        self.assertIn("原文（穿插分析）：", finalized["quote_first_markdown"])
        self.assertIn("单个LLM充当中央规划器。", finalized["quote_first_markdown"])
        self.assertIn("这句定义了架构核心。", finalized["quote_first_markdown"])

    def test_concept_alignment_gate_requires_belief_gap_and_named_course_object(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        cards = [
            {
                "title": "多智能体工作流框架",
                "course_transformation": "课程摘要",
                "teachable_one_liner": "这是一个重要框架。",
                "primary_section_ids": ["section_a"],
                "evidence": [{"section_id": "section_a"}],
                "judgement": {"color": "green", "reason": "信息完整。"},
            },
            {
                "title": "并行协作不是串行接力",
                "course_transformation": "并行协作模板：Outline→Parallel Expand",
                "teachable_one_liner": "别只让代理轮流接力，而是先统一大纲再并行扩写。",
                "primary_section_ids": ["section_b"],
                "evidence": [{"section_id": "section_b"}],
                "judgement": {"color": "green", "reason": "从串行到并行是明显认知转变。"},
            },
        ]
        kept, excluded = pipeline._gate_judged_cards_for_concept_alignment(cards)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["title"], "并行协作不是串行接力")
        self.assertEqual(len(excluded), 1)
        self.assertIn("CONCEPT alignment gate failed", excluded[0]["reason"])

    def test_concept_alignment_gate_keeps_direct_transfer_cards_without_explicit_belief_gap(self) -> None:
        pipeline = PaperPipeline(self.settings, self.repository)
        cards = [
            {
                "title": "记忆写回要分开处理修改与换出",
                "course_transformation": "记忆写回流程模板：修改与换出两步决策",
                "teachable_one_liner": "设计记忆系统时，把写回拆成修改旧记忆和腾出空间两类动作。",
                "paper_specific_object": "记忆写回的两步决策：修改与换出",
                "primary_section_ids": ["section_a"],
                "figure_ids": [],
                "evidence": [{"section_id": "section_a"}],
                "judgement": {"color": "yellow", "reason": "流程清楚，直接可迁移。"},
            }
        ]
        kept, excluded = pipeline._gate_judged_cards_for_concept_alignment(cards)
        self.assertEqual(len(kept), 1)
        self.assertEqual(excluded, [])

    def test_api_single_paper_validation_endpoint_returns_artifacts(self) -> None:
        run = self.repository.create_run("api single validation", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("api single validation")
        paper = self.repository.create_or_get_paper(
            title="API Single Validation",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::api-single-validation",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            services_module.enrich_sections_with_structure(
                [
                    {
                        "id": "section_api_sv_1",
                        "section_order": 1,
                        "section_title": "Results",
                        "paragraph_text": "Results: this section contains measurable improvements.",
                        "page_number": 1,
                        "embedding": [0.0] * 64,
                    }
                ],
                "pdf_markitdown",
            ),
        )
        response = self.client.post(
            f"/api/papers/{paper['id']}/validate-single",
            json={"topic_id": topic["id"], "run_id": run["id"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()["validation"]
        self.assertEqual(payload["paper_id"], paper["id"])
        self.assertEqual(payload["topic_id"], topic["id"])
        self.assertEqual(payload["run_id"], run["id"])
        self.assertIn("artifacts", payload)

        understanding_response = self.client.get(
            f"/api/papers/{paper['id']}/understanding",
            params={"topic_id": topic["id"], "run_id": run["id"]},
        )
        self.assertEqual(understanding_response.status_code, 200, understanding_response.text)
        card_plan_response = self.client.get(
            f"/api/papers/{paper['id']}/card-plan",
            params={"topic_id": topic["id"], "run_id": run["id"]},
        )
        self.assertEqual(card_plan_response.status_code, 200, card_plan_response.text)

    def test_pipeline_records_llm_failure_and_creates_no_cards(self) -> None:
        run = self.repository.create_run("agentic workflow", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("agentic workflow")
        paper = self.repository.create_or_get_paper(
            title="Agentic Workflows in Practice",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::agentic-workflows-practice",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": "section_beta",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "We find that verifier checks improve agentic workflow reliability by 23 percent in production-style tasks.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )

        class FailingCardEngine:
            def is_enabled(self) -> bool:
                return True

            def extract_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                sections: list[dict],
                figures: list[dict] | None = None,
                planned_cards: list[dict] | None = None,
                planning_context: dict | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                return {"cards": [], "excluded_content": []}

            def judge_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                extracted_cards: list[dict],
                figures: list[dict] | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                raise LLMGenerationError("request to https://api.example.com/v1/chat/completions failed: boom")

        pipeline = PaperPipeline(self.settings, self.repository, card_engine=FailingCardEngine())
        created_count = pipeline.build_cards(paper, topic, run["id"])

        self.assertEqual(created_count, 0)
        cards = self.repository.list_cards(run_id=run["id"])
        self.assertEqual(len(cards), 0)
        updated = self.repository._fetchone("SELECT * FROM papers WHERE id = ?", (paper["id"],))
        self.assertEqual(updated["card_generation_status"], "llm_failed")
        self.assertIn("https://api.example.com/v1/chat/completions", updated["card_generation_failure_reason"])

    def test_pipeline_requires_llm_provider_for_card_generation(self) -> None:
        run = self.repository.create_run("LLM agent", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("LLM agent")
        paper = self.repository.create_or_get_paper(
            title="LLM Agents in Practice",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::llm-only-required",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": "section_gamma",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "Verifier agents reduce contradiction rates in multi-agent workflows.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )

        pipeline = PaperPipeline(self.settings, self.repository)
        created_count = pipeline.build_cards(paper, topic, run["id"])

        self.assertEqual(created_count, 0)
        updated = self.repository._fetchone("SELECT * FROM papers WHERE id = ?", (paper["id"],))
        self.assertEqual(updated["card_generation_status"], "llm_unavailable")
        self.assertIn("LLM-only card generation is enabled", updated["card_generation_failure_reason"])

    def test_llm_engine_builds_openai_compatible_client(self) -> None:
        settings = Settings(
            data_dir=self.settings.data_dir,
            db_path=self.settings.db_path,
            max_workers=4,
            google_docs_mode="artifact_only",
            llm_mode="openai_compatible",
            llm_base_url="https://api.example.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-test",
        )
        engine = LLMCardEngine(settings)
        self.assertIsInstance(engine.client, OpenAICompatibleLLMClient)

    def test_llm_engine_uses_default_openai_base_url_when_not_provided(self) -> None:
        settings = Settings(
            data_dir=self.settings.data_dir,
            db_path=self.settings.db_path,
            max_workers=4,
            google_docs_mode="artifact_only",
            llm_mode="openai_compatible",
            llm_api_key="test-key",
            llm_model="gpt-test",
        )
        engine = LLMCardEngine(settings)
        self.assertIsInstance(engine.client, OpenAICompatibleLLMClient)
        self.assertEqual(engine.client.base_url, "https://api.openai.com/v1")

    def test_llm_engine_builds_anthropic_client(self) -> None:
        settings = Settings(
            data_dir=self.settings.data_dir,
            db_path=self.settings.db_path,
            max_workers=4,
            google_docs_mode="artifact_only",
            llm_mode="anthropic",
            llm_base_url="https://api.anthropic.com/v1",
            llm_api_key="test-key",
            llm_model="claude-test",
        )
        engine = LLMCardEngine(settings)
        self.assertIsInstance(engine.client, AnthropicLLMClient)

    def test_llm_engine_builds_gemini_client(self) -> None:
        settings = Settings(
            data_dir=self.settings.data_dir,
            db_path=self.settings.db_path,
            max_workers=4,
            google_docs_mode="artifact_only",
            llm_mode="gemini",
            llm_base_url="https://generativelanguage.googleapis.com",
            llm_api_key="test-key",
            llm_model="gemini-2.0-flash",
        )
        engine = LLMCardEngine(settings)
        self.assertIsInstance(engine.client, GeminiLLMClient)

    def test_llm_engine_routes_to_fallback_provider_after_network_error(self) -> None:
        settings = Settings(
            data_dir=self.settings.data_dir,
            db_path=self.settings.db_path,
            max_workers=4,
            google_docs_mode="artifact_only",
            llm_mode="openai_compatible",
            llm_base_url="https://primary.example.com/v1",
            llm_api_key="primary-key",
            llm_model="primary-model",
            llm_providers_json=json.dumps(
                [
                    {
                        "provider_id": "primary",
                        "provider_type": "openai_compatible",
                        "base_url": "https://primary.example.com/v1",
                        "api_key": "primary-key",
                        "model": "primary-model",
                        "priority": 0,
                    },
                    {
                        "provider_id": "fallback",
                        "provider_type": "anthropic",
                        "base_url": "https://fallback.example.com/v1",
                        "api_key": "fallback-key",
                        "model": "fallback-model",
                        "priority": 1,
                    },
                ]
            ),
            llm_provider_cooldown_seconds=60,
        )

        class FailingClient:
            model = "primary-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                raise LLMGenerationError("request to https://primary.example.com/v1/chat/completions failed: ssl handshake")

        class FallbackClient:
            model = "fallback-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                prompt = json.loads(user_prompt)
                if prompt["stage"] == "candidate_extraction":
                    return {
                        "cards": [
                            {
                                "title": "验证者先找冲突再继续协作",
                                "section_ids": ["section_demo_1", "section_demo_2"],
                                "granularity_level": "subpattern",
                                "draft_body": "多智能体协作里先显式找冲突，再继续汇总。",
                                "evidence_analysis": [
                                    {"section_id": "section_demo_1", "analysis": "第一段说明模式本体。"},
                                    {"section_id": "section_demo_2", "analysis": "第二段给出量化结果。"},
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "验证者先找冲突再继续协作",
                            "course_transformation": "多智能体协作：验证者冲突检查",
                            "teachable_one_liner": "多个执行者可能打架时，先加验证者找冲突，再进入汇总。",
                            "draft_body": "多智能体协作里先显式找冲突，再继续汇总。",
                            "evidence_localization": [
                                {"section_id": "section_demo_1", "quote_zh": "当多个执行智能体可能互相矛盾时，需要单独设置一个验证者来显式检查冲突。"},
                                {"section_id": "section_demo_2", "quote_zh": "加入验证者后，不一致率下降了 23%。"},
                            ],
                            "judgement": {"color": "green", "reason": "这是可直接迁移的协作模式。"},
                        }
                    ]
                }

        engine = LLMCardEngine(
            settings,
            provider_clients={
                "primary": FailingClient(),
                "fallback": FallbackClient(),
            },
        )

        payload = engine.smoke_test()

        self.assertEqual(payload["card_count"], 1)
        self.assertEqual(payload["provider_route"]["selected_provider"]["provider_id"], "fallback")
        extraction_route = payload["provider_routes"]["candidate_extraction"]
        attempt_statuses = [attempt["status"] for attempt in extraction_route["attempts"]]
        self.assertIn("failed", attempt_statuses)
        self.assertIn("success", attempt_statuses)

    def test_openai_compatible_client_parses_chat_completion_shape(self) -> None:
        client = OpenAICompatibleLLMClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-test",
            timeout_seconds=30,
        )
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse({"choices": [{"message": {"content": '{"cards":[]}'}}]})):
            payload = client.chat_json("system", "user")
        self.assertEqual(payload, {"cards": []})

    def test_anthropic_client_parses_messages_shape(self) -> None:
        client = AnthropicLLMClient(
            base_url="https://api.anthropic.com/v1",
            api_key="test-key",
            model="claude-test",
            timeout_seconds=30,
            anthropic_version="2023-06-01",
        )
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse({"content": [{"type": "text", "text": '{"cards":[]}' }]})):
            payload = client.chat_json("system", "user")
        self.assertEqual(payload, {"cards": []})

    def test_gemini_client_parses_generate_content_shape(self) -> None:
        client = GeminiLLMClient(
            base_url="https://generativelanguage.googleapis.com",
            api_key="test-key",
            model="gemini-2.0-flash",
            timeout_seconds=30,
            api_version="v1beta",
        )
        with patch(
            "urllib.request.urlopen",
            return_value=FakeHTTPResponse({"candidates": [{"content": {"parts": [{"text": '{"cards":[]}' }]}}]}),
        ):
            payload = client.chat_json("system", "user")
        self.assertEqual(payload, {"cards": []})

    def test_openai_client_surfaces_dns_errors_with_endpoint_context(self) -> None:
        client = OpenAICompatibleLLMClient(
            base_url="https://bad-host.invalid/v1",
            api_key="test-key",
            model="gpt-test",
            timeout_seconds=30,
        )
        dns_error = urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))
        with patch("urllib.request.urlopen", side_effect=dns_error):
            with self.assertRaises(LLMGenerationError) as context:
                client.chat_json("system", "user")
        self.assertIn("DNS lookup failed", str(context.exception))
        self.assertIn("bad-host.invalid", str(context.exception))

    def test_openai_client_retries_transient_url_error_then_succeeds(self) -> None:
        client = OpenAICompatibleLLMClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-test",
            timeout_seconds=30,
        )
        transient_error = urllib.error.URLError(TimeoutError("timed out"))
        with patch(
            "urllib.request.urlopen",
            side_effect=[transient_error, FakeHTTPResponse({"choices": [{"message": {"content": '{"cards":[]}'}}]})],
        ) as mock_open, patch("app.llm.time.sleep", return_value=None) as mock_sleep:
            payload = client.chat_json("system", "user")
        self.assertEqual(payload, {"cards": []})
        self.assertEqual(mock_open.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_openai_client_retries_429_then_succeeds(self) -> None:
        client = OpenAICompatibleLLMClient(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model="gpt-test",
            timeout_seconds=30,
        )
        throttled = urllib.error.HTTPError(
            url="https://api.example.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0"},
            fp=None,
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=[throttled, FakeHTTPResponse({"choices": [{"message": {"content": '{"cards":[]}'}}]})],
        ) as mock_open, patch("app.llm.time.sleep", return_value=None) as mock_sleep:
            payload = client.chat_json("system", "user")
        self.assertEqual(payload, {"cards": []})
        self.assertEqual(mock_open.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_llm_smoke_test_returns_normalized_cards_with_stub_client(self) -> None:
        class StubClient:
            model = "stub-live-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                prompt = json.loads(user_prompt)
                if prompt["stage"] == "candidate_extraction":
                    return {
                        "cards": [
                            {
                                "title": "验证者智能体能降低多智能体流程中的不一致",
                                "section_ids": ["section_demo_1", "section_demo_2"],
                                "granularity_level": "subpattern",
                                "draft_body": "当多个执行智能体可能相互矛盾时，单独加一个验证者会显著提高稳定性。",
                                "evidence_analysis": [
                                    {
                                        "section_id": "section_demo_1",
                                        "analysis": "这段解释了为什么这个模式有效：矛盾不是自然消失的，而是要被显式检查出来。",
                                    },
                                    {
                                        "section_id": "section_demo_2",
                                        "analysis": "这段给出了量化结果，所以它不是经验判断，而是能拿来教的证据型方法。",
                                    },
                                ],
                            }
                        ],
                        "excluded_content": [
                            {
                                "label": "多智能体流水线的一般铺垫",
                                "section_ids": ["section_demo_1"],
                                "exclusion_type": "background",
                                "reason": "这些铺垫有上下文价值，但真正值得出卡的是验证者模式本身。",
                            }
                        ],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "验证者智能体能降低多智能体流程中的不一致",
                            "course_transformation": "多智能体协作：验证者模式",
                            "teachable_one_liner": "只要多个执行智能体可能互相打架，就要单独加一个验证者来找矛盾。",
                            "draft_body": "当多个执行智能体可能相互矛盾时，单独加一个验证者会显著提高稳定性。",
                            "evidence_localization": [
                                {
                                    "section_id": "section_demo_1",
                                    "quote_zh": "在多智能体流程里，只要多个执行智能体可能彼此产生冲突，就需要单独设置一个验证者来显式检查这些矛盾；否则这些矛盾不会自己消失，而会继续累积到后续步骤中。",
                                },
                                {
                                    "section_id": "section_demo_2",
                                    "quote_zh": "另一段证据进一步给出了量化结果，说明加入验证者后流程稳定性确实得到了提升；因此这不是抽象经验，而是可以直接带入教学的证据型协作模式。",
                                },
                            ],
                            "judgement": {
                                "color": "green",
                                "reason": "这是一个有量化证据支撑、又能直接转成课堂方法的协作模式。",
                            },
                        }
                    ]
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        payload = engine.smoke_test()
        self.assertEqual(payload["card_count"], 1)
        self.assertEqual(payload["cards"][0]["judgement"]["model_version"], "stub-live-model")
        self.assertIn("验证者智能体", payload["cards"][0]["title"])
        self.assertEqual(payload["cards"][0]["teachable_one_liner"], "只要多个执行智能体可能互相打架，就要单独加一个验证者来找矛盾。")
        self.assertEqual(len(payload["excluded_content"]), 1)

    def test_llm_engine_requires_teachable_shape_for_normalized_cards(self) -> None:
        class StubClient:
            model = "stub-shape-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                return {
                    "cards": [
                        {
                            "title": "这只是一个摘要式标题",
                            "section_ids": ["section_demo_1"],
                            "granularity_level": "subpattern",
                            "course_transformation": "多智能体：摘要性复述",
                            "teachable_one_liner": "",
                            "draft_body": "这只是把内容重复了一遍，没有形成可教学的一句话。",
                            "evidence_analysis": [],
                            "judgement": {
                                "color": "yellow",
                                "reason": "缺少真正面向学员的可教表达。",
                            },
                        }
                    ],
                    "excluded_content": [
                        {
                            "label": "智能体协作的一般讨论",
                            "section_ids": ["section_demo_1"],
                            "exclusion_type": "summary",
                            "reason": "这只是摘要，没有形成清晰的学员认知跃迁。",
                        }
                    ],
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        payload = engine.smoke_test()
        self.assertEqual(payload["card_count"], 0)
        self.assertEqual(len(payload["excluded_content"]), 1)

    def test_llm_engine_understanding_blocks_objects_when_paper_is_rejected(self) -> None:
        class StubClient:
            model = "stub-understanding-gate-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                self_payload = {
                    "paper_relevance_verdict": "borderline_reject",
                    "paper_relevance_reason": "主题相邻，但仍无法明确命名课程对象。",
                    "relevance_failure_type": "cannot_name_course_object",
                    "global_contribution_objects": [
                        {
                            "id": "obj_1",
                            "label": "看起来相关但其实无法成卡的对象",
                            "object_type": "framework",
                            "level_hint": "overall",
                            "evidence_section_ids": ["section_demo_1"],
                            "evidence_figure_ids": [],
                            "summary": "这个对象不应该继续。",
                            "importance_score": 0.9,
                        }
                    ],
                    "contribution_graph": [],
                    "candidate_level_hints": {"obj_1": "overall"},
                }
                if payload["stage"] != "paper_understanding":
                    raise AssertionError("This test only exercises understanding.")
                return self_payload

        engine = LLMCardEngine(self.settings, client=StubClient())
        outputs = engine.build_paper_understanding(
            topic_name="LLM roleplay",
            paper_title="Borderline Paper",
            sections=[
                {
                    "id": "section_demo_1",
                    "section_title": "Page 1",
                    "section_kind": "results",
                    "body_role": "results",
                    "selection_score": 0.8,
                    "paragraph_text": "The paper discusses evaluation setup details but still does not expose a clean course object.",
                    "page_number": 1,
                }
            ],
            figures=[],
        )

        self.assertEqual(outputs["paper_relevance_verdict"], "borderline_reject")
        self.assertEqual(outputs["relevance_failure_type"], "cannot_name_course_object")
        self.assertEqual(outputs["global_contribution_objects"], [])

    def test_llm_engine_card_plan_inherits_rejected_paper_verdict(self) -> None:
        class StubClient:
            model = "stub-plan-gate-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                if payload["stage"] != "card_planning":
                    raise AssertionError("This test only exercises planning.")
                return {
                    "paper_relevance_verdict": "off_topic_hard",
                    "paper_relevance_reason": "论文核心对象是训练算法调参，不是课程对象。",
                    "relevance_failure_type": "pure_technical_mismatch",
                    "planned_cards": [
                        {
                            "plan_id": "plan_obj_1",
                            "level": "detail",
                            "target_object_id": "obj_1",
                            "target_object_label": "训练技巧",
                            "why_valuable_for_course": "这里不该被保留",
                            "must_have_evidence_ids": ["section_demo_1"],
                            "optional_supporting_ids": [],
                            "disposition": "produce",
                            "disposition_reason": "",
                        }
                    ],
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        outputs = engine.build_card_plan(
            topic_name="LLM roleplay",
            paper_title="Pure Technical Paper",
            understanding={
                "paper_relevance_verdict": "off_topic_hard",
                "paper_relevance_reason": "论文核心对象是训练算法调参，不是课程对象。",
                "relevance_failure_type": "pure_technical_mismatch",
                "global_contribution_objects": [
                    {
                        "id": "obj_1",
                        "label": "训练技巧",
                        "level_hint": "detail",
                        "evidence_section_ids": ["section_demo_1"],
                        "evidence_figure_ids": [],
                        "importance_score": 0.8,
                    }
                ],
                "evidence_index": {"obj_1": {"section_ids": ["section_demo_1"], "figure_ids": []}},
                "candidate_level_hints": {"obj_1": "detail"},
            },
            max_cards=3,
            calibration_examples=[],
            calibration_set_name="",
        )

        self.assertEqual(outputs["paper_relevance_verdict"], "off_topic_hard")
        self.assertEqual(outputs["relevance_failure_type"], "pure_technical_mismatch")
        self.assertEqual(outputs["planned_cards"], [])
        self.assertEqual(outputs["coverage_report"]["produce"], 0)

    def test_pipeline_zero_slot_plan_does_not_fallback_to_judged_cards(self) -> None:
        run = self.repository.create_run("LLM roleplay", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("LLM roleplay")
        paper = self.repository.create_or_get_paper(
            title="Pure Technical Optimizer Paper",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::pure-technical-gate",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": "section_theta",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "We optimize training stability with a new gradient clipping schedule and ablation-heavy tuning recipe.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )

        class StubCardEngine:
            def is_enabled(self) -> bool:
                return True

            def build_paper_understanding(self, **kwargs) -> dict:
                return {
                    "paper_relevance_verdict": "off_topic_hard",
                    "paper_relevance_reason": "核心对象是训练算法调参与优化，不是当前课程对象。",
                    "relevance_failure_type": "pure_technical_mismatch",
                    "global_contribution_objects": [],
                    "contribution_graph": [],
                    "evidence_index": {},
                    "candidate_level_hints": {},
                }

            def build_card_plan(self, **kwargs) -> dict:
                return {
                    "paper_relevance_verdict": "off_topic_hard",
                    "paper_relevance_reason": "核心对象是训练算法调参与优化，不是当前课程对象。",
                    "relevance_failure_type": "pure_technical_mismatch",
                    "planned_cards": [],
                    "coverage_report": {"produce": 0, "exclude": 0, "overall": 0, "local": 0, "detail": 0},
                }

            def extract_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                sections: list[dict],
                figures: list[dict] | None = None,
                planned_cards: list[dict] | None = None,
                planning_context: dict | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                assert planned_cards == []
                assert (planning_context or {}).get("paper_relevance_verdict") == "off_topic_hard"
                return {
                    "cards": [
                        {
                            "title": "训练稳定性调参技巧",
                            "primary_section_ids": ["section_theta"],
                            "supporting_section_ids": [],
                            "granularity_level": "detail",
                            "claim_type": "method",
                            "paper_specific_object": "gradient clipping schedule",
                            "body_grounding_reason": "正文只是在讲训练技巧",
                            "evidence_level": "strong",
                            "possible_duplicate_signature": "gradient-clipping-schedule",
                            "draft_body": "这是一张不该被放行的技术调参卡。",
                            "evidence": [
                                {
                                    "section_id": "section_theta",
                                    "quote": "We optimize training stability with a new gradient clipping schedule and ablation-heavy tuning recipe.",
                                    "quote_zh": "",
                                    "page_number": 1,
                                    "analysis": "这是训练调参，不是当前课程对象。",
                                }
                            ],
                            "figure_ids": [],
                            "status": "candidate",
                            "source_plan_id": "",
                        }
                    ],
                    "excluded_content": [],
                }

            def judge_candidates(self, **kwargs) -> dict:
                return {
                    "cards": [
                        {
                            "title": "训练稳定性调参技巧",
                            "course_transformation": "课程里的训练调参案例",
                            "teachable_one_liner": "这条本来会被误放行，但现在不应该再漏出。",
                            "draft_body": "这是一张不该被放行的技术调参卡。",
                            "evidence": [
                                {
                                    "section_id": "section_theta",
                                    "quote": "We optimize training stability with a new gradient clipping schedule and ablation-heavy tuning recipe.",
                                    "quote_zh": "我们用新的梯度裁剪调度和大量调参来优化训练稳定性。",
                                    "page_number": 1,
                                    "analysis": "这是训练调参，不是当前课程对象。",
                                }
                            ],
                            "primary_section_ids": ["section_theta"],
                            "supporting_section_ids": [],
                            "figure_ids": [],
                            "claim_type": "method",
                            "paper_specific_object": "gradient clipping schedule",
                            "body_grounding_reason": "正文只是在讲训练技巧",
                            "evidence_level": "strong",
                            "possible_duplicate_signature": "gradient-clipping-schedule",
                            "source_plan_id": "",
                            "judgement": {"color": "green", "reason": "故意模拟 judgement 越界放行。"},
                            "status": "candidate",
                            "granularity_level": "detail",
                        }
                    ]
                }

        pipeline = PaperPipeline(self.settings, self.repository, card_engine=StubCardEngine())
        created_count = pipeline.build_cards(paper, topic, run["id"])

        self.assertEqual(created_count, 0)
        self.assertEqual(self.repository.list_cards(run_id=run["id"]), [])

    def test_pipeline_passes_active_calibration_examples_to_judgement(self) -> None:
        run = self.repository.create_run("Context Engineering", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("Context Engineering")
        paper = self.repository.create_or_get_paper(
            title="Standards in Practice",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::standards-practice",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
        self.repository.replace_sections(
            paper["id"],
            [
                {
                    "id": "section_delta",
                    "section_order": 1,
                    "section_title": "Page 1",
                    "paragraph_text": "Practitioners still use retired standards after successors are published.",
                    "page_number": 1,
                    "embedding": [0.0] * 64,
                }
            ],
        )
        calibration_set = self.repository.import_calibration_set(
            name="context-engineering-v1",
            description="Calibration examples for context engineering.",
            metadata={},
            examples=[
                {
                    "example_type": "positive",
                    "topic_name": "Context Engineering",
                    "audience": "operators",
                    "title": "Legacy standard inertia",
                    "source_text": "Old standards can outlive their replacements in practice.",
                    "evidence": [{"quote": "A retired standard remains widely referenced."}],
                    "expected_cards": [{"title": "Legacy-standard inertia is real"}],
                    "expected_exclusions": [],
                    "rationale": "A clear learner-facing aha.",
                    "tags": ["positive"],
                }
            ],
        )
        self.repository.activate_calibration_set(calibration_set["id"])

        class RecordingCardEngine:
            def __init__(self):
                self.judgement_inputs = None

            def is_enabled(self) -> bool:
                return True

            def extract_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                sections: list[dict],
                figures: list[dict] | None = None,
                planned_cards: list[dict] | None = None,
                planning_context: dict | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                return {
                    "cards": [
                        {
                            "title": "旧标准会在正式退役后继续主导真实工作流",
                            "granularity_level": "detail",
                            "draft_body": "这说明正式替换不等于实践替换，流程惯性本身就是值得教的内容。",
                            "evidence": [
                                {
                                    "section_id": sections[0]["id"],
                                    "quote": sections[0]["paragraph_text"],
                                    "page_number": sections[0]["page_number"],
                                    "analysis": "这是一条很典型的 learner-facing 候选：它解释了制度更新和真实执行之间为什么会错位。",
                                }
                            ],
                            "figure_ids": [],
                            "status": "candidate",
                        }
                    ],
                    "excluded_content": [],
                }

            def judge_candidates(
                self,
                *,
                topic_name: str,
                paper_title: str,
                extracted_cards: list[dict],
                figures: list[dict] | None = None,
                calibration_examples: list[dict] | None = None,
                calibration_set_name: str = "",
            ) -> dict:
                self.judgement_inputs = {
                    "topic_name": topic_name,
                    "paper_title": paper_title,
                    "figures": figures or [],
                    "calibration_examples": calibration_examples,
                    "calibration_set_name": calibration_set_name,
                }
                return {
                    "cards": [
                        {
                            "title": extracted_cards[0]["title"],
                            "granularity_level": extracted_cards[0]["granularity_level"],
                            "course_transformation": "遗留标准迁移：流程惯性讲解卡",
                            "teachable_one_liner": "官方说它退役了，不代表一线团队第二天就真的不用了。",
                            "draft_body": extracted_cards[0]["draft_body"],
                            "evidence": extracted_cards[0]["evidence"],
                            "figure_ids": [],
                            "status": "candidate",
                            "judgement": {
                                "color": "yellow",
                                "reason": "这是一张边界型但有明显教学价值的卡，需要人再判断受众熟悉度。",
                                "model_version": "recording-model",
                                "prompt_version": JUDGEMENT_PROMPT_VERSION,
                                "rubric_version": CARD_RUBRIC_VERSION,
                            },
                        }
                    ]
                }

        engine = RecordingCardEngine()
        pipeline = PaperPipeline(self.settings, self.repository, card_engine=engine)
        created_count = pipeline.build_cards(paper, topic, run["id"])

        self.assertEqual(created_count, 1)
        self.assertIsNotNone(engine.judgement_inputs)
        self.assertEqual(engine.judgement_inputs["calibration_set_name"], "context-engineering-v1")
        self.assertEqual(len(engine.judgement_inputs["calibration_examples"]), 1)
        self.assertEqual(engine.judgement_inputs["calibration_examples"][0]["title"], "Legacy standard inertia")
        self.assertEqual(engine.judgement_inputs["calibration_examples"][0]["source_text"], "Old standards can outlive their replacements in practice.")

    def test_llm_engine_judgement_prompt_uses_calibration_examples(self) -> None:
        test_case = self

        class StubClient:
            model = "stub-calibration-model"

            def __init__(self):
                self.calls = []

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                self.calls.append(payload)
                if payload["stage"] == "candidate_extraction":
                    test_case.assertEqual(payload["output_language"], "zh-CN")
                    test_case.assertEqual(payload["figures"][0]["caption"], "Legacy migration curve")
                    test_case.assertEqual(payload["calibration_examples"][0]["source_text"], "Old standards can outlive their replacements in practice.")
                    test_case.assertIn("shared_policy", payload)
                    test_case.assertIn("stage_spec", payload)
                    test_case.assertIn("stage_examples", payload)
                    return {
                        "cards": [
                            {
                                "title": "退役标准仍会在真实工作流里长期存在",
                                "section_ids": ["section_demo_1"],
                                "figure_ids": ["figure_demo_1"],
                                "granularity_level": "detail",
                                "draft_body": "这条候选强调的是流程惯性，而不是制度文本本身。",
                                "evidence_analysis": [
                                    {
                                        "section_id": "section_demo_1",
                                        "analysis": "证据指向的不是抽象制度，而是标准迁移在真实组织中的惯性。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "退役标准仍会在真实工作流里长期存在",
                            "course_transformation": "遗留标准迁移：现实阻力案例",
                            "teachable_one_liner": "制度文件更新了，不代表流程现场会立刻跟着更新。",
                            "draft_body": "这条候选强调的是流程惯性，而不是制度文本本身。",
                            "evidence_localization": [
                                {
                                    "section_id": "section_demo_1",
                                    "quote_zh": "一项已经退役的标准，在正式替代方案出现之后，仍然可能继续留在日常实际使用中；这说明制度文件的更新，并不等于流程现场也会立刻完成相同幅度的更新。",
                                }
                            ],
                            "judgement": {
                                "color": "yellow",
                                "reason": "结合校准样本，这更像是一条可教学的现实惯性洞见，而不只是摘要。",
                            },
                        }
                    ]
                }

        client = StubClient()
        engine = LLMCardEngine(self.settings, client=client)
        outputs = engine.generate_outputs(
            topic_name="Context Engineering",
            paper_title="Smoke Test Paper",
            sections=[
                {
                    "id": "section_demo_1",
                    "page_number": 1,
                    "paragraph_text": "A retired standard can remain in everyday use after a formal replacement appears.",
                }
            ],
            calibration_examples=[
                {
                    "example_type": "positive",
                    "topic_name": "Context Engineering",
                    "audience": "operators",
                    "title": "Legacy standard inertia",
                    "source_text": "Old standards can outlive their replacements in practice.",
                    "expected_cards": [{"title": "Legacy-standard inertia is real"}],
                    "expected_exclusions": [],
                    "rationale": "Useful learner-facing pattern.",
                    "tags": ["positive"],
                }
            ],
            figures=[
                {
                    "id": "figure_demo_1",
                    "figure_label": "Figure 1",
                    "caption": "Legacy migration curve",
                    "linked_section_ids": ["section_demo_1"],
                }
            ],
            calibration_set_name="context-engineering-v1",
        )

        self.assertEqual(len(outputs["cards"]), 1)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[1]["stage"], "candidate_judgement")
        self.assertEqual(client.calls[1]["active_calibration_set"], "context-engineering-v1")
        self.assertEqual(client.calls[1]["calibration_examples"][0]["title"], "Legacy standard inertia")
        self.assertIn("shared_policy", client.calls[1])
        self.assertIn("stage_spec", client.calls[1])
        self.assertIn("stage_examples", client.calls[1])
        self.assertIn(
            "complete Simplified Chinese translation",
            client.calls[1]["judgement_rules"]["evidence_translation_rules"][0],
        )

    def test_prompt_version_records_include_understanding_and_planning_stages(self) -> None:
        records = get_prompt_version_records()
        stages = {record["stage"] for record in records}

        self.assertIn("paper_understanding", stages)
        self.assertIn("card_planning", stages)
        extraction_record = next(record for record in records if record["stage"] == "candidate_extraction")
        self.assertEqual(extraction_record["details"]["shared_policy_version"], "llm-shared-policy-v4-off-topic-gate")

    def test_discovery_service_deduplicates_cross_provider_candidates(self) -> None:
        class OpenAlexProvider:
            def discover(self, topic: str, strategy: dict | None = None) -> list[dict]:
                return [
                    {
                        "provider": "openalex",
                        "strategy_type": "topic_query",
                        "query_text": topic,
                        "title": "Verifier Feedback Loops",
                        "authors": ["Ada Researcher"],
                        "publication_year": 2026,
                        "source_external_id": "https://openalex.org/W123",
                        "original_url": "https://openalex.org/W123",
                        "asset_url": "",
                        "confidence": 0.7,
                        "ids": {"doi": "10.1234/verifier", "openalex": "https://openalex.org/W123"},
                        "metadata": {"source": "openalex"},
                    }
                ]

        class SemanticScholarProvider:
            def discover(self, topic: str, strategy: dict | None = None) -> list[dict]:
                return [
                    {
                        "provider": "semantic_scholar",
                        "strategy_type": "topic_query",
                        "query_text": topic,
                        "title": "Verifier Feedback Loops",
                        "authors": ["Ada Researcher"],
                        "publication_year": 2026,
                        "source_external_id": "S2-123",
                        "original_url": "https://www.semanticscholar.org/paper/S2-123",
                        "asset_url": "https://example.com/verifier.pdf",
                        "confidence": 0.9,
                        "ids": {"doi": "10.1234/verifier", "semantic_scholar": "S2-123"},
                        "metadata": {"source": "semantic_scholar"},
                    }
                ]

        service = services_module.DiscoveryService(providers=[OpenAlexProvider(), SemanticScholarProvider()])
        results = ORIGINAL_DISCOVERY_DISCOVER(service, "verifier feedback")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["external_id"], "doi::10.1234/verifier")
        self.assertEqual(results[0]["provider"], "semantic_scholar")
        self.assertEqual(results[0]["asset_url"], "https://example.com/verifier.pdf")
        self.assertEqual(len(results[0]["discovery_sources"]), 8)
        self.assertTrue(all(source["strategy_order"] >= 1 for source in results[0]["discovery_sources"]))

    def test_topic_search_strategy_matrix_contains_ordered_families(self) -> None:
        strategies = services_module.build_topic_search_strategies("verifier feedback", current_year=2026)
        self.assertEqual([item["strategy_order"] for item in strategies], [1, 2, 3, 4])
        self.assertEqual([item["strategy_family"] for item in strategies], ["core", "mechanism", "application", "recency"])
        self.assertEqual(strategies[-1]["params"]["year_from"], 2023)

    def test_discovery_service_passes_strategy_context_to_provider(self) -> None:
        captured_calls: list[tuple[str, str, int]] = []

        class RecordingProvider:
            def discover(self, topic: str, strategy: dict | None = None) -> list[dict]:
                strategy = strategy or {}
                captured_calls.append((topic, str(strategy.get("strategy_type", "")), int(strategy.get("strategy_order", 0))))
                if int(strategy.get("strategy_order", 0)) != 1:
                    return []
                return [
                    {
                        "provider": "recording",
                        "strategy_family": strategy.get("strategy_family", "core"),
                        "strategy_type": strategy.get("strategy_type", "topic_query"),
                        "strategy_order": int(strategy.get("strategy_order", 0)),
                        "query_text": strategy.get("query_text", topic),
                        "title": "Verifier Feedback Loops",
                        "authors": ["Ada Researcher"],
                        "publication_year": 2026,
                        "source_external_id": "recording::1",
                        "original_url": "https://example.com/recording/1",
                        "asset_url": "",
                        "confidence": 0.5,
                        "strategy_params": strategy.get("params", {}),
                        "ids": {"doi": "10.1234/verifier"},
                        "metadata": {"source": "recording"},
                    }
                ]

        service = services_module.DiscoveryService(providers=[RecordingProvider()])
        results = ORIGINAL_DISCOVERY_DISCOVER(service, "verifier feedback")

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(len(captured_calls), 4)
        self.assertEqual(captured_calls[0], ("verifier feedback", "topic_query", 1))
        self.assertEqual(results[0]["discovery_sources"][0]["strategy_type"], "topic_query")
        self.assertEqual(results[0]["discovery_sources"][0]["strategy_order"], 1)

    def test_run_records_discovery_strategies_and_deduped_results(self) -> None:
        discovery_payload = [
            {
                "provider": "semantic_scholar",
                "title": "Verifier Feedback Loops",
                "authors": ["Ada Researcher"],
                "publication_year": 2026,
                "external_id": "doi::10.1234/verifier",
                "source_external_id": "S2-123",
                "original_url": "https://www.semanticscholar.org/paper/S2-123",
                "asset_url": "",
                "confidence": 0.9,
                "metadata": {"identifiers": {"doi": "10.1234/verifier"}, "primary_provider": "semantic_scholar"},
                "discovery_sources": [
                    {
                        "provider": "semantic_scholar",
                        "strategy_family": "core",
                        "strategy_type": "topic_query",
                        "strategy_order": 1,
                        "query_text": "verifier feedback",
                        "source_external_id": "S2-123",
                        "original_url": "https://www.semanticscholar.org/paper/S2-123",
                        "asset_url": "",
                        "confidence": 0.9,
                        "ids": {"doi": "10.1234/verifier", "semantic_scholar": "S2-123"},
                        "metadata": {"source": "semantic_scholar"},
                    },
                    {
                        "provider": "crossref",
                        "strategy_family": "core",
                        "strategy_type": "topic_query",
                        "strategy_order": 1,
                        "query_text": "verifier feedback",
                        "source_external_id": "10.1234/verifier",
                        "original_url": "https://doi.org/10.1234/verifier",
                        "asset_url": "",
                        "confidence": 0.55,
                        "ids": {"doi": "10.1234/verifier"},
                        "metadata": {"source": "crossref"},
                    },
                ],
            }
        ]

        with patch.object(services_module.DiscoveryService, "discover", return_value=discovery_payload), patch.object(
            services_module.PaperPipeline, "acquire_remote_asset", return_value=""
        ):
            create_response = self.client.post(
                "/api/runs",
                json={
                    "topics_text": "verifier feedback",
                    "metadata": {},
                    "local_pdfs": [],
                },
            )
            self.assertEqual(create_response.status_code, 200, create_response.text)
            run_id = create_response.json()["run"]["id"]

            for _ in range(30):
                run_response = self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(run_response.status_code, 200, run_response.text)
                if run_response.json()["run"]["status"] in {"completed", "partial_failed", "failed"}:
                    break
                time.sleep(0.2)

        payload = run_response.json()
        self.assertEqual(payload["run"]["status"], "completed")
        self.assertEqual(len(payload["discovery_strategies"]), 2)
        self.assertEqual(len(payload["discovery_results"]), 2)
        self.assertIn("progress_summary", payload["run"])
        self.assertGreaterEqual(payload["run"]["progress_summary"]["discovered"], 1)
        self.assertEqual(payload["topic_runs"][0]["stats"]["discovered_raw"], 2)
        self.assertEqual(payload["topic_runs"][0]["stats"]["deduped_candidates"], 1)
        self.assertEqual(payload["topic_runs"][0]["stats"]["duplicate_candidates_collapsed"], 1)
        self.assertEqual(payload["topic_runs"][0]["stats"]["queued_for_access"], 1)
        self.assertEqual(payload["topic_runs"][0]["current_stage"], "completed")
        self.assertEqual(payload["topic_runs"][0]["derived_status"], "completed")
        self.assertGreaterEqual(payload["topic_runs"][0]["elapsed_seconds"], 0)
        self.assertEqual(len(payload["access_queue"]), 1)
        papers = self.repository._fetchall("SELECT * FROM papers")
        self.assertEqual(len(papers), 1)
        self.assertEqual({item["dedupe_status"] for item in payload["discovery_results"]}, {"canonical", "duplicate_source"})
        saturation_response = self.client.get("/api/saturation/topics", params={"topic": "verifier feedback", "history_limit": 5})
        self.assertEqual(saturation_response.status_code, 200, saturation_response.text)
        saturation_payload = saturation_response.json()
        self.assertGreaterEqual(len(saturation_payload["snapshots"]), 1)
        self.assertGreaterEqual(len(saturation_payload["trends"]), 1)
        self.assertEqual(saturation_payload["trends"][0]["topic_name"].lower(), "verifier feedback")
        self.assertIn("latest_likely_flattening", saturation_payload["trends"][0])

    def test_run_keeps_topic_completed_when_asset_acquire_times_out(self) -> None:
        discovery_payload = [
            {
                "provider": "semantic_scholar",
                "title": "Verifier Feedback Loops",
                "authors": ["Ada Researcher"],
                "publication_year": 2026,
                "external_id": "doi::10.1234/verifier-timeout",
                "source_external_id": "S2-timeout",
                "original_url": "https://www.semanticscholar.org/paper/S2-timeout",
                "asset_url": "https://example.com/timeout.pdf",
                "confidence": 0.9,
                "metadata": {"identifiers": {"doi": "10.1234/verifier-timeout"}, "primary_provider": "semantic_scholar"},
                "discovery_sources": [
                    {
                        "provider": "semantic_scholar",
                        "strategy_family": "core",
                        "strategy_type": "topic_query",
                        "strategy_order": 1,
                        "query_text": "verifier timeout",
                        "source_external_id": "S2-timeout",
                        "original_url": "https://www.semanticscholar.org/paper/S2-timeout",
                        "asset_url": "https://example.com/timeout.pdf",
                        "confidence": 0.9,
                        "ids": {"doi": "10.1234/verifier-timeout", "semantic_scholar": "S2-timeout"},
                        "metadata": {"source": "semantic_scholar"},
                    }
                ],
            }
        ]

        with patch.object(services_module.DiscoveryService, "discover", return_value=discovery_payload), patch.object(
            services_module.PaperPipeline, "acquire_remote_asset", side_effect=TimeoutError("The read operation timed out")
        ):
            create_response = self.client.post(
                "/api/runs",
                json={"topics_text": "verifier timeout", "metadata": {}, "local_pdfs": []},
            )
            self.assertEqual(create_response.status_code, 200, create_response.text)
            run_id = create_response.json()["run"]["id"]

            for _ in range(30):
                run_response = self.client.get(f"/api/runs/{run_id}")
                self.assertEqual(run_response.status_code, 200, run_response.text)
                if run_response.json()["run"]["status"] in {"completed", "partial_failed", "failed"}:
                    break
                time.sleep(0.2)

        payload = run_response.json()
        self.assertEqual(payload["run"]["status"], "completed")
        topic_stats = payload["topic_runs"][0]["stats"]
        self.assertEqual(payload["topic_runs"][0]["status"], "completed")
        self.assertEqual(topic_stats["queued_for_access"], 1)
        self.assertGreaterEqual(topic_stats["acquisition_errors"], 1)
        self.assertIn("saturation_metrics", topic_stats)
        self.assertIn("flattening_signal", topic_stats["saturation_metrics"])
        self.assertNotIn("error", topic_stats)

    def test_topic_runs_derive_waiting_for_access_and_stalled_statuses(self) -> None:
        run = self.repository.create_run("observable topic", {"operator": "tester"})
        topic_waiting = self.repository.create_or_get_topic("observable topic waiting")
        topic_stalled = self.repository.create_or_get_topic("observable topic stalled")
        waiting_run = self.repository.create_topic_run(run["id"], topic_waiting["id"])
        stalled_run = self.repository.create_topic_run(run["id"], topic_stalled["id"])

        waiting_stats = services_module.initial_topic_run_stats()
        waiting_stats["current_stage"] = "acquisition"
        waiting_stats["queued_for_access"] = 2
        waiting_stats["accessible"] = 0
        waiting_stats["stage_started_at"] = "2026-03-07T00:00:00+00:00"
        waiting_stats["last_progress_at"] = services_module.utc_now()
        self.repository.update_topic_run(waiting_run["id"], "running", stats=waiting_stats, started=True)

        stalled_stats = services_module.initial_topic_run_stats()
        stalled_stats["current_stage"] = "parsing"
        stalled_stats["accessible"] = 3
        stalled_stats["last_progress_at"] = "2026-03-07T00:00:00+00:00"
        stalled_stats["stage_started_at"] = "2026-03-07T00:00:00+00:00"
        self.repository.update_topic_run(stalled_run["id"], "running", stats=stalled_stats, started=True)

        waiting_payload = self.client.get(f"/api/runs/{run['id']}").json()["topic_runs"]
        derived = {item["topic_name"]: item["derived_status"] for item in waiting_payload}
        self.assertEqual(derived["observable topic waiting"], "waiting_for_access")
        self.assertEqual(derived["observable topic stalled"], "stalled")

    def test_discovery_timeout_budget_marks_topic_failed_predictably(self) -> None:
        self.discovery_patcher.stop()
        self.addCleanup(self.discovery_patcher.start)
        timeout_settings = Settings(
            data_dir=self.settings.data_dir,
            db_path=self.settings.db_path,
            max_workers=2,
            discovery_timeout_seconds=1,
            google_docs_mode="artifact_only",
            llm_mode="disabled",
        )

        def slow_discover(_self, topic: str) -> list[dict]:
            time.sleep(1.5)
            return []

        with patch.object(services_module.DiscoveryService, "discover", new=slow_discover):
            timeout_client = TestClient(create_app(timeout_settings))
            create_response = timeout_client.post("/api/runs", json={"topics_text": "slow topic", "metadata": {}, "local_pdfs": []})
            self.assertEqual(create_response.status_code, 200, create_response.text)
            run_id = create_response.json()["run"]["id"]

            for _ in range(20):
                run_response = timeout_client.get(f"/api/runs/{run_id}")
                self.assertEqual(run_response.status_code, 200, run_response.text)
                if run_response.json()["run"]["status"] in {"completed", "partial_failed", "failed"}:
                    break
                time.sleep(0.2)

        payload = run_response.json()
        self.assertEqual(payload["run"]["status"], "failed")
        self.assertEqual(payload["topic_runs"][0]["status"], "failed")
        self.assertEqual(payload["topic_runs"][0]["current_stage"], "failed")
        self.assertTrue(payload["topic_runs"][0]["latest_failures"])
        self.assertIn("timed out", payload["topic_runs"][0]["latest_failures"][-1]["message"])

    def test_concurrent_runs_complete_with_small_topic_pool(self) -> None:
        self.discovery_patcher.stop()
        self.addCleanup(self.discovery_patcher.start)
        concurrent_dir = Path(self.temp_dir.name) / "concurrent"
        concurrent_settings = Settings(
            data_dir=concurrent_dir / "data",
            db_path=concurrent_dir / "data" / "paper2bullet.sqlite3",
            max_workers=2,
            google_docs_mode="artifact_only",
            llm_mode="disabled",
        )
        concurrent_settings.data_dir.mkdir(parents=True, exist_ok=True)
        init_db(concurrent_settings.db_path)
        concurrent_client = TestClient(create_app(concurrent_settings))

        artifact_path = concurrent_settings.artifacts_dir / "concurrent.pdf"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(build_minimal_pdf_bytes("Concurrent run paper body."))

        def discover_side_effect(_discovery_self, topic: str) -> list[dict]:
            slug = topic.strip().lower().replace(" ", "-")
            return [
                {
                    "provider": "semantic_scholar",
                    "title": f"{topic} Paper",
                    "authors": ["Ada Researcher"],
                    "publication_year": 2026,
                    "external_id": f"doi::10.1234/{slug}",
                    "source_external_id": f"S2-{slug}",
                    "original_url": f"https://example.com/{slug}",
                    "asset_url": f"https://example.com/{slug}.pdf",
                    "confidence": 0.9,
                    "metadata": {"identifiers": {"doi": f"10.1234/{slug}"}},
                    "discovery_sources": [
                        {
                            "provider": "semantic_scholar",
                            "strategy_family": "core",
                            "strategy_type": "topic_query",
                            "strategy_order": 1,
                            "query_text": topic,
                            "source_external_id": f"S2-{slug}",
                            "original_url": f"https://example.com/{slug}",
                            "asset_url": f"https://example.com/{slug}.pdf",
                            "confidence": 0.9,
                            "ids": {"doi": f"10.1234/{slug}", "semantic_scholar": f"S2-{slug}"},
                            "metadata": {"source": "semantic_scholar"},
                        }
                    ],
                }
            ]

        def parse_paper_side_effect(_coordinator_self, paper: dict) -> tuple[str, bool]:
            time.sleep(0.2)
            return paper["id"], False

        with patch.object(
            services_module.DiscoveryService, "discover", new=discover_side_effect
        ), patch.object(
            services_module.PaperPipeline, "acquire_remote_asset", return_value=str(artifact_path)
        ), patch.object(
            services_module.RunCoordinator, "_parse_paper", autospec=True, side_effect=parse_paper_side_effect
        ):
            run_ids = []
            for topic_name in ["concurrency topic one", "concurrency topic two"]:
                create_response = concurrent_client.post(
                    "/api/runs",
                    json={"topics_text": topic_name, "metadata": {}, "local_pdfs": []},
                )
                self.assertEqual(create_response.status_code, 200, create_response.text)
                run_ids.append(create_response.json()["run"]["id"])

            final_statuses: dict[str, str] = {}
            for _ in range(40):
                all_done = True
                for run_id in run_ids:
                    run_response = concurrent_client.get(f"/api/runs/{run_id}")
                    self.assertEqual(run_response.status_code, 200, run_response.text)
                    status = run_response.json()["run"]["status"]
                    final_statuses[run_id] = status
                    if status not in {"completed", "partial_failed", "failed"}:
                        all_done = False
                if all_done:
                    break
                time.sleep(0.2)

        self.assertTrue(all(status == "completed" for status in final_statuses.values()), final_statuses)

    def test_saturation_trends_compare_latest_and_previous_snapshots(self) -> None:
        run1 = self.repository.create_run("trend topic", {"operator": "tester"})
        run2 = self.repository.create_run("trend topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("trend topic")
        topic_run1 = self.repository.create_topic_run(run1["id"], topic["id"])
        topic_run2 = self.repository.create_topic_run(run2["id"], topic["id"])
        self.repository.create_topic_saturation_snapshot(
            run_id=run1["id"],
            topic_run_id=topic_run1["id"],
            topic_id=topic["id"],
            saturation_metrics={
                "card_count": 6,
                "near_duplicate_cards": 5,
                "same_pattern_cards": 0,
                "novel_cards": 1,
                "semantic_duplication_ratio": 0.8,
                "search_strategy_comparison": [],
                "flattening_signal": {"tail_size": 3, "tail_incremental_new_cards": [2, 1, 0], "likely_flattening": False},
                "stop_decision": {
                    "decision": "continue_search",
                    "reason": "flattening signal not met",
                    "policy": services_module.default_saturation_stop_policy(),
                },
            },
        )
        self.repository.create_topic_saturation_snapshot(
            run_id=run2["id"],
            topic_run_id=topic_run2["id"],
            topic_id=topic["id"],
            saturation_metrics={
                "card_count": 6,
                "near_duplicate_cards": 6,
                "same_pattern_cards": 0,
                "novel_cards": 0,
                "semantic_duplication_ratio": 1.0,
                "search_strategy_comparison": [],
                "flattening_signal": {"tail_size": 3, "tail_incremental_new_cards": [0, 0, 0], "likely_flattening": True},
                "stop_decision": {
                    "decision": "candidate_stop",
                    "reason": "Recent strategies yielded no new cards while duplication stayed high and stable.",
                    "policy": services_module.default_saturation_stop_policy(),
                },
            },
        )

        response = self.client.get("/api/saturation/topics", params={"topic": "trend topic", "history_limit": 5})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(len(payload["trends"]), 1)
        trend = payload["trends"][0]
        self.assertEqual(trend["topic_name"], "trend topic")
        self.assertTrue(trend["latest_likely_flattening"])
        self.assertAlmostEqual(trend["latest_duplication_ratio"], 1.0, places=4)
        self.assertAlmostEqual(trend["previous_duplication_ratio"], 0.8, places=4)
        self.assertGreater(trend["duplication_ratio_delta"], 0)
        self.assertEqual(trend["latest_stop_decision"], "candidate_stop")
        self.assertIn("duplication stayed high", trend["latest_stop_reason"])

    def test_saturation_stop_evaluator_requires_history_before_candidate_stop(self) -> None:
        metrics = {
            "semantic_duplication_ratio": 0.95,
            "flattening_signal": {
                "tail_size": 3,
                "tail_incremental_new_cards": [0, 0, 0],
                "likely_flattening": True,
            },
        }
        decision = services_module.evaluate_saturation_stop(
            current_metrics=metrics,
            previous_snapshots=[],
            policy=services_module.default_saturation_stop_policy(),
        )
        self.assertEqual(decision["decision"], "insufficient_history")
        self.assertIn("Need at least", decision["reason"])

    def test_saturation_stop_evaluator_marks_candidate_stop_when_tail_and_duplication_converge(self) -> None:
        metrics = {
            "semantic_duplication_ratio": 0.9,
            "flattening_signal": {
                "tail_size": 3,
                "tail_incremental_new_cards": [0, 0, 0],
                "likely_flattening": True,
            },
        }
        previous_snapshots = [{"semantic_duplication_ratio": 0.86}]
        decision = services_module.evaluate_saturation_stop(
            current_metrics=metrics,
            previous_snapshots=previous_snapshots,
            policy=services_module.default_saturation_stop_policy(),
        )
        self.assertEqual(decision["decision"], "candidate_stop")
        self.assertTrue(decision["checks"]["flattening_met"])
        self.assertTrue(decision["checks"]["duplication_met"])
        self.assertEqual(metrics["stop_decision"]["decision"], "candidate_stop")

    def test_run_level_primary_topic_routing_suppresses_cross_topic_repeat_cards(self) -> None:
        run = self.repository.create_run("alpha topic\nbeta topic", {"operator": "tester"})
        alpha = self.repository.create_or_get_topic("alpha topic")
        beta = self.repository.create_or_get_topic("beta topic")
        self.repository.create_topic_run(run["id"], alpha["id"])
        self.repository.create_topic_run(run["id"], beta["id"])
        paper = self.repository.create_or_get_paper(
            title="Shared Paper",
            authors=["Test Author"],
            publication_year=2026,
            external_id="paper::shared-primary-routing",
            source_type="local",
            local_path="",
            original_url="",
            access_status="open_fulltext",
            ingestion_status="artifact_ready",
            parse_status="parsed",
            artifact_path="",
        )
        self.repository.link_paper_to_topic(paper["id"], alpha["id"], run["id"], "discovery")
        self.repository.link_paper_to_topic(paper["id"], beta["id"], run["id"], "discovery")

        coordinator = services_module.RunCoordinator(self.settings, self.repository)
        with patch.object(coordinator.pipeline, "build_cards", return_value=1) as mocked_build_cards:
            built_for_primary = coordinator._build_cards_for_paper(paper, alpha, run["id"])
            built_for_secondary = coordinator._build_cards_for_paper(paper, beta, run["id"])

        self.assertEqual(built_for_primary, 1)
        self.assertEqual(built_for_secondary, 0)
        self.assertEqual(mocked_build_cards.call_count, 1)

    def test_card_detail_exposes_structured_dedupe_assistance(self) -> None:
        run = self.repository.create_run("Verifier Topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("Verifier Topic")
        shared_embedding = services_module.embedding_for_text("Verifier feedback loops improve coordination reliability.")

        for suffix in ["a", "b"]:
            paper = self.repository.create_or_get_paper(
                title=f"Verifier Paper {suffix}",
                authors=["Test Author"],
                publication_year=2026,
                external_id=f"paper::dedupe::{suffix}",
                source_type="local",
                local_path="",
                original_url="",
                access_status="open_fulltext",
                ingestion_status="ready",
                parse_status="parsed",
                artifact_path="",
            )
            self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "local_pdf")
            self.repository.replace_sections(
                paper["id"],
                [
                    {
                        "id": f"section_dedupe_{suffix}",
                        "section_order": 1,
                        "section_title": "Page 1",
                        "paragraph_text": "Verifier feedback loops improve coordination reliability.",
                        "page_number": 1,
                        "embedding": shared_embedding,
                    }
                ],
            )
            self.repository.replace_generation_outputs_for_paper_topic(
                paper["id"],
                topic["id"],
                run["id"],
                [
                    {
                        "id": f"card_dedupe_{suffix}",
                        "title": f"验证反馈回路 {suffix}",
                        "granularity_level": "detail",
                        "course_transformation": "验证反馈回路",
                        "teachable_one_liner": "把验证结果持续喂回执行链，而不是最后才审计。",
                        "draft_body": "这张卡强调验证反馈本身就是工作流的一部分。",
                        "evidence": [
                            {
                                "section_id": f"section_dedupe_{suffix}",
                                "quote": "Verifier feedback loops improve coordination reliability.",
                                "quote_zh": "验证反馈回路会提升协同可靠性。",
                                "page_number": 1,
                                "analysis": "相同模式的证据应当被拿来做近重复比较。",
                            }
                        ],
                        "figure_ids": [],
                        "status": "candidate",
                        "embedding": shared_embedding,
                        "created_at": "2026-03-06T00:00:02+00:00",
                        "judgement": {
                            "color": "green",
                            "reason": "这是明确可教学的流程模式。",
                            "model_version": "stub-model",
                            "prompt_version": JUDGEMENT_PROMPT_VERSION,
                            "rubric_version": CARD_RUBRIC_VERSION,
                            "created_at": "2026-03-06T00:00:03+00:00",
                        },
                    }
                ],
                [],
            )

        response = self.client.get("/api/cards/card_dedupe_a")
        self.assertEqual(response.status_code, 200, response.text)
        neighbors = response.json()["card"]["neighbors"]
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["id"], "card_dedupe_b")
        self.assertEqual(neighbors[0]["relationship"], "near_duplicate")
        self.assertIn("same teaching point", neighbors[0]["relationship_reason"])

    def test_topic_run_stats_include_early_saturation_metrics(self) -> None:
        run = self.repository.create_run("Verifier Topic", {"operator": "tester"})
        topic = self.repository.create_or_get_topic("Verifier Topic")
        topic_run = self.repository.create_topic_run(run["id"], topic["id"])
        coordinator = services_module.RunCoordinator(self.settings, self.repository)
        shared_embedding = services_module.embedding_for_text("Verifier feedback loops improve coordination reliability.")

        for suffix, provider in [("a", "semantic_scholar"), ("b", "crossref")]:
            paper = self.repository.create_or_get_paper(
                title=f"Verifier Metrics Paper {suffix}",
                authors=["Test Author"],
                publication_year=2026,
                external_id=f"doi::10.1234/metrics-{suffix}",
                source_type=provider,
                local_path="",
                original_url=f"https://example.com/{suffix}",
                access_status="open_fulltext",
                ingestion_status="artifact_ready",
                parse_status="parsed",
                artifact_path="",
            )
            self.repository.link_paper_to_topic(paper["id"], topic["id"], run["id"], "search")
            strategy = self.repository.create_discovery_strategy(
                run_id=run["id"],
                topic_run_id=topic_run["id"],
                topic_id=topic["id"],
                provider=provider,
                strategy_family="core",
                strategy_type="topic_query",
                strategy_order=1,
                query_text="verifier topic",
                result_count=1,
                metadata={},
            )
            self.repository.create_discovery_result(
                run_id=run["id"],
                topic_run_id=topic_run["id"],
                strategy_id=strategy["id"],
                dedupe_key=f"doi::10.1234/metrics-{suffix}",
                provider=provider,
                source_external_id=f"source::{suffix}",
                paper_title=f"Verifier Metrics Paper {suffix}",
                authors=["Test Author"],
                publication_year=2026,
                original_url=f"https://example.com/{suffix}",
                asset_url="",
                confidence=0.8,
                dedupe_status="canonical",
                paper_id=paper["id"],
                metadata={},
            )
            self.repository.replace_sections(
                paper["id"],
                [
                    {
                        "id": f"section_metrics_{suffix}",
                        "section_order": 1,
                        "section_title": "Page 1",
                        "paragraph_text": "Verifier feedback loops improve coordination reliability.",
                        "page_number": 1,
                        "embedding": shared_embedding,
                    }
                ],
            )
            self.repository.replace_generation_outputs_for_paper_topic(
                paper["id"],
                topic["id"],
                run["id"],
                [
                    {
                        "id": f"card_metrics_{suffix}",
                        "title": f"验证反馈回路指标 {suffix}",
                        "granularity_level": "detail",
                        "course_transformation": "验证反馈回路",
                        "teachable_one_liner": "把验证结果持续喂回执行链，而不是最后才审计。",
                        "draft_body": "这张卡强调验证反馈本身就是工作流的一部分。",
                        "evidence": [
                            {
                                "section_id": f"section_metrics_{suffix}",
                                "quote": "Verifier feedback loops improve coordination reliability.",
                                "quote_zh": "验证反馈回路会提升协同可靠性。",
                                "page_number": 1,
                                "analysis": "重复模式应体现在饱和指标里。",
                            }
                        ],
                        "figure_ids": [],
                        "status": "candidate",
                        "embedding": shared_embedding,
                        "created_at": "2026-03-06T00:00:02+00:00",
                        "judgement": {
                            "color": "green",
                            "reason": "这是明确可教学的流程模式。",
                            "model_version": "stub-model",
                            "prompt_version": JUDGEMENT_PROMPT_VERSION,
                            "rubric_version": CARD_RUBRIC_VERSION,
                            "created_at": "2026-03-06T00:00:03+00:00",
                        },
                    }
                ],
                [],
            )

        stats = coordinator._build_topic_run_metrics(run["id"], topic, topic_run, services_module.initial_topic_run_stats())
        metrics = stats["saturation_metrics"]
        self.assertEqual(metrics["card_count"], 2)
        self.assertEqual(metrics["near_duplicate_cards"], 2)
        self.assertEqual(metrics["novel_cards"], 0)
        self.assertGreater(metrics["semantic_duplication_ratio"], 0.9)
        self.assertEqual(metrics["independent_aha_class_count"], 1)
        self.assertEqual(metrics["reportable_aha_class_count"], 1)
        self.assertGreater(metrics["aha_class_duplication_ratio"], 0.4)
        self.assertEqual(len(metrics["search_strategy_comparison"]), 2)
        self.assertEqual({item["yielded_cards"] for item in metrics["search_strategy_comparison"]}, {1})
        self.assertEqual({item["incremental_new_cards"] for item in metrics["search_strategy_comparison"]}, {1})
        self.assertEqual([item["incremental_new_aha_classes"] for item in metrics["search_strategy_comparison"]], [1, 0])
        self.assertEqual({item["strategy_order"] for item in metrics["search_strategy_comparison"]}, {1})
        self.assertFalse(metrics["flattening_signal"]["likely_flattening"])
        self.assertEqual(metrics["flattening_signal"]["tail_incremental_new_aha_classes"], [1, 0])
        stats = coordinator._attach_topic_stop_decision(topic, stats)
        self.assertIn("saturation_stop", stats)
        self.assertEqual(stats["saturation_stop"]["decision"], "insufficient_history")

    def test_llm_engine_rejects_summary_like_evidence_translation(self) -> None:
        class StubClient:
            model = "stub-translation-gate-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                if payload["stage"] == "candidate_extraction":
                    return {
                        "cards": [
                            {
                                "title": "把验证反馈变成持续纠偏回路",
                                "section_ids": ["section_demo_1"],
                                "granularity_level": "subpattern",
                                "draft_body": "这条候选强调形式化验证不只是审计，而是持续反馈。",
                                "evidence_analysis": [
                                    {
                                        "section_id": "section_demo_1",
                                        "analysis": "这段证据讲的是把验证反馈持续喂回 LLM。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "把验证反馈变成持续纠偏回路",
                            "course_transformation": "验证驱动的对话迭代",
                            "teachable_one_liner": "验证的价值不只在最后抓 bug，还在于每轮都能纠偏。",
                            "draft_body": "这条候选强调形式化验证不只是审计，而是持续反馈。",
                            "evidence_localization": [
                                {
                                    "section_id": "section_demo_1",
                                    "quote_zh": "形式化方法可以让 vibe coding 更可靠。",
                                }
                            ],
                            "judgement": {
                                "color": "green",
                                "reason": "观点很强，但这条测试应该被翻译质量闸门拒掉。",
                            },
                        }
                    ]
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        outputs = engine.generate_outputs(
            topic_name="Vibe Coding",
            paper_title="Verification Feedback",
            sections=[
                {
                    "id": "section_demo_1",
                    "page_number": 1,
                    "paragraph_text": (
                        "Abstract Vibe coding has exploded in popularity, but developers report technical debt, security issues, "
                        "and code churn. Given LLMs' receptiveness to verification-based feedback, we argue that formal methods "
                        "can mitigate these pitfalls and make vibe coding more reliable. We advocate a side-car system that "
                        "autoformalizes specifications, validates against targets, delivers actionable feedback to the LLM, "
                        "and preserves intuitive developer control."
                    ),
                }
            ],
            calibration_examples=[],
            figures=[],
            calibration_set_name="",
        )

        self.assertEqual(outputs["cards"], [])
        self.assertEqual(len(outputs["cards"]), 0)

    def test_llm_engine_keeps_supporting_evidence_without_analysis_text(self) -> None:
        class StubClient:
            model = "stub-missing-support-analysis-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                if payload["stage"] == "candidate_extraction":
                    return {
                        "cards": [
                            {
                                "title": "先统一骨架，再并行展开",
                                "primary_section_ids": ["section_demo_1"],
                                "supporting_section_ids": ["section_demo_2"],
                                "section_ids": ["section_demo_1", "section_demo_2"],
                                "granularity_level": "subpattern",
                                "claim_type": "method",
                                "draft_body": "先给多个代理统一一个骨架，再分别并行展开，能减少相互冲突。",
                                "evidence_analysis": [
                                    {
                                        "section_id": "section_demo_1",
                                        "analysis": "主证据明确描述了先骨架后并行扩写的流程。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "先统一骨架，再并行展开",
                            "course_transformation": "并行扩写流程模板",
                            "teachable_one_liner": "先统一框架，再让多个代理并行填充细节。",
                            "draft_body": "先给多个代理统一一个骨架，再分别并行展开，能减少相互冲突。",
                            "evidence_localization": [
                                {"section_id": "section_demo_1", "quote_zh": "先产出骨架，再并行扩写。"}
                            ],
                            "judgement": {
                                "color": "green",
                                "reason": "这是可直接迁移的工作流模式。",
                            },
                        }
                    ]
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        outputs = engine.extract_candidates(
            topic_name="Agentic workflow",
            paper_title="Workflow Skeleton",
            sections=[
                {
                    "id": "section_demo_1",
                    "page_number": 1,
                    "paragraph_text": "The workflow first creates an outline and then expands each point in parallel.",
                },
                {
                    "id": "section_demo_2",
                    "page_number": 1,
                    "paragraph_text": "Agents share structure information while expanding their assigned points.",
                },
            ],
            calibration_examples=[],
            figures=[],
            calibration_set_name="",
        )

        self.assertEqual(len(outputs["cards"]), 1)
        self.assertEqual(outputs["cards"][0]["title"], "先统一骨架，再并行展开")
        self.assertEqual(
            outputs["cards"][0]["evidence"][0]["quote"],
            "The workflow first creates an outline and then expands each point in parallel.",
        )

    def test_llm_engine_extract_candidates_keeps_full_paragraph_evidence(self) -> None:
        long_paragraph = (
            "The workflow first creates an outline and then expands each point in parallel while preserving a shared structure. "
            "Each agent receives the same scaffold, keeps its local reasoning visible, and synchronizes key decisions through a common checkpoint. "
            "This reduces conflict between branches, lowers rewrite cost, and makes it easier to audit where a bad decision first entered the workflow. "
            "The paper emphasizes that this works because the outline is not a summary artifact but the control surface for later expansion, review, and correction."
        )

        class StubClient:
            model = "stub-full-paragraph-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                if payload["stage"] == "candidate_extraction":
                    return {
                        "cards": [
                            {
                                "title": "先统一骨架，再并行展开",
                                "section_ids": ["section_demo_1"],
                                "granularity_level": "subpattern",
                                "draft_body": "这张卡强调骨架是后续并行扩写的控制面。",
                                "evidence_analysis": [
                                    {
                                        "section_id": "section_demo_1",
                                        "analysis": "这里讲清楚了为什么骨架不是摘要，而是后续扩写和纠偏的控制面。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                raise AssertionError("This test only exercises extraction.")

        engine = LLMCardEngine(self.settings, client=StubClient())
        outputs = engine.extract_candidates(
            topic_name="Agentic workflow",
            paper_title="Workflow Skeleton",
            sections=[
                {
                    "id": "section_demo_1",
                    "page_number": 1,
                    "paragraph_text": long_paragraph,
                }
            ],
            calibration_examples=[],
            figures=[],
            calibration_set_name="",
        )

        self.assertEqual(len(outputs["cards"]), 1)
        self.assertEqual(outputs["cards"][0]["evidence"][0]["quote"], long_paragraph)
        self.assertNotIn("...", outputs["cards"][0]["evidence"][0]["quote"])

    def test_llm_engine_keeps_supporting_evidence_without_localized_quote(self) -> None:
        class StubClient:
            model = "stub-missing-support-quote-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                payload = json.loads(user_prompt)
                if payload["stage"] == "candidate_extraction":
                    return {
                        "cards": [
                            {
                                "title": "先统一骨架，再并行展开",
                                "primary_section_ids": ["section_demo_1"],
                                "supporting_section_ids": ["section_demo_2"],
                                "section_ids": ["section_demo_1", "section_demo_2"],
                                "granularity_level": "subpattern",
                                "claim_type": "method",
                                "draft_body": "先给多个代理统一一个骨架，再分别并行展开，能减少相互冲突。",
                                "evidence_analysis": [
                                    {
                                        "section_id": "section_demo_1",
                                        "analysis": "主证据明确描述了先骨架后并行扩写的流程。",
                                    }
                                ],
                            }
                        ],
                        "excluded_content": [],
                    }
                return {
                    "cards": [
                        {
                            "candidate_index": 0,
                            "title": "先统一骨架，再并行展开",
                            "course_transformation": "并行扩写流程模板",
                            "teachable_one_liner": "先统一框架，再让多个代理并行填充细节。",
                            "draft_body": "先给多个代理统一一个骨架，再分别并行展开，能减少相互冲突。",
                            "evidence_localization": [
                                {
                                    "section_id": "section_demo_1",
                                    "quote_zh": "工作流先生成一个骨架，然后把每个要点并行展开。",
                                }
                            ],
                            "judgement": {
                                "color": "green",
                                "reason": "这是可直接迁移的工作流模式。",
                            },
                        }
                    ]
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        outputs = engine.generate_outputs(
            topic_name="Agentic workflow",
            paper_title="Workflow Skeleton",
            sections=[
                {
                    "id": "section_demo_1",
                    "page_number": 1,
                    "paragraph_text": "The workflow first creates an outline and then expands each point in parallel.",
                },
                {
                    "id": "section_demo_2",
                    "page_number": 1,
                    "paragraph_text": "Agents share structure information while expanding their assigned points.",
                },
            ],
            calibration_examples=[],
            figures=[],
            calibration_set_name="",
        )

        self.assertEqual(len(outputs["cards"]), 1)
        self.assertEqual(outputs["cards"][0]["judgement"]["color"], "green")

    def test_import_calibration_examples_script_persists_set(self) -> None:
        source_json = Path(self.temp_dir.name) / "calibration-set.json"
        source_json.write_text(
            json.dumps(
                {
                    "name": "script-imported-v1",
                    "description": "Imported from script.",
                    "metadata": {"source": "script"},
                    "examples": [
                        {
                            "example_type": "boundary",
                            "topic_name": "LLM agent",
                            "audience": "operators",
                            "title": "Borderline evidence case",
                            "source_text": "Some evidence exists, but the transfer distance is debatable.",
                            "evidence": [{"quote": "The paper reports a partial result."}],
                            "expected_cards": [{"title": "Borderline orchestration lesson"}],
                            "expected_exclusions": [{"label": "Background setup", "exclusion_type": "background"}],
                            "rationale": "Useful for yellow-zone calibration.",
                            "tags": ["boundary"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["P2B_DATA_DIR"] = str(self.settings.data_dir)
        env["P2B_DB_PATH"] = str(self.settings.db_path)
        result = subprocess.run(
            [sys.executable, "scripts/import_calibration_examples.py", str(source_json), "--activate"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["calibration_set"]["status"], "active")

        active_set = self.repository.get_active_calibration_set()
        self.assertIsNotNone(active_set)
        self.assertEqual(active_set["name"], "script-imported-v1")
        self.assertEqual(len(active_set["examples"]), 1)


if __name__ == "__main__":
    unittest.main()
