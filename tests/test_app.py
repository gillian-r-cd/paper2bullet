"""
This file contains Phase 0 integration tests for the Paper to Bullet application.
Main tests: local PDF intake, card generation, review actions, and export artifact creation.
Data structures: temporary app settings, a minimal PDF fixture, and API-level assertions.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
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
    JUDGEMENT_PROMPT_VERSION,
    AnthropicLLMClient,
    GeminiLLMClient,
    LLMGenerationError,
    LLMCardEngine,
    OpenAICompatibleLLMClient,
)
from app.main import create_app
from app.db import ensure_migrations, get_connection, init_db
from app.services import EvaluationService, PaperPipeline, PdfParser, Repository, split_paragraphs


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
    ) -> dict:
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
                    "figure_ids": [],
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
            self.assertIn("证据原文（EN）", markdown_text)
            self.assertIn("证据译文（ZH）", markdown_text)
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
        self.assertIn("Promote", response.text)
        self.assertIn('id="export-card-picker"', response.text)
        self.assertIn('<select id="export-run-id">', response.text)
        self.assertNotIn('id="review-item-detail"', response.text)

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
                calibration_examples: list[dict],
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
                calibration_examples: list[dict],
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
                calibration_examples: list[dict],
                calibration_set_name: str = "",
            ) -> dict:
                self.judgement_inputs = {
                    "topic_name": topic_name,
                    "paper_title": paper_title,
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
        self.assertIn(
            "complete Simplified Chinese translation",
            client.calls[1]["judgement_rules"]["evidence_translation_rules"][0],
        )

    def test_discovery_service_deduplicates_cross_provider_candidates(self) -> None:
        self.discovery_patcher.stop()
        self.addCleanup(self.discovery_patcher.start)

        class OpenAlexProvider:
            def discover(self, topic: str) -> list[dict]:
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
            def discover(self, topic: str) -> list[dict]:
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
        results = service.discover("verifier feedback")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["external_id"], "doi::10.1234/verifier")
        self.assertEqual(results[0]["provider"], "semantic_scholar")
        self.assertEqual(results[0]["asset_url"], "https://example.com/verifier.pdf")
        self.assertEqual(len(results[0]["discovery_sources"]), 2)

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
                        "strategy_type": "topic_query",
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
                        "strategy_type": "topic_query",
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
        self.assertEqual(payload["topic_runs"][0]["stats"]["discovered_raw"], 2)
        self.assertEqual(payload["topic_runs"][0]["stats"]["deduped_candidates"], 1)
        self.assertEqual(payload["topic_runs"][0]["stats"]["duplicate_candidates_collapsed"], 1)
        self.assertEqual(payload["topic_runs"][0]["stats"]["queued_for_access"], 1)
        self.assertEqual(len(payload["access_queue"]), 1)
        papers = self.repository._fetchall("SELECT * FROM papers")
        self.assertEqual(len(papers), 1)
        self.assertEqual({item["dedupe_status"] for item in payload["discovery_results"]}, {"canonical", "duplicate_source"})

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
                strategy_type="topic_query",
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
        self.assertEqual(len(metrics["search_strategy_comparison"]), 2)
        self.assertEqual({item["yielded_cards"] for item in metrics["search_strategy_comparison"]}, {1})

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
