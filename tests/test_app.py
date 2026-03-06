"""
This file contains Phase 0 integration tests for the Paper to Bullet application.
Main tests: local PDF intake, card generation, review actions, and export artifact creation.
Data structures: temporary app settings, a minimal PDF fixture, and API-level assertions.
"""
from __future__ import annotations

import json
import os
import socket
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
    AnthropicLLMClient,
    GeminiLLMClient,
    LLMGenerationError,
    LLMCardEngine,
    OpenAICompatibleLLMClient,
)
from app.main import create_app
from app.db import init_db
from app.services import PaperPipeline, PdfParser, Repository, split_paragraphs


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
        )
        self.client = TestClient(create_app(self.settings))
        init_db(self.settings.db_path)
        self.repository = Repository(self.settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_local_pdf_to_export_flow(self) -> None:
        source_pdf = Path(self.temp_dir.name) / "adaptive-selling.pdf"
        source_pdf.write_bytes(
            build_minimal_pdf_bytes(
                "We find that adaptive selling beats rigid scripts by 24 percent in perceived customer fit."
            )
        )

        class StubClient:
            model = "stub-api-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                prompt = json.loads(user_prompt)
                section_id = prompt["sections"][0]["section_id"]
                return {
                    "cards": [
                        {
                            "title": "Adaptive selling improves perceived customer fit",
                            "section_ids": [section_id],
                            "granularity_level": "detail",
                            "course_transformation": "adaptive selling: evidence-backed talking point",
                            "teachable_one_liner": "Adaptive selling works because it fits the customer better than rigid scripts.",
                            "draft_body": "Adaptive selling can outperform rigid scripts when fit matters.",
                            "evidence_analysis": [
                                {
                                    "section_id": section_id,
                                    "analysis": "This gives a measurable, learner-facing contrast between adaptive selling and rigid scripts.",
                                }
                            ],
                            "judgement": {
                                "color": "green",
                                "reason": "Direct finding with measurable evidence.",
                            },
                        }
                    ],
                    "excluded_content": [],
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

    def test_homepage_prefills_safe_metadata_default(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('<textarea id="metadata" rows="4">{}</textarea>', response.text)

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

            def generate_outputs(self, *, topic_name: str, paper_title: str, sections: list[dict]) -> dict:
                return {
                    "cards": [
                        {
                            "title": "Agents improve planning reliability through explicit coordination",
                            "granularity_level": "subpattern",
                            "course_transformation": f"{topic_name}: coordination lesson",
                            "teachable_one_liner": "Make agents explain and check one another if you want planning reliability.",
                            "draft_body": "This card comes from the stubbed LLM card generator.",
                            "evidence": [
                                {
                                    "section_id": sections[0]["id"],
                                    "quote": sections[0]["paragraph_text"],
                                    "page_number": sections[0]["page_number"],
                                    "analysis": "The evidence links explicit coordination to a practical reliability gain.",
                                }
                            ],
                            "figure_ids": [],
                            "status": "candidate",
                            "judgement": {
                                "color": "green",
                                "reason": "Stubbed LLM judged this as a strong actionable pattern.",
                                "model_version": "stub-llm-model",
                                "prompt_version": "llm-card-generator-v1",
                                "rubric_version": "llm-card-rubric-v1",
                            },
                        }
                    ],
                    "excluded_content": [
                        {
                            "label": "General background on agents",
                            "exclusion_type": "background",
                            "reason": "Background setup is useful context but not the main learner-facing insight.",
                            "section_ids": [sections[0]["id"]],
                        }
                    ],
                }

        pipeline = PaperPipeline(self.settings, self.repository, card_engine=StubCardEngine())
        created_count = pipeline.build_cards(paper, topic, run["id"])

        self.assertEqual(created_count, 1)
        cards = self.repository.list_cards(run_id=run["id"])
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], "Agents improve planning reliability through explicit coordination")
        card_detail = self.repository.get_card(cards[0]["id"])
        self.assertEqual(card_detail["judgement"]["model_version"], "stub-llm-model")
        self.assertEqual(card_detail["teachable_one_liner"], "Make agents explain and check one another if you want planning reliability.")
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

            def generate_outputs(self, *, topic_name: str, paper_title: str, sections: list[dict]) -> dict:
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
                return {
                    "cards": [
                        {
                            "title": "Verifier agents reduce inconsistency in multi-agent workflows",
                            "section_ids": ["section_demo_1", "section_demo_2"],
                            "granularity_level": "subpattern",
                            "course_transformation": "LLM agent: verifier pattern",
                            "teachable_one_liner": "Use a verifier agent when multiple worker agents may contradict one another.",
                            "draft_body": "Use a verifier agent to catch contradictions between worker agents.",
                            "evidence_analysis": [
                                {
                                    "section_id": "section_demo_1",
                                    "analysis": "This explains the mechanism: contradictions have to be checked explicitly.",
                                },
                                {
                                    "section_id": "section_demo_2",
                                    "analysis": "This provides the measurable result that makes the pattern teachable.",
                                },
                            ],
                            "judgement": {
                                "color": "green",
                                "reason": "Actionable orchestration pattern with measurable evidence.",
                            },
                        }
                    ],
                    "excluded_content": [
                        {
                            "label": "General setup for multi-agent pipelines",
                            "section_ids": ["section_demo_1"],
                            "exclusion_type": "background",
                            "reason": "The setup matters as context, but the verifier pattern is the real teachable point.",
                        }
                    ],
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        payload = engine.smoke_test()
        self.assertEqual(payload["card_count"], 1)
        self.assertEqual(payload["cards"][0]["judgement"]["model_version"], "stub-live-model")
        self.assertEqual(payload["cards"][0]["teachable_one_liner"], "Use a verifier agent when multiple worker agents may contradict one another.")
        self.assertEqual(len(payload["excluded_content"]), 1)

    def test_llm_engine_requires_teachable_shape_for_normalized_cards(self) -> None:
        class StubClient:
            model = "stub-shape-model"

            def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
                return {
                    "cards": [
                        {
                            "title": "This is only a summary-like title",
                            "section_ids": ["section_demo_1"],
                            "granularity_level": "subpattern",
                            "course_transformation": "LLM agent: summary",
                            "teachable_one_liner": "",
                            "draft_body": "This repeats the content without a teachable articulation.",
                            "evidence_analysis": [],
                            "judgement": {
                                "color": "yellow",
                                "reason": "Weak learner-facing articulation.",
                            },
                        }
                    ],
                    "excluded_content": [
                        {
                            "label": "General discussion of agent orchestration",
                            "section_ids": ["section_demo_1"],
                            "exclusion_type": "summary",
                            "reason": "This is only a summary and does not create a clear learner-facing insight.",
                        }
                    ],
                }

        engine = LLMCardEngine(self.settings, client=StubClient())
        payload = engine.smoke_test()
        self.assertEqual(payload["card_count"], 0)
        self.assertEqual(len(payload["excluded_content"]), 1)


if __name__ == "__main__":
    unittest.main()
