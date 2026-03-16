import json
import os
import pathlib
import typing
from unittest.mock import AsyncMock, patch

import anchorite
import pytest
from google.cloud import documentai  # type: ignore[import-untyped]

from gemini_ocr import docai_layout, docai_ocr
from gemini_ocr import gemini as gemini_module

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

PROJECT = "test-project"
LOCATION = "us-central1"


@pytest.mark.asyncio
async def test_hubble_regression() -> None:
    pdf_path = pathlib.Path("tests/data/hubble-1929.pdf")
    if not pdf_path.exists():
        pytest.skip("Regression test PDF not found")

    with open(FIXTURES_DIR / "hubble_gemini_responses.json") as f:
        gemini_responses = json.load(f)

    with open(FIXTURES_DIR / "hubble_docai_bboxes.json") as f_json:
        bboxes_raw = json.load(f_json)
    docai_bboxes = [
        [
            anchorite.Anchor(text=a["text"], page=a["page"], boxes=tuple(anchorite.BBox(**b) for b in a["boxes"]))
            for a in chunk
        ]
        for chunk in bboxes_raw
    ]

    async def mock_gemini_side_effect(chunk: anchorite.document.DocumentChunk) -> str:
        idx = chunk.start_page // 10
        return str(gemini_responses[idx])

    async def mock_ocr_side_effect(chunk: anchorite.document.DocumentChunk) -> list[anchorite.Anchor]:
        idx = chunk.start_page // 10
        return typing.cast("list[anchorite.Anchor]", docai_bboxes[idx])

    markdown_provider = gemini_module.GeminiMarkdownProvider(
        project_id=PROJECT,
        location=LOCATION,
        model_name="gemini-2.0-flash",
    )
    anchor_provider = docai_ocr.DocAIAnchorProvider(project_id=PROJECT, location=LOCATION, processor_id="test-ocr")

    with (
        patch.object(markdown_provider, "generate_markdown", new=AsyncMock(side_effect=mock_gemini_side_effect)),
        patch.object(anchor_provider, "generate_anchors", new=AsyncMock(side_effect=mock_ocr_side_effect)),
    ):
        chunks = anchorite.document.chunks(pdf_path)
        result = await anchorite.process_document(chunks, markdown_provider, anchor_provider, renumber=True)

    output_md = result.annotate()
    golden_path = FIXTURES_DIR / "hubble_golden.md"

    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.write_text(output_md)

    if not golden_path.exists():
        pytest.fail("Golden file not found. Run with UPDATE_GOLDEN=1 to generate it.")

    assert output_md == golden_path.read_text()


@pytest.mark.asyncio
async def test_hubble_docai_regression() -> None:
    pdf_path = pathlib.Path("tests/data/hubble-1929.pdf")
    if not pdf_path.exists():
        pytest.skip("Regression test PDF not found")

    with open(FIXTURES_DIR / "hubble_docai_layout_responses.json") as f:
        docai_responses_json = json.load(f)

    docai_responses = [
        typing.cast("documentai.Document", documentai.Document.from_json(j)) for j in docai_responses_json
    ]

    with open(FIXTURES_DIR / "hubble_docai_bboxes.json") as f_json:
        bboxes_raw = json.load(f_json)
    docai_bboxes = [
        [
            anchorite.Anchor(text=a["text"], page=a["page"], boxes=tuple(anchorite.BBox(**b) for b in a["boxes"]))
            for a in chunk
        ]
        for chunk in bboxes_raw
    ]

    async def mock_docai_side_effect(
        _project_id: str,
        _location: str,
        _processor_id: str,
        _process_options: documentai.ProcessOptions,
        chunk: anchorite.document.DocumentChunk,
        **_kwargs: object,
    ) -> documentai.Document:
        idx = chunk.start_page // 10
        return docai_responses[idx]

    async def mock_ocr_side_effect(chunk: anchorite.document.DocumentChunk) -> list[anchorite.Anchor]:
        idx = chunk.start_page // 10
        return typing.cast("list[anchorite.Anchor]", docai_bboxes[idx])

    markdown_provider = docai_layout.DocAIMarkdownProvider(
        project_id=PROJECT,
        location=LOCATION,
        processor_id="test-layout-id",
    )
    anchor_provider = docai_ocr.DocAIAnchorProvider(project_id=PROJECT, location=LOCATION, processor_id="test-ocr")

    with (
        patch("gemini_ocr.docai.process", new=AsyncMock(side_effect=mock_docai_side_effect)),
        patch.object(anchor_provider, "generate_anchors", new=AsyncMock(side_effect=mock_ocr_side_effect)),
    ):
        chunks = anchorite.document.chunks(pdf_path)
        result = await anchorite.process_document(chunks, markdown_provider, anchor_provider, renumber=True)

    output_md = result.annotate()
    golden_path = FIXTURES_DIR / "hubble_docai_golden.md"

    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.write_text(output_md)

    if not golden_path.exists():
        pytest.fail("Golden file not found. Run with UPDATE_GOLDEN=1 to generate it.")

    assert output_md == golden_path.read_text()
