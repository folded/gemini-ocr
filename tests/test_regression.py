import json
import os
import pathlib
import pickle
import typing
from unittest.mock import AsyncMock, patch

import pytest
from google.cloud import documentai  # type: ignore[import-untyped]

from gemini_ocr import document, gemini_ocr, settings

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def regression_settings() -> settings.Settings:
    return settings.Settings(
        project_id="test-project",
        location="us-central1",
        layout_processor_id="test-layout",
        ocr_processor_id="test-ocr",
        mode=settings.OcrMode.GEMINI,
        cache_dir=None,
    )


@pytest.mark.asyncio
async def test_hubble_regression(regression_settings: settings.Settings) -> None:
    pdf_path = pathlib.Path("tests/data/hubble-1929.pdf")
    if not pdf_path.exists():
        pytest.skip("Regression test PDF not found")

    # Load fixtures
    # Load fixtures
    with open(FIXTURES_DIR / "hubble_gemini_responses.json") as f:
        gemini_responses = json.load(f)

    with open(FIXTURES_DIR / "hubble_docai_bboxes.pkl", "rb") as f_bin:
        docai_bboxes = pickle.load(f_bin)  # noqa: S301

    async def mock_gemini_side_effect(_settings: settings.Settings, chunk: document.DocumentChunk) -> str:
        idx = chunk.start_page // 10
        return str(gemini_responses[idx])

    async def mock_ocr_side_effect(
        _settings: settings.Settings,
        chunk: document.DocumentChunk,
    ) -> list[document.BoundingBox]:
        idx = chunk.start_page // 10
        return typing.cast("list[document.BoundingBox]", docai_bboxes[idx])

    # Patch
    with patch("gemini_ocr.gemini.generate_markdown", new_callable=AsyncMock) as mock_gemini:
        mock_gemini.side_effect = mock_gemini_side_effect

        with patch("gemini_ocr.docai_ocr.generate_bounding_boxes", new_callable=AsyncMock) as mock_ocr:
            mock_ocr.side_effect = mock_ocr_side_effect

            # Run
            result = await gemini_ocr.process_document(pdf_path, settings=regression_settings)

            # Annotate
            output_md = result.annotate()

            # Compare with golden
            golden_path = FIXTURES_DIR / "hubble_golden.md"

            if os.environ.get("UPDATE_GOLDEN"):
                golden_path.write_text(output_md)

            if not golden_path.exists():
                pytest.fail("Golden file not found. Run with UPDATE_GOLDEN=1 to generate it.")

            expected = golden_path.read_text()
            assert output_md == expected


@pytest.mark.asyncio
async def test_hubble_docai_regression(regression_settings: settings.Settings) -> None:
    pdf_path = pathlib.Path("tests/data/hubble-1929.pdf")
    if not pdf_path.exists():
        pytest.skip("Regression test PDF not found")

    docai_settings = regression_settings
    docai_settings.mode = settings.OcrMode.DOCUMENTAI
    docai_settings.layout_processor_id = "test-layout-id"  # Mocked anyway

    # Load fixtures
    with open(FIXTURES_DIR / "hubble_docai_layout_responses.json") as f:
        docai_responses_json = json.load(f)

    # Deserialize list of JSON strings to documentai.Document objects
    docai_responses = [
        typing.cast("documentai.Document", documentai.Document.from_json(j)) for j in docai_responses_json
    ]

    with open(FIXTURES_DIR / "hubble_docai_bboxes.pkl", "rb") as f_bin:
        docai_bboxes = pickle.load(f_bin)  # noqa: S301

    async def mock_docai_side_effect(
        _settings: settings.Settings,
        _process_options: documentai.ProcessOptions,
        _processor_id: str,
        chunk: document.DocumentChunk,
    ) -> documentai.Document:
        idx = chunk.start_page // 10
        return docai_responses[idx]

    async def mock_ocr_side_effect(
        _settings: settings.Settings,
        chunk: document.DocumentChunk,
    ) -> list[document.BoundingBox]:
        idx = chunk.start_page // 10
        return typing.cast("list[document.BoundingBox]", docai_bboxes[idx])

    # Patch
    with patch("gemini_ocr.docai.process", new_callable=AsyncMock) as mock_process:
        mock_process.side_effect = mock_docai_side_effect

        with patch("gemini_ocr.docai_ocr.generate_bounding_boxes", new_callable=AsyncMock) as mock_ocr:
            mock_ocr.side_effect = mock_ocr_side_effect

            # Run
            result = await gemini_ocr.process_document(pdf_path, settings=docai_settings)

            # Annotate
            output_md = result.annotate()

            # Compare with golden
            golden_path = FIXTURES_DIR / "hubble_docai_golden.md"

            if os.environ.get("UPDATE_GOLDEN"):
                golden_path.write_text(output_md)

            if not golden_path.exists():
                pytest.fail("Golden file not found. Run with UPDATE_GOLDEN=1 to generate it.")

            expected = golden_path.read_text()
            assert output_md == expected
