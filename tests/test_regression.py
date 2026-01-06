import json
import pickle
import pathlib
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from gemini_ocr import gemini_ocr, settings, document

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
async def test_hubble_regression(regression_settings, tmp_path) -> None:
    pdf_path = pathlib.Path("tests/data/hubble-1929.pdf")
    if not pdf_path.exists():
        pytest.skip("Regression test PDF not found")

    # Load fixtures
    with open(FIXTURES_DIR / "hubble_gemini_responses.json") as f:
        gemini_responses = json.load(f)

    with open(FIXTURES_DIR / "hubble_docai_bboxes.pkl", "rb") as f:
        docai_bboxes = pickle.load(f)

    async def mock_gemini_side_effect(settings, chunk):
        idx = chunk.start_page // 10
        return gemini_responses[idx]

    async def mock_ocr_side_effect(settings, chunk):
        idx = chunk.start_page // 10
        return docai_bboxes[idx]

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

            import os

            if os.environ.get("UPDATE_GOLDEN"):
                golden_path.write_text(output_md)

            if not golden_path.exists():
                pytest.fail("Golden file not found. Run with UPDATE_GOLDEN=1 to generate it.")

            expected = golden_path.read_text()
            assert output_md == expected
