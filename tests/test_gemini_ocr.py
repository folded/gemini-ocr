import pathlib
from unittest.mock import AsyncMock, patch

import pytest

# Create a dummy PDF for testing
from fitz import open as fitz_open

from gemini_ocr.document import BBox, BoundingBox
from gemini_ocr.gemini_ocr import OcrResult, process_document
from gemini_ocr.settings import Settings

dummy_pdf_path = pathlib.Path("test.pdf")
if not dummy_pdf_path.exists():
    doc = fitz_open()
    doc.new_page()
    doc.save(dummy_pdf_path)

# Dummy settings for the test
TEST_SETTINGS = Settings(
    project="test-project",
    location="test-location",
    layout_processor_id="test-processor",
    ocr_processor_id="test-processor",
    markdown_page_batch_size=2,
    ocr_page_batch_size=2,
)


@pytest.mark.asyncio
async def test_process_document_full_flow() -> None:
    """
    Tests the full process_document flow, mocking the external API calls.
    """
    # 1. Mock the input data and API responses
    test_pdf_path = dummy_pdf_path

    # Mock return value for _generate_markdown_for_batch
    mock_markdown_batch = "<!--- page: 1 -->\nThis is a test document."

    # Mock return value for _get_ocr_for_page
    mock_bbox_page_1 = [
        BoundingBox(page=1, rect=BBox(10, 20, 30, 40), text="test document"),
    ]

    # 2. Patch the external-facing functions in the gemini_ocr module
    with (
        patch("gemini_ocr.gemini_ocr._generate_markdown_for_chunk", new_callable=AsyncMock) as mock_md,
        patch("gemini_ocr.docai_ocr.generate_bounding_boxes", new_callable=AsyncMock) as mock_ocr,
    ):
        # Set the return values for the mocks
        mock_md.return_value = mock_markdown_batch
        mock_ocr.return_value = mock_bbox_page_1

        # 3. Call the function under test
        result = await process_document(
            test_pdf_path,
            settings=TEST_SETTINGS,
        )

        # 4. Assert the results
        # Check that the mock functions were called correctly
        mock_md.assert_called()
        mock_ocr.assert_called()

        # Check the content of the OcrResult
        # Check the content of the OcrResult
        assert isinstance(result, OcrResult)

        # Check the assignment map
        assert isinstance(result.bounding_boxes, dict)
        assert len(result.bounding_boxes) == 1
        assert mock_bbox_page_1[0] in result.bounding_boxes

        # Verify the span (start, end) matches the text position
        start, end = result.bounding_boxes[mock_bbox_page_1[0]]
        assert mock_markdown_batch[start:end] == "test document"


# Cleanup the dummy file
def teardown_module() -> None:
    if dummy_pdf_path.exists():
        dummy_pdf_path.unlink()
