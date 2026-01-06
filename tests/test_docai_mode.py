import pathlib
from unittest.mock import MagicMock, patch

import fitz
import pytest
from google.cloud import documentai

from gemini_ocr import gemini_ocr, settings


@pytest.fixture
def ocr_settings() -> settings.Settings:
    return settings.Settings(
        project_id="test-project",
        location="us-central1",
        ocr_processor_id="test-processor",
        layout_processor_id="test-layout-processor",
        mode=settings.OcrMode.DOCUMENTAI,
        cache_dir=None,
    )


@patch("gemini_ocr.document.fitz.open")
@patch("gemini_ocr.docai.documentai.DocumentProcessorServiceClient")
@pytest.mark.asyncio
async def test_process_document_docai_mode(
    mock_client_class: MagicMock,
    mock_fitz_open: MagicMock,
    ocr_settings: settings.Settings,
    tmp_path: pathlib.Path,
) -> None:
    # Create a dummy PDF file
    dummy_pdf_path = tmp_path / "dummy.pdf"
    dummy_pdf_path.write_bytes(b"%PDF-1.5\n%dummy")

    # Setup Mock API Client
    mock_client = mock_client_class.return_value
    mock_client.processor_path.return_value = "projects/p/locations/l/processors/p"

    # Create a mock Document object
    mock_document = documentai.Document()
    page = documentai.Document.Page()
    page.dimension.width = 100
    page.dimension.height = 100

    # Add a Line to the Page (docai_ocr uses page.lines)
    line = documentai.Document.Page.Line()
    line.layout.text_anchor.text_segments = [documentai.Document.TextAnchor.TextSegment(start_index=0, end_index=5)]

    # Bbox for line
    v1 = documentai.NormalizedVertex(x=0.1, y=0.1)
    v2 = documentai.NormalizedVertex(x=0.2, y=0.1)
    v3 = documentai.NormalizedVertex(x=0.2, y=0.2)
    v4 = documentai.NormalizedVertex(x=0.1, y=0.2)
    line.layout.bounding_poly.normalized_vertices = [v1, v2, v3, v4]

    # Assign line to page for docai_ocr
    page.lines = [line]

    # Setup DocumentLayout for docai_layout
    layout_block = documentai.Document.DocumentLayout.DocumentLayoutBlock()
    layout_block.text_block.text = "Hello"
    layout_block.text_block.type_ = "paragraph"
    mock_document.document_layout.blocks = [layout_block]

    mock_document.text = "Hello"

    # Add a Visual Element (Image)
    image_el = documentai.Document.Page.VisualElement()
    image_el.type_ = "image"
    # 4 vertices
    iv1 = documentai.NormalizedVertex(x=0.5, y=0.5)
    iv2 = documentai.NormalizedVertex(x=0.6, y=0.5)
    iv3 = documentai.NormalizedVertex(x=0.6, y=0.6)
    iv4 = documentai.NormalizedVertex(x=0.5, y=0.6)
    image_el.layout.bounding_poly.normalized_vertices = [iv1, iv2, iv3, iv4]

    page.visual_elements = [image_el]
    mock_document.pages = [page]
    mock_process_response = MagicMock()
    mock_process_response.document = mock_document
    mock_client.process_document.return_value = mock_process_response

    # Setup Mock fitz (PyMuPDF)
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = MagicMock()
    mock_page.rect = fitz.Rect(0, 0, 1000, 1000)
    # Mock get_pixmap to return bytes
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"fake_image_bytes"
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc.tobytes.return_value = b"fake_pdf_bytes"
    mock_doc.__getitem__.return_value = mock_page
    mock_fitz_open.return_value = mock_doc

    # Run process_document
    result = await gemini_ocr.process_document(dummy_pdf_path, settings=ocr_settings)

    # Assertions
    print("Markdown Content:", result.markdown_content)

    # We now look at result.markdown_content because OcrResult has markdown_content string field.
    # The bounding_boxes is a dict mapping BoundingBox -> span (start, end).

    assert "Hello" in result.markdown_content

    # Check bounding box assignment
    assert len(result.bounding_boxes) == 1
    bbox, span = next(iter(result.bounding_boxes.items()))
    assert bbox.text == "Hello"
    assert result.markdown_content[span[0] : span[1]] == "Hello"
