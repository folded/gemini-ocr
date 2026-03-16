import pathlib
from unittest.mock import MagicMock, patch

import pytest
from google.cloud import documentai

from gemini_ocr import docai_layout, docai_ocr, gemini_ocr


@patch("anchorite.document.pdfium.PdfDocument")
@patch("gemini_ocr.docai.documentai.DocumentProcessorServiceClient")
@pytest.mark.asyncio
async def test_process_document_docai_mode(
    mock_client_class: MagicMock,
    mock_pdf_doc: MagicMock,
    tmp_path: pathlib.Path,
) -> None:
    dummy_pdf_path = tmp_path / "dummy.pdf"
    dummy_pdf_path.write_bytes(b"%PDF-1.5\n%dummy")

    mock_client = mock_client_class.return_value
    mock_client.processor_path.return_value = "projects/p/locations/l/processors/p"

    mock_document = documentai.Document()
    page = documentai.Document.Page()
    page.dimension.width = 100
    page.dimension.height = 100

    line = documentai.Document.Page.Line()
    line.layout.text_anchor.text_segments = [documentai.Document.TextAnchor.TextSegment(start_index=0, end_index=5)]
    v1 = documentai.NormalizedVertex(x=0.1, y=0.1)
    v2 = documentai.NormalizedVertex(x=0.2, y=0.1)
    v3 = documentai.NormalizedVertex(x=0.2, y=0.2)
    v4 = documentai.NormalizedVertex(x=0.1, y=0.2)
    line.layout.bounding_poly.normalized_vertices = [v1, v2, v3, v4]
    page.lines = [line]

    layout_block = documentai.Document.DocumentLayout.DocumentLayoutBlock()
    layout_block.text_block.text = "Hello"
    layout_block.text_block.type_ = "paragraph"
    mock_document.document_layout.blocks = [layout_block]
    mock_document.text = "Hello"

    image_el = documentai.Document.Page.VisualElement()
    image_el.type_ = "image"
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

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_pdf_doc.return_value = mock_doc

    markdown_provider = docai_layout.DocAIMarkdownProvider(
        project_id="test-project",
        location="us-central1",
        processor_id="test-layout-processor",
    )
    anchor_provider = docai_ocr.DocAIAnchorProvider(
        project_id="test-project",
        location="us-central1",
        processor_id="test-processor",
    )

    result = await gemini_ocr.process_document(dummy_pdf_path, markdown_provider, anchor_provider)

    assert "Hello" in result.markdown_content
    assert len(result.anchor_spans) == 1
    bbox, span = next(iter(result.anchor_spans.items()))
    assert bbox.text == "Hello"
    assert result.markdown_content[span[0] : span[1]] == "Hello"
