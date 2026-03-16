from unittest import mock

import anchorite
import pytest

from gemini_ocr import gemini as gemini_module
from gemini_ocr import gemini_ocr


def test_inference_pdf() -> None:
    data = b"%PDF-1.4\n..."

    with mock.patch("anchorite.document.pdfium.PdfDocument") as mock_pdf_doc:
        mock_pdf_doc.return_value.__len__.return_value = 1

        chunks = list(anchorite.document.chunks(data))
        assert len(chunks) == 1
        assert chunks[0].mime_type == "application/pdf"


def test_inference_png() -> None:
    data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    chunks = list(anchorite.document.chunks(data))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/png"


def test_inference_jpeg() -> None:
    data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    chunks = list(anchorite.document.chunks(data))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/jpeg"


def test_inference_webp() -> None:
    data = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    chunks = list(anchorite.document.chunks(data))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/webp"


def test_explicit_mime_type_override() -> None:
    data = b"\x89PNG\r\n\x1a\n"
    chunks = list(anchorite.document.chunks(data, mime_type="image/jpeg"))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/jpeg"


@pytest.mark.asyncio
async def test_process_document_propagates_mime_type() -> None:
    data = b"some bytes"
    mime_type = "image/png"
    provider = gemini_module.GeminiMarkdownProvider(
        project_id="p", location="us-central1", model_name="gemini-2.0-flash",
    )

    with mock.patch("anchorite.document.chunks") as mock_chunks:
        mock_chunks.return_value = iter([])
        with mock.patch("anchorite.process_document", new_callable=mock.AsyncMock):
            await gemini_ocr.process_document(data, provider, mime_type=mime_type)

        mock_chunks.assert_called_once()
        assert mock_chunks.call_args.kwargs.get("mime_type") == mime_type
