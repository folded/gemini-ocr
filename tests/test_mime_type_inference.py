from unittest import mock

import pytest

from gemini_ocr import document, gemini_ocr, settings


@pytest.fixture
def mock_settings() -> settings.Settings:
    return settings.Settings(
        project_id="test",
        location="us",
        layout_processor_id="layout",
        ocr_processor_id="ocr",
        cache_dir=None,
    )


def test_inference_pdf() -> None:
    data = b"%PDF-1.4\n..."

    with mock.patch("gemini_ocr.document.fitz") as mock_fitz:
        mock_fitz.open.return_value.__enter__.return_value = mock.Mock()
        mock_fitz.open.return_value.__len__.return_value = 1

        chunks = list(document.chunks(data))
        assert len(chunks) == 1
        assert chunks[0].mime_type == "application/pdf"


def test_inference_png() -> None:
    data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    chunks = list(document.chunks(data))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/png"


def test_inference_jpeg() -> None:
    data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    chunks = list(document.chunks(data))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/jpeg"


def test_inference_webp() -> None:
    data = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    chunks = list(document.chunks(data))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/webp"


def test_explicit_mime_type_override() -> None:
    data = b"\x89PNG\r\n\x1a\n"
    chunks = list(document.chunks(data, mime_type="image/jpeg"))
    assert len(chunks) == 1
    assert chunks[0].mime_type == "image/jpeg"


@pytest.mark.asyncio
async def test_process_document_propagates_mime_type(mock_settings: settings.Settings) -> None:
    data = b"some bytes"
    mime_type = "image/png"

    with mock.patch("gemini_ocr.gemini_ocr.extract_raw_data") as mock_extract:
        mock_extract.return_value = gemini_ocr.RawOcrData(markdown_content="", bounding_boxes=[])

        await gemini_ocr.process_document(data, settings=mock_settings, mime_type=mime_type)

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args.kwargs.get("mime_type") == mime_type
