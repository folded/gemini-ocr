from unittest import mock

import anchorite


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
