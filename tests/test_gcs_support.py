import pathlib
from unittest.mock import MagicMock, patch

import fitz
import pytest

from gemini_ocr import document


@pytest.fixture
def valid_pdf_bytes() -> bytes:
    doc = fitz.open()
    doc.new_page()
    return doc.tobytes()


def test_chunks_gcs_path(valid_pdf_bytes: bytes) -> None:
    """Test chunks with a GCS path using mock fsspec."""
    gcs_path = "gs://bucket/file.pdf"

    # Mock fsspec.open
    mock_file = MagicMock()
    mock_file.read.return_value = valid_pdf_bytes
    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = None

    with patch("fsspec.open") as mock_open:
        mock_open.return_value = mock_file

        # Call chunks
        chunks = list(document.chunks(gcs_path))

        # Verify fsspec.open called
        mock_open.assert_called_once_with(gcs_path, "rb")

        # Verify chunks created
        assert len(chunks) > 0
        assert chunks[0].data == valid_pdf_bytes
        assert chunks[0].mime_type == "application/pdf"


def test_chunks_local_file(tmp_path: pathlib.Path, valid_pdf_bytes: bytes) -> None:
    """Test chunks with local file path (not using fsspec logic in priority)."""
    # This ensures we didn't break local file handling
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(valid_pdf_bytes)

    chunks = list(document.chunks(pdf_path))
    assert len(chunks) > 0
    assert chunks[0].data == valid_pdf_bytes
