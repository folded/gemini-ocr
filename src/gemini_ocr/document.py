import dataclasses
import hashlib
import io
import mimetypes
import pathlib
from collections.abc import Iterator
from typing import NamedTuple

import fitz


class BBox(NamedTuple):
    """A bounding box tuple (top, left, bottom, right)."""

    top: int
    """Top coordinate (y-min: [0-1000])."""
    left: int
    """Left coordinate (x-min: [0-1000])."""
    bottom: int
    """Bottom coordinate (y-max: [0-1000])."""
    right: int
    """Right coordinate (x-max: [0-1000])."""


@dataclasses.dataclass(frozen=True)
class BoundingBox:
    """A text segment with its bounding box and page number."""

    text: str
    """The text content."""
    page: int
    """Page number (0-indexed)."""
    rect: BBox
    """The bounding box coordinates."""


@dataclasses.dataclass
class DocumentChunk:
    """A chunk of a document (e.g., a subset of pages extracted from a PDF)."""

    document_sha256: str
    """SHA256 hash of the original document."""
    start_page: int
    """Start page number of this chunk in the original document."""
    end_page: int
    """End page number (exclusive) of this chunk."""
    data: bytes
    """Raw bytes of the chunk (PDF or image)."""
    mime_type: str
    """MIME type of the chunk data."""


def _split_pdf(file_path: pathlib.Path, page_count: int | None = None) -> Iterator[DocumentChunk]:
    file_bytes = file_path.read_bytes()
    doc = fitz.open(io.BytesIO(file_bytes))
    doc_page_count = len(doc)
    document_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if page_count is None:
        yield DocumentChunk(document_sha256, 0, doc_page_count, file_bytes, "application/pdf")
        return

    for start_page in range(0, doc_page_count, page_count):
        new_doc = fitz.open()
        end_page = min(start_page + page_count, doc_page_count)
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
        yield DocumentChunk(document_sha256, start_page, end_page, new_doc.tobytes(), "application/pdf")
        new_doc.close()


def chunks(file_path: pathlib.Path, *, page_count: int | None = None) -> Iterator[DocumentChunk]:
    """Splits a PDF into single-page PDF chunks or reads a single image file."""
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type and mime_type.startswith("image/"):
        yield DocumentChunk(hashlib.sha256(file_path.read_bytes()).hexdigest(), 0, 0, file_path.read_bytes(), mime_type)
        return

    if mime_type != "application/pdf":
        raise ValueError(f"Unsupported file type: {mime_type}")

    yield from _split_pdf(file_path, page_count)
