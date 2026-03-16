import dataclasses
import enum
import os

import anchorite
import anchorite.document
import anchorite.providers

from gemini_ocr import docai_layout, docai_ocr, docling
from gemini_ocr import gemini as gemini_module


class _OcrMode(enum.StrEnum):
    GEMINI = "gemini"
    DOCUMENTAI = "documentai"
    DOCLING = "docling"


@dataclasses.dataclass
class FixedMarkdownProvider(anchorite.providers.MarkdownProvider):
    """Markdown provider that returns a fixed string."""

    markdown_content: str

    async def generate_markdown(self, _chunk: anchorite.document.DocumentChunk) -> str:
        return self.markdown_content


def from_env(
    prefix: str = "GEMINI_OCR_",
) -> tuple[anchorite.providers.MarkdownProvider, anchorite.providers.AnchorProvider | None]:
    """Build providers from environment variables."""

    def get(key: str) -> str | None:
        return os.getenv(prefix + key.upper())

    def require(key: str) -> str:
        val = get(key)
        if val is None:
            raise ValueError(f"{prefix}{key.upper()} environment variable is required.")
        return val

    def getdefault(key: str, default: str) -> str:
        return os.getenv(prefix + key.upper(), default)

    project_id = require("project_id")
    location = getdefault("location", "us-central1")
    documentai_location = get("documentai_location")
    cache_dir = get("cache_dir")
    mode = _OcrMode(getdefault("mode", _OcrMode.GEMINI))
    include_bboxes = getdefault("include_bboxes", "true").lower() in ("true", "1", "yes")

    match mode:
        case _OcrMode.GEMINI:
            markdown_provider: anchorite.providers.MarkdownProvider = gemini_module.GeminiMarkdownProvider(
                project_id=project_id,
                location=location,
                model_name=require("gemini_model_name"),
                quota_project_id=get("quota_project_id"),
                prompt=get("gemini_prompt"),
                cache_dir=cache_dir,
            )
        case _OcrMode.DOCUMENTAI:
            markdown_provider = docai_layout.DocAIMarkdownProvider(
                project_id=project_id,
                location=location,
                processor_id=require("layout_processor_id"),
                documentai_location=documentai_location,
                cache_dir=cache_dir,
            )
        case _OcrMode.DOCLING:
            markdown_provider = docling.DoclingMarkdownProvider()
        case _:
            raise ValueError(f"Unknown mode: {mode}")

    anchor_provider: anchorite.providers.AnchorProvider | None = None
    if include_bboxes:
        ocr_processor_id = get("ocr_processor_id")
        if ocr_processor_id:
            anchor_provider = docai_ocr.DocAIAnchorProvider(
                project_id=project_id,
                location=location,
                processor_id=ocr_processor_id,
                documentai_location=documentai_location,
                cache_dir=cache_dir,
            )

    return markdown_provider, anchor_provider


async def process_document(  # noqa: PLR0913
    document_input: anchorite.document.DocumentInput,
    markdown_provider: anchorite.providers.MarkdownProvider | None = None,
    anchor_provider: anchorite.providers.AnchorProvider | None = None,
    *,
    page_count: int = 10,
    mime_type: str | None = None,
    alignment_uniqueness_threshold: float = 0.5,
    alignment_min_overlap: float = 0.9,
) -> anchorite.AlignmentResult:
    """Process a document, generating annotated markdown with OCR bounding boxes."""
    if markdown_provider is None:
        markdown_provider, anchor_provider = from_env()

    chunks = anchorite.document.chunks(document_input, page_count=page_count, mime_type=mime_type)
    return await anchorite.process_document(
        chunks,
        markdown_provider,
        anchor_provider,
        alignment_uniqueness_threshold=alignment_uniqueness_threshold,
        alignment_min_overlap=alignment_min_overlap,
        renumber=True,
    )
