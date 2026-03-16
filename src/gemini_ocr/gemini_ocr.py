import enum
import os

import anchorite

from gemini_ocr import docai_layout, docai_ocr, docling
from gemini_ocr import gemini as gemini_module


class _OcrMode(enum.StrEnum):
    GEMINI = "gemini"
    DOCUMENTAI = "documentai"
    DOCLING = "docling"


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
