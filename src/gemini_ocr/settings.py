import dataclasses
import enum
import os
from typing import Self


class OcrMode(enum.StrEnum):
    """Processing mode."""

    GEMINI = "gemini"
    """Use Gemini for markdown generation."""
    DOCUMENTAI = "documentai"
    """Use Document AI layout mode for markdown generation."""
    DOCLING = "docling"
    """Use Docling for markdown generation."""


@dataclasses.dataclass
class Settings:
    """gemini-ocr settings."""

    location: str
    """Gemini api endpoint location (e.g. 'us-central1')."""
    layout_processor_id: str | None
    """Document AI layout processor ID (required for Document AI mode)."""
    ocr_processor_id: str | None
    """Document AI OCR processor ID."""

    project_id: str
    """GCP project ID."""
    quota_project_id: str | None = None
    """GCP quota project ID (defaults to project if None)."""
    gemini_model_name: str | None = None
    """Name of the Gemini model to use. (required for Gemini mode)"""

    mode: OcrMode = OcrMode.GEMINI
    """Processing mode to use."""

    documentai_location: str | None = None
    """DocumentAI api endpoint location (e.g. 'us', 'eu'). If `None`, infers from `location`."""

    alignment_uniqueness_threshold: float = 0.5
    """Minimum score ratio between best and second-best match."""
    alignment_min_overlap: float = 0.9
    """Minimum overlap fraction required for a valid match."""
    include_bboxes: bool = True
    """Whether to perform bounding box alignment."""
    markdown_page_batch_size: int = 10
    """Pages per batch for Markdown generation."""
    ocr_page_batch_size: int = 10
    """Pages per batch for OCR."""
    num_jobs: int = 10
    """Max concurrent jobs."""
    cache_dir: str | None = None
    """Directory to store API response cache. `None` disables caching."""
    cache_gemini: bool = True
    """Whether to cache Gemini API responses."""
    cache_docai: bool = True
    """Whether to cache DocAI API responses."""

    def get_documentai_location(self) -> str:
        if self.documentai_location is None:
            return "eu" if self.location.startswith("eu") else "us"
        return self.documentai_location

    @classmethod
    def from_env(cls, prefix: str = "GEMINI_OCR_") -> Self:
        """Create Settings from environment variables."""

        def get(key: str) -> str | None:
            return os.getenv(prefix + key.upper())

        def getdefault(key: str, default: str) -> str:
            return os.getenv(prefix + key.upper(), default)

        project_id = get("project_id")
        if project_id is None:
            raise ValueError(f"{prefix}PROJECT_ID environment variable is required.")

        return cls(
            project_id=project_id,
            location=getdefault("location", "us-central1"),
            quota_project_id=get("quota_project_id"),
            layout_processor_id=get("layout_processor_id"),
            ocr_processor_id=get("ocr_processor_id"),
            gemini_model_name=get("gemini_model_name"),
        )
