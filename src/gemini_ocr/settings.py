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

    project: str
    """GCP project name."""
    location: str
    """GCP location (e.g. 'us', 'eu')."""

    layout_processor_id: str | None
    """Document AI layout processor ID (required for Document AI mode)."""
    ocr_processor_id: str | None
    """Document AI OCR processor ID."""
    gemini_model_name: str | None = None
    """Name of the Gemini model to use. (required for Gemini mode)"""

    mode: OcrMode = OcrMode.GEMINI
    """Processing mode to use."""

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
    """Directory to store API response cache."""

    @classmethod
    def from_env(cls, prefix: str = "GEMINI_OCR_") -> Self:
        """Create Settings from environment variables."""

        def get(key: str, default: str | None = None) -> str | None:
            return os.getenv(prefix + key.upper(), default)

        # Helper to ensure we don't pass None to fields that require str
        project = get("project")
        if project is None:
            raise ValueError(f"{prefix}PROJECT environment variable is required.")

        location = get("location", "us")
        if location is None:  # Should be "us" default, but for safety
            raise ValueError(f"{prefix}LOCATION environment variable is required.")

        return cls(
            project=project,
            location=location,
            layout_processor_id=get("layout_processor_id"),
            ocr_processor_id=get("ocr_processor_id"),
            gemini_model_name=get("gemini_model_name"),
        )
