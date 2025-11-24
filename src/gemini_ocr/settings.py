import dataclasses
import enum


class OcrMode(enum.StrEnum):
    """Processing mode."""

    GEMINI = "gemini"
    """Use Gemini for markdown generation."""
    DOCUMENTAI = "documentai"
    """Use Document AI layout mode for markdown generation."""
    DOCLING = "docling"
    """Use Docling for markdown generation."""


# Settings for the APIs
@dataclasses.dataclass
class Settings:
    """gemini-ocr settings."""

    project: str
    """GCP project name."""
    location: str
    """GCP location (e.g. 'us', 'eu')."""
    gcp_project_id: str
    """GCP project ID (can be same as project)."""
    layout_processor_id: str
    """Document AI layout processor ID."""
    ocr_processor_id: str
    """Document AI OCR processor ID."""

    mode: OcrMode = OcrMode.GEMINI
    """Processing mode to use."""
    gemini_model_name: str = "gemini-2.5-flash"
    """Name of the Gemini model to use."""
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
    cache_dir: str | None = ".docai_cache"
    """Directory to store API response cache."""
