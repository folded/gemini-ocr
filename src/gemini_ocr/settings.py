import dataclasses
import enum
import os


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

    project: str | None = None
    """GCP project name."""
    location: str = "us"
    """GCP location (e.g. 'us', 'eu')."""
    gcp_project_id: str | None = None
    """GCP project ID (can be same as project)."""
    layout_processor_id: str | None = None
    """Document AI layout processor ID."""
    ocr_processor_id: str | None = None
    """Document AI OCR processor ID."""

    mode: OcrMode = OcrMode.GEMINI
    """Processing mode to use."""
    gemini_model_name: str = "gemini-2.0-flash-exp"
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
    cache_dir: str | None = None
    """Directory to store API response cache."""

    def __post_init__(self) -> None:
        self._load_from_env()
        self._validate()

    def _load_from_env(self) -> None:
        if self.project is None:
            self.project = os.environ.get("GEMINI_OCR_PROJECT")
        if self.gcp_project_id is None:
            self.gcp_project_id = os.environ.get("GEMINI_OCR_GCP_PROJECT_ID") or self.project
        if self.layout_processor_id is None:
            self.layout_processor_id = os.environ.get("GEMINI_OCR_LAYOUT_PROCESSOR_ID")
        if self.ocr_processor_id is None:
            self.ocr_processor_id = os.environ.get("GEMINI_OCR_OCR_PROCESSOR_ID")

        if self.location == "us" and "GEMINI_OCR_LOCATION" in os.environ:
            self.location = os.environ["GEMINI_OCR_LOCATION"]

    def _validate(self) -> None:
        if not self.project:
            raise ValueError("project is required (or GEMINI_OCR_PROJECT env var)")
        if not self.location:
            raise ValueError("location is required")
        if not self.gcp_project_id:
            raise ValueError("gcp_project_id is required (or GEMINI_OCR_GCP_PROJECT_ID env var)")
        if not self.layout_processor_id:
            raise ValueError("layout_processor_id is required (or GEMINI_OCR_LAYOUT_PROCESSOR_ID env var)")
        if not self.ocr_processor_id:
            raise ValueError("ocr_processor_id is required (or GEMINI_OCR_OCR_PROCESSOR_ID env var)")
