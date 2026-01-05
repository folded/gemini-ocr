import os
from unittest.mock import patch

import pytest

from gemini_ocr.settings import OcrMode, Settings


def test_settings_defaults() -> None:
    """Test default values."""
    # We need to ensure required env vars are present or mocked for this to pass validation
    with patch.dict(
        os.environ,
        {
            "GEMINI_OCR_PROJECT": "test-project",
            "GEMINI_OCR_GCP_PROJECT_ID": "test-project",
            "GEMINI_OCR_LAYOUT_PROCESSOR_ID": "layout-id",
            "GEMINI_OCR_OCR_PROCESSOR_ID": "ocr-id",
        },
        clear=True,
    ):
        s = Settings()  # Should load defaults
        assert s.project == "test-project"
        assert s.location == "us"
        assert s.mode == OcrMode.GEMINI


def test_settings_env_vars() -> None:
    """Test loading from environment variables."""
    env = {
        "GEMINI_OCR_PROJECT": "env-project",
        "GEMINI_OCR_LOCATION": "eu",
        "GEMINI_OCR_GCP_PROJECT_ID": "env-gcp",
        "GEMINI_OCR_LAYOUT_PROCESSOR_ID": "env-layout",
        "GEMINI_OCR_OCR_PROCESSOR_ID": "env-ocr",
    }
    with patch.dict(os.environ, env, clear=True):
        s = Settings()
        assert s.project == "env-project"
        assert s.location == "eu"
        assert s.gcp_project_id == "env-gcp"
        assert s.layout_processor_id == "env-layout"
        assert s.ocr_processor_id == "env-ocr"


def test_settings_explicit_overrides_env() -> None:
    """Test that explicit arguments override environment variables."""
    env = {
        "GEMINI_OCR_PROJECT": "env-project",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch.dict(
            os.environ,
            {
                "GEMINI_OCR_GCP_PROJECT_ID": "gcp",
                "GEMINI_OCR_LAYOUT_PROCESSOR_ID": "lay",
                "GEMINI_OCR_OCR_PROCESSOR_ID": "ocr",
            },
        ),
    ):
        s = Settings(project="explicit-project")
        assert s.project == "explicit-project"


def test_settings_validation_error() -> None:
    """Test validation raises error if missing config."""
    with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="project is required"):
        Settings()
