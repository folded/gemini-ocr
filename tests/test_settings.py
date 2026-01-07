import os
from unittest.mock import patch

import pytest

from gemini_ocr.settings import Settings


def test_settings_from_env() -> None:
    """Test loading from environment variable using from_env factory."""
    env = {
        "GEMINI_OCR_PROJECT_ID": "env-project",
        "GEMINI_OCR_LOCATION": "eu",
        "GEMINI_OCR_LAYOUT_PROCESSOR_ID": "env-layout",
        "GEMINI_OCR_OCR_PROCESSOR_ID": "env-ocr",
    }
    with patch.dict(os.environ, env, clear=True):
        s = Settings.from_env()
        assert s.project_id == "env-project"
        assert s.get_documentai_location() == "eu"

        assert s.layout_processor_id == "env-layout"
        assert s.ocr_processor_id == "env-ocr"
        assert s.quota_project_id is None

    # Test setting QUOTA_PROJECT_ID
    env["GEMINI_OCR_QUOTA_PROJECT_ID"] = "env-quota"
    with patch.dict(os.environ, env, clear=True):
        s = Settings.from_env()
        assert s.quota_project_id == "env-quota"

    # Test overridden locations
    env_loc = {
        "GEMINI_OCR_PROJECT_ID": "p",
        "GEMINI_OCR_LOCATION": "europe-west1",
        "GEMINI_OCR_DOCUMENTAI_LOCATION": "eu",
    }
    with patch.dict(os.environ, env_loc, clear=True):
        s = Settings.from_env()
        assert s.get_documentai_location() == "eu"
        assert s.location == "europe-west1"


def test_settings_from_env_defaults() -> None:
    """Test default values when using from_env."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_OCR_PROJECT_ID": "test-project",
            # LOCATION allows defaults/fallback
        },
        clear=True,
    ):
        # layout_processor_id and ocr_processor_id return None if missing in env
        s = Settings.from_env()
        assert s.project_id == "test-project"
        assert s.get_documentai_location() == "us"  # default
        assert s.location == "us-central1"

        assert s.layout_processor_id is None
        assert s.ocr_processor_id is None


def test_settings_validation_error() -> None:
    """Test validation raises error if missing required env vars in from_env."""
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="PROJECT_ID environment variable is required"),
    ):
        Settings.from_env()
