import os
from unittest.mock import patch

import pytest

from gemini_ocr.settings import Settings


def test_settings_from_env() -> None:
    """Test loading from environment variable using from_env factory."""
    env = {
        "GEMINI_OCR_PROJECT": "env-project",
        "GEMINI_OCR_LOCATION": "eu",
        "GEMINI_OCR_LAYOUT_PROCESSOR_ID": "env-layout",
        "GEMINI_OCR_OCR_PROCESSOR_ID": "env-ocr",
    }
    with patch.dict(os.environ, env, clear=True):
        s = Settings.from_env()
        assert s.project == "env-project"
        assert s.location == "eu"

        assert s.layout_processor_id == "env-layout"
        assert s.ocr_processor_id == "env-ocr"


def test_settings_from_env_defaults() -> None:
    """Test default values when using from_env."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_OCR_PROJECT": "test-project",
            # GCP_PROJECT_ID and LOCATION allow defaults/fallback
        },
        clear=True,
    ):
        # layout_processor_id and ocr_processor_id return None if missing in env
        s = Settings.from_env()
        assert s.project == "test-project"
        assert s.location == "us"  # default

        assert s.layout_processor_id is None
        assert s.ocr_processor_id is None


def test_settings_validation_error() -> None:
    """Test validation raises error if missing required env vars in from_env."""
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="PROJECT environment variable is required"),
    ):
        Settings.from_env()
