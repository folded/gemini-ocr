import os
from unittest.mock import patch

import pytest

from gemini_ocr.docai_layout import DocAIMarkdownProvider
from gemini_ocr.docai_ocr import DocAIAnchorProvider
from gemini_ocr.gemini import GeminiMarkdownProvider
from gemini_ocr.gemini_ocr import from_env


def test_from_env_gemini_mode() -> None:
    env = {
        "GEMINI_OCR_PROJECT_ID": "my-project",
        "GEMINI_OCR_LOCATION": "us-central1",
        "GEMINI_OCR_GEMINI_MODEL_NAME": "gemini-2.0-flash",
        "GEMINI_OCR_OCR_PROCESSOR_ID": "ocr-proc",
    }
    with patch.dict(os.environ, env, clear=True):
        markdown_provider, anchor_provider = from_env()

    assert isinstance(markdown_provider, GeminiMarkdownProvider)
    assert markdown_provider.project_id == "my-project"
    assert markdown_provider.model_name == "gemini-2.0-flash"
    assert isinstance(anchor_provider, DocAIAnchorProvider)
    assert anchor_provider.processor_id == "ocr-proc"


def test_from_env_documentai_mode() -> None:
    env = {
        "GEMINI_OCR_PROJECT_ID": "my-project",
        "GEMINI_OCR_LOCATION": "europe-west1",
        "GEMINI_OCR_MODE": "documentai",
        "GEMINI_OCR_LAYOUT_PROCESSOR_ID": "layout-proc",
        "GEMINI_OCR_DOCUMENTAI_LOCATION": "eu",
    }
    with patch.dict(os.environ, env, clear=True):
        markdown_provider, anchor_provider = from_env()

    assert isinstance(markdown_provider, DocAIMarkdownProvider)
    assert markdown_provider.processor_id == "layout-proc"
    assert markdown_provider.documentai_location == "eu"
    assert anchor_provider is None  # no ocr_processor_id set


def test_from_env_no_bboxes() -> None:
    env = {
        "GEMINI_OCR_PROJECT_ID": "p",
        "GEMINI_OCR_GEMINI_MODEL_NAME": "gemini-2.0-flash",
        "GEMINI_OCR_INCLUDE_BBOXES": "false",
    }
    with patch.dict(os.environ, env, clear=True):
        _, anchor_provider = from_env()

    assert anchor_provider is None


def test_from_env_missing_project_id() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="PROJECT_ID environment variable is required"),
    ):
        from_env()
