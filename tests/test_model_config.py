from unittest.mock import MagicMock, patch

import pytest

from gemini_ocr import document, gemini, settings


@pytest.mark.asyncio
async def test_generate_markdown_uses_configured_model() -> None:
    # Setup settings with custom model
    ocr_settings = settings.Settings(
        project_id="test-project",
        location="us-central1",
        layout_processor_id="layout-id",
        ocr_processor_id="ocr-id",
        gemini_model_name="gemini-1.5-pro-preview-0409",
    )

    chunk = document.DocumentChunk(
        document_sha256="hash",
        start_page=0,
        end_page=1,
        data=b"pdf-content",
        mime_type="application/pdf",
    )

    # Mock genai.Client
    with patch("google.genai.Client") as mock_client:
        mock_client_instance = mock_client.return_value
        mock_models = mock_client_instance.models
        mock_response = MagicMock()
        mock_response.text = "Markdown content"
        mock_models.generate_content.return_value = mock_response

        # Execute
        result = await gemini.generate_markdown(ocr_settings, chunk)

        # Verify
        assert result == "Markdown content"

        # Check if generate_content was called with correct model
        _args, kwargs = mock_models.generate_content.call_args
        assert kwargs["model"] == "gemini-1.5-pro-preview-0409"
