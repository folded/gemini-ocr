from unittest.mock import MagicMock, patch

import pytest
from anchorite.document import DocumentChunk

from gemini_ocr.gemini import GeminiMarkdownProvider


@pytest.mark.asyncio
async def test_generate_markdown_uses_configured_model() -> None:
    provider = GeminiMarkdownProvider(
        project_id="test-project",
        location="us-central1",
        model_name="gemini-1.5-pro-preview-0409",
    )

    chunk = DocumentChunk(
        document_sha256="hash",
        start_page=0,
        end_page=1,
        data=b"pdf-content",
        mime_type="application/pdf",
    )

    with patch("google.genai.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.text = "Markdown content"
        mock_client.return_value.models.generate_content.return_value = mock_response

        result = await provider.generate_markdown(chunk)

        assert result == "Markdown content"
        _args, kwargs = mock_client.return_value.models.generate_content.call_args
        assert kwargs["model"] == "gemini-1.5-pro-preview-0409"
