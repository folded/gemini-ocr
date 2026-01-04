from gemini_ocr import document, settings


async def generate_markdown(
    settings: settings.Settings,
    chunk: document.DocumentChunk,
) -> str:
    raise NotImplementedError
