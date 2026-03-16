# Gemini OCR

<img src="https://raw.githubusercontent.com/folded/gemini-ocr/main/docs/source/_static/gemini-ocr.svg" alt="gemini-ocr" width="200">

## Traceable Generative Markdown for PDFs

`gemini-ocr` provides [anchorite](https://github.com/folded/anchorite) provider
plugins that convert PDFs to traceable Markdown using Google Cloud APIs.

- **`GeminiMarkdownProvider`** — generates Markdown via the Gemini API
- **`DocAIMarkdownProvider`** — generates Markdown via Document AI Layout
- **`DocAIAnchorProvider`** — extracts bounding boxes via Document AI OCR
- **`DoclingMarkdownProvider`** — generates Markdown via Docling (stub)

## Quick Start

```python
import asyncio
from pathlib import Path

import anchorite
from gemini_ocr import DocAIAnchorProvider, GeminiMarkdownProvider

async def main():
    markdown_provider = GeminiMarkdownProvider(
        project_id="my-gcp-project",
        location="us-central1",
        model_name="gemini-2.5-flash",
    )
    anchor_provider = DocAIAnchorProvider(
        project_id="my-gcp-project",
        location="us-central1",
        processor_id="projects/.../processors/...",
    )

    chunks = anchorite.document.chunks(Path("document.pdf"))
    result = await anchorite.process_document(
        chunks, markdown_provider, anchor_provider, renumber=True
    )

    print(result.markdown_content)
    print(result.annotate())   # Markdown with inline <span data-bbox="..."> tags

asyncio.run(main())
```

## Configuration via Environment Variables

`from_env()` builds providers from environment variables, useful for
twelve-factor deployments:

```python
import anchorite
from gemini_ocr import from_env

markdown_provider, anchor_provider = from_env()
chunks = anchorite.document.chunks(Path("document.pdf"))
result = await anchorite.process_document(chunks, markdown_provider, anchor_provider)
```

| Variable                          | Description                                                      |
| :-------------------------------- | :--------------------------------------------------------------- |
| `GEMINI_OCR_PROJECT_ID`           | GCP project ID (required)                                        |
| `GEMINI_OCR_LOCATION`             | GCP location (default: `us-central1`)                            |
| `GEMINI_OCR_MODE`                 | `gemini` (default), `documentai`, or `docling`                   |
| `GEMINI_OCR_GEMINI_MODEL_NAME`    | Gemini model name (required in `gemini` mode)                    |
| `GEMINI_OCR_LAYOUT_PROCESSOR_ID`  | Document AI processor ID (required in `documentai` mode)         |
| `GEMINI_OCR_OCR_PROCESSOR_ID`     | Document AI OCR processor ID (enables bounding box extraction)   |
| `GEMINI_OCR_DOCUMENTAI_LOCATION`  | Document AI endpoint location override                           |
| `GEMINI_OCR_QUOTA_PROJECT_ID`     | Quota project override for Gemini API calls                      |
| `GEMINI_OCR_GEMINI_PROMPT`        | Additional prompt appended to the default Gemini prompt          |
| `GEMINI_OCR_CACHE_DIR`            | Directory for caching API responses                              |
| `GEMINI_OCR_INCLUDE_BBOXES`       | Set to `false` to skip bounding box extraction (default: `true`) |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
