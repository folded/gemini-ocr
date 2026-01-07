# Gemini OCR

<img src="https://raw.githubusercontent.com/folded/gemini-ocr/main/docs/source/_static/gemini-ocr.svg" alt="gemini-ocr" width="200">

## Traceable Generative Markdown for PDFs

Gemini OCR is a library designed to convert PDF documents into clean, semantic Markdown while maintaining precise traceability back to the source coordinates. It bridges the gap between the readability of Generative AI (Gemini, Document AI Chunking) and the grounded accuracy of traditional OCR (Google Document AI).

## Key Features

- **Generative Markdown**: Uses Google's Gemini Pro or Document AI Layout models to generate human-readable Markdown with proper structure (headers, tables, lists).
- **Precision Traceability**: Aligns the generated Markdown text back to the original PDF coordinates using detailed OCR data from Google Document AI.
- **Reverse-Alignment Algorithm**: Implements a robust "reverse-alignment" strategy that starts with the readable text and finds the corresponding bounding boxes, ensuring the Markdown is the ground truth for content.
- **Confidence Metrics**: (New) Includes coverage metrics to quantify how much of the Markdown content is successfully backed by OCR data.
- **Pagination Support**: Automatically handles PDF page splitting and merging logic.

## Architecture

The library processes documents in two parallel streams:

1. **Semantic Stream**: The PDF is sent to a Generative AI model (e.g., Gemini 2.5 Flash) to produce a clean Markdown representation.
2. **Positional Stream**: The PDF is sent to Google Document AI to extract raw bounding boxes and text segments.

These two streams are then merged using a custom alignment engine (`seq_smith` + `bbox_alignment.py`) which:

1. Normalizes both text sources.
2. Identifies "anchor" comparisons for reliable alignment.
3. Computes a global alignment using the anchors to constrain the search space.
4. Identifies significant gaps or mismatches.
5. Recursively re-aligns mismatched regions until a high-quality alignment is achieved.

**Key Features:**

- **Robust to Cleanliness Issues:** Handles extra headers/footers, watermarks, and noisy OCR artifacts.
- **Scale-Invariant:** Recursion ensures even small missed sections in large documents are recovered.

## Quick Start

```python
import asyncio
from pathlib import Path
from gemini_ocr import gemini_ocr, settings

async def main():
    # Configure settings
    ocr_settings = settings.Settings(
        project="my-gcp-project",
        location="us",
        gcp_project_id="my-gcp-project",
        layout_processor_id="projects/.../processors/...",
        ocr_processor_id="projects/.../processors/...",
        mode=settings.OcrMode.GEMINI,
    )

    file_path = Path("path/to/document.pdf")

    # Process the document
    result = await gemini_ocr.process_document(ocr_settings, file_path)

    # Access results
    print(f"Coverage: {result.coverage_percent:.2%}")

    # Get annotated HTML-compatible Markdown
    annotated_md = result.annotate()
    print(annotated_md[:500])  # View first 500 chars

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

The `gemini_ocr.settings.Settings` class controls the behavior:

| Parameter                        | Type      | Description                                                      |
| :------------------------------- | :-------- | :--------------------------------------------------------------- |
| `project`                        | `str`     | GCP Project Name                                                 |
| `location`                       | `str`     | GCP Location (e.g., `us`, `eu`)                                  |
| `gcp_project_id`                 | `str`     | GCP Project ID (might be same as `project`)                      |
| `layout_processor_id`            | `str`     | Document AI Processor ID for Layout (if using `DOCUMENTAI` mode) |
| `ocr_processor_id`               | `str`     | Document AI Processor ID for OCR (required for bounding boxes)   |
| `mode`                           | `OcrMode` | `GEMINI` (default), `DOCUMENTAI`, or `DOCLING`                   |
| `gemini_model_name`              | `str`     | Gemini model to use (default: `gemini-2.5-flash`)                |
| `alignment_uniqueness_threshold` | `float`   | Min score ratio for unique match (default: `0.5`)                |
| `alignment_min_overlap`          | `float`   | Min overlap fraction for valid match (default: `0.9`)            |
| `include_bboxes`                 | `bool`    | Whether to perform alignment (default: `True`)                   |
| `markdown_page_batch_size`       | `int`     | Pages per batch for Markdown generation (default: `10`)          |
| `ocr_page_batch_size`            | `int`     | Pages per batch for OCR (default: `10`)                          |
| `num_jobs`                       | `int`     | Max concurrent jobs (default: `10`)                              |
| `cache_dir`                      | `str`     | Directory to store API response cache (default: `.docai_cache`)  |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
