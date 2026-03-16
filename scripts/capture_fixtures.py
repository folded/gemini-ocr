"""Capture Gemini markdown and DocAI bounding-box fixtures for regression tests."""

import asyncio
import json
import os
import sys
from pathlib import Path

import anchorite
import dotenv

sys.path.insert(0, str(Path.cwd() / "src"))

from gemini_ocr import DocAIAnchorProvider, GeminiMarkdownProvider


async def capture() -> None:
    dotenv.load_dotenv()

    # Support legacy env-var names used in CI/local setups
    mapping = {
        "GOOGLE_OCR_PROJECT": "GEMINI_OCR_PROJECT_ID",
        "GOOGLE_OCR_LAYOUT_PARSER_PROCESSOR_ID": "GEMINI_OCR_LAYOUT_PROCESSOR_ID",
        "GOOGLE_OCR_OCR_PROCESSOR_ID": "GEMINI_OCR_OCR_PROCESSOR_ID",
        "GOOGLE_OCR_LOCATION": "GEMINI_OCR_LOCATION",
    }
    for src, dst in mapping.items():
        val = os.getenv(src)
        if val and not os.getenv(dst):
            os.environ[dst] = val

    project_id = os.environ["GEMINI_OCR_PROJECT_ID"]
    location = os.getenv("GEMINI_OCR_LOCATION", "us-central1")
    model_name = os.environ["GEMINI_OCR_GEMINI_MODEL_NAME"]
    ocr_processor_id = os.environ["GEMINI_OCR_OCR_PROCESSOR_ID"]
    cache_dir = os.getenv("GEMINI_OCR_CACHE_DIR")

    markdown_provider = GeminiMarkdownProvider(
        project_id=project_id,
        location=location,
        model_name=model_name,
        cache_dir=cache_dir,
    )
    anchor_provider = DocAIAnchorProvider(
        project_id=project_id,
        location=location,
        processor_id=ocr_processor_id,
        cache_dir=cache_dir,
    )

    pdf_path = Path("tests/data/hubble-1929.pdf")
    chunks = list(anchorite.document.chunks(pdf_path))

    print(f"Processing {pdf_path} ({len(chunks)} chunks)...")

    gemini_responses = []
    for i, chunk in enumerate(chunks):
        print(f"Generating Gemini markdown for chunk {i}...")
        text = await markdown_provider.generate_markdown(chunk)
        gemini_responses.append(text)

    with open("tests/fixtures/hubble_gemini_responses.json", "w") as f:
        json.dump(gemini_responses, f)
    print("Saved Gemini responses.")

    all_chunks_bboxes = []
    for i, chunk in enumerate(chunks):
        print(f"Generating DocAI bboxes for chunk {i}...")
        anchors = await anchor_provider.generate_anchors(chunk)
        all_chunks_bboxes.append(
            [{"text": a.text, "page": a.page, "boxes": [b._asdict() for b in a.boxes]} for a in anchors],
        )

    with open("tests/fixtures/hubble_docai_bboxes.json", "w") as f:
        json.dump(all_chunks_bboxes, f)
    print("Saved DocAI bboxes.")


if __name__ == "__main__":
    asyncio.run(capture())
