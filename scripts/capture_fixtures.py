import asyncio
import json
import os
import pickle

# Add src to path so we can import gemini_ocr modules
import sys
from pathlib import Path

import dotenv

sys.path.append(str(Path.cwd() / "src"))

import typing

from gemini_ocr import docai_ocr, document, gemini, settings

# For serializing BBox


async def capture() -> None:
    # Load .env
    dotenv.load_dotenv()

    # ... (skipping some unchanged lines in between) ...

    # Re-running DocAI OCR (bboxes) is safer.
    # Note: process_document uses batched gather.
    # We'll reproduce logic from extract_raw_data roughly but per chunk.

    # Map GOOGLE_OCR_ vars to GEMINI_OCR_ vars if needed
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

    pdf_path = Path("tests/data/hubble-1929.pdf")

    ocr_settings = settings.Settings.from_env()
    # Ensure Gemini mode
    ocr_settings.mode = settings.OcrMode.GEMINI

    print(f"Processing {pdf_path}...")
    print(f"Settings: {ocr_settings}")

    chunks = list(document.chunks(pdf_path, page_count=ocr_settings.markdown_page_batch_size))

    # 1. Capture Gemini Markdown Responses
    gemini_responses = []
    for i, chunk in enumerate(chunks):
        print(f"Generating Gemini markdown for chunk {i}...")
        text = await gemini.generate_markdown(ocr_settings, chunk)
        gemini_responses.append(text)

    with open("tests/fixtures/hubble_gemini_responses.json", "w") as f:
        json.dump(gemini_responses, f)
    print("Saved Gemini responses.")

    print("Generating DocAI BBoxes...")

    all_chunks_bboxes: list[typing.Any] = []

    for i, chunk in enumerate(chunks):
        print(f"Generating DocAI bboxes for chunk {i}...")
        bboxes = await docai_ocr.generate_bounding_boxes(ocr_settings, chunk)
        all_chunks_bboxes.append(bboxes)

    with open("tests/fixtures/hubble_docai_bboxes.pkl", "wb") as f:
        pickle.dump(all_chunks_bboxes, f)

    print("Saved DocAI bboxes.")


if __name__ == "__main__":
    asyncio.run(capture())
