"""Capture Document AI layout fixtures for regression tests."""

import asyncio
import json
import os
import sys
from pathlib import Path

import anchorite
import dotenv
from google.cloud import documentai

sys.path.insert(0, str(Path.cwd() / "src"))

from gemini_ocr import docai


async def capture() -> None:
    dotenv.load_dotenv()

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
    layout_processor_id = os.environ["GEMINI_OCR_LAYOUT_PROCESSOR_ID"]
    documentai_location = os.getenv("GEMINI_OCR_DOCUMENTAI_LOCATION")
    cache_dir = os.getenv("GEMINI_OCR_CACHE_DIR")

    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            return_bounding_boxes=True,
        ),
    )

    pdf_path = Path("tests/data/hubble-1929.pdf")
    chunks = list(anchorite.document.chunks(pdf_path))

    print(f"Processing {pdf_path} ({len(chunks)} chunks) with Document AI Layout...")

    documents = []
    for i, chunk in enumerate(chunks):
        print(f"Calling DocAI Layout for chunk {i}...")
        doc = await docai.process(
            project_id,
            location,
            layout_processor_id,
            process_options,
            chunk,
            documentai_location=documentai_location,
            cache_dir=cache_dir,
        )
        documents.append(type(doc).to_json(doc))

    with open("tests/fixtures/hubble_docai_layout_responses.json", "w") as f:
        json.dump(documents, f)
    print("Saved DocAI layout responses.")


if __name__ == "__main__":
    asyncio.run(capture())
