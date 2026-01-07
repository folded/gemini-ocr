import asyncio
import json
import os
from pathlib import Path

import dotenv
from google.cloud import documentai

from gemini_ocr import docai, document, settings


async def capture() -> None:
    # Load .env
    dotenv.load_dotenv()

    mapping = {
        "GOOGLE_OCR_PROJECT": "GEMINI_OCR_PROJECT_ID",
        "GOOGLE_OCR_LAYOUT_PARSER_PROCESSOR_ID": "GEMINI_OCR_LAYOUT_PROCESSOR_ID",
        "GOOGLE_OCR_OCR_PROCESSOR_ID": "GEMINI_OCR_OCR_PROCESSOR_ID",
        "GOOGLE_OCR_LOCATION": "GEMINI_OCR_LOCATION",
    }
    for src, dst in mapping.items():
        if os.getenv(src) and not os.getenv(dst):
            os.environ[dst] = os.getenv(src)

    pdf_path = Path("tests/data/hubble-1929.pdf")

    ocr_settings = settings.Settings.from_env()
    ocr_settings.mode = settings.OcrMode.DOCUMENTAI

    print(f"Processing with settings: {ocr_settings}")

    chunks = list(document.chunks(pdf_path, page_count=ocr_settings.markdown_page_batch_size))

    documents = []
    for i, chunk in enumerate(chunks):
        print(f"Calling DocAI Layout for chunk {i}...")

        # docai.process returns documentai.Document
        # We need the processor ID.
        if ocr_settings.layout_processor_id is None:
            raise ValueError("Layout processor ID required")

        process_options = documentai.ProcessOptions(
            layout_config=documentai.ProcessOptions.LayoutConfig(
                return_bounding_boxes=True,
            ),
        )

        doc = await docai.process(ocr_settings, process_options, ocr_settings.layout_processor_id, chunk)

        # Serialize to JSON using protojson (built-in to the class usually or via library)
        # documentai.Document is a proto-plus wrapper. verify .to_json() or similar.
        # Actually Google Proto objects usually have ._pb methods or we can use type(doc).to_json(doc)

        # Let's try standard serialization
        json_str = type(doc).to_json(doc)
        documents.append(json_str)

    # Save list of JSON strings
    with open("tests/fixtures/hubble_docai_layout_responses.json", "w") as f:
        json.dump(documents, f)

    print("Saved DocAI layout responses.")


if __name__ == "__main__":
    asyncio.run(capture())
