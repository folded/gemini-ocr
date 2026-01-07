import asyncio
import itertools
import json
import numpy as np
import os
import pickle
import dotenv
from pathlib import Path

# Add src to path so we can import gemini_ocr modules
import sys

sys.path.append(str(Path.cwd() / "src"))

from gemini_ocr import settings, document, docai_ocr, gemini_ocr, gemini

# For serializing BBox
from gemini_ocr.document import BoundingBox, BBox


async def capture():
    # Load .env
    dotenv.load_dotenv()

    # Map GOOGLE_OCR_ vars to GEMINI_OCR_ vars if needed
    mapping = {
        "GOOGLE_OCR_PROJECT": "GEMINI_OCR_PROJECT",
        "GOOGLE_OCR_LAYOUT_PARSER_PROCESSOR_ID": "GEMINI_OCR_LAYOUT_PROCESSOR_ID",
        "GOOGLE_OCR_OCR_PROCESSOR_ID": "GEMINI_OCR_OCR_PROCESSOR_ID",
        "GOOGLE_OCR_LOCATION": "GEMINI_OCR_LOCATION",
    }
    for src, dst in mapping.items():
        if os.getenv(src) and not os.getenv(dst):
            os.environ[dst] = os.getenv(src)

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

    # 2. Capture DocAI OCR Bounding Boxes (if not already cached/saved)
    # We can probably skipping re-capturing DocAI bboxes if the file hasn't changed,
    # but to be safe and complete, let's re-capture or verify.
    # The existing fixture is hubble_docai_bboxes.pkl.
    # The user asked only to improve Gemini prompt.
    # But for a consistent set, good to refresh. However, DocAI costs money/quota.
    # I'll check if the file exists and skip if so?
    # Actually, the user's request doesn't affect DocAI output.
    # But I deleted the original generation script.

    # Let's re-generate DocAI bboxes just in case, or load existing if I wanted to rely on them.
    # Since I claimed I am establishing a regression test, I should own the fixtures.
    # Re-running DocAI OCR (bboxes) is safer.

    docai_bboxes_list = []
    # Note: process_document uses batched gather.
    # We'll reproduce logic from extract_raw_data roughly but per chunk.

    print("Generating DocAI BBoxes...")
    # Parallelize?
    # docai_ocr.generate_bounding_boxes

    # docai_ocr.generate_bounding_boxes returns list[BoundingBox]

    # We'll just execute it.
    all_chunks_bboxes = []

    # To match the regression structure (list of lists of bboxes corresponding to chunks is NOT what the current regression test does!)
    # Wait, the regression test logic I wrote earlier:
    # "mock_ocr.side_effect = mock_ocr_side_effect"
    # "return docai_bboxes[idx]"
    # So docai_bboxes in pickle should be a LIST OF LISTS of BoundingBoxes, one per chunk.

    # Let's verify what I loaded in the test.
    # "docai_bboxes = pickle.load(f)"
    # "return docai_bboxes[idx]" -> So it is indexed by chunk index.

    # Okay, so I need to save a List[List[BoundingBox]].

    for i, chunk in enumerate(chunks):
        print(f"Generating DocAI bboxes for chunk {i}...")
        bboxes = await docai_ocr.generate_bounding_boxes(ocr_settings, chunk)
        all_chunks_bboxes.append(bboxes)

    with open("tests/fixtures/hubble_docai_bboxes.pkl", "wb") as f:
        pickle.dump(all_chunks_bboxes, f)

    print("Saved DocAI bboxes.")


if __name__ == "__main__":
    asyncio.run(capture())
