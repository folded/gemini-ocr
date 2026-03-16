import pickle
import json
import os
import sys
from pathlib import Path

# Mock gemini_ocr.document so pickle.load can find the classes
import dataclasses
from typing import NamedTuple, Any


class BBox(NamedTuple):
    top: int
    left: int
    bottom: int
    right: int


@dataclasses.dataclass(frozen=True)
class BoundingBox:
    text: str
    page: int
    rect: BBox


@dataclasses.dataclass
class DocumentChunk:
    document_sha256: str
    start_page: int
    end_page: int
    data: bytes
    mime_type: str


# Inject into sys.modules BEFORE any other imports
import types

doc_mod = types.ModuleType("gemini_ocr.document")
doc_mod.BBox = BBox
doc_mod.BoundingBox = BoundingBox
doc_mod.DocumentChunk = DocumentChunk
doc_mod.DocumentInput = Any  # TypeAlias
sys.modules["gemini_ocr.document"] = doc_mod


def convert():
    pkl_path = Path("tests/fixtures/hubble_docai_bboxes.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # data is list[list[BoundingBox]]
    json_data = []
    for chunk in data:
        json_chunk = []
        for b in chunk:
            json_chunk.append(
                {
                    "text": b.text,
                    "page": b.page,
                    "box": {"top": b.rect.top, "left": b.rect.left, "bottom": b.rect.bottom, "right": b.rect.right},
                }
            )
        json_data.append(json_chunk)

    dest_dir = Path("../anchorite/tests/fixtures")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / "hubble_docai_bboxes.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Converted {pkl_path} to json in anchorite.")


if __name__ == "__main__":
    convert()
