from unittest.mock import MagicMock

import pytest

from gemini_ocr import bbox_alignment, document, gemini_ocr


@pytest.mark.asyncio
async def test_coverage_calculation() -> None:
    # Setup
    markdown = "Hello World"  # length 11
    # Spans: "Hello" (0-5), "World" (6-11). Space (5-6) is missing.
    # Total covered: 5 + 5 = 10. Coverage: 10/11 ~ 0.909

    bbox1 = document.BoundingBox(text="Hello", page=1, rect=document.BBox(0, 0, 0, 0))
    bbox2 = document.BoundingBox(text="World", page=1, rect=document.BBox(0, 0, 0, 0))

    annotated = {bbox1: (0, 5), bbox2: (6, 11)}

    raw_data = gemini_ocr.RawOcrData(markdown, [bbox1, bbox2])

    # We need to mock extract_raw_data or just test the logic directly if possible.
    # gemini_ocr.process_document calls extract_raw_data then bbox_alignment.create_annotated_markdown
    # (which is slow/complex). Since the logic is inside process_document, we should mock the deps.

    # Easier: Mock extract_raw_data and bbox_alignment.create_annotated_markdown

    with pytest.MonkeyPatch.context() as m:

        async def mock_extract(*_args: object, **_kwargs: object) -> gemini_ocr.RawOcrData:
            return raw_data

        m.setattr(gemini_ocr, "extract_raw_data", mock_extract)
        m.setattr(bbox_alignment, "create_annotated_markdown", lambda *_, **__: annotated)

        settings = MagicMock()

        result = await gemini_ocr.process_document("dummy_path", settings=settings, markdown_content=markdown)  # type: ignore[arg-type]

        expected_coverage = 10.0 / 11.0
        assert result.coverage_percent == pytest.approx(expected_coverage)


@pytest.mark.asyncio
async def test_coverage_overlap() -> None:
    markdown = "Hello"  # 5
    # Span 1: 0-3 "Hel"
    # Span 2: 2-5 "llo"
    # Union: 0-5. Covered: 5/5 = 1.0

    bbox1 = document.BoundingBox(text="Hel", page=1, rect=document.BBox(0, 0, 0, 0))
    bbox2 = document.BoundingBox(text="llo", page=1, rect=document.BBox(0, 0, 0, 0))

    annotated = {bbox1: (0, 3), bbox2: (2, 5)}

    raw_data = gemini_ocr.RawOcrData(markdown, [bbox1, bbox2])

    with pytest.MonkeyPatch.context() as m:

        async def mock_extract(*_args: object, **_kwargs: object) -> gemini_ocr.RawOcrData:
            return raw_data

        m.setattr(gemini_ocr, "extract_raw_data", mock_extract)
        m.setattr(bbox_alignment, "create_annotated_markdown", lambda *_, **__: annotated)

        settings = MagicMock()
        result = await gemini_ocr.process_document("dummy_path", settings=settings, markdown_content=markdown)  # type: ignore[arg-type]

        assert result.coverage_percent == 1.0
