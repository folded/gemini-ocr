import re
from unittest.mock import patch

import pytest

from gemini_ocr import gemini_ocr, settings


@pytest.mark.asyncio
async def test_multi_chunk_renumbering() -> None:
    # multiple chunks with overlapping table/figure numbers
    # Mock _generate_markdown_for_chunk to return specific text with generic markers
    with (
        patch("gemini_ocr.gemini_ocr._generate_markdown_for_chunk") as mock_gen_md,
        patch("gemini_ocr.document.chunks", return_value=["c1", "c2"]),
        patch("gemini_ocr.docai_ocr.generate_bounding_boxes", return_value=[]),
    ):
        mock_gen_md.side_effect = [
            "Part 1: <!--table--> Content <!--table--> <!--figure-->",
            "Part 2: <!--table--> Content <!--figure--> <!--figure-->",
        ]
        res = await gemini_ocr.process_document(
            "dummy.pdf",
            settings=settings.Settings(
                project_id="test",
                location="us",
                ocr_processor_id="id",
                layout_processor_id="id",
            ),
        )

    content = res.markdown_content

    # Check tables are 1, 2, 3
    tables = re.findall(r"<!--table: \d+-->", content)
    assert tables == ["<!--table: 1-->", "<!--table: 2-->", "<!--table: 3-->"]

    # Check figures are 1, 2, 3
    figures = re.findall(r"<!--figure: \d+-->", content)
    assert figures == ["<!--figure: 1-->", "<!--figure: 2-->", "<!--figure: 3-->"]
