from gemini_ocr import bbox_alignment, document, gemini_ocr


def test_missed_match_histone_modifier() -> None:
    markdown_content = """
| Cell cycle | 3 | 34 |
| DNA repair | 4 | 52 |
| Histone modifier | 5 | 29 |
| OTHER GROWTH/PROLIFERATION SIGNALING | | |
"""

    bbox_text = "Histone modifier"
    bbox = document.BoundingBox(text=bbox_text, page=1, rect=document.BBox(0, 0, 0, 0))
    bboxes = [bbox]

    alignments = bbox_alignment.create_annotated_markdown(markdown_content, bboxes)

    assert bbox in alignments

    result = gemini_ocr.OcrResult(markdown_content, alignments, coverage_percent=0.0)
    annotated = result.annotate()

    assert '<span class="ocr_bbox"' in annotated
    assert ">Histone modifier</span>" in annotated


def test_missed_match_mek12() -> None:
    markdown_content = """
| Trametinib | MEK1/2 | BRAF | SKCM |
| Vemurafenib | BRAF | BRAF | SKCM |
"""
    bbox_text = "MEK1/2"
    bbox = document.BoundingBox(text=bbox_text, page=1, rect=document.BBox(0, 0, 0, 0))
    bboxes = [bbox]

    alignments = bbox_alignment.create_annotated_markdown(markdown_content, bboxes)

    assert bbox in alignments

    result = gemini_ocr.OcrResult(markdown_content, alignments, coverage_percent=0.0)
    annotated = result.annotate()
    assert ">MEK1/2</span>" in annotated


def test_missed_match_duplicate_handling() -> None:
    markdown_content = """
Here is duplicate.
Here is duplicate.
"""
    bbox_text = "duplicate"
    bbox = document.BoundingBox(text=bbox_text, page=1, rect=document.BBox(0, 0, 0, 0))
    bboxes = [bbox]

    alignments = bbox_alignment.create_annotated_markdown(markdown_content, bboxes)

    assert bbox in alignments

    result = gemini_ocr.OcrResult(markdown_content, alignments, coverage_percent=0.0)
    annotated = result.annotate()

    assert annotated.count('<span class="ocr_bbox"') == 1
    assert ">duplicate</span>" in annotated
