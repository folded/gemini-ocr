from gemini_ocr import document, gemini_ocr


def test_nested_zero_length_at_start() -> None:
    content = "Hello"
    bbox_a = document.BoundingBox(text="Hello", page=1, rect=document.BBox(0, 0, 5, 1))
    bbox_b = document.BoundingBox(text="", page=1, rect=document.BBox(0, 0, 0, 0))

    result = gemini_ocr.OcrResult(content, {bbox_a: (0, 5), bbox_b: (0, 0)}, coverage_percent=0.0)
    annotated = result.annotate()

    tag_a = '<span class="ocr_bbox" data-bbox="0,0,5,1" data-page="1">'
    tag_b = '<span class="ocr_bbox" data-bbox="0,0,0,0" data-page="1">'

    assert f"{tag_a}{tag_b}</span>Hello</span>" == annotated


def test_nested_zero_length_at_end() -> None:
    content = "Hello"
    bbox_a = document.BoundingBox(text="Hello", page=1, rect=document.BBox(0, 0, 5, 1))
    bbox_c = document.BoundingBox(text="", page=1, rect=document.BBox(5, 5, 5, 5))

    result = gemini_ocr.OcrResult(content, {bbox_a: (0, 5), bbox_c: (5, 5)}, coverage_percent=0.0)
    annotated = result.annotate()

    tag_a = '<span class="ocr_bbox" data-bbox="0,0,5,1" data-page="1">'
    tag_c = '<span class="ocr_bbox" data-bbox="5,5,5,5" data-page="1">'

    assert f"{tag_a}Hello{tag_c}</span></span>" == annotated
