import logging

from google.cloud import documentai

from gemini_ocr import docai, document, settings


async def _run_document_ai(settings: settings.Settings, chunk: document.DocumentChunk) -> documentai.Document:
    """Runs Document AI OCR."""

    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=True,
            premium_features=documentai.OcrConfig.PremiumFeatures(
                compute_style_info=True,
                enable_math_ocr=True,
            ),
        ),
    )

    return await docai.process(settings, process_options, settings.ocr_processor_id, chunk)


async def generate_bounding_boxes(
    settings: settings.Settings,
    chunk: document.DocumentChunk,
) -> list[document.BoundingBox]:
    doc = await _run_document_ai(settings, chunk)

    def _get_text(text_anchor: documentai.Document.TextAnchor) -> str:
        if not text_anchor.text_segments:
            return ""
        return "".join(
            doc.text[int(segment.start_index) : int(segment.end_index)] for segment in text_anchor.text_segments
        )

    bboxes = []
    for page_num, page in enumerate(doc.pages):
        for block in page.lines:
            text = _get_text(block.layout.text_anchor).strip()
            vertices = block.layout.bounding_poly.normalized_vertices
            num_vertices = 4
            if len(vertices) == num_vertices:
                top = int(vertices[0].y * 1000)
                left = int(vertices[0].x * 1000)
                bottom = int(vertices[2].y * 1000)
                right = int(vertices[2].x * 1000)
                rect = document.BBox(top, left, bottom, right)
                bboxes.append(document.BoundingBox(page=page_num + chunk.start_page, rect=rect, text=text))

    logging.debug("Generated %d bounding boxes", len(bboxes))
    return bboxes
