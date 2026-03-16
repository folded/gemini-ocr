import dataclasses
import logging

import anchorite
from anchorite.document import DocumentChunk
from anchorite.providers import AnchorProvider
from google.cloud import documentai

from gemini_ocr import docai

_BBOX_VERTEX_COUNT = 4


@dataclasses.dataclass
class DocAIAnchorProvider(AnchorProvider):
    """Anchor provider that generates bounding boxes using Document AI OCR."""

    project_id: str
    location: str
    processor_id: str
    documentai_location: str | None = None
    cache_dir: str | None = None
    cache: bool = True

    async def generate_anchors(self, chunk: DocumentChunk) -> list[anchorite.Anchor]:
        process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_native_pdf_parsing=True,
                premium_features=documentai.OcrConfig.PremiumFeatures(
                    compute_style_info=True,
                    enable_math_ocr=True,
                ),
            ),
        )
        doc = await docai.process(
            self.project_id,
            self.location,
            self.processor_id,
            process_options,
            chunk,
            documentai_location=self.documentai_location,
            cache_dir=self.cache_dir,
            cache=self.cache,
        )

        def _get_text(text_anchor: documentai.Document.TextAnchor) -> str:
            if not text_anchor.text_segments:
                return ""
            return "".join(doc.text[int(s.start_index) : int(s.end_index)] for s in text_anchor.text_segments)

        anchors = []
        for page_num, page in enumerate(doc.pages):
            for block in page.lines:
                text = _get_text(block.layout.text_anchor).strip()
                vertices = block.layout.bounding_poly.normalized_vertices
                if len(vertices) == _BBOX_VERTEX_COUNT:
                    anchors.append(
                        anchorite.Anchor(
                            text=text,
                            page=page_num + chunk.start_page,
                            boxes=(
                                anchorite.BBox(
                                    top=int(vertices[0].y * 1000),
                                    left=int(vertices[0].x * 1000),
                                    bottom=int(vertices[2].y * 1000),
                                    right=int(vertices[2].x * 1000),
                                ),
                            ),
                        ),
                    )

        logging.debug("Generated %d anchors", len(anchors))
        return anchors
