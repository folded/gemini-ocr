import asyncio
import collections
import dataclasses
import itertools
import re
import typing

from gemini_ocr import bbox_alignment, docai_layout, docai_ocr, docling, document, gemini
from gemini_ocr import settings as settings_module

T = typing.TypeVar("T")


@dataclasses.dataclass
class RawOcrData:
    """Intermediate data structure holding raw OCR/Markdown output."""

    markdown_content: str
    """The generated markdown string."""
    bounding_boxes: list[document.BoundingBox]
    """List of all bounding boxes extracted from the document."""


@dataclasses.dataclass
class OcrResult:
    """The final result of the OCR and annotation process."""

    markdown_content: str
    """The generated markdown content."""
    bounding_boxes: dict[document.BoundingBox, tuple[int, int]]
    """Mapping of bounding boxes to their span ranges in the markdown."""
    coverage_percent: float
    """Percentage of markdown content covered by aligned bounding boxes."""

    def annotate(self) -> str:
        """Annotates the markdown content with bounding box spans."""

        insertions = []
        for bbox, (start, end) in self.bounding_boxes.items():
            length = end - start
            bbox_str = f"{bbox.rect.top},{bbox.rect.left},{bbox.rect.bottom},{bbox.rect.right}"
            start_tag = f'<span class="ocr_bbox" data-bbox="{bbox_str}" data-page="{bbox.page}">'
            end_tag = "</span>"

            insertions.append((start, False, length, start_tag))
            insertions.append((end, True, length, end_tag))

        # Sort:
        # 1. Index Descending.
        # 2. is_end Descending (True/End processed before False/Start).
        # 3. Length Ascending (Short processed before Long).

        insertions.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)

        chars = list(self.markdown_content)
        for index, _, _, text in insertions:
            chars.insert(index, text)

        return "".join(chars)


async def _generate_markdown_for_chunk(
    ocr_settings: settings_module.Settings,
    chunk: document.DocumentChunk,
) -> str:
    """Generates markdown for a chunk using the Gemini API."""

    match ocr_settings.mode:
        case settings_module.OcrMode.GEMINI:
            text = await gemini.generate_markdown(ocr_settings, chunk)
        case settings_module.OcrMode.DOCUMENTAI:
            text = await docai_layout.generate_markdown(ocr_settings, chunk)
        case settings_module.OcrMode.DOCLING:
            text = await docling.generate_markdown(ocr_settings, chunk)
        case _:
            text = None

    if text is None:
        return ""

    def _adjust_page_num(match: re.Match) -> str:
        relative_page_num = int(match.group(1))
        actual_page_num = chunk.start_page + relative_page_num
        return f"<!--page: {actual_page_num}-->"

    return re.sub(r"<!-{2,3}\s*page:\s*(\d+)\s*-->", _adjust_page_num, text)


# --- Merging and Annotation ---


async def _batched_gather(tasks: collections.abc.Sequence[collections.abc.Awaitable[T]], batch_size: int) -> list[T]:
    """Runs awaitables in batches."""
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        results.extend(await asyncio.gather(*batch))
    return results


async def extract_raw_data(
    document_input: document.DocumentInput,
    settings: settings_module.Settings | None = None,
    markdown_content: str | None = None,
) -> RawOcrData:
    """
    Extracts raw OCR data (markdown and bounding boxes) from a file.

    Args:
        document_input: The document to process (Path, str, bytes, or stream).
        settings: Configuration settings.
        markdown_content: Optional existing markdown content.

    Returns:
        RawOcrData containing markdown and bounding boxes.
    """
    if settings is None:
        settings = settings_module.Settings.from_env()

    chunks = list(document.chunks(document_input, page_count=settings.markdown_page_batch_size))
    if not markdown_content:
        markdown_work = [_generate_markdown_for_chunk(settings, chunk) for chunk in chunks]
        markdown_chunks = await _batched_gather(markdown_work, settings.num_jobs)
        markdown_content = "\n".join(markdown_chunks)

    bounding_box_work = [docai_ocr.generate_bounding_boxes(settings, chunk) for chunk in chunks]
    bboxes = list(itertools.chain.from_iterable(await _batched_gather(bounding_box_work, settings.num_jobs)))

    return RawOcrData(
        markdown_content=markdown_content,
        bounding_boxes=bboxes,
    )


async def process_document(
    document_input: document.DocumentInput,
    settings: settings_module.Settings | None = None,
    markdown_content: str | None = None,
) -> OcrResult:
    """
    Processes a document to generate annotated markdown with OCR bounding boxes.

    Args:
        document_input: The document to process (Path, str, bytes, or stream).
        settings: Configuration settings.
        markdown_content: Optional existing markdown content.

    Returns:
        OcrResult containing annotated markdown and stats.
    """
    if settings is None:
        settings = settings_module.Settings.from_env()

    raw_data = await extract_raw_data(document_input, settings, markdown_content)
    annotated_markdown = bbox_alignment.create_annotated_markdown(
        raw_data.markdown_content,
        raw_data.bounding_boxes,
        uniqueness_threshold=settings.alignment_uniqueness_threshold,
        min_overlap=settings.alignment_min_overlap,
    )

    # Calculate coverage
    if not raw_data.markdown_content:
        coverage_percent = 0.0
    else:
        spans = list(annotated_markdown.values())
        spans.sort()
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        covered_len = sum(end - start for start, end in merged)
        coverage_percent = covered_len / len(raw_data.markdown_content)

    return OcrResult(
        markdown_content=raw_data.markdown_content,
        bounding_boxes=annotated_markdown,
        coverage_percent=coverage_percent,
    )
