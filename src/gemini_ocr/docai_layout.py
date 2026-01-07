import dataclasses
import re
import textwrap
from collections.abc import Generator, Sequence

from google.cloud import documentai

from gemini_ocr import docai, document, settings


@dataclasses.dataclass
class TableCell:
    """Represents a single cell within a table structure."""

    content: str
    """The text content of the cell."""
    row_pos: int
    """The file-global row index."""
    col_pos: int
    """The file-global column index."""
    row_span: int
    """Number of rows this cell spans (>= 1)."""
    col_span: int
    """Number of columns this cell spans (>= 1)."""
    is_header: bool
    """True if this cell falls within a header row."""


class LayoutProcessor:
    def process(
        self,
        blocks: Sequence[documentai.Document.DocumentLayout.DocumentLayoutBlock],
    ) -> Generator[str, None, None]:
        yield from self._process_layout_blocks(blocks)

    def _process_text_block(
        self,
        block: documentai.Document.DocumentLayout.DocumentLayoutBlock,
    ) -> Generator[str, None, None]:
        text_type = block.text_block.type_
        text = block.text_block.text
        # Fix math formatting: DocAI uses \(...\) for inline math, convert to $...$
        # We also trim whitespace inside the delimiters to ensure correct rendering (e.g. $ = 32$ instead of $ = 32 $)
        text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text)
        text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", text)

        if text_type.startswith("heading-"):
            level = int(text_type.split("-")[1])
            yield f"{'#' * level} {text}\n\n"
        elif text_type == "title":
            yield f"# {text}\n\n"
        elif text_type == "subtitle":
            yield f"## {text}\n\n"
        elif text_type == "paragraph":
            for line in textwrap.wrap(text):
                yield line + "\n"
            yield "\n"

        yield from self._process_layout_blocks(block.text_block.blocks)

    def _process_list_block(
        self,
        block: documentai.Document.DocumentLayout.DocumentLayoutBlock,
    ) -> Generator[str, None, None]:
        for entry in block.list_block.list_entries:
            entry_text = "".join(self._process_layout_blocks(entry.blocks)).rstrip("\n")
            entry_text = textwrap.indent(entry_text, "  ")

            if entry_text:
                yield f"- {entry_text[2:]}\n"
        yield "\n"

    def _process_table_block(
        self,
        block: documentai.Document.DocumentLayout.DocumentLayoutBlock,
    ) -> Generator[str, None, None]:
        yield "<!--table-->\n"

        table_block = block.table_block
        grid, num_rows, num_cols = self._build_table_grid(table_block)

        if num_rows == 0 or num_cols == 0:
            yield "<!--end-->\n"
            return

        yield from self._render_table(
            grid,
            num_rows,
            num_cols,
            len(table_block.header_rows) > 0,
            len(table_block.header_rows),
        )
        yield "<!--end-->\n"

    def _build_table_grid(
        self,
        table_block: documentai.Document.DocumentLayout.DocumentLayoutBlock.LayoutTableBlock,
    ) -> tuple[dict[tuple[int, int], TableCell], int, int]:
        all_rows = [(r, True) for r in table_block.header_rows] + [(r, False) for r in table_block.body_rows]

        occupied = set()
        grid = {}
        max_col = 0
        current_row_idx = 0

        for row_obj, is_header in all_rows:
            current_col_idx = 0
            for cell in row_obj.cells:
                # Advance column pointer if current position is occupied
                while (current_row_idx, current_col_idx) in occupied:
                    current_col_idx += 1

                cell_text = "".join(self._process_layout_blocks(cell.blocks)).rstrip("\n")
                cell_text = cell_text.replace("|", "\\|").replace("\n", "<br>")

                row_span = max(1, cell.row_span)
                col_span = max(1, cell.col_span)

                grid[(current_row_idx, current_col_idx)] = TableCell(
                    content=cell_text,
                    row_pos=current_row_idx,
                    col_pos=current_col_idx,
                    row_span=row_span,
                    col_span=col_span,
                    is_header=is_header,
                )

                for r in range(row_span):
                    for c in range(col_span):
                        occupied.add((current_row_idx + r, current_col_idx + c))

                current_col_idx += col_span
                max_col = max(max_col, current_col_idx)

            current_row_idx += 1

        return grid, current_row_idx, max_col

    def _render_table(
        self,
        grid: dict[tuple[int, int], TableCell],
        num_rows: int,
        num_cols: int,
        has_header: bool,
        header_row_count: int,
    ) -> Generator[str, None, None]:
        table_matrix = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for (r, c), cell in grid.items():
            table_matrix[r][c] = cell.content

        def row_to_md(values: list[str]) -> str:
            return "| " + " | ".join(values) + " |"

        if not has_header:
            yield row_to_md(["" for _ in range(num_cols)]) + "\n"
            yield row_to_md(["---" for _ in range(num_cols)]) + "\n"

        for r in range(num_rows):
            yield row_to_md(table_matrix[r]) + "\n"
            if has_header and r == header_row_count - 1:
                yield row_to_md(["---" for _ in range(num_cols)]) + "\n"

        yield "\n"

    def _process_layout_blocks(
        self,
        blocks: Sequence[documentai.Document.DocumentLayout.DocumentLayoutBlock],
    ) -> Generator[str, None, None]:
        for block in blocks:
            if block.text_block:
                yield from self._process_text_block(block)
            elif block.list_block:
                yield from self._process_list_block(block)
            elif block.table_block:
                yield from self._process_table_block(block)
            else:
                raise ValueError(f"Unknown block type: {block}")


async def _run_document_ai(settings: settings.Settings, chunk: document.DocumentChunk) -> documentai.Document:
    process_options = documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            return_bounding_boxes=True,
        ),
    )

    if settings.layout_processor_id is None:
        raise ValueError("Layout processor ID is not set")

    return await docai.process(settings, process_options, settings.layout_processor_id, chunk)


async def generate_markdown(
    settings: settings.Settings,
    chunk: document.DocumentChunk,
) -> str:
    """Generates Markdown from a Document AI chunk.

    Args:
        settings: OCR settings.
        chunk: The document chunk (usually a single page or small range).

    Returns:
        The generated Markdown string.
    """
    doc_result = await _run_document_ai(settings, chunk)
    processor = LayoutProcessor()

    return "".join(processor.process(doc_result.document_layout.blocks))
