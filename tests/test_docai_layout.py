from unittest.mock import Mock

from gemini_ocr import docai_layout


def test_layout_processor_table_tags() -> None:
    # Text block mock
    text_block = Mock()
    text_block.text_block.type_ = "paragraph"
    text_block.text_block.text = "Header text"
    text_block.text_block.blocks = []
    # Ensure other types are False
    text_block.list_block = None
    text_block.table_block = None

    # Table block mock
    # 2x2 Table: 1 Header row, 1 Body row

    # Helper to make a cell mock
    def make_cell(text: str) -> Mock:
        cell = Mock()
        cell.row_span = 1
        cell.col_span = 1
        block = Mock()
        block.text_block.type_ = "paragraph"
        block.text_block.text = text
        block.text_block.blocks = []
        block.list_block = None
        block.table_block = None
        # The recursion check in _process_table_block iterates cell.blocks
        cell.blocks = [block]
        return cell

    cell_h1 = make_cell("H1")
    cell_h2 = make_cell("H2")
    cell_c1 = make_cell("C1")
    cell_c2 = make_cell("C2")

    row_1 = Mock()
    row_1.cells = [cell_h1, cell_h2]

    row_2 = Mock()
    row_2.cells = [cell_c1, cell_c2]

    table_block = Mock()
    table_block.text_block = None
    table_block.list_block = None
    table_block.table_block = Mock()
    table_block.table_block.header_rows = [row_1]
    table_block.table_block.body_rows = [row_2]

    processor = docai_layout.LayoutProcessor()
    # Processor expects a list of blocks
    result = "".join(processor.process([text_block, table_block]))

    print(f"Result:\n{result}")

    assert "# Header text" not in result  # Paragraphs don't get #
    assert "Header text" in result
    assert "<!--table-->" in result
    assert "| H1 | H2 |" in result
    assert "| C1 | C2 |" in result
    assert "<!--end-->" in result


def test_multiple_tables() -> None:
    table_block = Mock()
    table_block.text_block = None
    table_block.list_block = None
    table_block.table_block = Mock()
    table_block.table_block.header_rows = []
    table_block.table_block.body_rows = []

    processor = docai_layout.LayoutProcessor()
    result = "".join(processor.process([table_block, table_block]))

    assert result.count("<!--table-->") == 2
