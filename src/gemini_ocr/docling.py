import dataclasses

from anchorite.document import DocumentChunk
from anchorite.providers import MarkdownProvider


@dataclasses.dataclass
class DoclingMarkdownProvider(MarkdownProvider):
    """Markdown provider that generates markdown using Docling."""

    async def generate_markdown(self, chunk: DocumentChunk) -> str:
        raise NotImplementedError
