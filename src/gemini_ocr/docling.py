import dataclasses

import anchorite.document
import anchorite.providers


@dataclasses.dataclass
class DoclingMarkdownProvider(anchorite.providers.MarkdownProvider):
    """Markdown provider that generates markdown using Docling."""

    async def generate_markdown(self, chunk: anchorite.document.DocumentChunk) -> str:
        raise NotImplementedError
