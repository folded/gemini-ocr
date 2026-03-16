import dataclasses

import anchorite


@dataclasses.dataclass
class DoclingMarkdownProvider(anchorite.providers.MarkdownProvider):
    """Markdown provider that generates markdown using Docling."""

    async def generate_markdown(self, chunk: anchorite.document.DocumentChunk) -> str:
        raise NotImplementedError
