import asyncio
import dataclasses
import hashlib
import logging
import pathlib
from typing import Final

import google.auth
from anchorite.document import DocumentChunk
from anchorite.providers import MarkdownProvider
from google import genai

GEMINI_PROMPT: Final[str] = """
Carefully transcribe the text for this pdf into a text file with
markdown annotations.

**The final output must be formatted as text that visually
mimics in markdown the layout and hierarchy of the original PDF
when rendered.**

* Do not include headers or footers that are repeated on each page.
* Do not include page numbers.
* Preserve the reading order of the text as it appears in the PDF.
* Remove hyphens that break words at the end of lines.
  * e.g. "uti- lized" -> "utilized"
* Use Markdown headings (`#`, `##`, `###`) to reflect the size and
  hierarchy of titles and subtitles in the PDF.
* Ensure that there are blank lines before and after headings, lists,
  tables, and images.
* End each paragraph with a blank line.
* Do not break lines within paragraphs or headings.
* Render bullet points and numbered lettered lists as markdown lists.
  * It is ok to remove brackets and other consistent punctuation around
    list identifiers
    * e.g. "a)" -> "a."
* Use blockquotes for any sidebars or highlighted text.
* Bold all words and phrases that appear bolded in the original
  source material. Similarly, italicise all text in italics.
* Render tables as markdown, paying particular attention to copying
  identifiers exactly.
* Break text into paragraphs and lists exactly as they appear in
  the PDF.
* Replace any images with a text description of their content.
  * Convert bar charts into markdown tables.
* Convert tables contained in images into markdown.
* Render all mathematical equations and symbols using LaTeX formatting.
  * e.g. use `\alpha` instead of `α`, `\\cos` instead of `cos`.
  * Enclose equations in `$` or `$$`.
  * Pay close attention to distinguishing Latin and Greek characters, e.g. 'a' vs '\alpha'.
* Insert markers at the start of each page of the form `<!--page-->`
* Surround tables and figure descriptions with markers:
  * `<!--table-->` ... `<!--end-->`
  * `<!--figure-->` ... `<!--end-->`
"""  # noqa: RUF001


@dataclasses.dataclass
class GeminiMarkdownProvider(MarkdownProvider):
    """Markdown provider that generates markdown using the Gemini API."""

    project_id: str
    location: str
    model_name: str
    quota_project_id: str | None = None
    prompt: str | None = None
    cache_dir: str | None = None
    cache: bool = True

    def _cache_path(self, chunk: DocumentChunk) -> pathlib.Path | None:
        if not self.cache_dir or not self.cache:
            return None
        hasher = hashlib.sha256()
        hasher.update(GEMINI_PROMPT.encode())
        if self.prompt:
            hasher.update(self.prompt.encode())
        hasher.update(chunk.document_sha256.encode())
        hasher.update((self.model_name or "").encode())
        cache_key = f"{hasher.hexdigest()}_{chunk.start_page}_{chunk.end_page}"
        return pathlib.Path(self.cache_dir) / "gemini" / f"{cache_key}.txt"

    def _call(self, chunk: DocumentChunk) -> genai.types.GenerateContentResponse:
        credentials, _ = google.auth.default()
        if self.quota_project_id:
            credentials = credentials.with_quota_project(self.quota_project_id)
        elif self.project_id:
            credentials = credentials.with_quota_project(self.project_id)

        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            credentials=credentials,
        )

        contents: list[genai.types.Part | str] = [
            genai.types.Part(inline_data=genai.types.Blob(data=chunk.data, mime_type=chunk.mime_type)),
            GEMINI_PROMPT,
        ]
        if self.prompt:
            contents.append(self.prompt)

        return client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=genai.types.GenerateContentConfig(response_mime_type="text/plain"),
        )

    async def generate_markdown(self, chunk: DocumentChunk) -> str:
        cache_path = self._cache_path(chunk)

        if cache_path and cache_path.exists():
            logging.debug("Loaded from Gemini cache: %s", cache_path)
            return cache_path.read_text()

        response = await asyncio.to_thread(self._call, chunk)
        text = response.text or ""

        if cache_path and text:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text)
            logging.debug("Saved to Gemini cache: %s", cache_path)

        return text
