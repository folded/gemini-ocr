import asyncio
from typing import Final

from google import genai

from gemini_ocr import document, settings

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
* Insert markers at the start of each page of the form `<!--page: {num}-->` starting at page 0
* Surround tables and figure descriptions with markers, where {num} is
  determined by the document:
  * `<!--table: {num}-->` ... `<!--end-->`
  * `<!--figure: {num}-->` ... `<!--end-->`
"""


async def generate_markdown(
    settings: settings.Settings,
    chunk: document.DocumentChunk,
) -> str | None:
    """Generates markdown for a chunk using the Gemini API."""

    def _call_gemini() -> genai.types.GenerateContentResponse:
        client = genai.Client(vertexai=True, project=settings.project, location=settings.location)

        contents: list[genai.types.Part | str] = []
        contents.append(genai.types.Part(inline_data=genai.types.Blob(data=chunk.data, mime_type=chunk.mime_type)))
        contents.append(GEMINI_PROMPT)

        return client.models.generate_content(
            model=settings.gemini_model_name,
            contents=contents,
            config=genai.types.GenerateContentConfig(response_mime_type="text/plain"),
        )

    return (await asyncio.to_thread(_call_gemini)).text
