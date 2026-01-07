import asyncio
from typing import Final

import google.auth
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
* Render all mathematical equations and symbols using LaTeX formatting.
  * e.g. use `\alpha` instead of `α`, `\\cos` instead of `cos`.
  * Enclose equations in `$` or `$$`.
  * Pay close attention to distinguishing Latin and Greek characters, e.g. 'a' vs '\alpha'.
* Insert markers at the start of each page of the form `<!--page-->`
* Surround tables and figure descriptions with markers:
  * `<!--table-->` ... `<!--end-->`
  * `<!--figure-->` ... `<!--end-->`
"""  # noqa: RUF001


async def generate_markdown(
    settings: settings.Settings,
    chunk: document.DocumentChunk,
) -> str | None:
    """Generates markdown for a chunk using the Gemini API."""

    def _call_gemini() -> genai.types.GenerateContentResponse:
        # TODO: consider reusing client
        credentials, _ = google.auth.default()
        if settings.quota_project_id:
            credentials = credentials.with_quota_project(settings.quota_project_id)
        elif settings.project_id:
            # Fallback to project if quota_project_id is not set
            credentials = credentials.with_quota_project(settings.project_id)

        client = genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.location,
            credentials=credentials,
        )

        model_name = settings.gemini_model_name
        if model_name is None:
            raise ValueError("gemini_model_name is required for Gemini mode.")

        contents: list[genai.types.Part | str] = []
        contents.append(genai.types.Part(inline_data=genai.types.Blob(data=chunk.data, mime_type=chunk.mime_type)))
        contents.append(GEMINI_PROMPT)

        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=genai.types.GenerateContentConfig(response_mime_type="text/plain"),
        )

    return (await asyncio.to_thread(_call_gemini)).text
