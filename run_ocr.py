import argparse
import asyncio
import logging
import os
import pathlib
import sys
import traceback

import anchorite
import dotenv
import google.auth
from google import genai

from gemini_ocr import DocAIAnchorProvider, DocAIMarkdownProvider, GeminiMarkdownProvider


def _list_models(project: str | None, location: str, quota_project: str | None) -> None:
    if not project:
        print("Error: --project or GOOGLE_CLOUD_PROJECT env var required.")
        sys.exit(1)

    credentials, _ = google.auth.default()
    if quota_project:
        credentials = credentials.with_quota_project(quota_project)
    elif project:
        credentials = credentials.with_quota_project(project)

    client = genai.Client(vertexai=True, project=project, location=location, credentials=credentials)
    print("Available Gemini Models:")
    for model in client.models.list():
        if model.name and "gemini" in model.name:
            print(f" - {model.name}")
    sys.exit(0)


async def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Run Gemini OCR on a PDF.")
    parser.add_argument(
        "input_pdf",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("main.pdf"),
        help="Input PDF file.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Vertex AI Project ID",
    )
    parser.add_argument(
        "--quota-project",
        default=os.environ.get("GEMINI_OCR_QUOTA_PROJECT_ID"),
        help="GCP Quota Project ID (for billing)",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="GCP Location",
    )
    parser.add_argument(
        "--processor-id",
        default=os.environ.get("DOCUMENTAI_LAYOUT_PARSER_PROCESSOR_ID"),
        help="Document AI Layout Parser Processor ID",
    )
    parser.add_argument(
        "--ocr-processor-id",
        default=os.environ.get("DOCUMENTAI_OCR_PROCESSOR_ID"),
        help="Document AI OCR Processor ID (for bounding box extraction)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_OCR_GEMINI_MODEL_NAME"),
        help="Gemini Model Name (e.g. gemini-2.0-flash)",
    )
    parser.add_argument(
        "--gemini-prompt",
        default=None,
        help="Additional instructions to append to the default Gemini prompt.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output.md"),
        help="Output markdown file",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        help="Directory to cache OCR results",
    )
    parser.add_argument(
        "--mode",
        choices=["gemini", "documentai"],
        default="gemini",
        help="OCR generation mode",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Gemini models and exit",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Disable bounding box output in markdown",
    )

    args = parser.parse_args()

    if args.list_models:
        _list_models(args.project, args.location, args.quota_project)

    if not args.input_pdf.exists():
        print(f"Error: Input file {args.input_pdf} not found.")
        sys.exit(1)

    if not args.project:
        print("Error: --project or GOOGLE_CLOUD_PROJECT env var required.")
        sys.exit(1)

    cache_dir = str(args.cache_dir) if args.cache_dir else None

    if args.mode == "gemini":
        if not args.model:
            print("Error: --model or GEMINI_OCR_GEMINI_MODEL_NAME required in gemini mode.")
            sys.exit(1)
        markdown_provider: anchorite.providers.MarkdownProvider = GeminiMarkdownProvider(
            project_id=args.project,
            location=args.location,
            model_name=args.model,
            quota_project_id=args.quota_project,
            prompt=args.gemini_prompt,
            cache_dir=cache_dir,
        )
    else:
        if not args.processor_id:
            print("Error: --processor-id required in documentai mode.")
            sys.exit(1)
        markdown_provider = DocAIMarkdownProvider(
            project_id=args.project,
            location=args.location,
            processor_id=args.processor_id,
            cache_dir=cache_dir,
        )

    anchor_provider: anchorite.providers.AnchorProvider | None = None
    if not args.no_bbox and args.ocr_processor_id:
        anchor_provider = DocAIAnchorProvider(
            project_id=args.project,
            location=args.location,
            processor_id=args.ocr_processor_id,
            cache_dir=cache_dir,
        )

    print(f"Processing {args.input_pdf}...")

    try:
        chunks = anchorite.document.chunks(args.input_pdf)
        result = await anchorite.process_document(chunks, markdown_provider, anchor_provider, renumber=True)

        output_content = result.annotate() if anchor_provider else result.markdown_content
        args.output.write_text(output_content)
        print(f"Done! Output saved to {args.output}")

    except Exception as e:  # noqa: BLE001
        print(f"Error processing document: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
