import argparse
import asyncio
import logging
import os
import pathlib
import sys
import traceback

import dotenv
import google.auth
from google import genai

from gemini_ocr import gemini_ocr, settings


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
        help="Document AI OCR Processor ID (for secondary bbox pass)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_OCR_GEMINI_MODEL_NAME"),
        help="Gemini Model Name (e.g. gemini-2.0-flash-exp)",
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

    if not args.processor_id:
        print("Error: --processor-id or DOCUMENTAI_LAYOUT_PARSER_PROCESSOR_ID env var required.")
        sys.exit(1)

    ocr_settings = settings.Settings(
        project_id=args.project,
        location=args.location,
        quota_project_id=args.quota_project,
        layout_processor_id=args.processor_id,
        ocr_processor_id=args.ocr_processor_id,
        gemini_model_name=args.model,
        mode=args.mode,
        include_bboxes=not args.no_bbox,
        cache_dir=args.cache_dir,
    )

    print(f"Processing {args.input_pdf}...")
    print(f"Settings: {ocr_settings}")

    try:
        result = await gemini_ocr.process_document(args.input_pdf, settings=ocr_settings)

        output_content = result.annotate() if ocr_settings.include_bboxes else result.markdown_content

        output_path = args.output
        output_path.write_text(output_content)

        print(f"Done! Output saved to {output_path}")

    except Exception as e:  # noqa: BLE001
        print(f"Error processing document: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
