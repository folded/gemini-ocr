import argparse
import asyncio
import logging
import os
import pathlib
import sys
import traceback

import dotenv

from gemini_ocr import gemini_ocr, settings


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
        "--gcp-project-id",
        default=os.environ.get("GCP_PROJECT_ID"),
        help="GCP Project ID (if different from project)",
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
        "--no-bbox",
        action="store_true",
        help="Disable bounding box output in markdown",
    )

    parser.add_argument(
        "--enable-chunking",
        action="store_true",
        help="Enable V1.5 chunking",
    )

    args = parser.parse_args()

    if not args.input_pdf.exists():
        print(f"Error: Input file {args.input_pdf} not found.")
        sys.exit(1)

    if not args.project:
        print("Error: --project or GOOGLE_CLOUD_PROJECT env var required.")
        sys.exit(1)

    if not args.processor_id:
        print("Error: --processor-id or DOCUMENTAI_LAYOUT_PARSER_PROCESSOR_ID env var required.")
        sys.exit(1)

    gcp_project = args.gcp_project_id or args.project

    ocr_settings = settings.Settings(
        project=args.project,
        location=args.location,
        layout_processor_id=args.processor_id,
        ocr_processor_id=args.ocr_processor_id,
        gcp_project_id=gcp_project,
        mode=args.mode,
        include_bboxes=not args.no_bbox,
        cache_dir=str(args.cache_dir) if args.cache_dir else ".docai_cache",
    )

    print(f"Processing {args.input_pdf}...")
    print(f"Settings: {ocr_settings}")

    try:
        result = await gemini_ocr.process_document(ocr_settings, args.input_pdf)

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
