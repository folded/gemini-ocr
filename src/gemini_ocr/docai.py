import asyncio
import hashlib
import logging
import pathlib

from google.api_core import client_options
from google.cloud import documentai

from gemini_ocr import document, settings


async def process(
    settings: settings.Settings,
    process_options: documentai.ProcessOptions,
    processor_id: str,
    chunk: document.DocumentChunk,
) -> documentai.Document:
    """Runs Document AI OCR."""

    hasher = hashlib.sha256()
    hasher.update(documentai.ProcessOptions.to_json(process_options, sort_keys=True).encode())
    hasher.update(processor_id.encode())
    hasher.update(chunk.document_sha256.encode())
    cache_key = f"{hasher.hexdigest()}_{chunk.start_page}_{chunk.end_page}"

    # --- Caching Logic ---
    cache_path = None
    if settings.cache_dir:
        cache_path = pathlib.Path(settings.cache_dir) / f"{cache_key}.json"

        if cache_path.exists():
            logging.debug("Loaded from DocAI cache: %s", cache_path)
            return documentai.Document.from_json(cache_path.read_text())

    def _call_docai() -> documentai.Document:
        location = settings.location
        if location == "us-central1":
            location = "us"

        client = documentai.DocumentProcessorServiceClient(
            client_options=client_options.ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com"),
        )

        name = client.processor_path(settings.gcp_project_id, location, processor_id)

        raw_document = documentai.RawDocument(content=chunk.data, mime_type=chunk.mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document, process_options=process_options)
        result = client.process_document(request=request)
        return result.document

    doc = await asyncio.to_thread(_call_docai)

    # Save to Cache
    if settings.cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(documentai.Document.to_json(doc))
        logging.debug("Saved to DocAI cache: %s", cache_path)

    return doc
