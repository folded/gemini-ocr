import asyncio
import hashlib
import logging
import pathlib
import typing

from google.api_core import client_options
from google.cloud import documentai

from gemini_ocr import document, settings


def _call_docai(
    ocr_settings: settings.Settings,
    process_options: documentai.ProcessOptions,
    processor_id: str,
    chunk: document.DocumentChunk,
) -> documentai.Document:
    location = ocr_settings.get_documentai_location()

    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options.ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com"),
    )

    name = client.processor_path(ocr_settings.project_id, location, processor_id)

    raw_document = documentai.RawDocument(content=chunk.data, mime_type=chunk.mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document, process_options=process_options)
    result = client.process_document(request=request)
    return result.document


def _generate_cache_path(
    ocr_settings: settings.Settings,
    process_options: documentai.ProcessOptions,
    processor_id: str,
    chunk: document.DocumentChunk,
) -> pathlib.Path | None:
    if not ocr_settings.cache_dir or not ocr_settings.cache_docai:
        return None
    hasher = hashlib.sha256()
    hasher.update(documentai.ProcessOptions.to_json(process_options, sort_keys=True).encode())
    hasher.update(processor_id.encode())
    hasher.update(chunk.document_sha256.encode())
    cache_key = f"{hasher.hexdigest()}_{chunk.start_page}_{chunk.end_page}"

    return pathlib.Path(ocr_settings.cache_dir) / "docai" / f"{cache_key}.json"


async def process(
    ocr_settings: settings.Settings,
    process_options: documentai.ProcessOptions,
    processor_id: str,
    chunk: document.DocumentChunk,
) -> documentai.Document:
    """Runs Document AI OCR."""

    cache_path = _generate_cache_path(ocr_settings, process_options, processor_id, chunk)

    if cache_path and cache_path.exists():
        logging.debug("Loaded from DocAI cache: %s", cache_path)
        return typing.cast("documentai.Document", documentai.Document.from_json(cache_path.read_text()))

    doc = await asyncio.to_thread(_call_docai, ocr_settings, process_options, processor_id, chunk)

    # Save to Cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(documentai.Document.to_json(doc))
        logging.debug("Saved to DocAI cache: %s", cache_path)

    return doc
