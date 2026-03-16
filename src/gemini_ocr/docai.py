import asyncio
import hashlib
import logging
import pathlib
import typing

from anchorite.document import DocumentChunk
from google.api_core import client_options
from google.cloud import documentai


def _resolve_documentai_location(location: str, documentai_location: str | None) -> str:
    if documentai_location is not None:
        return documentai_location
    return "eu" if location.startswith("eu") else "us"


def _call_docai(  # noqa: PLR0913
    project_id: str,
    location: str,
    documentai_location: str | None,
    processor_id: str,
    process_options: documentai.ProcessOptions,
    chunk: DocumentChunk,
) -> documentai.Document:
    resolved_location = _resolve_documentai_location(location, documentai_location)
    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options.ClientOptions(api_endpoint=f"{resolved_location}-documentai.googleapis.com"),
    )
    name = client.processor_path(project_id, resolved_location, processor_id)
    raw_document = documentai.RawDocument(content=chunk.data, mime_type=chunk.mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document, process_options=process_options)
    return client.process_document(request=request).document


def _generate_cache_path(
    cache_dir: str | None,
    cache: bool,
    processor_id: str,
    process_options: documentai.ProcessOptions,
    chunk: DocumentChunk,
) -> pathlib.Path | None:
    if not cache_dir or not cache:
        return None
    hasher = hashlib.sha256()
    hasher.update(documentai.ProcessOptions.to_json(process_options, sort_keys=True).encode())
    hasher.update(processor_id.encode())
    hasher.update(chunk.document_sha256.encode())
    cache_key = f"{hasher.hexdigest()}_{chunk.start_page}_{chunk.end_page}"
    return pathlib.Path(cache_dir) / "docai" / f"{cache_key}.json"


async def process(  # noqa: PLR0913
    project_id: str,
    location: str,
    processor_id: str,
    process_options: documentai.ProcessOptions,
    chunk: DocumentChunk,
    *,
    documentai_location: str | None = None,
    cache_dir: str | None = None,
    cache: bool = True,
) -> documentai.Document:
    cache_path = _generate_cache_path(cache_dir, cache, processor_id, process_options, chunk)

    if cache_path and cache_path.exists():
        logging.debug("Loaded from DocAI cache: %s", cache_path)
        return typing.cast("documentai.Document", documentai.Document.from_json(cache_path.read_text()))

    doc = await asyncio.to_thread(
        _call_docai,
        project_id,
        location,
        documentai_location,
        processor_id,
        process_options,
        chunk,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(documentai.Document.to_json(doc))
        logging.debug("Saved to DocAI cache: %s", cache_path)

    return doc
