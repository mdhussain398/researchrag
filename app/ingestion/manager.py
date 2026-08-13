"""
Ingestion Manager coordinating parsing, cleaning, chunking, and disk-based caching.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.models.schemas import DocumentMetadata, DocumentChunk
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import AcademicChunker
from app.utils.config import UPLOADS_DIR, CACHE_DIR, logger


class IngestionManager:
    """Coordinates parsing, chunking, and persistent caching of research documents."""

    def __init__(
        self,
        target_chunk_size: int = 750,
        chunk_overlap: int = 150,
    ):
        self.parser = PDFParser()
        self.chunker = AcademicChunker(
            target_chunk_size=target_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.documents: Dict[str, DocumentMetadata] = {}
        self.chunks_by_doc: Dict[str, List[DocumentChunk]] = {}
        self._load_cached_registry()

    def _get_cache_path(self, sha256_hash: str) -> Path:
        """Returns cache JSON path for a given file hash."""
        return CACHE_DIR / f"{sha256_hash}.json"

    def _load_cached_registry(self) -> None:
        """Loads cached document metadata and chunks from disk cache."""
        for cache_file in CACHE_DIR.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    meta = DocumentMetadata(**data["metadata"])
                    chunks = [DocumentChunk(**c) for c in data["chunks"]]
                    self.documents[meta.document_id] = meta
                    self.chunks_by_doc[meta.document_id] = chunks
            except Exception as e:
                logger.warning(f"Could not load cache file {cache_file}: {e}")

    def _save_to_cache(self, metadata: DocumentMetadata, chunks: List[DocumentChunk]) -> None:
        """Persists processed document and chunks to JSON cache."""
        cache_path = self._get_cache_path(metadata.sha256_hash)
        payload = {
            "metadata": metadata.model_dump(mode="json"),
            "chunks": [c.model_dump(mode="json") for c in chunks],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def process_file(self, file_path: str, force_reprocess: bool = False) -> Tuple[DocumentMetadata, List[DocumentChunk]]:
        """Processes a single PDF file (with caching)."""
        path = Path(file_path)
        file_hash = self.parser.compute_sha256(str(path))
        cache_file = self._get_cache_path(file_hash)

        if not force_reprocess and cache_file.exists():
            logger.info(f"Loading cached parsing for '{path.name}' ({file_hash[:8]})...")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                meta = DocumentMetadata(**data["metadata"])
                chunks = [DocumentChunk(**c) for c in data["chunks"]]
                self.documents[meta.document_id] = meta
                self.chunks_by_doc[meta.document_id] = chunks
                return meta, chunks

        logger.info(f"Processing PDF '{path.name}'...")
        metadata, pages_data = self.parser.parse_pdf(str(path))
        chunks = self.chunker.chunk_document(metadata, pages_data)

        self._save_to_cache(metadata, chunks)
        self.documents[metadata.document_id] = metadata
        self.chunks_by_doc[metadata.document_id] = chunks
        return metadata, chunks

    def process_files(self, file_paths: List[str], force_reprocess: bool = False) -> Dict[str, Any]:
        """Processes multiple PDF files."""
        processed_meta = []
        all_chunks = []
        errors = []

        for fp in file_paths:
            try:
                meta, chunks = self.process_file(fp, force_reprocess=force_reprocess)
                processed_meta.append(meta)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {fp}: {e}")
                errors.append({"file": fp, "error": str(e)})

        return {
            "processed_documents": processed_meta,
            "total_chunks": len(all_chunks),
            "chunks": all_chunks,
            "errors": errors,
        }

    def save_uploaded_file(self, file_bytes: bytes, filename: str) -> str:
        """Saves uploaded bytes to data/uploads directory and returns absolute path."""
        dest_path = UPLOADS_DIR / filename
        with open(dest_path, "wb") as f:
            f.write(file_bytes)
        return str(dest_path.resolve())

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Returns all chunks across all ingested documents."""
        all_chunks = []
        for chunks in self.chunks_by_doc.values():
            all_chunks.extend(chunks)
        return all_chunks

    def get_all_documents(self) -> List[DocumentMetadata]:
        """Returns metadata of all currently loaded documents."""
        return list(self.documents.values())

    def clear_all(self) -> None:
        """Clears all in-memory and disk cached documents."""
        self.documents.clear()
        self.chunks_by_doc.clear()
        for f in CACHE_DIR.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        for f in UPLOADS_DIR.glob("*.pdf"):
            try:
                f.unlink()
            except Exception:
                pass
        logger.info("Cleared all documents, chunks, and cache.")
