"""
Intelligent Section-Aware Recursive Chunking for academic research papers.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from app.models.schemas import DocumentChunk, DocumentMetadata
from app.utils.config import logger


class AcademicChunker:
    """
    Splits research paper text into semantically cohesive chunks.
    Preserves:
    - Page numbers
    - Detected section hierarchy
    - Sentence boundaries
    - Overlap for context continuity
    """

    def __init__(
        self,
        target_chunk_size: int = 750,
        chunk_overlap: int = 150,
        min_chunk_size: int = 120,
    ):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Heuristic token estimation: ~4 chars per token in English academic text."""
        return max(1, len(text) // 4)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences while respecting common academic abbreviations (e.g., et al., e.g., i.e., Fig.)."""
        # Protect abbreviations with temporary placeholder
        abbrs = [
            (r"\bet al\.", "ET_AL_PLACEHOLDER"),
            (r"\be\.g\.", "EG_PLACEHOLDER"),
            (r"\bi\.e\.", "IE_PLACEHOLDER"),
            (r"\bFig\.", "FIG_PLACEHOLDER"),
            (r"\bEq\.", "EQ_PLACEHOLDER"),
            (r"\bvs\.", "VS_PLACEHOLDER"),
            (r"\bDr\.", "DR_PLACEHOLDER"),
            (r"\bProf\.", "PROF_PLACEHOLDER"),
            (r"\bpp\.", "PP_PLACEHOLDER"),
            (r"\bvol\.", "VOL_PLACEHOLDER"),
            (r"\bno\.", "NO_PLACEHOLDER"),
        ]
        protected = text
        for pattern, placeholder in abbrs:
            protected = re.sub(pattern, placeholder, protected, flags=re.IGNORECASE)

        # Split on sentence boundaries: period, exclamation, or question mark followed by whitespace and capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", protected)

        # Restore abbreviations
        restored = []
        for s in sentences:
            for pattern, placeholder in abbrs:
                orig = pattern.replace(r"\b", "").replace(r"\.", ".")
                s = s.replace(placeholder, orig)
            if s.strip():
                restored.append(s.strip())
        return restored

    def chunk_document(
        self, metadata: DocumentMetadata, pages_data: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Chunks pages extracted from a single document.
        Returns a list of DocumentChunk objects with full provenance metadata.
        """
        chunks: List[DocumentChunk] = []
        current_section = "Abstract" if metadata.abstract else "Introduction"
        chunk_idx = 0

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["cleaned_text"]
            page_sections = page.get("sections", [])

            if not page_text.strip():
                continue

            # Update current section if the page introduced a new section
            if page_sections:
                current_section = page_sections[0]

            # Split page text into paragraphs
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

            current_chunk_text = ""
            char_offset = 0

            for para in paragraphs:
                # Check if paragraph contains section header line
                lines = para.split("\n")
                first_line = lines[0].strip()
                for sec in metadata.detected_sections:
                    if sec.lower() in first_line.lower() and len(first_line) < 60:
                        current_section = sec
                        break

                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    sentence_len = len(sentence)
                    
                    # If adding this sentence exceeds target size and current chunk is non-empty:
                    if len(current_chunk_text) + sentence_len > self.target_chunk_size:
                        if len(current_chunk_text) >= self.min_chunk_size:
                            chunk_id = f"{metadata.document_id}_p{page_num}_c{chunk_idx}"
                            chunk_hash = hashlib.sha256(current_chunk_text.encode("utf-8")).hexdigest()
                            
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=chunk_id,
                                    document_id=metadata.document_id,
                                    filename=metadata.filename,
                                    page_number=page_num,
                                    section=current_section,
                                    char_start=char_offset,
                                    char_end=char_offset + len(current_chunk_text),
                                    token_count_est=self._estimate_tokens(current_chunk_text),
                                    text=current_chunk_text,
                                    sha256_hash=chunk_hash,
                                )
                            )
                            chunk_idx += 1
                            char_offset += len(current_chunk_text)

                            # Calculate overlap from trailing sentences
                            overlap_text = ""
                            if self.chunk_overlap > 0 and len(current_chunk_text) > self.chunk_overlap:
                                overlap_text = current_chunk_text[-self.chunk_overlap:].strip()
                                # Adjust to start at word boundary
                                if " " in overlap_text:
                                    overlap_text = overlap_text[overlap_text.find(" ") + 1:]
                            
                            current_chunk_text = (overlap_text + " " + sentence).strip() if overlap_text else sentence
                        else:
                            current_chunk_text += (" " + sentence if current_chunk_text else sentence)
                    else:
                        current_chunk_text += (" " + sentence if current_chunk_text else sentence)

            # Flush remaining text on page if meets minimum size or is last chunk
            if len(current_chunk_text.strip()) >= self.min_chunk_size:
                chunk_id = f"{metadata.document_id}_p{page_num}_c{chunk_idx}"
                chunk_hash = hashlib.sha256(current_chunk_text.encode("utf-8")).hexdigest()
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=metadata.document_id,
                        filename=metadata.filename,
                        page_number=page_num,
                        section=current_section,
                        char_start=char_offset,
                        char_end=char_offset + len(current_chunk_text),
                        token_count_est=self._estimate_tokens(current_chunk_text),
                        text=current_chunk_text,
                        sha256_hash=chunk_hash,
                    )
                )
                chunk_idx += 1

        logger.info(f"Chunked '{metadata.filename}': generated {len(chunks)} chunks across {metadata.page_count} pages.")
        return chunks
