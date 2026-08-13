"""
PDF Parser using PyMuPDF (fitz) with metadata extraction and section detection.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF

from app.models.schemas import DocumentMetadata
from app.ingestion.cleaner import TextCleaner
from app.utils.config import logger


class PDFParser:
    """Extracts text, metadata, page numbers, and detected sections from PDF research papers."""

    COMMON_SECTIONS = [
        "abstract",
        "introduction",
        "background",
        "related work",
        "literature review",
        "methodology",
        "method",
        "proposed method",
        "model architecture",
        "system design",
        "experiments",
        "experimental setup",
        "evaluation",
        "results",
        "results and discussion",
        "discussion",
        "limitations",
        "threats to validity",
        "ethical considerations",
        "broader impact",
        "future work",
        "conclusion",
        "conclusions",
        "acknowledgments",
        "references",
        "appendix",
    ]

    SECTION_HEADER_PATTERN = re.compile(
        r"^(?:\d+[\.\d]*\s+|[I|V|X]+\.\s+|[A-Z]\.\s+)?(" + "|".join(COMMON_SECTIONS) + r")\b.*$",
        re.IGNORECASE
    )

    def __init__(self):
        self.cleaner = TextCleaner()

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """Calculates SHA256 hash of a file for caching and deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def parse_pdf(self, file_path: str) -> Tuple[DocumentMetadata, List[Dict[str, Any]]]:
        """
        Parses a PDF research paper.
        Returns:
            - DocumentMetadata
            - List of page dicts: [{"page_number": int, "raw_text": str, "cleaned_text": str, "sections": List[str]}]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(f"PDF file is empty: {file_path}")

        file_hash = self.compute_sha256(file_path)
        doc_id = f"doc_{file_hash[:12]}"
        filename = path.name

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            raise ValueError(f"Corrupted or invalid PDF file {filename}: {str(e)}")

        page_count = len(doc)
        if page_count == 0:
            doc.close()
            raise ValueError(f"PDF has no pages: {filename}")

        pages_data = []
        all_detected_sections = set()
        first_page_text = ""

        for page_idx in range(page_count):
            page_num = page_idx + 1
            try:
                page = doc[page_idx]
                # Extract text using block ordering to preserve multi-column flow
                blocks = page.get_text("blocks")
                # Sort blocks primarily by y0 (vertical) and secondarily by x0 (horizontal)
                # For standard 2-column or 1-column layouts, block extraction in fitz handles sorting well
                page_text_pieces = []
                page_sections = []

                for block in blocks:
                    block_text = block[4] if len(block) >= 5 else ""
                    if not block_text.strip():
                        continue
                    
                    # Detect section headers
                    first_line = block_text.strip().split("\n")[0].strip()
                    match = self.SECTION_HEADER_PATTERN.match(first_line)
                    if match:
                        sec_name = match.group(1).strip().title()
                        page_sections.append(sec_name)
                        all_detected_sections.add(sec_name)

                    page_text_pieces.append(block_text)

                raw_page_text = "\n".join(page_text_pieces)
                cleaned_page_text = self.cleaner.clean(raw_page_text)

                if page_num == 1:
                    first_page_text = cleaned_page_text

                pages_data.append({
                    "page_number": page_num,
                    "raw_text": raw_page_text,
                    "cleaned_text": cleaned_page_text,
                    "sections": page_sections,
                })
            except Exception as e:
                logger.warning(f"Error reading page {page_num} in {filename}: {e}")
                pages_data.append({
                    "page_number": page_num,
                    "raw_text": "",
                    "cleaned_text": "",
                    "sections": [],
                })

        doc.close()

        # Extract paper metadata (title, authors, year, abstract) from page 1 / doc metadata
        title, authors, year, abstract = self._extract_metadata(filename, first_page_text, doc)

        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            file_path=str(path.resolve()),
            file_size_bytes=file_size,
            page_count=page_count,
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            detected_sections=sorted(list(all_detected_sections)),
            sha256_hash=file_hash,
        )

        return metadata, pages_data

    def _extract_metadata(
        self, filename: str, first_page_text: str, doc_obj: Any
    ) -> Tuple[Optional[str], List[str], Optional[int], Optional[str]]:
        """Extracts Title, Authors, Year, and Abstract using heuristics."""
        title = None
        authors = []
        year = None
        abstract = None

        # Try fitz document metadata first
        try:
            doc_meta = doc_obj.metadata if hasattr(doc_obj, "metadata") and doc_obj.metadata else {}
            if doc_meta.get("title") and len(doc_meta["title"].strip()) > 4:
                title = doc_meta["title"].strip()
            if doc_meta.get("author"):
                raw_authors = doc_meta["author"].replace(";", ",").split(",")
                authors = [a.strip() for a in raw_authors if a.strip()]
        except Exception:
            pass

        # Parse from first page text
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]

        # Heuristic for title if not in doc metadata
        if not title and lines:
            candidate_lines = []
            for line in lines[:6]:
                # Stop if we hit abstract or authors markers
                if re.match(r"^(abstract|introduction|\d+\.\s*introduction)", line, re.IGNORECASE):
                    break
                # Skip standalone arxiv stamps
                if "arxiv" in line.lower() or "doi" in line.lower() or "http" in line.lower():
                    continue
                candidate_lines.append(line)
                if len(" ".join(candidate_lines)) > 140:
                    break
            if candidate_lines:
                title = " ".join(candidate_lines)
            else:
                title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        elif not title:
            title = Path(filename).stem.replace("_", " ").replace("-", " ").title()

        # Heuristic for year
        year_matches = re.findall(r"\b(19\d\d|20[0-2]\d)\b", first_page_text[:1500])
        if year_matches:
            # Pick the most plausible recent year (up to 2026)
            valid_years = [int(y) for y in year_matches if 1980 <= int(y) <= 2026]
            if valid_years:
                year = max(valid_years)

        # Heuristic for abstract
        abstract_match = re.search(
            r"\babstract\b[:\s]*(.*?)(?=\b(?:1\.?\s*)?introduction\b|\bkeywords\b|\bindex terms\b|\b1\s+[A-Z]|\n\n\n|$)",
            first_page_text,
            re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            abstract_candidate = abstract_match.group(1).strip()
            if len(abstract_candidate) > 40:
                abstract = self.cleaner.normalize_whitespace(abstract_candidate)

        # Heuristic for authors if empty
        if not authors and len(lines) > 2:
            # Lines between title and abstract
            potential_author_lines = []
            capture = False
            for line in lines[:15]:
                if line == lines[0]:
                    capture = True
                    continue
                if re.match(r"^(abstract|keywords|1\.?\s*introduction)", line, re.IGNORECASE):
                    break
                if capture and not any(kw in line.lower() for kw in ["university", "department", "institute", "email", "@"]):
                    if len(line.split()) <= 8 and not re.search(r"\d{4}", line):
                        potential_author_lines.append(line)
            if potential_author_lines:
                authors = [a.strip() for a in potential_author_lines[0].split(",") if a.strip()]

        return title, authors, year, abstract
