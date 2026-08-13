"""
Text cleaning and normalization pipeline for extracted research paper text.
"""

import re
import unicodedata
from typing import Optional, List, Dict


class TextCleaner:
    """Cleans raw PDF text extractions, removing hyphenations, headers, and artifacts."""

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalizes unicode characters and replaces ligatures."""
        if not text:
            return ""
        # NFKC decomposes combined characters and standardizes compatibility chars
        text = unicodedata.normalize("NFKC", text)
        # Common ligature replacements if any survived
        ligatures = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬆ": "st",
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "—": " -- ",
            "–": " - ",
        }
        for lig, rep in ligatures.items():
            text = text.replace(lig, rep)
        return text

    @staticmethod
    def fix_line_break_hyphenation(text: str) -> str:
        """
        Rejoins words that were split across lines with hyphens.
        e.g., 'transfor-\nmers' -> 'transformers'
        """
        # Match word char followed by hyphen, newline, and word char (lowercase usually)
        pattern = r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)"
        return re.sub(pattern, r"\1\2", text)

    @staticmethod
    def remove_running_headers_footers(text: str) -> str:
        """Removes common page headers/footers like 'Page 1 of 12' or 'arXiv:2301.xxxx'."""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip page numbers alone
            if re.match(r"^(Page\s+)?\d+(\s+of\s+\d+)?$", stripped, re.IGNORECASE):
                continue
            # Skip standalone arXiv header stamps
            if re.match(r"^arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[.*\]\s*.*$", stripped, re.IGNORECASE):
                continue
            # Skip copyright lines
            if re.match(r"^©\s*\d{4}\s+IEEE|ACM|Springer|Elsevier.*$", stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapses excessive empty lines and spaces while preserving paragraph breaks."""
        # Replace multiple spaces with a single space (except newlines)
        text = re.sub(r"[ \t]+", " ", text)
        # Replace 3 or more newlines with double newline (paragraph break)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()

    @classmethod
    def clean(cls, text: str) -> str:
        """Runs full end-to-end cleaning on extracted text."""
        if not text:
            return ""
        text = cls.normalize_unicode(text)
        text = cls.fix_line_break_hyphenation(text)
        text = cls.remove_running_headers_footers(text)
        text = cls.normalize_whitespace(text)
        return text
