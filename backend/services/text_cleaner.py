"""
backend/services/text_cleaner.py
─────────────────────────────────────────────────────────────
Text Cleaning and Normalization Service.

Key responsibilities:
  - Standardize line endings (\r\n and \r to \n)
  - Strip unprintable non-whitespace control characters
  - Collapse excessive spaces and tabs while preserving indentation/tables where relevant
  - Normalize paragraph boundaries (collapse 3+ newlines to 2)
  - Preserve markdown headings, lists, quotes, punctuation, and code blocks
─────────────────────────────────────────────────────────────
"""

import re
from typing import Optional


class TextCleaner:
    """
    Normalizes extracted raw text from PDF, DOCX, and TXT documents
    before semantic chunking.
    """

    # Regex matching control characters except \t (0x09) and \n (0x0A)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # Consecutive horizontal whitespace (spaces, non-breaking spaces, tabs)
    _HORIZONTAL_WS_RE = re.compile(r"[^\S\r\n]+")

    # 3 or more consecutive newlines
    _EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")

    @classmethod
    def clean(cls, text: Optional[str]) -> str:
        """
        Clean and normalize raw extracted document text.
        
        Args:
            text: Raw string extracted from document.
            
        Returns:
            Normalized string with preserved structural paragraphs and punctuation.
        """
        if not text:
            return ""

        # 1. Normalize line endings (\r\n and \r -> \n)
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Strip non-printable control characters (retaining \t and \n)
        cleaned = cls._CONTROL_CHAR_RE.sub("", cleaned)

        # 3. Normalize non-breaking spaces (\xa0) and thin spaces to standard space
        cleaned = cleaned.replace("\xa0", " ").replace("\u200b", "")

        # 4. Normalize lines: collapse redundant horizontal spaces per line while preserving indentation
        lines = []
        for line in cleaned.split("\n"):
            # Strip trailing spaces on each line
            rstripped = line.rstrip()
            if not rstripped:
                lines.append("")
                continue

            # Detect leading whitespace if any (for indentation/lists)
            indent_len = len(rstripped) - len(rstripped.lstrip())
            indent = rstripped[:indent_len]
            content = rstripped[indent_len:]

            # Collapse internal runs of spaces/tabs to a single space
            content_norm = cls._HORIZONTAL_WS_RE.sub(" ", content)
            lines.append(indent + content_norm)

        cleaned = "\n".join(lines)

        # 5. Collapse 3+ consecutive newlines to double newlines (\n\n) to preserve paragraph breaks
        cleaned = cls._EXCESSIVE_NEWLINES_RE.sub("\n\n", cleaned)

        # 6. Final trim of leading/trailing whitespace
        return cleaned.strip()
