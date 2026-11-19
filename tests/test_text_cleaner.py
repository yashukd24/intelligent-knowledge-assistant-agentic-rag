"""
tests/test_text_cleaner.py
─────────────────────────────────────────────────────────────
Unit tests for TextCleaner service.
─────────────────────────────────────────────────────────────
"""

import pytest
from backend.services.text_cleaner import TextCleaner


def test_clean_empty_or_none():
    assert TextCleaner.clean("") == ""
    assert TextCleaner.clean(None) == ""
    assert TextCleaner.clean("   \n\t  ") == ""


def test_clean_line_endings():
    raw = "Line 1\r\nLine 2\rLine 3\nLine 4"
    expected = "Line 1\nLine 2\nLine 3\nLine 4"
    assert TextCleaner.clean(raw) == expected


def test_clean_control_characters():
    # Null byte \x00, bell \x07, form feed \x0c should be stripped,
    # while standard whitespace formatting (newlines and leading indents) is kept.
    raw = "Hello\x00 World!\x07 This is\x0c a test.\n\tIndented line.\nNormal line."
    expected = "Hello World! This is a test.\n\tIndented line.\nNormal line."
    assert TextCleaner.clean(raw) == expected


def test_clean_excessive_whitespace():
    raw = "Word1     Word2\t\t\tWord3"
    expected = "Word1 Word2 Word3"
    assert TextCleaner.clean(raw) == expected


def test_clean_excessive_newlines_preserves_paragraphs():
    raw = "Paragraph 1.\n\n\n\n\n\nParagraph 2.\n\n\nParagraph 3."
    expected = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    assert TextCleaner.clean(raw) == expected


def test_clean_preserves_markdown_and_punctuation():
    raw = """
    # Section 1: Academic Regulations (2026)

    * Clause 1.1: Students must maintain 75% attendance.
    * Clause 1.2: Minimum passing grade is 40.0%.

    > Note: Exception applies for approved medical leave.

    Code snippet:
        def compute_gpa(credits, marks):
            return sum(credits * marks) / sum(credits)
    """
    cleaned = TextCleaner.clean(raw)
    assert "# Section 1: Academic Regulations (2026)" in cleaned
    assert "* Clause 1.1: Students must maintain 75% attendance." in cleaned
    assert "> Note: Exception applies for approved medical leave." in cleaned
    assert "def compute_gpa(credits, marks):" in cleaned


def test_clean_non_breaking_spaces():
    raw = "Hello\xa0World\u200b!"
    expected = "Hello World!"
    assert TextCleaner.clean(raw) == expected


def test_clean_strips_trailing_line_whitespace():
    raw = "Line 1 with trailing spaces      \nLine 2 with trailing tabs\t\t\nLine 3"
    expected = "Line 1 with trailing spaces\nLine 2 with trailing tabs\nLine 3"
    assert TextCleaner.clean(raw) == expected
