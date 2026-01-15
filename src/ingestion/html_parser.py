"""
HTML parser for SEC 10-K filings.

Uses readability-lxml to extract main content and BeautifulSoup
for cleaning and text extraction.
"""

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from readability import Document as ReadabilityDocument

logger = logging.getLogger(__name__)


class HTMLParser:
    """
    Parse and clean HTML from SEC 10-K filings.

    SEC filings are XBRL/iXBRL documents with significant boilerplate.
    This parser extracts the main readable content.
    """

    ENCODINGS = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    REMOVE_TAGS = [
        "script",
        "style",
        "meta",
        "link",
        "head",
        "ix:header",
        "ix:hidden",
        "ix:references",
    ]

    NOISE_PATTERNS = [
        re.compile(r"^\s*$"),
        re.compile(r"^[\d\s,.$%()-]+$"),
        re.compile(r"^Table of Contents\s*$", re.IGNORECASE),
        re.compile(r"^Index\s*$", re.IGNORECASE),
        re.compile(r"^\s*Page\s+\d+\s*$", re.IGNORECASE),
    ]

    def __init__(self, min_text_length: int = 50):
        """
        Initialize HTML parser.

        Args:
            min_text_length: Minimum text length to return (skip near-empty files).
        """
        self.min_text_length = min_text_length

    def parse_file(self, file_path: str | Path) -> str:
        """
        Parse HTML file and extract clean text.

        Args:
            file_path: Path to HTML file.

        Returns:
            str: Cleaned text content.

        Raises:
            ValueError: If file is too short or parsing fails.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        html_content = self._read_file(file_path)
        return self.parse_html(html_content)

    def parse_html(self, html_content: str) -> str:
        """
        Parse HTML string and extract clean text.

        Uses a two-stage approach:
        1. readability-lxml extracts main content
        2. BeautifulSoup cleans and extracts text

        Args:
            html_content: Raw HTML string.

        Returns:
            str: Cleaned text content.

        Raises:
            ValueError: If content is too short after cleaning.
        """
        try:
            main_content = self._extract_main_content(html_content)
        except Exception as e:
            logger.warning(f"Readability failed, using direct parsing: {e}")
            main_content = html_content

        text = self._extract_text(main_content)
        text = self._clean_text(text)

        if len(text) < self.min_text_length:
            raise ValueError(
                f"Extracted text too short ({len(text)} chars, "
                f"min {self.min_text_length})"
            )

        return text

    def _read_file(self, file_path: Path) -> str:
        """Read file with encoding fallback."""
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _extract_main_content(self, html_content: str) -> str:
        """
        Use readability-lxml to extract main content.

        This removes navigation, headers, footers, and other boilerplate.
        """
        doc = ReadabilityDocument(html_content)
        return doc.summary()

    def _extract_text(self, html_content: str) -> str:
        """
        Extract text from HTML using BeautifulSoup.

        Preserves some structure by adding newlines after block elements.
        """
        soup = BeautifulSoup(html_content, "lxml")

        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for ix_tag in soup.find_all(re.compile(r"^ix:")):
            ix_tag.unwrap()

        text_parts = []
        self._recursive_extract(soup.body or soup, text_parts)

        return "\n".join(text_parts)

    def _recursive_extract(
        self,
        element,
        text_parts: list[str],
    ) -> None:
        """
        Recursively extract text from element tree.

        Adds newlines after block-level elements for structure preservation.
        """
        block_tags = {
            "p",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "tr",
            "section",
            "article",
            "br",
        }

        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text and not self._is_noise(text):
                    text_parts.append(text)
            elif hasattr(child, "name"):
                self._recursive_extract(child, text_parts)
                if child.name in block_tags:
                    if text_parts and text_parts[-1] != "":
                        text_parts.append("")

    def _is_noise(self, text: str) -> bool:
        """Check if text matches noise patterns (skip it)."""
        for pattern in self.NOISE_PATTERNS:
            if pattern.match(text):
                return True
        return False

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        - Normalize whitespace
        - Remove excessive blank lines
        - Strip leading/trailing whitespace
        """
        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def extract_headers(self, html_content: str) -> list[dict]:
        """
        Extract header hierarchy from HTML.

        Used for section detection in chunking.

        Args:
            html_content: HTML string.

        Returns:
            list[dict]: List of headers with level and text.
        """
        soup = BeautifulSoup(html_content, "lxml")
        headers = []

        for level in range(1, 7):
            for tag in soup.find_all(f"h{level}"):
                text = tag.get_text(strip=True)
                if text and len(text) > 2:
                    headers.append({
                        "level": level,
                        "text": text,
                        "tag": str(tag),
                    })

        return headers
