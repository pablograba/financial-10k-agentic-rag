"""
Text chunking for SEC 10-K filings with section detection.

Uses LangChain text splitters with hybrid section detection:
1. HTML headers (h1-h6) for section boundaries
2. Regex fallback for standard 10-K section patterns
"""

import logging
import re
from dataclasses import dataclass
from typing import Iterator

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
)

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    text: str
    section_title: str
    chunk_index: int
    char_count: int


SEC_SECTION_PATTERNS = [
    (re.compile(r"ITEM\s*1\.?\s*BUSINESS", re.IGNORECASE), "Item 1. Business"),
    (
        re.compile(r"ITEM\s*1A\.?\s*RISK\s*FACTORS?", re.IGNORECASE),
        "Item 1A. Risk Factors",
    ),
    (
        re.compile(r"ITEM\s*1B\.?\s*UNRESOLVED\s*STAFF", re.IGNORECASE),
        "Item 1B. Unresolved Staff Comments",
    ),
    (
        re.compile(r"ITEM\s*1C\.?\s*CYBERSECURITY", re.IGNORECASE),
        "Item 1C. Cybersecurity",
    ),
    (re.compile(r"ITEM\s*2\.?\s*PROPERTIES", re.IGNORECASE), "Item 2. Properties"),
    (
        re.compile(r"ITEM\s*3\.?\s*LEGAL\s*PROCEEDINGS", re.IGNORECASE),
        "Item 3. Legal Proceedings",
    ),
    (
        re.compile(r"ITEM\s*4\.?\s*MINE\s*SAFETY", re.IGNORECASE),
        "Item 4. Mine Safety Disclosures",
    ),
    (
        re.compile(r"ITEM\s*5\.?\s*MARKET\s*FOR", re.IGNORECASE),
        "Item 5. Market for Common Equity",
    ),
    (
        re.compile(r"ITEM\s*6\.?\s*(\[RESERVED\]|SELECTED)", re.IGNORECASE),
        "Item 6. Reserved/Selected Financial Data",
    ),
    (
        re.compile(r"ITEM\s*7\.?\s*MANAGEMENT.?S?\s*DISCUSSION", re.IGNORECASE),
        "Item 7. Management's Discussion and Analysis",
    ),
    (
        re.compile(r"ITEM\s*7A\.?\s*QUANTITATIVE", re.IGNORECASE),
        "Item 7A. Quantitative Disclosures About Market Risk",
    ),
    (
        re.compile(r"ITEM\s*8\.?\s*FINANCIAL\s*STATEMENTS", re.IGNORECASE),
        "Item 8. Financial Statements",
    ),
    (
        re.compile(r"ITEM\s*9\.?\s*CHANGES?\s*(IN\s*AND\s*)?DISAGREEMENTS?", re.IGNORECASE),
        "Item 9. Changes in and Disagreements with Accountants",
    ),
    (
        re.compile(r"ITEM\s*9A\.?\s*CONTROLS?\s*(AND\s*)?PROCEDURES?", re.IGNORECASE),
        "Item 9A. Controls and Procedures",
    ),
    (
        re.compile(r"ITEM\s*9B\.?\s*OTHER\s*INFORMATION", re.IGNORECASE),
        "Item 9B. Other Information",
    ),
    (
        re.compile(r"ITEM\s*9C\.?\s*DISCLOSURE", re.IGNORECASE),
        "Item 9C. Disclosure Regarding Foreign Jurisdictions",
    ),
    (
        re.compile(r"ITEM\s*10\.?\s*DIRECTORS", re.IGNORECASE),
        "Item 10. Directors, Executive Officers and Corporate Governance",
    ),
    (
        re.compile(r"ITEM\s*11\.?\s*EXECUTIVE\s*COMPENSATION", re.IGNORECASE),
        "Item 11. Executive Compensation",
    ),
    (
        re.compile(r"ITEM\s*12\.?\s*SECURITY\s*OWNERSHIP", re.IGNORECASE),
        "Item 12. Security Ownership",
    ),
    (
        re.compile(r"ITEM\s*13\.?\s*CERTAIN\s*RELATIONSHIPS", re.IGNORECASE),
        "Item 13. Certain Relationships and Related Transactions",
    ),
    (
        re.compile(r"ITEM\s*14\.?\s*PRINCIPAL\s*(ACCOUNTANT|ACCOUNTING)", re.IGNORECASE),
        "Item 14. Principal Accountant Fees and Services",
    ),
    (
        re.compile(r"ITEM\s*15\.?\s*EXHIBITS?", re.IGNORECASE),
        "Item 15. Exhibits and Financial Statement Schedules",
    ),
    (
        re.compile(r"ITEM\s*16\.?\s*FORM\s*10-?K\s*SUMMARY", re.IGNORECASE),
        "Item 16. Form 10-K Summary",
    ),
    (re.compile(r"PART\s+I\b", re.IGNORECASE), "Part I"),
    (re.compile(r"PART\s+II\b", re.IGNORECASE), "Part II"),
    (re.compile(r"PART\s+III\b", re.IGNORECASE), "Part III"),
    (re.compile(r"PART\s+IV\b", re.IGNORECASE), "Part IV"),
    (re.compile(r"SIGNATURES?\s*$", re.IGNORECASE), "Signatures"),
]


class Chunker:
    """
    Chunk text with section detection.

    Uses hybrid approach:
    1. Try to detect sections from text patterns
    2. Fall back to generic section name if no pattern matches
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_length: int | None = None,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
            min_chunk_length: Minimum chunk length to keep.
        """
        settings = get_settings()
        chars_per_token = 4

        self.chunk_size = chunk_size or (settings.chunk_size * chars_per_token)
        self.chunk_overlap = chunk_overlap or (settings.chunk_overlap * chars_per_token)
        self.min_chunk_length = min_chunk_length or (
            settings.min_chunk_length * chars_per_token
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str) -> list[TextChunk]:
        """
        Split text into chunks with section detection.

        Args:
            text: Text to chunk.

        Returns:
            list[TextChunk]: List of chunks with metadata.
        """
        chunks = list(self._chunk_with_sections(text))
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _chunk_with_sections(self, text: str) -> Iterator[TextChunk]:
        """
        Chunk text and detect sections.

        Maintains section state across chunks.
        """
        raw_chunks = self._splitter.split_text(text)

        current_section = "Preamble"
        chunk_index = 0

        for raw_text in raw_chunks:
            if len(raw_text.strip()) < self.min_chunk_length:
                continue

            detected_section = self._detect_section(raw_text)
            if detected_section:
                current_section = detected_section

            yield TextChunk(
                text=raw_text.strip(),
                section_title=current_section,
                chunk_index=chunk_index,
                char_count=len(raw_text),
            )
            chunk_index += 1

    def _detect_section(self, text: str) -> str | None:
        """
        Detect section from text using regex patterns.

        Returns section name if detected, None otherwise.
        """
        text_start = text[:500]

        for pattern, section_name in SEC_SECTION_PATTERNS:
            if pattern.search(text_start):
                return section_name

        return None


class HTMLChunker(Chunker):
    """
    Chunk HTML content with header-based section detection.

    Uses HTMLHeaderTextSplitter first to preserve structure,
    then falls back to regex patterns.
    """

    HEADERS_TO_SPLIT_ON = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]

    def __init__(self, **kwargs):
        """Initialize HTML chunker."""
        super().__init__(**kwargs)
        self._html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
        )

    def chunk_html(self, html_content: str) -> list[TextChunk]:
        """
        Chunk HTML content with header-based section detection.

        Args:
            html_content: Raw HTML string.

        Returns:
            list[TextChunk]: List of chunks with metadata.
        """
        try:
            header_splits = self._html_splitter.split_text(html_content)

            all_chunks = []
            chunk_index = 0
            current_section = "Preamble"

            for doc in header_splits:
                header_section = self._extract_section_from_headers(doc.metadata)
                if header_section:
                    current_section = header_section

                sub_chunks = self._splitter.split_text(doc.page_content)

                for sub_text in sub_chunks:
                    if len(sub_text.strip()) < self.min_chunk_length:
                        continue

                    regex_section = self._detect_section(sub_text)
                    section = regex_section or current_section

                    all_chunks.append(
                        TextChunk(
                            text=sub_text.strip(),
                            section_title=section,
                            chunk_index=chunk_index,
                            char_count=len(sub_text),
                        )
                    )
                    chunk_index += 1

            logger.info(f"Created {len(all_chunks)} chunks from HTML")
            return all_chunks

        except Exception as e:
            logger.warning(f"HTML chunking failed, falling back to text: {e}")
            from src.ingestion.html_parser import HTMLParser

            parser = HTMLParser()
            text = parser.parse_html(html_content)
            return self.chunk_text(text)

    def _extract_section_from_headers(self, metadata: dict) -> str | None:
        """
        Extract section name from header metadata.

        Checks if any header text matches a 10-K section pattern.
        """
        for key in ["Header 1", "Header 2", "Header 3", "Header 4"]:
            header_text = metadata.get(key, "")
            if header_text:
                for pattern, section_name in SEC_SECTION_PATTERNS:
                    if pattern.search(header_text):
                        return section_name

                if len(header_text) > 3 and len(header_text) < 100:
                    return header_text

        return None
