"""
Metadata extraction from SEC 10-K filings.

Extracts company information, filing date, and other metadata from:
1. full-submission.txt header (primary source)
2. File path (fallback)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FilingMetadata:
    """Metadata extracted from a 10-K filing."""

    cik: str
    ticker: str
    company_name: str
    filing_date: str
    fiscal_year: int
    accession_number: str
    source_path: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "cik": self.cik,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_date": self.filing_date,
            "fiscal_year": self.fiscal_year,
            "accession_number": self.accession_number,
            "source_path": self.source_path,
        }


class MetadataExtractor:
    """Extract metadata from SEC 10-K filings."""

    HEADER_PATTERNS = {
        "accession_number": re.compile(
            r"ACCESSION NUMBER:\s*(\d{10}-\d{2}-\d{6})", re.IGNORECASE
        ),
        "company_name": re.compile(
            r"COMPANY CONFORMED NAME:\s*(.+)", re.IGNORECASE
        ),
        "cik": re.compile(
            r"CENTRAL INDEX KEY:\s*(\d+)", re.IGNORECASE
        ),
        "filing_date": re.compile(
            r"FILED AS OF DATE:\s*(\d{8})", re.IGNORECASE
        ),
        "fiscal_year_end": re.compile(
            r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", re.IGNORECASE
        ),
        "fiscal_year_end_alt": re.compile(
            r"FISCAL YEAR END:\s*(\d{4})", re.IGNORECASE
        ),
    }

    PATH_PATTERN = re.compile(
        r"sec-edgar-filings/([A-Z]+)/10-K/(\d{10}-\d{2}-\d{6})"
    )

    def extract(
        self,
        file_path: str | Path,
        ticker: str | None = None,
        fallback_year: int | None = None,
    ) -> FilingMetadata:
        """
        Extract metadata from a 10-K filing.

        Attempts to parse full-submission.txt first, falls back to path parsing.

        Args:
            file_path: Path to the primary-document.html file.
            ticker: Known ticker symbol (used as fallback).
            fallback_year: Known year (used as fallback).

        Returns:
            FilingMetadata: Extracted metadata.
        """
        file_path = Path(file_path)
        filing_dir = file_path.parent

        path_info = self._extract_from_path(str(file_path))

        submission_file = filing_dir / "full-submission.txt"
        header_info = {}
        if submission_file.exists():
            header_info = self._extract_from_header(submission_file)

        cik = header_info.get("cik") or path_info.get("cik") or "unknown"
        accession = (
            header_info.get("accession_number")
            or path_info.get("accession_number")
            or "unknown"
        )
        company_name = header_info.get("company_name") or ticker or "Unknown Company"
        filing_date = self._format_date(header_info.get("filing_date"))
        fiscal_year = self._extract_fiscal_year(header_info, fallback_year)
        ticker_final = ticker or path_info.get("ticker") or "UNKNOWN"

        return FilingMetadata(
            cik=self._normalize_cik(cik),
            ticker=ticker_final.upper(),
            company_name=company_name.strip(),
            filing_date=filing_date,
            fiscal_year=fiscal_year,
            accession_number=accession,
            source_path=str(file_path),
        )

    def _extract_from_path(self, file_path: str) -> dict:
        """
        Extract metadata from file path.

        Expected format: .../sec-edgar-filings/{TICKER}/10-K/{ACCESSION}/...
        """
        match = self.PATH_PATTERN.search(file_path)
        if match:
            ticker, accession = match.groups()
            cik = accession.split("-")[0]
            return {
                "ticker": ticker,
                "accession_number": accession,
                "cik": cik,
            }
        return {}

    def _extract_from_header(self, submission_file: Path) -> dict:
        """
        Extract metadata from full-submission.txt SEC header.

        Only reads the first 5KB (header section).
        """
        result = {}
        try:
            with open(submission_file, "r", encoding="utf-8", errors="ignore") as f:
                header_content = f.read(5000)

            for key, pattern in self.HEADER_PATTERNS.items():
                match = pattern.search(header_content)
                if match:
                    result[key] = match.group(1).strip()

        except Exception as e:
            logger.warning(f"Failed to parse submission header: {e}")

        return result

    def _normalize_cik(self, cik: str) -> str:
        """Normalize CIK to 10-digit zero-padded format."""
        if cik == "unknown":
            return cik
        digits = re.sub(r"\D", "", cik)
        return digits.zfill(10)

    def _format_date(self, date_str: str | None) -> str:
        """
        Format date from YYYYMMDD to YYYY-MM-DD.

        Args:
            date_str: Date in YYYYMMDD format.

        Returns:
            str: Date in YYYY-MM-DD format, or "unknown".
        """
        if not date_str or len(date_str) != 8:
            return "unknown"
        try:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except Exception:
            return "unknown"

    def _extract_fiscal_year(
        self,
        header_info: dict,
        fallback_year: int | None,
    ) -> int:
        """
        Extract fiscal year from header info.

        Tries fiscal_year_end first, then fiscal_year_end_alt, then fallback.
        """
        period = header_info.get("fiscal_year_end")
        if period and len(period) >= 4:
            try:
                return int(period[:4])
            except ValueError:
                pass

        year_end = header_info.get("fiscal_year_end_alt")
        if year_end:
            try:
                return int(year_end)
            except ValueError:
                pass

        if fallback_year:
            return fallback_year

        return 0


def extract_accession_from_path(file_path: str | Path) -> str | None:
    """
    Quick utility to extract accession number from file path.

    Args:
        file_path: Path containing accession number.

    Returns:
        str: Accession number or None.
    """
    match = MetadataExtractor.PATH_PATTERN.search(str(file_path))
    if match:
        return match.group(2)
    return None


def extract_ticker_from_path(file_path: str | Path) -> str | None:
    """
    Quick utility to extract ticker from file path.

    Args:
        file_path: Path containing ticker.

    Returns:
        str: Ticker symbol or None.
    """
    match = MetadataExtractor.PATH_PATTERN.search(str(file_path))
    if match:
        return match.group(1)
    return None
