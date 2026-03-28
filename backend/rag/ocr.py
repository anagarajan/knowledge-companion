"""
ocr.py — PDF text extraction with OCR fallback

Handles two types of PDFs:
  - Text PDFs: extract text directly with PyMuPDF (fast, accurate)
  - Scanned PDFs: convert each page to an image, run Tesseract OCR

Returns a list of pages, each with its text and page number.
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Minimum characters per page to consider it a text PDF.
# Below this threshold we treat the page as scanned and run OCR.
MIN_TEXT_CHARS = 50


@dataclass
class PageContent:
    page_number: int   # 1-indexed
    text: str
    was_ocr: bool      # True if OCR was used for this page


def extract_text_from_pdf(pdf_path: str) -> list[PageContent]:
    """
    Open a PDF and return text for every page.
    Automatically detects scanned pages and applies OCR.
    Tables are extracted as structured markdown to preserve column order.
    """
    pages: list[PageContent] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        raise

    logger.info(f"Processing {pdf_path} — {len(doc)} pages")

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        # Attempt direct text extraction first (fast path)
        raw_text = page.get_text("text")
        clean_text = _clean_text(raw_text)

        raw_stripped = raw_text.strip()
        # How much text did garble-line filtering remove from this page?
        # If we stripped >40% of the raw content, the PDF font encoding is
        # unreliable — OCR the rendered image instead for better coverage.
        heavy_garble = (
            len(raw_stripped) > MIN_TEXT_CHARS
            and len(clean_text) < len(raw_stripped) * 0.60
        )

        if len(clean_text) >= MIN_TEXT_CHARS and not _is_garbled(clean_text) and not heavy_garble:
            # Enough clean text found — this is a native text page.
            # Try to extract any tables as structured markdown — this preserves
            # column order and tick/cross characters that flat text extraction loses.
            table_md = _extract_tables_as_markdown(page)
            if table_md:
                final_text = clean_text + "\n\n" + table_md
                logger.info(f"  Page {page_number}: structured table(s) extracted")
            else:
                final_text = clean_text
            pages.append(PageContent(
                page_number=page_number,
                text=final_text,
                was_ocr=False,
            ))
        else:
            # Fall back to OCR — covers scanned pages, garbled encoding,
            # and pages where >40% of content was stripped by the garble filter.
            reason = (
                "scanned" if len(raw_stripped) < MIN_TEXT_CHARS
                else "garbled encoding — falling back to OCR"
            )
            logger.info(f"  Page {page_number}: {reason}")
            ocr_text = _ocr_page(page)
            clean_ocr = _clean_text(ocr_text)
            # Prefer OCR result if it produced more text than the filtered native text
            final_text = clean_ocr if len(clean_ocr) > len(clean_text) else clean_text
            pages.append(PageContent(
                page_number=page_number,
                text=final_text,
                was_ocr=True,
            ))

    doc.close()

    total_ocr = sum(1 for p in pages if p.was_ocr)
    logger.info(
        f"Done — {len(pages)} pages extracted, "
        f"{total_ocr} via OCR, {len(pages) - total_ocr} native text"
    )

    return pages


def _extract_tables_as_markdown(page: fitz.Page) -> str | None:
    """
    Use PyMuPDF's built-in table detector to extract tables as markdown.

    This preserves column order and special characters (✓ ✗) that are lost
    when reading table content as a flat text stream.  Returns None if no
    tables are found or extraction fails.
    """
    try:
        tabs = page.find_tables()
        if not tabs.tables:
            return None

        md_tables = []
        for table in tabs.tables:
            rows = table.extract()
            if not rows or len(rows) < 2:
                continue

            md_rows = []
            for i, row in enumerate(rows):
                cells = [_normalise_cell(cell) for cell in row]
                md_rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")

            md_tables.append("\n".join(md_rows))

        return "\n\n".join(md_tables) if md_tables else None
    except Exception as e:
        logger.debug(f"Table extraction skipped: {e}")
        return None


def _normalise_cell(cell: object) -> str:
    """Normalise a table cell value for markdown output."""
    if cell is None:
        return ""
    text = str(cell).strip()
    text = _normalise_tick_cross(text)
    # Escape pipe so markdown table structure is not broken
    return text.replace("|", "\\|")


def _normalise_tick_cross(text: str) -> str:
    """
    Map common OCR / font-encoding misreads of ✓ and ✗ to proper Unicode.
    Only applies when the entire cell content is a single tick or cross symbol —
    this avoids corrupting words that legitimately contain 'v' or 'x'.
    """
    stripped = text.strip()
    if stripped in {"v", "V", "y", "Y", "J", "√", "\u2713", "\u2714", "✓"}:
        return "✓"
    if stripped in {"x", "X", "×", "\u2717", "\u2718", "✗", "✕"}:
        return "✗"
    return text


def _ocr_page(page: fitz.Page, dpi: int = 300) -> str:
    """
    Render a PDF page to an image and extract text with Tesseract.
    Higher DPI = better OCR accuracy but slower. 300 is the sweet spot.
    """
    # Render page to a pixmap (in-memory image)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is PDF's native DPI
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert pixmap to PIL Image for Tesseract
    img_bytes = pixmap.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))

    # Run Tesseract OCR
    text = pytesseract.image_to_string(image, lang="eng")
    return text


def _is_garbled(text: str) -> bool:
    """
    Detect pages where PDF font encoding produced garbage characters.

    Some PDFs use non-standard font encoding — PyMuPDF extracts the raw
    character codes which look like percent-encoded junk (EMcW%*-"%-*23%...).
    A healthy page of English text should have >70% letters/spaces/common punct.
    If the ratio drops below that, OCR on the rendered image will be more accurate.
    """
    if not text:
        return True
    readable = sum(
        1 for c in text
        if c.isalpha() or c.isspace() or c in ".,;:!?-()'\"[]0123456789"
    )
    return (readable / len(text)) < 0.70


def _clean_text(text: str) -> str:
    """
    Remove excessive whitespace, blank lines, and garbled lines from extracted text.

    Some PDFs produce pages with mixed clean + garbled lines (custom font encoding).
    Lines where <60% of characters are normal readable text are dropped — they contain
    no useful information and corrupt the chunk embeddings.
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Drop individual lines that look like garbled encoding artifacts.
        # Garbled PDF font encoding produces lines with many % signs and
        # non-English character sequences (e.g. "EMcW%*-"%-*23%9:-5)74$").
        # Two signals: high % density OR low alpha+space ratio (excluding %).
        percent_count = stripped.count("%")
        alpha_space = sum(1 for c in stripped if c.isalpha() or c.isspace())
        if len(stripped) > 10:
            if percent_count / len(stripped) > 0.08:     # >8% percent signs = garbled
                continue
            if alpha_space / len(stripped) < 0.50:       # <50% letters/spaces = garbled
                continue
        cleaned.append(stripped)
    return "\n".join(cleaned)
