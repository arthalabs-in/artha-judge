from __future__ import annotations

import importlib.util
import io
import shutil
from typing import Any

import fitz
from langchain_core.documents import Document
from PIL import Image


def detect_ocr_need(pdf_path: str, *, min_text_chars_per_page: int = 80) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    sparse_pages = []
    total_chars = 0

    for page_index in range(doc.page_count):
        text = doc.load_page(page_index).get_text("text").strip()
        total_chars += len(text)
        if len(text) < min_text_chars_per_page:
            sparse_pages.append(page_index + 1)

    doc.close()
    return {
        "needs_ocr": len(sparse_pages) > 0,
        "page_count": page_count,
        "sparse_pages": sparse_pages,
        "total_text_chars": total_chars,
        "ocr_available": _ocr_available(),
    }


def ocr_pdf_with_tesseract(
    pdf_path: str,
    *,
    target_pages: list[int] | None = None,
) -> tuple[list[Document], dict[str, Any]]:
    if not _ocr_available():
        return [], {"ocr_used": False, "reason": "tesseract_unavailable"}

    import pytesseract  # type: ignore

    doc = fitz.open(pdf_path)
    pages = target_pages or [index + 1 for index in range(doc.page_count)]
    documents: list[Document] = []

    for page_number in pages:
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        text = pytesseract.image_to_string(image).strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_number,
                    "document_type": "pdf",
                    "degraded_extraction": True,
                    "locator_confidence": 0.35,
                    "ocr_used": True,
                },
            )
        )

    doc.close()
    return documents, {"ocr_used": bool(documents), "pages": pages}


def _ocr_available() -> bool:
    return bool(importlib.util.find_spec("pytesseract")) and bool(shutil.which("tesseract"))
