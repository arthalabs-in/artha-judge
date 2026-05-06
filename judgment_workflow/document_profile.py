from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import fitz

from .ocr import detect_ocr_need


def compute_document_hash(pdf_path: str) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_pdf(pdf_path: str) -> dict[str, Any]:
    ocr_status = detect_ocr_need(pdf_path)
    page_count = int(ocr_status.get("page_count", 0) or 0)
    sparse_pages = list(ocr_status.get("sparse_pages", []))
    total_chars = int(ocr_status.get("total_text_chars", 0) or 0)
    sparse_ratio = (len(sparse_pages) / page_count) if page_count else 1.0
    if total_chars == 0:
        profile_type = "scanned"
    elif sparse_ratio >= 0.5:
        profile_type = "mixed"
    elif sparse_pages:
        profile_type = "low_text"
    else:
        profile_type = "digital"
    return {
        "file_name": Path(pdf_path).name,
        "profile_type": profile_type,
        "page_count": page_count,
        "total_text_chars": total_chars,
        "sparse_pages": sparse_pages,
        "ocr_available": bool(ocr_status.get("ocr_available")),
        "ocr_used": False,
        "source_quality": "text" if profile_type == "digital" else "mixed",
    }


def first_page_preview(pdf_path: str, max_chars: int = 1200) -> str:
    doc = fitz.open(pdf_path)
    try:
        if doc.page_count == 0:
            return ""
        return doc.load_page(0).get_text("text")[:max_chars]
    finally:
        doc.close()

