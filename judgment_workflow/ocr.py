from __future__ import annotations

import importlib.util
import io
import re
import shutil
from typing import Any

import fitz
from langchain_core.documents import Document
from PIL import Image


def detect_ocr_need(pdf_path: str, *, min_text_chars_per_page: int = 80) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    sparse_pages = []
    unreliable_text_pages = []
    page_quality_scores: dict[int, float] = {}
    image_pages = []
    total_chars = 0

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text").strip()
        has_images = bool(page.get_images(full=True))
        if has_images:
            image_pages.append(page_index + 1)
        total_chars += len(text)
        if len(text) < min_text_chars_per_page:
            sparse_pages.append(page_index + 1)
        else:
            quality = _embedded_text_quality(text)
            page_quality_scores[page_index + 1] = quality["score"]
            if quality["unreliable"] or (has_images and quality["score"] < 0.5):
                unreliable_text_pages.append(page_index + 1)

    doc.close()
    text_layer_reliable = not unreliable_text_pages
    return {
        "needs_ocr": bool(sparse_pages or unreliable_text_pages),
        "page_count": page_count,
        "sparse_pages": sparse_pages,
        "unreliable_text_pages": unreliable_text_pages,
        "image_pages": image_pages,
        "text_layer_reliable": text_layer_reliable,
        "page_text_quality": page_quality_scores,
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


_LEGAL_TEXT_MARKERS = {
    "court",
    "high",
    "petition",
    "petitioner",
    "respondent",
    "judgment",
    "justice",
    "writ",
    "appeal",
    "order",
    "dated",
    "versus",
    "bench",
    "coram",
    "allowed",
    "dismissed",
    "directed",
    "learned",
    "counsel",
}


def _embedded_text_quality(text: str) -> dict[str, Any]:
    normalized = " ".join(text.split())
    tokens = re.findall(r"[A-Za-z0-9.]+", normalized)
    word_tokens = re.findall(r"[A-Za-z]{2,}", normalized)
    if not normalized or not tokens:
        return {"score": 0.0, "unreliable": True}

    mixed_alnum = [
        token for token in tokens
        if any(char.isalpha() for char in token) and any(char.isdigit() for char in token)
    ]
    vowel_words = [token for token in word_tokens if re.search(r"[aeiouAEIOU]", token)]
    legal_hits = sum(1 for token in word_tokens if token.lower().strip(".") in _LEGAL_TEXT_MARKERS)
    symbol_chars = sum(1 for char in normalized if not (char.isalnum() or char.isspace() or char in ".,:;/-()[]&'\""))

    mixed_ratio = len(mixed_alnum) / max(len(tokens), 1)
    vowel_ratio = len(vowel_words) / max(len(word_tokens), 1)
    symbol_ratio = symbol_chars / max(len(normalized), 1)
    legal_density = legal_hits / max(len(word_tokens), 1)
    score = max(0.0, min(1.0, (vowel_ratio * 0.55) + (legal_density * 3.0) - (mixed_ratio * 0.9) - (symbol_ratio * 0.8)))

    unreliable = (
        len(normalized) >= 80
        and (
            mixed_ratio >= 0.18
            or (vowel_ratio < 0.52 and legal_hits < 2)
            or score < 0.38
        )
    )
    return {
        "score": round(score, 3),
        "unreliable": unreliable,
        "mixed_alnum_ratio": round(mixed_ratio, 3),
        "vowel_word_ratio": round(vowel_ratio, 3),
        "legal_marker_hits": legal_hits,
    }
