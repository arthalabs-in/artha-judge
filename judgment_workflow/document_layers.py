from __future__ import annotations

import re
from typing import Any

import fitz
from langchain_core.documents import Document


def extract_layered_pdf_documents(pdf_path: str) -> list[Document]:
    """Extract page text with lightweight layout metadata for later layers."""
    documents: list[Document] = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            if not text.strip():
                continue
            blocks = page.get_text("blocks") or []
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_index,
                        "chunk_id": f"p{page_index}",
                        "extraction_method": "pymupdf_text",
                        "extraction_layer": "digital_text",
                        "page_role": classify_page_role(text, page_index),
                        "block_count": len(blocks),
                    },
                )
            )
    return documents


def classify_page_role(text: str, page_number: int) -> str:
    clean = " ".join(text.lower().split())
    if page_number <= 2 and re.search(r"\b(court|petition|appeal|versus|respondent|petitioner)\b", clean):
        return "case_header"
    if re.search(r"\b(judgment|order|reasons|analysis|held)\b", clean):
        return "judgment_body"
    if re.search(r"\b(annexure|memo|affidavit|schedule|appendix)\b", clean):
        return "attachment"
    if re.search(r"\b(directed|shall|within|disposed|allowed|dismissed|quashed)\b", clean):
        return "operative_order"
    return "unknown"


def metadata_for_metrics(documents: list[Document]) -> dict[str, Any]:
    roles: dict[str, int] = {}
    methods: dict[str, int] = {}
    for document in documents:
        metadata = document.metadata or {}
        role = str(metadata.get("page_role", "unknown"))
        method = str(metadata.get("extraction_method", "unknown"))
        roles[role] = roles.get(role, 0) + 1
        methods[method] = methods.get(method, 0) + 1
    return {"page_roles": roles, "extraction_methods": methods}
