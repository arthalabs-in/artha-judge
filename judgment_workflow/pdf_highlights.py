from __future__ import annotations

from pathlib import Path
from typing import Any
import difflib
import re

import fitz

def build_highlight_positions(review_package_or_record: Any) -> list[dict[str, Any]]:
    payload = review_package_or_record
    if not isinstance(payload, dict):
        from .serialization import serialize_review_package

        payload = serialize_review_package(review_package_or_record)

    positions: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    counts: dict[str, int] = {}

    def add_evidence(evidence: dict[str, Any], position_type: str):
        counts[position_type] = counts.get(position_type, 0) + 1
        label = _position_label(position_type, counts[position_type])
        source_label = _source_label(evidence, label)
        key = (
            evidence.get("page"),
            tuple(evidence.get("bbox") or []),
            evidence.get("snippet"),
            position_type,
        )
        if key in seen or not evidence.get("page"):
            return
        seen.add(key)
        positions.append(
            {
                "page": evidence.get("page"),
                "bbox": evidence.get("bbox"),
                "text": evidence.get("snippet", ""),
                "type": position_type,
                "label": label,
                "source_label": source_label,
                "extraction_method": evidence.get("extraction_method"),
                "source_quality": evidence.get("source_quality"),
                "match_strategy": evidence.get("match_strategy"),
                "confidence": evidence.get("confidence"),
            }
        )

    extraction = payload.get("extraction", {})
    for field_name, field_value in extraction.items():
        if field_name == "risk_flags":
            continue
        if field_name == "directions":
            for direction in field_value:
                for evidence in direction.get("evidence", []):
                    add_evidence(evidence, "direction")
            continue
        for evidence in field_value.get("evidence", []):
            add_evidence(evidence, field_name)

    for action_item in payload.get("action_items", []):
        for evidence in action_item.get("evidence", []):
            add_evidence(evidence, "action_item")

    return positions


def generate_highlighted_pdf(
    original_pdf_path: str,
    output_pdf_path: str,
    review_package_or_record: Any,
) -> str:
    positions = build_highlight_positions(review_package_or_record)
    positions = enrich_highlight_positions(original_pdf_path, positions)
    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    _create_highlighted_pdf(original_pdf_path, output_pdf_path, positions)
    return output_pdf_path


def render_highlighted_page_png(
    highlighted_pdf_path: str,
    page_number: int,
    *,
    zoom: float = 1.75,
) -> bytes:
    doc = fitz.open(highlighted_pdf_path)
    try:
        page_index = max(0, min(int(page_number) - 1, doc.page_count - 1))
        page = doc.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")
    finally:
        doc.close()


def _create_highlighted_pdf(original_pdf_path: str, output_pdf_path: str, positions: list[dict[str, Any]]) -> None:
    doc = fitz.open(original_pdf_path)
    try:
        for position in positions:
            page_number = position.get("page")
            if not page_number:
                continue
            page_index = int(page_number) - 1
            if page_index < 0 or page_index >= doc.page_count:
                continue
            page = doc.load_page(page_index)
            markers = _markers_for_position(page, position)
            for marker in markers:
                highlight = page.add_highlight_annot(marker)
                highlight.set_colors(stroke=(1, 0.86, 0.15))
                highlight.set_opacity(0.42)
                highlight.set_info(content=_annotation_content(position))
                highlight.update()
            if markers:
                _add_visible_source_label(page, markers[0], position)
        doc.save(output_pdf_path, garbage=4, deflate=True)
    finally:
        doc.close()


def _markers_for_position(page: fitz.Page, position: dict[str, Any]) -> list[Any]:
    quads = _quads_from_position(position)
    if quads:
        return quads

    bbox = position.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [fitz.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))]

    match = _best_page_match(page, str(position.get("text") or ""))
    return match.get("quads") or match.get("rects") or []


def enrich_highlight_positions(original_pdf_path: str, positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc = fitz.open(original_pdf_path)
    try:
        enriched = []
        for position in positions:
            if not position.get("page"):
                enriched.append(position)
                continue
            page_index = int(position["page"]) - 1
            if page_index < 0 or page_index >= doc.page_count:
                enriched.append(position)
                continue
            match = _best_page_match(doc.load_page(page_index), str(position.get("text") or ""))
            if match.get("rects") or match.get("quads"):
                rect = match["rects"][0]
                updated = dict(position)
                updated.setdefault("bbox", [rect.x0, rect.y0, rect.x1, rect.y1])
                if match.get("quads"):
                    updated["quad_points"] = _serialize_quads(match["quads"])
                updated["match_strategy"] = match["strategy"]
                updated["locator_confidence"] = match["confidence"]
                updated["source_label"] = _source_label(updated, str(updated.get("label") or updated.get("type") or "Source"))
                enriched.append(updated)
            else:
                updated = dict(position)
                updated["match_strategy"] = "unmatched"
                updated["locator_confidence"] = 0.35
                updated["source_label"] = _source_label(updated, str(updated.get("label") or updated.get("type") or "Source"))
                enriched.append(updated)
        return enriched
    finally:
        doc.close()


def _best_page_match(page: fitz.Page, text: str) -> dict[str, Any]:
    for fragment in _search_fragments(text):
        quads = list(page.search_for(fragment, quads=True))
        if quads:
            rects = [quad.rect for quad in quads]
            return {"quads": quads, "rects": rects, "strategy": "pymupdf_quad_search", "confidence": 0.92}

    word_match = _word_window_match(page, text)
    if word_match:
        return word_match

    return {}


def _search_fragments(text: str) -> list[str]:
    words = [word.strip(".,:;()[]{}") for word in text.split() if word.strip(".,:;()[]{}")]
    if not words:
        return []
    fragments = [" ".join(words)]
    for size in (18, 14, 10, 8, 6, 4):
        if len(words) >= size:
            fragments.append(" ".join(words[:size]))
            fragments.append(" ".join(words[-size:]))
    seen: set[str] = set()
    return [fragment for fragment in fragments if fragment and not (fragment in seen or seen.add(fragment))]


def _word_window_match(page: fitz.Page, text: str) -> dict[str, Any] | None:
    target_tokens = _tokens(text)
    if len(target_tokens) < 4:
        return None

    words = []
    for raw in page.get_text("words"):
        token = _normalise_token(raw[4])
        if token:
            words.append({"token": token, "rect": fitz.Rect(raw[:4])})
    if not words:
        return None

    best: tuple[float, int, int] | None = None
    target_text = " ".join(target_tokens)
    max_window = min(max(len(target_tokens) + 5, 8), 36)
    min_window = min(max(4, len(target_tokens) - 5), max_window)
    for start in range(len(words)):
        for end in range(start + min_window, min(len(words), start + max_window) + 1):
            window_tokens = [item["token"] for item in words[start:end]]
            token_overlap = len(set(window_tokens) & set(target_tokens)) / max(1, len(set(target_tokens)))
            order_score = difflib.SequenceMatcher(None, " ".join(window_tokens), target_text).ratio()
            score = (token_overlap * 0.68) + (order_score * 0.32)
            if best is None or score > best[0]:
                best = (score, start, end)

    if not best or best[0] < 0.58:
        return None

    matched = words[best[1] : best[2]]
    rects = _merge_line_rects([item["rect"] for item in matched])
    quads = [_quad_from_rect(rect) for rect in rects]
    return {
        "quads": quads,
        "rects": rects,
        "strategy": "word_window_fuzzy",
        "confidence": round(min(0.88, max(0.62, best[0])), 2),
    }


def _merge_line_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    if not rects:
        return []
    lines: list[fitz.Rect] = []
    for rect in sorted(rects, key=lambda item: (round(item.y0 / 3), item.x0)):
        if lines and abs(lines[-1].y0 - rect.y0) < 3 and abs(lines[-1].y1 - rect.y1) < 3:
            lines[-1].include_rect(rect)
        else:
            lines.append(fitz.Rect(rect))
    return lines


def _quad_from_rect(rect: fitz.Rect) -> fitz.Quad:
    return fitz.Quad(
        [
            fitz.Point(rect.x0, rect.y0),
            fitz.Point(rect.x1, rect.y0),
            fitz.Point(rect.x0, rect.y1),
            fitz.Point(rect.x1, rect.y1),
        ]
    )


def _tokens(text: str) -> list[str]:
    return [token for token in (_normalise_token(part) for part in text.split()) if token]


def _normalise_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _position_label(position_type: str, index: int) -> str:
    labels = {
        "action_item": "Action Item",
        "direction": "Direction",
        "case_number": "Case Number",
        "court": "Court",
        "bench": "Bench",
        "judgment_date": "Judgment Date",
        "parties": "Parties",
        "petitioners": "Petitioners",
        "respondents": "Respondents",
        "disposition": "Disposition",
        "advocates": "Advocates",
        "departments": "Departments",
    }
    base = labels.get(str(position_type), str(position_type).replace("_", " ").title())
    return f"{base} {index}" if index > 1 and base in {"Action Item", "Direction"} else base


def _source_label(evidence: dict[str, Any], label: str) -> str:
    page = evidence.get("page") or "?"
    method = evidence.get("extraction_method") or evidence.get("source_quality") or "source"
    locator = evidence.get("match_strategy")
    confidence = evidence.get("locator_confidence") or evidence.get("confidence")
    parts = [label, f"p. {page}", str(method).replace("_", " ")]
    if locator:
        parts.append(str(locator).replace("_", " "))
    if confidence not in (None, ""):
        try:
            parts.append(f"{float(confidence):.2f}")
        except (TypeError, ValueError):
            pass
    return " | ".join(parts)


def _annotation_content(position: dict[str, Any]) -> str:
    source_label = str(position.get("source_label") or position.get("label") or "Source")
    snippet = str(position.get("text") or "").strip()
    if len(snippet) > 420:
        snippet = snippet[:417].rstrip() + "..."
    return f"{source_label}\n{snippet}".strip()


def _add_visible_source_label(page: fitz.Page, marker: Any, position: dict[str, Any]) -> None:
    try:
        rect = marker.rect if hasattr(marker, "rect") else fitz.Rect(marker)
        label = str(position.get("source_label") or position.get("label") or "Source")
        label = " | ".join(label.split(" | ")[:2])
        label_width = min(max(72, len(label) * 3.8), 190)
        y0 = max(2, rect.y0 - 14)
        label_rect = fitz.Rect(rect.x0, y0, min(page.rect.width - 2, rect.x0 + label_width), y0 + 11)
        annot = page.add_freetext_annot(
            label_rect,
            label,
            fontsize=6.5,
            text_color=(0.12, 0.15, 0.2),
            fill_color=(1, 0.96, 0.68),
            border_color=(0.86, 0.68, 0.18),
        )
        annot.set_opacity(0.92)
        annot.set_info(content=_annotation_content(position))
        annot.update()
    except Exception:
        return


def _serialize_quads(quads: list[fitz.Quad]) -> list[list[list[float]]]:
    return [
        [[point.x, point.y] for point in (quad.ul, quad.ur, quad.ll, quad.lr)]
        for quad in quads
    ]


def _quads_from_position(position: dict[str, Any]) -> list[fitz.Quad]:
    raw = position.get("quad_points")
    if not raw:
        return []
    if raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 2:
        raw = [raw[index : index + 4] for index in range(0, len(raw), 4)]
    quads = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            try:
                quads.append(fitz.Quad([fitz.Point(point[0], point[1]) for point in item]))
            except (TypeError, ValueError):
                continue
    return quads
