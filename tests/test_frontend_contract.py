from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_review_pane_is_wide_and_animates_without_display_toggle():
    css = (ROOT / "frontend" / "judgments.css").read_text(encoding="utf-8")
    fixed_pane_rule = re.search(r"body:has\(\.review-pane\.open\).*?\.review-pane\s*\{(?P<body>.*?)\n\}", css, re.S)

    assert "width: min(1120px, calc(100vw - 28px));" in css
    assert ".review-pane.closing" in css
    assert "visibility: hidden;" in css
    assert ".review-pane.open" in css and "visibility: visible;" in css
    assert fixed_pane_rule
    assert "display: none;" not in fixed_pane_rule.group("body")


def test_action_plan_cards_use_decision_ready_layout():
    js = (ROOT / "frontend" / "judgments.js").read_text(encoding="utf-8")
    css = (ROOT / "frontend" / "judgments.css").read_text(encoding="utf-8")

    assert "action-card-main" in js
    assert "action-card-controls" in js
    assert "action-meta-strip" in js
    assert "source-proof" in js
    assert ".action-card-main" in css
    assert ".source-proof" in css


def test_review_ui_exposes_manual_override_for_unsourced_human_edits():
    js = (ROOT / "frontend" / "judgments.js").read_text(encoding="utf-8")

    assert "data-field-manual-override" in js
    assert 'data-prop="manual_override"' in js


def test_review_screen_source_pdf_order_and_readability():
    html = (ROOT / "frontend" / "judgments.html").read_text(encoding="utf-8")
    css = (ROOT / "frontend" / "judgments.css").read_text(encoding="utf-8")
    js = (ROOT / "frontend" / "judgments.js").read_text(encoding="utf-8")

    assert html.index("extraction-review-card") < html.index("source-review-card") < html.index("review-details")
    assert '<img id="pdfFrame"' in html
    assert 'id="pdfPrevBtn"' in html
    assert 'id="pdfNextBtn"' in html
    assert 'id="pdfPageInput"' in html
    assert "height: clamp(560px, 68vh, 820px);" in css
    assert "grid-template-columns: 24px minmax(142px, 0.54fr) minmax(280px, 1fr) minmax(390px, 1.12fr) minmax(210px, auto);" in css
    assert "max-width: 210px;" in css
    assert "data-edit-icon" in js
    assert "source-proof-note" in js
    assert "renderEvidenceList" in js
    assert "source-chip" in js
    assert "loadHighlightedPage" in js
    assert "pdfPrevBtn" in js
    assert "pdfNextBtn" in js
    assert "highlightedPdfUrl(recordId, userId)" in js
    assert "highlightedPageUrl(recordId, userId, 1)" in js
    assert ".source-proof-note" in css
    assert ".source-chip" in css
    assert ".pdf-viewer-toolbar" in css
