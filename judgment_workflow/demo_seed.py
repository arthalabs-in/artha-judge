from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz


def _real_demo_pdf_path() -> Path | None:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "user_data" / "evaluation_inputs" / "34897.pdf",
        project_root.parent / "user_data" / "evaluation_inputs" / "34897.pdf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_demo_pdf_bytes() -> bytes:
    real_demo_pdf = _real_demo_pdf_path()
    if real_demo_pdf:
        return real_demo_pdf.read_bytes()

    doc = fitz.open()
    page = doc.new_page()
    text = (
        "REPORTABLE\n"
        "IN THE HIGH COURT OF KARNATAKA AT BENGALURU\n"
        "WRIT PETITION NO. 1234 OF 2026\n"
        "Asha Residents Association ... Petitioner(s)\n"
        "VERSUS\n"
        "The State of Karnataka and Bruhat Bengaluru Mahanagara Palike ... Respondent(s)\n"
        "JUDGMENT DATED 15 March 2026\n"
        "CORAM: HON'BLE JUSTICE R. DEMO\n\n"
        "The writ petition is partly allowed. The BBMP is directed to remove the encroachment "
        "on the public road within four weeks. The Urban Development Department shall file "
        "a compliance report within 30 days. The Revenue Department shall release arrears "
        "payable to eligible petitioners within six weeks. Liberty is reserved to the State "
        "to seek legal review in accordance with law.\n"
    )
    page.insert_textbox(fitz.Rect(72, 72, 520, 760), text, fontsize=11)
    buffer = BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()

