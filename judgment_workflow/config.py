from __future__ import annotations

import os
from pathlib import Path


CANVAS_APP_ID = os.getenv("CANVAS_APP_ID", "gold-verve-463709-n2")
JUDGMENT_DATA_ROOT = Path(
    os.getenv(
        "JUDGMENT_DATA_ROOT",
        os.path.join("user_data", "judgments"),
    )
)
CCMS_BASE_URL = os.getenv("CCMS_BASE_URL")
CCMS_API_KEY = os.getenv("CCMS_API_KEY")
JUDGMENT_SYNC_PROCESSING = os.getenv("JUDGMENT_SYNC_PROCESSING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CCMS_PUSHBACK_ENABLED = os.getenv("CCMS_PUSHBACK_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
