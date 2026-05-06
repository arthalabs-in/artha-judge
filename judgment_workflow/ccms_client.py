from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
import httpx

from .config import CCMS_PUSHBACK_ENABLED
from .demo_seed import build_demo_pdf_bytes


@dataclass
class CCMSClient:
    base_url: str | None
    api_key: str | None
    timeout_s: float = 20.0
    mock_cases: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def fetch_case(self, ccms_case_id: str) -> dict[str, Any]:
        if ccms_case_id in self.mock_cases:
            case = self.mock_cases[ccms_case_id]
            pdf_bytes = case.get("pdf_bytes")
            pdf_path = case.get("pdf_path")
            if pdf_bytes is None and pdf_path:
                pdf_bytes = Path(pdf_path).read_bytes()
            return {
                "ccms_case_id": ccms_case_id,
                "metadata": dict(case.get("metadata", {})),
                "pdf_bytes": pdf_bytes,
            }

        if not self.base_url:
            return {
                "ccms_case_id": ccms_case_id,
                "metadata": {
                    "source_system": "mock_ccms",
                    "case_source": "generated_fallback",
                },
                "pdf_bytes": _build_mock_pdf_bytes(ccms_case_id),
            }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(f"{self.base_url.rstrip('/')}/cases/{ccms_case_id}", headers=headers)
            response.raise_for_status()
            payload = response.json()
            pdf_url = payload.get("pdf_url")
            if not pdf_url:
                raise ValueError(f"CCMS response for '{ccms_case_id}' did not include 'pdf_url'.")
            pdf_response = await client.get(pdf_url, headers=headers)
            pdf_response.raise_for_status()
            return {
                "ccms_case_id": ccms_case_id,
                "metadata": payload.get("metadata", {}),
                "pdf_bytes": pdf_response.content,
            }

    async def push_verified_action_plan(self, ccms_case_id: str, action_plan: dict[str, Any]) -> dict[str, Any]:
        if not CCMS_PUSHBACK_ENABLED or not self.base_url:
            return {"enabled": False, "status": "not_configured"}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/cases/{ccms_case_id}/action-plan",
                headers=headers,
                json=action_plan,
            )
            response.raise_for_status()
            return {"enabled": True, "status": "pushed", "response": response.json()}


def _build_mock_pdf_bytes(ccms_case_id: str) -> bytes:
    del ccms_case_id
    return build_demo_pdf_bytes()
