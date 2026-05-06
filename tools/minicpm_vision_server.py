from __future__ import annotations

import argparse
import base64
import io
import json
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel


class ChatCompletionRequest(BaseModel):
    model: str = "openbmb/MiniCPM-o-4_5"
    messages: list[dict[str, Any]]
    temperature: float | None = 0.0
    max_tokens: int | None = 1400


def create_app(model_id: str, device: str = "cuda") -> FastAPI:
    app = FastAPI(title="MiniCPM-o Vision Sidecar")
    state: dict[str, Any] = {}

    @app.on_event("startup")
    def load_model() -> None:
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        actual_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype,
            init_vision=True,
            init_audio=False,
            init_tts=False,
        )
        model.eval()
        if actual_device == "cuda":
            model.cuda()
        state["model"] = model
        state["device"] = actual_device

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": "model" in state, "model": model_id, "device": state.get("device")}

    @app.post("/v1/chat/completions")
    def chat_completion(payload: ChatCompletionRequest) -> dict[str, Any]:
        model = state.get("model")
        if model is None:
            raise HTTPException(status_code=503, detail="MiniCPM-o model is not loaded yet.")
        prompt, images = _extract_prompt_and_images(payload.messages)
        if not images:
            raise HTTPException(status_code=400, detail="At least one image_url item is required.")
        msgs = [{"role": "user", "content": [*images, prompt]}]
        with torch.inference_mode():
            response = model.chat(
                msgs=msgs,
                max_new_tokens=payload.max_tokens or 1400,
                do_sample=bool(payload.temperature and payload.temperature > 0),
                temperature=payload.temperature or 0.0,
                use_image_id=False,
                max_slice_nums=1,
                use_tts_template=False,
                enable_thinking=False,
            )
        content = _jsonish_response(response)
        return {
            "id": "minicpm-vision-local",
            "object": "chat.completion",
            "model": model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    return app


def _extract_prompt_and_images(messages: list[dict[str, Any]]) -> tuple[str, list[Image.Image]]:
    prompt_parts: list[str] = []
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            prompt_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                prompt_parts.append(str(item.get("text") or ""))
            elif item.get("type") == "image_url":
                url = ((item.get("image_url") or {}).get("url") or "")
                images.append(_decode_data_url_image(url))
    return "\n\n".join(part for part in prompt_parts if part.strip()), images


def _decode_data_url_image(url: str) -> Image.Image:
    if "," not in url:
        raise HTTPException(status_code=400, detail="Only data URL images are supported by the local sidecar.")
    encoded = url.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc


def _jsonish_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    return json.dumps(response, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve MiniCPM-o vision extraction behind an OpenAI-compatible endpoint.")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args.model, device=args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
