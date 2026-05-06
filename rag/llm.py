"""LLM utilities extracted from rag_core.py (router-based, multi-provider)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Union

import tiktoken

from llm_router import call_llm, call_llm_stream, get_request_identity

from .config import DEFAULT_REQUESTS_PER_MINUTE, DEFAULT_TOKENS_PER_MINUTE


class TokenRateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps: List[float] = []
        self.token_timestamps: List[tuple[float, int]] = []

    async def wait(self, tokens_to_send: int):
        now = time.time()

        # Clean up old timestamps
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
        self.token_timestamps = [t for t in self.token_timestamps if now - t[0] < 60]

        # Check request rate
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            await asyncio.sleep(max(0.0, sleep_time))

        # Check token rate
        if (
            sum(t[1] for t in self.token_timestamps) + tokens_to_send
            > self.tokens_per_minute
        ):
            sleep_time = 60 - (now - self.token_timestamps[0][0])
            await asyncio.sleep(max(0.0, sleep_time))

        self.request_timestamps.append(now)
        self.token_timestamps.append((now, tokens_to_send))


rate_limiter = TokenRateLimiter(
    requests_per_minute=DEFAULT_REQUESTS_PER_MINUTE,
    tokens_per_minute=DEFAULT_TOKENS_PER_MINUTE,
)


MessageLike = Union[str, Dict[str, str], Any]
MessagesLike = Union[MessageLike, Iterable[MessageLike]]
TelemetryCallback = Callable[[dict[str, Any]], None]

_CACHE_VERSION = "llm-raw-cache-v1"
_singleflight_lock: asyncio.Lock | None = None
_singleflight_tasks: dict[str, asyncio.Task[str]] = {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cache_dir() -> Path:
    return Path(os.getenv("LLM_RAW_CACHE_DIR", "user_data/llm_raw_cache"))


def _request_identity(
    *,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    provider_identity = get_request_identity(model=model_name)
    return {
        "cache_version": _CACHE_VERSION,
        "provider": provider_identity.get("provider"),
        "base_url": provider_identity.get("base_url"),
        "model": provider_identity.get("model") or model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }


def _cache_key(identity: dict[str, Any]) -> str:
    data = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _read_cached_raw_response(key: str) -> str | None:
    path = _cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = payload.get("raw_response")
    return raw if isinstance(raw, str) else None


def _write_cached_raw_response(key: str, identity: dict[str, Any], raw_response: str) -> None:
    directory = _cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / f"{key}.json"
    payload = {
        "cache_version": _CACHE_VERSION,
        "request_identity": identity,
        "raw_response": raw_response,
    }
    fd, temp_name = tempfile.mkstemp(prefix=f"{key}.", suffix=".tmp", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        os.replace(temp_name, final_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _emit(callback: TelemetryCallback | None, event: dict[str, Any]) -> None:
    if callback is not None:
        callback(event)


def _timeout_indicator(exc: BaseException) -> bool:
    name = exc.__class__.__name__.lower()
    return "timeout" in name or "timed out" in str(exc).lower()


def _to_dict_messages(messages: MessagesLike) -> List[Dict[str, str]]:
    """
    Normalize any message input to a list of dicts: {"role": "...", "content": "..."}.
    Supports strings, dicts, LangChain message objects (duck-typed), or lists of those.
    """

    def convert_one(m: MessageLike) -> Dict[str, str]:
        if isinstance(m, dict) and "role" in m and "content" in m:
            return {"role": str(m["role"]), "content": str(m["content"])}
        if hasattr(m, "content"):  # LangChain BaseMessage-like
            cls = m.__class__.__name__.lower()
            if "system" in cls:
                role = "system"
            elif "human" in cls or "user" in cls:
                role = "user"
            elif "ai" in cls or "assistant" in cls:
                role = "assistant"
            else:
                role = "user"
            return {"role": role, "content": str(getattr(m, "content", ""))}
        if isinstance(m, str):
            return {"role": "user", "content": m}
        return {"role": "user", "content": str(m)}

    if isinstance(messages, (str, dict)) or hasattr(messages, "content"):
        return [convert_one(messages)]
    try:
        return [convert_one(m) for m in messages]  # type: ignore[arg-type]
    except TypeError:
        return [convert_one(messages)]  # type: ignore[arg-type]


async def invoke_llm_with_retry(
    model_name: str,
    messages: MessagesLike,
    logger: Optional[Any] = None,
    temperature: float = 0.2,
    max_tokens: int = 8092,
    retries: int = 2,
    stream: bool = False,
    telemetry_callback: TelemetryCallback | None = None,
) -> AsyncIterator[str]:
    """
    Unified LLM invoker via llm_router.

    This matches the current `rag_core.invoke_llm_with_retry` contract:
    it is an async generator yielding either streaming chunks (stream=True)
    or a single full response (stream=False).
    """
    msg_dicts = _to_dict_messages(messages)

    # simple token estimation
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = sum(
            len(encoding.encode(m.get("content", ""))) for m in msg_dicts
        )
    except Exception:
        prompt_tokens = 0

    identity = _request_identity(
        model_name=model_name,
        messages=msg_dicts,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    request_key = _cache_key(identity)
    provider = identity.get("provider")

    cache_enabled = _env_bool("ENABLE_LLM_RAW_CACHE", False) and not stream
    singleflight_enabled = _env_bool("ENABLE_LLM_SINGLEFLIGHT", False) and not stream
    cache_event = "cache_disabled"
    if cache_enabled:
        cached = _read_cached_raw_response(request_key)
        if cached is not None:
            _emit(
                telemetry_callback,
                {
                    "stage_name": "llm_transport",
                    "model": model_name,
                    "provider": provider,
                    "prompt_chars": sum(len(m.get("content", "")) for m in msg_dicts),
                    "response_chars": len(cached),
                    "estimated_prompt_tokens": prompt_tokens,
                    "max_tokens": max_tokens,
                    "rate_limiter_wait_s": 0.0,
                    "http_elapsed_s": 0.0,
                    "retry_attempt_index": 0,
                    "retry_count": retries,
                    "retry_sleep_s": 0.0,
                    "json_attempt_index": None,
                    "json_parse_s": None,
                    "json_valid_dict": None,
                    "cache_hit": True,
                    "cache_event": "cache_hit",
                    "singleflight_joined": False,
                    "singleflight_owner": False,
                    "exception_type": None,
                    "timeout": False,
                },
            )
            yield cached
            return
        cache_event = "cache_miss"

    wait_started_at = time.perf_counter()
    await rate_limiter.wait(prompt_tokens)
    rate_limiter_wait_s = time.perf_counter() - wait_started_at

    delay = 1.5
    for attempt in range(retries + 1):
        retry_sleep_s = 0.0
        try:
            if stream:
                http_started_at = time.perf_counter()
                for chunk in call_llm_stream(
                    messages=msg_dicts,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk
                _emit(
                    telemetry_callback,
                    {
                        "stage_name": "llm_transport",
                        "model": model_name,
                        "provider": provider,
                        "prompt_chars": sum(len(m.get("content", "")) for m in msg_dicts),
                        "response_chars": None,
                        "estimated_prompt_tokens": prompt_tokens,
                        "max_tokens": max_tokens,
                        "rate_limiter_wait_s": rate_limiter_wait_s,
                        "http_elapsed_s": time.perf_counter() - http_started_at,
                        "retry_attempt_index": attempt,
                        "retry_count": retries,
                        "retry_sleep_s": 0.0,
                        "json_attempt_index": None,
                        "json_parse_s": None,
                        "json_valid_dict": None,
                        "cache_hit": False,
                        "cache_event": "cache_disabled",
                        "singleflight_joined": False,
                        "singleflight_owner": True,
                        "exception_type": None,
                        "timeout": False,
                    },
                )
                break

            singleflight_joined = False
            singleflight_owner = True
            http_started_at = time.perf_counter()
            if singleflight_enabled:
                global _singleflight_lock
                if _singleflight_lock is None:
                    _singleflight_lock = asyncio.Lock()
                async with _singleflight_lock:
                    task = _singleflight_tasks.get(request_key)
                    if task is None:
                        task = asyncio.create_task(
                            asyncio.to_thread(
                                call_llm,
                                messages=msg_dicts,
                                model=model_name,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                        )
                        _singleflight_tasks[request_key] = task
                    else:
                        singleflight_joined = True
                        singleflight_owner = False
                try:
                    result = await task
                finally:
                    if singleflight_owner:
                        async with _singleflight_lock:
                            if _singleflight_tasks.get(request_key) is task:
                                _singleflight_tasks.pop(request_key, None)
            else:
                result = await asyncio.to_thread(
                    call_llm,
                    messages=msg_dicts,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            raw_result = result.strip() if isinstance(result, str) else str(result)
            http_elapsed_s = time.perf_counter() - http_started_at
            cache_write = False
            if cache_enabled and not singleflight_joined:
                _write_cached_raw_response(request_key, identity, raw_result)
                cache_write = True
            _emit(
                telemetry_callback,
                {
                    "stage_name": "llm_transport",
                    "model": model_name,
                    "provider": provider,
                    "prompt_chars": sum(len(m.get("content", "")) for m in msg_dicts),
                    "response_chars": len(raw_result),
                    "estimated_prompt_tokens": prompt_tokens,
                    "max_tokens": max_tokens,
                    "rate_limiter_wait_s": rate_limiter_wait_s,
                    "http_elapsed_s": http_elapsed_s,
                    "retry_attempt_index": attempt,
                    "retry_count": retries,
                    "retry_sleep_s": retry_sleep_s,
                    "json_attempt_index": None,
                    "json_parse_s": None,
                    "json_valid_dict": None,
                    "cache_hit": False,
                    "cache_event": "cache_write" if cache_write else cache_event,
                    "singleflight_joined": singleflight_joined,
                    "singleflight_owner": singleflight_owner,
                    "exception_type": None,
                    "timeout": False,
                },
            )
            yield raw_result
            break
        except Exception as e:
            http_elapsed_s = time.perf_counter() - http_started_at if "http_started_at" in locals() else 0.0
            if logger:
                logger.warning(
                    f"[LLM Retry] Call failed (attempt {attempt + 1}/{retries + 1}): {e}"
                )
            event = {
                "stage_name": "llm_transport",
                "model": model_name,
                "provider": provider,
                "prompt_chars": sum(len(m.get("content", "")) for m in msg_dicts),
                "response_chars": 0,
                "estimated_prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "rate_limiter_wait_s": rate_limiter_wait_s,
                "http_elapsed_s": http_elapsed_s,
                "retry_attempt_index": attempt,
                "retry_count": retries,
                "retry_sleep_s": 0.0,
                "json_attempt_index": None,
                "json_parse_s": None,
                "json_valid_dict": None,
                "cache_hit": False,
                "cache_event": cache_event,
                "singleflight_joined": False,
                "singleflight_owner": True,
                "exception_type": e.__class__.__name__,
                "timeout": _timeout_indicator(e),
            }
            if attempt < retries:
                retry_sleep_s = delay
                event["retry_sleep_s"] = retry_sleep_s
                _emit(telemetry_callback, event)
                await asyncio.sleep(delay)
                delay *= 1.6
            else:
                _emit(telemetry_callback, event)
                raise


async def llm_complete(
    model_name: str,
    messages: MessagesLike,
    logger: Optional[Any] = None,
    temperature: float = 0.2,
    max_tokens: int = 8092,
    retries: int = 2,
    telemetry_callback: TelemetryCallback | None = None,
) -> str:
    """Convenience wrapper to collect a full completion as a string."""
    parts: List[str] = []
    async for chunk in invoke_llm_with_retry(
        model_name=model_name,
        messages=messages,
        logger=logger,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        stream=False,
        telemetry_callback=telemetry_callback,
    ):
        parts.append(chunk)
    return "".join(parts)


async def llm_stream(
    model_name: str,
    messages: MessagesLike,
    logger: Optional[Any] = None,
    temperature: float = 0.2,
    max_tokens: int = 8092,
    retries: int = 2,
) -> AsyncIterator[str]:
    """Convenience wrapper to yield only streaming chunks."""
    async for chunk in invoke_llm_with_retry(
        model_name=model_name,
        messages=messages,
        logger=logger,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
        stream=True,
    ):
        yield chunk
