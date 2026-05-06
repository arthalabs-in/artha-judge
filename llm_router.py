# llm_router.py - Multi-provider LLM support
"""
Swap LLM providers by changing ACTIVE_PROVIDER below.
Each provider has its own call function with identical interface.

Usage in rag_core.py:
    from llm_router import call_llm, call_llm_stream

    response = call_llm(messages=[...], temperature=0.1)
"""

import os
import requests
import json
import threading
from typing import List, Dict, Any, Optional, Generator, Tuple
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

load_dotenv()

# Default models per provider
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-4o-mini",
    "glm": "glm-4.6",
    "deepseek": "deepseek-chat",
    "opencode": "deepseek-v4-flash",
    "kimi": "kimi-k2.5",
    "ollama": os.getenv("OLLAMA_TEXT_MODEL", "llama3.1:8b"),
}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _request_timeout() -> Tuple[float, float]:
    """Return (connect_timeout_s, read_timeout_s) for requests calls."""
    connect_timeout = _float_env("LLM_ROUTER_CONNECT_TIMEOUT_S", 5.0)
    read_timeout = _float_env("LLM_ROUTER_READ_TIMEOUT_S", 120.0)
    return (connect_timeout, read_timeout)


_thread_local = threading.local()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _http_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def _post(*args, **kwargs):
    if _bool_env("ENABLE_LLM_KEEPALIVE_SESSION", True):
        return _http_session().post(*args, **kwargs)
    return requests.post(*args, **kwargs)


def _infer_provider_from_model(model: Optional[str]) -> Optional[str]:
    """Best-effort provider inference when caller passes a provider-specific model name."""
    if not model:
        return None

    m = model.lower().strip()

    # Keep Groq-hosted OpenAI OSS models on Groq.
    if m.startswith("openai/"):
        return "groq"

    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("glm"):
        return "glm"
    if m == "deepseek-v4-flash" or m.startswith("opencode/"):
        return "opencode"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("kimi"):
        return "kimi"
    if m.startswith(("llama", "mistral", "qwen", "gemma", "phi", "minicpm", "openbmb/")) or ":" in m:
        return "ollama"

    return None


def _resolve_provider(
    model: Optional[str] = None, provider: Optional[str] = None
) -> str:
    preferred = provider or _infer_provider_from_model(model) or ACTIVE_PROVIDER
    selected = preferred.lower().strip()
    if selected not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider: {selected}. Options: {list(PROVIDER_MAP.keys())}"
        )
    return selected


# ============================================================
# CHANGE THIS TO SWAP PROVIDERS
# Options: "groq", "gemini", "openai", "glm", "deepseek", "opencode", "kimi", "ollama"
# ============================================================
ACTIVE_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "groq").lower().strip()
if ACTIVE_PROVIDER not in DEFAULT_MODELS:
    ACTIVE_PROVIDER = "groq"


# ============================================================
# GROQ
# ============================================================
def call_groq(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """Groq API call"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    response = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["groq"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Groq API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def call_groq_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """Groq streaming API call"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    response = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["groq"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Groq API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except:
                    pass


# ============================================================
# GEMINI (Google)
# ============================================================
def call_gemini(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """Google Gemini API call"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    model_name = model or DEFAULT_MODELS["gemini"]

    # Convert OpenAI message format to Gemini format
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    response = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API error {response.status_code}: {response.text}")

    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]


def call_gemini_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """Google Gemini streaming API call"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    model_name = model or DEFAULT_MODELS["gemini"]

    # Convert message format
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    response = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={api_key}&alt=sse",
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Gemini API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "candidates" in data:
                        text = data["candidates"][0]["content"]["parts"][0].get(
                            "text", ""
                        )
                        if text:
                            yield text
                except:
                    pass


# ============================================================
# OPENAI
# ============================================================
def call_openai(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """OpenAI API call"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    response = _post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["openai"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"OpenAI API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def call_openai_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """OpenAI streaming API call"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    response = _post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["openai"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"OpenAI API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except:
                    pass


# ============================================================
# GLM (Zhipu AI)
# ============================================================
def call_glm(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """GLM (Zhipu AI) API call"""
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise ValueError("GLM_API_KEY not set")

    response = _post(
        "https://api.z.ai/api/paas/v4/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["glm"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"GLM API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def call_glm_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """GLM streaming API call"""
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise ValueError("GLM_API_KEY not set")

    response = _post(
        "https://api.z.ai/api/paas/v4/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["glm"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"GLM API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except:
                    pass


# ============================================================
# DEEPSEEK
# ============================================================
def call_deepseek(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """DeepSeek API call (OpenAI-compatible)"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    response = _post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["deepseek"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"DeepSeek API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def call_deepseek_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """DeepSeek streaming API call"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    response = _post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["deepseek"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"DeepSeek API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except:
                    pass


# ============================================================
# OPENCODE / OPENAI-COMPATIBLE
# ============================================================
def _opencode_api_key() -> str:
    api_key = (
        os.getenv("OPENCODE_API_KEY")
        or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("OPENCODE_API_KEY or OPENAI_COMPATIBLE_API_KEY not set")
    return api_key


def _opencode_base_url() -> str:
    return os.getenv(
        "OPENCODE_CHAT_COMPLETIONS_URL",
        "https://opencode.ai/zen/go/v1/chat/completions",
    )


def call_opencode(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """OpenCode OpenAI-compatible chat completions call."""
    response = _post(
        _opencode_base_url(),
        headers={
            "Authorization": f"Bearer {_opencode_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["opencode"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"OpenCode API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def call_opencode_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """OpenCode OpenAI-compatible streaming chat completions call."""
    response = _post(
        _opencode_base_url(),
        headers={
            "Authorization": f"Bearer {_opencode_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["opencode"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"OpenCode API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except:
                    pass


# ============================================================
# KIMI (Moonshot AI)
# ============================================================
def call_kimi(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.6,  # Kimi recommends 0.6
    max_tokens: int = 8192,
) -> str:
    """Kimi (Moonshot AI) API call - OpenAI-compatible"""
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError("MOONSHOT_API_KEY not set")

    response = _post(
        "https://api.moonshot.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["kimi"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Kimi API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def call_kimi_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.6,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    """Kimi streaming API call"""
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError("MOONSHOT_API_KEY not set")

    response = _post(
        "https://api.moonshot.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or DEFAULT_MODELS["kimi"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Kimi API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except:
                    pass


# ============================================================
# OLLAMA / LOCAL CHAT
# ============================================================
def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def call_ollama(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> str:
    """Local Ollama chat call. Keeps judgment text on the machine."""
    response = _post(
        f"{_ollama_base_url()}/api/chat",
        json={
            "model": model or DEFAULT_MODELS["ollama"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Ollama API error {response.status_code}: {response.text}")

    payload = response.json()
    return str((payload.get("message") or {}).get("content") or payload.get("response") or "")


def call_ollama_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Generator[str, None, None]:
    response = _post(
        f"{_ollama_base_url()}/api/chat",
        json={
            "model": model or DEFAULT_MODELS["ollama"],
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        stream=True,
        timeout=_request_timeout(),
    )

    if response.status_code != 200:
        raise Exception(f"Ollama API error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        content = (chunk.get("message") or {}).get("content")
        if content:
            yield content


# ============================================================
# ROUTER - Uses ACTIVE_PROVIDER
# ============================================================
PROVIDER_MAP = {
    "groq": (call_groq, call_groq_stream),
    "gemini": (call_gemini, call_gemini_stream),
    "openai": (call_openai, call_openai_stream),
    "glm": (call_glm, call_glm_stream),
    "deepseek": (call_deepseek, call_deepseek_stream),
    "opencode": (call_opencode, call_opencode_stream),
    "kimi": (call_kimi, call_kimi_stream),
    "ollama": (call_ollama, call_ollama_stream),
}


def call_llm(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    provider: str = None,  # Override ACTIVE_PROVIDER
) -> str:
    """
    Unified LLM call - uses ACTIVE_PROVIDER or override.

    To swap providers, either:
    1. Change ACTIVE_PROVIDER at top of file
    2. Pass provider="gemini" etc. to this function
    """
    p = _resolve_provider(model=model, provider=provider)

    call_fn, _ = PROVIDER_MAP[p]
    return call_fn(messages, model, temperature, max_tokens)


def call_llm_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    provider: str = None,
) -> Generator[str, None, None]:
    """Unified streaming LLM call"""
    p = _resolve_provider(model=model, provider=provider)

    _, stream_fn = PROVIDER_MAP[p]
    return stream_fn(messages, model, temperature, max_tokens)


def get_active_provider(model: str = None, provider: str = None) -> str:
    """Expose effective provider resolution for request-time policy checks."""
    return _resolve_provider(model=model, provider=provider)


def get_request_identity(model: str = None, provider: str = None) -> Dict[str, Any]:
    """Expose stable request identity fields for instrumentation/cache keys."""
    selected = _resolve_provider(model=model, provider=provider)
    base_urls = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "gemini": "https://generativelanguage.googleapis.com",
        "openai": "https://api.openai.com/v1/chat/completions",
        "glm": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "deepseek": "https://api.deepseek.com/chat/completions",
        "opencode": _opencode_base_url(),
        "kimi": "https://api.moonshot.cn/v1/chat/completions",
        "ollama": f"{_ollama_base_url()}/api/chat",
    }
    return {
        "provider": selected,
        "base_url": base_urls.get(selected),
        "model": model or DEFAULT_MODELS.get(selected),
    }


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print(f"Testing {ACTIVE_PROVIDER}...")
    response = call_llm(
        messages=[
            {"role": "user", "content": "Say 'Hello from LLM router!' in 5 words"}
        ],
        temperature=0.1,
    )
    print(f"Response: {response}")
