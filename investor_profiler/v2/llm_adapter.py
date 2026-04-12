"""
LLM Adapter — InvestorDNA v2
Single entry point for all LLM calls.

Supports: ollama | ollama_cloud | openrouter
Config priority: env vars > llm_config.json > defaults

SECURITY: Never put API keys in llm_config.json. Use env vars.
"""
from __future__ import annotations

import json
import os
import re
import requests
from pathlib import Path

PROVIDER_OLLAMA       = "ollama"
PROVIDER_OLLAMA_CLOUD = "ollama_cloud"
PROVIDER_OPENROUTER   = "openrouter"
_VALID_PROVIDERS      = {PROVIDER_OLLAMA, PROVIDER_OLLAMA_CLOUD, PROVIDER_OPENROUTER}

_DEFAULT_MODELS = {
    PROVIDER_OLLAMA:       "llama3.1:8b",
    PROVIDER_OLLAMA_CLOUD: "llama3.1:8b",
    PROVIDER_OPENROUTER:   "meta-llama/llama-3.1-8b-instruct:free",
}
_DEFAULT_URLS = {
    PROVIDER_OLLAMA:       "http://localhost:11434",
    PROVIDER_OLLAMA_CLOUD: "https://ollama.com",
    PROVIDER_OPENROUTER:   "https://openrouter.ai/api/v1",
}

DEFAULT_TIMEOUT = 120  # seconds — never 0 in production


def _load_config_file() -> dict:
    cfg = Path(__file__).parent / "llm_config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())
        except Exception:
            pass
    return {}


class _Config:
    def __init__(self):
        self._load()

    def _load(self):
        f = _load_config_file()
        self.provider = (os.environ.get("LLM_PROVIDER") or f.get("provider") or PROVIDER_OLLAMA).lower()
        if self.provider not in _VALID_PROVIDERS:
            raise ValueError(f"Unknown LLM_PROVIDER '{self.provider}'")

        self.model    = os.environ.get("LLM_MODEL") or f.get("model") or _DEFAULT_MODELS[self.provider]
        self.base_url = _DEFAULT_URLS[self.provider]

        if self.provider == PROVIDER_OLLAMA:
            self.base_url = os.environ.get("OLLAMA_BASE_URL") or f.get("base_url") or self.base_url
            self.api_key  = None
        elif self.provider == PROVIDER_OLLAMA_CLOUD:
            self.base_url = os.environ.get("OLLAMA_CLOUD_URL") or f.get("base_url") or self.base_url
            self.api_key  = os.environ.get("OLLAMA_CLOUD_KEY") or f.get("api_key")
        elif self.provider == PROVIDER_OPENROUTER:
            self.api_key  = os.environ.get("OPENROUTER_API_KEY") or f.get("api_key")
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY env var is required for openrouter provider.")

        raw_timeout = int(os.environ.get("LLM_TIMEOUT", 0) or f.get("timeout", 0))
        self.timeout = raw_timeout if raw_timeout > 0 else DEFAULT_TIMEOUT

    def override(self, **kw):
        if "provider" in kw and kw["provider"] and kw["provider"] != self.provider:
            self.base_url = _DEFAULT_URLS.get(kw["provider"], self.base_url)
            self.api_key  = None
        for k, v in kw.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)


_cfg = _Config()


def configure(provider=None, model=None, base_url=None, api_key=None, timeout=None):
    if provider and provider not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'")
    _cfg.override(provider=provider, model=model, base_url=base_url, api_key=api_key, timeout=timeout)


def get_config() -> dict:
    return {
        "provider": _cfg.provider, "model": _cfg.model,
        "base_url": _cfg.base_url, "api_key": "***" if _cfg.api_key else None,
        "timeout":  _cfg.timeout,
    }


# ---------------------------------------------------------------------------
# JSON parsing + repair
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """Close unclosed braces/brackets. Also strips trailing incomplete key-value pairs."""
    text = text.rstrip()
    # Remove trailing comma
    text = re.sub(r",\s*$", "", text)
    # Remove trailing incomplete key (e.g. "key": )
    text = re.sub(r',\s*"[^"]*"\s*:\s*$', "", text)
    # Close open arrays then objects
    text += "]" * max(0, text.count("[") - text.count("]"))
    text += "}" * max(0, text.count("{") - text.count("}"))
    return text


def parse_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_repair_json(text))


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, num_predict: int) -> dict:
    resp = requests.post(
        f"{_cfg.base_url}/api/generate",
        json={"model": _cfg.model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0, "num_predict": num_predict}, "format": "json"},
        timeout=_cfg.timeout,
    )
    resp.raise_for_status()
    return parse_json(resp.json().get("response", ""))


def _call_ollama_cloud(prompt: str, num_predict: int) -> dict:
    resp = requests.post(
        f"{_cfg.base_url}/api/chat",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_cfg.api_key}"},
        json={"model": _cfg.model, "messages": [{"role": "user", "content": prompt}],
              "stream": False, "options": {"temperature": 0, "num_predict": num_predict},
              "format": "json"},
        timeout=_cfg.timeout,
    )
    resp.raise_for_status()
    return parse_json(resp.json()["message"]["content"])


def _call_openrouter(prompt: str, num_predict: int) -> dict:
    resp = requests.post(
        f"{_cfg.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {_cfg.api_key}", "Content-Type": "application/json",
                 "X-Title": "InvestorDNA"},
        json={"model": _cfg.model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": num_predict,
              "response_format": {"type": "json_object"}},
        timeout=_cfg.timeout,
    )
    resp.raise_for_status()
    return parse_json(resp.json()["choices"][0]["message"]["content"])


def llm_call(prompt: str, num_predict: int = 512) -> dict:
    """Single LLM call → parsed dict. Raises on failure."""
    if _cfg.provider == PROVIDER_OLLAMA:
        return _call_ollama(prompt, num_predict)
    elif _cfg.provider == PROVIDER_OLLAMA_CLOUD:
        return _call_ollama_cloud(prompt, num_predict)
    elif _cfg.provider == PROVIDER_OPENROUTER:
        return _call_openrouter(prompt, num_predict)
    raise ValueError(f"Unknown provider: {_cfg.provider}")
