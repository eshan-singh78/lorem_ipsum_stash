"""
LLM Adapter — InvestorDNA
Single entry point for all LLM calls. Supports:
  - ollama        (local, default)
  - ollama_cloud  (Ollama Cloud API)
  - openrouter    (OpenRouter — access to 200+ models)

Configuration (in priority order):
  1. Environment variables  (recommended for production)
  2. llm_config.json        (optional file-based config)
  3. Built-in defaults      (ollama local)

Environment variables:
  LLM_PROVIDER        = ollama | ollama_cloud | openrouter
  LLM_MODEL           = model name (provider-specific)
  OLLAMA_BASE_URL     = http://localhost:11434  (ollama only)
  OLLAMA_CLOUD_URL    = https://...             (ollama_cloud only)
  OLLAMA_CLOUD_KEY    = <api key>               (ollama_cloud only)
  OPENROUTER_API_KEY  = <api key>               (openrouter only)
  OPENROUTER_MODEL    = meta-llama/llama-3.1-8b-instruct:free  (openrouter default)

Usage:
  from llm_adapter import llm_call, configure

  # Use defaults (reads from env / config file)
  response = llm_call(prompt, num_predict=512)

  # Override at runtime
  configure(provider="openrouter", model="mistralai/mistral-7b-instruct")
  response = llm_call(prompt)
"""

from __future__ import annotations

import json
import os
import re
import requests
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

PROVIDER_OLLAMA        = "ollama"
PROVIDER_OLLAMA_CLOUD  = "ollama_cloud"
PROVIDER_OPENROUTER    = "openrouter"

_VALID_PROVIDERS = {PROVIDER_OLLAMA, PROVIDER_OLLAMA_CLOUD, PROVIDER_OPENROUTER}

# Default model names per provider
_DEFAULT_MODELS = {
    PROVIDER_OLLAMA:       "llama3.1:8b",
    PROVIDER_OLLAMA_CLOUD: "llama3.1:8b",
    PROVIDER_OPENROUTER:   "meta-llama/llama-3.1-8b-instruct:free",
}

# Default base URLs
_DEFAULT_URLS = {
    PROVIDER_OLLAMA:       "http://localhost:11434",
    PROVIDER_OLLAMA_CLOUD: "https://ollama.com",   # direct cloud API host
    PROVIDER_OPENROUTER:   "https://openrouter.ai/api/v1",
}


# ---------------------------------------------------------------------------
# Config file loader (defined before _Config so it's available at class init)
# ---------------------------------------------------------------------------

def _load_config_file() -> dict:
    """
    Load LLM config with a two-file merge strategy:

    1. investor_profiler/llm_config.json       — base config (provider, api_key, base_url, timeout)
    2. investor_profiler/v2/llm_config.json    — override layer (model switching by benchmark)

    The v2 file wins for any key it defines. This means:
      - benchmark.py writes only the "model" field to v2/llm_config.json
      - all other settings (api_key, provider, timeout) come from the root file
      - main.py always picks up the current model without any extra wiring
    """
    base_dir = Path(__file__).parent
    cfg: dict = {}

    # Layer 1: root config
    root_cfg = base_dir / "llm_config.json"
    if root_cfg.exists():
        try:
            with open(root_cfg) as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass

    # Layer 2: v2 override (benchmark writes model here)
    v2_cfg = base_dir / "v2" / "llm_config.json"
    if v2_cfg.exists():
        try:
            with open(v2_cfg) as f:
                overrides = json.load(f)
            # Only apply non-null, non-empty values from v2 config
            cfg.update({k: v for k, v in overrides.items() if v not in (None, "", 0)})
        except (json.JSONDecodeError, OSError):
            pass

    return cfg


# ---------------------------------------------------------------------------
# Runtime config — mutable, set once at startup
# ---------------------------------------------------------------------------

class _Config:
    provider:  str
    model:     str
    base_url:  str
    api_key:   str | None
    timeout:   int | None   # seconds; None = no timeout

    def __init__(self):
        self._load()

    def _load(self):
        """Load config from env vars, then llm_config.json, then defaults."""
        file_cfg = _load_config_file()

        self.provider = (
            os.environ.get("LLM_PROVIDER")
            or file_cfg.get("provider")
            or PROVIDER_OLLAMA
        ).lower()

        if self.provider not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                f"Valid: {sorted(_VALID_PROVIDERS)}"
            )

        default_model = _DEFAULT_MODELS[self.provider]
        default_url   = _DEFAULT_URLS[self.provider]

        if self.provider == PROVIDER_OLLAMA:
            self.model   = os.environ.get("LLM_MODEL") or file_cfg.get("model") or default_model
            self.base_url = os.environ.get("OLLAMA_BASE_URL") or file_cfg.get("base_url") or default_url
            self.api_key  = None

        elif self.provider == PROVIDER_OLLAMA_CLOUD:
            self.model   = os.environ.get("LLM_MODEL") or file_cfg.get("model") or default_model
            self.base_url = os.environ.get("OLLAMA_CLOUD_URL") or file_cfg.get("base_url") or default_url
            self.api_key  = os.environ.get("OLLAMA_CLOUD_KEY") or file_cfg.get("api_key")

        elif self.provider == PROVIDER_OPENROUTER:
            self.model   = (
                os.environ.get("LLM_MODEL")
                or os.environ.get("OPENROUTER_MODEL")
                or file_cfg.get("model")
                or default_model
            )
            self.base_url = default_url
            self.api_key  = os.environ.get("OPENROUTER_API_KEY") or file_cfg.get("api_key")
            if not self.api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required for the openrouter provider. "
                    "Set it as an environment variable or in llm_config.json."
                )

        self.timeout = int(os.environ.get("LLM_TIMEOUT", 0) or file_cfg.get("timeout", 0)) or None
        # Enforce a minimum 120s timeout — never allow infinite hang in production
        if self.timeout is None:
            self.timeout = 120

    def override(self, **kwargs):
        """Apply runtime overrides. When provider changes, reset base_url to its default."""
        new_provider = kwargs.get("provider")
        if new_provider and new_provider != self.provider:
            # Reset base_url to the new provider's default before applying overrides
            self.base_url = _DEFAULT_URLS.get(new_provider, self.base_url)
            self.api_key  = None  # clear key when switching providers
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)


_cfg = _Config()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure(
    provider:  str | None = None,
    model:     str | None = None,
    base_url:  str | None = None,
    api_key:   str | None = None,
    timeout:   int | None = None,
) -> None:
    """
    Override LLM config at runtime.

    Args:
        provider:  "ollama" | "ollama_cloud" | "openrouter"
        model:     Model name (provider-specific)
        base_url:  Override base URL
        api_key:   API key (required for ollama_cloud and openrouter)
        timeout:   Request timeout in seconds (None = no timeout)

    Example:
        configure(provider="openrouter", model="mistralai/mistral-7b-instruct")
    """
    if provider and provider not in _VALID_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Valid: {sorted(_VALID_PROVIDERS)}")
    _cfg.override(provider=provider, model=model, base_url=base_url,
                  api_key=api_key, timeout=timeout)


def get_config() -> dict:
    """Return current active config (api_key masked)."""
    return {
        "provider":  _cfg.provider,
        "model":     _cfg.model,
        "base_url":  _cfg.base_url,
        "api_key":   ("***" if _cfg.api_key else None),
        "timeout":   _cfg.timeout,
    }


def llm_call(prompt: str, num_predict: int = 1024) -> dict:
    """
    Make a single LLM call and return the parsed JSON response.

    Args:
        prompt:      The prompt string.
        num_predict: Max tokens to generate.

    Returns:
        Parsed dict from the model response.

    Raises:
        requests.RequestException: On network/HTTP failure.
        ValueError:                On JSON parse failure.
    """
    provider = _cfg.provider

    if provider == PROVIDER_OLLAMA:
        return _call_ollama(prompt, num_predict)
    elif provider == PROVIDER_OLLAMA_CLOUD:
        return _call_ollama_cloud(prompt, num_predict)
    elif provider == PROVIDER_OPENROUTER:
        return _call_openrouter(prompt, num_predict)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """Extract and parse JSON from model response text. Attempts repair on truncation."""
    text = text.strip()
    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)

    # Try clean parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt repair: close any unclosed braces/brackets
    repaired = _repair_json(text)
    return json.loads(repaired)


def _repair_json(text: str) -> str:
    """
    Best-effort repair of truncated JSON.
    Closes unclosed strings, arrays, and objects so json.loads can parse partial output.
    """
    # Close any unclosed string (odd number of unescaped quotes after last key)
    # Simple heuristic: strip trailing incomplete key-value pair
    text = text.rstrip()

    # Remove trailing comma before attempting to close
    text = re.sub(r",\s*$", "", text)

    # Count open braces and brackets
    depth_brace   = text.count("{") - text.count("}")
    depth_bracket = text.count("[") - text.count("]")

    # Close open arrays first, then objects
    text += "]" * max(0, depth_bracket)
    text += "}" * max(0, depth_brace)

    return text


def _call_ollama(prompt: str, num_predict: int) -> dict:
    """Ollama local — /api/generate endpoint."""
    payload = {
        "model":   _cfg.model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": num_predict},
        "format":  "json",
    }
    resp = requests.post(
        f"{_cfg.base_url}/api/generate",
        json=payload,
        timeout=_cfg.timeout,
    )
    resp.raise_for_status()
    return _parse_json(resp.json().get("response", ""))


def _call_ollama_cloud(prompt: str, num_predict: int) -> dict:
    """
    Ollama Cloud — native /api/chat endpoint on ollama.com.
    Uses the same wire format as local Ollama, with Bearer auth.
    Docs: https://docs.ollama.com/cloud
    """
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {_cfg.api_key}",
    }
    payload = {
        "model":  _cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
        "format": "json",
    }
    resp = requests.post(
        f"{_cfg.base_url}/api/chat",
        headers=headers,
        json=payload,
        timeout=_cfg.timeout,
    )
    resp.raise_for_status()
    # Native Ollama /api/chat response: {"message": {"role": "assistant", "content": "..."}}
    content = resp.json()["message"]["content"]
    return _parse_json(content)


def _call_openrouter(prompt: str, num_predict: int) -> dict:
    """
    OpenRouter — OpenAI-compatible /chat/completions endpoint.
    Supports 200+ models from multiple providers.
    """
    headers = {
        "Authorization":  f"Bearer {_cfg.api_key}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "https://github.com/investordna",   # optional but good practice
        "X-Title":        "InvestorDNA",
    }
    payload = {
        "model": _cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": num_predict,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        f"{_cfg.base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=_cfg.timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_json(content)
