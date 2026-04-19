from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_TIMEOUT_SECONDS = 30


def _read_streamlit_secret(name: str) -> str | None:
    project_secret_file = Path.cwd() / ".streamlit" / "secrets.toml"
    user_secret_file = Path.home() / ".streamlit" / "secrets.toml"
    if not project_secret_file.exists() and not user_secret_file.exists():
        return None

    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_ollama_url() -> str:
    return (
        os.getenv("OLLAMA_URL")
        or _read_streamlit_secret("OLLAMA_URL")
        or DEFAULT_OLLAMA_URL
    )


@lru_cache(maxsize=1)
def get_ollama_model() -> str:
    return (
        os.getenv("OLLAMA_MODEL")
        or _read_streamlit_secret("OLLAMA_MODEL")
        or DEFAULT_OLLAMA_MODEL
    )


def call_ollama(prompt: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str | None:
    """Call Ollama and return the `response` text when available.

    Returns None when the request fails or an unexpected payload is returned.
    """

    try:
        response = requests.post(
            get_ollama_url(),
            json={"model": get_ollama_model(), "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    content = payload.get("response") if isinstance(payload, dict) else None
    if content is None:
        return None

    return str(content).strip()
