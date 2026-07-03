"""
Thin wrapper around Gemini Flash. Kept separate so swapping models later
touches one file, not the whole codebase.

Includes retry-with-backoff on 429/quota errors, since the free tier caps
gemini-flash at 5 requests/minute and a single Mnemosyne pass can easily
use 4-6 calls (planner + N steps + reflection).
"""
from __future__ import annotations
import json
import os
import time

import google.generativeai as genai

MODEL_NAME = "gemini-2.5-flash-lite"

MAX_QUOTA_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 5


def _configure() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")
    genai.configure(api_key=api_key)


def generate(prompt: str, expect_json: bool = False) -> str:
    _configure()
    model = genai.GenerativeModel(MODEL_NAME)

    delay = INITIAL_BACKOFF_SECONDS
    last_error: Exception | None = None

    for attempt in range(MAX_QUOTA_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if expect_json:
                text = _strip_code_fences(text)
            return text
        except Exception as e:  # noqa: BLE001
            last_error = e
            if _is_quota_error(e) and attempt < MAX_QUOTA_RETRIES:
                time.sleep(delay)
                delay *= 2
                continue
            raise

    raise last_error  # type: ignore[misc]


def generate_json(prompt: str) -> dict:
    raw = generate(prompt, expect_json=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {raw[:300]}") from e


def _is_quota_error(e: Exception) -> bool:
    text = str(e).lower()
    return "429" in text or "quota" in text or "rate limit" in text


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
