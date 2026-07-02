"""
Thin wrapper around Gemini Flash. Kept separate so swapping models later
touches one file, not the whole codebase.
"""
from __future__ import annotations
import json
import os

import google.generativeai as genai

MODEL_NAME = "gemini-flash-latest"


def _configure() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")
    genai.configure(api_key=api_key)


def generate(prompt: str, expect_json: bool = False) -> str:
    _configure()
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    text = response.text.strip()
    if expect_json:
        text = _strip_code_fences(text)
    return text


def generate_json(prompt: str) -> dict:
    raw = generate(prompt, expect_json=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {raw[:300]}") from e


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
