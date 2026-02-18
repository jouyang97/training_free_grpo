from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    out = []
    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            if getattr(part, "type", "") == "output_text":
                value = getattr(part, "text", "")
                if value:
                    out.append(value)

    return "\n".join(out).strip()


@dataclass(slots=True)
class OpenAIResponsesBackend:
    """
    Frozen LLM backend using the OpenAI Responses API.

    Set OPENAI_API_KEY in your environment.
    """

    model: str = "gpt-5.2"
    api_key: str | None = None

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=key)

    def generate(self, prompt: str, context: str | None = None, temperature: float = 0.7) -> str:
        blocks: list[str] = []
        if context and context.strip():
            blocks.append(context.strip())
        blocks.append("Task:\n" + prompt.strip())
        blocks.append("Respond with concise reasoning and a final line starting with 'Answer:'.")
        full_input = "\n\n".join(blocks)

        response = self.client.responses.create(
            model=self.model,
            input=full_input,
            temperature=temperature,
        )
        text = _extract_response_text(response)
        if not text:
            return "Answer: (empty)"
        return text
