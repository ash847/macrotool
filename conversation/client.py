"""
Thin Anthropic SDK wrapper.

Provides a streaming interface that yields text chunks while buffering the full
response for tag parsing by the flow layer.
"""

from __future__ import annotations

import time
from typing import Generator

import anthropic

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048  # Conversational responses; not long-form generation
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]  # seconds between attempts


class MacroToolClient:
    def __init__(self, api_key: str | None = None, model: str = MODEL):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = MAX_TOKENS,
    ) -> Generator[str, None, None]:
        """
        Stream a response, yielding text chunks.

        Retries on 529 (overloaded) and 5xx errors with backoff.
        Buffers the full response in self.last_response after exhaustion.
        """
        self.last_response = ""

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                with self._client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        self.last_response += text
                        yield text
                return  # success
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 529) or e.status_code >= 500:
                    last_exc = e
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_DELAYS[attempt])
                        self.last_response = ""  # reset buffer for retry
                        continue
                raise
            except anthropic.APIConnectionError as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAYS[attempt])
                    self.last_response = ""
                    continue
                raise

        if last_exc:
            raise last_exc
