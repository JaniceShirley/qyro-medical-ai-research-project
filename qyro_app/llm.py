"""Layer 6 provider wrapper. Provider-specific code lives only in this file.

Choose with QYRO_LLM_PROVIDER = groq | gemini | none (default: whichever API key is set).
Keys are read from the environment (GROQ_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY), never from code.
"""
import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from qyro_app.schemas import ExplanationDraft

# Provider model catalogs change; check `Groq().models.list()` and override with QYRO_LLM_MODEL if one is retired.
DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b", "gemini": "gemini-2.5-flash"}
TIMEOUT_SECONDS = float(os.environ.get("QYRO_LLM_TIMEOUT_SECONDS", "20"))
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2048


class LLMError(Exception):
    """Any provider failure (auth, rate limit, timeout, empty output). The caller falls back safely."""


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: float = 0.0


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str) -> LLMResult: ...


class GroqProvider:
    name = "groq"

    def __init__(self, model: Optional[str] = None):
        from groq import Groq

        self.model = model or DEFAULT_MODELS["groq"]
        self._client = Groq(timeout=TIMEOUT_SECONDS, max_retries=1)

    def complete_json(self, system: str, user: str) -> LLMResult:
        import groq

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            )
        except groq.AuthenticationError as exc:
            raise LLMError("groq: invalid or missing API key") from exc
        except groq.RateLimitError as exc:
            raise LLMError("groq: rate limited") from exc
        except groq.APITimeoutError as exc:
            raise LLMError("groq: timed out") from exc
        except groq.APIConnectionError as exc:
            raise LLMError("groq: network error") from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"groq: API error {exc.status_code}") from exc
        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise LLMError("groq: empty response")
        usage = response.usage
        return LLMResult(text=text, model=self.model,
                         input_tokens=getattr(usage, "prompt_tokens", None), output_tokens=getattr(usage, "completion_tokens", None),
                         latency_ms=round((time.perf_counter() - started) * 1000, 1))


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: Optional[str] = None):
        from google import genai
        from google.genai import types

        self.model = model or DEFAULT_MODELS["gemini"]
        self._types = types
        self._client = genai.Client(http_options=types.HttpOptions(timeout=int(TIMEOUT_SECONDS * 1000)))

    def complete_json(self, system: str, user: str) -> LLMResult:
        from google.genai import errors

        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=self._types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_json_schema=ExplanationDraft.model_json_schema(),
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
        except errors.ClientError as exc:
            raise LLMError(f"gemini: request rejected ({exc.code})") from exc
        except errors.ServerError as exc:
            raise LLMError(f"gemini: server error ({exc.code})") from exc
        except errors.APIError as exc:
            raise LLMError("gemini: API error") from exc
        except Exception as exc:  # timeouts and network errors surface as httpx exceptions
            raise LLMError(f"gemini: {type(exc).__name__}") from exc
        text = response.text
        if not text:
            raise LLMError("gemini: empty response")
        usage = response.usage_metadata
        return LLMResult(text=text, model=self.model,
                         input_tokens=getattr(usage, "prompt_token_count", None), output_tokens=getattr(usage, "candidates_token_count", None),
                         latency_ms=round((time.perf_counter() - started) * 1000, 1))


def configured_provider_name() -> str:
    choice = os.environ.get("QYRO_LLM_PROVIDER", "auto").lower()
    if choice != "auto":
        return choice
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "none"


def get_provider() -> Optional[LLMProvider]:
    """Returns None when the LLM is switched off or no key is configured (the app then uses the template)."""
    name = configured_provider_name()
    model = os.environ.get("QYRO_LLM_MODEL") or None
    if name == "groq":
        return GroqProvider(model)
    if name == "gemini":
        return GeminiProvider(model)
    return None
