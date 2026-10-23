import os
from time import perf_counter
import requests
from ..config import (
    OPENROUTER_API_KEY,
    OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_MODEL,
)


class OpenRouterClient:

    def __init__(self):

        self.api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
        self.model = OPENROUTER_MODEL
        self.fallback_models = _parse_fallback_models(OPENROUTER_FALLBACK_MODELS)

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in .env"
            )

        if not self.model:
            raise ValueError(
                "OPENROUTER_MODEL is not set in config.py"
            )

        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.call_log = []

    def ask(self, prompt, purpose="unknown"):
        from ..telemetry import log

        errors = []
        models = self._model_sequence()
        for attempt, model in enumerate(models):
            started = perf_counter()
            try:
                response = self._post(prompt, model)
            except requests.RequestException as exc:
                elapsed = perf_counter() - started
                log(
                    f"OpenRouter call purpose={purpose} model={model} "
                    f"error={type(exc).__name__} time={elapsed:.2f}s"
                )
                errors.append(f"{model}: {type(exc).__name__}: {exc}")
                continue
            elapsed = perf_counter() - started
            call = {
                "purpose": purpose,
                "model": model,
                "prompt_chars": len(prompt) if isinstance(prompt, str) else 0,
                "duration": elapsed,
                "provider_status": response.status_code,
                "retry_count": attempt,
            }
            self.call_log.append(call)
            log(
                f"OpenRouter call purpose={purpose} model={model} "
                f"status={response.status_code} prompt_chars={call['prompt_chars']} "
                f"time={elapsed:.2f}s"
            )
            if not response.ok:
                error_text = f"OpenRouter returned {response.status_code}: {response.text}"
                if _should_try_next_model(response):
                    errors.append(error_text)
                    continue
                raise RuntimeError(error_text)

            try:
                return self._parse_completion(response)
            except RuntimeError as exc:
                errors.append(str(exc))
                if model != models[-1]:
                    continue
                raise

        raise RuntimeError(
            "OpenRouter did not return a completion from any configured free "
            f"model. Errors: {' | '.join(errors)}"
        )

    def _post(self, prompt, model):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        return requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=60
        )

    def _parse_completion(self, response):
        if not response.ok:
            raise RuntimeError(
                f"OpenRouter returned {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"OpenRouter returned invalid JSON ({response.status_code}): "
                f"{response.text}"
            ) from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            provider_error = data.get("error") if isinstance(data, dict) else data
            raise RuntimeError(
                "OpenRouter response did not contain any choices. "
                f"Provider response: {provider_error!r}"
            )

        try:
            content = choices[0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"OpenRouter returned an invalid completion shape: {data!r}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                f"OpenRouter returned empty completion content: {data!r}"
            )
        return content

    def _model_sequence(self):
        models = [self.model, *self.fallback_models]
        sequence = []
        for model in models:
            if model and model not in sequence:
                sequence.append(model)
        return sequence


def _parse_fallback_models(value):
    if not value:
        return []
    return [
        model.strip()
        for model in value.split(",")
        if model.strip()
    ]


def _should_try_next_model(response):
    # Rate limits and provider-side failures are model specific; auth and
    # request errors would fail the same way on every fallback model.
    if response.status_code == 429 or response.status_code >= 500:
        return True
    return _is_free_model_unavailable(response)


def _is_free_model_unavailable(response):
    if response.status_code != 404:
        return False
    text = response.text.lower()
    return "unavailable for free" in text or "use this slug instead" in text
