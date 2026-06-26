import json
import logging
from typing import Any

import requests

from campusmind.config import get_settings

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096},
        }
        if system:
            payload["system"] = system
        try:
            response = requests.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
        except requests.RequestException as exc:
            logger.exception("Ollama request failed")
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama and pull the configured model."
            ) from exc

    def generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
    ):
        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": 3072,
                "num_predict": 260,
            },
        }
        if system:
            payload["system"] = system
        try:
            with requests.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=(5, 180),
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    token = str(data.get("response", ""))
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            logger.exception("Ollama streaming request failed")
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama and pull the configured model."
            ) from exc
