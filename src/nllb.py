"""Cliente REST e utilitários de tradução para a API NLLB."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import httpx


TARGET_LANGUAGE = "por_Latn"
MAX_TRANSLATION_TOKENS = 512
MAX_TRANSLATION_CHARACTERS = 4096
LANGUAGE_SAMPLE_CHARACTERS = 512


class NLLBApiError(RuntimeError):
    """Falha de transporte, HTTP ou contrato na API NLLB."""


class NLLBClient:
    """Cliente síncrono para as rotas REST v4 do serviço NLLB."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("NLLB_TIMEOUT_SECONDS deve ser maior que zero.")
        self.base_url = base_url.rstrip("/")
        self._http = http_client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._owns_http = http_client is None

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def _get_json(self, route: str, *, params: dict[str, str]) -> dict[str, Any]:
        try:
            response = self._http.get(route, params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise NLLBApiError(
                f"HTTP {exc.response.status_code} em {route}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise NLLBApiError(f"Falha de conexão em {route}: {exc}") from exc
        except ValueError as exc:
            raise NLLBApiError(f"Resposta JSON inválida em {route}.") from exc
        if not isinstance(payload, dict):
            raise NLLBApiError(f"Resposta inesperada em {route}: esperado objeto JSON.")
        return payload

    def _post_json(self, route: str, *, payload: dict[str, str]) -> dict[str, Any]:
        try:
            response = self._http.post(route, json=payload)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise NLLBApiError(
                f"HTTP {exc.response.status_code} em {route}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise NLLBApiError(f"Falha de conexão em {route}: {exc}") from exc
        except ValueError as exc:
            raise NLLBApiError(f"Resposta JSON inválida em {route}.") from exc
        if not isinstance(result, dict):
            raise NLLBApiError(f"Resposta inesperada em {route}: esperado objeto JSON.")
        return result

    def detect_language(self, text: str) -> str:
        # O contrato da rota de idioma limita a amostra a 512 caracteres.
        sample = text.strip()[:LANGUAGE_SAMPLE_CHARACTERS]
        payload = self._get_json("/api/v4/language", params={"text": sample})
        language = payload.get("language")
        if not isinstance(language, str) or not language:
            raise NLLBApiError("A resposta de /api/v4/language não contém 'language'.")
        return language

    @lru_cache(maxsize=4096)
    def count_tokens(self, text: str) -> int:
        payload = self._get_json(
            "/api/v4/translator/tokens", params={"text": text}
        )
        length = payload.get("length")
        if not isinstance(length, int) or length < 0:
            raise NLLBApiError(
                "A resposta de /api/v4/translator/tokens não contém 'length' válido."
            )
        return length

    def translate(self, text: str, source: str) -> str:
        payload = self._post_json(
            "/api/v4/translator",
            payload={"text": text, "source": source, "target": TARGET_LANGUAGE},
        )
        result = payload.get("result")
        if not isinstance(result, str):
            raise NLLBApiError(
                "A resposta de /api/v4/translator não contém 'result' textual."
            )
        return result

    def _fits(self, text: str) -> bool:
        return (
            len(text) <= MAX_TRANSLATION_CHARACTERS
            and self.count_tokens(text) <= MAX_TRANSLATION_TOKENS
        )

    def _split_oversized(self, text: str) -> list[str]:
        """Divide um segmento sem espaços usando contagens do tokenizer remoto."""
        chunks: list[str] = []
        remainder = text
        while remainder:
            if self._fits(remainder):
                chunks.append(remainder)
                break

            low, high = 1, min(len(remainder), MAX_TRANSLATION_CHARACTERS)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if self._fits(remainder[:middle]):
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise NLLBApiError(
                    "Não foi possível criar um trecho dentro do limite de 512 tokens."
                )
            chunks.append(remainder[:best])
            remainder = remainder[best:]
        return chunks

    def _split_long_clause(self, clause: str) -> list[str]:
        """Último recurso: divide uma cláusula longa nos limites de palavras."""
        words_and_spaces = re.findall(r"\s+|\S+", clause)
        chunks: list[str] = []
        current = ""
        for part in words_and_spaces:
            candidate = current + part
            if self._fits(candidate):
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if self._fits(part):
                current = part
            else:
                chunks.extend(self._split_oversized(part))
        if current:
            chunks.append(current)
        return chunks

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Preserva a sentença; para sentenças longas, tenta cortar após vírgulas."""
        clauses = re.split(r"(?<=,)", sentence)
        chunks: list[str] = []
        current = ""
        for clause in clauses:
            clause_chunks = (
                [clause]
                if self._fits(clause)
                else self._split_long_clause(clause)
            )
            for part in clause_chunks:
                candidate = current + part
                if self._fits(candidate):
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part
        if current:
            chunks.append(current)
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Divide nos limites de frases e mantém os separadores originais."""
        chunks: list[str] = []
        # Manter os delimitadores de parágrafo evita perder ou colapsar quebras.
        paragraphs = re.split(r"(\n+)", text)
        for paragraph in paragraphs:
            if not paragraph:
                continue
            if paragraph.isspace() and "\n" in paragraph:
                chunks.append(paragraph)
                continue

            sentences = re.split(r"(?<=[.!?])(?=\s)", paragraph)
            current = ""
            for sentence in sentences:
                sentence_chunks = (
                    [sentence]
                    if self._fits(sentence)
                    else self._split_long_sentence(sentence)
                )
                for part in sentence_chunks:
                    candidate = current + part
                    if self._fits(candidate):
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = part
            if current:
                chunks.append(current)

        if "".join(chunks) != text:
            raise AssertionError("A divisão NLLB não preservou integralmente o texto.")
        return chunks

    def translate_text(self, text: str) -> str:
        """Detecta idioma e traduz os trechos para português, sem truncar."""
        if not text or not text.strip():
            return text
        source = self.detect_language(text)
        if source == TARGET_LANGUAGE:
            return text

        chunks = self.split_text(text)
        translations: list[str] = []
        for chunk in chunks:
            # Separadores de parágrafo/linhas são mantidos sem enviar texto vazio.
            if not chunk.strip():
                translations.append(chunk)
                continue
            translated = self.translate(chunk, source)
            if not translated:
                raise NLLBApiError("A API retornou uma tradução vazia.")
            translations.append(translated)
        return "".join(translations)
