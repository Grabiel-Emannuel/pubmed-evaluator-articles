"""Fábrica de modelos de chat LangChain compatíveis com LangGraph."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


_PROVIDERS = {
    "openai": "openai",
    "google": "google_genai",
    "gemini": "google_genai",
    "google_genai": "google_genai",
    "google-genai": "google_genai",
    "claude": "anthropic",
    "anthropic": "anthropic",
    "ollama": "ollama",
}


def create_chat(provider: str, model: str, **kwargs: Any) -> BaseChatModel:
    """Cria um chat LangChain para uso em chains e grafos LangGraph.

    ``provider`` aceita ``openai``, ``google``/``gemini``,
    ``claude``/``anthropic`` ou ``ollama``. ``model`` é o identificador
    específico do modelo no provedor. Opções adicionais, como ``temperature``,
    são encaminhadas para ``init_chat_model``.

    As credenciais devem estar configuradas nas variáveis de ambiente esperadas
    pela integração escolhida. Para Ollama, o serviço local também deve estar
    disponível.
    """
    if not provider or not provider.strip():
        raise ValueError("O provedor do chat deve ser informado.")
    if not model or not model.strip():
        raise ValueError("O modelo do chat deve ser informado.")

    provider_key = provider.strip().lower()
    try:
        model_provider = _PROVIDERS[provider_key]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Provedor '{provider}' não suportado. Opções: {supported}."
        ) from exc

    if model_provider == "ollama":
        kwargs["keep_alive"] = "30m"

    return init_chat_model(
        model=model.strip(),
        model_provider=model_provider,
        **kwargs,
    )
