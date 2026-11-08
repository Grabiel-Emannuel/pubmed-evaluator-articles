"""Grafo LangGraph para traduzir artigos científicos para português."""

import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from .chat import create_chat
from .models import ArticleInformation, ConfigurationPrompt
from .nllb import NLLBApiError, NLLBClient
from .settings import Settings
from .util import ROLE2TEMPLATE


DEFAULT_CONFIGURATION_PATH = (
    Path(__file__).resolve().parent / "prompts" / "translate" / "translator.json"
)


class TranslationState(TypedDict):
    article: ArticleInformation
    translated_article: NotRequired[ArticleInformation]


def _load_configuration(path: str | Path) -> ConfigurationPrompt:
    configuration_path = Path(path)
    return ConfigurationPrompt.model_validate_json(
        configuration_path.read_text(encoding="utf-8")
    )


def _response_text(response: Any) -> str:
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if text_blocks:
            return "".join(text_blocks).strip()
    raise TypeError("O modelo não retornou conteúdo textual para a tradução.")


def create_translation_graph(
    configuration_path: str | Path = DEFAULT_CONFIGURATION_PATH,
    *,
    thinking: bool | None = None,
):
    """Cria um grafo de tradução; ``thinking`` ativa raciocínio no Ollama."""
    configuration = _load_configuration(configuration_path)
    chat_options = (
        {"base_url": Settings.OLLAMA_HOST}
        if Settings.PROVIDER == "ollama" and Settings.OLLAMA_HOST
        else {}
    )
    if Settings.PROVIDER == "ollama":
        configured_think = configuration.think if thinking is None else thinking
        chat_options["think"] = bool(configured_think)
    chat = create_chat(Settings.PROVIDER, configuration.model, **chat_options)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (ROLE2TEMPLATE[prompt.role], prompt.instruction)
            for prompt in configuration.prompts
        ]
    )
    translation_chain = prompt_template | chat

    def translate_node(state: TranslationState) -> dict[str, ArticleInformation]:
        article = state["article"]
        translated_title = _response_text(
            translation_chain.invoke({"text": article.title})
        )
        translated_abstract = _response_text(
            translation_chain.invoke({"text": article.abstract})
        )
        translated_article = ArticleInformation(
            title=translated_title,
            abstract=translated_abstract,
        )
        return {"translated_article": translated_article}

    graph = StateGraph(TranslationState)
    graph.add_node("translate", translate_node)
    graph.add_edge(START, "translate")
    graph.add_edge("translate", END)
    return graph.compile()


def create_nllb_translation_graph(client: NLLBClient | None = None):
    """Cria um grafo LangGraph de um nó usando a API REST NLLB."""
    nllb_client = client or NLLBClient(
        Settings.NLLB_API_URL,
        timeout_seconds=Settings.NLLB_TIMEOUT_SECONDS,
    )

    def translate_node(state: TranslationState) -> dict[str, ArticleInformation]:
        article = state["article"]
        translated_fields: dict[str, str] = {}
        for field_name in ("title", "abstract"):
            source_text = getattr(article, field_name)
            if not source_text or not source_text.strip():
                translated_fields[field_name] = source_text
                continue
            try:
                translated_fields[field_name] = nllb_client.translate_text(source_text)
            except NLLBApiError as exc:
                article_title = article.title[:160]
                if len(article.title) > 160:
                    article_title += "…"
                raise NLLBApiError(
                    f"Falha no artigo '{article_title}', campo '{field_name}': {exc}"
                ) from exc

        return {
            "translated_article": ArticleInformation.model_validate(translated_fields)
        }

    graph = StateGraph(TranslationState)
    graph.add_node("translate", translate_node)
    graph.add_edge(START, "translate")
    graph.add_edge("translate", END)
    return graph.compile()


def translate_article(
    article: ArticleInformation,
    configuration_path: str | Path = DEFAULT_CONFIGURATION_PATH,
    *,
    graph: Any | None = None,
) -> ArticleInformation:
    """Traduz título e abstract em chamadas separadas e devolve o artigo."""
    translation_graph = graph or create_translation_graph(configuration_path)
    result = translation_graph.invoke({"article": article})
    return ArticleInformation.model_validate(result["translated_article"])


def translate_articles_file(
    articles_path: str | Path,
    configuration_path: str | Path = DEFAULT_CONFIGURATION_PATH,
) -> list[ArticleInformation]:
    """Lê uma lista JSON de artigos e devolve a lista traduzida."""
    raw_articles = json.loads(Path(articles_path).read_text(encoding="utf-8"))
    if not isinstance(raw_articles, list):
        raise ValueError("O arquivo de artigos deve conter uma lista JSON.")

    articles = [ArticleInformation.model_validate(item) for item in raw_articles]
    graph = create_translation_graph(configuration_path)
    return [translate_article(article, graph=graph) for article in articles]
