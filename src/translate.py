"""Grafo LangGraph para traduzir artigos científicos para português."""

import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .chat import create_chat
from .models import ArticleInformation, ConfigurationPrompt
from .settings import Settings


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


def _make_messages(
    configuration: ConfigurationPrompt, text: str
) -> list[Any]:
    role_to_message = {
        "System": SystemMessage,
        "User": HumanMessage,
        "Ai": AIMessage,
    }
    messages = [
        role_to_message[prompt.role](content=prompt.instruction)
        for prompt in configuration.prompts
    ]
    messages.append(HumanMessage(content=text))
    return messages


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
):
    """Cria e compila um grafo de um nó para traduzir um ArticleInformation."""
    configuration = _load_configuration(configuration_path)
    chat_options = (
        {"base_url": Settings.OLLAMA_HOST}
        if Settings.PROVIDER == "ollama" and Settings.OLLAMA_HOST
        else {}
    )
    chat = create_chat(Settings.PROVIDER, configuration.model, **chat_options)

    def translate_node(state: TranslationState) -> dict[str, ArticleInformation]:
        article = state["article"]
        translated_title = _response_text(
            chat.invoke(_make_messages(configuration, article.title))
        )
        translated_abstract = _response_text(
            chat.invoke(_make_messages(configuration, article.abstract))
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
