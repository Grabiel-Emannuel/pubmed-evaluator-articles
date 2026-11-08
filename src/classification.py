"""Grafo LangGraph para classificar artigos conforme critérios em Markdown."""

import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from .chat import create_chat
from .models import ArticleInformation, ConfigurationPrompt
from .settings import Settings
from .util import ROLE2TEMPLATE


DEFAULT_CONFIGURATION_PATH = (
    Path(__file__).resolve().parent
    / "prompts"
    / "classification"
    / "classificador.py"
)


class ClassifiedArticle(ArticleInformation):
    """Artigo acompanhado do rótulo binário de validade (1 ou 0)."""

    valid: Literal[0, 1]


class ClassificationLabel(BaseModel):
    """Saída estruturada do modelo, limitada a válido (1) ou inválido (0)."""

    valid: Literal[0, 1] = Field(
        description="1 se o artigo atende aos critérios de inclusão; 0 caso contrário."
    )


class ClassificationState(TypedDict):
    article: ArticleInformation
    valid: NotRequired[Literal[0, 1]]


def _load_configuration(path: str | Path) -> ConfigurationPrompt:
    configuration_path = Path(path)
    return ConfigurationPrompt.model_validate_json(
        configuration_path.read_text(encoding="utf-8")
    )


def create_classification_graph(
    criteria_path: str | Path,
    configuration_path: str | Path = DEFAULT_CONFIGURATION_PATH,
):
    """Cria um grafo para classificar artigos usando critérios de um arquivo Markdown."""
    criteria = Path(criteria_path).read_text(encoding="utf-8").strip()
    if not criteria:
        raise ValueError("O arquivo Markdown de critérios está vazio.")

    configuration = _load_configuration(configuration_path)
    chat_options = (
        {"base_url": Settings.OLLAMA_HOST}
        if Settings.PROVIDER == "ollama" and Settings.OLLAMA_HOST
        else {}
    )
    if Settings.PROVIDER == "ollama":
        chat_options["think"] = bool(configuration.think)
    chat = create_chat(Settings.PROVIDER, configuration.model, **chat_options)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (ROLE2TEMPLATE[prompt.role], prompt.instruction)
            for prompt in configuration.prompts
        ]
    )
    classification_chain = prompt_template | chat.with_structured_output(
        ClassificationLabel
    )

    def classify_node(state: ClassificationState) -> dict[str, Literal[0, 1]]:
        article = state["article"]
        classification = classification_chain.invoke(
            {
                "criteria": criteria,
                "title": article.title,
                "abstract": article.abstract,
            }
        )
        return {"valid": classification.valid}

    graph = StateGraph(ClassificationState)
    graph.add_node("classify", classify_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    return graph.compile()


def classify_article(
    article: ArticleInformation,
    criteria_path: str | Path,
    configuration_path: str | Path = DEFAULT_CONFIGURATION_PATH,
    *,
    graph: Any | None = None,
) -> Literal[0, 1]:
    """Classifica um artigo: 1 se válido, 0 se inválido segundo os critérios."""
    classification_graph = graph or create_classification_graph(
        criteria_path, configuration_path
    )
    result = classification_graph.invoke({"article": article})
    return result["valid"]


def classify_articles_file(
    articles_path: str | Path,
    criteria_path: str | Path,
    configuration_path: str | Path = DEFAULT_CONFIGURATION_PATH,
) -> list[ClassifiedArticle]:
    """Lê uma lista JSON de artigos e devolve artigos com o rótulo binário."""
    raw_articles = json.loads(Path(articles_path).read_text(encoding="utf-8"))
    if not isinstance(raw_articles, list):
        raise ValueError("O arquivo de artigos deve conter uma lista JSON.")

    articles = [ArticleInformation.model_validate(item) for item in raw_articles]
    graph = create_classification_graph(criteria_path, configuration_path)
    return [
        ClassifiedArticle(
            title=article.title,
            abstract=article.abstract,
            valid=classify_article(article, criteria_path, graph=graph),
        )
        for article in articles
    ]
