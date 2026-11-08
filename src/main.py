"""Interface de linha de comando do harness PubMed."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ArticleInformation
from src.batch import (
    cache_fingerprint,
    file_sha256,
    load_articles,
    process_articles_in_batches,
)
from src.classification import (
    DEFAULT_CONFIGURATION_PATH as DEFAULT_CLASSIFICATION_CONFIGURATION_PATH,
    ClassifiedArticle,
    classify_article,
    create_classification_graph,
)
from src.translate import (
    DEFAULT_CONFIGURATION_PATH,
    create_nllb_translation_graph,
    create_translation_graph,
    translate_article,
)
from src.nllb import NLLBApiError
from src.settings import Settings


def _json_file(path: str) -> Path:
    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError("O arquivo de entrada deve ter extensão .json.")
    if not file_path.is_file():
        raise argparse.ArgumentTypeError(f"Arquivo não encontrado: {file_path}")
    return file_path


def _markdown_file(path: str) -> Path:
    file_path = Path(path)
    if file_path.suffix.lower() != ".md":
        raise argparse.ArgumentTypeError("O arquivo de critérios deve ter extensão .md.")
    if not file_path.is_file():
        raise argparse.ArgumentTypeError(f"Arquivo não encontrado: {file_path}")
    return file_path


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("O tamanho do lote deve ser um inteiro.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("O tamanho do lote deve ser maior que zero.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pubmed",
        description="Harness CLI para processar artigos baixados do PubMed.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    translate_parser = commands.add_parser(
        "translate",
        help="Traduz os artigos de um arquivo JSON.",
        description="Traduz título e abstract de cada artigo de uma lista JSON.",
    )
    translate_parser.add_argument(
        "articles_json",
        type=_json_file,
        help="Caminho obrigatório para o arquivo JSON com a lista de artigos.",
    )
    translate_parser.add_argument(
        "--prompt-config",
        type=Path,
        default=DEFAULT_CONFIGURATION_PATH,
        help=(
            "Arquivo JSON com a configuração e os prompts de tradução "
            f"(padrão: {DEFAULT_CONFIGURATION_PATH})."
        ),
    )
    translate_parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Caminho do JSON consolidado de saída. Por padrão, grava ao lado do "
            "arquivo de entrada com o sufixo _translated."
        ),
    )
    translate_parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=100,
        help="Quantidade de artigos por arquivo temporário (padrão: 100).",
    )
    translation_provider = translate_parser.add_mutually_exclusive_group()
    translation_provider.add_argument(
        "--with-chat",
        action="store_true",
        help="Usa o provedor de chat LangChain configurado (padrão).",
    )
    translation_provider.add_argument(
        "--with-nllb",
        action="store_true",
        help="Usa a API REST NLLB para traduzir títulos e abstracts.",
    )
    thinking_group = translate_parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        default=None,
        help="Ativa o raciocínio do modelo no Ollama, sobrescrevendo o prompt.",
    )
    thinking_group.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        default=None,
        help="Desativa o raciocínio do modelo no Ollama.",
    )

    classify_parser = commands.add_parser(
        "classify",
        help="Classifica os artigos como válidos (1) ou inválidos (0).",
        description=(
            "Classifica título e abstract de cada artigo usando os critérios "
            "de um arquivo Markdown."
        ),
    )
    classify_parser.add_argument(
        "articles_json",
        type=_json_file,
        help="Caminho para o arquivo JSON com a lista de artigos.",
    )
    classify_parser.add_argument(
        "--criteria",
        type=_markdown_file,
        required=True,
        help="Arquivo Markdown (.md) com os critérios de inclusão.",
    )
    classify_parser.add_argument(
        "--prompt-config",
        type=Path,
        default=DEFAULT_CLASSIFICATION_CONFIGURATION_PATH,
        help=(
            "Arquivo JSON com a configuração e os prompts de classificação "
            f"(padrão: {DEFAULT_CLASSIFICATION_CONFIGURATION_PATH})."
        ),
    )
    classify_parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Caminho do JSON de saída. Por padrão, grava ao lado do arquivo "
            "de entrada com o sufixo _classified."
        ),
    )
    classify_parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=100,
        help="Quantidade de artigos por arquivo temporário (padrão: 100).",
    )
    return parser


def _run_translation(args: argparse.Namespace) -> int:
    output_path = args.output or args.articles_json.with_name(
        f"{args.articles_json.stem}_translated.json"
    )
    temp_dir = Path(__file__).resolve().parent.parent / "data" / "temp"

    try:
        articles, source_hash = load_articles(args.articles_json)
        if args.with_nllb:
            cache_key = {
                "source_sha256": source_hash,
                "translation_mode": "nllb",
                "nllb_api_url": Settings.NLLB_API_URL,
            }
            graph = create_nllb_translation_graph()
        else:
            prompt_hash = file_sha256(args.prompt_config)
            cache_key = {
                "source_sha256": source_hash,
                "prompt_sha256": prompt_hash,
                "provider": Settings.PROVIDER or "",
                "ollama_host": Settings.OLLAMA_HOST or "",
                "thinking": args.thinking,
            }
            graph = create_translation_graph(
                args.prompt_config,
                thinking=args.thinking,
            )
        cache_id = cache_fingerprint(cache_key)
        batch_prefix = (
            f"{args.articles_json.stem}_translated_{cache_id}_"
            f"{args.batch_size}.temp"
        )
        checkpoint_path = temp_dir / f"{batch_prefix}.state.json"
        process_articles_in_batches(
            articles,
            process_article=lambda article: translate_article(article, graph=graph),
            result_model=ArticleInformation,
            output_path=output_path,
            temp_dir=temp_dir,
            batch_prefix=batch_prefix,
            batch_size=args.batch_size,
            cache_key=cache_key,
            progress_description="Traduzindo",
        )
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
        TypeError,
        NLLBApiError,
    ) as exc:
        print(f"Erro ao traduzir artigos: {exc}", file=sys.stderr)
        return 1

    print(f"Traduções salvas em lotes em: {temp_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Arquivo consolidado salvo em: {output_path}")
    return 0


def _run_classification(args: argparse.Namespace) -> int:
    output_path = args.output or args.articles_json.with_name(
        f"{args.articles_json.stem}_classified.json"
    )
    temp_dir = Path(__file__).resolve().parent.parent / "data" / "temp"

    try:
        articles, source_hash = load_articles(args.articles_json)
        criteria_hash = file_sha256(args.criteria)
        prompt_hash = file_sha256(args.prompt_config)
        cache_key = {
            "source_sha256": source_hash,
            "criteria_sha256": criteria_hash,
            "prompt_sha256": prompt_hash,
            "provider": Settings.PROVIDER or "",
            "ollama_host": Settings.OLLAMA_HOST or "",
        }
        cache_id = cache_fingerprint(cache_key)
        batch_prefix = (
            f"{args.articles_json.stem}_classified_{cache_id}_"
            f"{args.batch_size}.temp"
        )
        checkpoint_path = temp_dir / f"{batch_prefix}.state.json"
        graph = create_classification_graph(args.criteria, args.prompt_config)
        process_articles_in_batches(
            articles,
            process_article=lambda article: ClassifiedArticle(
                title=article.title,
                abstract=article.abstract,
                valid=classify_article(article, args.criteria, graph=graph),
            ),
            result_model=ClassifiedArticle,
            output_path=output_path,
            temp_dir=temp_dir,
            batch_prefix=batch_prefix,
            batch_size=args.batch_size,
            cache_key=cache_key,
            progress_description="Classificando",
        )
    except (OSError, json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        print(f"Erro ao classificar artigos: {exc}", file=sys.stderr)
        return 1

    print(f"Classificações salvas em lotes em: {temp_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Arquivo consolidado salvo em: {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "translate":
        return _run_translation(args)
    if args.command == "classify":
        return _run_classification(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
