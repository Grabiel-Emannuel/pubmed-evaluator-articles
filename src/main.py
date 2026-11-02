"""Interface de linha de comando do harness PubMed."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.translate import DEFAULT_CONFIGURATION_PATH, translate_articles_file


def _json_file(path: str) -> Path:
    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError("O arquivo de entrada deve ter extensão .json.")
    if not file_path.is_file():
        raise argparse.ArgumentTypeError(f"Arquivo não encontrado: {file_path}")
    return file_path


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
            "Caminho do JSON de saída. Por padrão, grava ao lado do arquivo de "
            "entrada com o sufixo _translated."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "translate":
        output_path = args.output or args.articles_json.with_name(
            f"{args.articles_json.stem}_translated.json"
        )
        try:
            translated_articles = translate_articles_file(
                args.articles_json,
                configuration_path=args.prompt_config,
            )
            output_path.write_text(
                json.dumps(
                    [article.model_dump() for article in translated_articles],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            print(f"Erro ao traduzir artigos: {exc}", file=sys.stderr)
            return 1

        print(f"Traduções salvas em: {output_path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
