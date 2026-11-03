"""Interface de linha de comando do harness PubMed."""

import argparse
import hashlib
import json
import sys
from math import ceil
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ArticleInformation
from src.translate import (
    DEFAULT_CONFIGURATION_PATH,
    create_translation_graph,
    translate_article,
)


def _json_file(path: str) -> Path:
    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        raise argparse.ArgumentTypeError("O arquivo de entrada deve ter extensão .json.")
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


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary_path = path.with_name(f"{path.name}.writing")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_articles(path: Path, articles: list[ArticleInformation]) -> None:
    _write_json_atomic(path, [article.model_dump() for article in articles])


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "translate":
        output_path = args.output or args.articles_json.with_name(
            f"{args.articles_json.stem}_translated.json"
        )
        temp_dir = Path(__file__).resolve().parent.parent / "data" / "temp"
        batch_prefix = f"{args.articles_json.stem}_translated.temp"
        checkpoint_path = temp_dir / f"{batch_prefix}.state.json"

        try:
            source_bytes = args.articles_json.read_bytes()
            raw_articles = json.loads(source_bytes)
            if not isinstance(raw_articles, list):
                raise ValueError("O arquivo de artigos deve conter uma lista JSON.")
            articles = [ArticleInformation.model_validate(item) for item in raw_articles]

            prompt_bytes = args.prompt_config.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
            total_batches = ceil(len(articles) / args.batch_size)
            temp_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "source_sha256": source_hash,
                "prompt_sha256": prompt_hash,
                "batch_size": args.batch_size,
                "completed_batches": [],
            }
            if checkpoint_path.exists():
                try:
                    saved_checkpoint = json.loads(
                        checkpoint_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    saved_checkpoint = None
                if (
                    isinstance(saved_checkpoint, dict)
                    and saved_checkpoint.get("source_sha256") == source_hash
                    and saved_checkpoint.get("prompt_sha256") == prompt_hash
                    and saved_checkpoint.get("batch_size") == args.batch_size
                    and isinstance(saved_checkpoint.get("completed_batches"), list)
                ):
                    checkpoint = saved_checkpoint

            declared_completed = set()
            for batch_number in checkpoint["completed_batches"]:
                if isinstance(batch_number, int) and 1 <= batch_number <= total_batches:
                    declared_completed.add(batch_number)

            cached_batches: dict[int, list[ArticleInformation]] = {}
            for batch_number in range(1, total_batches + 1):
                if batch_number not in declared_completed:
                    break
                start = (batch_number - 1) * args.batch_size
                expected_count = min(args.batch_size, len(articles) - start)
                batch_path = temp_dir / f"{batch_prefix}.{batch_number}.json"
                try:
                    raw_batch = json.loads(batch_path.read_text(encoding="utf-8"))
                    if not isinstance(raw_batch, list) or len(raw_batch) != expected_count:
                        break
                    cached_batches[batch_number] = [
                        ArticleInformation.model_validate(item) for item in raw_batch
                    ]
                except (OSError, json.JSONDecodeError, ValidationError):
                    break

            checkpoint["completed_batches"] = list(cached_batches)
            _write_json_atomic(checkpoint_path, checkpoint)
            graph = create_translation_graph(args.prompt_config)
            translated_articles: list[ArticleInformation] = []

            with tqdm(total=len(articles), desc="Traduzindo", unit="artigo") as progress:
                for batch_number, start in enumerate(
                    range(0, len(articles), args.batch_size), start=1
                ):
                    batch = articles[start : start + args.batch_size]
                    if batch_number in cached_batches:
                        translated_batch = cached_batches[batch_number]
                        progress.update(len(batch))
                    else:
                        translated_batch = []
                        for article in batch:
                            translated_batch.append(translate_article(article, graph=graph))
                            progress.update(1)

                        batch_path = temp_dir / f"{batch_prefix}.{batch_number}.json"
                        _write_articles(batch_path, translated_batch)
                        checkpoint["completed_batches"].append(batch_number)
                        _write_json_atomic(checkpoint_path, checkpoint)

                    translated_articles.extend(translated_batch)
                    progress.set_postfix(batch=f"{batch_number}/{total_batches}")

            _write_articles(output_path, translated_articles)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            print(f"Erro ao traduzir artigos: {exc}", file=sys.stderr)
            return 1

        print(f"Traduções salvas em lotes em: {temp_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Arquivo consolidado salvo em: {output_path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
