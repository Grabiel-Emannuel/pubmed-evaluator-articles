#!/usr/bin/env python3
"""Render downloaded article titles and abstracts into a Word document."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from docxtpl import DocxTemplate, Listing


def load_articles(path: Path) -> list[dict[str, str]]:
    """Load and validate the JSON list produced by download_titles.py."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"Não foi possível ler '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"O arquivo '{path}' não contém JSON válido: {error}") from error

    if not isinstance(data, list):
        raise RuntimeError("O JSON deve conter uma lista de artigos.")

    articles: list[dict[str, str]] = []
    for number, article in enumerate(data, start=1):
        if not isinstance(article, dict) or not isinstance(article.get("title"), str):
            raise RuntimeError(f"O artigo {number} não possui um campo 'title' válido.")
        abstract = article.get("abstract", "")
        if not isinstance(abstract, str):
            raise RuntimeError(f"O abstract do artigo {number} não é um texto válido.")
        # Listing preserves the section breaks from PubMed as line breaks in Word.
        articles.append({"title": article["title"], "abstract": Listing(abstract)})

    return articles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera um DOCX legível com os títulos e abstracts baixados do PubMed."
    )
    parser.add_argument("json_file", type=Path, help="arquivo JSON com títulos e abstracts")
    parser.add_argument("template_file", type=Path, help="template DOCX com campos Jinja")
    parser.add_argument("output_file", type=Path, help="caminho do DOCX que será gerado")
    args = parser.parse_args(argv)

    try:
        articles = load_articles(args.json_file)
        template = DocxTemplate(args.template_file)
        template.render({"articles": articles})
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        template.save(args.output_file)
    except (RuntimeError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Documento gerado com {len(articles)} artigos em {args.output_file}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
