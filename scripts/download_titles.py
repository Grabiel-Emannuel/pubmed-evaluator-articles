#!/usr/bin/env python3
"""Download article titles and abstracts matching PubMed search strings.

Usage:
    python download_titles.py <search_strings_file> <max_titles>

Each non-empty line in the input file is treated as a separate PubMed query.
The output is written as JSON to the same path with a ``.json`` extension.
Set ``max_titles`` to 0 to retrieve every result; otherwise it is the maximum
number of articles to retrieve across all queries.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MIN_REQUEST_INTERVAL = 1 / 3
FETCH_BATCH_SIZE = 10
MAX_RESULTS_PER_SEARCH = 10_000
MAX_RETRIES = 5


class PubMedClient:
    """Small E-utilities client that spaces all requests by at least 1/3 sec."""

    def __init__(self) -> None:
        self.last_request_at: float | None = None

    def request(self, endpoint: str, params: dict[str, str]) -> bytes:
        url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
        for attempt in range(MAX_RETRIES):
            if self.last_request_at is not None:
                elapsed = time.monotonic() - self.last_request_at
                if elapsed < MIN_REQUEST_INTERVAL:
                    time.sleep(MIN_REQUEST_INTERVAL - elapsed)

            self.last_request_at = time.monotonic()
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "pubmed-title-downloader/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == MAX_RETRIES - 1:
                    detail = error.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(
                        f"Erro HTTP {error.code} do PubMed: {detail or error.reason}"
                    ) from error
                time.sleep(min(2**attempt, 10))
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"Falha ao consultar o PubMed: {error}") from error
                time.sleep(min(2**attempt, 10))
        raise RuntimeError("Não foi possível consultar o PubMed.")

    def search(self, query: str) -> tuple[int, str, str]:
        """Start a history-backed search and return count, WebEnv, and query key."""
        payload = self.request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmax": "0",
                "usehistory": "y",
                "retmode": "xml",
            },
        )
        root = ET.fromstring(payload)
        error = root.findtext("ERROR")
        if error:
            raise RuntimeError(f"Busca PubMed inválida: {error}")
        count = int(root.findtext("Count", "0"))
        webenv = root.findtext("WebEnv")
        query_key = root.findtext("QueryKey")
        if count and (not webenv or not query_key):
            raise RuntimeError("O PubMed não retornou os dados necessários para buscar os resultados.")
        return count, webenv or "", query_key or ""

    def search_partitions(
        self, query: str, initial_result: tuple[int, str, str], needed: int
    ) -> list[tuple[int, str, str]]:
        """Split PubMed searches into date ranges below its 10,000-result cap."""
        if initial_result[0] <= MAX_RESULTS_PER_SEARCH or needed <= MAX_RESULTS_PER_SEARCH:
            return [initial_result]

        partitions: list[tuple[int, str, str]] = []
        first_day = date(1800, 1, 1)
        last_day = date.today()

        def search_range(start: date, end: date) -> None:
            date_query = (
                f'({query}) AND '
                f'({start:%Y/%m/%d}:{end:%Y/%m/%d}[edat])'
            )
            result = self.search(date_query)
            count = result[0]
            if count <= MAX_RESULTS_PER_SEARCH:
                if count:
                    partitions.append(result)
                return
            if start == end:
                raise RuntimeError(
                    f"O intervalo de {start:%Y-%m-%d} ainda contém mais de "
                    f"{MAX_RESULTS_PER_SEARCH:,} resultados do PubMed."
                )

            midpoint = start + (end - start) // 2
            search_range(start, midpoint)
            search_range(midpoint + timedelta(days=1), end)

        search_range(first_day, last_day)
        return partitions

    def fetch_articles(
        self, webenv: str, query_key: str, start: int, amount: int
    ) -> list[dict[str, str]]:
        payload = self.request(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "query_key": query_key,
                "WebEnv": webenv,
                "retstart": str(start),
                "retmax": str(amount),
                "rettype": "abstract",
                "retmode": "xml",
            },
        )
        root = ET.fromstring(payload)
        articles: list[dict[str, str]] = []
        for article in root.findall(".//PubmedArticle"):
            article_title = article.find(".//ArticleTitle")
            if article_title is None:
                continue

            # Titles and abstract sections can contain inline XML elements.
            title = "".join(article_title.itertext()).strip()
            if not title:
                continue

            abstract_parts: list[str] = []
            abstract = article.find(".//Abstract")
            if abstract is not None:
                for section in abstract.findall("AbstractText"):
                    text = "".join(section.itertext()).strip()
                    if not text:
                        continue
                    label = section.get("Label")
                    abstract_parts.append(f"{label}: {text}" if label else text)

            articles.append({"title": title, "abstract": "\n".join(abstract_parts)})
        return articles


def read_queries(path: Path) -> list[str]:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise RuntimeError(f"Não foi possível ler o arquivo '{path}': {error}") from error
    queries = [line.strip() for line in content.splitlines() if line.strip()]
    if not queries:
        raise RuntimeError(f"O arquivo '{path}' não contém strings de busca.")
    return queries


def download_articles(queries: list[str], max_titles: int) -> list[dict[str, str]]:
    client = PubMedClient()
    articles: list[dict[str, str]] = []

    for query_number, query in enumerate(queries, start=1):
        if max_titles and len(articles) >= max_titles:
            break

        initial_result = client.search(query)
        count = initial_result[0]
        remaining = max_titles - len(articles) if max_titles else count
        partitions = client.search_partitions(query, initial_result, remaining)
        partition_total = sum(partition[0] for partition in partitions)
        to_fetch = min(partition_total, remaining)
        with tqdm(
            total=to_fetch,
            desc=f"Busca {query_number}/{len(queries)}",
            unit="artigo",
            file=sys.stderr,
        ) as progress:
            remaining_in_query = to_fetch
            for partition_count, webenv, query_key in partitions:
                if remaining_in_query <= 0:
                    break
                partition_fetch = min(partition_count, remaining_in_query)
                for start in range(0, partition_fetch, FETCH_BATCH_SIZE):
                    batch_size = min(FETCH_BATCH_SIZE, partition_fetch - start)
                    batch = client.fetch_articles(webenv, query_key, start, batch_size)
                    articles.extend(batch)
                    progress.update(batch_size)
                    remaining_in_query -= batch_size

    return articles


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(
            "Uso: python download_titles.py <arquivo_strings> <quantidade_artigos>",
            file=sys.stderr,
        )
        return 2

    input_path = Path(argv[0])
    try:
        max_titles = int(argv[1])
        if max_titles < 0:
            raise ValueError
        queries = read_queries(input_path)
        articles = download_articles(queries, max_titles)
        output_path = input_path.with_suffix(".json")
        output_path.write_text(
            json.dumps(articles, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except ValueError:
        print("A quantidade de artigos deve ser um inteiro não negativo.", file=sys.stderr)
        return 2
    except (RuntimeError, ET.ParseError, OSError) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"Baixados {len(articles)} artigos (título e resumo) em {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
