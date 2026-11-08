"""Utilitários reutilizáveis para processar artigos em lotes retomáveis."""

import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from .models import ArticleInformation


Result = TypeVar("Result", bound=BaseModel)


def load_articles(path: str | Path) -> tuple[list[ArticleInformation], str]:
    """Carrega artigos de JSON e retorna também o hash dos bytes de origem."""
    source_bytes = Path(path).read_bytes()
    raw_articles = json.loads(source_bytes)
    if not isinstance(raw_articles, list):
        raise ValueError("O arquivo de artigos deve conter uma lista JSON.")
    articles = [ArticleInformation.model_validate(item) for item in raw_articles]
    return articles, hashlib.sha256(source_bytes).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Calcula o SHA-256 de um arquivo sem pressupor seu formato."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_fingerprint(cache_key: dict[str, str]) -> str:
    """Gera uma assinatura curta e estável para isolar os arquivos de lote."""
    canonical_key = json.dumps(cache_key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16]


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.writing")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def process_articles_in_batches(
    articles: list[ArticleInformation],
    *,
    process_article: Callable[[ArticleInformation], Result],
    result_model: type[Result],
    output_path: str | Path,
    temp_dir: str | Path,
    batch_prefix: str,
    batch_size: int,
    cache_key: dict[str, str],
    progress_description: str,
) -> list[Result]:
    """Processa artigos, recupera lotes válidos e grava saída e checkpoint.

    ``cache_key`` deve identificar todas as entradas que afetam o resultado,
    como os hashes do arquivo de origem, dos critérios e do prompt.
    """
    if batch_size < 1:
        raise ValueError("O tamanho do lote deve ser maior que zero.")

    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = temp_path / f"{batch_prefix}.state.json"
    total_batches = ceil(len(articles) / batch_size)
    checkpoint: dict[str, Any] = {
        "cache_key": cache_key,
        "batch_size": batch_size,
        "completed_batches": [],
    }

    if checkpoint_path.exists():
        try:
            saved_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved_checkpoint = None
        if (
            isinstance(saved_checkpoint, dict)
            and saved_checkpoint.get("cache_key") == cache_key
            and saved_checkpoint.get("batch_size") == batch_size
            and isinstance(saved_checkpoint.get("completed_batches"), list)
        ):
            checkpoint = saved_checkpoint

    declared_completed = {
        batch_number
        for batch_number in checkpoint["completed_batches"]
        if isinstance(batch_number, int) and 1 <= batch_number <= total_batches
    }
    cached_batches: dict[int, list[Result]] = {}
    for batch_number in range(1, total_batches + 1):
        if batch_number not in declared_completed:
            break
        start = (batch_number - 1) * batch_size
        expected_count = min(batch_size, len(articles) - start)
        batch_path = temp_path / f"{batch_prefix}.{batch_number}.json"
        try:
            raw_batch = json.loads(batch_path.read_text(encoding="utf-8"))
            if not isinstance(raw_batch, list) or len(raw_batch) != expected_count:
                break
            cached_batches[batch_number] = [
                result_model.model_validate(item) for item in raw_batch
            ]
        except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
            break

    checkpoint["completed_batches"] = list(cached_batches)
    _write_json_atomic(checkpoint_path, checkpoint)
    results: list[Result] = []

    with tqdm(total=len(articles), desc=progress_description, unit="artigo") as progress:
        for batch_number, start in enumerate(range(0, len(articles), batch_size), start=1):
            batch = articles[start : start + batch_size]
            if batch_number in cached_batches:
                processed_batch = cached_batches[batch_number]
                progress.update(len(batch))
            else:
                processed_batch = []
                for article in batch:
                    processed_batch.append(process_article(article))
                    progress.update(1)

                batch_path = temp_path / f"{batch_prefix}.{batch_number}.json"
                _write_json_atomic(
                    batch_path,
                    [result.model_dump() for result in processed_batch],
                )
                checkpoint["completed_batches"].append(batch_number)
                _write_json_atomic(checkpoint_path, checkpoint)

            results.extend(processed_batch)
            progress.set_postfix(batch=f"{batch_number}/{total_batches}")

    _write_json_atomic(
        Path(output_path),
        [result.model_dump() for result in results],
    )
    return results
