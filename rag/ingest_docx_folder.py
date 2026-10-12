"""Batch-index DOCX documents with the existing FOS RAG store.

Examples:
    python ingest_docx_folder.py ./rpd --type rpd --discipline-code Б1.В.07
    python ingest_docx_folder.py ./documents --manifest documents.json --db index.sqlite3
    python ingest_docx_folder.py ./documents --manifest documents.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fos_rag import DOCUMENT_TYPES, OpenAIEmbedder, RAGStore


@dataclass(frozen=True)
class DocumentJob:
    path: Path
    relative_path: str
    document_type: str
    discipline_code: str | None
    competencies: tuple[str, ...]


def discover_docx(folder: Path, recursive: bool = True) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Папка не найдена: {folder}")
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (p for p in iterator if p.is_file() and not p.is_symlink()
         and p.suffix.lower() == ".docx" and not p.name.startswith("~$")),
        key=lambda p: p.relative_to(folder).as_posix().casefold(),
    )


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, dict)
        for key, value in data.items()
    ):
        raise ValueError("Манифест должен быть объектом {относительный_путь: метаданные}")
    return data


def load_curriculum(path: Path | None) -> dict[str, dict[str, Any]]:
    """Index the parsed plan by component code; retain ambiguous codes."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    components = data.get("components")
    if not isinstance(components, list):
        raise ValueError("Учебный план JSON должен содержать список components")
    catalog: dict[str, dict[str, Any]] = {}
    for item in components:
        if not isinstance(item, dict) or not item.get("code") or not item.get("name"):
            continue
        code = str(item["code"]).strip().upper()
        if code in catalog:
            catalog[code]["ambiguous"] = True
        else:
            catalog[code] = {
                "name": str(item["name"]).strip(),
                "competencies": [
                    comp["code"] for comp in item.get("competencies", [])
                    if isinstance(comp, dict) and comp.get("code")
                ],
                "ambiguous": False,
            }
    return catalog


def normalized_name(value: str) -> str:
    text = value.casefold().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return " ".join(text.split())


def clean_filename_title(value: str) -> str:
    name = normalized_name(value)
    name = re.sub(r"\b(?:рпд|фос|рп|202[0-9]|203[0-9])\b", " ", name)
    return " ".join(name.split())


CODE_IN_FILENAME_RE = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])(?:Б[1-3](?:\.[А-ЯA-Z0-9]+)+|ФТД(?:\.[0-9]+)+)"
    r"(?![A-Za-zА-Яа-я0-9])", re.IGNORECASE,
)


def infer_code(
    filename: str,
    catalog: dict[str, dict[str, Any]],
    allow_unknown_codes: bool = False,
) -> str:
    """Match a code or full discipline name in the filename, never fuzzy-guess."""
    stem = Path(filename).stem
    code_matches = [
        code for code in catalog
        if re.search(
            # A parent code is not a match inside a longer code:
            # Б1.В.ДВ.01 must not match Б1.В.ДВ.01.01.
            rf"(?<![A-Za-zА-Яа-я0-9]){re.escape(code)}(?![A-Za-zА-Яа-я0-9]|\.[0-9])",
            stem, flags=re.IGNORECASE,
        )
    ]
    if len(code_matches) == 1:
        code = code_matches[0]
        remaining = re.sub(re.escape(code), " ", stem, count=1, flags=re.IGNORECASE)
        supplied_title = clean_filename_title(remaining)
        expected_title = normalized_name(catalog[code]["name"])
        if supplied_title and expected_title not in supplied_title:
            raise ValueError(
                f"код {code} соответствует «{catalog[code]['name']}», "
                f"а в имени файла указано «{supplied_title}»"
            )
        return code
    if len(code_matches) > 1:
        raise ValueError("несколько кодов в имени файла: " + ", ".join(code_matches))

    mentioned_codes = list(dict.fromkeys(
        match.group(0).upper() for match in CODE_IN_FILENAME_RE.finditer(stem)
    ))
    if len(mentioned_codes) > 1:
        raise ValueError("несколько кодов в имени файла: " + ", ".join(mentioned_codes))
    if mentioned_codes:
        if allow_unknown_codes:
            return mentioned_codes[0]
        raise ValueError(
            f"код {mentioned_codes[0]} отсутствует в учебном плане; "
            "используйте соответствующий план или --allow-unknown-codes"
        )

    name = clean_filename_title(stem)
    exact = [code for code, item in catalog.items()
             if normalized_name(item["name"]) == name]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError("название встречается у нескольких дисциплин: " + ", ".join(exact))
    contained = [
        code for code, item in catalog.items()
        if f" {normalized_name(item['name'])} " in f" {name} "
    ]
    if len(contained) == 1:
        return contained[0]
    if len(contained) > 1:
        raise ValueError("неоднозначное название: " + ", ".join(contained))
    raise ValueError("дисциплина не найдена по имени файла; укажите её в манифесте")


def build_jobs(
    folder: Path,
    files: list[Path],
    manifest: dict[str, dict[str, Any]],
    default_type: str | None,
    default_code: str | None,
    default_competencies: tuple[str, ...],
    catalog: dict[str, dict[str, Any]] | None = None,
    allow_unknown_codes: bool = False,
) -> list[DocumentJob]:
    catalog = catalog or {}
    found = {path.relative_to(folder).as_posix() for path in files}
    unknown = set(manifest) - found
    if unknown:
        raise ValueError("В манифесте нет соответствующих DOCX: " + ", ".join(sorted(unknown)))

    jobs = []
    errors = []
    for path in files:
        relative = path.relative_to(folder).as_posix()
        item = manifest.get(relative, {})
        kind = item.get("document_type", default_type)
        code = item.get("discipline_code", default_code)
        if not code and (catalog or allow_unknown_codes) and kind in {"rpd", "fos_example"}:
            try:
                code = infer_code(path.name, catalog, allow_unknown_codes)
            except ValueError as exc:
                errors.append(f"{relative}: {exc}")
                continue
        if code and catalog:
            code = str(code).strip().upper()
            if code not in catalog and not allow_unknown_codes:
                errors.append(f"{relative}: код {code} отсутствует в учебном плане")
                continue
            if code in catalog and catalog[code]["ambiguous"]:
                errors.append(f"{relative}: код {code} неоднозначен в учебном плане")
                continue
        competencies = item.get(
            "competencies",
            list(default_competencies) or (catalog.get(code, {}).get("competencies", []) if code else []),
        )
        if kind not in DOCUMENT_TYPES:
            errors.append(f"{relative}: укажите тип из {sorted(DOCUMENT_TYPES)}")
            continue
        if kind in {"rpd", "fos_example"} and not code:
            errors.append(f"{relative}: для {kind} нужен discipline_code")
            continue
        if code is not None and (not isinstance(code, str) or not code.strip()):
            errors.append(f"{relative}: discipline_code должен быть строкой")
            continue
        if not isinstance(competencies, list) or not all(
            isinstance(value, str) and value.strip() for value in competencies
        ):
            errors.append(f"{relative}: competencies должен быть списком кодов")
            continue
        jobs.append(DocumentJob(
            path=path,
            relative_path=relative,
            document_type=kind,
            discipline_code=code.strip().upper() if code else None,
            competencies=tuple(value.strip().upper() for value in competencies),
        ))
    if errors:
        raise ValueError("Проверьте метаданные до загрузки:\n" + "\n".join(errors))
    return jobs


def ingest_jobs(
    jobs: list[DocumentJob],
    db_path: Path,
    embedding_model: str,
    embedder_factory: Callable[[str], Any] = OpenAIEmbedder,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "found": len(jobs), "indexed": 0, "unchanged": 0, "failed": 0, "files": []
    }
    if not jobs:
        return report
    embedder = embedder_factory(embedding_model)
    with RAGStore(db_path, embedder) as store:
        for job in jobs:
            try:
                count = store.add_document(
                    job.path, job.document_type, job.discipline_code,
                    job.competencies,
                )
            except Exception as exc:
                report["failed"] += 1
                report["files"].append({
                    "file": job.relative_path, "status": "failed", "error": str(exc),
                })
                continue
            status = "indexed" if count else "unchanged"
            report[status] += 1
            report["files"].append({
                "file": job.relative_path, "status": status, "chunks": count,
                "document_type": job.document_type,
                "discipline_code": job.discipline_code,
            })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--db", type=Path, default=Path("fos_rag.sqlite3"))
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--type", choices=sorted(DOCUMENT_TYPES))
    parser.add_argument("--discipline-code")
    parser.add_argument("--competencies", default="", help="Коды через запятую")
    parser.add_argument("--manifest", type=Path, help="JSON: путь DOCX -> метаданные")
    parser.add_argument("--curriculum", type=Path, help="JSON учебного плана для сопоставления по имени файла")
    parser.add_argument(
        "--allow-unknown-codes", action="store_true",
        help="Загружать коды из имени файла, отсутствующие в плане; используйте отдельный --db",
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Проверить файлы и метаданные без API и записи в БД")
    parser.add_argument("--report", type=Path, help="Сохранить JSON отчёт")
    args = parser.parse_args()

    try:
        folder = args.folder.resolve()
        files = discover_docx(folder, recursive=not args.no_recursive)
        manifest = load_manifest(args.manifest)
        catalog = load_curriculum(args.curriculum)
        jobs = build_jobs(
            folder, files, manifest, args.type, args.discipline_code,
            tuple(code.strip() for code in args.competencies.split(",") if code.strip()),
            catalog,
            args.allow_unknown_codes,
        )
        if args.dry_run:
            report = {
                "found": len(jobs),
                "files": [
                    {"file": job.relative_path, "document_type": job.document_type,
                     "discipline_code": job.discipline_code,
                     "competencies": list(job.competencies)}
                    for job in jobs
                ],
            }
        else:
            report = ingest_jobs(jobs, args.db, args.embedding_model)
        output = json.dumps(report, ensure_ascii=False, indent=2)
        print(output)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output + "\n", encoding="utf-8")
        return 1 if report.get("failed", 0) else 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
