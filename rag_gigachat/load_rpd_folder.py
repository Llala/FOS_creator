"""Load every DOCX RPD from a folder into the FOS RAG index.

Examples from the rag/ directory:
    python load_rpd_folder.py ./rpd --dry-run
    python load_rpd_folder.py ./rpd --db rpd_index.sqlite3
    python load_rpd_folder.py ./rpd --manifest rpd_map.json --dry-run

The discipline code is taken from each filename (for example Б1.О.05).
Documents without a code require a manifest entry. This script deliberately
does not attach an unrelated curriculum's competencies to the documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ingest_docx_folder import build_jobs, discover_docx, ingest_jobs, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Папка с РПД в формате DOCX")
    parser.add_argument("--db", type=Path, default=Path("rpd_gigachat.sqlite3"))
    parser.add_argument("--manifest", type=Path,
                        help="JSON с кодами для файлов без кода в названии")
    parser.add_argument("--embedding-model", default="EmbeddingsGigaR")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Проверить имена и коды без вызова API")
    parser.add_argument("--report", type=Path, help="Записать отчёт в JSON")
    args = parser.parse_args()

    try:
        folder = args.folder.resolve()
        files = discover_docx(folder, recursive=not args.no_recursive)
        if not files:
            raise ValueError(f"В {folder} не найдены файлы .docx")
        jobs = build_jobs(
            folder=folder,
            files=files,
            manifest=load_manifest(args.manifest),
            default_type="rpd",
            default_code=None,
            default_competencies=(),
            catalog={},
            allow_unknown_codes=True,
        )
        if any(job.document_type != "rpd" for job in jobs):
            raise ValueError("Эта команда загружает только rpd; проверьте document_type в манифесте")
        if args.dry_run:
            report = {
                "found": len(jobs),
                "database": str(args.db.resolve()),
                "files": [
                    {"file": job.relative_path,
                     "discipline_code": job.discipline_code,
                     "competencies": list(job.competencies)}
                    for job in jobs
                ],
            }
        else:
            report = ingest_jobs(jobs, args.db, args.embedding_model)
            report["database"] = str(args.db.resolve())
        output = json.dumps(report, ensure_ascii=False, indent=2)
        print(output)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output + "\n", encoding="utf-8")
        return 1 if report.get("failed", 0) else 0
    except (OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
