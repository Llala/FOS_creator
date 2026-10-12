"""Offline regression checks; no API calls or model downloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fos_rag import (
    RAGStore, chunk_passages, curriculum_discipline, index_status,
    make_context, Passage,
)


class FakeEmbedder:
    model = "fake-v1"

    def embed(self, texts):
        vectors = []
        for text in texts:
            tokens = text.lower()
            vectors.append([
                float(tokens.count("маркетинг") + tokens.count("рынок")),
                float(tokens.count("спонсор")),
                float(tokens.count("критерии")),
                0.01,
            ])
        return vectors


class RAGTests(unittest.TestCase):
    def test_chunker_preserves_pages_and_long_paragraph(self):
        parts = chunk_passages([
            Passage("А" * 250, "стр. 1", ""),
            Passage("Б" * 250, "стр. 2", ""),
            Passage("В" * 800, "стр. 2", ""),
        ], max_chars=300)
        self.assertEqual("".join(p.text for p in parts), "А" * 250 + "Б" * 250 + "В" * 800)
        self.assertEqual(parts[0].location, "стр. 1")
        self.assertTrue(all(p.location == "стр. 2" for p in parts[1:]))

    def test_filters_reindexing_and_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rpd = root / "rpd.md"
            other = root / "other.md"
            method = root / "methodology.txt"
            rpd.write_text("Маркетинг: анализ рынка спортивных услуг и рисков.", encoding="utf-8")
            other.write_text("Спонсорство: договоры партнерства.", encoding="utf-8")
            method.write_text("Критерии оценки: полнота анализа и обоснование вывода.", encoding="utf-8")
            with RAGStore(root / "index.db", FakeEmbedder()) as rag:
                self.assertGreater(rag.add_document(rpd, "rpd", "Б1.В.07", ["ПК-7"]), 0)
                self.assertEqual(rag.add_document(rpd, "rpd", "Б1.В.07", ["ПК-7"]), 0)
                rag.add_document(other, "rpd", "Б1.В.25", ["ПК-6"])
                rag.add_document(method, "methodology")
                hits = rag.search("маркетинг рынок", "Б1.В.07", "ПК-7")
                self.assertTrue(hits)
                self.assertEqual(hits[0].document_name, "rpd.md")
                self.assertEqual({h.document_name for h in hits}, {"rpd.md", "methodology.txt"})
                context = make_context({"code": "Б1.В.07", "name": "Маркетинг"}, hits)
                self.assertEqual(context["sources"][0]["ref"], "S1")
                self.assertTrue(context["sources"][0]["source_id"])
                self.assertEqual(context["missing_inputs"], [])

                rpd.write_text("Маркетинг: новая редакция программы.", encoding="utf-8")
                self.assertGreater(rag.add_document(rpd, "rpd", "Б1.В.07", ["ПК-7"]), 0)
                updated = rag.search("маркетинг", "Б1.В.07")
                self.assertTrue(any("новая редакция" in h.text for h in updated))
                self.assertFalse(any("анализ рынка спортивных услуг" in h.text for h in updated))

    def test_duplicate_curriculum_code_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps({"components": [
                {"code": "ФТД.01", "source_row": 104},
                {"code": "ФТД.01", "source_row": 107},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "found 2"):
                curriculum_discipline(path, "ФТД.01")

    def test_status_is_read_only_and_shows_eligible_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = root / "wrong.db"
            with self.assertRaisesRegex(ValueError, "Индекс не найден"):
                index_status(missing)
            self.assertFalse(missing.exists())
            source = root / "rpd.txt"
            source.write_text("Зачёт по дисциплине.", encoding="utf-8")
            db = root / "index.db"
            with RAGStore(db, FakeEmbedder()) as rag:
                rag.add_document(source, "rpd", "Б1.О.05")
            status = index_status(db, "Б1.О.05")
            self.assertEqual(status["documents_total"], 1)
            self.assertEqual(status["eligible_documents"][0]["name"], "rpd.txt")
            other = index_status(db, "Б1.В.07")
            self.assertEqual(other["eligible_documents"], [])
            env = {key: value for key, value in os.environ.items()
                   if key != "OPENAI_API_KEY"}
            cli = subprocess.run(
                [sys.executable, "fos_rag.py", "--db", str(db), "status",
                 "--discipline-code", "Б1.О.05"],
                cwd=Path(__file__).parent, env=env,
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(cli.stdout)["documents_total"], 1)


if __name__ == "__main__":
    unittest.main()
