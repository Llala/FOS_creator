"""Offline tests: synthetic curriculum and sources, no model requests."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from generate_fos import generate_fos, validate_fos


CONTEXT = {
    "curriculum_facts": {"code": "DEMO.01", "name": "Учебная демонстрация",
        "competencies": [{"code": "ДЕМО-1", "description": "Анализировать данные"}],
        "assessments": [{"kind": "pass", "semester": 1}]},
    "sources": [{"ref": "S1", "type": "rpd", "document": "demo.txt",
                 "text": "Тема: анализ учебных данных. Нужно объяснить полученный результат."}],
    "missing_inputs": [],
}
CANDIDATE = {
    "discipline_code": "DEMO.01", "discipline_name": "Учебная демонстрация",
    "tasks": [{"id": "T1", "topic": "Анализ учебных данных",
        "competency_codes": ["ДЕМО-1"], "assessed_skill": "Объяснение результата",
        "stage": "final", "assessment_kind": "pass", "semester": 1,
        "format": "question", "prompt": "Объясните результат анализа учебных данных.",
        "expected_answer": "Описание результата и обоснование вывода.",
        "criteria": [{"description": "Обоснованность вывода", "max_points": 2,
                      "zero_points": "Вывод отсутствует", "full_points": "Вывод обоснован"}],
        "max_points": 2, "evidence": [{"ref": "S1", "quote": "Тема: анализ учебных данных."}]}],
    "missing_data": [],
}


class GenerationTests(unittest.TestCase):
    def test_valid_synthetic_fos(self):
        _, report = validate_fos(CANDIDATE, CONTEXT, min_tasks=1)
        self.assertTrue(report["valid"])

    def test_schema_and_scores(self):
        for field, value in [("expected_answer", "  "), ("max_points", 3),
                             ("max_points", True), ("semester", "1")]:
            with self.subTest(field=field, value=value):
                candidate = copy.deepcopy(CANDIDATE)
                candidate["tasks"][0][field] = value
                _, report = validate_fos(candidate, CONTEXT, 1)
                self.assertFalse(report["valid"])

    def test_domain_mismatches(self):
        changes = [
            lambda c: c.update(discipline_code="WRONG"),
            lambda c: c["tasks"][0].update(competency_codes=["UNKNOWN"]),
            lambda c: c["tasks"][0].update(assessment_kind="exam"),
            lambda c: c["tasks"][0]["evidence"][0].update(ref="S99"),
            lambda c: c["tasks"][0]["evidence"][0].update(quote="Выдуманная цитата из документа"),
            lambda c: c["tasks"].append(copy.deepcopy(c["tasks"][0])),
        ]
        for change in changes:
            candidate = copy.deepcopy(CANDIDATE)
            change(candidate)
            _, report = validate_fos(candidate, CONTEXT, 1)
            self.assertFalse(report["valid"])

    def test_coverage_and_rpd_evidence(self):
        context = copy.deepcopy(CONTEXT)
        context["curriculum_facts"]["competencies"].append({"code": "ДЕМО-2"})
        _, report = validate_fos(CANDIDATE, context, 1)
        self.assertTrue(any("Не покрыты" in e["message"] for e in report["errors"]))
        context = copy.deepcopy(CONTEXT)
        context["sources"].append({"ref": "S2", "type": "methodology", "text": "Пример методических требований"})
        candidate = copy.deepcopy(CANDIDATE)
        candidate["tasks"][0]["evidence"] = [{"ref": "S2", "quote": "Пример методических требований"}]
        _, report = validate_fos(candidate, context, 1)
        self.assertFalse(report["valid"])

    def test_missing_context_rejected_before_model_call(self):
        def forbidden(*args):
            self.fail("Model must not be called")
        with self.assertRaises(ValueError):
            generate_fos({}, generator=forbidden)

    def test_one_repair_with_feedback(self):
        calls = []
        invalid = copy.deepcopy(CANDIDATE)
        invalid["tasks"][0]["max_points"] = 10
        def fake(system, payload, model, max_tokens):
            calls.append(payload)
            return json.dumps(invalid if len(calls) == 1 else CANDIDATE)
        result = generate_fos(CONTEXT, min_tasks=1, generator=fake)
        self.assertEqual(len(calls), 2)
        self.assertIn("errors", calls[1])
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(result["status"], "requires_teacher_review")
        self.assertEqual(result["curriculum_facts"], CONTEXT["curriculum_facts"])

    def test_invalid_json_repair_is_bounded_and_saved(self):
        calls = []
        def fake(*args):
            calls.append(1)
            return "This is not JSON"
        result = generate_fos(CONTEXT, generator=fake)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "validation_failed")
        self.assertEqual(result["attempts"][0]["raw_output"], "This is not JSON")

    def test_validate_cli_needs_no_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            context = root / "context.json"
            fos = root / "fos.json"
            context.write_text(json.dumps(CONTEXT), encoding="utf-8")
            fos.write_text(json.dumps({"fos": CANDIDATE}), encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("generate_fos.py")),
                "validate", str(fos), "--context", str(context), "--min-tasks", "1"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["valid"])


if __name__ == "__main__":
    unittest.main()
