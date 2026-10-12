from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from docx import Document


class LoadRPDTests(unittest.TestCase):
    def test_dry_run_multiple_disciplines_without_api_key(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "rpd"
            folder.mkdir()
            for filename in ("Б1.О.05 Физическая культура РП.docx",
                             "Б1.В.ДВ.01.02 Речевое общение РП.docx"):
                doc = Document()
                doc.add_paragraph("Темы дисциплины")
                doc.save(folder / filename)
            environment = {key: value for key, value in os.environ.items()
                           if key != "OPENAI_API_KEY"}
            command = [sys.executable, "load_rpd_folder.py", str(folder),
                       "--db", str(Path(temp) / "index.sqlite3"), "--dry-run"]
            result = subprocess.run(
                command, cwd=Path(__file__).parent, env=environment,
                capture_output=True, text=True, check=True,
            )
            data = json.loads(result.stdout)
            self.assertEqual(data["found"], 2)
            self.assertEqual({item["discipline_code"] for item in data["files"]},
                             {"Б1.О.05", "Б1.В.ДВ.01.02"})
            self.assertFalse((Path(temp) / "index.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
