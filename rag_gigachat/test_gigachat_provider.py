"""Offline adapter tests using the real gigachat SDK and a mock HTTP transport."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from gigachat import GigaChat
from fos_rag import GigaChatEmbedder, RAGStore, draft_fos, gigachat_client


def mock_client(handler):
    client = GigaChat(access_token="offline-test-token")
    client._client_instance = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://offline.invalid/v1/"
    )
    return client


class ProviderTests(unittest.TestCase):
    def test_embedding_batches_preserve_input_order(self):
        calls = []
        def handler(request):
            payload = json.loads(request.content)
            calls.append(payload)
            return httpx.Response(200, json={
                "object": "list", "model": "EmbeddingsGigaR",
                "data": [
                    {"object": "embedding", "index": i,
                     "embedding": [float(text), 1.0], "usage": {"prompt_tokens": 1}}
                    for i, text in reversed(list(enumerate(payload["input"])))
                ],
            })
        with patch("fos_rag.gigachat_client", return_value=mock_client(handler)):
            embedder = GigaChatEmbedder()
            try:
                vectors = embedder.embed([str(i) for i in range(19)])
                # The client must remain usable across separate documents/queries.
                self.assertEqual(embedder.embed(["20"]), [[20.0, 1.0]])
            finally:
                embedder.close()
        self.assertEqual([len(c["input"]) for c in calls], [16, 3, 1])
        self.assertTrue(all(c["model"] == "EmbeddingsGigaR" for c in calls))
        self.assertEqual([v[0] for v in vectors], list(map(float, range(19))))

    def test_bad_embedding_indices_are_rejected(self):
        def handler(request):
            return httpx.Response(200, json={
                "object": "list", "model": "EmbeddingsGigaR", "data": [
                    {"object": "embedding", "index": 4, "embedding": [1.0],
                     "usage": {"prompt_tokens": 1}},
                ],
            })
        with patch("fos_rag.gigachat_client", return_value=mock_client(handler)):
            embedder = GigaChatEmbedder()
            try:
                with self.assertRaisesRegex(ValueError, "индексы"):
                    embedder.embed(["текст"])
            finally:
                embedder.close()

    def test_generation_request_and_citation_warnings(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={
                "object": "chat.completion", "model": "GigaChat-2", "created": 1,
                "choices": [{"index": 0, "finish_reason": "length",
                    "message": {"role": "assistant", "content": "Задание [S1] [S99]"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })
        context = {"curriculum_facts": {"code": "Б1.О.05"},
                   "sources": [{"ref": "S1", "text": "Критерии"}],
                   "missing_inputs": ["Нет методики"]}
        with patch("fos_rag.gigachat_client", return_value=mock_client(handler)):
            result = draft_fos(context, "GigaChat-2", max_tokens=100)
        self.assertEqual(calls[0]["model"], "GigaChat-2")
        self.assertEqual(calls[0]["max_tokens"], 100)
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertEqual(json.loads(calls[0]["messages"][1]["content"]), context)
        self.assertEqual(result["status"], "requires_teacher_review")
        self.assertTrue(any("S99" in w for w in result["warnings"]))
        self.assertTrue(any("обрезан" in w for w in result["warnings"]))

    def test_api_errors_do_not_include_response_body(self):
        def handler(request):
            return httpx.Response(403, json={"message": "secret-provider-response"})
        with patch("fos_rag.gigachat_client", return_value=mock_client(handler)):
            embedder = GigaChatEmbedder()
            try:
                with self.assertRaises(ValueError) as error:
                    embedder.embed(["текст"])
                self.assertNotIn("secret-provider-response", str(error.exception))
            finally:
                embedder.close()

    def test_openai_index_is_rejected_before_embedding_request(self):
        class OfflineEmbedder:
            model = "text-embedding-3-small"
            def embed(self, texts):
                return [[1.0, 0.0] for text in texts]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "rpd.txt"
            source.write_text("Физическая культура", encoding="utf-8")
            db = Path(temp) / "index.db"
            with RAGStore(db, OfflineEmbedder()) as store:
                store.add_document(source, "rpd", "Б1.О.05")
            embedder = OfflineEmbedder()
            embedder.model = "EmbeddingsGigaR"
            with RAGStore(db, embedder) as store:
                with self.assertRaisesRegex(ValueError, "different embedding models"):
                    store.search("зачёт", "Б1.О.05")

    def test_auth_and_sources_are_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GIGACHAT_CREDENTIALS"):
                gigachat_client()
        with self.assertRaisesRegex(ValueError, "without retrieved sources"):
            draft_fos({}, "GigaChat-2")


if __name__ == "__main__":
    unittest.main()
