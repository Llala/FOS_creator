import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
import dotenv
import os
import time
import httpx
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from fos_rag import RAGStore, CloudRuEmbedder


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"

# print("ENV path:", ENV_PATH)
# print("ENV exists:", ENV_PATH.exists())


dotenv.load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Задайте OPENAI_API_KEY — ключ OpenAI API")

http_client = httpx.AsyncClient(
    proxy="http://127.0.0.1:10809",
    timeout=httpx.Timeout(45.0, connect=15.0),
    trust_env=False
    )

async def measure(label, metric, **kwargs):
    started = time.monotonic()
    print(f"Начало: {label}", flush=True)

    try:
        result = await asyncio.wait_for(
            metric.ascore(**kwargs),
            timeout=360,
        )
    except TimeoutError:
        print(f"Таймаут: {label}, ожидание 360 секунд", flush=True)
        raise
    except Exception as exc:
        # Не печатаем ключи, заголовки и содержимое запросов.
        print(
            f"Ошибка: {label}: {type(exc).__name__}; "
            f"HTTP={getattr(exc, 'status_code', None)}",
            flush=True,
        )
        raise

    elapsed = time.monotonic() - started
    print(
        f"Готово: {label}: {result.value}, {elapsed:.1f} сек.",
        flush=True,
    )
    return result

async def main():
    questions_path = BASE_DIR / "rpd_questions.json"
    fos_path = BASE_DIR / "physical_culture_fos.json"
    output_path = BASE_DIR / "ragas_report.json"

    questions = read_json(questions_path)
    #questions = questions[:1]
    result = read_json(fos_path)

    if not isinstance(questions, list) or not questions:
        raise ValueError("rpd_questions.json должен содержать непустой список")

    # Используем проверенные данные и исходный контекст.

    retrieval_scores = []
    task_scores = []

    async with AsyncOpenAI(http_client=http_client,timeout=45, max_retries=1) as client:
        judge = llm_factory("gpt-4.1-mini", client=client)

        precision_metric = ContextPrecision(llm=judge)
        recall_metric = ContextRecall(llm=judge)
        faithfulness_metric = Faithfulness(llm=judge)

        # Оценка поиска: отдельный контекст для каждого вопроса.
        for number, question in enumerate(questions, start=1):
            query = question["query"]
            reference = question["reference"]

            # Относительный путь разрешается относительно файла вопросов.
            context_path = Path(question["context_file"])
            if not context_path.is_absolute():
                context_path = questions_path.parent / context_path
            # chunks = read_json(context_path)

            # if not isinstance(chunks, list):
            #     raise ValueError("Ожидается JSON со списком чанков")

            # for i, chunk in enumerate(chunks, start=1):
            #     print(f"\nChunk {i}")
            #     print("ID:", chunk.get("id"))
            #     print("Score:", chunk.get("score"))
            #     print("Text:", chunk.get("text", "")[:500])

            # retrieved_contexts = [chunk["text"] for chunk in chunks]
            # # Только для проверки работоспособности метрики.
            # print(
            #     f"Вопрос {number}/{len(questions)}: {query}\n"
            #     f"Чанков: {len(retrieved_contexts)}, "
            #     f"символов: {sum(len(text) for text in retrieved_contexts)}",
            #     flush=True,
            # )

            # Получаем результаты настоящего поиска
            embedder = CloudRuEmbedder("ai-sage/Giga-Embeddings-instruct-480M" )
            with RAGStore("rpd_gigachat.sqlite3", embedder) as store:
                results = store.search(
                    query=query,
                    discipline_code="Б1.О.05",
                )

                retrieved_contexts = [
                    result.text
                    for result in results
                ]

                precision = await measure(
                                    f"Вопрос {number}: ContextPrecision",
                                    precision_metric,
                                    user_input=query,
                                    retrieved_contexts=retrieved_contexts,
                                    reference=reference,
                                )

                recall = await measure(
                                f"Вопрос {number}: ContextRecall",
                                recall_metric,
                                user_input=query,
                                retrieved_contexts=retrieved_contexts,
                                reference=reference,
                            )

                retrieval_scores.append({
                    "id": question.get("id", f"question-{number}"),
                    "query": query,
                    "context_file": str(context_path),
                    "context_precision": precision.value,
                    "context_recall": recall.value,
                })

        # Оценка ответов ФОС: контекст, доступный генератору.
        generation_contexts = [
            source["text"] for source in result["sources"]
        ]
        generation_contexts.append(
            json.dumps(result["curriculum_facts"], ensure_ascii=False)
        )

        for task in result["fos"]["tasks"]:
            score = await measure(
                f"Задание {task['id']}: Faithfulness",
                faithfulness_metric,
                user_input=task["prompt"],
                response=task["expected_answer"],
                retrieved_contexts=generation_contexts,
            )

            task_scores.append({
                "task_id": task["id"],
                "expected_answer_faithfulness": score.value,
            })

    report = {
        "judge_model": "gpt-4.1-mini",
        "questions_file": str(questions_path),
        "fos_file": str(fos_path),
        "retrieval_scores": retrieval_scores,
        "task_scores": task_scores,
    }

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"Отчёт сохранён: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())