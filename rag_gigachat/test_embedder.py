import os
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

credentials = os.getenv("GIGACHAT_API_KEY")

with GigaChat(
    credentials=credentials,
    scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
    verify_ssl_certs=False,
) as giga:
    try:
        result = giga.embeddings(
            texts=["Тестовый текст"],
            model="EmbeddingsGigaR",
        )

        print("OK")
        print("dimension:", len(result.data[0].embedding))

    except Exception as exc:
        print(type(exc).__name__)
        print(repr(exc))