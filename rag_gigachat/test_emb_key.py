import os

from openai import OpenAI
from pathlib import Path
import dotenv
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"

# print("ENV path:", ENV_PATH)
# print("ENV exists:", ENV_PATH.exists())


dotenv.load_dotenv(dotenv_path=ENV_PATH)

api_key = os.environ["GIGACHAT_API_2"]
url = "https://foundation-models.api.cloud.ru/v1"

client = OpenAI(
    api_key=api_key,
    base_url=url
)

# response = client.embeddings.create(
#     model="ai-sage/Giga-Embeddings-instruct-480M",
#     input=["Как написать хороший код?"]
# )

response = client.embeddings.create(
    model="ai-sage/Giga-Embeddings-instruct-480M",
    input="Как написать хороший код?"
)

print(len(response.data[0].embedding))

print(response.data[0].embedding)