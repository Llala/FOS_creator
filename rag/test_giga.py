
import os
from dotenv import load_dotenv
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(".env")

url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"

print("ENV path:", ENV_PATH)
print("ENV exists:", ENV_PATH.exists())

load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("GIGACHAT_API_KEY"):
        raise ValueError("Set GIGACHAT_API_KEY to index or search documents")

payload={
  f'scope': f"{os.getenv('GIGACHAT_SCOPE')}"
}
api_key=os.getenv("GIGACHAT_API_KEY")
api_key = os.getenv("GIGACHAT_API_KEY", "").strip()

print(repr(api_key[:10]))
print(len(api_key))
headers = {
  'Content-Type': 'application/x-www-form-urlencoded',
  'Accept': 'application/json',
  'RqUID': '4550abfb-e00d-469f-98e4-1f752b407ed0',
  'Authorization': f"Basic {api_key}"
}
key = os.getenv("GIGACHAT_API_KEY")

print("Key exists:", key is not None)
print("Key length:", len(key) if key else 0)
print("Key prefix:", key[:10] if key else None)

response = requests.request("POST", url, headers=headers, data=payload, verify=False,)

# print(response.text)
access_token = response.json()["access_token"]
