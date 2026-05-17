import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "flask", "flask-socketio", "flask-cors", "pyngrok", "simple-websocket",
    "pymongo", "dnspython", "python-dotenv", "cloudscraper",
    "beautifulsoup4", "pandas", "emoji", "nltk", "tqdm",
    "accelerate", "sentencepiece", "bitsandbytes",
])

from google.colab import userdata

SECRET_KEYS = [
    "MONGO_URI",
    "MONGO_DB_NAME",
    "SERPER_API_KEYS",
    "NGROK_AUTH_TOKEN",
    "MONGO_SOURCE_COLLECTION",
    "SENTIMENT_BATCH_SIZE",
    "SENTIMENT_MAX_TEXT_LEN",
]

for key in SECRET_KEYS:
    try:
        value = userdata.get(key)
        if value:
            os.environ[key] = value
            print(f"  ✓ Loaded secret: {key}")
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        pass

for required in ["MONGO_URI", "NGROK_AUTH_TOKEN"]:
    if not os.environ.get(required):
        raise RuntimeError(f"Missing required Colab Secret: {required}")

os.environ.setdefault("MONGO_DB_NAME", "COM6064")
os.environ.setdefault("MONGO_SOURCE_COLLECTION", "posts_processed")
os.environ.setdefault("SENTIMENT_BATCH_SIZE", "16")
os.environ.setdefault("SENTIMENT_MAX_TEXT_LEN", "512")

REPO_URL = "https://github.com/sobdurated/com6064-pipeline.git"
PIPELINE_DIR = "/content/pipeline"

if not os.path.exists(PIPELINE_DIR):
    subprocess.check_call(["git", "clone", REPO_URL, PIPELINE_DIR])
else:
    subprocess.check_call(["git", "-C", PIPELINE_DIR, "pull"])

for d in ["output", "state", "pipeline_reports"]:
    os.makedirs(os.path.join(PIPELINE_DIR, d), exist_ok=True)

print(f"Pipeline dir: {PIPELINE_DIR}")
print(f"Contents: {os.listdir(PIPELINE_DIR)}")

sys.path.insert(0, PIPELINE_DIR)
from server import start_server
import time

server_thread = start_server(
    pipeline_dir=PIPELINE_DIR,
    port=5000,
    ngrok_token=os.environ["NGROK_AUTH_TOKEN"],
    host="127.0.0.1",
    background=True,
)

print("\nServer is running in the background!")

try:
    while True:
        time.sleep(30)
        print(f"[Heartbeat] {time.strftime('%Y-%m-%d %H:%M:%S')}")
except KeyboardInterrupt:
    print("\nStopped.")
