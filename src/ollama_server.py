import subprocess
import time
import requests

from src.config import OLLAMA_HOST, EMBEDDING_MODEL


def ollama_is_ready():
    try:
        response = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def start_ollama_server():
    if ollama_is_ready():
        print("Ollama server is already running.")
        return

    print("Starting Ollama server...")

    try:
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama is not installed or not on PATH.") from exc

    for _ in range(30):
        if ollama_is_ready():
            print("Ollama server is ready.")
            return

        if ollama_process.poll() is not None:
            raise RuntimeError("Ollama server exited unexpectedly.")

        time.sleep(1)

    ollama_process.terminate()
    raise RuntimeError("Ollama server failed to start.")


def load_models(model_names=None):
    if model_names is None:
        model_names = (EMBEDDING_MODEL,)

    if not ollama_is_ready():
        raise RuntimeError("Ollama server is not running.")

    response = requests.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()

    installed_models = {model["name"] for model in response.json().get("models", [])}

    for model_name in model_names:
        normalized_name = model_name if ":" in model_name else f"{model_name}:latest"

        if normalized_name in installed_models:
            print(f"{model_name} is already installed.")
            continue

        print(f"Downloading {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"{model_name} downloaded successfully.")

    print("Required Ollama models are ready.")


def setup_ollama(model_names=None):
    start_ollama_server()
    load_models(model_names)
