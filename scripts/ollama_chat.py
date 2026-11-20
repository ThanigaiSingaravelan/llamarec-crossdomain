"""Minimal chat loop against a local Ollama model (connection sanity check)."""
import os
import sys

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LLAMAREC_MODEL", "llama3:8b")


def check_ollama_connection() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama API connected successfully!")
            return True
        print("Ollama API connection failed!")
        return False
    except requests.RequestException:
        print("Cannot connect to Ollama API. Make sure Ollama is running!")
        return False


def ai_generate(prompt: str) -> str:
    response = requests.post(f"{OLLAMA_URL}/api/generate",
                             json={"model": MODEL, "prompt": prompt, "stream": False},
                             timeout=120)
    return response.json()["response"]


def main() -> int:
    if not check_ollama_connection():
        return 1
    print(f"Chatting with {MODEL} (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        print("AI:", ai_generate(user_input))
    return 0


if __name__ == "__main__":
    sys.exit(main())
