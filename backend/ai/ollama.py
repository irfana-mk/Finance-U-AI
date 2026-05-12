import requests
import json

def ask_ollama(prompt: str, model="llama3") -> str:
    """Send a prompt to local Ollama API."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        raise Exception(f"Ollama failed: {e}")
