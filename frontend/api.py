import os

import requests


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def api_is_up() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def get_recommendations(payload: dict) -> list[dict]:
    # POST /recommend -> {"recomendacoes"}
    resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["recomendacoes"]


def send_evaluation(payload: dict) -> dict:
    # POST /avaliacoes -> salva no SQLite via backend
    resp = requests.post(f"{API_URL}/avaliacoes", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()
