import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "game_images.csv"

API_URL = "https://boardgamegeek.com/xmlapi2/thing"
BATCH_SIZE = 20       # limite de IDs por requisição da API
PAUSE_SECONDS = 5     # intervalo recomendado pela BGG
MAX_RETRIES = 5
MAX_GAMES = 2000      # jogos mais populares primeiro


def fetch_batch(game_ids: list[int], token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(MAX_RETRIES):
        response = requests.get(
            API_URL,
            params={"id": ",".join(map(str, game_ids))},
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            break

        if response.status_code in (401, 403):
            sys.exit(
                "Token recusado pela BGG (status "
                f"{response.status_code}). Verifique o BGG_TOKEN."
            )

        # 202 = pedido na fila, 429 = limite de requisições
        time.sleep(PAUSE_SECONDS * (attempt + 1))
    else:
        print(f"Falhou para os IDs {game_ids} (status {response.status_code})")
        return []

    root = ET.fromstring(response.content)

    return [
        {
            "ID": int(item.get("id")),
            "image": (item.findtext("image") or "").strip(),
            "thumbnail": (item.findtext("thumbnail") or "").strip(),
        }
        for item in root.findall("item")
    ]


def main() -> None:
    token = os.getenv("BGG_TOKEN")
    if not token:
        sys.exit(
            "Defina a variável de ambiente BGG_TOKEN com o token da sua "
            "aplicação registrada na BGG (uso não comercial)."
        )

    games = pd.read_csv(
        PROCESSED_DIR / "games.csv",
        usecols=["ID", "Users Rated"],
    )

    game_ids = (
        games.sort_values("Users Rated", ascending=False)["ID"]
        .astype(int)
        .head(MAX_GAMES)
        .tolist()
    )

    # Permite retomar de onde parou se o script for interrompido
    fetched = (
        set(pd.read_csv(OUTPUT_PATH)["ID"])
        if OUTPUT_PATH.exists()
        else set()
    )
    pending = [game_id for game_id in game_ids if game_id not in fetched]

    for start in range(0, len(pending), BATCH_SIZE):
        rows = fetch_batch(pending[start:start + BATCH_SIZE], token)

        if rows:
            pd.DataFrame(rows).to_csv(
                OUTPUT_PATH,
                mode="a",
                header=not OUTPUT_PATH.exists(),
                index=False,
            )

        print(f"{min(start + BATCH_SIZE, len(pending))}/{len(pending)}")
        time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()
