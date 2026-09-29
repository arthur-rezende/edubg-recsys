from datetime import datetime, timezone
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "backend" / "data" / "app.db"
INTERACTIONS_PATH = PROJECT_ROOT / "backend" / "data" / "processed" / "interactions.csv"


CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS usuarios (
    user_id TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    rating_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _password_hash(password: str) -> str:
    return sha256(f"triforce-table:{password}".encode("utf-8")).hexdigest()


def _connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _select_demo_user() -> tuple[str, int]:
    interactions = pd.read_csv(INTERACTIONS_PATH, usecols=["user_id"])
    counts = interactions["user_id"].value_counts()
    eligible = counts[counts > 20]
    if eligible.empty:
        raise ValueError("Nenhum usuário com mais de 20 avaliações foi encontrado.")
    return str(eligible.index[0]), int(eligible.iloc[0])


def ensure_demo_user() -> dict:
    with _connection() as connection:
        connection.execute(CREATE_USERS_SQL)
        existing = connection.execute(
            "SELECT user_id, rating_count FROM usuarios ORDER BY rating_count DESC LIMIT 1"
        ).fetchone()
        if existing:
            return dict(existing)

        user_id, rating_count = _select_demo_user()
        connection.execute(
            "INSERT INTO usuarios (user_id, password_hash, rating_count, created_at) VALUES (?, ?, ?, ?)",
            (
                user_id,
                _password_hash("12345"),
                rating_count,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return {"user_id": user_id, "rating_count": rating_count}


def authenticate(user_id: str, password: str) -> bool:
    with _connection() as connection:
        user = connection.execute(
            "SELECT password_hash FROM usuarios WHERE user_id = ?",
            (user_id.strip(),),
        ).fetchone()
    return bool(user and compare_digest(user["password_hash"], _password_hash(password)))
