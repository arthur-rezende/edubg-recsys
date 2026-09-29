import logging
import sqlite3

from contextlib import contextmanager
from datetime import datetime
from datetime import timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "app.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS avaliacoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    game_id INTEGER NOT NULL,
    rating REAL NOT NULL,
    student_age INTEGER,
    student_count INTEGER,
    class_duration INTEGER,
    pedagogical_objective TEXT,
    context TEXT,
    created_at TEXT NOT NULL
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_avaliacoes_user ON avaliacoes(user_id);
"""

INSERT_SQL = """
INSERT INTO avaliacoes (
    user_id, game_id, rating, student_age, student_count,
    class_duration, pedagogical_objective, context, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    # Iniciar db, cria a tabela 'avaliacoes'
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.execute(CREATE_INDEX_SQL)
    logger.info("Banco de dados iniciado! Path: %s", DB_PATH)


def insert_avaliacao(
        user_id: str,
        game_id: int,
        rating: float,
        student_age: int | None = None,
        student_count: int | None = None,
        class_duration: int | None = None,
        pedagogical_objective: str | None = None,
        context: str | None = None,
) -> int:

    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            INSERT_SQL,
            (
                user_id, game_id, rating, student_age, student_count,
                class_duration, pedagogical_objective, context, created_at,
            ),
        )
        return cursor.lastrowid


def get_avaliacao_by_id(avaliacao_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM avaliacoes WHERE id = ?", (avaliacao_id,)
        ).fetchone()
        return dict(row) if row else None 


def get_avaliacoes(user_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM avaliacoes"
    params: tuple = ()
    if user_id is not None:
        query += " WHERE user_id = ?"
        params = (user_id,)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

def get_rated_game_id(user_id: str) -> set[int]:
    # Jogos que o professor já avaliou
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT game_id FROM avaliacoes WHERE user_id = ?", (user_id,),
        ).fetchall()
        return {int(row["game_id"]) for row in rows}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    novo_id = insert_avaliacao(
        user_id="usuario - 01",
        game_id=213788,
        rating=8.5,
        student_age=14,
        student_count=23,
        class_duration=50,
        pedagogical_objective="Desenvolver uso de lógica de programação",
        context="Turma do ensino fundamental",
    )
    print("Avaliação inserida!", get_avaliacao_by_id(novo_id))