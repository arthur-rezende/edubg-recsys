import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.recommenders.item_based import ItemBasedRecommender
from backend.app.recommenders.user_based import UserBasedRecommender

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MIN_RATINGS = 5
TEST_SIZE = 0.2
RELEVANCE_THRESHOLD = 7 # Nota mínima de relevância
TOP_K = 5


def train_test_split_per_user(
    interactions: pd.DataFrame,
    test_size: float = TEST_SIZE,
    min_ratings: int = MIN_RATINGS,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(random_seed)
    train_parts, test_parts = [], []

    for _, grupo in interactions.groupby("user_id"):
        if len(grupo) < min_ratings:
            train_parts.append(grupo)
            continue

        indices = grupo.index.to_numpy().copy()
        rng.shuffle(indices)
        n_teste = max(1, int(len(indices) * test_size))

        test_parts.append(grupo.loc[indices[:n_teste]])
        train_parts.append(grupo.loc[indices[n_teste:]])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = (
        pd.concat(test_parts).reset_index(drop=True)
        if test_parts
        else pd.DataFrame(columns=interactions.columns)
    )
    return train_df, test_df


def relevant_games(test_df: pd.DataFrame, user_id, threshold: float = RELEVANCE_THRESHOLD) -> set:
    avaliacoes_usuario = test_df[test_df["user_id"] == user_id]
    relevantes = avaliacoes_usuario[avaliacoes_usuario["rating"] >= threshold]
    return set(relevantes["game_id"])


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not recommended:
        return 0.0
    top_k = recommended[:k]
    acertos = sum(1 for game_id in top_k if game_id in relevant)
    return acertos / len(top_k)


def recall_at_k(recommended: list, relevant: set, k: int):
    if not relevant:
        return None  # sem itens relevantes no teste
    top_k = recommended[:k]
    acertos = sum(1 for game_id in top_k if game_id in relevant)
    return acertos / len(relevant)


def hit_rate_at_k(recommended: list, relevant: set, k: int):
    if not relevant:
        return None
    top_k = recommended[:k]
    return 1.0 if any(game_id in relevant for game_id in top_k) else 0.0


def evaluate_model(model, test_df: pd.DataFrame, top_k: int = TOP_K) -> dict:
    precisions, recalls, hit_rates, tempos = [], [], [], []

    for user_id in test_df["user_id"].unique():
        if user_id not in model.user_item_matrix.index:
            continue  # usuário sem histórico de treino

        relevantes = relevant_games(test_df, user_id)

        inicio = time.perf_counter()
        recomendacoes = model.recommend(user_id, top_k=top_k)
        tempos.append(time.perf_counter() - inicio)

        recomendados_ids = [item["game_id"] for item in recomendacoes]

        precisions.append(precision_at_k(recomendados_ids, relevantes, top_k))

        recall = recall_at_k(recomendados_ids, relevantes, top_k)
        if recall is not None:
            recalls.append(recall)

        hit = hit_rate_at_k(recomendados_ids, relevantes, top_k)
        if hit is not None:
            hit_rates.append(hit)

    return {
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "hit_rate@k": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "tempo_medio_s": float(np.mean(tempos)) if tempos else 0.0,
        "usuarios_avaliados": len(precisions),
    }


def main():
    logging.basicConfig(level=logging.INFO)

    interactions_path = Path(__file__).resolve().parent / "data" / "interactions.csv"
    interactions = pd.read_csv(interactions_path)

    train_df, test_df = train_test_split_per_user(interactions)
    logger.info("Treino: %d interações | Teste: %d interações", len(train_df), len(test_df))

    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    try:
        train_df.to_csv(tmp_file.name, index=False)
        tmp_file.close()

        item_based = ItemBasedRecommender(tmp_file.name)
        user_based = UserBasedRecommender(tmp_file.name)

        resultados = {
            "Item-Based": evaluate_model(item_based, test_df),
            "User-Based": evaluate_model(user_based, test_df),
        }
    finally:
        os.unlink(tmp_file.name)

    print(f"\n{'Métrica':<20}{'Item-Based':<15}{'User-Based':<15}")
    for metrica in ["precision@k", "recall@k", "hit_rate@k", "tempo_medio_s", "usuarios_avaliados"]:
        print(
            f"{metrica:<20}"
            f"{resultados['Item-Based'][metrica]:<15.4f}"
            f"{resultados['User-Based'][metrica]:<15.4f}"
        )


if __name__ == "__main__":
    main()