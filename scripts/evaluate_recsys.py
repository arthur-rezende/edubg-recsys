import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

interactions_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interactions.csv"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.recommenders.item_based import ItemBasedRecommender
from backend.app.recommenders.user_based import UserBasedRecommender


logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MIN_RATINGS = 5
TEST_SIZE = 0.2
RELEVANCE_THRESHOLD = 7
TOP_K = 5
EVAL_USERS = 5000


def train_test_split_per_user(
    interactions: pd.DataFrame,
    test_size: float = TEST_SIZE,
    min_ratings: int = MIN_RATINGS,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(
        random_seed
    )

    user_counts = (
        interactions
        .groupby("user_id")
        .size()
    )

    eligible_users = user_counts[
        user_counts >= min_ratings
    ].index

    eligible_mask = interactions[
        "user_id"
    ].isin(
        eligible_users
    )

    eligible = interactions.loc[
        eligible_mask
    ].copy()

    ineligible = interactions.loc[
        ~eligible_mask
    ].copy()

    eligible["_random"] = rng.random(
        len(eligible)
    )

    eligible["_rank"] = (
        eligible
        .groupby("user_id")["_random"]
        .rank(
            method="first",
            ascending=True
        )
    )

    test_counts = (
        eligible
        .groupby("user_id")
        .size()
        .mul(test_size)
        .astype(int)
        .clip(lower=1)
    )

    eligible = eligible.merge(
        test_counts.rename("_test_count"),
        left_on="user_id",
        right_index=True,
        how="left"
    )

    test_mask = (
        eligible["_rank"]
        <= eligible["_test_count"]
    )

    columns_to_remove = [
        "_random",
        "_rank",
        "_test_count"
    ]

    test_df = (
        eligible.loc[test_mask]
        .drop(columns=columns_to_remove)
        .reset_index(drop=True)
    )

    train_eligible = (
        eligible.loc[~test_mask]
        .drop(columns=columns_to_remove)
    )

    train_df = pd.concat(
        [
            train_eligible,
            ineligible
        ],
        ignore_index=True
    )

    return (
        train_df.reset_index(drop=True),
        test_df
    )


def sample_evaluation_users(
    test_df: pd.DataFrame,
    sample_size: int = EVAL_USERS,
    random_seed: int = RANDOM_SEED
) -> np.ndarray:

    users = (
        test_df["user_id"]
        .drop_duplicates()
        .to_numpy()
    )

    if len(users) <= sample_size:
        return users

    rng = np.random.default_rng(
        random_seed
    )

    return rng.choice(
        users,
        size=sample_size,
        replace=False
    )


def build_relevant_games(
    test_df: pd.DataFrame,
    users: np.ndarray,
    threshold: float = RELEVANCE_THRESHOLD
) -> dict:

    relevant_df = test_df[
        test_df["user_id"].isin(users)
        & (
            test_df["rating"]
            >= threshold
        )
    ]

    return (
        relevant_df
        .groupby("user_id")["game_id"]
        .agg(set)
        .to_dict()
    )


def precision_at_k(
    recommended: list,
    relevant: set,
    k: int
) -> float:

    if not recommended:
        return 0.0

    top_k = recommended[:k]

    hits = sum(
        game_id in relevant
        for game_id in top_k
    )

    return hits / len(top_k)


def recall_at_k(
    recommended: list,
    relevant: set,
    k: int
):

    if not relevant:
        return None

    top_k = recommended[:k]

    hits = sum(
        game_id in relevant
        for game_id in top_k
    )

    return hits / len(relevant)


def hit_rate_at_k(
    recommended: list,
    relevant: set,
    k: int
):

    if not relevant:
        return None

    top_k = recommended[:k]

    return float(
        any(
            game_id in relevant
            for game_id in top_k
        )
    )


def calculate_metrics(
    recommendations_by_user: dict,
    relevant_by_user: dict,
    users: np.ndarray,
    top_k: int
) -> dict:

    precision_sum = 0.0
    recall_sum = 0.0
    hit_rate_sum = 0.0

    precision_count = 0
    recall_count = 0
    hit_rate_count = 0

    for user_id in users:

        recommendations = (
            recommendations_by_user
            .get(user_id, [])
        )

        recommended_ids = [
            item["game_id"]
            for item in recommendations
        ]

        relevant = relevant_by_user.get(
            user_id,
            set()
        )

        precision = precision_at_k(
            recommended_ids,
            relevant,
            top_k
        )

        precision_sum += precision
        precision_count += 1

        recall = recall_at_k(
            recommended_ids,
            relevant,
            top_k
        )

        if recall is not None:
            recall_sum += recall
            recall_count += 1

        hit_rate = hit_rate_at_k(
            recommended_ids,
            relevant,
            top_k
        )

        if hit_rate is not None:
            hit_rate_sum += hit_rate
            hit_rate_count += 1

    return {
        "precision@k": (
            precision_sum / precision_count
            if precision_count
            else 0.0
        ),
        "recall@k": (
            recall_sum / recall_count
            if recall_count
            else 0.0
        ),
        "hit_rate@k": (
            hit_rate_sum / hit_rate_count
            if hit_rate_count
            else 0.0
        ),
        "usuarios_avaliados": len(users)
    }


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    users: np.ndarray,
    top_k: int = TOP_K
) -> dict:

    relevant_by_user = build_relevant_games(
        test_df,
        users
    )

    start = time.perf_counter()

    if hasattr(model, "recommend_batch"):

        recommendations_by_user = (
            model.recommend_batch(
                users.tolist(),
                top_k=top_k
            )
        )

    else:

        recommendations_by_user = {}

        for user_id in users:

            recommendations_by_user[user_id] = (
                model.recommend(
                    user_id,
                    top_k=top_k
                )
            )

    total_time = (
        time.perf_counter()
        - start
    )

    metrics = calculate_metrics(
        recommendations_by_user,
        relevant_by_user,
        users,
        top_k
    )

    metrics["tempo_total_s"] = total_time

    metrics["tempo_medio_s"] = (
        total_time / len(users)
        if len(users)
        else 0.0
    )

    return metrics


def main():

    logging.basicConfig(
        level=logging.INFO
    )

    logger.info(
        "Carregando interações..."
    )

    interactions = pd.read_csv(
        interactions_path
    )

    logger.info(
        "Interações carregadas: %d",
        len(interactions)
    )

    train_df, test_df = (
        train_test_split_per_user(
            interactions
        )
    )

    logger.info(
        "Treino: %d interações | Teste: %d interações",
        len(train_df),
        len(test_df)
    )

    evaluation_users = (
        sample_evaluation_users(
            test_df
        )
    )

    logger.info(
        "Usuários disponíveis para avaliação: %d",
        test_df["user_id"].nunique()
    )

    logger.info(
        "Usuários selecionados para avaliação: %d",
        len(evaluation_users)
    )

    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False
    )

    try:

        train_df.to_csv(
            tmp_file.name,
            index=False
        )

        tmp_file.close()

        logger.info(
            "Construindo Item-Based..."
        )

        item_based = ItemBasedRecommender(
            tmp_file.name
        )

        logger.info(
            "Construindo User-Based..."
        )

        user_based = UserBasedRecommender(
            tmp_file.name
        )

        logger.info(
            "Avaliando Item-Based..."
        )

        item_results = evaluate_model(
            item_based,
            test_df,
            evaluation_users
        )

        logger.info(
            "Item-Based concluído em %.2f segundos",
            item_results["tempo_total_s"]
        )

        logger.info(
            "Avaliando User-Based..."
        )

        user_results = evaluate_model(
            user_based,
            test_df,
            evaluation_users
        )

        logger.info(
            "User-Based concluído em %.2f segundos",
            user_results["tempo_total_s"]
        )

        resultados = {
            "Item-Based": item_results,
            "User-Based": user_results,
        }

    finally:

        if not tmp_file.closed:
            tmp_file.close()

        if os.path.exists(
            tmp_file.name
        ):
            os.unlink(
                tmp_file.name
            )

    print()

    print(
        f"{'Métrica':<20}"
        f"{'Item-Based':<15}"
        f"{'User-Based':<15}"
    )

    print("-" * 50)

    for metrica in [
        "precision@k",
        "recall@k",
        "hit_rate@k",
        "tempo_total_s",
        "tempo_medio_s",
        "usuarios_avaliados",
    ]:

        print(
            f"{metrica:<20}"
            f"{resultados['Item-Based'][metrica]:<15.4f}"
            f"{resultados['User-Based'][metrica]:<15.4f}"
        )


if __name__ == "__main__":
    main()