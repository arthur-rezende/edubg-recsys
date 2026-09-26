import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .collaborative_base import CollaborativeFilteringBase

logger = logging.getLogger(__name__)

class ItemBasedRecommender(CollaborativeFilteringBase):

    def __init__(self, interactions_path: str):
        super().__init__(interactions_path)
        self.items_similarity = self._build_item_similarity()

    def _build_item_similarity(self) -> pd.DataFrame:

        matrix_preenchida = self.user_item_matrix.fillna(0)

        similarity = cosine_similarity(matrix_preenchida.T)
        matrix = pd.DataFrame(
            similarity,
            index=matrix_preenchida.columns,
            columns=matrix_preenchida.columns,
        )

        logger.info(
            "Item_Based: %d usuários, %d jogos.",
            self.user_item_matrix.shape[0],
            self.user_item_matrix.shape[1]
        )
        return matrix

    def recommend(self, user_id: str, top_k: int = 5) -> list[dict]:
        known_games = self.get_known_games(user_id)
        if not known_games:
            logger.info("Usuário %s sem interações (cold start).", user_id)
            return []

        known_ratings = self.get_user_ratings(user_id).dropna()

        sim_conhecidos = self.item_similarity.loc[:, list(known_games)]
        numerador = sim_conhecidos.mul(known_ratings, axis=1).sum(axis=1)
        denominador = sim_conhecidos.abs().sum(axis=1)

        scores = numerador / denominador.replace(0, np.nan)

        scores = scores.drop(index=known_games, errors="ignore").dropna()

        top = scores.sort_values(ascending=False).head(top_k)

        return [
            {"game_id": int(game_id), "score": float(score)}
            for game_id, score in top.items()
        ]