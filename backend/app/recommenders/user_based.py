import logging

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .collaborative_base import CollaborativeFilteringBase

logger = logging.getLogger(__name__)


class UserBasedRecommender(CollaborativeFilteringBase):

    def __init__(self, interactions_path: str, k_neighbors: int = 20):
        super().__init__(interactions_path)
        self.k_neighbors = k_neighbors
        self.user_similarity = self._build_user_similarity()

    def _build_user_similarity(self) -> pd.DataFrame:
        matrix_preenchida = self.user_item_matrix.fillna(0)

        similarity = cosine_similarity(matrix_preenchida)
        matrix = pd.DataFrame(
            similarity,
            index=matrix_preenchida.index,
            columns=matrix_preenchida.index,
        )

        logger.info(
            "User-Based ajustado: %d usuários, %d jogos.",
            self.user_item_matrix.shape[0],
            self.user_item_matrix.shape[1],
        )
        return matrix

    def recommend(self, user_id: str, top_k: int = 5) -> list[dict]:

        if user_id not in self.user_item_matrix.index:
            logger.info("Usuário %s sem interações (cold start).", user_id)
            return []

        known_games = self.get_known_games(user_id)

        # k vizinhos mais parecidos (exclui o próprio usuário)
        similares = self.user_similarity.loc[user_id].drop(index=user_id)
        vizinhos = similares.sort_values(ascending=False).head(self.k_neighbors)
        vizinhos = vizinhos[vizinhos > 0]

        if vizinhos.empty:
            return []

        # score(j)
        notas_vizinhos = self.user_item_matrix.loc[vizinhos.index].fillna(0)

        numerador = notas_vizinhos.mul(vizinhos, axis=0).sum(axis=0)
        denominador = vizinhos.sum()

        scores = numerador / denominador

        # exclui jogos já avaliados
        scores = scores.drop(index=known_games, errors="ignore")
        scores = scores[scores > 0]

        top = scores.sort_values(ascending=False).head(top_k)

        return [
            {"game_id": int(game_id), "score": float(score)}
            for game_id, score in top.items()
        ]