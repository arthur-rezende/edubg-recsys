from pathlib import Path

import numpy as np
import pandas as pd

from ..recommenders.context_based import ContextBasedRecommender
from ..recommenders.item_based import ItemBasedRecommender
from ..recommenders.user_based import UserBasedRecommender


class RecommendationService:
    """Orquestra os recomendadores e mantém os artefatos carregados em memória."""

    def __init__(self, project_root: Path):
        data_dir = project_root / "data" / "processed"
        self.games = pd.read_csv(data_dir / "games.csv")
        self.games["ID"] = self.games["ID"].astype(int)
        self.games_by_id = self.games.set_index("ID")

        interactions_path = str(data_dir / "interactions.csv")
        self.recommenders = {"item": ItemBasedRecommender(interactions_path)}
        self.interactions_path = interactions_path
        self._user_recommender = None
        self.context_recommender = ContextBasedRecommender(self.games)

    def _get_recommender(self, algorithm: str):
        if algorithm == "item":
            return self.recommenders["item"]
        if self._user_recommender is None:
            self._user_recommender = UserBasedRecommender(self.interactions_path)
        return self._user_recommender

    def _context_recommendations(self, request, known_games: set[int]) -> list[dict]:
        return self.context_recommender.recommend(
            idade_alunos=request.idade_alunos,
            quantidade_alunos=request.quantidade_alunos,
            tempo_disponivel=request.tempo_disponivel,
            objetivo_pedagogico=request.objetivo_pedagogico,
            contexto=request.contexto,
            top_k=request.top_k,
            excluded_game_ids=known_games,
        )

    def recommend(self, request) -> list[dict]:
        known_games: set[int] = set()
        if request.user_id:
            known_games = {
                int(game_id)
                for game_id in self.recommenders["item"].get_known_games(request.user_id)
            }

        if request.mode == "context":
            recommendations = self._context_recommendations(request, known_games)
        else:
            recommender = self._get_recommender(request.algorithm)
            recommendations = (
                recommender.recommend(request.user_id, request.top_k)
                if request.user_id
                else []
            )
            if not recommendations:
                recommendations = self._context_recommendations(request, known_games)

        response = []
        for recommendation in recommendations:
            game_id = int(recommendation["game_id"])
            game = self.games_by_id.loc[game_id]
            response.append({
                "game_id": game_id,
                "nome": str(game["Name"]),
                "score": float(recommendation["score"]),
                "rating": float(game["Rating Average"])
                if pd.notna(game["Rating Average"])
                else None,
            })
        return response