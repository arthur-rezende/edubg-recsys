from pathlib import Path

import pandas as pd

from ..database import get_rated_game_id
from ..recommenders.context_based import ContextBasedRecommender
from ..recommenders.item_based import ItemBasedRecommender
from ..recommenders.user_based import UserBasedRecommender
from .embeddings import EmbeddingsService


class RecommendationService:
    def __init__(self, project_root: Path):
        data_dir = project_root / "data" / "processed"

        self.games = pd.read_csv(
            data_dir / "games.csv"
        )

        self.games["ID"] = self.games["ID"].astype(int)

        self.games_by_id = self.games.set_index("ID")

        interactions_path = str(
            data_dir / "interactions.csv"
        )

        scaler_path = data_dir / "scaler.pkl"

        self.interactions_path = interactions_path

        self.recommenders = {
            "item": ItemBasedRecommender(
                interactions_path
            )
        }

        self._user_recommender = None

        self.embeddings_service = EmbeddingsService()

        self.context_recommender = ContextBasedRecommender(
            games=self.games,
            scaler_path=scaler_path,
            embeddings_service=self.embeddings_service,
        )

    def _get_recommender(self, algorithm: str):
        if algorithm == "item":
            return self.recommenders["item"]

        if self._user_recommender is None:
            self._user_recommender = UserBasedRecommender(
                self.interactions_path
            )

        return self._user_recommender

    def _get_known_games(
        self,
        user_id: str | None,
    ) -> set[int]:

        if not user_id:
            return set()

        # Board Games avaliados pelo usuário no dataset
        csv_known = {
            int(game_id)
            for game_id in self.recommenders["item"].get_known_games(
                user_id
            )
        }

        # Board Games avaliados pelo usuário na aplicação
        db_known = get_rated_game_id(user_id)

        return csv_known | db_known

    def _context_recommendations(
        self,
        request,
        known_games: set[int],
    ) -> list[dict]:

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
        known_games = self._get_known_games(
            request.user_id
        )

        if request.mode == "context":
            recommendations = self._context_recommendations(
                request,
                known_games,
            )
        else:
            recommender = self._get_recommender(
                request.algorithm
            )

            recommendations = (
                recommender.recommend(
                    request.user_id,
                    request.top_k + len(known_games),
                )
                if request.user_id
                else []
            )

            if not recommendations:
                recommendations = self._context_recommendations(
                    request,
                    known_games,
                )

        response = []

        for recommendation in recommendations:
            game_id = int(
                recommendation["game_id"]
            )

            if game_id in known_games:
                continue

            if game_id not in self.games_by_id.index:
                continue

            game = self.games_by_id.loc[game_id]

            response.append(
                {
                    "game_id": game_id,
                    "nome": str(game["Name"]),
                    "score": float(
                        recommendation["score"]
                    ),
                    "rating": (
                        float(game["Rating Average"])
                        if pd.notna(game["Rating Average"])
                        else None
                    ),
                }
            )

            if len(response) == request.top_k:
                break

        return response