from pathlib import Path

import pandas as pd

from ..database import get_rated_game_id
from ..recommenders.context_based import ContextBasedRecommender
from ..recommenders.item_based import ItemBasedRecommender
from ..recommenders.user_based import UserBasedRecommender
from .embeddings import EmbeddingsService

print(">>> RECOMMENDATION SERVICE CARREGADO:", __file__)
class RecommendationService:
    def __init__(self, project_root: Path):
        data_dir = project_root / "data" / "processed"

        self.games = pd.read_csv(data_dir / "games.csv")
        self.games["ID"] = self.games["ID"].astype(int)
        self.games_by_id = self.games.set_index("ID")

        interactions_path = str(data_dir / "interactions.csv")
        scaler_path = data_dir / "scaler.pkl"

        self.interactions_path = interactions_path

        self.recommenders = {
            "item": ItemBasedRecommender(interactions_path),
            "user": UserBasedRecommender(interactions_path),
        }

        self.embeddings_service = EmbeddingsService()
        self.context_recommender = ContextBasedRecommender(
            games=self.games,
            scaler_path=scaler_path,
            embeddings_service=self.embeddings_service,
        )

    def _get_recommender(self, algorithm: str):
        if algorithm not in self.recommenders:
            raise ValueError(
                f"Algoritmo de recomendação inválido: {algorithm}"
            )

        return self.recommenders[algorithm]

    def _get_known_games(
        self,
        user_id: str | None,
    ) -> set[int]:
        if not user_id:
            return set()

        csv_known = {
            int(game_id)
            for game_id in self.recommenders["item"].get_known_games(
                user_id
            )
        }

        db_known = {
            int(game_id)
            for game_id in get_rated_game_id(user_id)
        }

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

    def _top_rated_games(
        self,
        user_id: str,
        limit: int = 3,
    ) -> list[tuple[int, float]]:
        ratings = self.recommenders["item"].get_user_ratings(user_id)

        if ratings is None:
            return []

        if isinstance(ratings, pd.Series):
            if ratings.empty:
                return []

            ranked = ratings.sort_values(
                ascending=False,
                kind="stable",
            )

            return [
                (int(game_id), float(rating))
                for game_id, rating in ranked.head(limit).items()
            ]

        if isinstance(ratings, pd.DataFrame):
            if ratings.empty:
                return []

            if "game_id" in ratings.columns and "rating" in ratings.columns:
                rows = ratings[["game_id", "rating"]].dropna()
                rows = rows.sort_values("rating", ascending=False)
                return [
                    (int(row.game_id), float(row.rating))
                    for row in rows.head(limit).itertuples(index=False)
                ]

            return []

        if isinstance(ratings, dict):
            if not ratings:
                return []

            return sorted(
                (
                    (int(game_id), float(rating))
                    for game_id, rating in ratings.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]

        if isinstance(ratings, list):
            parsed = []

            for item in ratings:
                if isinstance(item, dict):
                    game_id = item.get("game_id", item.get("BGGId", item.get("ID")))
                    rating = item.get("rating", item.get("Rating"))
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    game_id, rating = item[0], item[1]
                else:
                    continue

                if game_id is None or rating is None:
                    continue

                try:
                    parsed.append((int(game_id), float(rating)))
                except (TypeError, ValueError):
                    continue

            parsed.sort(key=lambda item: item[1], reverse=True)
            return parsed[:limit]

        return []

    def _game_name(self, game_id: int) -> str:
        if game_id not in self.games_by_id.index:
            return str(game_id)

        return str(self.games_by_id.loc[game_id, "Name"])

    def _hydrate(
        self,
        recommendations: list[dict],
        known_games: set[int],
    ) -> list[dict]:
        response = []

        for recommendation in recommendations:
            game_id = int(recommendation["game_id"])

            if (
                game_id in known_games
                or game_id not in self.games_by_id.index
            ):
                continue

            game = self.games_by_id.loc[game_id]

            response.append(
                {
                    "game_id": game_id,
                    "nome": str(game["Name"]),
                    "score": float(recommendation["score"]),
                    "rating": (
                        float(game["Rating Average"])
                        if pd.notna(game["Rating Average"])
                        else None
                    ),
                }
            )

        return response

    def recommend_home(
        self,
        user_id: str,
        top_k: int = 15,
    ) -> list[dict]:
        known_games = self._get_known_games(user_id)
        top_games = self._top_rated_games(user_id, limit=3)
        sections = []

        item_recommender = self.recommenders["item"]
        user_recommender = self.recommenders["user"]

        for game_id, _rating in top_games:
            similar = item_recommender.recommend_similar_game(
                game_id=game_id,
                top_k=top_k,
                excluded_game_ids=known_games,
            )

            items = self._hydrate(similar, known_games)

            if items:
                sections.append(
                    {
                        "type": "item_based",
                        "title": (
                            f"Games similar to {self._game_name(game_id)}"
                        ),
                        "items": items,
                    }
                )

        for game_id, _rating in top_games:
            liked_by = user_recommender.recommend_from_liked_game(
                game_id=game_id,
                current_user_id=user_id,
                top_k=top_k,
                min_rating=8.0,
            )

            items = self._hydrate(liked_by, known_games)

            if items:
                sections.append(
                    {
                        "type": "user_based",
                        "title": (
                            "What players who liked "
                            f"{self._game_name(game_id)} are playing"
                        ),
                        "items": items,
                    }
                )

        return sections

    def recommend(self, request) -> list[dict]:
        known_games = self._get_known_games(request.user_id)

        if request.mode == "context":
            recommendations = self._context_recommendations(
                request,
                known_games,
            )
        else:
            recommender = self._get_recommender(request.algorithm)

            recommendations = (
                recommender.recommend(
                    request.user_id,
                    request.top_k + len(known_games),
                )
                if request.user_id
                else []
            )

        return self._hydrate(
            recommendations,
            known_games,
        )[:request.top_k]
