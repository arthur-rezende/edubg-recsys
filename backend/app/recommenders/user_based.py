import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .collaborative_base import CollaborativeFilteringBase


logger = logging.getLogger(__name__)

print(">>> USER BASED CARREGADO:", __file__)
class UserBasedRecommender(CollaborativeFilteringBase):

    def __init__(
        self,
        interactions_path: str,
        k_neighbors: int = 20,
    ):
        super().__init__(interactions_path)

        self.k_neighbors = k_neighbors

        self.neighbor_model = NearestNeighbors(
            n_neighbors=min(
                self.k_neighbors + 1,
                self.user_item_matrix.shape[0],
            ),
            metric="cosine",
            algorithm="brute",
        )

        self.neighbor_model.fit(self.user_item_matrix)

        logger.info(
            "User-Based: %d usuários, %d jogos.",
            self.user_item_matrix.shape[0],
            self.user_item_matrix.shape[1],
        )

    def _get_similar_users_batch(self, user_indices: list[int]):
        if not user_indices:
            return []

        user_vectors = self.user_item_matrix[user_indices]

        distances, indices = self.neighbor_model.kneighbors(
            user_vectors,
            return_distance=True,
        )

        similarities = 1.0 - distances
        results = []

        for current_user_index, row_indices, row_similarities in zip(
            user_indices,
            indices,
            similarities,
        ):
            neighbors = []

            for neighbor_index, similarity in zip(
                row_indices,
                row_similarities,
            ):
                if neighbor_index == current_user_index:
                    continue

                if similarity <= 0:
                    continue

                neighbors.append(
                    (
                        neighbor_index,
                        float(similarity),
                    )
                )

            results.append(neighbors)

        return results

    def _recommend_from_neighbors(
        self,
        user_id: str,
        similar_users,
        top_k: int,
    ) -> list[dict]:
        known_games = self.get_known_games(user_id)

        if not similar_users:
            return []

        scores = {}
        similarity_sums = {}

        for neighbor_index, similarity in similar_users:
            neighbor_row = self.user_item_matrix[neighbor_index]

            for game_index, rating in zip(
                neighbor_row.indices,
                neighbor_row.data,
            ):
                game_id = self.index_to_game[game_index]

                if game_id in known_games:
                    continue

                scores[game_id] = (
                    scores.get(game_id, 0.0)
                    + similarity * float(rating)
                )

                similarity_sums[game_id] = (
                    similarity_sums.get(game_id, 0.0)
                    + similarity
                )

        if not scores:
            return []

        ranked = sorted(
            (
                (
                    game_id,
                    scores[game_id] / similarity_sums[game_id],
                )
                for game_id in scores
                if similarity_sums[game_id] > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return [
            {
                "game_id": int(game_id),
                "score": float(score),
            }
            for game_id, score in ranked
        ]

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        user_index = self.get_user_index(user_id)

        if user_index is None:
            logger.info(
                "Usuário %s sem interações (cold start).",
                user_id,
            )
            return []

        similar_users = self._get_similar_users_batch([user_index])[0]

        return self._recommend_from_neighbors(
            user_id,
            similar_users,
            top_k,
        )

    def recommend_batch(
        self,
        user_ids: list[str],
        top_k: int = 5,
    ) -> dict:
        valid_users = []
        valid_indices = []

        for user_id in user_ids:
            user_index = self.get_user_index(user_id)

            if user_index is None:
                continue

            valid_users.append(user_id)
            valid_indices.append(user_index)

        if not valid_users:
            return {}

        similar_users_batch = []
        batch_size = 256

        for start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[start:start + batch_size]
            similar_users_batch.extend(
                self._get_similar_users_batch(batch_indices)
            )

        results = {}

        for user_id, similar_users in zip(
            valid_users,
            similar_users_batch,
        ):
            results[user_id] = self._recommend_from_neighbors(
                user_id,
                similar_users,
                top_k,
            )

        return results

    def recommend_from_liked_game(
        self,
        game_id: int,
        current_user_id: str | None = None,
        top_k: int = 15,
        min_rating: float = 8.0,
    ) -> list[dict]:
        game_index = self.get_game_index(game_id)

        if game_index is None:
            return []

        game_column = self.user_item_matrix.getcol(game_index)

        liked_user_indices = np.flatnonzero(
            np.asarray(game_column.toarray()).ravel() >= min_rating
        )

        if current_user_id is not None:
            current_user_index = self.get_user_index(current_user_id)

            if current_user_index is not None:
                liked_user_indices = liked_user_indices[
                    liked_user_indices != current_user_index
                ]

        if liked_user_indices.size == 0:
            return []

        known_games = (
            {
                int(game_id)
                for game_id in self.get_known_games(current_user_id)
            }
            if current_user_id
            else set()
        )

        liked_rows = self.user_item_matrix[
            liked_user_indices
        ].tocsr()

        anchor_ratings = np.asarray(
            game_column[liked_user_indices].toarray()
        ).ravel()

        weights = anchor_ratings / 10.0

        weighted_rows = liked_rows.multiply(
            weights[:, np.newaxis]
        )

        scores = np.asarray(
            weighted_rows.sum(axis=0)
        ).ravel()

        presence = liked_rows.copy()
        presence.data = np.ones_like(
            presence.data,
            dtype=float,
        )

        similarity_sums = np.asarray(
            presence.multiply(
                weights[:, np.newaxis]
            ).sum(axis=0)
        ).ravel()

        valid = similarity_sums > 0
        candidate_indices = np.flatnonzero(valid)

        candidate_indices = candidate_indices[
            candidate_indices != game_index
        ]

        if candidate_indices.size == 0:
            return []

        candidate_scores = (
            scores[candidate_indices]
            / similarity_sums[candidate_indices]
        )

        order = np.argsort(candidate_scores)[::-1]

        recommendations = []

        for position in order:
            candidate_index = int(candidate_indices[position])
            candidate_game_id = int(
                self.index_to_game[candidate_index]
            )

            if candidate_game_id in known_games:
                continue

            recommendations.append(
                {
                    "game_id": candidate_game_id,
                    "score": float(candidate_scores[position]),
                }
            )

            if len(recommendations) >= top_k:
                break

        return recommendations
