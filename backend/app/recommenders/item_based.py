import logging

from sklearn.neighbors import NearestNeighbors

from .collaborative_base import CollaborativeFilteringBase


logger = logging.getLogger(__name__)
print(">>> ITEM BASED CARREGADO:", __file__)

class ItemBasedRecommender(CollaborativeFilteringBase):

    def __init__(
        self,
        interactions_path: str,
        n_neighbors: int = 50,
        batch_size: int = 256
    ):

        super().__init__(interactions_path)

        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

        self.item_user_matrix = (
            self.user_item_matrix
            .T
            .tocsr()
        )

        self.neighbor_model = NearestNeighbors(
            n_neighbors=min(
                self.n_neighbors,
                self.item_user_matrix.shape[0]
            ),
            metric="cosine",
            algorithm="brute"
        )

        self.neighbor_model.fit(
            self.item_user_matrix
        )

        logger.info(
            "Item-Based: %d usuários, %d jogos.",
            self.user_item_matrix.shape[0],
            self.user_item_matrix.shape[1]
        )

    def _get_similar_items_batch(
        self,
        game_indices: list[int]
    ):

        if not game_indices:
            return []

        game_vectors = self.item_user_matrix[
            game_indices
        ]

        distances, indices = self.neighbor_model.kneighbors(
            game_vectors,
            return_distance=True
        )

        similarities = 1.0 - distances

        results = []

        for row_indices, row_similarities in zip(
            indices,
            similarities
        ):

            neighbors = []

            for neighbor_index, similarity in zip(
                row_indices,
                row_similarities
            ):

                similarity = float(similarity)

                if similarity <= 0:
                    continue

                neighbors.append(
                    (
                        int(neighbor_index),
                        similarity
                    )
                )

            results.append(neighbors)

        return results

    def _recommend_from_user(
        self,
        user_id: str,
        top_k: int
    ) -> list[dict]:

        user_index = self.get_user_index(
            user_id
        )

        if user_index is None:
            return []

        user_row = self.user_item_matrix.getrow(
            user_index
        )

        if user_row.nnz == 0:
            return []

        rated_game_indices = user_row.indices
        rated_values = user_row.data

        similar_items_batch = (
            self._get_similar_items_batch(
                rated_game_indices.tolist()
            )
        )

        known_game_indices = set(
            rated_game_indices
        )

        scores = {}
        similarity_sums = {}

        for rating, similar_items in zip(
            rated_values,
            similar_items_batch
        ):

            rating = float(rating)

            for similar_index, similarity in similar_items:

                if similar_index in known_game_indices:
                    continue

                scores[similar_index] = (
                    scores.get(similar_index, 0.0)
                    + similarity * rating
                )

                similarity_sums[similar_index] = (
                    similarity_sums.get(similar_index, 0.0)
                    + similarity
                )

        if not scores:
            return []

        ranked = sorted(
            (
                (
                    game_index,
                    scores[game_index]
                    / similarity_sums[game_index]
                )
                for game_index in scores
                if similarity_sums[game_index] > 0
            ),
            key=lambda item: item[1],
            reverse=True
        )[:top_k]

        return [
            {
                "game_id": int(
                    self.index_to_game[game_index]
                ),
                "score": float(score)
            }
            for game_index, score in ranked
        ]

    def recommend(
        self,
        user_id: str,
        top_k: int = 5
    ) -> list[dict]:

        return self._recommend_from_user(
            user_id,
            top_k
        )

    def recommend_batch(
        self,
        user_ids: list[str],
        top_k: int = 5
    ) -> dict:

        results = {}

        valid_users = []
        valid_indices = []

        for user_id in user_ids:

            user_index = self.get_user_index(
                user_id
            )

            if user_index is None:
                continue

            user_row = self.user_item_matrix.getrow(
                user_index
            )

            if user_row.nnz == 0:
                continue

            valid_users.append(user_id)
            valid_indices.append(user_index)

        for start in range(
            0,
            len(valid_users),
            self.batch_size
        ):

            batch_user_ids = valid_users[
                start:start + self.batch_size
            ]

            batch_user_indices = valid_indices[
                start:start + self.batch_size
            ]

            batch_game_indices = []
            batch_ratings = []

            for user_index in batch_user_indices:

                user_row = self.user_item_matrix.getrow(
                    user_index
                )

                batch_game_indices.append(
                    user_row.indices.tolist()
                )

                batch_ratings.append(
                    user_row.data.tolist()
                )

            flat_game_indices = [
                game_index
                for user_games in batch_game_indices
                for game_index in user_games
            ]

            if not flat_game_indices:
                continue

            unique_game_indices = list(
                set(flat_game_indices)
            )

            similar_items = (
                self._get_similar_items_batch(
                    unique_game_indices
                )
            )

            similarity_map = {
                game_index: neighbors
                for game_index, neighbors in zip(
                    unique_game_indices,
                    similar_items
                )
            }

            for user_id, user_games, user_ratings in zip(
                batch_user_ids,
                batch_game_indices,
                batch_ratings
            ):

                known_games = set(
                    user_games
                )

                scores = {}
                similarity_sums = {}

                for game_index, rating in zip(
                    user_games,
                    user_ratings
                ):

                    for similar_index, similarity in (
                        similarity_map.get(
                            game_index,
                            []
                        )
                    ):

                        if similar_index in known_games:
                            continue

                        scores[similar_index] = (
                            scores.get(
                                similar_index,
                                0.0
                            )
                            + similarity * rating
                        )

                        similarity_sums[similar_index] = (
                            similarity_sums.get(
                                similar_index,
                                0.0
                            )
                            + similarity
                        )

                if not scores:
                    results[user_id] = []
                    continue

                ranked = sorted(
                    (
                        (
                            game_index,
                            scores[game_index]
                            / similarity_sums[game_index]
                        )
                        for game_index in scores
                        if similarity_sums[game_index] > 0
                    ),
                    key=lambda item: item[1],
                    reverse=True
                )[:top_k]

                results[user_id] = [
                    {
                        "game_id": int(
                            self.index_to_game[
                                game_index
                            ]
                        ),
                        "score": float(score)
                    }
                    for game_index, score in ranked
                ]

        return results
    def recommend_similar_game(
        self,
        game_id: int,
        top_k: int = 15,
        excluded_game_ids: set[int] | None = None,
    ) -> list[dict]:
        game_index = self.get_game_index(game_id)
        if game_index is None:
            return []

        distances, indices = self.neighbor_model.kneighbors(
            self.item_user_matrix.getrow(game_index),
            return_distance=True,
        )

        excluded_game_ids = excluded_game_ids or set()
        recommendations = []

        for neighbor_index, distance in zip(indices[0], distances[0]):
            candidate_id = int(self.index_to_game[int(neighbor_index)])
            similarity = 1.0 - float(distance)

            if candidate_id == int(game_id):
                continue
            if candidate_id in excluded_game_ids:
                continue
            if similarity <= 0:
                continue

            recommendations.append(
                {
                    "game_id": candidate_id,
                    "score": similarity,
                }
            )

            if len(recommendations) == top_k:
                break

        return recommendations
