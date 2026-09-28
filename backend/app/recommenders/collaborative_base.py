import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class CollaborativeFilteringBase:

    REQUIRED_COLUMNS = [
        "user_id",
        "game_id",
        "rating"
    ]

    def __init__(self, interactions_path: str):

        self.interactions_df = pd.read_csv(
            interactions_path
        )

        self._validate_data()

        self._build_mappings()

        self.user_item_matrix = self._build_matrix()

    def _validate_data(self):

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.interactions_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns in interactions data: "
                + ", ".join(missing_columns)
            )

        self.interactions_df = self.interactions_df.dropna(
            subset=self.REQUIRED_COLUMNS
        )

    def _build_mappings(self):

        self.users = self.interactions_df["user_id"].unique()
        self.games = self.interactions_df["game_id"].unique()

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(self.users)
        }

        self.game_to_index = {
            game_id: index
            for index, game_id in enumerate(self.games)
        }

        self.index_to_user = {
            index: user_id
            for user_id, index in self.user_to_index.items()
        }

        self.index_to_game = {
            index: game_id
            for game_id, index in self.game_to_index.items()
        }

    def _build_matrix(self):

        rows = self.interactions_df["user_id"].map(
            self.user_to_index
        )

        cols = self.interactions_df["game_id"].map(
            self.game_to_index
        )

        values = self.interactions_df["rating"].astype(
            np.float32
        )

        matrix = csr_matrix(
            (
                values,
                (rows, cols)
            ),
            shape=(
                len(self.users),
                len(self.games)
            ),
            dtype=np.float32
        )

        return matrix

    def get_user_index(self, user_id: str):

        return self.user_to_index.get(user_id)

    def get_game_index(self, game_id: int):

        return self.game_to_index.get(game_id)

    def get_user_ratings(self, user_id: str) -> pd.Series:

        user_index = self.get_user_index(user_id)

        if user_index is None:
            return pd.Series(dtype=float)

        row = self.user_item_matrix.getrow(
            user_index
        )

        ratings = {
            self.index_to_game[col]: value
            for col, value in zip(
                row.indices,
                row.data
            )
        }

        return pd.Series(ratings)

    def get_known_games(self, user_id: str) -> set:

        ratings = self.get_user_ratings(user_id)

        return set(ratings.index)