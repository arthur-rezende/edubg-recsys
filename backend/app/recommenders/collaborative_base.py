import pandas as pd

class CollaborativeFilteringBase:
    def __init__(self, interactions_path: str):
        # Carrega os dados de interações do arquivo CSV.
        self.interactions_df = pd.read_csv(interactions_path)

        self._validate_data()
        self.user_item_matrix = self._build_matrix()

    def _validate_data(self):
        required_columns =[ "user_id", "item_id", "rating" ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.interactions.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in interactions data: {', '.join(missing_columns)}"
            )

        def _build_matrix(self) -> pd.DataFrame:
            matrix = self.interactions.pivot(index="user_id", columns="game_id", values="rating")
            return matrix

        def get_user_ratings(self, user_id: str) -> pd.Series:
            if user_id not in self.user_item_matrix.index:
                return pd.Series(dtype=float)

            return self.user_item_matrix.loc[user_id]

        def get_known_games(self, user_id: str) -> set:
            ratings = self.get_user_ratings(user_id)

            return set(ratings.dropna().index)