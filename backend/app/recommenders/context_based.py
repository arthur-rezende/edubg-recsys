import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..services.embeddings import EmbeddingsService


class ContextBasedRecommender:
    NUMERIC_COLUMNS = [
        "Min Players",
        "Max Players",
        "Play Time",
        "Min Age",
        "Rating Average",
    ]

    OUTLIER_COLUMNS = {
        "Max Players",
        "Play Time",
    }

    def __init__(
        self,
        games: pd.DataFrame,
        scaler_path,
        embeddings_service: EmbeddingsService,
    ):
        self.games = games.reset_index(drop=True).copy()
        self.embeddings_service = embeddings_service
        self.scaler = joblib.load(scaler_path)

        numeric_data = (
            self.games[self.NUMERIC_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        numeric_data = self._clip_outliers(numeric_data)
        self.numeric_vectors = self.scaler.transform(
            numeric_data
        ).astype(np.float32)

        self.text_vectors = self._build_text_vectors()

    def _build_text_vectors(self) -> np.ndarray:
        texts = (
            self.games["texto_embedding"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        embeddings = self.embeddings_service.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def _clip_outliers(self, numeric_data: pd.DataFrame) -> pd.DataFrame:
        clipped = numeric_data.copy()

        for column_index, column in enumerate(self.NUMERIC_COLUMNS):
            if column in self.OUTLIER_COLUMNS:
                clipped[column] = clipped[column].clip(
                    upper=self.scaler.data_max_[column_index]
                )

        return clipped

    def _apply_hard_filters(
        self,
        idade_alunos: int,
        quantidade_alunos: int,
        tempo_disponivel: int,
    ) -> np.ndarray:
        min_players = pd.to_numeric(
            self.games["Min Players"],
            errors="coerce"
        )

        max_players = pd.to_numeric(
            self.games["Max Players"],
            errors="coerce"
        )

        play_time = pd.to_numeric(
            self.games["Play Time"],
            errors="coerce"
        )

        min_age = pd.to_numeric(
            self.games["Min Age"],
            errors="coerce"
        )

        mask = (
            (min_age <= idade_alunos)
            & (min_players <= quantidade_alunos)
            & (max_players >= quantidade_alunos)
            & (play_time <= tempo_disponivel)
        )

        return np.flatnonzero(mask.to_numpy())

    def recommend(
        self,
        idade_alunos: int,
        quantidade_alunos: int,
        tempo_disponivel: int,
        objetivo_pedagogico: str,
        contexto: str,
        top_k: int,
        excluded_game_ids: set[int] | None = None,
    ) -> list[dict]:

        candidate_indices = self._apply_hard_filters(
            idade_alunos,
            quantidade_alunos,
            tempo_disponivel,
        )

        if len(candidate_indices) == 0:
            return []

        query_text = (
            f"Objetivo pedagogico: {objetivo_pedagogico}. "
            f"Contexto da aula: {contexto}."
        )

        query_text_vector = self.embeddings_service.get_embeddings(
            query_text
        ).reshape(1, -1)

        query_numeric = np.array(
            [[
                quantidade_alunos,
                quantidade_alunos,
                tempo_disponivel,
                idade_alunos,
                self.games["Rating Average"].median(),
            ]],
            dtype=np.float32,
        )

        query_numeric = self._clip_outliers(
            pd.DataFrame(query_numeric, columns=self.NUMERIC_COLUMNS)
        )
        query_numeric = self.scaler.transform(
            query_numeric
        ).astype(np.float32)

        query_vector = np.concatenate(
            [
                query_text_vector,
                query_numeric,
            ],
            axis=1,
        )

        candidate_vectors = np.concatenate(
            [
                self.text_vectors[candidate_indices],
                self.numeric_vectors[candidate_indices],
            ],
            axis=1,
        )

        scores = cosine_similarity(
            query_vector,
            candidate_vectors,
        )[0]

        excluded = excluded_game_ids or set()

        ranked_indices = np.argsort(scores)[::-1]

        recommendations = []

        for position in ranked_indices:
            dataframe_index = candidate_indices[position]
            game_id = int(self.games.iloc[dataframe_index]["ID"])

            if game_id in excluded:
                continue

            recommendations.append(
                {
                    "game_id": game_id,
                    "score": float(scores[position]),
                }
            )

            if len(recommendations) >= top_k:
                break

        return recommendations