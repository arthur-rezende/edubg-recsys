import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

import joblib
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
                scaler_path: str | Path,
                embeddings_service: EmbeddingsService,
                numeric_weight: float = 0.4,
                text_weight: float = 0.6,
        ):
                required_columns = {"ID", "texto_embedding", *self.NUMERIC_COLUMNS}
                missing_columns = required_columns.difference(games.columns)
                if missing_columns:
                        raise ValueError(
                                "Colunas ausentes no catálogo de jogos: "
                                + ", ".join(sorted(missing_columns))
                        )

                if numeric_weight < 0 or text_weight < 0:
                        raise ValueError("Os pesos das características não podem ser negativos")
                if numeric_weight == 0 and text_weight == 0:
                        raise ValueError("Pelo menos um peso de característica deve ser positivo")

                self.games = games.reset_index(drop=True).copy()
                self.game_ids = self.games["ID"].astype(int).to_numpy()
                self.embeddings_service = embeddings_service
                self.numeric_weight = numeric_weight
                self.text_weight = text_weight
                self.scaler = joblib.load(scaler_path)

                numeric_data = (
                        self.games[self.NUMERIC_COLUMNS]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0)
                )
                self.numeric_vectors = self.scaler.transform(
                        self._clip_outliers(numeric_data)
                ).astype(np.float32)

                self.text_vectors = self._build_text_vectors()
                self.feature_vectors = np.concatenate(
                        [
                                self.text_weight * self.text_vectors,
                                self.numeric_weight * self.numeric_vectors,
                        ],
                        axis=1,
                )
                self.rating_median = float(
                        pd.to_numeric(self.games["Rating Average"], errors="coerce").median()
                )
                if not np.isfinite(self.rating_median):
                        self.rating_median = 0.0

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
                return np.asarray(embeddings, dtype=np.float32)

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
                min_players = pd.to_numeric(self.games["Min Players"], errors="coerce")
                max_players = pd.to_numeric(self.games["Max Players"], errors="coerce")
                play_time = pd.to_numeric(self.games["Play Time"], errors="coerce")
                min_age = pd.to_numeric(self.games["Min Age"], errors="coerce")

                mask = (
                        (min_age <= idade_alunos)
                        & (min_players <= quantidade_alunos)
                        & (max_players >= quantidade_alunos)
                        & (play_time <= tempo_disponivel)
                )
                return np.flatnonzero(mask.to_numpy())

        def _build_query_vector(
                self,
                idade_alunos: int,
                quantidade_alunos: int,
                tempo_disponivel: int,
                objetivo_pedagogico: str,
                contexto: str,
        ) -> np.ndarray:
                query_text = (
                        f"Objetivo pedagogico: {objetivo_pedagogico}. "
                        f"Contexto da aula: {contexto}."
                )
                text_vector = self.embeddings_service.get_embeddings(query_text)

                query_numeric = pd.DataFrame(
                        [[
                                quantidade_alunos,
                                quantidade_alunos,
                                tempo_disponivel,
                                idade_alunos,
                                self.rating_median,
                        ]],
                        columns=self.NUMERIC_COLUMNS,
                )
                numeric_vector = self.scaler.transform(
                        self._clip_outliers(query_numeric)
                ).astype(np.float32)

                return np.concatenate(
                        [
                                self.text_weight * text_vector.reshape(1, -1),
                                self.numeric_weight * numeric_vector,
                        ],
                        axis=1,
                )

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
                if top_k <= 0:
                        return []

                candidate_indices = self._apply_hard_filters(
                        idade_alunos,
                        quantidade_alunos,
                        tempo_disponivel,
                )
                if len(candidate_indices) == 0:
                        return []

                excluded = excluded_game_ids or set()
                if excluded:
                        candidate_indices = candidate_indices[
                                ~np.isin(self.game_ids[candidate_indices], list(excluded))
                        ]
                if len(candidate_indices) == 0:
                        return []

                query_vector = self._build_query_vector(
                        idade_alunos,
                        quantidade_alunos,
                        tempo_disponivel,
                        objetivo_pedagogico,
                        contexto,
                )
                scores = cosine_similarity(
                        query_vector,
                        self.feature_vectors[candidate_indices],
                )[0]

                result_count = min(top_k, len(scores))
                if result_count < len(scores):
                        selected_positions = np.argpartition(
                                scores,
                                len(scores) - result_count,
                        )[-result_count:]
                else:
                        selected_positions = np.arange(len(scores))

                selected_positions = selected_positions[
                        np.argsort(scores[selected_positions])[::-1]
                ]

                return [
                        {
                                "game_id": int(self.game_ids[candidate_indices[position]]),
                                "score": float(scores[position]),
                        }
                        for position in selected_positions
                ]
