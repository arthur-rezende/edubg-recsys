import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContextBasedRecommender:
        NUMERIC_COLUMNS = [
                "Min Players_norm", "Max Players_norm", "Play Time_norm",
                "Min Age_norm", "Rating Average_norm",
        ]

        def __init__(self, games: pd.DataFrame):
                self.games = games.reset_index(drop=True)
                self.vectorizer = TfidfVectorizer(max_features=30_000)
                texts = self.games["texto_embedding"].fillna("").astype(str)
                self.text_vectors = self.vectorizer.fit_transform(texts)
                self.numeric_vectors = self.games[self.NUMERIC_COLUMNS].fillna(0).to_numpy(
                        dtype=np.float32
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
                query_text = f"{objetivo_pedagogico} {contexto}"
                text_scores = cosine_similarity(
                        self.vectorizer.transform([query_text]), self.text_vectors
                )[0]

                query_numeric = np.array([
                        1.0 / max(idade_alunos, 1),
                        quantidade_alunos / 10.0,
                        tempo_disponivel / 240.0,
                        idade_alunos / 18.0,
                        0.5,
                ], dtype=np.float32)
                numeric_scores = cosine_similarity(
                        query_numeric.reshape(1, -1), self.numeric_vectors
                )[0]
                scores = 0.6 * text_scores + 0.4 * numeric_scores

                excluded = excluded_game_ids or set()
                candidates = [
                        index for index in np.argsort(scores)[::-1]
                        if int(self.games.iloc[index]["ID"]) not in excluded
                ][:top_k]
                return [
                        {
                                "game_id": int(self.games.iloc[index]["ID"]),
                                "score": float(scores[index]),
                        }
                        for index in candidates
                ]