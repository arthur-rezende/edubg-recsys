import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingsService:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def get_embeddings(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32
        )