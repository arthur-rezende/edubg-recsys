# Classe usada para transformar textos em vetores numéricos (embeddings).
from sentence_transformers import SentenceTransformer
import numpy as np

# Modelo pré-treinado que representa o significado semântico dos textos.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class EmbeddingsService:
    def __init__(self):
        # Carrega o modelo uma única vez para reutilizá-lo nas requisições.
        self.model = SentenceTransformer(MODEL_NAME)

    def get_embeddings(self, text: str) -> np.ndarray:
        # Converte o texto em um vetor e normaliza seu comprimento para facilitar
        # a comparação de similaridade com outros vetores.
        embeddings = self.model.encode(text, normalize_embeddings=True)

        # Garante que o resultado seja um array NumPy com números de 32 bits.
        return np.array(embeddings, dtype=np.float32)