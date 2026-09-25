import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(
            self,
            embeddings_service,
            alpha: float = 0.4,
            beta: float = 0.6,
    ):
                # Serviço responsável por gerar embeddings dos textos.
        self.embeddings_service = embeddings_service

                # Pesos usados para combinar as características numéricas e textuais.
        self.alpha = alpha
        self.beta = beta

    def build_query_text(
            self,
            objetivo_pedagogico: str,
            contexto: str,
        ) -> str:
        # Monta uma única descrição textual para gerar o embedding da consulta.
                return f"objetivo pedagógico: {objetivo_pedagogico}, contexto: {contexto}"

    def build_query_vector(
            self,
            numeric_vector: np.ndarray,
            text: str
    ) -> np.ndarray:
                # Gera o vetor semântico a partir da descrição textual da consulta.
        text_vector = self.embeddings_service.get_embeddings(text)

                # Converte as características numéricas para o mesmo tipo dos embeddings.
        numeric_vector = np.array(numeric_vector, dtype=np.float32)

                # Junta os dois vetores, aplicando um peso a cada fonte de informação.
        final_vector = np.concatenate([self.alpha * numeric_vector, self.beta * text_vector])

        return final_vector

    def calculate_similarity(
            self,
            query_vector: np.ndarray,
            game_vector: np.ndarray
        ) -> np.ndarray:
        # Compara a consulta com os vetores dos jogos usando similaridade cosseno.
                return cosine_similarity(query_vector.reshape(1, -1), game_vector)[0]

    def get_top_k(
            self,
            similarities: np.ndarray,
            top_k: int
    ) -> np.ndarray:

                # Ordena os índices pela maior similaridade e mantém apenas os primeiros.
        indices = np.argsort(similarities)[::-1]
        return indices[:top_k]