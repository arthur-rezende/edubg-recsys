# backend/app/build_pipeline.py
from pathlib import Path
import pandas as pd
import numpy as np

from preprocessing import Preprocessor
from embeddings import EmbeddingsService

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ALPHA = 0.3
BETA = 0.7


def main():
    df_bruto = pd.read_csv(DATA_DIR / "BGG_Data_Set.csv", encoding="latin-1")

    preprocessor = Preprocessor()
    df_limpo = preprocessor.clean_null(df_bruto)
    df_limpo = preprocessor.drop_duplicates(df_limpo)

    # games.csv guarda os valores REAIS (não normalizados)
    df_com_texto = preprocessor.textual_embedding(df_limpo.copy())
    df_com_texto.to_csv(DATA_DIR / "games.csv", index=False)

    # df normalizado
    df_normalizado = preprocessor.normalize_numeric_features(df_limpo)

    embeddings_service = EmbeddingsService()
    embeddings = embeddings_service.get_embeddings(df_com_texto["texto_embedding"].tolist())

    vetores_numericos = df_normalizado[Preprocessor.NUM_COLS].values.astype(np.float32)
    vetores_finais = np.hstack([ALPHA * vetores_numericos, BETA * embeddings])

    preprocessor.save_scaler(str(DATA_DIR / "scaler.pkl"))
    np.save(DATA_DIR / "vectors.npy", vetores_finais)

    print(f"Pipeline concluído: {len(df_com_texto)} jogos processados.")


if __name__ == "__main__":
    main()