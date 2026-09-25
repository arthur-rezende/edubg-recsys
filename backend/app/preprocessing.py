import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:

    ESSENTIAL_COLS = ["Min Players", "Max Players", "Play Time", "Min Age"]
    NUM_COLS = ["Min Players", "Max Players", "Play Time", "Min Age", "Rating Average"]
    OUTLIER_COLS = ["Max Players", "Play Time"]

    def __init__(self):
        self.scaler = MinMaxScaler()

    def clean_null(self, dataframe: pd.DataFrame):
        # Dropar colunas essenciais para os filtros de compatbilidade
        df = dataframe.dropna(subset=self.ESSENTIAL_COLS).copy()

        # ID e Year Published - Poucos nulos, sem perda relevante
        df = df.dropna(subset=["ID", "Year Published"])

        # Owned Users - Transformar nulo em 0 (não há registro de posse)
        df["Owned Users"] = df["Owned Users"].fillna(0)

        # Preencher com string vazia Mecânicas e Domains nulos
        df["Mechanics"] = df["Mechanics"].fillna("")
        df["Domains"] = df["Domains"].fillna("")

        return df

    def drop_duplicates(self, dataframe: pd.DataFrame):
        # Remove duplicatas com base na coluna 'ID'
        df = dataframe.copy()
        coluna_id = 'ID'
        tam_antes_limpo = len(df)

        if dataframe.duplicated(subset=[coluna_id]).any():
            # Remoção de duplicatas
            df_sem_duplicatas = df.drop_duplicates(subset=[coluna_id], keep='last')
        else:
            # Nenhuma duplicata encontrada
            df_sem_duplicatas = df

        return df_sem_duplicatas

    def normalize_numeric_features(self, dataframe: pd.DataFrame):
        # Normalização de valores para vetor numérico
        df = dataframe.copy()

        df[self.NUM_COLS] = df[self.NUM_COLS].apply(pd.to_numeric, errors="coerce")

        # Tratar outliers antes de normalizar
        for col in self.OUTLIER_COLS:
            teto = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=teto)

        df[self.NUM_COLS] = self.scaler.fit_transform(df[self.NUM_COLS])

        return df

    def textual_embedding(self, dataframe: pd.DataFrame):
        # Padronizar texto
        df = dataframe.copy()

        df["Mechanics"] = df["Mechanics"].fillna("").str.lower()
        df["Domains"] = df["Domains"].fillna("").str.lower()

        # Texto combinado para embedding
        df["texto_embedding"] = (
            "Nome: " + df["Name"] +
            " Categorias: " + df["Domains"] +
            " Mecanicas: " + df["Mechanics"]
        )

        return df

    # Método orquestrador
    def process(self, dataframe: pd.DataFrame):
        # Executa pipeline completo de pré-processamento
        df = self.clean_null(dataframe)
        df = self.drop_duplicates(df)
        df = self.normalize_numeric_features(df)
        df = self.textual_embedding(df)
        return df

    def save_scaler(self, path: str) -> None:
        """Persiste o scaler já ajustado, para reaplicar depois em novas consultas."""
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        """Carrega um scaler previamente ajustado e salvo em disco."""
        self.scaler = joblib.load(path)