from pathlib import Path

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


def read_csv_with_utf8_fallback(path: str | Path) -> pd.DataFrame:
    """Lê um CSV em UTF-8 e, se necessário, converte uma codificação legacy para UTF-8."""
    csv_path = Path(path)

    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raw_bytes = csv_path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            decoded = raw_bytes.decode(encoding)
            converted_path = csv_path.with_name(f"{csv_path.stem}_utf8{csv_path.suffix}")
            converted_path.write_text(decoded, encoding="utf-8")
            return pd.read_csv(converted_path, encoding="utf-8")
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Não foi possível decodificar o arquivo CSV em uma codificação suportada: {csv_path}")


class Preprocessor:

    ESSENTIAL_GAME_COLS = [
        "Min Players",
        "Max Players",
        "Play Time",
        "Min Age"
    ]

    NUM_COLS = [
        "Min Players",
        "Max Players",
        "Play Time",
        "Min Age",
        "Rating Average"
    ]

    OUTLIER_COLS = [
        "Max Players",
        "Play Time"
    ]

    def __init__(self):
        self.scaler = MinMaxScaler()

    def clean_games(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()

        required_cols = [
            "ID",
            "Name",
            "Year Published",
            "Min Players",
            "Max Players",
            "Play Time",
            "Min Age",
            "Rating Average",
            "Owned Users",
            "Mechanics",
            "Domains"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Colunas ausentes no dataset de jogos: {missing_cols}"
            )

        numeric_cols = [
            "ID",
            "Year Published",
            "Min Players",
            "Max Players",
            "Play Time",
            "Min Age",
            "Users Rated",
            "Rating Average",
            "BGG Rank",
            "Complexity Average",
            "Owned Users"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["ID", "Name", "Year Published"]
        )

        df = df.dropna(
            subset=self.ESSENTIAL_GAME_COLS
        )

        df["Owned Users"] = df["Owned Users"].fillna(0)

        df["Mechanics"] = (
            df["Mechanics"]
            .fillna("")
            .astype(str)
        )

        df["Domains"] = (
            df["Domains"]
            .fillna("")
            .astype(str)
        )

        df = df.drop_duplicates(
            subset=["ID"],
            keep="last"
        )

        df["ID"] = df["ID"].astype(int)

        return df.reset_index(drop=True)

    def normalize_numeric_features(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        df = dataframe.copy()

        df[self.NUM_COLS] = df[self.NUM_COLS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        numeric_data = df[self.NUM_COLS].copy()

        for col in self.OUTLIER_COLS:
            teto = numeric_data[col].quantile(0.99)
            numeric_data[col] = numeric_data[col].clip(
                upper=teto
            )

        normalized_data = self.scaler.fit_transform(
            numeric_data
        )

        for index, col in enumerate(self.NUM_COLS):
            df[f"{col}_norm"] = normalized_data[:, index]

        return df

    def prepare_text(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        df = dataframe.copy()

        df["Mechanics"] = (
            df["Mechanics"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        df["Domains"] = (
            df["Domains"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        df["Name"] = (
            df["Name"]
            .fillna("")
            .astype(str)
        )

        df["texto_embedding"] = (
            "Nome: " + df["Name"] +
            " Categorias: " + df["Domains"] +
            " Mecanicas: " + df["Mechanics"]
        )

        return df

    def process_games(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        df = self.clean_games(dataframe)
        df = self.normalize_numeric_features(df)
        df = self.prepare_text(df)

        return df

    def clean_ratings(
        self,
        ratings: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.DataFrame:

        df = ratings.copy()

        required_cols = [
            "BGGId",
            "Rating",
            "Username"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Colunas ausentes no dataset de ratings: {missing_cols}"
            )

        df["BGGId"] = pd.to_numeric(
            df["BGGId"],
            errors="coerce"
        )

        df["Rating"] = pd.to_numeric(
            df["Rating"],
            errors="coerce"
        )

        df["Username"] = (
            df["Username"]
            .astype("string")
            .str.strip()
        )

        df = df.dropna(
            subset=[
                "BGGId",
                "Rating",
                "Username"
            ]
        )

        df["BGGId"] = df["BGGId"].astype(int)

        df = df[
            (df["Rating"] >= 0) &
            (df["Rating"] <= 10)
        ]

        valid_game_ids = set(
            games["ID"].astype(int)
        )

        df = df[
            df["BGGId"].isin(valid_game_ids)
        ]

        df = (
            df.groupby(
                ["Username", "BGGId"],
                as_index=False
            )["Rating"]
            .mean()
        )

        df = df.rename(
            columns={
                "Username": "user_id",
                "BGGId": "game_id",
                "Rating": "rating"
            }
        )

        return df.reset_index(drop=True)

    def process(
        self,
        games: pd.DataFrame,
        ratings: pd.DataFrame
    ):

        processed_games = self.process_games(games)

        interactions = self.clean_ratings(
            ratings,
            processed_games
        )

        return processed_games, interactions

    def save_scaler(self, path: str) -> None:
        joblib.dump(
            self.scaler,
            path
        )

    def load_scaler(self, path: str) -> None:
        self.scaler = joblib.load(path)