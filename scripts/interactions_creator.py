from pathlib import Path

import pandas as pd

from preprocessing import Preprocessor, read_csv_with_utf8_fallback


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

games = read_csv_with_utf8_fallback(DATA_DIR / "BGG_Data_Set.csv")
ratings = read_csv_with_utf8_fallback(DATA_DIR / "user_ratings.csv")

preprocessor = Preprocessor()

processed_games, interactions = preprocessor.process(
    games,
    ratings
)

processed_games.to_csv(
    OUTPUT_DIR / "games.csv",
    index=False
)

interactions.to_csv(
    OUTPUT_DIR / "interactions.csv",
    index=False
)

preprocessor.save_scaler(
    str(OUTPUT_DIR / "scaler.pkl")
)