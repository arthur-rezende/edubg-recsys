from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().parents[1] / "backend" / "data" / "processed" / "games.csv"
# Gerado por scripts/fetch_game_images.py (URLs das capas na BGG)
IMAGES_PATH = DATA_PATH.with_name("game_images.csv")
IMAGE_URLS = [
    "https://picsum.photos/seed/meeple-market-1/1200/800",
    "https://picsum.photos/seed/meeple-market-2/1200/800",
    "https://picsum.photos/seed/meeple-market-3/1200/800",
    "https://picsum.photos/seed/meeple-market-4/1200/800",
    "https://picsum.photos/seed/meeple-market-5/1200/800",
]


@st.cache_data
def load_games() -> pd.DataFrame:
    games = pd.read_csv(DATA_PATH)
    numeric_columns = [
        "ID",
        "Rating Average",
        "Users Rated",
        "Min Age",
        "Play Time",
        "Min Players",
        "Max Players",
        "BGG Rank",
    ]
    for column in numeric_columns:
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games = games.dropna(subset=["ID", "Name", "Rating Average"])
    games["ID"] = games["ID"].astype(int)

    display_columns = [
        "Year Published",
        "Rating Average",
        "Min Age",
        "Play Time",
        "Min Players",
        "Max Players",
        "Users Rated",
        "BGG Rank",
        "Complexity Average",
        "Owned Users",
    ]
    for column in display_columns:
        games[column] = games[column].replace(0, pd.NA)

    games["Domains"] = games["Domains"].fillna("Strategy games")
    games["Mechanics"] = games["Mechanics"].fillna("")
    games["genre"] = games["Domains"].map(
        lambda value: str(value).split(",")[0].strip().title()
    )
    games["image"] = [
        IMAGE_URLS[index % len(IMAGE_URLS)]
        for index in range(len(games))
    ]

    # Capas reais da BGG, cruzadas pelo ID; sem capa, fica o placeholder
    games["thumbnail"] = games["image"]
    if IMAGES_PATH.exists():
        images = pd.read_csv(IMAGES_PATH).drop_duplicates("ID").set_index("ID")
        for column in ["image", "thumbnail"]:
            real = games["ID"].map(images[column])
            games[column] = real.where(
                real.notna() & (real != ""),
                games[column],
            )

    return games.sort_values("Rating Average", ascending=False).reset_index(drop=True)


def filter_games(
    games: pd.DataFrame,
    search: str,
    selected_genre: str,
    selected_age: str,
    selected_time: str,
) -> pd.DataFrame:
    filtered = games.copy()

    if search:
        search_value = search.lower().strip()
        filtered = filtered[
            filtered.apply(
                lambda game: search_value
                in f"{game['Name']} {game['genre']} {game['Mechanics']}".lower(),
                axis=1,
            )
        ]

    if selected_genre != "All genres":
        filtered = filtered[filtered["genre"] == selected_genre]

    if selected_age != "Any age":
        minimum_age = int(selected_age.replace("+", ""))
        filtered = filtered[filtered["Min Age"] >= minimum_age]

    if selected_time == "Under 1 hour":
        filtered = filtered[filtered["Play Time"] < 60]
    elif selected_time == "1–2 hours":
        filtered = filtered[
            (filtered["Play Time"] >= 60) & (filtered["Play Time"] < 120)
        ]
    elif selected_time == "2+ hours":
        filtered = filtered[filtered["Play Time"] >= 120]

    return filtered


def build_category_shelves(games: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    mechanics = games["Mechanics"].str.lower()
    domains = games["Domains"].str.lower()
    theme_text = mechanics + " " + domains

    def shelf(mask: pd.Series) -> pd.DataFrame:
        return games[mask].sort_values(
            ["Rating Average", "Users Rated"],
            ascending=[False, False],
        ).head(20)

    return [
        (
            "The solo shelf",
            "Great games for one",
            shelf(
                (games["Min Players"] <= 1)
                & (games["Max Players"] >= 1)
            ),
        ),
        (
            "The quick table",
            "Big fun in under an hour",
            shelf(
                (games["Play Time"] >= 15)
                & (games["Play Time"] <= 60)
            ),
        ),
        (
            "The cooperative corner",
            "Better together",
            shelf(theme_text.str.contains("cooperative", na=False)),
        ),
        (
            "The party table",
            "Bring the whole crew",
            shelf(theme_text.str.contains("party", na=False)),
        ),
        (
            "The deep shelf",
            "For strategy lovers",
            shelf(
                games["Complexity Average"].ge(3.0)
                & games["Rating Average"].ge(7.0)
            ),
        ),
        (
            "The crowd favorites",
            "Most played and talked about",
            games.sort_values(
                ["Users Rated", "Rating Average"],
                ascending=[False, False],
            ).head(20),
        ),
    ]
