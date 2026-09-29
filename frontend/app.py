import streamlit as st

from auth import authenticate, ensure_demo_user
from components import (
    render_empty_state,
    render_hero,
    render_hero_slideshow,
    render_game_details,
    render_filtered_grid,
    render_game_shelf,
    render_login,
    render_section_heading,
    render_topbar,
)
from data import build_category_shelves, filter_games, load_games
from styles import apply_styles


st.set_page_config(
    page_title="Triforce Table | Board game recommendations",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_styles()
games = load_games()

demo_user = ensure_demo_user()
if "authenticated_user" not in st.session_state:
    user_id, password, submitted = render_login(demo_user)
    if submitted:
        if authenticate(user_id, password):
            st.session_state.authenticated_user = user_id.strip()
            st.rerun()
        st.error("Username or password not recognized.")
    st.stop()

render_topbar()

selected_game_id = st.query_params.get("game_id")
if selected_game_id is not None:
    try:
        selected_game_id = int(selected_game_id)
    except ValueError:
        selected_game_id = None
    selected_game = games[games["ID"] == selected_game_id]
    if not selected_game.empty:
        if st.button(
            "Back to collection",
            icon=":material/arrow_back:",
            width="content",
        ):
            st.query_params.pop("game_id", None)
            st.rerun()
        render_game_details(selected_game.iloc[0])
        st.stop()

search_col, genre_col, age_col, time_col = st.columns([2.2, 1.25, 1.15, 1.15])
with search_col:
    st.markdown('<div class="filter-label">Find your next table</div>', unsafe_allow_html=True)
    search = st.text_input(
        "Search games",
        placeholder="Search by title, genre or mechanic",
        label_visibility="collapsed",
    )
with genre_col:
    st.markdown('<div class="filter-label">Genre</div>', unsafe_allow_html=True)
    genres = ["All genres"] + sorted(games["genre"].dropna().unique().tolist())
    selected_genre = st.selectbox("Genre", genres, label_visibility="collapsed")
with age_col:
    st.markdown('<div class="filter-label">Age</div>', unsafe_allow_html=True)
    selected_age = st.selectbox(
        "Age",
        ["Any age", "10+", "14+", "18+"],
        label_visibility="collapsed",
    )
with time_col:
    st.markdown('<div class="filter-label">Play time</div>', unsafe_allow_html=True)
    selected_time = st.selectbox(
        "Play time",
        ["Any length", "Under 1 hour", "1–2 hours", "2+ hours"],
        label_visibility="collapsed",
    )

filtered = filter_games(
    games,
    search,
    selected_genre,
    selected_age,
    selected_time,
)

has_filters = bool(search.strip()) or any(
    [
        selected_genre != "All genres",
        selected_age != "Any age",
        selected_time != "Any length",
    ]
)

if has_filters:
    sort_order = st.segmented_control(
        "Rating order",
        ["Highest rated", "Lowest rated"],
        default="Highest rated",
        key="filtered-rating-order",
    )
    filtered = filtered.sort_values(
        "Rating Average",
        ascending=sort_order == "Lowest rated",
    )

if has_filters and not filtered.empty:
    render_hero(filtered.iloc[0])
elif not has_filters:
    render_hero_slideshow(games)
st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

if filtered.empty:
    render_empty_state("No games match that combination. Try loosening a filter.")
elif has_filters:
    render_section_heading("Your filtered table", "Games that match your filters")
    render_filtered_grid(filtered)
else:
    render_game_shelf(
        filtered.head(10),
        "The leaderboard",
        "Top 10 · highest rated",
        "top",
        featured=True,
    )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    age_games = filtered[filtered["Min Age"] >= 10].head(20)
    render_game_shelf(age_games, "The family table", "Perfect for ages 10+", "age")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    long_games = filtered[filtered["Play Time"] >= 120].head(20)
    render_game_shelf(
        long_games,
        "The long haul",
        "Games worth clearing your evening for",
        "long",
    )

    for shelf_index, (eyebrow, title, shelf_games) in enumerate(
        build_category_shelves(filtered),
        start=1,
    ):
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        render_game_shelf(shelf_games, eyebrow, title, f"shelf-{shelf_index}")

st.markdown(
    """
    <div style="text-align:center; color:#7e806d; font-size:.75rem; margin-top:3.5rem;">
        TRIFORCE TABLE · Curated from the BoardGameGeek catalogue
    </div>
    """,
    unsafe_allow_html=True,
)