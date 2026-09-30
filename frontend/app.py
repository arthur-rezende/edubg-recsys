import streamlit as st

from api import (
    api_is_up,
    get_home_recommendations,
    get_recommendations,
    send_evaluation,
)
from auth import authenticate, ensure_demo_user
from components import (
    render_bgg_attribution,
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
from data import filter_games, load_games
from styles import apply_styles


st.set_page_config(
    page_title="Triforce Table | Board game recommendations",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_styles()


@st.cache_data(ttl=300, show_spinner=False)
def load_home_recommendations(user_id: str, top_k: int = 15):
    return get_home_recommendations(user_id, top_k=top_k)


@st.cache_data(ttl=5, show_spinner=False)
def backend_is_up_cached() -> bool:
    return api_is_up()


games = load_games()
demo_user = ensure_demo_user()

if "auth_user" in st.query_params:
    st.session_state.authenticated_user = st.query_params["auth_user"]

if "authenticated_user" not in st.session_state:
    user_id, password, submitted = render_login(demo_user)

    if submitted:
        if authenticate(user_id, password):
            st.session_state.authenticated_user = user_id.strip()
            st.query_params["auth_user"] = user_id.strip()
            st.rerun()

        st.error("Username or password not recognized.")

    st.stop()

if st.query_params.get("auth_user") != st.session_state.authenticated_user:
    st.query_params["auth_user"] = st.session_state.authenticated_user

render_topbar()

backend_online = backend_is_up_cached()

with st.expander("Recomendações para sua turma", expanded=False):
    if not backend_online:
        st.warning("Backend offline.")
    else:
        with st.form("recomendar"):
            modo = st.radio(
                "Modo",
                ["collaborative", "context"],
                horizontal=True,
            )

            c1, c2, c3 = st.columns(3)
            idade = c1.number_input(
                "Idade dos alunos",
                5,
                99,
                12,
            )
            qtd = c2.number_input(
                "Quantidade de alunos",
                1,
                100,
                4,
            )
            tempo = c3.number_input(
                "Tempo disponível (min)",
                5,
                600,
                50,
            )
            objetivo = st.text_input(
                "Objetivo pedagógico",
                "Desenvolver raciocínio lógico",
            )
            contexto = st.text_input(
                "Contexto da aula",
                "Turma do ensino fundamental",
            )
            enviar = st.form_submit_button("Recomendar")

        if enviar:
            with st.spinner("Calculando recomendações..."):
                recs = get_recommendations({
                    "user_id": st.session_state.authenticated_user,
                    "mode": modo,
                    "idade_alunos": idade,
                    "quantidade_alunos": qtd,
                    "tempo_disponivel": tempo,
                    "objetivo_pedagogico": objetivo,
                    "contexto": contexto,
                    "top_k": 5,
                })

            ids = [r["game_id"] for r in recs]
            game_ids = set(games["ID"])
            recomendados = games.set_index("ID").loc[
                [i for i in ids if i in game_ids]
            ].reset_index()

            render_filtered_grid(
                recomendados,
                key_prefix="recs",
            )


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

        with st.form("avaliar"):
            nota = st.slider(
                "Sua avaliação",
                0.0,
                10.0,
                7.0,
                0.5,
            )

            if st.form_submit_button("Enviar avaliação"):
                try:
                    send_evaluation({
                        "user_id": st.session_state.authenticated_user,
                        "game_id": selected_game_id,
                        "rating": nota,
                    })

                    load_home_recommendations.clear()
                    st.success("Avaliação salva!")
                    st.rerun()
                except Exception as erro:
                    st.error(f"Não foi possível salvar: {erro}")

        render_bgg_attribution()
        st.stop()

search_col, genre_col, age_col, time_col = st.columns(
    [2.2, 1.25, 1.15, 1.15]
)

# Filtros persistidos na URL, para sobreviverem à navegação entre páginas.
_filter_defaults = {
    "search": "",
    "genre": "All genres",
    "age": "Any age",
    "time": "Any length",
}


def _filter_value(key: str) -> str:
    return st.query_params.get(key) or _filter_defaults[key]


search_value = _filter_value("search")
genre_value = _filter_value("genre")
age_value = _filter_value("age")
time_value = _filter_value("time")

with search_col:
    st.markdown(
        '<div class="filter-label">Find your next table</div>',
        unsafe_allow_html=True,
    )
    search = st.text_input(
        "Search games",
        placeholder="Search by title, genre or mechanic",
        label_visibility="collapsed",
        value=search_value,
    )

with genre_col:
    st.markdown(
        '<div class="filter-label">Genre</div>',
        unsafe_allow_html=True,
    )
    genres = ["All genres"] + sorted(
        games["genre"].dropna().unique().tolist()
    )
    genre_index = genres.index(genre_value) if genre_value in genres else 0
    selected_genre = st.selectbox(
        "Genre",
        genres,
        index=genre_index,
        label_visibility="collapsed",
    )

with age_col:
    st.markdown(
        '<div class="filter-label">Age</div>',
        unsafe_allow_html=True,
    )
    age_options = ["Any age", "10+", "14+", "18+"]
    age_index = age_options.index(age_value) if age_value in age_options else 0
    selected_age = st.selectbox(
        "Age",
        age_options,
        index=age_index,
        label_visibility="collapsed",
    )

with time_col:
    st.markdown(
        '<div class="filter-label">Play time</div>',
        unsafe_allow_html=True,
    )
    time_options = ["Any length", "Under 1 hour", "1–2 hours", "2+ hours"]
    time_index = time_options.index(time_value) if time_value in time_options else 0
    selected_time = st.selectbox(
        "Play time",
        time_options,
        index=time_index,
        label_visibility="collapsed",
    )

# Sincroniza os filtros de volta para a URL (só quando houver mudança).
_chosen = {
    "search": search,
    "genre": selected_genre,
    "age": selected_age,
    "time": selected_time,
}
for _key in ["search", "genre", "age", "time"]:
    _normalized = _chosen[_key] if _chosen[_key] != _filter_defaults[_key] else ""
    if (st.query_params.get(_key) or "") != _normalized:
        if _normalized:
            st.query_params[_key] = _normalized
        elif _key in st.query_params:
            del st.query_params[_key]

filtered = filter_games(
    games,
    search,
    selected_genre,
    selected_age,
    selected_time,
)

has_filters = bool(search.strip()) or any([
    selected_genre != "All genres",
    selected_age != "Any age",
    selected_time != "Any length",
])

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

st.markdown(
    '<div class="section-rule"></div>',
    unsafe_allow_html=True,
)

if filtered.empty:
    render_empty_state(
        "No games match that combination. Try loosening a filter."
    )

elif has_filters:
    render_section_heading(
        "Your filtered table",
        "Games that match your filters",
    )
    render_filtered_grid(filtered)

else:
    if not backend_online:
        render_empty_state(
            "Backend offline. Start the API to load your recommendations."
        )
    else:
        try:
            with st.spinner("Montando suas recomendações..."):
                home = load_home_recommendations(
                    st.session_state.authenticated_user,
                    top_k=15,
                )

            sections = home.get("sections", [])

            if not sections:
                render_empty_state(
                    "Ainda não há recomendações suficientes para este usuário."
                )
            else:
                for index, section in enumerate(sections):
                    section_ids = [
                        item["game_id"]
                        for item in section.get("items", [])
                    ]
                    game_ids = set(games["ID"])
                    shelf_games = games.set_index("ID").loc[
                        [i for i in section_ids if i in game_ids]
                    ].reset_index()

                    render_game_shelf(
                        shelf_games,
                        (
                            "Item-Based"
                            if section.get("type") == "item_based"
                            else "User-Based"
                        ),
                        section.get("title", "Recomendações"),
                        f"home-{index}",
                    )

                    if index < len(sections) - 1:
                        st.markdown(
                            '<div class="section-rule"></div>',
                            unsafe_allow_html=True,
                        )
        except Exception as erro:
            st.error(
                f"Não foi possível carregar as recomendações: {erro}"
            )

render_bgg_attribution()
