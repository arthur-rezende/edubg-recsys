from html import escape
import time
import urllib.parse # <-- Adicionado para tratar URL do usuário

import pandas as pd
import streamlit as st


def game_image(game: pd.Series) -> str:
    return str(game["image"])


def game_thumbnail(game: pd.Series) -> str:
    return str(game.get("thumbnail", game["image"]))


def game_genre(game: pd.Series) -> str:
    return str(game["genre"])


def integer_label(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def decimal_label(value: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{decimals}f}"


def duration_label(minutes: object) -> str:
    if pd.isna(minutes):
        return "NA"
    minutes = int(minutes)
    if minutes >= 120:
        return f"{minutes // 60}h+"
    return f"{minutes}m"


def player_range(game: pd.Series) -> str:
    return f"{integer_label(game['Min Players'])}–{integer_label(game['Max Players'])}"


def _tags(value: object) -> list[str]:
    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def _triforce_logo(class_name: str = "triforce-logo") -> str:
    return (
        f'<span class="{class_name}" aria-label="Triforce logo">'
        '<i class="triangle triangle-top"></i>'
        '<i class="triangle triangle-left"></i>'
        '<i class="triangle triangle-right"></i>'
        "</span>"
    )


def render_login(demo_user: dict) -> tuple[str, str, bool]:
    st.markdown(
        f"""
        <div class="login-shell">
            {_triforce_logo()}
            <div class="eyebrow">Enter the quest</div>
            <h1>Welcome to Triforce Table</h1>
            <p>Find a game worthy of your next table.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login-form"):
        user_id = st.text_input("Username", value=demo_user["user_id"])
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Enter the table",
            icon=":material/login:",
            width="stretch",
        )
    st.caption(
        f"Demo access: {demo_user['user_id']} · password 12345 · "
        f"{demo_user['rating_count']:,} ratings in the catalogue"
    )
    return user_id, password, submitted


def render_topbar() -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">{_triforce_logo("triforce-mini")} TRIFORCE TABLE</div>
            <div class="top-note">A hand-picked quest for your next game night</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _detail_url(game_id: int) -> str:
    """URL de detalhes preservando filtros e autenticação atuais."""
    params = {key: value for key, value in st.query_params.items()}
    params["game_id"] = str(game_id)
    return "/?" + urllib.parse.urlencode(params)


def render_hero(game: pd.Series) -> None:
    st.markdown(
        f"""
        <a class="hero-link" href="{_detail_url(int(game['ID']))}" target="_self">
            <section class="hero" style="background-image: linear-gradient(90deg, rgba(23,49,31,.96) 0%, rgba(29,58,37,.86) 43%, rgba(29,58,37,.3) 100%), url('{game_image(game)}');">
                <div class="hero-copy">
                    <span class="hero-badge">#1 TOP RATED · {game_genre(game)}</span>
                    <h1>{escape(str(game['Name']))}</h1>
                    <p>Build your next perfect table around a game loved by thousands of players.</p>
                    <div class="hero-stats">
                        <span class="hero-stat"><b>★ {decimal_label(game['Rating Average'], 1)}</b> top rated</span>
                        <span class="hero-stat">◷ {duration_label(game['Play Time'])}</span>
                        <span class="hero-stat">♙ {integer_label(game['Min Age'])}+</span>
                        <span class="hero-stat">♟ {player_range(game)} players</span>
                    </div>
                </div>
            </section>
        </a>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="1s")
def render_hero_slideshow(games: pd.DataFrame) -> None:
    slides = (
        games.sort_values("Rating Average", ascending=False)
        .drop_duplicates("genre")
        .reset_index(drop=True)
    )
    if slides.empty:
        return

    now = time.monotonic()
    slide_index = st.session_state.get("hero_slide_index", 0) % len(slides)
    slide_started_at = st.session_state.get("hero_slide_started_at", now)
    if now - slide_started_at >= 5:
        slide_index = (slide_index + 1) % len(slides)
        slide_started_at = now
        st.session_state.hero_slide_index = slide_index
        st.session_state.hero_slide_started_at = slide_started_at

    render_hero(slides.iloc[slide_index])
    with st.container(key="hero-controls"):
        previous_col, _, next_col = st.columns([.15, .7, .15])
        with previous_col:
            if st.button(
                "Previous",
                key="hero-previous",
                icon=":material/chevron_left:",
            ):
                st.session_state.hero_slide_index = (slide_index - 1) % len(slides)
                st.session_state.hero_slide_started_at = time.monotonic()
                st.rerun()
        with next_col:
            if st.button(
                "Next",
                key="hero-next",
                icon=":material/chevron_right:",
            ):
                st.session_state.hero_slide_index = (slide_index + 1) % len(slides)
                st.session_state.hero_slide_started_at = time.monotonic()
                st.rerun()

    dots = "".join(
        f'<span class="hero-dot{" active" if index == slide_index else ""}"></span>'
        for index in range(len(slides))
    )
    st.markdown(f'<div class="hero-dots">{dots}</div>', unsafe_allow_html=True)
    st.session_state.hero_slide_index = slide_index
    st.session_state.hero_slide_started_at = slide_started_at


def render_card(
    game: pd.Series,
    featured: bool = False,
    motion: str = "next",
    compact: bool = False,
) -> None:
    title = escape(str(game["Name"]))
    title_short = title if len(title) <= 29 else f"{title[:27]}..."
    card_class = "game-card featured-card" if featured else "game-card"
    if compact:
        card_class = f"{card_class} grid-card"
    card_class = f"{card_class} shelf-motion shelf-{motion}"
    rank = integer_label(game["BGG Rank"])
    game_id = int(game["ID"])

    st.markdown(
        f"""
        <a class="game-card-link" href="{_detail_url(game_id)}" target="_self">
            <article class="{card_class}">
                <div class="card-image" style="background-image: url('{game_thumbnail(game)}');">
                    <span class="rating-pill">★ {decimal_label(game['Rating Average'], 1)}</span>
                    <span class="rank-stamp">#{rank}</span>
                </div>
                <div class="card-copy">
                    <div class="card-kicker">{escape(game_genre(game))}</div>
                    <h3 title="{title}">{title_short}</h3>
                    <div class="card-meta">
                        <span>◷ {duration_label(game['Play Time'])}</span>
                        <span>♙ {integer_label(game['Min Age'])}+</span>
                        <span>♟ {player_range(game)}</span>
                    </div>
                </div>
            </article>
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_game_button(game: pd.Series, key: str) -> bool:
    clicked = st.button(
        "View details",
        key=key,
        icon=":material/arrow_forward:",
        width="stretch",
    )
    if clicked:
        st.query_params["game_id"] = str(int(game["ID"]))
        st.rerun()
    return clicked


def render_game_shelf(
    games: pd.DataFrame,
    eyebrow: str,
    title: str,
    key_prefix: str,
    featured: bool = False,
) -> None:
    render_section_heading(eyebrow, title)
    if games.empty:
        render_empty_state("No games match this shelf with the active filters.")
        return

    shelf_key = f"shelf_offset_{key_prefix}"
    offset = st.session_state.get(shelf_key, 0) % len(games)
    previous_col, _, next_col = st.columns([.15, .7, .15])
    with previous_col:
        if st.button("Previous", key=f"previous-{key_prefix}", icon=":material/chevron_left:"):
            st.session_state[shelf_key] = (offset - 3) % len(games)
            st.session_state[f"shelf_direction_{key_prefix}"] = "previous"
            st.rerun()
    with next_col:
        if st.button("Next", key=f"next-{key_prefix}", icon=":material/chevron_right:"):
            st.session_state[shelf_key] = (offset + 3) % len(games)
            st.session_state[f"shelf_direction_{key_prefix}"] = "next"
            st.rerun()

    direction = st.session_state.get(f"shelf_direction_{key_prefix}", "next")
    columns = st.columns(3, gap="medium")
    for index in range(min(3, len(games))):
        game = games.iloc[(offset + index) % len(games)]
        with columns[index]:
            render_card(game, featured=featured, motion=direction)
            if render_game_button(game, f"{key_prefix}-{int(game['ID'])}"):
                return


def render_filtered_grid(games: pd.DataFrame, key_prefix: str = "filtered") -> None:
    controls = st.columns([.45, .8, .8, .2, .12, .42, .28, .12], gap="small")
    rows, columns = 10, 10
    with controls[0]:
        with st.container(key=f"{key_prefix}-exhibition-control"):
            with st.popover(
                ":material/grid_view:",
                help="Exhibition: choose the number of rows and columns",
            ):
                rows = st.number_input(
                    "Rows",
                    min_value=1,
                    max_value=10,
                    value=10,
                    step=1,
                    key=f"{key_prefix}-rows",
                )
                columns = st.number_input(
                    "Columns",
                    min_value=1,
                    max_value=10,
                    value=10,
                    step=1,
                    key=f"{key_prefix}-columns",
                )

    rows = min(max(int(rows), 1), 10)
    columns = min(max(int(columns), 1), 10)
    page_size = rows * columns
    page_count = max(1, (len(games) + page_size - 1) // page_size)
    page_key = f"{key_prefix}-page"
    page_index = min(st.session_state.get(page_key, 0), page_count - 1)

    if page_count > 1:
        with controls[4]:
            if st.button(
                "Previous page",
                key=f"{key_prefix}-previous-page",
                icon=":material/chevron_left:",
            ):
                st.session_state[page_key] = (page_index - 1) % page_count
                st.rerun()
        with controls[5]:
            st.markdown(
                f'<div class="grid-page-label">Page {page_index + 1} of {page_count}</div>',
                unsafe_allow_html=True,
            )
        with controls[6]:
            target_page = st.number_input(
                "Go to page",
                min_value=1,
                max_value=page_count,
                value=page_index + 1,
                step=1,
                key=f"{key_prefix}-jump-page",
            )
            if target_page - 1 != page_index:
                st.session_state[page_key] = target_page - 1
                st.rerun()
        with controls[7]:
            if st.button(
                "Next page",
                key=f"{key_prefix}-next-page",
                icon=":material/chevron_right:",
            ):
                st.session_state[page_key] = (page_index + 1) % page_count
                st.rerun()

    page_start = page_index * page_size
    page_games = games.iloc[page_start:page_start + page_size]
    grid_columns = st.columns(columns, gap="small")
    for index, (_, game) in enumerate(page_games.iterrows()):
        with grid_columns[index % columns]:
            render_card(game, compact=True)


def render_game_details(game: pd.Series) -> None:
    categories = _tags(game["Domains"])
    mechanics = _tags(game["Mechanics"])
    category_text = ", ".join(categories[:3]) or "Board game"
    mechanic_text = ", ".join(mechanics[:4]) or "varied gameplay"
    year = integer_label(game["Year Published"])
    rank = integer_label(game["BGG Rank"])
    complexity = (
        f"{game['Complexity Average']:.2f} / 5"
        if pd.notna(game["Complexity Average"])
        else "NA"
    )
    owned = integer_label(game["Owned Users"])
    users_rated = integer_label(game["Users Rated"])

    st.markdown(
        f"""
        <div class="detail-layout">
            <div class="detail-cover" style="background-image: url('{game_image(game)}');"></div>
            <div class="detail-main">
                <div class="eyebrow">{escape(game_genre(game))} · #{rank}</div>
                <h1>{escape(str(game['Name']))}</h1>
                <p class="detail-description">A board game in the {escape(category_text.lower())} category, built around {escape(mechanic_text.lower())}. A strong pick for tables looking for a focused, replayable experience.</p>
                <div class="detail-rating"><strong>★ {game['Rating Average']:.2f}</strong><span>{users_rated} ratings</span></div>
                <div class="detail-stats">
                    <div><span>Players</span><strong>{player_range(game)}</strong></div>
                    <div><span>Minimum age</span><strong>{integer_label(game['Min Age'])}+</strong></div>
                    <div><span>Play time</span><strong>{duration_label(game['Play Time'])}</strong></div>
                    <div><span>Published</span><strong>{year}</strong></div>
                </div>
            </div>
        </div>
        <div class="detail-section">
            <div class="detail-section-label">Categories</div>
            <div class="tag-list">{''.join(f'<span>{escape(tag.title())}</span>' for tag in categories) or '<span>Uncategorized</span>'}</div>
        </div>
        <div class="detail-section">
            <div class="detail-section-label">Mechanics</div>
            <div class="tag-list">{''.join(f'<span>{escape(tag.title())}</span>' for tag in mechanics) or '<span>Not listed</span>'}</div>
        </div>
        <div class="detail-section detail-facts">
            <div><span>Community rank</span><strong>#{rank}</strong></div>
            <div><span>Complexity</span><strong>{complexity}</strong></div>
            <div><span>Owned copies</span><strong>{owned}</strong></div>
            <div><span>BoardGameGeek ID</span><strong>{int(game['ID'])}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(eyebrow: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <div>
                <div class="eyebrow">{escape(eyebrow)}</div>
                <h2>{escape(title)}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="empty-state">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def render_bgg_attribution() -> None:
    # Atribuição exigida pelos termos da BGG XML API (uso não comercial)
    st.markdown(
        """
        <div class="bgg-attribution">
            TRIFORCE TABLE · Dados e imagens:
            <a href="https://boardgamegeek.com" target="_blank" rel="noopener">Powered by BoardGameGeek</a>
        </div>
        """,
        unsafe_allow_html=True,
    )