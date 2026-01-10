from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from recommender_core import *

RESULTS_PER_PAGE = 10

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="🎬 CineScope – Movie Recommender",
    layout="wide"
)

# ==================================================
# ADVANCED UI CSS (UNCHANGED + INFO CARD)
# ==================================================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}
.block-container {
    padding-top: 1.5rem;
}
h1, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #FFFFF0;
}
h2 {
    font-family: 'Segoe UI', sans-serif;
    color: #111213;
}
.card {
    background: rgba(180, 7, 16, 0.8);
    backdrop-filter: blur(12px);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.45);
    margin-bottom: 20px;
}
.poster {
    border-radius: 14px;
}
.badge {
    display: inline-block;
    padding: 6px 12px;
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: white;
    border-radius: 999px;
    font-weight: 700;
}
.empty-box {
    text-align: center;
    padding: 35px;
    background: rgba(2,6,23,0.7);
    border-radius: 18px;
    color: #94a3b8;
}
div.stButton > button {
    background: rgba(180, 7, 16, 0.8);
    color: white;
    border-radius: 999px;
    padding: 0.45rem 1.5rem;
    border: none;
    font-weight: 700;
}
.info-card {
    background: rgba(15, 23, 42, 0.85);
    border-left: 5px solid #f59e0b;
    padding: 18px 22px;
    border-radius: 14px;
    margin: 15px 0;
    color: #e5e7eb;
    font-size: 0.95rem;
    line-height: 1.6;
}
.info-card strong {
    color: #fbbf24;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================
defaults = {
    "logged_in": False,
    "username": "",
    "search_page": 0,
    "genre_page": 0,
    "similar_page": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================================================
# LOGIN
# ==================================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🎬 CineScope</h1>", unsafe_allow_html=True)
    name = st.text_input("Enter your name")
    if st.button("Enter App") and name.strip():
        st.session_state.logged_in = True
        st.session_state.username = name
    st.stop()

# ==================================================
# HEADER
# ==================================================
st.markdown(f"""
<div style="background:#E50914;padding:25px;border-radius:20px;">
<h1>👋 Welcome, {st.session_state.username}</h1>
<p style="color:#cbd5e1;">Smart movie discovery using IMDb + NLP</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 🔍 MOVIE SEARCH
# ==================================================
st.header("🔍 Search Movies")
query = st.text_input("Movie title")

if st.button("Search Movie"):
    st.session_state.search_page = 0
    st.session_state.search_results = search_movie(query)

if query and "search_results" in st.session_state:
    results = st.session_state.search_results

    if results.empty:
        st.markdown("<div class='empty-box'>❌ Movie not found</div>", unsafe_allow_html=True)
    else:
        start = st.session_state.search_page * RESULTS_PER_PAGE
        end = start + RESULTS_PER_PAGE

        for _, row in results.iloc[start:end].iterrows():
            c1, c2 = st.columns([1, 4])
            with c1:
                st.image(get_poster_url(row["title"]), width=140)
            with c2:
                st.markdown(f"""
                <div class="card">
                    <h3>{row['title']} ({row['year']})</h3>
                    <p>🎭 {row['genres']}</p>
                    <span class="badge">⭐ {row['rating']}</span>
                    <p>👥 {int(row['votes'])} votes</p>
                </div>
                """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        if col1.button("⬅ Previous", disabled=st.session_state.search_page == 0):
            st.session_state.search_page -= 1
            st.rerun()
        if col2.button("Next ➡", disabled=end >= len(results)):
            st.session_state.search_page += 1
            st.rerun()

# ==================================================
# 🎭 BROWSE BY GENRE (RESTORED)
# ==================================================
st.header("🎭 Browse by Genre")
genre = st.text_input("Genre (Action, Drama, Sci-Fi)")

if st.button("Search Genre"):
    st.session_state.genre_page = 0
    st.session_state.genre_results = search_by_genre(genre)

if genre and "genre_results" in st.session_state:
    results = st.session_state.genre_results

    if results.empty:
        st.markdown("<div class='empty-box'>❌ No movies found</div>", unsafe_allow_html=True)
    else:
        start = st.session_state.genre_page * RESULTS_PER_PAGE
        end = start + RESULTS_PER_PAGE

        for _, row in results.iloc[start:end].iterrows():
            c1, c2 = st.columns([1, 4])
            with c1:
                st.image(get_poster_url(row["title"]), width=140)
            with c2:
                st.markdown(f"""
                <div class="card">
                    <h3>{row['title']}</h3>
                    <span class="badge">⭐ {row['rating']}</span>
                </div>
                """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        if col1.button("⬅ Previous Genre", disabled=st.session_state.genre_page == 0):
            st.session_state.genre_page -= 1
            st.rerun()
        if col2.button("Next Genre ➡", disabled=end >= len(results)):
            st.session_state.genre_page += 1
            st.rerun()

# ==================================================
# 🎥 SIMILAR MOVIES (WITH PAGINATION)
# ==================================================
st.header("🎥 Similar Movies")
base_movie = st.text_input("Base movie")

if st.button("Find Similar Movies"):
    st.session_state.similar_page = 0
    st.session_state.similar_results = get_similar_movies(base_movie)

if base_movie and "similar_results" in st.session_state:
    results = st.session_state.similar_results

    if results.empty:
        st.markdown("<div class='empty-box'>❌ Movie not found</div>", unsafe_allow_html=True)
    else:
        start = st.session_state.similar_page * RESULTS_PER_PAGE
        end = start + RESULTS_PER_PAGE

        for _, row in results.iloc[start:end].iterrows():
            c1, c2 = st.columns([1, 4])
            with c1:
                st.image(get_poster_url(row["title"]), width=140)
            with c2:
                st.markdown(f"""
                <div class="card">
                    <h3>{row['title']}</h3>
                    <span class="badge">⭐ {row['rating']}</span>
                </div>
                """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        if col1.button("⬅ Previous Similar", disabled=st.session_state.similar_page == 0):
            st.session_state.similar_page -= 1
            st.rerun()
        if col2.button("Next Similar ➡", disabled=end >= len(results)):
            st.session_state.similar_page += 1
            st.rerun()

# ==================================================
# 📊 EXPLAINABILITY & EVALUATION
# ==================================================
st.header("📊 Explainability & Evaluation")

st.markdown("""
<div class="info-card">
<strong>Similarity Heatmap:</strong><br>
The heatmap visualizes how similar recommended movies are to each other using cosine
similarity over genre and description features. Darker cells indicate stronger similarity,
helping explain recommendation behavior.
</div>
""", unsafe_allow_html=True)

if st.button("Show Similarity Heatmap"):
    idx = get_movie_index(base_movie)
    if idx is not None:
        top_idx = compute_similarity_for_index(idx, top_n=10)
        labels = movies.iloc[top_idx]["title"]
        matrix = cosine_similarity(tfidf_matrix[top_idx], tfidf_matrix[top_idx])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="magma")
        plt.xticks(rotation=90)
        st.pyplot(fig)

st.markdown("""
<div class="info-card">
<strong>K Value:</strong> Number of top recommendations considered during evaluation.<br>
<strong>Rating Threshold:</strong> Minimum IMDb rating for a movie to be treated as relevant.
</div>
""", unsafe_allow_html=True)

k = st.slider("K value", 5, 20, 10)
threshold = st.slider("Rating threshold", 6.0, 9.0, 7.0)

if st.button("Evaluate Model"):
    p, r = precision_recall_at_k(k, threshold)
    st.metric("Precision@K", f"{p:.3f}")
    st.metric("Recall@K", f"{r:.3f}")

    st.markdown("""
    <div class="info-card">
    <strong>Precision@K:</strong> Fraction of recommended movies in the top K that are relevant.<br><br>
    <strong>Recall@K:</strong> Fraction of all relevant movies that appear in the top K.<br><br>
    <strong>Effect of Sliders:</strong> Increasing K improves recall but may reduce precision.
    Increasing the rating threshold makes evaluation stricter, often reducing recall while
    slightly improving precision. These sliders affect only evaluation metrics and do not
    change recommendation results.
    </div>
    """, unsafe_allow_html=True)




