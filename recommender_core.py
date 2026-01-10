import pandas as pd
import numpy as np
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================================================
# LOAD DATASET (WITH DESCRIPTION)
# =================================================
movies = pd.read_csv("movies_with_description.csv")

# =================================================
# BASIC CLEANING
# =================================================
movies["title"] = movies["title"].fillna("")
movies["genres"] = movies["genres"].fillna("")
movies["description"] = movies["description"].fillna("")
movies["year"] = movies["year"].fillna(0).astype(int)
movies["rating"] = movies["rating"].fillna(0)
movies["votes"] = movies["votes"].fillna(0)

movies["clean_title"] = movies["title"].str.lower().str.strip()
movies = movies.drop_duplicates(subset="clean_title").reset_index(drop=True)

# =================================================
# METADATA (GENRE BOOSTING + DESCRIPTION)
# =================================================
movies["metadata"] = (
    (movies["genres"].str.replace(",", " ").str.lower() + " ") * 3
    + movies["description"].str.lower()
)

# =================================================
# TF-IDF VECTORIZATION (NOISE CONTROL)
# =================================================
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85
)

tfidf_matrix = tfidf.fit_transform(movies["metadata"])

# =================================================
# QUALITY NORMALIZATION
# =================================================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

# =================================================
# SEARCH FUNCTIONS (UNCHANGED)
# =================================================
def search_movie(query, limit=100):
    if not query:
        return pd.DataFrame()

    df = movies[movies["clean_title"].str.contains(query.lower(), na=False)]
    df["rating_norm"] = normalize(df["rating"])
    df["votes_norm"] = normalize(np.log1p(df["votes"]))
    df["score"] = 0.6 * df["rating_norm"] + 0.4 * df["votes_norm"]

    return df.sort_values("score", ascending=False).head(limit)[
        ["title", "year", "genres", "rating", "votes"]
    ]

def search_by_genre(genre, limit=100):
    if not genre:
        return pd.DataFrame()

    df = movies[movies["genres"].str.lower().str.contains(genre.lower(), na=False)]
    df["rating_norm"] = normalize(df["rating"])
    df["votes_norm"] = normalize(np.log1p(df["votes"]))
    df["score"] = 0.6 * df["rating_norm"] + 0.4 * df["votes_norm"]

    return df.sort_values("score", ascending=False).head(limit)[
        ["title", "year", "genres", "rating", "votes"]
    ]

# =================================================
# SIMILAR MOVIES (FULLY FINE-TUNED)
# =================================================
def get_movie_index(query):
    if not query:
        return None

    matches = movies[movies["clean_title"].str.contains(query.lower(), na=False)]
    return None if matches.empty else matches.index[0]

def get_similar_movies(query, top_n=50):
    idx = get_movie_index(query)
    if idx is None:
        return pd.DataFrame()

    # ---------- Similarity ----------
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    df = movies.copy()
    df["similarity"] = similarity_scores
    df = df[df.index != idx]

    # ---------- Genre Overlap Filter ----------
    base_genres = set(movies.loc[idx, "genres"].lower().split(","))

    def genre_overlap(genres):
        g = set(genres.lower().split(","))
        return len(base_genres & g) / max(len(base_genres), 1)

    df["genre_overlap"] = df["genres"].apply(genre_overlap)
    df = df[df["genre_overlap"] >= 0.3]

    # ---------- Year Similarity ----------
    base_year = movies.loc[idx, "year"]
    df["year_diff"] = abs(df["year"] - base_year)
    df["year_score"] = 1 / (1 + df["year_diff"])

    # ---------- Quality ----------
    df["rating_norm"] = normalize(df["rating"])
    df["votes_norm"] = normalize(np.log1p(df["votes"]))

    # ---------- Popularity Penalty ----------
    df["popularity_penalty"] = 1 / (1 + np.log1p(df["votes"]))

    # ---------- Stage 1: Similarity Shortlist ----------
    df = df.sort_values("similarity", ascending=False).head(200)

    # ---------- Stage 2: Final Ranking ----------
    df["final_score"] = (
        0.45 * df["similarity"] +
        0.25 * df["rating_norm"] +
        0.15 * df["votes_norm"] +
        0.10 * df["year_score"]
    ) * df["popularity_penalty"]

    return df.sort_values("final_score", ascending=False).head(top_n)[
        ["title", "year", "genres", "rating", "votes"]
    ]

# =================================================
# HEATMAP SUPPORT (UNCHANGED)
# =================================================
def compute_similarity_for_index(idx, top_n=10):
    vec = tfidf_matrix[idx]
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    return scores.argsort()[::-1][1:top_n + 1]

# =================================================
# EVALUATION (GENRE-AWARE PRECISION@K)
# =================================================
def precision_recall_at_k(k=10, rating_threshold=7.0):
    precisions = []
    recalls = []

    # Relevant movies = good quality movies
    relevant = movies[movies["rating"] >= rating_threshold]
    relevant_indices = set(relevant.index)

    if not relevant_indices:
        return 0.0, 0.0

    for idx in range(min(50, len(movies))):
        recs = get_similar_movies(movies.loc[idx, "title"], k)

        if recs.empty:
            continue

        rec_indices = set(
            movies[movies["title"].isin(recs["title"])].index
        )

        true_positives = len(rec_indices & relevant_indices)

        precisions.append(true_positives / k)
        recalls.append(true_positives / len(relevant_indices))

    precision = np.mean(precisions) if precisions else 0.0
    recall = np.mean(recalls) if recalls else 0.0

    return precision, recall


# =================================================
# POSTER FETCH (TMDB)
# =================================================
TMDB_API_KEY = "1a1c768f0babc9f17bf0ea5ffcc011f2"

def get_poster_url(title):
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": title},
            timeout=5
        ).json()
        return f"https://image.tmdb.org/t/p/w500{r['results'][0]['poster_path']}"
    except:
        return "https://dummyimage.com/300x450/000/fff&text=No+Poster"
