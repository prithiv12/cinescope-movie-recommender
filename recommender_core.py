import pandas as pd
import numpy as np
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collaborative_filtering import predict_rating

# =========================================================
# LOAD DATA
# =========================================================
movies = pd.read_csv("movies_with_description.csv")

# =========================================================
# CLEAN DATA
# =========================================================
movies["title"] = movies["title"].fillna("")
movies["genres"] = movies["genres"].fillna("")
movies["description"] = movies["description"].fillna("")
movies["year"] = movies["year"].fillna(0).astype(int)
movies["rating"] = movies["rating"].fillna(0)
movies["votes"] = movies["votes"].fillna(0)

movies["clean_title"] = movies["title"].str.lower().str.strip()
movies = movies.drop_duplicates(subset="clean_title").reset_index(drop=True)

# =========================================================
# METADATA CREATION
# =========================================================
movies["metadata"] = (
    (movies["genres"].str.replace(",", " ").str.lower() + " ") * 3
    + movies["description"].str.lower()
)

# =========================================================
# TF-IDF VECTORIZATION
# =========================================================
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85
)

tfidf_matrix = tfidf.fit_transform(movies["metadata"])

# =========================================================
# NORMALIZATION HELPERS
# =========================================================
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def normalize_minmax(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin + 1e-6)

# =========================================================
# SEARCH FUNCTIONS
# =========================================================
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

# =========================================================
# SIMILAR MOVIES
# =========================================================
def get_movie_index(query):
    matches = movies[movies["clean_title"].str.contains(query.lower(), na=False)]
    return None if matches.empty else matches.index[0]

def get_similar_movies(query, top_n=50):
    idx = get_movie_index(query)
    if idx is None:
        return pd.DataFrame()

    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    df = movies.copy()
    df["similarity"] = similarity
    df = df[df.index != idx]

    df["rating_norm"] = normalize(df["rating"])
    df["votes_norm"] = normalize(np.log1p(df["votes"]))

    df["final_score"] = (
        0.6 * df["similarity"]
        + 0.25 * df["rating_norm"]
        + 0.15 * df["votes_norm"]
    )

    return df.sort_values("final_score", ascending=False).head(top_n)[
        ["title", "year", "genres", "rating", "votes"]
    ]

# =========================================================
# HEATMAP SUPPORT
# =========================================================
def compute_similarity_for_index(idx, top_n=10):
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = scores.argsort()[::-1]
    similar_indices = similar_indices[similar_indices != idx]
    return similar_indices[:top_n]

# =========================================================
# HYBRID RECOMMENDATION (CORRECTED)
# =========================================================
def get_hybrid_recommendations(base_movie, user_id=1, alpha=0.6, top_n=20):
    idx = get_movie_index(base_movie)
    if idx is None:
        return pd.DataFrame()

    similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    df = movies.copy()
    df["similarity"] = similarity
    df = df[df.index != idx]

    # Content relevance
    sim_min, sim_max = df["similarity"].min(), df["similarity"].max()
    df["content_score"] = df["similarity"].apply(
        lambda x: normalize_minmax(x, sim_min, sim_max)
    )

    # Quality modifier
    df["rating_norm"] = normalize(df["rating"])
    df["votes_norm"] = normalize(np.log1p(df["votes"]))
    df["quality_score"] = 0.6 * df["rating_norm"] + 0.4 * df["votes_norm"]

    df["content_score"] = (
        0.7 * df["content_score"] + 0.3 * df["quality_score"]
    )

    # Collaborative filtering
    cf_vals = []
    for i in df.index:
        cf_vals.append(predict_rating(user_id, i))

    df["cf_raw"] = cf_vals
    valid_cf = df["cf_raw"].dropna()

    if not valid_cf.empty:
        cf_min, cf_max = valid_cf.min(), valid_cf.max()
        df["cf_score"] = df["cf_raw"].apply(
            lambda x: normalize_minmax(x, cf_min, cf_max)
            if x is not None else None
        )
    else:
        df["cf_score"] = None

    # Fusion
    def fuse(row):
        if row["cf_score"] is None:
            return row["content_score"]
        return alpha * row["content_score"] + (1 - alpha) * row["cf_score"]

    df["hybrid_score"] = df.apply(fuse, axis=1)

    return df.sort_values("hybrid_score", ascending=False).head(top_n)[
        ["title", "year", "genres", "rating", "votes", "hybrid_score"]
    ]

# =========================================================
# EVALUATION METRICS
# =========================================================
def precision_recall_at_k(k=10, rating_threshold=7.0):
    precisions = []
    recalls = []

    sample_indices = movies.sample(
        n=min(100, len(movies)), random_state=42
    ).index

    relevant_movies = movies[movies["rating"] >= rating_threshold]

    for idx in sample_indices:
        base_movie = movies.loc[idx, "title"]
        recs = get_similar_movies(base_movie, top_n=k)

        if recs.empty:
            continue

        relevant_recs = recs[recs["rating"] >= rating_threshold]

        precision = len(relevant_recs) / len(recs)
        recall = len(relevant_recs) / max(1, len(relevant_movies))

        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))

# =========================================================
# POSTER FETCH
# =========================================================
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
