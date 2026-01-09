import pandas as pd
import numpy as np
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================================================
# LOAD IMDb DATA
# =================================================
BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

basics = pd.read_csv(BASICS_URL, sep="\t", compression="gzip", na_values="\\N", low_memory=False)
ratings = pd.read_csv(RATINGS_URL, sep="\t", compression="gzip", na_values="\\N")

movies = basics[
    (basics["titleType"] == "movie") &
    (basics["isAdult"] == 0)
][["tconst", "primaryTitle", "startYear", "genres"]]

movies = movies.merge(ratings, on="tconst", how="left")

movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce")
movies = movies.dropna(subset=["averageRating", "startYear"])
movies["startYear"] = movies["startYear"].astype(int)

movies.rename(columns={
    "primaryTitle": "title",
    "averageRating": "rating",
    "numVotes": "votes",
    "startYear": "year"
}, inplace=True)

movies["title"] = movies["title"].fillna("")
movies["genres"] = movies["genres"].fillna("")
movies["clean_title"] = movies["title"].str.lower().str.strip()

movies["metadata"] = (
    movies["genres"].str.replace(",", " ").str.lower()
    + " " +
    movies["clean_title"]
)

movies = movies.drop_duplicates(subset="clean_title").reset_index(drop=True)

# =================================================
# TF-IDF
# =================================================
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
tfidf_matrix = tfidf.fit_transform(movies["metadata"])

# =================================================
# WEIGHTED RANK
# =================================================
def add_weighted_score(df):
    df = df.copy()
    df["rating_norm"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min())
    df["votes_norm"] = np.log1p(df["votes"]) / np.log1p(df["votes"].max())
    df["score"] = 0.6 * df["rating_norm"] + 0.4 * df["votes_norm"]
    return df.sort_values("score", ascending=False)

# =================================================
# SEARCH
# =================================================
def search_movie(query, limit=100):
    if not query:
        return pd.DataFrame()
    df = movies[movies["clean_title"].str.contains(query.lower(), na=False)]
    return add_weighted_score(df).head(limit)[
        ["title", "year", "genres", "rating", "votes"]
    ]

def search_by_genre(genre, limit=100):
    df = movies[movies["genres"].str.lower().str.contains(genre.lower(), na=False)]
    return add_weighted_score(df).head(limit)[
        ["title", "year", "genres", "rating", "votes"]
    ]

# =================================================
# SIMILARITY
# =================================================
def get_movie_index(query):
    if not query:
        return None
    matches = movies[movies["clean_title"].str.contains(query.lower(), na=False)]
    return None if matches.empty else matches.index[0]

def compute_similarity_for_index(idx, top_n=50):
    vec = tfidf_matrix[idx]
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    return scores.argsort()[::-1][1:top_n + 1]

def get_similar_movies(query, top_n=50):
    idx = get_movie_index(query)
    if idx is None:
        return pd.DataFrame()
    indices = compute_similarity_for_index(idx, top_n)
    return add_weighted_score(movies.iloc[indices])[
        ["title", "year", "genres", "rating", "votes"]
    ]

# =================================================
# PRECISION / RECALL
# =================================================
def precision_recall_at_k(k=10, rating_threshold=7.0):
    relevant = movies[movies["rating"] >= rating_threshold]
    precisions, recalls = [], []

    for idx in range(min(100, len(movies))):
        indices = compute_similarity_for_index(idx, k)
        recommended = set(indices)
        relevant_set = set(relevant.index)

        tp = len(recommended & relevant_set)
        precisions.append(tp / k)
        recalls.append(tp / len(relevant_set))

    return np.mean(precisions), np.mean(recalls)

# =================================================
# POSTERS
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
