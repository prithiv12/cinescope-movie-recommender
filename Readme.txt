CineScope – Movie Recommendation System

Project Description:
This project is an intelligent movie recommendation system developed as a minor project for the MCA (Artificial Intelligence) program at Amrita Vishwa Vidyapeetham. The system helps users discover movies based on content similarity, genre preference, and movie quality using real-world datasets.

The project uses IMDb datasets for movie metadata and ratings, and TMDB API for movie descriptions and posters. Natural Language Processing techniques are applied to compute similarity between movies, and recommendations are displayed through a Streamlit web application.

Project Objectives:

To build a content-based movie recommendation system

To use NLP techniques for semantic similarity analysis

To combine genre, description, rating, and popularity for better recommendations

To evaluate the recommendation quality using standard metrics

To provide an interactive and explainable user interface

Files Description:

app.py
This file contains the Streamlit web application. It handles user interaction such as movie search, genre browsing, similar movie recommendations, explainability heatmaps, and evaluation metrics.

recommender_core.py
This file contains the core recommendation logic. It performs data cleaning, TF-IDF vectorization, cosine similarity calculation, multi-stage ranking, and evaluation using Precision@K and Recall@K.

moviedataset_downloadandpreprocessing.py
This file is used to download and preprocess movie data. It fetches IMDb movie metadata and ratings, retrieves movie descriptions from TMDB API, applies caching, and generates the final dataset.

movies_with_description.csv
This is the final processed dataset containing movie titles, genres, ratings, votes, release year, and movie descriptions.

tmdb_cache.csv
This file stores cached TMDB API responses to reduce repeated API calls and improve runtime efficiency.

requirements.txt
This file lists all the required Python libraries needed to run the project.

Algorithms and Techniques Used:

TF-IDF Vectorization for text feature extraction

Cosine Similarity for measuring movie similarity

Genre-based boosting and filtering

Rating and popularity normalization

Precision@K and Recall@K for evaluation

How to Run the Project:

Install the required libraries using the following command:
pip install -r requirements.txt

Run the Streamlit application using:
python -m streamlit run app.py

The application will open in a web browser.

Dataset Information:
IMDb public datasets are used for movie metadata and ratings.
TMDB API is used to fetch movie descriptions and posters.
All API responses are cached locally to avoid repeated requests.
