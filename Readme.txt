CineScope – Hybrid Movie Recommendation System (Version 1)

Project Description:
CineScope is a hybrid movie recommendation system developed as a minor project for the MCA (Artificial Intelligence) program at Amrita Vishwa Vidyapeetham. This version implements a hybrid approach by combining content-based filtering with collaborative filtering to recommend movies based on both relevance and user preferences.

The system uses traditional Natural Language Processing (NLP) techniques to analyze movie metadata such as genres and descriptions. User preferences are learned from historical rating data using collaborative filtering. The application is deployed as an interactive Streamlit web interface with evaluation and explainability features.

--------------------------------------------------

Project Objectives:

• To design a hybrid movie recommendation system  
• To apply traditional NLP techniques for movie similarity analysis  
• To learn user preferences using collaborative filtering  
• To combine relevance and personalization using hybrid score fusion  
• To evaluate the recommendation system using standard metrics  
• To provide an interactive and explainable user interface  

--------------------------------------------------

System Architecture Overview:

The system consists of three main components:

1. Content-Based Recommendation (Traditional NLP):
   Movie genres and descriptions are converted into numerical representations using TF-IDF vectorization. Cosine similarity is used to measure similarity between movies based on these representations.

2. Collaborative Filtering:
   User preferences are learned using matrix factorization on the MovieLens user–movie ratings dataset. The model predicts user ratings and is evaluated using RMSE.

3. Hybrid Recommendation:
   Content-based similarity scores and collaborative filtering scores are combined using a weighted hybrid fusion strategy to generate personalized recommendations.

--------------------------------------------------

Hybrid Recommendation Logic:

For a given base movie and user ID, the final recommendation score is computed as:

HybridScore = α × ContentScore + (1 − α) × UserPreferenceScore

Where:
• ContentScore is derived from TF-IDF–based cosine similarity  
• UserPreferenceScore is predicted using collaborative filtering  
• α controls the balance between relevance and personalization  

--------------------------------------------------

Explainability and Evaluation:

• Similarity Heatmap:
  Visualizes cosine similarity between recommended movies using TF-IDF feature vectors to provide explainability.

• Evaluation Metrics:
  – RMSE is used to evaluate collaborative filtering accuracy  
  – Precision@K and Recall@K are used to evaluate recommendation quality  

--------------------------------------------------

Files Description:

app.py  
Contains the Streamlit web application, including movie search, genre browsing, similar movies, hybrid recommendations, explainability visualizations, and evaluation metrics.

recommender_core.py  
Implements the core recommendation logic, including TF-IDF vectorization, cosine similarity computation, hybrid score calculation, heatmap support, and evaluation metrics.

collaborative_filtering.py  
Implements collaborative filtering using matrix factorization and computes RMSE for model evaluation.

moviedataset_downloadandpreprocessing.py  
Downloads and preprocesses movie metadata and retrieves movie descriptions using the TMDB API with caching.

ratingsdataset.py  
Downloads the MovieLens dataset and prepares ratings.csv for collaborative filtering.

movies_with_description.csv  
Processed dataset containing movie titles, genres, ratings, votes, release year, and descriptions.

ratings.csv  
MovieLens user–movie rating dataset used for collaborative filtering.

tmdb_cache.csv  
Cache file for TMDB API responses.

requirements.txt  
Lists all required Python dependencies.

--------------------------------------------------

Technologies and Libraries Used:

• Python  
• Streamlit  
• scikit-learn  
• pandas, numpy  
• matplotlib, seaborn  
• TMDB API  

--------------------------------------------------

How to Run the Project:

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Streamlit application:
   streamlit run app.py

3. Open the displayed local URL in a web browser.

--------------------------------------------------

Academic Note:

This version demonstrates the implementation of a classical hybrid recommender system using traditional NLP techniques and collaborative filtering. It serves as a stable baseline for further enhancement using deep learning approaches.
