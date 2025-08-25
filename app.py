import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import re # For text cleaning
import pickle # To save/load trained models and data

# --- Function to load data (centralized to avoid re-loading on every interaction) ---
@st.cache_data # Cache the data loading to speed up app
def load_data():
    ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies_df = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                                   'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    movies_df = movies_df[['movie_id', 'title', 'release_date', 'Action', 'Adventure', 'Animation',
                           'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                           'Thriller', 'War', 'Western']]

    # Preprocessing for Content-Based
    def get_genres_string(row):
        genres = [col for col in movies_df.columns[5:] if row[col] == 1]
        return ' '.join(genres)
    movies_df['genres_str'] = movies_df.apply(get_genres_string, axis=1)
    movies_df['clean_title'] = movies_df['title'].apply(lambda x: re.sub(r' \(\d{4}\)', '', x).strip())
    movies_df['content_description'] = movies_df['clean_title'] + ' ' + movies_df['genres_str']

    # Preprocessing for Collaborative Filtering (Surprise library needs Reader object)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

    # Train SVD model (can be pre-trained and loaded)
    trainset = data.build_full_trainset()
    svd_model = SVD(random_state=42)
    svd_model.fit(trainset)

    # Content-Based: TF-IDF and Cosine Similarity
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['content_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    title_to_index = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

    return ratings_df, movies_df, svd_model, cosine_sim, title_to_index

# Load data, models, and matrices once
ratings_df, movies_df, svd_model, cosine_sim, title_to_index = load_data()

# --- Recommendation Functions (as defined before) ---

def get_content_based_recommendations_for_app(movie_title, cosine_sim_matrix, df_movies, top_n=10):
    if movie_title not in df_movies['title'].values:
        return []
    idx = title_to_index[movie_title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df_movies['title'].iloc[movie_indices].tolist()

def get_collaborative_recommendations_for_app(user_id, model, df_ratings, df_movies, top_n=10):
    all_movie_ids = df_movies['movie_id'].unique()
    rated_movie_ids = df_ratings[df_ratings['user_id'] == user_id]['movie_id'].unique()
    unrated_movie_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

    predictions = []
    for movie_id in unrated_movie_ids:
        pred = model.predict(uid=user_id, iid=movie_id).est
        predictions.append((movie_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:top_n]
    recommended_movie_titles = [df_movies[df_movies['movie_id'] == mid]['title'].iloc[0] for mid, _ in top_recommendations]
    return recommended_movie_titles

# --- Streamlit App ---
st.title('Movie Recommendation System 🎬')
st.write('Explore movie recommendations using two different approaches: Content-Based and Collaborative Filtering.')

st.sidebar.header('Choose Recommendation Type')
recommendation_type = st.sidebar.radio(
    "Select a recommender:",
    ('Content-Based Filtering', 'Collaborative Filtering')
)

top_n_recommendations = st.sidebar.slider('Number of Recommendations', 5, 20, 10)

if recommendation_type == 'Content-Based Filtering':
    st.header('Content-Based Recommendations')
    movie_list = movies_df['title'].tolist()
    selected_movie = st.selectbox('Select a movie you like:', movie_list)

    if st.button('Get Content-Based Recommendations'):
        if selected_movie:
            with st.spinner('Generating recommendations...'):
                recommendations = get_content_based_recommendations_for_app(selected_movie, cosine_sim, movies_df, top_n=top_n_recommendations)
                if recommendations:
                    st.subheader(f'Movies similar to "{selected_movie}":')
                    for i, movie in enumerate(recommendations):
                        st.write(f'{i+1}. {movie}')
                else:
                    st.write("Could not find recommendations for this movie.")
        else:
            st.write("Please select a movie.")

elif recommendation_type == 'Collaborative Filtering':
    st.header('Collaborative Filtering Recommendations')
    # Get unique user IDs for the slider
    user_ids = sorted(ratings_df['user_id'].unique())
    selected_user_id = st.slider('Select a User ID:', min_value=min(user_ids), max_value=max(user_ids), value=1)

    if st.button('Get Collaborative Recommendations'):
        with st.spinner('Generating recommendations...'):
            recommendations = get_collaborative_recommendations_for_app(selected_user_id, svd_model, ratings_df, movies_df, top_n=top_n_recommendations)
            if recommendations:
                st.subheader(f'Recommendations for User ID {selected_user_id}:')
                for i, movie in enumerate(recommendations):
                    st.write(f'{i+1}. {movie}')
            else:
                st.write("Could not find recommendations for this user.")

st.sidebar.markdown("""
---
**Project by:** Leonardo Flores Gonzalez
""")