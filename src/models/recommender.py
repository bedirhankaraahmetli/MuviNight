import numpy as np
import pandas as pd
import torch
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import ast

class MovieRecommender:
    def __init__(self, data_dir, models_dir):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.models = {}
        self.movies = None
        self.features = {}
        self.id_to_index = {}
        self.load_data()
        self.load_models()
        
    def load_data(self):
        """Load processed data and features"""
        print("Loading data...")
        
        # Load processed movies
        self.movies = pd.read_csv(os.path.join(self.data_dir, 'processed_movies.csv'))
        
        # Load features
        self.features['text'] = sparse.load_npz(os.path.join(self.data_dir, 'text_features.npz'))
        self.features['genre'] = pd.read_csv(os.path.join(self.data_dir, 'genre_features.csv'))
        self.features['numeric'] = pd.read_csv(os.path.join(self.data_dir, 'numeric_features.csv'))
        self.features['language'] = np.load(os.path.join(self.data_dir, 'language_features.npy'), allow_pickle=True)
        
        # Combine features
        numeric_features = self.features['numeric'].values
        genre_features = self.features['genre'].values
        language_features = self.features['language'].reshape(-1, 1)
        text_features = self.features['text'].toarray()
        
        self.X = np.hstack([numeric_features, genre_features, language_features, text_features])
        
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.movies['id'])}
        
    def load_models(self):
        """Load trained models"""
        print("Loading models...")
        
        # Load content-based model
        self.models['content_based'] = joblib.load(os.path.join(self.models_dir, 'content_based_model.joblib'))
        
        # Load genre-based model
        self.models['genre'] = joblib.load(os.path.join(self.models_dir, 'genre_model.joblib'))
        
        # Load text-based model
        self.models['text'] = joblib.load(os.path.join(self.models_dir, 'text_model.joblib'))
        
        # Load cast-based model if it exists
        cast_model_path = os.path.join(self.models_dir, 'cast_model.joblib')
        if os.path.exists(cast_model_path):
            self.models['cast'] = joblib.load(cast_model_path)
        
    def get_recommendations(self, movie_ids, n_recommendations=5, model_name='content_based'):
        """Get movie recommendations based on selected movies"""
        if not movie_ids:
            return self._get_random_recommendations(n_recommendations)
            
        # Map movie_ids to indices
        indices = [self.id_to_index[mid] for mid in movie_ids if mid in self.id_to_index]
        if not indices:
            raise ValueError("No valid movie IDs provided.")
        
        # Get recommendations based on model
        if model_name in self.models:
            model = self.models[model_name]
            similarity_matrix = model['similarity_matrix']
            
            # Get average similarity scores for selected movies
            selected_similarities = similarity_matrix[indices].mean(axis=0)
            
            # Get top N recommendations (excluding the selected movies)
            top_indices = np.argsort(selected_similarities)[::-1]
            top_indices = [idx for idx in top_indices if idx not in indices][:n_recommendations]
            
            # Get movie details for recommendations
            recommended_movies = self.movies.iloc[top_indices][
                ['id', 'title', 'overview', 'poster_path', 'release_date', 'vote_average']
            ].to_dict('records')
            
            return recommended_movies
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
    def _get_random_recommendations(self, n_recommendations=5):
        """Get random movie recommendations"""
        random_indices = np.random.choice(len(self.movies), n_recommendations, replace=False)
        random_movies = self.movies.iloc[random_indices][
            ['id', 'title', 'overview', 'poster_path', 'release_date', 'vote_average']
        ].to_dict('records')
        
        return random_movies
        
    def get_movie_details(self, movie_id):
        """Get details for a specific movie"""
        matches = self.movies[self.movies['id'] == movie_id]
        print(f"Looking for movie_id={movie_id}, matches found: {len(matches)}")
        if matches.empty:
            print("No match found!")
            return None
        movie = matches.iloc[0]
        print("Movie row:", movie.to_dict())
        # Handle genres (use genre_ids if genres is missing)
        genres = None
        if 'genres' in self.movies.columns:
            genres = movie['genres']
        elif 'genre_ids' in self.movies.columns:
            genres = movie['genre_ids']
        # Parse genres if it's a string
        if isinstance(genres, str):
            try:
                genres_list = ast.literal_eval(genres)
                if isinstance(genres_list, list):
                    genres = [g.strip() for g in genres_list]
                else:
                    genres = [g.strip() for g in genres.split(',') if g.strip()]
            except Exception:
                genres = [g.strip() for g in genres.split(',') if g.strip()]
        elif not isinstance(genres, list):
            genres = []
        # Helper for safe value
        def safe_value(col):
            return movie[col] if col in self.movies.columns and not pd.isna(movie[col]) else None
        return {
            'id': self.to_python_type(safe_value('id')),
            'title': self.to_python_type(safe_value('title')),
            'overview': self.to_python_type(safe_value('overview')),
            'poster_path': self.to_python_type(safe_value('poster_path')),
            'release_date': self.to_python_type(safe_value('release_date')),
            'vote_average': self.to_python_type(safe_value('vote_average')),
            'genres': genres,
            'runtime': self.to_python_type(safe_value('runtime')),
            'budget': self.to_python_type(safe_value('budget')),
            'revenue': self.to_python_type(safe_value('revenue'))
        }
        
    def search_movies(self, query, limit=10):
        """Search movies by title"""
        query = query.lower()
        matches = self.movies[self.movies['title'].str.lower().str.contains(query)]
        return matches.head(limit)[
            ['id', 'title', 'overview', 'poster_path', 'release_date', 'vote_average']
        ].to_dict('records')

    def to_python_type(self, val):
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        return val 