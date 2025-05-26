import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class PosterManager:
    def __init__(self):
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')
        self.base_url = 'https://api.themoviedb.org/3'
        self.image_base_url = 'https://image.tmdb.org/t/p'
        self.poster_sizes = {
            'small': 'w185',
            'medium': 'w342',
            'large': 'w500',
            'original': 'original'
        }
        
    def get_poster_url(self, poster_path, size='medium'):
        """Get full poster URL from poster path"""
        if not poster_path:
            return None
            
        size = self.poster_sizes.get(size, 'w342')
        return f"{self.image_base_url}/{size}{poster_path}"
        
    def get_movie_details(self, tmdb_id):
        """Get movie details from TMDB API"""
        if not self.tmdb_api_key:
            return None
            
        url = f"{self.base_url}/movie/{tmdb_id}"
        params = {
            'api_key': self.tmdb_api_key,
            'language': 'en-US'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie details: {e}")
            return None
            
    def get_poster_path(self, tmdb_id):
        """Get poster path for a movie from TMDB API"""
        movie_details = self.get_movie_details(tmdb_id)
        if movie_details:
            return movie_details.get('poster_path')
        return None
        
    def download_poster(self, poster_path, save_dir, size='medium'):
        """Download movie poster and save to local directory"""
        if not poster_path:
            return None
            
        url = self.get_poster_url(poster_path, size)
        if not url:
            return None
            
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Create save directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save poster
            filename = f"{poster_path[1:]}"  # Remove leading slash
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            return filepath
        except requests.exceptions.RequestException as e:
            print(f"Error downloading poster: {e}")
            return None
            
    def batch_download_posters(self, movie_ids, save_dir, size='medium'):
        """Download posters for multiple movies"""
        results = {}
        for movie_id in movie_ids:
            poster_path = self.get_poster_path(movie_id)
            if poster_path:
                filepath = self.download_poster(poster_path, save_dir, size)
                results[movie_id] = filepath
        return results 