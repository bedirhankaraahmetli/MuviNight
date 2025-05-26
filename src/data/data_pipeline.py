import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
from datetime import datetime
import os
from scipy import sparse

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with error handling
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
        print("Please run the following commands manually:")
        print("python -m nltk.downloader punkt")
        print("python -m nltk.downloader stopwords")

download_nltk_data()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MovieDataPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load the movie dataset"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} movies")
        print("\nAvailable columns:")
        print(self.df.columns.tolist())
        
    def clean_data(self):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        # Convert release_date to datetime
        self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')
        self.df['release_year'] = self.df['release_date'].dt.year
        
        # Clean numeric columns
        numeric_cols = ['popularity', 'vote_average', 'vote_count']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(self.df[col].median())
            
        # Clean text columns
        text_cols = ['overview', 'original_title']
        for col in text_cols:
            self.df[col] = self.df[col].fillna('')
            
        # Process genre_ids
        self.df['genre_ids'] = self.df['genre_ids'].apply(self._parse_list)
        
        # Process keywords
        self.df['keywords'] = self.df['keywords'].apply(self._parse_json_list)
        
        # Process cast and crew
        self.df['cast'] = self.df['cast'].apply(self._parse_json_list)
        self.df['crew'] = self.df['crew'].apply(self._parse_json_list)
        
    def _parse_list(self, x):
        """Parse string representation of list to actual list"""
        try:
            if pd.isna(x):
                return []
            return eval(x)
        except:
            return []
            
    def _parse_json_list(self, x):
        """Parse JSON string to list of names"""
        try:
            if pd.isna(x):
                return []
            return [item['name'] for item in json.loads(x)]
        except:
            return []
            
    def create_features(self):
        """Create features for the model"""
        print("Creating features...")
        
        # TF-IDF for text features
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        text_features = tfidf.fit_transform(self.df['overview'] + ' ' + self.df['original_title'])
        
        # Save vocabulary for later use
        self.vocabulary = {k: int(v) for k, v in tfidf.vocabulary_.items()}
        
        # Label encoding for categorical features
        le = LabelEncoder()
        self.df['original_language_encoded'] = le.fit_transform(self.df['original_language'])
        
        # Create genre features
        all_genres = list(set([genre for genres in self.df['genre_ids'] for genre in genres]))
        genre_features = pd.DataFrame(0, index=self.df.index, columns=all_genres)
        for idx, genres in enumerate(self.df['genre_ids']):
            for genre in genres:
                genre_features.loc[idx, genre] = 1
                
        # Create keyword features
        all_keywords = list(set([keyword for keywords in self.df['keywords'] for keyword in keywords]))
        keyword_features = pd.DataFrame(0, index=self.df.index, columns=all_keywords)
        for idx, keywords in enumerate(self.df['keywords']):
            for keyword in keywords:
                keyword_features.loc[idx, keyword] = 1
                
        # Create cast features (top 100 most frequent actors)
        all_cast = list(set([actor for cast in self.df['cast'] for actor in cast]))
        cast_counts = pd.Series([actor for cast in self.df['cast'] for actor in cast]).value_counts()
        top_cast = cast_counts.head(100).index
        if len(top_cast) > 0:
            cast_features = pd.DataFrame(0, index=self.df.index, columns=top_cast)
            for idx, cast in enumerate(self.df['cast']):
                for actor in cast:
                    if actor in top_cast:
                        cast_features.loc[idx, actor] = 1
        else:
            cast_features = pd.DataFrame(index=self.df.index)
                    
        # Combine features
        self.features = {
            'text_features': text_features,
            'genre_features': genre_features,
            'keyword_features': keyword_features,
            'cast_features': cast_features,
            'numeric_features': self.df[['popularity', 'vote_average', 'vote_count', 'release_year']],
            'language_features': self.df['original_language_encoded']
        }
        
    def save_processed_data(self, output_dir):
        """Save processed data and features"""
        print("Saving processed data...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed dataframe
        self.df.to_csv(os.path.join(output_dir, 'processed_movies.csv'), index=False)
        
        # Save features
        for name, feature in self.features.items():
            if isinstance(feature, pd.DataFrame):
                feature.to_csv(os.path.join(output_dir, f'{name}.csv'))
            elif sparse.issparse(feature):
                # Save sparse matrix in CSR format
                sparse.save_npz(os.path.join(output_dir, f'{name}.npz'), feature)
            else:
                np.save(os.path.join(output_dir, f'{name}.npy'), feature)
                
        # Save vocabulary
        with open(os.path.join(output_dir, 'vocabulary.json'), 'w') as f:
            json.dump(self.vocabulary, f, cls=NumpyEncoder)
                
def main():
    # Initialize pipeline
    pipeline = MovieDataPipeline('dataset.csv')
    
    # Run pipeline
    pipeline.load_data()
    pipeline.clean_data()
    pipeline.create_features()
    pipeline.save_processed_data('data/processed')
    
if __name__ == "__main__":
    main() 