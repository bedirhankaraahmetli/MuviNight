import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MovieDataVisualizer:
    def __init__(self, processed_data_dir='data/processed'):
        self.processed_data_dir = processed_data_dir
        self.df = None
        self.features = {}
        
    def load_processed_data(self):
        """Load the processed data and features"""
        print("Loading processed data for visualization...")
        
        # Load processed dataframe
        self.df = pd.read_csv(os.path.join(self.processed_data_dir, 'processed_movies.csv'))
        
        # Load features
        feature_files = {
            'genre_features': 'genre_features.csv',
            'keyword_features': 'keyword_features.csv',
            'cast_features': 'cast_features.csv',
            'numeric_features': 'numeric_features.csv'
        }
        
        for name, filename in feature_files.items():
            filepath = os.path.join(self.processed_data_dir, filename)
            if os.path.exists(filepath):
                self.features[name] = pd.read_csv(filepath, index_col=0)
            else:
                print(f"Warning: {filename} not found")
                
        # Load sparse features
        sparse_files = ['text_features.npz']
        for filename in sparse_files:
            filepath = os.path.join(self.processed_data_dir, filename)
            if os.path.exists(filepath):
                feature_name = filename.replace('.npz', '')
                self.features[feature_name] = sparse.load_npz(filepath)
                
        print(f"Loaded {len(self.df)} movies with features")
        
    def create_overview_plots(self, save_dir='plots'):
        """Create overview plots of the dataset"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Dataset Overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Movie Dataset Overview', fontsize=16, fontweight='bold')
        
        # Movies per year
        year_counts = self.df['release_year'].value_counts().sort_index()
        axes[0, 0].plot(year_counts.index, year_counts.values, linewidth=2, color='steelblue')
        axes[0, 0].set_title('Movies Released per Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Movies')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Vote average distribution
        axes[0, 1].hist(self.df['vote_average'].dropna(), bins=30, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Distribution of Vote Averages')
        axes[0, 1].set_xlabel('Vote Average')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Popularity distribution
        axes[1, 0].hist(self.df['popularity'].dropna(), bins=30, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Distribution of Popularity Scores')
        axes[1, 0].set_xlabel('Popularity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Vote count distribution (log scale)
        vote_counts = self.df['vote_count'].dropna()
        axes[1, 1].hist(np.log10(vote_counts + 1), bins=30, alpha=0.7, color='gold')
        axes[1, 1].set_title('Distribution of Vote Counts (Log Scale)')
        axes[1, 1].set_xlabel('Log10(Vote Count + 1)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_genre_analysis(self, save_dir='plots'):
        """Create genre analysis plots"""
        if 'genre_features' not in self.features:
            print("Genre features not available")
            return
            
        genre_features = self.features['genre_features']
        
        # Top genres by frequency
        genre_counts = genre_features.sum().sort_values(ascending=False).head(15)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Genre Analysis', fontsize=16, fontweight='bold')
        
        # Top genres bar plot
        axes[0, 0].barh(range(len(genre_counts)), genre_counts.values, color='skyblue')
        axes[0, 0].set_yticks(range(len(genre_counts)))
        axes[0, 0].set_yticklabels(genre_counts.index)
        axes[0, 0].set_title('Top 15 Genres by Frequency')
        axes[0, 0].set_xlabel('Number of Movies')
        
        # Genre distribution pie chart (top 10)
        top_10_genres = genre_counts.head(10)
        other_count = genre_counts[10:].sum()
        pie_data = list(top_10_genres.values) + [other_count]
        pie_labels = list(top_10_genres.index) + ['Others']
        
        axes[0, 1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Genre Distribution (Top 10)')
        
        # Average vote by genre
        genre_votes = {}
        for genre in genre_counts.head(10).index:
            movies_with_genre = genre_features[genre] == 1
            avg_vote = self.df.loc[movies_with_genre, 'vote_average'].mean()
            genre_votes[genre] = avg_vote
            
        genre_votes_series = pd.Series(genre_votes).sort_values(ascending=False)
        axes[1, 0].barh(range(len(genre_votes_series)), genre_votes_series.values, color='lightcoral')
        axes[1, 0].set_yticks(range(len(genre_votes_series)))
        axes[1, 0].set_yticklabels(genre_votes_series.index)
        axes[1, 0].set_title('Average Vote by Genre')
        axes[1, 0].set_xlabel('Average Vote')
        
        # Genre popularity correlation
        genre_popularity = {}
        for genre in genre_counts.head(10).index:
            movies_with_genre = genre_features[genre] == 1
            avg_popularity = self.df.loc[movies_with_genre, 'popularity'].mean()
            genre_popularity[genre] = avg_popularity
            
        genre_popularity_series = pd.Series(genre_popularity).sort_values(ascending=False)
        axes[1, 1].barh(range(len(genre_popularity_series)), genre_popularity_series.values, color='lightgreen')
        axes[1, 1].set_yticks(range(len(genre_popularity_series)))
        axes[1, 1].set_yticklabels(genre_popularity_series.index)
        axes[1, 1].set_title('Average Popularity by Genre')
        axes[1, 1].set_xlabel('Average Popularity')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'genre_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_keyword_analysis(self, save_dir='plots'):
        """Create keyword analysis plots"""
        if 'keyword_features' not in self.features:
            print("Keyword features not available")
            return
            
        keyword_features = self.features['keyword_features']
        
        # Top keywords
        keyword_counts = keyword_features.sum().sort_values(ascending=False).head(20)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Keyword Analysis', fontsize=16, fontweight='bold')
        
        # Top keywords bar plot
        axes[0].barh(range(len(keyword_counts)), keyword_counts.values, color='gold')
        axes[0].set_yticks(range(len(keyword_counts)))
        axes[0].set_yticklabels(keyword_counts.index)
        axes[0].set_title('Top 20 Keywords by Frequency')
        axes[0].set_xlabel('Number of Movies')
        
        # Keyword distribution
        axes[1].hist(keyword_counts.values, bins=20, alpha=0.7, color='orange')
        axes[1].set_title('Distribution of Keyword Frequencies')
        axes[1].set_xlabel('Number of Movies')
        axes[1].set_ylabel('Number of Keywords')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'keyword_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_correlation_analysis(self, save_dir='plots'):
        """Create correlation analysis plots"""
        numeric_cols = ['popularity', 'vote_average', 'vote_count', 'release_year']
        numeric_data = self.df[numeric_cols].dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Correlation heatmap
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0, 0])
        axes[0, 0].set_title('Correlation Matrix')
        
        # Popularity vs Vote Average
        axes[0, 1].scatter(numeric_data['popularity'], numeric_data['vote_average'], 
                          alpha=0.6, color='steelblue')
        axes[0, 1].set_xlabel('Popularity')
        axes[0, 1].set_ylabel('Vote Average')
        axes[0, 1].set_title('Popularity vs Vote Average')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Vote Count vs Vote Average
        axes[1, 0].scatter(numeric_data['vote_count'], numeric_data['vote_average'], 
                          alpha=0.6, color='lightcoral')
        axes[1, 0].set_xlabel('Vote Count')
        axes[1, 0].set_ylabel('Vote Average')
        axes[1, 0].set_title('Vote Count vs Vote Average')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Release Year vs Vote Average
        axes[1, 1].scatter(numeric_data['release_year'], numeric_data['vote_average'], 
                          alpha=0.6, color='lightgreen')
        axes[1, 1].set_xlabel('Release Year')
        axes[1, 1].set_ylabel('Vote Average')
        axes[1, 1].set_title('Release Year vs Vote Average')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_language_analysis(self, save_dir='plots'):
        """Create language analysis plots"""
        language_counts = self.df['original_language'].value_counts().head(15)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Language Analysis', fontsize=16, fontweight='bold')
        
        # Top languages bar plot
        axes[0].barh(range(len(language_counts)), language_counts.values, color='purple')
        axes[0].set_yticks(range(len(language_counts)))
        axes[0].set_yticklabels(language_counts.index)
        axes[0].set_title('Top 15 Languages by Number of Movies')
        axes[0].set_xlabel('Number of Movies')
        
        # Average vote by language
        language_votes = {}
        for lang in language_counts.head(10).index:
            movies_in_lang = self.df[self.df['original_language'] == lang]
            avg_vote = movies_in_lang['vote_average'].mean()
            language_votes[lang] = avg_vote
            
        language_votes_series = pd.Series(language_votes).sort_values(ascending=False)
        axes[1].barh(range(len(language_votes_series)), language_votes_series.values, color='pink')
        axes[1].set_yticks(range(len(language_votes_series)))
        axes[1].set_yticklabels(language_votes_series.index)
        axes[1].set_title('Average Vote by Language')
        axes[1].set_xlabel('Average Vote')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'language_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_feature_importance_plot(self, save_dir='plots'):
        """Create feature importance visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Engineering Summary', fontsize=16, fontweight='bold')
        
        # Text features (TF-IDF)
        if 'text_features' in self.features:
            text_features = self.features['text_features']
            axes[0, 0].hist(text_features.data, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title(f'TF-IDF Features Distribution\n({text_features.shape[1]} features)')
            axes[0, 0].set_xlabel('TF-IDF Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Genre features
        if 'genre_features' in self.features:
            genre_features = self.features['genre_features']
            axes[0, 1].bar(range(len(genre_features.columns)), genre_features.sum(), color='green')
            axes[0, 1].set_title(f'Genre Features\n({len(genre_features.columns)} genres)')
            axes[0, 1].set_xlabel('Genre Index')
            axes[0, 1].set_ylabel('Number of Movies')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Keyword features
        if 'keyword_features' in self.features:
            keyword_features = self.features['keyword_features']
            axes[1, 0].hist(keyword_features.sum(), bins=30, alpha=0.7, color='orange')
            axes[1, 0].set_title(f'Keyword Features Distribution\n({len(keyword_features.columns)} keywords)')
            axes[1, 0].set_xlabel('Number of Movies per Keyword')
            axes[1, 0].set_ylabel('Number of Keywords')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Cast features
        if 'cast_features' in self.features:
            cast_features = self.features['cast_features']
            if len(cast_features.columns) > 0:
                axes[1, 1].hist(cast_features.sum(), bins=30, alpha=0.7, color='red')
                axes[1, 1].set_title(f'Cast Features Distribution\n({len(cast_features.columns)} actors)')
                axes[1, 1].set_xlabel('Number of Movies per Actor')
                axes[1, 1].set_ylabel('Number of Actors')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_pipeline_summary(self, save_dir='plots'):
        """Create a comprehensive pipeline summary"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Movie Data Pipeline Summary', fontsize=18, fontweight='bold')
        
        # Dataset size
        axes[0, 0].text(0.5, 0.5, f'Total Movies:\n{len(self.df):,}', 
                       ha='center', va='center', fontsize=20, fontweight='bold',
                       transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Dataset Size')
        axes[0, 0].axis('off')
        
        # Year range
        year_range = f"{self.df['release_year'].min():.0f} - {self.df['release_year'].max():.0f}"
        axes[0, 1].text(0.5, 0.5, f'Year Range:\n{year_range}', 
                       ha='center', va='center', fontsize=20, fontweight='bold',
                       transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Release Year Range')
        axes[0, 1].axis('off')
        
        # Average vote
        avg_vote = self.df['vote_average'].mean()
        axes[0, 2].text(0.5, 0.5, f'Average Vote:\n{avg_vote:.2f}/10', 
                       ha='center', va='center', fontsize=20, fontweight='bold',
                       transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Overall Rating')
        axes[0, 2].axis('off')
        
        # Feature counts
        feature_counts = {}
        if 'text_features' in self.features:
            feature_counts['Text Features'] = self.features['text_features'].shape[1]
        if 'genre_features' in self.features:
            feature_counts['Genre Features'] = len(self.features['genre_features'].columns)
        if 'keyword_features' in self.features:
            feature_counts['Keyword Features'] = len(self.features['keyword_features'].columns)
        if 'cast_features' in self.features:
            feature_counts['Cast Features'] = len(self.features['cast_features'].columns)
            
        feature_names = list(feature_counts.keys())
        feature_values = list(feature_counts.values())
        
        axes[1, 0].bar(feature_names, feature_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 0].set_title('Feature Counts')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Missing data
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        axes[1, 1].bar(range(len(missing_percent)), missing_percent, color='lightcoral')
        axes[1, 1].set_title('Missing Data Percentage')
        axes[1, 1].set_xlabel('Columns')
        axes[1, 1].set_ylabel('Missing (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Data quality score
        quality_score = (1 - missing_percent.mean() / 100) * 100
        axes[1, 2].text(0.5, 0.5, f'Data Quality Score:\n{quality_score:.1f}%', 
                       ha='center', va='center', fontsize=20, fontweight='bold',
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Overall Data Quality')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pipeline_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_all_plots(self, save_dir='plots'):
        """Create all visualization plots"""
        print("Creating comprehensive data visualizations...")
        
        # Create all plots
        self.create_overview_plots(save_dir)
        self.create_genre_analysis(save_dir)
        self.create_keyword_analysis(save_dir)
        self.create_correlation_analysis(save_dir)
        self.create_language_analysis(save_dir)
        self.create_feature_importance_plot(save_dir)
        self.create_pipeline_summary(save_dir)
        
        print(f"All plots saved to '{save_dir}' directory")
        
def main():
    # Initialize visualizer
    visualizer = MovieDataVisualizer()
    
    # Load data
    visualizer.load_processed_data()
    
    # Create all plots
    visualizer.create_all_plots()
    
if __name__ == "__main__":
    main() 