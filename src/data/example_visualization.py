#!/usr/bin/env python3
"""
Example: Custom Movie Data Visualization

This script demonstrates how to use the MovieDataVisualizer class
for custom analysis and visualization of movie data.

It shows how to:
- Load and analyze specific aspects of the data
- Create custom visualizations
- Extract insights from the data pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_visualization import MovieDataVisualizer

def example_custom_analysis():
    """Example of custom analysis using the visualization system"""
    
    print("=" * 50)
    print("CUSTOM MOVIE DATA ANALYSIS EXAMPLE")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = MovieDataVisualizer('data/processed')
    
    try:
        # Load data
        visualizer.load_processed_data()
        
        # Example 1: Analyze top-rated movies
        print("\n1. Analyzing Top-Rated Movies...")
        top_movies = visualizer.df.nlargest(10, 'vote_average')[['original_title', 'vote_average', 'vote_count', 'release_year']]
        print("Top 10 Movies by Rating:")
        print(top_movies.to_string(index=False))
        
        # Example 2: Analyze recent vs old movies
        print("\n2. Analyzing Recent vs Old Movies...")
        recent_movies = visualizer.df[visualizer.df['release_year'] >= 2020]
        old_movies = visualizer.df[visualizer.df['release_year'] < 2000]
        
        print(f"Recent movies (2020+): {len(recent_movies)} movies, avg rating: {recent_movies['vote_average'].mean():.2f}")
        print(f"Old movies (pre-2000): {len(old_movies)} movies, avg rating: {old_movies['vote_average'].mean():.2f}")
        
        # Example 3: Create custom plot
        print("\n3. Creating Custom Analysis Plot...")
        create_custom_plot(visualizer.df)
        
        # Example 4: Feature analysis
        print("\n4. Analyzing Feature Engineering Results...")
        analyze_features(visualizer)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        print("Make sure the data pipeline has been run first.")

def create_custom_plot(df):
    """Create a custom analysis plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Custom Movie Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Rating trends over time
    yearly_avg = df.groupby('release_year')['vote_average'].mean()
    axes[0, 0].plot(yearly_avg.index, yearly_avg.values, linewidth=2, color='blue')
    axes[0, 0].set_title('Average Rating by Year')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Average Rating')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Popularity vs Rating scatter
    axes[0, 1].scatter(df['popularity'], df['vote_average'], alpha=0.6, color='green')
    axes[0, 1].set_title('Popularity vs Rating')
    axes[0, 1].set_xlabel('Popularity')
    axes[0, 1].set_ylabel('Rating')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Vote count distribution by decade
    df['decade'] = (df['release_year'] // 10) * 10
    decade_counts = df['decade'].value_counts().sort_index()
    axes[1, 0].bar(decade_counts.index, decade_counts.values, color='orange')
    axes[1, 0].set_title('Movies by Decade')
    axes[1, 0].set_xlabel('Decade')
    axes[1, 0].set_ylabel('Number of Movies')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Rating distribution by popularity quartile
    df['popularity_quartile'] = pd.qcut(df['popularity'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    quartile_ratings = df.groupby('popularity_quartile')['vote_average'].mean()
    axes[1, 1].bar(quartile_ratings.index, quartile_ratings.values, color='red')
    axes[1, 1].set_title('Average Rating by Popularity Quartile')
    axes[1, 1].set_xlabel('Popularity Quartile')
    axes[1, 1].set_ylabel('Average Rating')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('custom_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Custom plot saved as 'custom_analysis.png'")

def analyze_features(visualizer):
    """Analyze the feature engineering results"""
    
    print("\nFeature Engineering Analysis:")
    print("-" * 30)
    
    # Analyze text features
    if 'text_features' in visualizer.features:
        text_features = visualizer.features['text_features']
        print(f"Text Features: {text_features.shape[1]} TF-IDF features")
        print(f"  - Sparsity: {1 - text_features.nnz / (text_features.shape[0] * text_features.shape[1]):.2%}")
        print(f"  - Average non-zero values per movie: {text_features.nnz / text_features.shape[0]:.1f}")
    
    # Analyze genre features
    if 'genre_features' in visualizer.features:
        genre_features = visualizer.features['genre_features']
        print(f"Genre Features: {len(genre_features.columns)} genres")
        print(f"  - Most common genre: {genre_features.sum().idxmax()}")
        print(f"  - Average genres per movie: {genre_features.sum(axis=1).mean():.1f}")
    
    # Analyze keyword features
    if 'keyword_features' in visualizer.features:
        keyword_features = visualizer.features['keyword_features']
        print(f"Keyword Features: {len(keyword_features.columns)} keywords")
        print(f"  - Most common keyword: {keyword_features.sum().idxmax()}")
        print(f"  - Average keywords per movie: {keyword_features.sum(axis=1).mean():.1f}")
    
    # Analyze cast features
    if 'cast_features' in visualizer.features:
        cast_features = visualizer.features['cast_features']
        if len(cast_features.columns) > 0:
            print(f"Cast Features: {len(cast_features.columns)} actors/actresses")
            print(f"  - Most frequent actor: {cast_features.sum().idxmax()}")
            print(f"  - Average cast members per movie: {cast_features.sum(axis=1).mean():.1f}")
        else:
            print("Cast Features: No cast data available")

def example_data_insights():
    """Extract and display key insights from the data"""
    
    print("\n" + "=" * 50)
    print("KEY DATA INSIGHTS")
    print("=" * 50)
    
    visualizer = MovieDataVisualizer('data/processed')
    visualizer.load_processed_data()
    
    df = visualizer.df
    
    # Insight 1: Rating distribution
    print("\n1. Rating Distribution:")
    print(f"   - Average rating: {df['vote_average'].mean():.2f}")
    print(f"   - Median rating: {df['vote_average'].median():.2f}")
    print(f"   - Rating range: {df['vote_average'].min():.1f} - {df['vote_average'].max():.1f}")
    
    # Insight 2: Temporal coverage
    print("\n2. Temporal Coverage:")
    print(f"   - Year range: {df['release_year'].min():.0f} - {df['release_year'].max():.0f}")
    print(f"   - Most productive year: {df['release_year'].mode().iloc[0]:.0f}")
    
    # Insight 3: Language diversity
    print("\n3. Language Diversity:")
    language_counts = df['original_language'].value_counts()
    print(f"   - Total languages: {len(language_counts)}")
    print(f"   - Most common language: {language_counts.index[0]} ({language_counts.iloc[0]} movies)")
    
    # Insight 4: Popularity analysis
    print("\n4. Popularity Analysis:")
    print(f"   - Average popularity: {df['popularity'].mean():.2f}")
    print(f"   - Most popular movie: {df.loc[df['popularity'].idxmax(), 'original_title']}")
    print(f"   - Popularity range: {df['popularity'].min():.1f} - {df['popularity'].max():.1f}")

if __name__ == "__main__":
    # Run the example analysis
    example_custom_analysis()
    
    # Extract insights
    example_data_insights()
    
    print("\n" + "=" * 50)
    print("EXAMPLE COMPLETE!")
    print("=" * 50)
    print("This example demonstrates how to:")
    print("- Load and analyze movie data")
    print("- Create custom visualizations")
    print("- Extract meaningful insights")
    print("- Understand feature engineering results") 