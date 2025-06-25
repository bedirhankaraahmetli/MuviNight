#!/usr/bin/env python3
"""
Movie Data Pipeline Visualization Runner

This script runs comprehensive visualizations for the movie data pipeline,
creating explanatory plots that help understand the data structure, 
feature engineering process, and data quality.

Usage:
    python run_visualizations.py
"""

import os
import sys
import matplotlib.pyplot as plt
from data_visualization import MovieDataVisualizer

def main():
    """Main function to run all visualizations"""
    print("=" * 60)
    print("MOVIE DATA PIPELINE VISUALIZATION")
    print("=" * 60)
    
    # Check if processed data exists
    processed_dir = 'data/processed'
    if not os.path.exists(processed_dir):
        print(f"Error: Processed data directory '{processed_dir}' not found!")
        print("Please run the data pipeline first:")
        print("python src/data/data_pipeline.py")
        return
    
    # Check for required files
    required_files = ['processed_movies.csv']
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(processed_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please run the data pipeline first to generate these files.")
        return
    
    try:
        # Initialize visualizer
        print("Initializing data visualizer...")
        visualizer = MovieDataVisualizer(processed_dir)
        
        # Load data
        print("Loading processed data...")
        visualizer.load_processed_data()
        
        # Create plots directory
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        print(f"\nCreating visualizations in '{plots_dir}' directory...")
        print("-" * 40)
        
        # Create all plots
        visualizer.create_all_plots(plots_dir)
        
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETE!")
        print("=" * 60)
        print(f"All plots have been saved to the '{plots_dir}' directory.")
        print("\nGenerated plots:")
        print("1. dataset_overview.png - Basic dataset statistics and distributions")
        print("2. genre_analysis.png - Genre frequency and performance analysis")
        print("3. keyword_analysis.png - Keyword frequency analysis")
        print("4. correlation_analysis.png - Feature correlation matrix and scatter plots")
        print("5. language_analysis.png - Language distribution and performance")
        print("6. feature_importance.png - Feature engineering summary")
        print("7. pipeline_summary.png - Overall pipeline statistics and data quality")
        
        print("\nThese visualizations help understand:")
        print("- Data distribution and quality")
        print("- Feature relationships and correlations")
        print("- Genre and keyword patterns")
        print("- Data preprocessing effectiveness")
        print("- Overall dataset characteristics")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        print("Please check that all required data files are available.")
        return

if __name__ == "__main__":
    main() 