# Movie Data Pipeline Visualizations

This directory contains comprehensive visualization tools for the movie data pipeline, designed to provide insights into data quality, feature engineering, and dataset characteristics.

## Overview

The visualization system creates explanatory plots that help understand:
- **Data Distribution**: How the data is spread across different features
- **Feature Relationships**: Correlations and patterns between variables
- **Data Quality**: Missing data, outliers, and data integrity
- **Feature Engineering**: Effectiveness of the preprocessing pipeline
- **Domain Insights**: Genre patterns, language distributions, and keyword analysis

## Files

### Core Files
- `data_visualization.py` - Main visualization class with all plotting methods
- `run_visualizations.py` - Simple script to run all visualizations
- `VISUALIZATION_README.md` - This documentation file

## Generated Plots

### 1. Dataset Overview (`dataset_overview.png`)
**Purpose**: Provides a high-level view of the dataset characteristics

**Contains**:
- **Movies per Year**: Time series showing movie production trends
- **Vote Average Distribution**: Histogram of user ratings
- **Popularity Distribution**: Distribution of popularity scores
- **Vote Count Distribution**: Log-scaled distribution of vote counts

**Insights**: Helps identify temporal trends, rating patterns, and popularity distributions.

### 2. Genre Analysis (`genre_analysis.png`)
**Purpose**: Analyzes movie genres and their performance characteristics

**Contains**:
- **Top 15 Genres by Frequency**: Most common genres in the dataset
- **Genre Distribution Pie Chart**: Visual representation of genre proportions
- **Average Vote by Genre**: Which genres receive higher ratings
- **Average Popularity by Genre**: Which genres are most popular

**Insights**: Reveals genre preferences, performance differences, and market trends.

### 3. Keyword Analysis (`keyword_analysis.png`)
**Purpose**: Examines movie keywords and their frequency patterns

**Contains**:
- **Top 20 Keywords by Frequency**: Most common keywords/tags
- **Keyword Distribution**: Histogram of keyword frequencies

**Insights**: Shows thematic patterns and content characteristics in the dataset.

### 4. Correlation Analysis (`correlation_analysis.png`)
**Purpose**: Explores relationships between numerical features

**Contains**:
- **Correlation Matrix**: Heatmap showing feature correlations
- **Popularity vs Vote Average**: Scatter plot relationship
- **Vote Count vs Vote Average**: Relationship between engagement and rating
- **Release Year vs Vote Average**: Temporal rating trends

**Insights**: Identifies feature dependencies and potential multicollinearity.

### 5. Language Analysis (`language_analysis.png`)
**Purpose**: Analyzes movie languages and their performance

**Contains**:
- **Top 15 Languages by Frequency**: Most common languages
- **Average Vote by Language**: Rating performance by language

**Insights**: Reveals international movie patterns and language preferences.

### 6. Feature Importance (`feature_importance.png`)
**Purpose**: Summarizes the feature engineering process

**Contains**:
- **TF-IDF Features Distribution**: Text feature characteristics
- **Genre Features**: Genre encoding effectiveness
- **Keyword Features**: Keyword feature distribution
- **Cast Features**: Actor/actress feature patterns

**Insights**: Shows the effectiveness of different feature engineering techniques.

### 7. Pipeline Summary (`pipeline_summary.png`)
**Purpose**: Provides an overall assessment of the data pipeline

**Contains**:
- **Dataset Size**: Total number of movies
- **Year Range**: Temporal coverage
- **Average Vote**: Overall rating statistics
- **Feature Counts**: Number of features by type
- **Missing Data Percentage**: Data quality assessment
- **Data Quality Score**: Overall quality metric

**Insights**: Comprehensive overview of data pipeline effectiveness and data quality.

## Usage

### Quick Start
```bash
# Navigate to the data directory
cd src/data

# Run all visualizations
python run_visualizations.py
```

### Custom Usage
```python
from data_visualization import MovieDataVisualizer

# Initialize visualizer
visualizer = MovieDataVisualizer('data/processed')

# Load data
visualizer.load_processed_data()

# Create specific plots
visualizer.create_overview_plots('my_plots')
visualizer.create_genre_analysis('my_plots')
visualizer.create_correlation_analysis('my_plots')
```

## Prerequisites

Before running visualizations, ensure:
1. The data pipeline has been executed (`python data_pipeline.py`)
2. Processed data files exist in `data/processed/`
3. Required Python packages are installed:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scipy

## Output

All plots are saved as high-resolution PNG files (300 DPI) in the specified output directory. Each plot includes:
- Clear titles and labels
- Color-coded visualizations
- Grid lines for readability
- Professional styling

## Interpretation Guide

### Data Quality Indicators
- **Missing Data**: Look for high percentages indicating data collection issues
- **Correlations**: Strong correlations (>0.7) may indicate redundant features
- **Distributions**: Skewed distributions may require transformation
- **Outliers**: Extreme values that may need special handling

### Feature Engineering Insights
- **Feature Counts**: Higher counts may indicate over-engineering
- **Sparse Features**: Many zero values may indicate feature selection needed
- **Distribution Patterns**: Help identify appropriate preprocessing steps

### Domain Insights
- **Genre Patterns**: Popular genres may need special handling
- **Temporal Trends**: Recent vs. older movie patterns
- **Language Diversity**: International content considerations

## Troubleshooting

### Common Issues
1. **Missing Data Files**: Ensure the data pipeline has been run
2. **Memory Errors**: Large datasets may require chunked processing
3. **Plot Display Issues**: Ensure matplotlib backend is properly configured

### Error Messages
- `"Processed data directory not found"`: Run the data pipeline first
- `"Missing required files"`: Check that all CSV files exist
- `"Feature not available"`: Some features may be optional

## Customization

The visualization system is designed to be extensible. You can:
- Add new plot types by extending the `MovieDataVisualizer` class
- Modify plot styles by changing matplotlib parameters
- Create custom analysis functions for specific insights
- Adjust plot sizes and layouts for different use cases

## Contributing

When adding new visualizations:
1. Follow the existing naming conventions
2. Include proper error handling
3. Add documentation for new plot types
4. Ensure plots are self-explanatory with clear titles and labels 