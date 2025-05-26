# MuviNight - Technical Documentation

## Overview
MuviNight is a movie recommendation system that utilizes multiple machine learning models to provide personalized movie recommendations. The system processes movie data, trains various recommendation models, and serves recommendations through a web interface.

## System Architecture

### 1. Data Pipeline (`src/data/data_pipeline.py`)

#### 1.1 Data Loading
- The system begins by loading the movie dataset using the `MovieDataPipeline` class
- Initial data is loaded from `dataset.csv` using pandas
- The pipeline handles SSL certificate issues and downloads required NLTK data for text processing

#### 1.2 Data Cleaning (`clean_data()` method)
- Converts release dates to datetime format
- Cleans numeric columns (popularity, vote_average, vote_count)
- Handles missing values in text columns (overview, original_title)
- Processes complex fields:
  - Genre IDs (parsed from string representation)
  - Keywords (parsed from JSON)
  - Cast and crew information (parsed from JSON)

#### 1.3 Feature Engineering (`create_features()` method)
- Text Features:
  - Uses TF-IDF vectorization on movie overviews and titles
  - Maximum 5000 features with English stop words removed
- Categorical Features:
  - Label encoding for original language
- Genre Features:
  - One-hot encoding of movie genres
- Keyword Features:
  - One-hot encoding of movie keywords
- Cast Features:
  - One-hot encoding of top 100 most frequent actors
- Numeric Features:
  - Popularity
  - Vote average
  - Vote count
  - Release year

#### 1.4 Data Storage
- Processed data is saved in the `data/processed` directory
- Features are saved in various formats:
  - Sparse matrices (.npz)
  - DataFrames (.csv)
  - Numpy arrays (.npy)
  - Vocabulary dictionary (JSON)

### 2. Model Training (`src/models/model_trainer.py`)

#### 2.1 Model Types
The system implements multiple recommendation models:

1. Content-Based Model:
   - Uses all available features
   - Implements PCA for dimensionality reduction
   - Optimizes using Optuna for hyperparameter tuning
   - Combines features:
     * Text features (TF-IDF)
     * Genre features (one-hot encoded)
     * Cast features (one-hot encoded)
     * Numeric features (scaled)
     * Language features (one-hot encoded)
   - Uses cosine similarity for final recommendations

2. Genre-Based Model:
   - Uses cosine similarity on genre features
   - Simple but effective for genre-based recommendations
   - One-hot encoded genre matrix
   - Computes similarity between movies based on shared genres
   - Weighted by genre importance in the dataset

3. Cast-Based Model:
   - Uses cosine similarity on cast features
   - Recommends movies with similar actors
   - Focuses on top 100 most frequent actors
   - One-hot encoded cast matrix
   - Computes actor-based similarity between movies

4. Text-Based Model:
   - Uses TF-IDF features from movie overviews and titles
   - Implements cosine similarity for text-based recommendations
   - Maximum 5000 features
   - Removes English stop words
   - Considers both movie titles and descriptions

5. Neural Network Model:
   - Custom PyTorch implementation
   - Architecture:
     * Input layer: matches feature dimension
     * Multiple hidden layers with ReLU activation
     * Dropout layers (0.3 dropout rate) for regularization
     * Output layer: reconstructs input features
   - Training features:
     * Early stopping with patience=5
     * Adam optimizer
     * MSE loss function
     * Learning rate optimization
     * Batch processing

6. Gradient Boosting Model:
   - Uses scikit-learn's GradientBoostingClassifier
   - Optimized for clustering-based recommendations
   - Features:
     * Fixed number of estimators (50)
     * Learning rate of 0.1
     * Maximum depth of 3
     * Subsampling of 0.8
     * Square root feature selection

7. SVM Model:
   - Uses scikit-learn's SVC
   - Linear kernel with probability estimates
   - Features:
     * C parameter fixed at 1
     * Scale-based gamma
     * Probability estimates enabled
     * Increased max iterations (2000)
     * Large cache size (2000)

#### 2.2 Training Process
- Each model undergoes hyperparameter optimization:
  * Content-Based: PCA components (50-200)
  * Neural Network: hidden layers (1-3), units, dropout rate
  * Gradient Boosting: tree parameters
  * SVM: kernel parameters
- Cross-validation:
  * 5-fold cross-validation for most models
  * Stratified sampling for classification tasks
  * Multiple metrics evaluation
- Early stopping implementation:
  * Patience-based stopping
  * Validation loss monitoring
  * Best model checkpointing
- Model persistence:
  * Saves model parameters
  * Stores feature scalers
  * Maintains vocabulary mappings
  * Saves similarity matrices

### 3. Recommendation System (`src/models/recommender.py`)

#### 3.1 Data Loading
- Loads processed data and trained models:
  * Text features (sparse matrix)
  * Genre features (DataFrame)
  * Numeric features (DataFrame)
  * Language features (numpy array)
- Maintains movie ID to index mapping:
  * Efficient lookup dictionary
  * Bidirectional mapping
  * Handles missing IDs
- Combines different feature types:
  * Horizontal stacking of features
  * Proper scaling and normalization
  * Sparse matrix handling

#### 3.2 Recommendation Generation
- Implements multiple recommendation strategies:
  * Content-based filtering
  * Genre-based filtering
  * Cast-based filtering
  * Text-based filtering
  * Hybrid recommendations
- Uses cosine similarity for content-based recommendations:
  * Computes similarity matrices
  * Handles sparse matrices efficiently
  * Normalizes feature vectors
- Combines results from different models:
  * Weighted ensemble approach
  * Model performance-based weighting
  * Diversity-aware combination
- Provides personalized recommendations:
  * User preference incorporation
  * Genre preference weighting
  * Actor preference consideration
  * Release year preferences

## Model Evaluation

### Metrics Used
1. Mean Squared Error (MSE):
   - Measures average squared difference
   - Penalizes larger errors more heavily
   - Used for regression tasks
   - Scaled to [0,1] range

2. Mean Absolute Error (MAE):
   - Measures average absolute difference
   - More interpretable than MSE
   - Less sensitive to outliers
   - Used for error magnitude assessment

3. Reconstruction Error:
   - Measures feature reconstruction quality
   - Used for autoencoder evaluation
   - Assesses information preservation
   - Helps in dimensionality reduction

4. Cross-validation scores:
   - 5-fold cross-validation
   - Multiple metric tracking
   - Stratified sampling
   - Performance stability assessment

### Model Comparison
- Each model's performance is tracked and compared:
  * Individual metric scores
  * Cross-validation results
  * Training time
  * Inference speed
- Metrics are stored in `model_metrics` dictionary:
  * Best parameters
  * Best scores
  * Final metrics
  * Training history
- Best performing models are used for final recommendations:
  * Performance-based selection
  * Diversity consideration
  * Computational efficiency
  * Real-time requirements

## Technical Requirements

### Dependencies
- Python 3.x
- Key libraries:
  - pandas
  - numpy
  - scikit-learn
  - PyTorch
  - NLTK
  - scipy
  - optuna

### Data Requirements
- Movie dataset in CSV format
- Required columns:
  - id
  - title
  - overview
  - release_date
  - genre_ids
  - keywords
  - cast
  - crew
  - popularity
  - vote_average
  - vote_count

## Performance Considerations

### Optimization Techniques
1. Feature Reduction:
   - PCA for dimensionality reduction
   - SVD for sparse matrices
   - Feature selection based on importance

2. Training Optimizations:
   - Early stopping
   - Batch processing
   - Parallel processing where applicable

3. Memory Management:
   - Sparse matrix storage
   - Efficient data structures
   - Batch processing for large datasets

## Future Improvements

1. Model Enhancements:
   - Implement deep learning models
   - Add collaborative filtering
   - Incorporate user feedback

2. Performance Optimizations:
   - Implement caching
   - Add batch processing
   - Optimize memory usage

3. Feature Engineering:
   - Add more text features
   - Incorporate temporal features
   - Add user interaction features 