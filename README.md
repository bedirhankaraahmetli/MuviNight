# MuviNight - Hybrid Movie Recommendation System

A sophisticated movie recommendation system that combines multiple machine learning approaches to provide personalized movie recommendations. The system utilizes content-based filtering, collaborative filtering, and deep learning techniques to deliver accurate and diverse movie suggestions.

## Features

- **Hybrid Recommendation System**
  - Content-based filtering using movie metadata and NLP
  - Genre-based recommendations
  - Cast-based suggestions
  - Text-based matching using TF-IDF
  - Neural network embeddings
  - Collaborative filtering using matrix factorization

- **Advanced Machine Learning Models**
  - K-Nearest Neighbors with cosine similarity
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting
  - Support Vector Machine (SVM)
  - Neural Network with PyTorch
  - SVD (Singular Value Decomposition)

- **User Interface**
  - Modern, responsive web interface
  - Movie poster previews
  - Detailed movie information
  - Support for selecting 2-5 movies for personalized recommendations
  - "Surprise Me" feature for random recommendations
  - User preference management

## Project Structure

```
MuviNight/
├── data/                  # Data storage
│   ├── raw/              # Original dataset
│   └── processed/        # Processed features and data
├── models/               # Trained model storage
├── src/                  # Source code
│   ├── data/            # Data processing scripts
│   │   └── data_pipeline.py
│   ├── models/          # Model training scripts
│   │   ├── model_trainer.py
│   │   └── recommender.py
│   ├── api/             # Flask API
│   │   └── api.py
│   └── utils/           # Utility functions
├── frontend-react/      # React frontend
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm 6+
- Virtual environment (recommended)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MuviNight.git
cd MuviNight
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Run the data pipeline:
```bash
python src/data/data_pipeline.py
```

5. Train the models:
```bash
python src/models/model_trainer.py
```

6. Start the Flask API:
```bash
python src/api/api.py
```

7. Start the frontend (in a new terminal):
```bash
cd frontend-react
npm install
npm start
```

## API Endpoints

### Movie Recommendations
- `POST /api/recommend`
  - Get personalized movie recommendations
  - Request body: List of movie IDs
  - Response: List of recommended movies with scores

### Movie Information
- `GET /api/movies`
  - Get list of all movies
  - Query parameters: page, limit, search
- `GET /api/movies/<id>`
  - Get detailed movie information
  - Includes: title, overview, genres, cast, etc.

### Special Features
- `GET /api/surprise`
  - Get random movie recommendations
  - Query parameters: count, genre

## Model Details

### Content-Based Models
- **Text-Based Model**
  - TF-IDF vectorization
  - 5000 maximum features
  - English stop words removal
- **Genre-Based Model**
  - One-hot encoded genres
  - Cosine similarity
- **Cast-Based Model**
  - Top 100 actors
  - One-hot encoding
  - Cosine similarity

### Machine Learning Models
- **Neural Network**
  - PyTorch implementation
  - Multiple hidden layers
  - Dropout regularization
  - Early stopping
- **Gradient Boosting**
  - 50 estimators
  - Learning rate: 0.1
  - Max depth: 3
- **SVM**
  - Linear kernel
  - Probability estimates
  - Scale-based gamma

## Performance

The system uses multiple metrics for evaluation:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Reconstruction Error
- Cross-validation scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Movie Database (TMDb) for the dataset
- Create React App for the frontend framework
- Flask for the backend API
- All the open-source libraries used in this project 