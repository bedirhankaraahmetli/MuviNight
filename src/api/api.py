from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import sys
import os
import json
import numpy as np

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recommender import MovieRecommender

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize recommender
recommender = MovieRecommender('data/processed', 'models')

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get movie recommendations based on selected movies"""
    data = request.get_json()
    print("Received data:", data)
    movie_ids = data.get('movie_ids', [])
    n_recommendations = data.get('n_recommendations', 5)
    model_name = data.get('model_name', 'knn')
    
    try:
        recommendations = recommender.get_recommendations(
            movie_ids, 
            n_recommendations=n_recommendations,
            model_name=model_name
        )
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        print("Error in /api/recommend:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get list of all movies"""
    try:
        # Replace NaN with None
        movies_df = recommender.movies[
            ['id', 'title', 'overview', 'poster_path', 'release_date', 'vote_average']
        ].replace({np.nan: None})
        movies = movies_df.to_dict('records')
        print("DEBUG: movies type:", type(movies))
        print("DEBUG: first movie:", movies[0] if movies else "No movies")
        return Response(json.dumps({'movies': movies}), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """Get details for a specific movie"""
    try:
        movie = recommender.get_movie_details(movie_id)
        if movie is None:
            return jsonify({'error': 'Movie not found'}), 404
        return jsonify(movie)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search movies by title"""
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 10))
    
    try:
        results = recommender.search_movies(query, limit=limit)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/surprise', methods=['GET'])
def get_surprise_recommendations():
    """Get random movie recommendations"""
    n_recommendations = int(request.args.get('n_recommendations', 5))
    
    try:
        recommendations = recommender._get_random_recommendations(n_recommendations)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/best_model', methods=['GET'])
def get_best_model():
    """Return the best model name based on model_metrics.json (lowest MSE)"""
    try:
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        # Find the model with the lowest MSE
        best_model = min(
            metrics.items(),
            key=lambda x: x[1].get('final_metrics', {}).get('mse', x[1].get('best_score', float('inf')))
        )[0]
        return jsonify({'best_model': best_model})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_metrics', methods=['GET'])
def get_model_metrics():
    """Return the comparison metrics for all models"""
    try:
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test')
def test():
    return jsonify({'hello': 'world'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0') 