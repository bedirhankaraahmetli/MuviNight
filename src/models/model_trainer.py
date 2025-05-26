import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import sparse
import json
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import optuna
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

class MovieDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.3):
        super(NeuralNet, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_units)-1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_units[-1], input_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class ModelTrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.features = {}
        self.models = {}
        self.scalers = {}
        self.model_metrics = {}
        self.load_data()
        
    def load_data(self):
        """Load processed data and features"""
        print("Loading processed data...")
        
        # Load text features (sparse matrix)
        self.features['text'] = sparse.load_npz(os.path.join(self.data_dir, 'text_features.npz'))
        
        # Load vocabulary
        with open(os.path.join(self.data_dir, 'vocabulary.json'), 'r') as f:
            self.vocabulary = json.load(f)
        
        # Load other features
        feature_files = {
            'genre': 'genre_features.csv',
            'keyword': 'keyword_features.csv',
            'cast': 'cast_features.csv',
            'numeric': 'numeric_features.csv',
            'language': 'language_features.npy'
        }
        
        for name, file in feature_files.items():
            if file.endswith('.csv'):
                self.features[name] = pd.read_csv(os.path.join(self.data_dir, file), index_col=0)
            else:
                self.features[name] = np.load(os.path.join(self.data_dir, file))
                
    def train_models(self):
        """Train different recommendation models with hyperparameter tuning"""
        print("Training models with hyperparameter tuning...")
        
        # 1. Content-based model using all features
        self._train_content_based_model()
        
        # 2. Genre-based model
        self._train_genre_model()
        
        # 3. Cast-based model
        self._train_cast_model()
        
        # 4. Text-based model
        self._train_text_model()
        
        # 5. Neural Network model
        self._train_neural_network()
        
        # 6. Gradient Boosting model
        self._train_gradient_boosting()
        
        # 7. SVM model
        self._train_svm()
        
        # Print model comparison
        self._print_model_comparison()

    def _train_content_based_model(self):
        """Train a content-based model using all features with hyperparameter tuning"""
        print("Training content-based model with hyperparameter tuning...")
        
        # Combine all features
        feature_matrices = []
        
        # Add text features
        feature_matrices.append(self.features['text'].toarray())
        
        # Add genre features
        feature_matrices.append(self.features['genre'].values)
        
        # Add keyword features
        feature_matrices.append(self.features['keyword'].values)
        
        # Add cast features
        feature_matrices.append(self.features['cast'].values)
        
        # Add numeric features
        numeric_features = self.features['numeric'].values
        scaler = StandardScaler()
        numeric_features_scaled = scaler.fit_transform(numeric_features)
        feature_matrices.append(numeric_features_scaled)
        
        # Add language features (one-hot encoded)
        language_features = np.eye(len(np.unique(self.features['language'])))[self.features['language']]
        feature_matrices.append(language_features)
        
        # Combine all features
        combined_features = np.hstack(feature_matrices)
        
        # Define hyperparameter space for PCA
        def objective(trial):
            n_components = trial.suggest_int('n_components', 50, 200)
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(combined_features)
            similarity_matrix = cosine_similarity(reduced_features)
            
            # Evaluate using cross-validation with multiple metrics
            scores = {
                'mse': [],
                'mae': [],
                'reconstruction_error': []
            }
            
            for i in range(5):  # 5-fold cross validation
                test_indices = np.random.choice(len(combined_features), size=len(combined_features)//5, replace=False)
                train_indices = np.setdiff1d(np.arange(len(combined_features)), test_indices)
                
                # Original features
                test_features = combined_features[test_indices]
                train_features = combined_features[train_indices]
                
                # Reconstructed features
                test_reconstructed = pca.inverse_transform(pca.transform(test_features))
                
                # Calculate various metrics
                mse = mean_squared_error(test_features, test_reconstructed)
                mae = mean_absolute_error(test_features, test_reconstructed)
                reconstruction_error = np.mean(np.abs(test_features - test_reconstructed))
                
                scores['mse'].append(mse)
                scores['mae'].append(mae)
                scores['reconstruction_error'].append(reconstruction_error)
            
            # Return weighted combination of metrics
            return np.mean(scores['mse']) + 0.5 * np.mean(scores['mae']) + 0.3 * np.mean(scores['reconstruction_error'])
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        # Train final model with best parameters
        best_n_components = study.best_params['n_components']
        pca = PCA(n_components=best_n_components)
        reduced_features = pca.fit_transform(combined_features)
        similarity_matrix = cosine_similarity(reduced_features)
        
        # Calculate final metrics
        reconstructed_features = pca.inverse_transform(reduced_features)
        final_mse = mean_squared_error(combined_features, reconstructed_features)
        final_mae = mean_absolute_error(combined_features, reconstructed_features)
        final_reconstruction_error = np.mean(np.abs(combined_features - reconstructed_features))
        
        # Save model components
        self.models['content_based'] = {
            'pca': pca,
            'similarity_matrix': similarity_matrix,
            'scaler': scaler,
            'best_params': study.best_params
        }
        
        # Store metrics
        self.model_metrics['content_based'] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'final_metrics': {
                'mse': final_mse,
                'mae': final_mae,
                'reconstruction_error': final_reconstruction_error
            }
        }

    def _train_genre_model(self):
        """Train a genre-based model"""
        print("Training genre-based model...")
        
        # Compute genre similarity matrix
        genre_similarity = cosine_similarity(self.features['genre'])
        
        # Save model
        self.models['genre'] = {
            'similarity_matrix': genre_similarity
        }
        
    def _train_cast_model(self):
        """Train a cast-based model"""
        print("Training cast-based model...")
        cast_matrix = self.features['cast']
        if cast_matrix.shape[1] == 0:
            print("Cast features are empty. Skipping cast-based model.")
            return
        # Compute cast similarity matrix
        cast_similarity = cosine_similarity(cast_matrix)
        # Save model
        self.models['cast'] = {
            'similarity_matrix': cast_similarity
        }
        
    def _train_text_model(self):
        """Train a text-based model"""
        print("Training text-based model...")
        
        # Compute text similarity matrix
        text_similarity = cosine_similarity(self.features['text'])
        
        # Save model
        self.models['text'] = {
            'similarity_matrix': text_similarity,
            'vocabulary': self.vocabulary
        }
        
    def _train_neural_network(self):
        """Train a neural network model with hyperparameter tuning"""
        print("Training neural network model...")
        
        # Prepare data
        X = self.features['text'].toarray()
        original_dim = X.shape[1]
        
        # Reduce dimensionality if needed
        if original_dim > 1000:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=1000)
            X = svd.fit_transform(X)
            print(f"Reduced features from {original_dim} to {X.shape[1]} dimensions")
        
        # Convert to PyTorch tensors and create dataset
        X_tensor = torch.FloatTensor(X)
        dataset = MovieDataset(X_tensor)
        batch_size = min(256, len(X))  # Use larger batch size for faster training
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define hyperparameter space
        def objective(trial):
            hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
            hidden_units = []
            for i in range(hidden_layers):
                hidden_units.append(trial.suggest_int(f'hidden_units_{i}', 64, 512))
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            
            model = NeuralNet(X.shape[1], hidden_units, dropout_rate)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop with validation
            best_val_loss = float('inf')
            patience = 3  # Reduced patience for faster training
            patience_counter = 0
            n_epochs = 30  # Reduced epochs for faster training
            
            for epoch in range(n_epochs):
                model.train()
                total_loss = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch)
                    loss = criterion(outputs, batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_tensor)
                    val_loss = criterion(val_outputs, X_tensor)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        break
            
            # Load best model for final evaluation
            model.load_state_dict(best_model_state)
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                embeddings = model(X_tensor).numpy()
                similarity_matrix = cosine_similarity(embeddings)
                
                # Calculate multiple metrics
                mse = mean_squared_error(X, embeddings)
                mae = mean_absolute_error(X, embeddings)
                reconstruction_error = np.mean(np.abs(X - embeddings))
                
                # Return weighted combination of metrics
                return mse + 0.5 * mae + 0.3 * reconstruction_error
        
        # Optimize hyperparameters with reduced trials
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)  # Reduced number of trials
        
        # Train final model with best parameters
        best_params = study.best_params
        model = NeuralNet(
            X.shape[1],
            [best_params[f'hidden_units_{i}'] for i in range(best_params['hidden_layers'])],
            best_params['dropout_rate']
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()
        
        # Final training with early stopping
        print("Starting final training with best parameters...")
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        n_epochs = 50  # More epochs for final training
        
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_tensor)
                val_loss = criterion(val_outputs, X_tensor)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Get final embeddings and metrics
        print("Generating final embeddings...")
        model.eval()
        with torch.no_grad():
            embeddings = model(X_tensor).numpy()
            similarity_matrix = cosine_similarity(embeddings)
            
            # Calculate final metrics
            final_mse = mean_squared_error(X, embeddings)
            final_mae = mean_absolute_error(X, embeddings)
            reconstruction_error = np.mean(np.abs(X - embeddings))
        
        # Save model
        self.models['neural_network'] = {
            'model': model,
            'similarity_matrix': similarity_matrix,
            'best_params': best_params,
            'svd': svd if 'svd' in locals() else None
        }
        
        # Store metrics
        self.model_metrics['neural_network'] = {
            'best_params': best_params,
            'best_score': study.best_value,
            'final_metrics': {
                'mse': final_mse,
                'mae': final_mae,
                'reconstruction_error': reconstruction_error
            }
        }
        print("Neural network model training completed")

    def _train_gradient_boosting(self):
        """Train a gradient boosting model with hyperparameter tuning"""
        print("Training gradient boosting model...")
        
        # Use a subset of features to speed up training
        X = self.features['text'].toarray()
        original_dim = X.shape[1]
        if original_dim > 500:  # Reduced threshold for feature reduction
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=500)  # Reduced number of components
            X = svd.fit_transform(X)
            print(f"Reduced features from {original_dim} to {X.shape[1]} dimensions")
        
        # Create synthetic labels for clustering with fewer clusters
        kmeans = KMeans(n_clusters=min(5, len(X)), random_state=42, n_init=10)
        y = kmeans.fit_predict(X)
        
        # Define smaller hyperparameter space
        param_dist = {
            'n_estimators': [50],  # Fixed number of estimators
            'learning_rate': [0.1],  # Fixed learning rate
            'max_depth': [3],  # Fixed depth
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }
        
        # Initialize model with reduced complexity
        model = GradientBoostingClassifier(
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        # Perform random search with fewer iterations
        scoring = {
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error'
        }
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=1,  # Single iteration since we're using fixed parameters
            cv=min(3, len(np.unique(y))),
            scoring=scoring,
            refit='mse',
            random_state=42,
            n_jobs=-1
        )
        
        print("Starting hyperparameter search...")
        random_search.fit(X, y)
        print("Hyperparameter search completed")
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Generate predictions and similarity matrix
        print("Generating predictions...")
        predictions = best_model.predict_proba(X)
        similarity_matrix = cosine_similarity(predictions)
        
        # Calculate final metrics
        final_mse = -random_search.cv_results_['mean_test_mse'][random_search.best_index_]
        final_mae = -random_search.cv_results_['mean_test_mae'][random_search.best_index_]
        
        # Calculate reconstruction error using cluster assignments
        cluster_centers = kmeans.cluster_centers_
        predicted_clusters = best_model.predict(X)
        reconstruction_error = np.mean(np.abs(X - cluster_centers[predicted_clusters]))
        
        # Save model
        self.models['gradient_boosting'] = {
            'model': best_model,
            'similarity_matrix': similarity_matrix,
            'best_params': random_search.best_params_,
            'kmeans': kmeans,
            'svd': svd if 'svd' in locals() else None
        }
        
        # Store metrics
        self.model_metrics['gradient_boosting'] = {
            'best_params': random_search.best_params_,
            'best_score': final_mse,
            'final_metrics': {
                'mse': final_mse,
                'mae': final_mae,
                'reconstruction_error': reconstruction_error
            }
        }
        print("Gradient boosting model training completed")

    def _train_svm(self):
        """Train an SVM model with hyperparameter tuning"""
        print("Training SVM model...")
        
        # Use a subset of features to speed up training
        X = self.features['text'].toarray()
        original_dim = X.shape[1]
        if original_dim > 500:  # Reduced threshold for feature reduction
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=500)  # Reduced number of components
            X = svd.fit_transform(X)
            print(f"Reduced features from {original_dim} to {X.shape[1]} dimensions")
        
        # Scale the features to help with convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create synthetic labels for clustering with fewer clusters
        kmeans = KMeans(n_clusters=min(5, len(X)), random_state=42, n_init=10)
        y = kmeans.fit_predict(X_scaled)
        
        # Define smaller hyperparameter space with fixed parameters
        param_dist = {
            'C': [1],  # Fixed C parameter
            'kernel': ['linear'],  # Only use linear kernel
            'gamma': ['scale']
        }
        
        # Initialize model with optimized settings
        model = SVC(
            probability=True,
            cache_size=2000,
            random_state=42,
            max_iter=2000  # Increased max iterations
        )
        
        # Perform random search with single iteration
        scoring = {
            'mse': 'neg_mean_squared_error',
            'mae': 'neg_mean_absolute_error'
        }
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=1,  # Single iteration since we're using fixed parameters
            cv=min(3, len(np.unique(y))),
            scoring=scoring,
            refit='mse',
            random_state=42,
            n_jobs=-1
        )
        
        print("Starting hyperparameter search...")
        random_search.fit(X_scaled, y)
        print("Hyperparameter search completed")
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Generate predictions and similarity matrix
        print("Generating predictions...")
        predictions = best_model.predict_proba(X_scaled)
        similarity_matrix = cosine_similarity(predictions)
        
        # Calculate final metrics
        final_mse = -random_search.cv_results_['mean_test_mse'][random_search.best_index_]
        final_mae = -random_search.cv_results_['mean_test_mae'][random_search.best_index_]
        
        # Calculate reconstruction error using cluster assignments
        cluster_centers = kmeans.cluster_centers_
        predicted_clusters = best_model.predict(X_scaled)
        reconstruction_error = np.mean(np.abs(X_scaled - cluster_centers[predicted_clusters]))
        
        # Save model
        self.models['svm'] = {
            'model': best_model,
            'similarity_matrix': similarity_matrix,
            'best_params': random_search.best_params_,
            'kmeans': kmeans,
            'svd': svd if 'svd' in locals() else None,
            'scaler': scaler
        }
        
        # Store metrics
        self.model_metrics['svm'] = {
            'best_params': random_search.best_params_,
            'best_score': final_mse,
            'final_metrics': {
                'mse': final_mse,
                'mae': final_mae,
                'reconstruction_error': reconstruction_error
            }
        }
        print("SVM model training completed")

    def _print_model_comparison(self):
        """Print comparison table of all models"""
        print("\nModel Comparison Table:")
        print("-" * 100)
        print(f"{'Model':<20} {'Best Parameters':<40} {'MSE':<10} {'MAE':<10} {'Reconstruction Error':<15}")
        print("-" * 100)
        
        for model_name, metrics in self.model_metrics.items():
            params_str = str(metrics['best_params'])
            if len(params_str) > 37:
                params_str = params_str[:34] + "..."
            
            # Get metrics
            mse = metrics.get('final_metrics', {}).get('mse', metrics['best_score'])
            mae = metrics.get('final_metrics', {}).get('mae', 'N/A')
            recon_error = metrics.get('final_metrics', {}).get('reconstruction_error', 'N/A')
            
            print(f"{model_name:<20} {params_str:<40} {mse:<10.4f} {mae:<10.4f} {recon_error:<15.4f}")
        
        print("-" * 100)
        
        # Find best model based on MSE
        best_model = min(self.model_metrics.items(), key=lambda x: x[1].get('final_metrics', {}).get('mse', x[1]['best_score']))
        print(f"\nBest performing model: {best_model[0]}")
        print(f"Best MSE: {best_model[1].get('final_metrics', {}).get('mse', best_model[1]['best_score']):.4f}")
        print(f"Best parameters: {best_model[1]['best_params']}")

    def save_models(self, output_dir):
        """Save trained models and metrics"""
        print("Saving models and metrics...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{name}_model.joblib'))
        
        # Save metrics
        with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
            json.dump(self.model_metrics, f, indent=4)

def main():
    # Initialize trainer
    trainer = ModelTrainer('data/processed')
    
    # Train models
    trainer.train_models()
    
    # Save models
    trainer.save_models('models')
    
if __name__ == "__main__":
    main() 