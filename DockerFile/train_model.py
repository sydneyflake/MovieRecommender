"""
Model Training Script
Train the LensKit BiasedMF model and save it for use in the app.
"""

import pandas as pd
import pickle
from lenskit.algorithms.als import BiasedMF
from lenskit import Recommender
from pathlib import Path
import json


def load_data():
    """Load and prepare the MovieLens data."""
    print("Loading data...")
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    return ratings, movies

def normalize_title(title: str) -> str:
    parts = title.rsplit("(", 1)  # separate year if present
    name = parts[0].strip()
    year = "(" + parts[1] if len(parts) > 1 else ""

    for article in [" The", " A", " An"]:
        suffix = f", {article}"
        if name.endswith(suffix):
            name = f"{article} {name[: -len(suffix)]}"

    return f"{name.strip()} {year}".strip()

def prepare_training_data(ratings):
    """Prepare data in LensKit format."""
    print("Preparing training data...")
    train = ratings.rename(columns={
        'userId': 'user', 
        'movieId': 'item'
    })[['user', 'item', 'rating']].copy()
    return train

def train_model(train_data, features=50, iterations=40, reg=0.02, damping=5.0):
    """Train the BiasedMF model."""
    print("Training model...")
    mf = BiasedMF(features, iterations=iterations, reg=reg, damping=damping)
    rec = Recommender.adapt(mf).fit(train_data)
    print("Model training complete!")
    return rec

def save_model(model, path='models/lenskit_model.pkl'):
    """Save the trained model to disk."""
    print(f"Saving model to {path}...")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")


def main():
    """Main training pipeline."""
    # Load data
    ratings, movies = load_data()
    
    # Prepare training data
    train = prepare_training_data(ratings)
    
    # Train model
    model = train_model(train)
    
    # Save model
    save_model(model)
    
    # Save movies data for the app
    movies.to_csv('models/movies.csv', index=False)
    print("Movies data saved!")
    
    print("\nTraining complete! Model ready for use.")

if __name__ == "__main__":
    main()