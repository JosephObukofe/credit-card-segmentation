import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from credit_card_segmentation.data.custom import (
    train_kmeans_clusterer,
    train_dbscan_clusterer,
    train_hierarchical_clusterer,
)
from credit_card_segmentation.config.config import (
    MODEL_TRAINING_DATA,
    TRAINED_KMEANS_CLUSTERER,
    TRAINED_DBSCAN_CLUSTERER,
    TRAINED_HIERARCHICAL_CLUSTERER,
    TRAINED_KMEANS_FEATURES,
    TRAINED_DBSCAN_FEATURES,
    TRAINED_HIERARCHICAL_FEATURES,
)

X_train = joblib.load(MODEL_TRAINING_DATA)

features = [feature for feature in X_train.columns]

kmeans_model_param_grid = {
    "n_clusters": [2, 3, 4, 5, 6],
    "init": ["k-means++", "random"],
    "n_init": [10, 20],
    "max_iter": [300, 500, 700],
}

dbscan_model_param_grid = {
    "eps": [0.1, 0.3, 0.5, 0.7],
    "min_samples": [5, 10, 15],
    "metric": ["euclidean", "manhattan"],
}

hierarchical_model_param_grid = {
    "n_clusters": [2, 3, 4, 5, 6],
    "linkage": ["ward", "complete", "average", "single"],
    "metric": ["euclidean", "manhattan"],
}


def parallel_model_training():
    tasks = [
        delayed(train_kmeans_clusterer)(X_train, kmeans_model_param_grid, features),
        delayed(train_dbscan_clusterer)(X_train, dbscan_model_param_grid, features),
        delayed(train_hierarchical_clusterer)(
            X_train, hierarchical_model_param_grid, features
        ),
    ]

    trained_models = Parallel(n_jobs=-1, verbose=10)(tasks)

    for i, model_output in enumerate(trained_models):
        print(f"Model {i + 1} Training Output:")
        print(f"Model: {model_output['Model']}")
        print(f"Optimal Features: {model_output['Optimal Features']}")
        print(f"Silhouette Score: {model_output['Silhouette Score']}")
        print("\n")

    return trained_models


# Training each model in parallel
if __name__ == "__main__":
    trained_models_output = parallel_model_training()

# Save each trained model to its respective path
joblib.dump(trained_models_output[0]["Model"], TRAINED_KMEANS_CLUSTERER)
joblib.dump(trained_models_output[1]["Model"], TRAINED_DBSCAN_CLUSTERER)
joblib.dump(trained_models_output[2]["Model"], TRAINED_HIERARCHICAL_CLUSTERER)

# Save the optimal features of each model
joblib.dump(trained_models_output[0]["Optimal Features"], TRAINED_KMEANS_FEATURES)
joblib.dump(trained_models_output[1]["Optimal Features"], TRAINED_DBSCAN_FEATURES)
joblib.dump(trained_models_output[2]["Optimal Features"], TRAINED_HIERARCHICAL_FEATURES)
