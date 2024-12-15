import numpy as np
import pandas as pd
import joblib
import mlflow
import logging
from skopt.space import Categorical, Integer
from joblib import Parallel, delayed
from credit_card_segmentation.data.custom import (
    train_kmeans_clusterer_with_grid_search,
    train_dbscan_clusterer_with_grid_search,
    train_hierarchical_clusterer_with_grid_search,
)
from credit_card_segmentation.config.config import (
    MODEL_TRAINING_DATA,
    TRAINED_KMEANS_CLUSTERER,
    TRAINED_DBSCAN_CLUSTERER,
    TRAINED_HIERARCHICAL_CLUSTERER,
    TRAINED_KMEANS_FEATURES,
    TRAINED_DBSCAN_FEATURES,
    TRAINED_HIERARCHICAL_FEATURES,
    MLFLOW_TRACKING_URI
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mlflow.set_experiment("Credit Card Segmentation: Model Training")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

X_train = joblib.load(MODEL_TRAINING_DATA)
features = [feature for feature in X_train.columns]
trained_models = [TRAINED_KMEANS_CLUSTERER, TRAINED_DBSCAN_CLUSTERER, TRAINED_HIERARCHICAL_CLUSTERER]
optimal_features = [TRAINED_KMEANS_FEATURES, TRAINED_DBSCAN_FEATURES, TRAINED_HIERARCHICAL_FEATURES]

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


def safe_task(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}")
        return None
    

def parallel_model_training():
    tasks = [
        delayed(safe_task)(
            train_kmeans_clusterer_with_grid_search, 
            X_train, 
            kmeans_model_param_grid, 
            features
        ),
        delayed(safe_task)(
            train_dbscan_clusterer_with_grid_search, 
            X_train, 
            dbscan_model_param_grid, 
            features
        ),
        delayed(safe_task)(
            train_hierarchical_clusterer_with_grid_search,
            X_train, 
            hierarchical_model_param_grid, 
            features
        ),
    ]

    trained_models_output = Parallel(n_jobs=4, verbose=10)(tasks)
    for i, model_output in enumerate(trained_models_output):
        if model_output:
            logging.info(f"Model {i + 1} Training Successful")
            logging.info(f"Silhouette Score: {model_output['Silhouette Score']}")
        else:
            logging.warning(f"Model {i + 1} Training Failed")
    return trained_models_output


if __name__ == "__main__":
    trained_models_output = parallel_model_training()

    for i, model_output in enumerate(trained_models_output):
        if model_output:
            joblib.dump(model_output["Model"], trained_models[i])
            joblib.dump(model_output["Optimal Features"], optimal_features[i])