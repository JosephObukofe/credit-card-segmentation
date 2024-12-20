import os
import io
import boto3
import pickle
import logging
import tempfile
import mlflow.tracking
import numpy as np
import pandas as pd
import urllib.parse
import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from botocore.exceptions import BotoCoreError, ClientError
from minio import Minio
from minio.error import MinioException, S3Error
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union, Optional
from matplotlib import pyplot as plt
from skopt import BayesSearchCV
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin, clone
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
    davies_bouldin_score,
)
from credit_card_segmentation.config.config import (
    MLFLOW_TRACKING_URI,
    MINIO_CONSOLE_ADDRESS,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
)


class Transformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies various transformations to numerical features,
    including Yeo-Johnson stabilization, binary encoding, and weighted encoding for zero values.

    Parameters
    ----------
    encoding_type : str, default="weighted"
        The type of transformation to apply. Options are:
        - "yeo-johnson": Applies Yeo-Johnson transformation to non-zero values.
        - "binary": Creates a binary feature indicating the presence or absence of non-zero values.
        - "weighted": Replaces zero values with a specified weight (`zero_weight`).
        - "none": No transformation applied, but can encode zero values using `none_encoding_type`.
    zero_weight : float, default=0.5
        The weight to assign to zero values when `encoding_type="weighted"` or `none_encoding_type="weighted"` is used.
    none_encoding_type : str, default="binary"
        The method used to encode zero values when `encoding_type="none"`. Options are:
        - "binary": Replace zero values with 1.
        - "weighted": Replace zero values with the specified `zero_weight`.

    Attributes
    ----------
    yeo_johnson : PowerTransformer
        An instance of sklearn's PowerTransformer for applying Yeo-Johnson transformation.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the data. If `encoding_type` is not "none", applies Yeo-Johnson
        transformation to non-zero values of the feature matrix.
    transform(X)
        Applies the transformation based on the specified `encoding_type`. Can apply Yeo-Johnson,
        binary encoding, weighted encoding, or no transformation based on the configuration.
    get_feature_names_out(input_features=None)
        Returns transformed feature names. If no input features are provided, generates generic feature names.
    """

    def __init__(
        self,
        encoding_type="weighted",
        zero_weight=0.5,
        none_encoding_type="binary",
    ):
        self.encoding_type = encoding_type
        self.zero_weight = zero_weight
        self.none_encoding_type = none_encoding_type
        self.yeo_johnson = PowerTransformer(method="yeo-johnson", standardize=False)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if self.encoding_type != "none":
            non_zero_values = X[X != 0].reshape(-1, 1)
            if len(non_zero_values) > 0:
                self.yeo_johnson.fit(non_zero_values)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        X_transformed = np.copy(X).astype(float)
        non_zero_mask = X_transformed != 0

        if self.encoding_type != "none" and non_zero_mask.any():
            X_transformed[non_zero_mask] = self.yeo_johnson.transform(
                X_transformed[non_zero_mask].reshape(-1, 1)
            ).flatten()

        zero_mask = ~non_zero_mask

        if self.encoding_type == "yeo-johnson":
            return X_transformed
        elif self.encoding_type == "binary":
            binary_column = non_zero_mask.astype(float)
            X_transformed_with_binary = np.column_stack([X_transformed, binary_column])
            return X_transformed_with_binary
        elif self.encoding_type == "weighted":
            X_transformed[zero_mask] = self.zero_weight
            return X_transformed
        elif self.encoding_type == "none":
            if self.none_encoding_type == "binary":
                X_transformed[zero_mask] = 1
            elif self.none_encoding_type == "weighted":
                X_transformed[zero_mask] = self.zero_weight
            return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(len(input_features))])
        return np.array([f"{feature}_transformed" for feature in input_features])


class ImputeAndStabilize(TransformerMixin, BaseEstimator):
    """
    A custom transformer that combines an imputer and a transformer for stabilizing features.
    First imputes missing values, then applies a transformation to stabilize the feature distribution.

    Parameters
    ----------
    imputer : sklearn.base.BaseEstimator
        An estimator for imputing missing values (e.g., SimpleImputer).

    transformer : Transformer
        A custom transformer object that applies a feature-wise transformation, such as stabilizing skewed features
        or applying scaling techniques like Yeo-Johnson.

    Methods
    -------
    fit(X, y=None)
        First imputes the missing values in the data using the specified imputer,
        then fits the transformer to the imputed data.

    transform(X)
        Imputes missing values and then applies the transformation to the imputed data.
    """

    def __init__(self, imputer, transformer):
        self.imputer = imputer
        self.transformer = transformer

    def fit(self, X, y=None):
        X_imputed = self.imputer.fit_transform(X)
        self.transformer.fit(X_imputed)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        return self.transformer.transform(X_imputed)


class NoCV(BaseCrossValidator):
    """
    Custom CV splitter that returns the entire dataset as training and validation.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        yield indices, indices


class DBSCANWithPredict(BaseEstimator):
    def __init__(self, **kwargs):
        self.model = DBSCAN(**kwargs)

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def predict(self, X):
        return self.model.fit_predict(X)

    def fit_predict(self, X, y=None):
        return self.model.fit_predict(X)


def preprocess_training_set(
    X: pd.DataFrame,
    missing_value_features: List[str],
    missing_value_skewed_features: List[str],
    sparse_skewed_features: List[str],
    sparse_features: List[str],
    skewed_features: List[str],
) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by applying the following transformations:

    1. Imputes missing values and stabilizes skewness for certain features.
    2. Imputes missing values for other features.
    3. Densifies sparse and skewed features and applies transformation to handle skewness.
    4. Densifies sparse features.
    5. Stabilizes skewed features.
    6. Scales the data using StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing features to be preprocessed.
    missing_value_features : List[str]
        List of features that have missing values and require imputation (but no skewness correction).
    missing_value_skewed_features : List[str]
        List of features that have missing values and need both imputation and skewness correction.
    sparse_skewed_features : List[str]
        List of sparse and skewed features that need to be densified and stabilized.
    sparse_features : List[str]
        List of sparse features that need to be densified (but no skewness correction).
    skewed_features : List[str]
        List of features that need stabilization (skewness correction).

    Returns
    -------
    Tuple[pd.DataFrame, Pipeline]
        A tuple where:
        - pd.DataFrame: A new DataFrame where the original features have been imputed, densified, stabilized, and scaled,
          based on the transformations applied to each subset of features.
        - Pipeline: A fitted pipeline that includes the preprocessing steps and scaling, which can be used to transform the test set.
    """

    imputer_stabilizer = ImputeAndStabilize(
        imputer=SimpleImputer(strategy="mean"),
        transformer=Transformer(encoding_type="yeo-johnson"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("impute_stabilize", imputer_stabilizer, missing_value_skewed_features),
            ("impute", SimpleImputer(strategy="mean"), missing_value_features),
            (
                "densify_stabilize",
                Transformer(encoding_type="weighted", zero_weight=0.5),
                sparse_skewed_features,
            ),
            (
                "densify",
                Transformer(
                    encoding_type="none", none_encoding_type="weighted", zero_weight=0.5
                ),
                sparse_features,
            ),
            (
                "stabilize",
                Transformer(encoding_type="yeo-johnson"),
                skewed_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[("preprocessing", preprocessor), ("scaling", StandardScaler())]
    )

    processed_array = pipeline.fit_transform(X)

    transformed_column_names = (
        [feature + "_imputed_stabilized" for feature in missing_value_skewed_features]
        + [feature + "_imputed" for feature in missing_value_features]
        + [feature + "_stabilized_weighted" for feature in sparse_skewed_features]
        + [feature + "_weighted" for feature in sparse_features]
        + [feature + "_stabilized" for feature in skewed_features]
    )

    processed_df = pd.DataFrame(processed_array, columns=transformed_column_names)
    X_dropped = X.drop(
        columns=missing_value_skewed_features
        + missing_value_features
        + sparse_skewed_features
        + sparse_features
        + skewed_features
    )

    final_df = pd.concat(
        [X_dropped.reset_index(drop=True), processed_df.reset_index(drop=True)],
        axis=1,
    )

    return final_df, pipeline


def preprocess_test_set(
    X: pd.DataFrame,
    preprocessor: BaseEstimator,
    missing_value_features: List[str],
    missing_value_skewed_features: List[str],
    sparse_skewed_features: List[str],
    sparse_features: List[str],
    skewed_features: List[str],
) -> pd.DataFrame:
    """
    Preprocesses the test set using a previously fitted pipeline and transformer.

    This function applies transformations learned during training to the test set.
    It does not fit or impute any information from the test set, thus preventing data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        The input test DataFrame to be preprocessed. It should contain the same features as the training set.
    preprocessor : BaseEstimator
        A previously fitted scikit-learn transformer or pipeline used for preprocessing.
        This preprocessor has been trained on the training data and will be used to transform the test set.
    missing_value_features : List[str]
        List of features that have missing values and require imputation during training.
    missing_value_skewed_features : List[str]
        List of features that have missing values and need imputation and skewness correction during training.
    sparse_skewed_features : List[str]
        List of sparse and skewed features that need densification and stabilization during training.
    sparse_features : List[str]
        List of sparse features that need densification (without skewness correction).
    skewed_features : List[str]
        List of features that need stabilization (skewness correction) during training.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the preprocessed test set, with the same transformed features as in the training set.
        The processed features will be concatenated with the original test set (if any columns are retained).
    """

    processed_array = preprocessor.transform(X)

    transformed_column_names = (
        [feature + "_imputed_stabilized" for feature in missing_value_skewed_features]
        + [feature + "_imputed" for feature in missing_value_features]
        + [feature + "_stabilized_weighted" for feature in sparse_skewed_features]
        + [feature + "_weighted" for feature in sparse_features]
        + [feature + "_stabilized" for feature in skewed_features]
    )

    processed_df = pd.DataFrame(processed_array, columns=transformed_column_names)
    X_dropped = X.drop(
        columns=missing_value_skewed_features
        + missing_value_features
        + sparse_skewed_features
        + sparse_features
        + skewed_features
    )

    final_df = pd.concat(
        [X_dropped.reset_index(drop=True), processed_df.reset_index(drop=True)],
        axis=1,
    )

    return final_df


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the CSV file that contains the data to be loaded.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the data from the specified CSV file.
    """

    raw_data = pd.read_csv(file_path)
    return raw_data


def read_csv_from_minio(
    bucket_name: str,
    object_name: str,
) -> pd.DataFrame:
    """
    Read a CSV file from MinIO and load it into a pandas DataFrame.

    Parameters
    ----------
    bucket_name : str
        The name of the MinIO bucket where the CSV file is stored.
    object_name : str
        The name of the CSV file (object) in the MinIO bucket.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data from the CSV file.

    Raises
    -------
    S3Error
        If the specified object does not exist in the MinIO bucket or other S3-related error.
    Exception
        For any other errors that may occur while reading the CSV file.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_CONSOLE_ADDRESS,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
    )

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        df = pd.read_csv(response['Body'])
        logger.info(f"Successfully loaded CSV data from '{object_name}' in bucket '{bucket_name}'")
        return df
    except S3Error as e:
        logger.error(f"S3 error: The object '{object_name}' does not exist in bucket '{bucket_name}' or another S3-related error occurred: {e}")
        return None
    except Minio.S3Error as e:
        logger.error(f"MinIO error connecting to bucket: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None


def upload_data_to_minio(
    dataframe: pd.DataFrame,
    bucket_name: str,
    object_name: str,
) -> None:
    """
    Uploads a pandas DataFrame to MinIO as a pickled object.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame to be uploaded to MinIO.
    bucket_name : str
        The name of the MinIO bucket where the pickled dataframe will be stored.
    object_name : str
        The name of the pickled dataframe (object) in the MinIO bucket.
        
    Returns
    -------
    None
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_CONSOLE_ADDRESS,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
    )

    try:
        with io.BytesIO() as buffer:
            pickle.dump(dataframe, buffer)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, bucket_name, object_name)
            print(f"Uploaded dataframe to MinIO bucket {bucket_name} as {object_name}")
    except ClientError as e:
        logger.error(f"Error uploading dataframe to MinIO: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def load_data_from_minio():
    pass


def recursive_cluster_feature_elimination(
    estimator: ClusterMixin,
    X_train: pd.DataFrame,
    features: List[str],
    group_size: int = 2,
    patience: int = 4,
) -> Tuple[ClusterMixin, List[str], float]:
    """
    Perform Recursive Cluster Feature Elimination (RCFE) with group evaluation to find the optimal set of features based on the silhouette score by iteratively removing groups of features, excluding noise points.

    Parameters:
    ----------
    estimator : ClusterMixin
        The clustering model (e.g., KMeans, DBSCAN, etc.) used for feature evaluation.
    X_train : pd.DataFrame
        The training dataset with all features.
    features : List[str]
        The initial list of feature names to consider.
    group_size : int, optional (default=2)
        The number of features to remove as a group during each iteration.
    patience : int, optional (default=3)
        Number of iterations to wait for improvement before stopping early.

    Returns:
    -------
    optimal_model : ClusterMixin
        The clustering model trained on the optimal set of features.
    optimal_features : List[str]
        The list of optimal features that achieved the best silhouette score.
    best_silhouette_score : float
        The highest silhouette score achieved during the process.
    """

    n_features = X_train.shape[1]
    current_features = features.copy()
    best_silhouette_score = -1
    optimal_features = current_features.copy()
    optimal_model = None
    no_improvement_count = 0

    print(
        f"Starting Recursive Cluster Feature Elimination with group size {group_size}."
    )

    while len(current_features) > group_size:
        scores = []
        models = []
        feature_groups = list(combinations(current_features, group_size))

        print(f"Evaluating {len(feature_groups)} groups of size {group_size}...")

        # Evaluate removing each group of features
        for group in feature_groups:
            remaining_features = [f for f in current_features if f not in group]

            if len(remaining_features) == 0:
                continue

            X_train_subset = X_train[remaining_features]
            model = clone(estimator)
            labels = model.fit_predict(X_train_subset)

            if len(set(labels)) > 1:
                # Exclude noise for DBSCAN-like algorithms
                mask = labels != -1
                if np.sum(mask) > 1:
                    score = silhouette_score(X_train_subset[mask], labels[mask])
                else:
                    score = -1
            else:
                score = -1

            scores.append(score)
            models.append((remaining_features, model))

        if not scores:
            break  # No valid scores, stop

        # Find the best group removal
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_remaining_features, best_model = models[best_idx]

        print(f"Best group removal leads to silhouette score: {best_score:.4f}")

        # Check for improvement
        if best_score > best_silhouette_score:
            best_silhouette_score = best_score
            optimal_features = best_remaining_features
            optimal_model = best_model
            no_improvement_count = 0  # Reset counter
            print(f"Improved score: {best_silhouette_score:.4f}. Continuing...")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} iteration(s).")

        # Early stopping condition
        if no_improvement_count >= patience:
            print(f"Early stopping after {patience} iterations with no improvement.")
            break

        current_features = optimal_features

    return optimal_model, optimal_features, best_silhouette_score


def calinski_harabasz_scorer(
    estimator: ClusterMixin,
    X: pd.DataFrame,
) -> float:
    """
    Computes the Calinski-Harabasz score for the given estimator's clustering results, ensuring there are at least 2 valid clusters.

    Parameters:
    ----------
    estimator : ClusterMixin
        The clustering model to evaluate.
    X : pd.DataFrame
        The dataset on which to perform clustering and scoring.

    Returns:
    -------
    float
        The Calinski-Harabasz score if valid, or -1 otherwise.
    """

    labels = estimator.fit_predict(X)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Exclude noise label (-1) for DBSCAN

    if len(unique_labels) < 2:
        # Return an invalid score when fewer than 2 clusters are present
        print(f"Invalid clustering: Only {len(unique_labels)} valid cluster(s) found.")
        return -1.0

    return calinski_harabasz_score(X, labels)


def hyperparameter_tuning(
    estimator: ClusterMixin,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    optimal_features: List[str],
) -> ClusterMixin:
    """
    Performs hyperparameter tuning on the clustering estimator using GridSearchCV optimizing for the Calinski-Harabasz score.

    Parameters:
    ----------
    estimator : ClusterMixin
        The clustering model (e.g., KMeans, DBSCAN) to tune.
    param_grid : Dict[str, List[Any]]
        Dictionary of hyperparameters to search through in grid search.
    X_train : pd.DataFrame
        The training dataset containing all features.
    optimal_features : List[str]
        The list of optimal feature names for use in model training.

    Returns:
    -------
    ClusterMixin
        The best clustering model found by GridSearchCV after hyperparameter tuning.
    """

    grid_search = GridSearchCV(
        estimator,
        param_grid,
        scoring=calinski_harabasz_scorer,
        cv=NoCV(),
        n_jobs=-1,
        verbose=1,
    )

    X_train_optimal = X_train[optimal_features]
    grid_search.fit(X_train_optimal)
    tuned_model = grid_search.best_estimator_
    return tuned_model


def bayesian_hyperparameter_tuning(
    estimator: ClusterMixin,
    search_spaces: Dict[str, Any],
    X_train: pd.DataFrame,
    optimal_features: List[str],
) -> ClusterMixin:
    """
    Performs hyperparameter tuning for a given clustering model using BayesSearchCV optimizing for the Calinski-Harabasz score.

    Parameters
    ----------
    estimator : ClusterMixin
        The clustering model to tune (e.g., KMeans, DBSCAN).
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    X_train : pd.DataFrame
        The training dataset containing all features.
    optimal_features : List[str]
        The list of optimal feature names for use in model training.

    Returns
    -------
    ClusterMixin
        The best clustering model found by BayesSearchCV after hyperparameter tuning.
    """

    bayes_search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        scoring=calinski_harabasz_scorer,
        cv=NoCV(),
        refit=True,
        random_state=42,
        n_iter=50,
        n_jobs=6,
        verbose=1,
    )

    X_train_optimal = X_train[optimal_features]
    bayes_search.fit(X_train_optimal)
    tuned_model = bayes_search.best_estimator_
    return tuned_model


def train_kmeans_clusterer_with_grid_search(
    X_train: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains a KMeans clustering model, perform recursive feature elimination to find the optimal set of features, and tunes the model using grid search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    param_grid : Dict[str, List[Any]]
        Dictionary of hyperparameters to search through in grid search.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned KMeans model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"KMeans Clusterer Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Credit Card KMeans Clusterer"

    with mlflow.start_run(run_name=model_name) as run:
        kmeans_model = KMeans(random_state=42)

        (
            kmeans_fitted_model,
            kmeans_optimal_features,
            kmeans_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            kmeans_model,
            X_train,
            features,
        )

        kmeans_tuned_model = hyperparameter_tuning(
            kmeans_fitted_model,
            param_grid,
            X_train,
            kmeans_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=kmeans_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": kmeans_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"KMeans Clusterer for Credit Card Segmentation with "
            f"optimal feature(s): {kmeans_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump(kmeans_optimal_features, temp_file)
                temp_file_path = temp_file.name

            mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
            os.remove(temp_file_path)
            logger.info("Optimal features logged as artifact to MLflow.")
        except Exception as e:
            logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": kmeans_tuned_model,
        "Optimal Features": kmeans_optimal_features,
        "Silhouette Score": kmeans_silhouette_score,
    }


def train_kmeans_clusterer_with_bayes_search(
    X_train: pd.DataFrame,
    search_spaces: Dict[str, Any],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains a KMeans clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using Bayesian search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned KMeans model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"KMeans Clusterer Experiment Run (with Bayesian search) - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Credit Card KMeans Clusterer (with Bayesian search)"

    with mlflow.start_run(run_name=model_name) as run:
        kmeans_model = KMeans(random_state=42)

        (
            kmeans_fitted_model,
            kmeans_optimal_features,
            kmeans_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            kmeans_model,
            X_train,
            features,
        )

        kmeans_tuned_model = bayesian_hyperparameter_tuning(
            kmeans_fitted_model,
            search_spaces,
            X_train,
            kmeans_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=kmeans_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": kmeans_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"KMeans Clusterer (with Bayesian search) for Credit Card Segmentation with "
            f"optimal feature(s): {kmeans_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            pickle.dump(kmeans_optimal_features, temp_file)
            temp_file_path = temp_file.name

        mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
        os.remove(temp_file_path)
        logger.info("Optimal features logged as artifact to MLflow.")
    except Exception as e:
        logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": kmeans_tuned_model,
        "Optimal Features": kmeans_optimal_features,
        "Silhouette Score": kmeans_silhouette_score,
    }


def train_dbscan_clusterer_with_grid_search(
    X_train: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains a DBSCAN clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using grid search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    param_grid : Dict[str, List[Any]]
        Dictionary of hyperparameters to search through in grid search.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned DBSCAN model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"DBSCAN Clusterer Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Credit Card DBSCAN Clusterer"

    with mlflow.start_run(run_name=model_name) as run:
        dbscan_model = DBSCAN()

        (
            dbscan_fitted_model,
            dbscan_optimal_features,
            dbscan_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            dbscan_model,
            X_train,
            features,
        )

        dbscan_tuned_model = hyperparameter_tuning(
            dbscan_fitted_model,
            param_grid,
            X_train,
            dbscan_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=dbscan_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": dbscan_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"DBSCAN Clusterer for Credit Card Segmentation with "
            f"optimal features: {dbscan_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri, name=registered_model_name
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            pickle.dump(dbscan_optimal_features, temp_file)
            temp_file_path = temp_file.name

        mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
        os.remove(temp_file_path)
        logger.info("Optimal features logged as artifact to MLflow.")
    except Exception as e:
        logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": dbscan_tuned_model,
        "Optimal Features": dbscan_optimal_features,
        "Silhouette Score": dbscan_silhouette_score,
    }


def train_dbscan_clusterer_with_bayes_search(
    X_train: pd.DataFrame,
    search_spaces: Dict[str, Any],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains a DBSCAN clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using Bayesian search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned DBSCAN model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"DBSCAN Clusterer Experiment Run (with Bayesian search) - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Credit Card DBSCAN Clusterer (with Bayesian search)"

    with mlflow.start_run(run_name=model_name) as run:
        dbscan_model = DBSCAN()

        (
            dbscan_fitted_model,
            dbscan_optimal_features,
            dbscan_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            dbscan_model,
            X_train,
            features,
        )

        dbscan_tuned_model = bayesian_hyperparameter_tuning(
            dbscan_fitted_model,
            search_spaces,
            X_train,
            dbscan_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=dbscan_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": dbscan_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"DBSCAN Clusterer (with Bayesian search) for Credit Card Segmentation with "
            f"optimal features: {dbscan_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump(dbscan_optimal_features, temp_file)
                temp_file_path = temp_file.name

            mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
            os.remove(temp_file_path)
            logger.info("Optimal features logged as artifact to MLflow.")
        except Exception as e:
            logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": dbscan_tuned_model,
        "Optimal Features": dbscan_optimal_features,
        "Silhouette Score": dbscan_silhouette_score,
    }


def train_hierarchical_clusterer_with_grid_search(
    X_train: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains an Agglomerative Hierarchical clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using grid search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    param_grid : Dict[str, List[Any]]
        Dictionary of hyperparameters to search through in grid search.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned Agglomerative Hierarchical model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Agglomerative Hierarchical Clusterer Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Credit Card Agglomerative Hierarchical Clusterer"

    with mlflow.start_run(run_name=model_name) as run:
        hierarchical_model = AgglomerativeClustering()

        (
            hierarchical_fitted_model,
            hierarchical_optimal_features,
            hierarchical_silhouette_score,
        ) = recursive_cluster_feature_elimination(
            hierarchical_model,
            X_train,
            features,
        )

        hierarchical_tuned_model = hyperparameter_tuning(
            hierarchical_fitted_model,
            param_grid,
            X_train,
            hierarchical_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=hierarchical_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": hierarchical_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Agglomerative Hierarchical Clusterer for Credit Card Segmentation with "
            f"optimal features: {hierarchical_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump(hierarchical_optimal_features, temp_file)
                temp_file_path = temp_file.name

            mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
            os.remove(temp_file_path)
            logger.info("Optimal features logged as artifact to MLflow.")
        except Exception as e:
            logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": hierarchical_tuned_model,
        "Optimal Features": hierarchical_optimal_features,
        "Silhouette Score": hierarchical_silhouette_score,
    }


def train_hierarchical_clusterer_with_bayes_search(
    X_train: pd.DataFrame,
    search_spaces: Dict[str, Any],
    features: List[str],
) -> Dict[str, Any]:
    """
    Trains an Agglomerative Hierarchical clustering model, performs recursive feature elimination to find the optimal set of features, and tunes the model using Bayesian search for hyperparameter optimization.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training dataset containing all features.
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    features : List[str]
        The list of feature names to consider for recursive feature elimination.

    Returns:
    -------
    Dict[str, Any]
        A dictionary containing the tuned Agglomerative Hierarchical model, optimal features, and the best silhouette score.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Agglomerative Hierarchical Clusterer Experiment Run (with Bayesian search) - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = (
        "Credit Card Agglomerative Hierarchical Clusterer (with Bayesian search)"
    )

    with mlflow.start_run(run_name=model_name) as run:
        hierarchical_model = AgglomerativeClustering()

        (
            hierarchical_fitted_model,
            hierarchical_optimal_features,
            hierarchical_silhouette_score,
        ) = recursive_cluster_feature_elimination(hierarchical_model, X_train, features)

        hierarchical_tuned_model = bayesian_hyperparameter_tuning(
            hierarchical_fitted_model,
            search_spaces,
            X_train,
            hierarchical_optimal_features,
        )

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=hierarchical_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error logging model: {e}")

        try:
            mlflow.log_metrics({"Silhouette Score": hierarchical_silhouette_score})
        except Exception as e:
            print(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Agglomerative Hierarchical Clusterer (with Bayesian search) for Credit Card Segmentation with "
            f"optimal features: {hierarchical_optimal_features} after RCFE."
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump(hierarchical_optimal_features, temp_file)
                temp_file_path = temp_file.name

            mlflow.log_artifact(temp_file_path, artifact_path="optimal_features")
            os.remove(temp_file_path)
            logger.info("Optimal features logged as artifact to MLflow.")
        except Exception as e:
            logger.error(f"Error logging optimal features to MLflow: {e}")

    return {
        "Model": hierarchical_tuned_model,
        "Optimal Features": hierarchical_optimal_features,
        "Silhouette Score": hierarchical_silhouette_score,
    }


def load_model_from_mlflow(
    run_id: str,
    artifact_path: str,
) -> Tuple[ClusterMixin, Any]:
    """
    Loads a clustering model from MLflow by run_id and artifact_path.

    Parameters:
    ----------
    run_id : str
        The unique identifier for the MLflow run from which to load the model.
    artifact_path : str
        The path to the model artifact within the specified MLflow run.

    Returns:
    -------
    Tuple[ClusterMixin, Any]
        The loaded clustering model and the associated optimal features.
    """

    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.sklearn.load_model(model_uri)
    optimal_features_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="optimal_features",
    )
    with open(optimal_features_path, "rb") as f:
        optimal_features = pickle.load(f)

    return model, optimal_features


def scatter_plot(
    X_reduced: Union[np.ndarray, pd.DataFrame],
    clusters: Union[np.ndarray, List[int]],
    cluster_centers: Optional[np.ndarray] = None,
    title: str = "Scatter Plot",
    xlabel: str = "Index",
    ylabel: str = "Feature Value",
) -> None:
    """
    Generates a scatter plot to visualize clustering results in either 1D or 2D space.
    The function adapts based on the number of dimensions in the input data (X_reduced).

    Parameters:
    -----------
    X_reduced : numpy.ndarray or pandas.DataFrame
        The reduced input features after dimensionality reduction (e.g., PCA) or selected features.
        Should either be a 1D array (single feature) or a 2D array (two features).
        For 1D data, the points will be plotted against their index.
    clusters : array-like
        The cluster labels for each data point. Used to color the data points according to their cluster.
    cluster_centers : numpy.ndarray, optional
        The coordinates of the cluster centers if available. Applicable for algorithms like KMeans.
        For 1D, cluster centers are plotted on the x-axis index. For 2D, centers are plotted on both axes.
    title : str, optional
        The title of the plot. Default is "Scatter Plot".
    xlabel : str, optional
        Label for the x-axis. Default is "Index" for 1D plots.
    ylabel : str, optional
        Label for the y-axis. Default is "Feature Value" for 1D plots.
    """

    if isinstance(X_reduced, pd.DataFrame):
        X_reduced = X_reduced.values

    if X_reduced.ndim == 2 and X_reduced.shape[1] == 1:
        X_reduced = X_reduced.flatten()

    if X_reduced.ndim == 1:
        plt.scatter(
            range(len(X_reduced)),
            X_reduced,
            c=clusters,
            cmap="viridis",
            marker="o",
            label="Data Points",
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # For KMeans if applicable
        if cluster_centers is not None:
            plt.scatter(
                range(len(cluster_centers)),
                cluster_centers,
                s=300,
                c="red",
                marker="x",
                label="Cluster Centers",
            )

    elif X_reduced.ndim == 2 and X_reduced.shape[1] == 2:
        plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=clusters,
            cmap="viridis",
            marker="o",
            label="Data Points",
        )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        if cluster_centers is not None:
            plt.scatter(
                cluster_centers[:, 0],
                cluster_centers[:, 1],
                s=300,
                c="red",
                marker="x",
                label="Cluster Centers",
            )

    else:
        raise ValueError(
            "The input features must have either one or two dimensions for plotting"
        )

    plt.title(title)
    plt.colorbar(label="Cluster")
    plt.legend()
    plt.show()


def evaluate_clustering_models(
    training_clustering: dict,
    test_clustering: dict,
    X_train_dict: dict,
    X_test_dict: dict,
) -> pd.DataFrame:
    """
    Evaluates the performance of multiple clustering models using Silhouette Score, Davies-Bouldin Score,
    and Calinski-Harabasz Index, and stores the results in a DataFrame.

    Parameters:
    -----------
    training_clustering (dict)
        Dictionary containing training cluster labels for each model.
    test_clustering (dict)
        Dictionary containing test cluster labels for each model.
    X_train_dict (dict)
        Dictionary containing optimized training datasets for each model.
    X_test_dict (dict)
        Dictionary containing optimized test datasets for each model.

    Returns:
    -----------
    pd.DataFrame
        A DataFrame showing the evaluation metrics for each model.
    """

    results = []

    for model_name in training_clustering.keys():
        train_labels = training_clustering[model_name]
        X_train_optimal = X_train_dict[model_name]

        silhouette_train = silhouette_score(X_train_optimal, train_labels)
        davies_bouldin_train = davies_bouldin_score(X_train_optimal, train_labels)
        calinski_harabasz_train = calinski_harabasz_score(X_train_optimal, train_labels)

        test_labels = test_clustering[model_name]
        X_test_optimal = X_test_dict[model_name]

        silhouette_test = silhouette_score(X_test_optimal, test_labels)
        davies_bouldin_test = davies_bouldin_score(X_test_optimal, test_labels)
        calinski_harabasz_test = calinski_harabasz_score(X_test_optimal, test_labels)

        results.append(
            {
                "Model": model_name,
                "Silhouette Score (Train)": silhouette_train,
                "Silhouette Score (Test)": silhouette_test,
                "Davies-Bouldin Score (Train)": davies_bouldin_train,
                "Davies-Bouldin Score (Test)": davies_bouldin_test,
                "Calinski-Harabasz Score (Train)": calinski_harabasz_train,
                "Calinski-Harabasz Score (Test)": calinski_harabasz_test,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df
