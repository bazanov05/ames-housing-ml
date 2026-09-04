import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from collections.abc import Callable
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.pipeline import build_linear_pipeline, build_catboost_pipeline, build_tree_pipeline
from src.config import create_dict_of_param_distr, make_narrow_grid
from src.preprocessing import AmesPreprocessor


def load_data(path: str = "data/raw/AmesHousing.csv") -> tuple[pd.DataFrame, np.ndarray]:
    try:
        df = pd.read_csv(filepath_or_buffer=path)
        y = df["SalePrice"].to_numpy()
        X = df.drop(columns=["SalePrice"])

        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def save_model(model: Pipeline, filepath: str | Path) -> None:
    """Saves a fitted scikit-learn model/pipeline to disk using joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # create a parent dir if it did not exist
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def get_models(X_train: pd.DataFrame) -> dict:
    # run preprocessor once to see the exact columns reaching CatBoost
    X_preprocessed = AmesPreprocessor().fit_transform(X_train)
    categorical_features = X_preprocessed.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    models = {
        "CatBoost Regressor": (
            CatBoostRegressor(
                cat_features=tuple(categorical_features), 
                verbose=False,
                thread_count=1
            ), 
            build_catboost_pipeline
        ),
        "Linear Regression": (LinearRegression(), build_linear_pipeline),
        "XGBoost Regressor": (XGBRegressor(), build_tree_pipeline),
        "LGBM Regressor": (LGBMRegressor(n_jobs=1), build_tree_pipeline)
    }

    return models


def find_best_pipeline_per_model(models: dict, X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, tuple[Pipeline, float]]:
    """
    Tunes and evaluates each model pipeline using a two-stage hyperparameter search.

    For Linear Regression, which has no hyperparameters to tune, the pipeline is evaluated
    directly using 5-fold cross-validation and stored with its mean CV RMSE score.

    For all other models (XGBoost, LightGBM, CatBoost), a two-stage search is performed:
        1. RandomizedSearchCV samples 50 random hyperparameter combinations to identify
           a promising region of the search space.
        2. GridSearchCV then performs a focused search around the best parameters found
           in stage one, using narrow intervals centered on those values.

    The best fitted pipeline per model is stored alongside its best CV RMSE score.

    Args:
        models (dict): Mapping of model name to a tuple of (estimator, pipeline_builder_function),
                       as returned by get_models().
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training target values.

    Returns:
        dict[str, tuple[Pipeline, float]]: Mapping of model name to a tuple of
            (best fitted pipeline, best negative CV RMSE score).
    """
    best_pipelines = dict()
    param_distributions = create_dict_of_param_distr()

    for name, model_params in models.items():
        model, build_method = model_params
        pipeline = build_method(model)
        parameters = param_distributions.get(name)

        # Linear Regression does no have parameters to tune
        # just fit it on train data and push the pipeline with mean rmse from 5 folds
        if not parameters:
            rmse_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
            best_pipelines[name] = (pipeline, rmse_scores.mean())
            pipeline.fit(X_train, y_train)

        # for XGBoost, LGBM and CatBoost tune the parameters
        else:
            # first - shuffle 50 random combinations and find the best among them
            random_cv = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameters,
                n_iter=50,      # sample 50 random combinations
                scoring="neg_root_mean_squared_error",
                cv=5,           # 5 folds - 4 to train, 1 to validate
                n_jobs=4,       # use 4 CPU cores
                random_state=42
            )

            random_cv.fit(X=X_train, y=y_train)

            # after best among random was found - find the best config of these parameters
            grid_cv = GridSearchCV(
                estimator=pipeline,
                param_grid=make_narrow_grid(random_cv.best_params_),
                scoring="neg_root_mean_squared_error",  # use negative since sklearn's logic is the bigger the better
                cv=5,
                n_jobs=4
            )

            grid_cv.fit(X=X_train, y=y_train)

            # save the best pipeline
            best_pipelines[name] = (grid_cv.best_estimator_, grid_cv.best_score_)

    return best_pipelines


def split_data_into_train_test(X: pd.DataFrame, y: np.ndarray) -> list[tuple]:
    # split data into: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return [(X_train, y_train), (X_test, y_test)]


if __name__ == "__main__":
    X, y = load_data()

    (X_train, y_train), (X_test, y_test) = split_data_into_train_test(X, y)

    # dict of: model_name: (class, method_to_build_pipeline)
    models: dict[str, tuple[Callable, Callable]] = get_models(X_train=X_train)
    best_pipelines: dict[str, tuple[Pipeline, float]] = find_best_pipeline_per_model(models=models, X_train=X_train, y_train=y_train)

    sorted_pipelines = sorted(
        best_pipelines.items(),
        key=lambda item: item[1][1],
        reverse=True
    )

    model_name, (best_pipeline, best_score) = sorted_pipelines[0]
    print(f"Overall Best Model: {model_name} (CV RMSE: {abs(best_score):.2f})\n\n")

    # iterate through best pipelines and compare results
    for name, (pipeline, cv_score) in sorted_pipelines:
        y_pred = pipeline.predict(X_test)
        test_rmse = root_mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        print(f"Model: {name}")
        print(f"CV RMSE:   {abs(cv_score):.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Test MAE:  {test_mae:.2f}")
        print(f"Test R2:   {test_r2:.4f}\n")

    save_model(model=best_pipeline, filepath="src/models/pipeline.joblib")