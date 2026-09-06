import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from src.train import load_data, split_data_into_train_test


def load_model(filepath: str | Path = "src/models/pipeline.joblib") -> Pipeline:
    """
    Loads a serialized scikit-learn pipeline from disk.

    Args:
        filepath (str | Path, optional): Path to the saved .joblib model file.
            Defaults to "src/models/pipeline.joblib".

    Raises:
        FileNotFoundError: If the specified file does not exist.

    Returns:
        Pipeline: The deserialized, fitted scikit-learn Pipeline instance.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    model: Pipeline = joblib.load(filepath)
    print(f"Model successfully loaded from: {filepath}")
    return model


def analyze_residuals(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model: Pipeline, 
    plot_path: str | Path = "src/plots/residual_analysis.png"
) -> None:
    """
    Generates and saves a residual plot to evaluate model prediction errors.

    Computes prediction residuals (y_true - y_pred), plots them against
    the predicted values to inspect for heteroscedasticity or bias, and saves
    the resulting figure to disk.

    Args:
        X_test (pd.DataFrame): Raw evaluation feature matrix.
        y_test (np.ndarray): Ground truth target values.
        model (Pipeline): Fitted scikit-learn pipeline used for generating predictions.
        plot_path (str | Path, optional): Filepath where the plot image will be saved.
            Defaults to "src/plots/residual_analysis.png".
    """
    # predict and convert both true and predicted values back to dollars
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    
    # calculate residuals in dollars
    residuals = y_true - y_pred

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (True - Predicted)")
    plt.title("Residual Analysis")
    plt.axhline(y=0, color="red", linestyle="--")

    plt.savefig(plot_path)
    plt.close()


def inspect_worst_errors(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model: Pipeline,
    n_top: int = 10,
) -> pd.DataFrame:
    """Identifies test samples with the highest absolute prediction errors.

    Args:
        X_test (pd.DataFrame): Raw evaluation feature matrix.
        y_test (np.ndarray): Ground truth target values.
        model (Pipeline): Fitted scikit-learn pipeline used for inference.
        n_top (int, optional): Number of worst instances to return. Defaults to 10.

    Returns:
        pd.DataFrame: Top N samples sorted by absolute error descending, containing
            original features, predictions, true values, residuals, and absolute errors.
    """
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    
    abs_errors = np.abs(y_true - y_pred)

    df_errors = X_test.copy()
    df_errors["True_SalePrice"] = y_true
    df_errors["Predicted_SalePrice"] = y_pred
    df_errors["Residual"] = y_true - y_pred
    df_errors["Absolute_Error"] = abs_errors

    worst_samples = df_errors.sort_values(
        by="Absolute_Error", ascending=False
    ).head(n_top)

    return worst_samples


def get_most_important_features(
    X_test: pd.DataFrame, 
    model: Pipeline, 
    n_top: int = 10
) -> pd.DataFrame:
    """Extracts and displays feature importance gain scores from a CatBoost pipeline.

    Passes raw test features through the pipeline's preprocessing step to align
    column names with the trained model, extracts gain-based importance scores,
    and returns the highest-ranking features.

    Args:
        X_test (pd.DataFrame): Raw test features prior to pipeline transformation.
        model (Pipeline): Fitted pipeline containing 'preprocessing' and 'regressor' steps.
        n_top (int, optional): Number of top features to return. Defaults to 10.

    Returns:
        pd.DataFrame: Sorted DataFrame with 'Feature' and 'Gain' columns.
    """
    catboost_model = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessing"]
    feature_names = preprocessor.transform(X_test).columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Gain": catboost_model.get_feature_importance()
    }).sort_values(by="Gain", ascending=False)

    return importance_df.head(n_top)


if __name__ == "__main__":
    best_model = load_model()
    X, y = load_data()
    
    # we do not need train data for evaluation of model
    _, (X_test, y_test) = split_data_into_train_test(X, y)

    analyze_residuals(X_test, y_test, model=best_model)

    # fetch top 10 worst data rows based on abs residual (diff between pred and true value)
    worst_samples = inspect_worst_errors(X_test, y_test, best_model, n_top=10)

    # save worst cases to .csv file
    output_path = Path("src/reports/worst_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    worst_samples.to_csv(output_path, index=False)
    print(f"Full worst predictions report saved to: {output_path}")

    # print in console short summary 
    summary_cols = [
        "Neighborhood",
        "Overall Qual",
        "Gr Liv Area",
        "Sale Condition",
        "True_SalePrice",
        "Predicted_SalePrice",
        "Residual",
        "Absolute_Error",
    ]

    display_cols = [col for col in summary_cols if col in worst_samples.columns]
    print("\nTop 10 Worst Predictions Summary:")
    print(worst_samples[display_cols].to_string(index=False))

    # fetch top 10 most important features for our winner CatBoost
    top_features = get_most_important_features(X_test, best_model, n_top=10)

    # print these top 10 in console
    print("\nTop 10 Most Important Features (Gain):")
    print(top_features.to_string(index=False))

    # save full importance(across all features) in .csv file
    features_output_path = Path("src/reports/feature_importance.csv")
    features_output_path.parent.mkdir(parents=True, exist_ok=True)
    catboost_model = best_model.named_steps["regressor"]
    preprocessor = best_model.named_steps["preprocessing"]
    feature_names = preprocessor.transform(X_test).columns
    
    full_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Gain": catboost_model.get_feature_importance()
    }).sort_values(by="Gain", ascending=False)
    
    full_importance_df.to_csv(features_output_path, index=False)
    print(f"\nFull feature importance saved to: {features_output_path}")
