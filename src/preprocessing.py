import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AmesPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible transformer for Ames Housing dataset preprocessing.
    
    Learns statistics from training data in fit() and applies all data cleaning,
    missing value imputation, and feature engineering transformations in transform(). 
    Designed to be used as a step inside a sklearn Pipeline.
    
    Args:
        _mode_for_electrical (str): Most frequent electrical system type learned from training data.
        _neighborhood_medians (pd.Series): Median Lot Frontage per Neighborhood learned from training data.
        _global_lot_frontage_median (float): Global median Lot Frontage, used as fallback for imputation.
    """
    def __init__(self):
        super().__init__()
        self._mode_for_electrical: str = None
        self._neighborhood_medians: pd.Series = None
        self._global_lot_frontage_median: float = None

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """
        Learn statistics from training data needed for transformation.
        Nothing is applied to the data here — only learned and remembered on self.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (np.ndarray, optional): Target values, not used but accepted for Pipeline compatibility.

        Returns:
            self: Returns the instance itself for method chaining.
        """
        self._mode_for_electrical = X["Electrical"].mode()[0]
        self._neighborhood_medians = X.groupby("Neighborhood")["Lot Frontage"].median()
        self._global_lot_frontage_median = X["Lot Frontage"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning, imputation, and feature engineering transformations to the data.
        Uses statistics learned during fit() — never recomputes anything from X.

        Transformations applied in order:
            - Drop high-missing columns and redundant collinear features
            - Fill missing categorical values for garage and basement features with "None"
            - Impute missing Electrical values with the mode learned in fit()
            - Fill missing numerical values for garage and basement features with 0
            - Impute missing Lot Frontage hierarchically:
                1. Neighborhood median Lot Frontage
                2. Global median Lot Frontage (fallback)
            - Create binary indicator flags (HasBasement, HasGarage, Has2ndFloor, IsRemodeled)
            - Calculate structural ages (HouseAge, RemodelAge) and TotalSF surface area
            - Drop redundant raw date columns (Year Built, Garage Yr Blt)

        Args:
            X (pd.DataFrame): Feature matrix to transform, train or test.

        Returns:
            pd.DataFrame: Transformed DataFrame ready for model input.
        """
        X = X.copy()

        # feature drops
        drop_cols = [
            "Pool QC",
            "Alley",
            "Fence",
            "Misc Feature",
            "Fireplace Qu",
            "Garage Cars",
            "TotRms AbvGrd",
            "1st Flr SF",
        ]
        X = X.drop(columns=drop_cols)

        # categorical imputation (fillna 'None')
        cat_none_cols = [
            "Garage Type",
            "Garage Finish",
            "Garage Qual",
            "Garage Cond",
            "Bsmt Qual",
            "Bsmt Cond",
            "Bsmt Exposure",
            "BsmtFin Type 1",
            "BsmtFin Type 2",
            "Mas Vnr Type",
        ]
        X[cat_none_cols] = X[cat_none_cols].fillna("None")

        # categorical Imputation (Mode)
        X["Electrical"] = X["Electrical"].fillna(self._mode_for_electrical)

        # numerical Imputation (fillna 0)
        num_zero_cols = [
            "Garage Area",
            "BsmtFin SF 1",
            "BsmtFin SF 2",
            "Bsmt Unf SF",
            "Bsmt Full Bath",
            "Bsmt Half Bath",
            "Total Bsmt SF",
            "Mas Vnr Area",
        ]
        X[num_zero_cols] = X[num_zero_cols].fillna(0)

        # Lot Frontage Imputation (Neighborhood Median -> Fallback Global Median)
        group_impute = X["Neighborhood"].map(self._neighborhood_medians)
        X["Lot Frontage"] = (
            X["Lot Frontage"]
            .fillna(group_impute)
            .fillna(self._global_lot_frontage_median)
        )

        # Feature Engineering - Binary Flags
        X["HasBasement"] = (X["Total Bsmt SF"] > 0).astype(int)
        X["HasGarage"] = (X["Garage Area"] > 0).astype(int)
        X["Has2ndFloor"] = (X["2nd Flr SF"] > 0).astype(int)
        X["IsRemodeled"] = (X["Year Remod/Add"] != X["Year Built"]).astype(int)

        # Feature Engineering - Ages & Total Surfaces
        X["HouseAge"] = (X["Yr Sold"] - X["Year Built"]).clip(lower=0)
        X["RemodelAge"] = (X["Yr Sold"] - X["Year Remod/Add"]).clip(lower=0)
        X["TotalSF"] = X["Gr Liv Area"] + X["Total Bsmt SF"]

        # Drop Redundant Raw Date Columns
        X = X.drop(columns=["Year Built", "Garage Yr Blt"])

        return X