import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.preprocessing import AmesPreprocessor


ORDINAL_COLS = [
    "Kitchen Qual",
    "Bsmt Qual",
    "Bsmt Cond",
    "Exter Qual",
    "Garage Qual",
    "Garage Cond",
    "Garage Finish",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
]

ORDINAL_CATEGORIES = [
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # Kitchen Qual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # Bsmt Qual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # Bsmt Cond
    ["Po", "Fa", "TA", "Gd", "Ex"],          # Exter Qual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # Garage Qual
    ["None", "Po", "Fa", "TA", "Gd", "Ex"],  # Garage Cond
    ["None", "Unf", "RFn", "Fin"],           # Garage Finish
    ["None", "No", "Mn", "Av", "Gd"],        # Bsmt Exposure
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFin Type 1
    ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],  # BsmtFin Type 2
]

# encode Neighborhood with target encoder since there is no logical ordinary logic between them
# Neighborhood will be encoded with the mean for this neighborhood
# but from other folds to prevent this row influence it's own encoding
TARGET_ENCODE_COLS = ["Neighborhood"]

ONE_HOT_COLS = [
    "House Style",
    "Garage Type",
    "Mas Vnr Type",
    "Electrical",
]


def build_linear_pipeline(model: LinearRegression) -> Pipeline:
    """
    Constructs a scikit-learn pipeline tailored for linear models.
    Applies base preprocessing, comprehensive categorical encoding (One-Hot, Target, Ordinal), 
    StandardScaler for numerical features, and LassoCV for feature selection.

    Args:
        model (LinearRegression): The linear regression estimator to train.

    Returns:
        Pipeline: The assembled pipeline ready for fitting.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), make_column_selector(dtype_include=np.number)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT_COLS),
            ("target", TargetEncoder(), TARGET_ENCODE_COLS),
            ("ordinal", OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_COLS)
        ]
    )
    
    # create a full pipeline:
    # clean data and add features -> encode and scale -> drop noisy features -> pass to model
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessing", AmesPreprocessor()),
            ("transformer", column_transformer),
            ("feature_selection", SelectFromModel(LassoCV(cv=5, max_iter=10000))),
            ("regressor", model)
        ]
    )

    return linear_pipeline


def build_tree_pipeline(model: LGBMRegressor | XGBRegressor) -> Pipeline:
    """
    Constructs a scikit-learn pipeline for standard tree-based models (XGBoost, LightGBM).
    Applies base preprocessing and categorical encoding, but explicitly bypasses 
    scaling and external feature selection.

    Args:
        model (LGBMRegressor | XGBRegressor): The tree-based estimator to train.

    Returns:
        Pipeline: The assembled pipeline ready for fitting.
    """
    # trees do not require scaling since we do not use Lasso to drop features
    column_transformer = ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT_COLS),
            ("target", TargetEncoder(), TARGET_ENCODE_COLS),
            ("ordinal", OrdinalEncoder(categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_COLS)
        ],
        remainder="passthrough"     # crucial: keeps numerical cols
    )
    
    tree_pipeline = Pipeline(
        steps=[
            ("preprocessing", AmesPreprocessor()),
            ("transformer", column_transformer),
            ("regressor", model)
        ]
    )

    return tree_pipeline


def build_catboost_pipeline(model: CatBoostRegressor) -> Pipeline:
    """
    Constructs a minimal scikit-learn pipeline for CatBoost.
    Applies only base preprocessing and passes raw categorical data directly 
    to the estimator to leverage CatBoost's internal ordered target encoding.

    Args:
        model (CatBoostRegressor): The CatBoost estimator (must be initialized with cat_features).

    Returns:
        Pipeline: The assembled pipeline ready for fitting.
    """
    # we do not need column transformer for CatBoost for 2 reasons
    # we do not use Lasso - we do not scale numeric features
    # CatBoost encodes on the fly with Ordered Target Encoding - we do not need Encoders for data
    catboost_pipeline = Pipeline(
        steps=[
            ("preprocessing", AmesPreprocessor()),
            ("regressor", model)
        ]
    )

    return catboost_pipeline
