import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from src.preprocessing import AmesPreprocessor


def build_pipeline(model) -> Pipeline:
    """
    Construct a full end-to-end scikit-learn Pipeline for the Ames Housing dataset.

    Combines custom cleaning and feature engineering (AmesPreprocessor), 
    feature encoding and scaling (ColumnTransformer), and the specified estimator.

    Args:
        model: A scikit-learn compatible regressor estimator.

    Returns:
        Pipeline: Assembled Pipeline ready for fit and transform operations.
    """
    ordinal_cols = [
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

    ordinal_categories = [
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
    target_encode_cols = ["Neighborhood"]

    one_hot_cols = [
        "House Style",
        "Garage Type",
        "Mas Vnr Type",
        "Electrical",
    ]

    column_transformer = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), make_column_selector(dtype_include=np.number)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), one_hot_cols),
            ("target", TargetEncoder(), target_encode_cols),
            ("ordinal", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols)
        ]
    )

    # create a full pipeline:
    # clean data and add features -> encode and scale -> pass to model
    pipeline = Pipeline(
        steps=[
            ("preprocessing", AmesPreprocessor()),
            ("transformer", column_transformer),
            ("regressor", model)
        ]
    )

    return pipeline
