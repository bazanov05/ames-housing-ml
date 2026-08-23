import numpy as np
import pandas as pd
from wrapped_models.xgboost_regressor import XGBoostRegressor, compute_baseline_prediction, compute_gradients
from wrapped_models.lightgbm_tree import LightGBMRegressionTree


class LightGBMRegressor(XGBoostRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_leaves=31,
                 max_depth=6, min_samples=10, reg_lambda=1.0, gamma=0.0):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples,
            reg_lambda=reg_lambda,
            gamma=gamma
        )
        self.max_leaves = max_leaves

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.y_mean = compute_baseline_prediction(y=y)
        predictions = np.full(shape=y.shape, fill_value=self.y_mean)

        for _ in range(self.n_estimators):
            gradients = compute_gradients(y_pred=predictions, y=y)

            tree = LightGBMRegressionTree(
                max_leaves=self.max_leaves,
                max_depth=self.max_depth,
                min_samples=self.min_samples_split,
                lam=self.reg_lambda,
                gamma=self.gamma
            )
            tree.fit(X=X, gradients=gradients)
            self.trees.append(tree)

            predictions = predictions + self.learning_rate * np.array(tree.predict(X))
        