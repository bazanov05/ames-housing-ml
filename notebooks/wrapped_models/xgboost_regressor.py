import numpy as np
import pandas as pd
from wrapped_models.xg_boost_regression_tree import XG_Boost_Regressor_Tree


def compute_baseline_prediction(y: np.ndarray) -> float:
    return np.mean(y)


def compute_gradients(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y_pred - y


class XGBoostRegressor:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.y_mean = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        # start with the baseline for every sample
        self.y_mean = compute_baseline_prediction(y=y)

        predictions = np.full(shape=y.shape, fill_value=self.y_mean)

        for _ in range(self.n_estimators):
            gradients = compute_gradients(y_pred=predictions, y=y)

            new_tree = XG_Boost_Regressor_Tree(
                max_depth=self.max_depth,
                min_samples=self.min_samples_split,
                lam=self.reg_lambda,
                gamma=self.gamma
            )
            
            new_tree.fit(X=X, gradients=gradients)  # build a new tree, find best splits

            self.trees.append(new_tree)

            # update predictions based on the new residuals
            residuals = np.array(new_tree.predict(X))
            predictions = predictions + self.learning_rate * residuals


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # start from initial baseline
        predictions = np.full(shape=X.shape[0], fill_value=self.y_mean)
        
        for tree in self.trees:
            # predict() returns list of floats, not np.array, so convert it 
            predictions += self.learning_rate * np.array(tree.predict(X))
        
        return predictions
