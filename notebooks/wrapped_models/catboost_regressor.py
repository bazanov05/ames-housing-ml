import numpy as np
import pandas as pd
from collections import defaultdict

from wrapped_models.catboost_tree import CatBoostTree
from wrapped_models.xgboost_regressor import compute_baseline_prediction, compute_gradients


class CatBoostRegressor:
    def __init__(
            self, 
            n_estimators=10, 
            learning_rate=0.1, 
            max_depth=3, 
            reg_lambda=1.0, 
            gamma=0.0,
            alpha=1.0,
            n_permutations=4
        ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.trees = []
        self.y_mean = None
        self.permutations_of_X = []

    def _fetch_categorical_features(self, X: pd.DataFrame) -> list[str]:
        categorical_features = []
        for feature in X.columns:
            if not pd.api.types.is_numeric_dtype(X[feature]):
                categorical_features.append(feature)

        return categorical_features

    def _get_target_encoding_per_permutation(
            self,
            X: pd.DataFrame,
            y: np.ndarray,
            categorical_features: list[str],
            permutations: list[list[int]],
            global_mean: float
    ) -> None:
        for p in permutations:
            running_sum = defaultdict(dict)     # feature: threshold: (running sum, count)
            X_shuffled = X.iloc[p].copy().reset_index(drop=True)
            y_shuffled = y[p]

            for i, row in X_shuffled.iterrows():
                for feature in categorical_features:
                    threshold = row[feature]
                    # if threshold for curr category is met for the 1st time - assign the global mean to it
                    if threshold not in running_sum[feature]:
                        X_shuffled.at[i, feature] = global_mean
                        running_sum[feature][threshold] = (y_shuffled[i], 1)
                    else:
                        curr_sum, curr_count = running_sum[feature][threshold]
                        # the core idea is to calculate the mean for that threshold without curr value
                        # prevents data leakage
                        X_shuffled.at[i, feature] = (curr_sum + self.alpha * global_mean) / (curr_count + self.alpha)

                        # update running sum and count and store for curr feature/threshold pair
                        curr_sum += y_shuffled[i]
                        curr_count += 1
                        running_sum[feature][threshold] = (curr_sum, curr_count)

            self.permutations_of_X.append(X_shuffled)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        # start with the baseline for every sample
        self.y_mean = compute_baseline_prediction(y=y)
        predictions = np.full(shape=y.shape, fill_value=self.y_mean)

        categorical_features = self._fetch_categorical_features(X=X)
        self.categorical_features = categorical_features

        # this dict will be used in predict() method
        # we will replace very categorical threshold with the mean
        # we do not use ordered target encoding in predict() since there is no risk of data leakage
        self.category_means = {}
        for col in categorical_features:
            self.category_means[col] = {}
            for cat in X[col].unique():
                # smoothed global target mean for category
                cat_y = y[X[col] == cat]
                self.category_means[col][cat] = (np.sum(cat_y) + self.alpha * self.y_mean) / (len(cat_y) + self.alpha)

        permutations = []

        for _ in range(self.n_permutations):
            permutations.append(np.random.permutation(len(X)))

        self._get_target_encoding_per_permutation(
            X=X,
            y=y,
            categorical_features=categorical_features,
            permutations=permutations,
            global_mean=self.y_mean
        )

        num_of_tree = 0

        for _ in range(self.n_estimators):
            i = num_of_tree % 4
            gradients = compute_gradients(y_pred=predictions, y=y)

            new_tree = CatBoostTree(
                max_depth=self.max_depth,
                lam=self.reg_lambda,
                gamma=self.gamma
            )
            
            new_tree.fit(X=self.permutations_of_X[i], gradients=gradients[permutations[i]])
            self.trees.append(new_tree)

            permuted_residuals = np.array(new_tree.predict(X=self.permutations_of_X[i]))
            # create an empty array to hold the un-permuted residuals
            residuals = np.zeros(len(X))
            
            # place them back in their original index positions
            residuals[permutations[i]] = permuted_residuals
            
            # update the global predictions
            predictions = predictions + self.learning_rate * residuals
            num_of_tree += 1


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # start from initial baseline
        predictions = np.full(shape=X.shape[0], fill_value=self.y_mean)

        X_encoded = X.copy()
        
        for feature in self.categorical_features:
            # replace categorical thresholds with the means per threshold
            # which were computed on training data
            # global mean as a fallback if some threshold was not found
            X_encoded[feature] = X_encoded[feature].map(self.category_means[feature]).fillna(self.y_mean)
    
        for tree in self.trees:
            # predict() returns list of floats, not np.array, so convert it 
            predictions += self.learning_rate * np.array(tree.predict(X_encoded))
        
        return predictions
