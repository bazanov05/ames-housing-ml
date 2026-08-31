import numpy as np
import pandas as pd


class CatBoostTree:
    def __init__(self, max_depth: int = 5, lam: float = 1.0, gamma: float = 0.0):
        self.max_depth = max_depth
        self.lam = lam      
        self.gamma = gamma 
        self.splits = []       # list of tuples: [(feature, threshold), ...]
        self.leaf_values = []  # flat list/array of 2^depth floats 
        self.partitions = None

    def fit(self, X: pd.DataFrame, gradients: np.ndarray) -> None:
        self._build_tree(X, gradients, list(X.columns), set())

    def predict(self, X: pd.DataFrame) -> list[float]:
        predictions = []

        for _, row in X.iterrows():
            prediction = self._predict_sample(sample=row)
            predictions.append(prediction)

        return predictions

    def _calculate_similarity_score(self, gradients: np.ndarray) -> float:
        return np.sum(gradients) ** 2 / (gradients.shape[0] + self.lam)

    def _calculate_gain(self, parent_ss: float, left_child_ss: float, right_child_ss: float) -> float:
        # gain must be > parent's similarity score + y
        return left_child_ss + right_child_ss - parent_ss - self.gamma

    def _best_split(
        self, 
        X: pd.DataFrame, 
        gradients: list[np.ndarray], 
        features: list[str], 
        used_features: set[str]
        ) -> tuple[str, str | int | float]:
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in features:
            is_numeric = pd.api.types.is_numeric_dtype(X[feature])

            # there is no point at reusing categorical feature since the data was already splitted based on it 
            if not is_numeric and feature in used_features:
                continue

            thresholds = X[feature].unique()
            thresholds.sort()

            # if have the numeric feature we should check the midpoints
            # midpoint = (x[i] + x[i - 1]) / 2
            if is_numeric:
                thresholds = thresholds[0:len(thresholds) - 1] + thresholds[1:]
                thresholds = thresholds / 2

            for threshold in thresholds:
                # for each threshold compute the sum of gains of every node in case we split by this threshold
                curr_total_gain = 0.0
                for i, idx in enumerate(self.partitions):
                    mask = X[feature].iloc[idx] <= threshold if is_numeric else X[feature].iloc[idx] == threshold
                    left_g = gradients[i][mask]
                    right_g = gradients[i][~mask]

                    gain = self._calculate_gain(
                        parent_ss=self._calculate_similarity_score(gradients=gradients[i]),
                        left_child_ss=self._calculate_similarity_score(gradients=left_g),
                        right_child_ss=self._calculate_similarity_score(gradients=right_g)
                    )
                    curr_total_gain += gain

                if curr_total_gain > best_gain:
                    best_gain = curr_total_gain
                    best_feature = feature
                    best_threshold = threshold
                    

        return best_feature, best_threshold

    def _leaf_values(self, gradients: list[np.ndarray]) -> list[float]:
        for grad in gradients:
            if len(grad) == 0:
                self.leaf_values.append(0.0)
            else:
                self.leaf_values.append(-1 * (np.sum(grad) / (grad.shape[0] + self.lam)))

    def _build_tree(
        self,
        X: pd.DataFrame,
        gradients: np.ndarray,
        features: list[str],
        used_features: set[str],
    ):
        # we start from 1x1 matrices since the root is not splitted yet
        gradients = [gradients]
        self.partitions = [np.arange(len(X))]
        for _ in range(self.max_depth):
            feature, threshold = self._best_split(X, gradients, features, used_features)

            # if no children ss > parent ss for any features - new level cannot be created
            if feature is None:
                self._leaf_values(gradients=gradients)
                return 

            is_numeric = pd.api.types.is_numeric_dtype(X[feature])
            if not is_numeric:
                used_features.add(feature)
            self.splits.append((feature, threshold))

            # we need to track gradients and indices for every node 
            # because we do not build tree recursively and work on the whole X all the time
            new_partitions = []
            new_gradients = []
            for i, idx in enumerate(self.partitions):
                mask = X[feature].iloc[idx] <= threshold if is_numeric else X[feature].iloc[idx] == threshold
                left_idx = idx[mask]
                right_idx = idx[~mask]
                new_partitions.append(left_idx)
                new_partitions.append(right_idx)

                left_grad = gradients[i][mask]
                right_grad = gradients[i][~mask]
                new_gradients.append(left_grad)
                new_gradients.append(right_grad)

            gradients = new_gradients
            self.partitions = new_partitions

        self._leaf_values(gradients=gradients)
        return

    def _predict_sample(self, sample: pd.Series) -> float:
        candidates = self.leaf_values

        for feature, threshold in self.splits:
            mid = len(candidates) // 2
            
            is_categorical = isinstance(threshold, str)

            on_the_left = sample[feature] == threshold if is_categorical else sample[feature] <= threshold

            if on_the_left:
                candidates = candidates[:mid]
            else:
                candidates = candidates[mid:]

        return candidates[0]
