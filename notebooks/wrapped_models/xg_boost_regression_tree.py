import numpy as np
import pandas as pd


class XG_Boost_Regressor_Tree:
    def __init__(self, max_depth: int = 5, min_samples: int = 10, lam: float = 1.0, gamma: float = 0.0):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
        self.lam = lam      
        self.gamma = gamma  

    def fit(self, X: pd.DataFrame, gradients: np.ndarray) -> None:
        self.tree = self._build_tree(X, gradients, list(X.columns), set())

    def predict(self, X: pd.DataFrame) -> list[float]:
        predictions = []

        for _, row in X.iterrows():
            prediction = self._predict_sample(row, node=self.tree)
            predictions.append(prediction)

        return predictions

    def _calculate_similarity_score(self, gradients: np.ndarray):
        # since the hessian = 2 and it cancels out with the 2 in 2(p - y)
        # ss is just the mean residual - it is the main diff from the classifier
        # in classifier hessian was responsible for the step size - how confident is model about the prediction
        return np.sum(gradients) ** 2 / (gradients.shape[0] + self.lam)

    def _calculate_gain(self, parent_ss: float, left_child_ss: float, right_child_ss: float) -> float:
        # gain must be > parent's similarity score + y
        return left_child_ss + right_child_ss - parent_ss - self.gamma

    def _best_split(
        self, 
        X: pd.DataFrame, 
        gradients: np.ndarray, 
        features: list[str], 
        used_features: set[str]
        ) -> tuple[str, str | int | float]:
        parent_ss = self._calculate_similarity_score(gradients=gradients)
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
                # apply mask based on the feature's type
                mask = X[feature] <= threshold if is_numeric else X[feature] == threshold
                left_g, right_g = gradients[mask], gradients[~mask]

                # prevent empty split
                if len(left_g) == 0 or len(right_g) == 0:
                    continue

                # gain = left_ss + right_ss - parent_ss
                gain = self._calculate_gain(
                    parent_ss,
                    left_child_ss=self._calculate_similarity_score(gradients=left_g),
                    right_child_ss=self._calculate_similarity_score(gradients=right_g)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _leaf_value(self, gradients: np.ndarray) -> float:
        # The vertex of the parabola for which the loss is minimized, 
        # where the loss is approximated as a quadratic function using a Taylor series
        # vertex = - b / 2 * a, b = 2(p - y), a = h
        # so vertex is just mean residual with the minus sign
        return  -1 * (np.sum(gradients) / (gradients.shape[0] + self.lam))

    def _build_tree(
        self,
        X: pd.DataFrame,
        gradients: np.ndarray,
        features: list[str],
        used_features: set[str],
        depth: int = 0
    ):
        if depth == self.max_depth or len(gradients) < self.min_samples:
            return {"leaf": True, "prediction": self._leaf_value(gradients=gradients)}

        feature, threshold = self._best_split(X, gradients, features, used_features)

        # if no better feature than parent was found - there is no better split for that node
        if feature is None:
            return {"leaf": True, "prediction": self._leaf_value(gradients=gradients)}

        is_categorical = not pd.api.types.is_numeric_dtype(X[feature])

        if is_categorical:
            mask = X[feature] == threshold
            used_features = used_features.copy()
            used_features.add(feature)
        else:
            mask = X[feature] <= threshold

        left_X, right_X = X[mask], X[~mask]
        left_g, right_g = gradients[mask], gradients[~mask]

        if len(left_X) == 0 or len(right_X) == 0:
            return {"leaf": True, "prediction": self._leaf_value(gradients=gradients)}

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(left_X, left_g, features, used_features, depth + 1),
            "right": self._build_tree(right_X, right_g, features, used_features, depth + 1),
        }

    def _predict_sample(self, sample: pd.Series, node) -> float:
        if node["leaf"]:
            return node["prediction"]

        feature, threshold = node["feature"], node["threshold"]
        if isinstance(threshold, str):
            child = node["left"] if sample[feature] == threshold else node["right"]
        else:
            child = node["left"] if sample[feature] <= threshold else node["right"]
        return self._predict_sample(sample, child)
