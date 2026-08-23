import numpy as np
import pandas as pd
from heapq import heappop, heappush
from dataclasses import dataclass
from itertools import count


@dataclass(frozen=False)
class Node:
    is_leaf: bool = True
    prediction: float = None        # leaf only
    feature: str = None             # internal only
    threshold: float | str = None   # internal only
    left: "Node" = None             # internal only
    right: "Node" = None            # internal only

    # building metadata (temporary)
    X: pd.DataFrame = None
    gradients: np.ndarray = None
    depth: int = 0
    used_features: set = None


class LightGBMRegressionTree:
    def __init__(self, max_leaves: int = 31, max_depth: int = 5, min_samples: int = 10, lam: float = 1.0, gamma: float = 0.0):
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
        self.lam = lam      
        self.gamma = gamma  

    def fit(self, X: pd.DataFrame, gradients: np.ndarray) -> None:
        self._build_tree(X, gradients, list(X.columns))
    
    def predict(self, X: pd.DataFrame) -> list[float]:
        predictions = []

        for _, row in X.iterrows():
            prediction = self._predict_sample(row, node=self.tree)
            predictions.append(prediction)

        return predictions

    def _calculate_similarity_score(self, gradients: np.ndarray):
        # the SS in LightGBM is the same like in XGBoost 
        return np.sum(gradients) ** 2 / (gradients.shape[0] + self.lam)

    def _calculate_gain(self, parent_ss: float, left_child_ss: float, right_child_ss: float) -> float:
        # gain must be bigger than gamma
        return left_child_ss + right_child_ss - parent_ss - self.gamma

    def _best_split(
        self, 
        X: pd.DataFrame, 
        gradients: np.ndarray, 
        features: list[str], 
        used_features: set[str]
        ) -> tuple[str, str | int | float, float]:
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

        return best_feature, best_threshold, best_gain

    def _leaf_value(self, gradients: np.ndarray) -> float:
        return  -1 * (np.sum(gradients) / (gradients.shape[0] + self.lam))

    def _build_tree(
        self,
        X: pd.DataFrame,
        gradients: np.ndarray,
        features: list[str]
    ):
        used_features = set()
        counter = count()
        root_feature, root_threshold, root_gain = self._best_split(X, gradients, features, used_features)
        root_prediction = self._leaf_value(gradients=gradients)

        is_categorical = not pd.api.types.is_numeric_dtype(X[root_feature])

        if is_categorical:
            used_features.add(root_feature)

        root_node = Node(
            prediction=root_prediction,
            feature=root_feature,
            threshold=root_threshold,
            X=X,
            gradients=gradients,
            used_features=used_features
        )

        # start with root_node, every node tracks it's left and right child if it's not leaf
        self.tree = root_node
        leaves = [(-root_gain, next(counter), root_node)]
        num_of_leaves = 1

        while leaves and num_of_leaves < self.max_leaves:
            _, _, leaf = heappop(leaves)    # do not need gain and counter for split logic

            # if we reached the max depth or curr leaf does not have feature to split - skip it
            curr_depth = leaf.depth + 1
            if curr_depth > self.max_depth or leaf.feature is None:
                continue
        
            is_categorical = not pd.api.types.is_numeric_dtype(X[leaf.feature])

            mask = leaf.X[leaf.feature] == leaf.threshold if is_categorical else leaf.X[leaf.feature] <= leaf.threshold
            left_g, right_g = leaf.gradients[mask], leaf.gradients[~mask]
            left_X, right_X = leaf.X[mask], leaf.X[~mask]

            # if some of the children after the split contains less than min samples - skip that leaf
            if len(left_g) < self.min_samples or len(right_g) < self.min_samples:
                continue

            # find the best split for each of children
            l_feature, l_threshold, l_gain = self._best_split(left_X, left_g, features, leaf.used_features)
            r_feature, r_threshold, r_gain = self._best_split(right_X, right_g, features, leaf.used_features)

            # gain is 1 since we turn one leaf into the node with two leaf children
            leaf.is_leaf = False
            num_of_leaves += 1

            l_used_features = leaf.used_features.copy()
            if l_feature is not None:
                if not pd.api.types.is_numeric_dtype(X[l_feature]):
                    l_used_features.add(l_feature)
            
            left_node = Node(
                prediction=self._leaf_value(gradients=left_g),
                feature=l_feature,
                threshold=l_threshold,
                gradients=left_g,
                X=left_X,
                depth=curr_depth,
                used_features=l_used_features
            )

            r_used_features = leaf.used_features.copy()
            if r_feature is not None:
                if not pd.api.types.is_numeric_dtype(X[r_feature]):
                    r_used_features.add(r_feature)
                        
            right_node = Node(
                prediction=self._leaf_value(gradients=right_g),
                feature=r_feature,
                threshold=r_threshold,
                gradients=right_g,
                X=right_X,
                depth=curr_depth,
                used_features=r_used_features
            )

            leaf.left = left_node
            leaf.right = right_node

            # use counter since we do not provide __lt__ for Node class
            # heap compares gain first, if it is equal - it compares counter
            heappush(leaves, (-l_gain, next(counter), left_node))
            heappush(leaves, (-r_gain, next(counter), right_node))

    def _predict_sample(self, sample: pd.Series, node: Node) -> float:
        if node.is_leaf:
            return node.prediction

        feature, threshold = node.feature, node.threshold
        if isinstance(threshold, str):
            child = node.left if sample[feature] == threshold else node.right
        else:
            child = node.left if sample[feature] <= threshold else node.right
        return self._predict_sample(sample, child)
    