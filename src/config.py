from scipy.stats import loguniform, randint


def create_dict_of_param_distr() -> dict[str, dict]:
    """Generates hyperparameter search distributions for RandomizedSearchCV across supported models.

    Maps each model name to a dictionary of statistical distributions or empty spaces.
    All parameter names include the 'regressor__' prefix to directly target the pipeline estimator.

    Returns:
        dict[str, dict]: Mapping of model names to their respective hyperparameter search spaces.
    """
    # Linear Regression has no hyperparameters to optimize
    LINEAR_SEARCH_SPACE: dict = {}


    # controls tree depth, learning rate, number of trees
    XGB_SEARCH_SPACE: dict = {
        "regressor__n_estimators": randint(100, 1000),
        "regressor__max_depth": randint(3, 10),
        "regressor__learning_rate": loguniform(1e-3, 3e-1),
    }

    # differs from XGB one - num of leaves are more important than max depth
    LGBM_SEARCH_SPACE: dict = {
        "regressor__n_estimators": randint(100, 1000),
        "regressor__num_leaves": randint(15, 128),
        "regressor__max_depth": randint(3, 12),
        "regressor__learning_rate": loguniform(1e-3, 3e-1),
    }


    CATBOOST_SEARCH_SPACE: dict = {
        "regressor__iterations": randint(200, 1000),
        "regressor__depth": randint(4, 10),
        "regressor__learning_rate": loguniform(1e-3, 2e-1),
    }


    # master dictionary mapping model names to their search spaces
    SEARCH_SPACES: dict[str, dict] = {
        "Linear Regression": LINEAR_SEARCH_SPACE,
        "XGBoost Regressor": XGB_SEARCH_SPACE,
        "LGBM Regressor": LGBM_SEARCH_SPACE,
        "CatBoost Regressor": CATBOOST_SEARCH_SPACE,
    }

    return SEARCH_SPACES


def make_narrow_grid(best_params: dict) -> dict:
    """Constructs a localized parameter grid around the best parameters found by RandomizedSearchCV.

    Generates adjacent integer values for tree counts and depths, scales continuous rates
    multiplicatively, and preserves existing pipeline prefixes.

    Args:
        best_params (dict): Optimal hyperparameter key-value pairs from a fitted search object.

    Returns:
        dict: Parameter grid formatted for GridSearchCV exploration.
    """
    grid = {}
    for param, val in best_params.items():
        if "max_depth" in param:
            # depth +- 1
            grid[param] = [max(2, val - 1), val, val + 1]
        elif "n_estimators" in param or "iterations" in param:
            # num of trees +- 50
            grid[param] = [max(50, val - 50), val, val + 50]
        elif "num_leaves" in param:
            # num of leaves +- 15
            grid[param] = [max(10, val - 15), val, val + 15]
        elif "learning_rate" in param:
            grid[param] = [round(val * 0.7, 4), round(val, 4), round(val * 1.3, 4)]
        else:
            # keep categorical or other parameters fixed at their best value
            grid[param] = [val]
    return grid