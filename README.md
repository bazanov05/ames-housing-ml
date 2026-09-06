# Ames Housing ML

An end-to-end machine learning project predicting residential house sale prices using the Ames Housing dataset. This repository bridges algorithmic theory and production engineering by implementing core regression models from scratch in NumPy, followed by a highly optimized, leak-free Scikit-Learn production pipeline.

---

## Project Structure

```plaintext
ames-housing-ml/
├── .gitignore
├── Dockerfile
├── requirements.txt
├── eda/
│   └── eda.ipynb                        # Missing value analysis and preprocessing decisions
├── notebooks/
│   ├── 01_linear_regression.ipynb
│   ├── 02_xg_boost_regressor.ipynb
│   ├── 03_lightgbm_regressor.ipynb
│   ├── 04_lasso_feature_selection.ipynb
│   ├── 05_catboost_regressor.ipynb
│   └── wrapped_models/                  # Custom from-scratch implementations
│       ├── catboost_regressor.py
│       ├── catboost_tree.py
│       ├── lightgbm_regressor.py
│       ├── lightgbm_tree.py
│       ├── xgboost_regressor.py
│       └── xg_boost_regression_tree.py
├── src/
│   ├── config.py                        # Hyperparameter search spaces (Randomized & Grid)
│   ├── evaluate.py                      # Test set evaluation, residual analysis, CSV export
│   ├── pipeline.py                      # Model-specific pipeline factories
│   ├── preprocessing.py                 # Custom sklearn-compatible AmesPreprocessor
│   ├── train.py                         # Training loop, data splitting, target transformation
│   ├── plots/                           # Generated residual diagnostics
│   └── reports/                         # Worst-prediction and feature importance CSVs
└── tests/                               # Unit tests for AmesPreprocessor
```

---

## Algorithmic Foundations From Scratch

Beyond standard library usage, this project includes manual NumPy implementations to understand the underlying mathematics:

- **Linear Regression** — features batch gradient descent using analytical matrix gradients $\frac{2}{N}X^T(\hat{y}-y)$ and custom Z-score standardization $x' = \frac{x-\mu}{\sigma}$ computed strictly on training folds to prevent data leakage.
- **XGBoost** — implements level-wise gradient boosting with a second-order Taylor approximation. Designed with a constant Hessian ($h=2$), simplifying the similarity score to $SS = \frac{(\sum g_i)^2}{N + \lambda}$ and leaf outputs to the negative mean residual.
- **LightGBM** — adapts the boosting loop for greedy leaf-wise tree growth using a max-heap structure to chase the highest gain, controlled by a `max_leaves` constraint.
- **CatBoost** — builds symmetric (oblivious) trees allowing $O(\text{depth})$ inference. Handles categoricals natively via ordered target encoding, utilizing smoothed running averages $\frac{\text{sum} + \alpha \cdot \bar{y}}{\text{count} + \alpha}$ across multiple permutations to prevent target leakage.

---

## Production Pipeline

The production architecture utilizes a custom `AmesPreprocessor` executing hierarchical imputation and feature engineering solely on training data. Known outliers (living area $\ge$ 4000 sq ft) are dropped. The target is log-transformed using `np.log1p` to stabilize variance and reverted via `np.expm1` during evaluation for accurate dollar-scale metrics.

Three separate pipelines are constructed depending on the algorithm's optimal processing:

| Model | Scaling | Categorical Encoding | Feature Selection |
|---|---|---|---|
| Linear Regression | StandardScaler | One-Hot + Target + Ordinal | LassoCV ($\alpha \approx 0.00064$) |
| XGBoost / LightGBM | None | One-Hot + Target + Ordinal | None |
| CatBoost | None | Ordered Target Encoding (internal) | None |

---

## Hyperparameter Tuning

A two-stage search is applied to all tree-based models using 5-fold cross-validation:

1. **RandomizedSearchCV** — 50 random combinations sampled from log-uniform and integer distributions.
2. **GridSearchCV** — narrow grid search concentrated strictly around the best parameters found in stage one.

Linear Regression uses analytical feature pruning via LassoCV and requires no further grid tuning.

---

## Benchmark Results

### Before Log Transformation & Outlier Removal

| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| **CatBoost** | 22,293 | 21,827 | 13,711 | 0.9406 |
| XGBoost | 22,618 | 22,013 | 13,838 | 0.9396 |
| LightGBM | 23,000 | 23,980 | 14,059 | 0.9283 |
| Linear Regression | 31,612 | 33,920 | 21,062 | 0.8565 |

### After Log Transformation & Outlier Removal (Evaluated in $)

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| **CatBoost** | 19,152 | 11,877 | 0.9484 |
| XGBoost | 19,314 | 12,818 | 0.9475 |
| Linear Regression | 19,593 | 13,273 | 0.9460 |
| LightGBM | 19,744 | 12,935 | 0.9452 |

Log transformation reduced CatBoost's Test RMSE by ~12% and corrected the severe linear assumptions violation for Linear Regression, causing its Test RMSE to plummet from 33,920 to 19,593.

---

## Diagnostics & Interpretability

### Top 10 Features (CatBoost Gain)

| Feature | Gain (%) | Description |
|---|---|---|
| TotalSF | 23.14 | Engineered combined surface area |
| Overall Qual | 17.57 | Overall material/finish quality |
| Bsmt Qual | 8.21 | Basement condition |
| Gr Liv Area | 4.27 | Above-ground living area |
| RemodelAge | 3.30 | Engineered age since last remodel |
| Kitchen Qual | 3.15 | Kitchen quality rating |
| Overall Cond | 3.07 | Overall condition rating |
| Lot Area | 2.96 | Total lot size |
| MS Zoning | 2.82 | General zoning classification |
| Garage Area | 2.72 | Total garage square footage |

### Error Analysis

Residual plots and absolute error sorting indicate that the worst predictions are concentrated in the `NridgHt` and `Crawfor` neighborhoods under `Partial` sale conditions. Tree ensembles average terminal leaf nodes and inherently struggle to extrapolate beyond extreme training boundaries for top-tier luxury new constructions (true price > $550,000).

---

## Execution Guide

### Using Docker

Executes the fully containerized pipeline, running training and evaluation consecutively, saving serialized models and CSV reports locally via a volume mount.

```bash
docker build -t ames-housing-ml .
docker run --rm -v $(pwd)/src:/app/src ames-housing-ml
```

### Local Environment

```bash
# Install exact locked dependencies
pip install -r requirements.txt

# Run two-stage tuning and save best pipeline artifact
python -m src.train

# Generate residual plots, worst-case CSVs, and feature importance
python -m src.evaluate
```
