# Student Performance Prediction

A machine learning project that predicts a student's **Performance Index** (0–100) from study habits and previous exam results. The notebook compares up to 16 regression models, selects the best one on unseen test data, and provides a reusable prediction function.

The analysis lives in `Student_performance.ipynb`; the data lives in `Student_Performance.csv`.

## Dataset

The raw file has **10,000 rows and 6 columns**.

| Column | Type | Description |
| --- | --- | --- |
| Hours Studied | Numeric | Study hours per day |
| Previous Scores | Numeric | Score in the previous exam |
| Extracurricular Activities | Categorical | `Yes` or `No` |
| Sleep Hours | Numeric | Sleep hours per day |
| Sample Question Papers Practiced | Numeric | Number of practice papers completed |
| Performance Index | Numeric (target) | Academic performance score to predict |

### Data quality

- **132 duplicate rows** were removed, leaving **9,868 rows**.
- `Extracurricular Activities` had **1,572 missing values** in the raw file (15.7%). They were filled with the most frequent value (`No`).
- `Sample Question Papers Practiced` had **2,856 missing values** (28.6%). They were filled with the median (`5`).
- `Extracurricular Activities` was encoded as `Yes → 1`, `No → 0`.

No rows were dropped because of missing values: only exact duplicate rows were removed. Imputation can make two rows look identical, but those rows are kept because they may represent different students whose values were unknown.

## Workflow

1. Load the CSV and inspect data types, ranges, missing values, and duplicates.
2. Clean the data: remove duplicates, impute missing values, encode the binary feature.
3. Explore the target distribution, correlations, and feature/target relationships.
4. Split the data into 80% training and 20% test sets (`random_state=42`).
5. Fit `MinMaxScaler` on the **training set only** and apply it to both sets. This prevents data leakage.
6. Train and evaluate the models with MAE, MSE, RMSE, and R² on both train and test data.
7. Rank the models by test R² and inspect the best model with actual-vs-predicted and residual plots.
8. Show feature importance and provide a `predict_performance()` helper for new students.

## Models Compared

- **Linear:** Linear Regression, Ridge, Lasso
- **Distance / kernel:** K-Nearest Neighbors, Support Vector Regression
- **Tree-based:** Decision Tree, Random Forest, ExtraTrees, Bagging
- **Boosting / ensembles:** AdaBoost, Gradient Boosting, HistGradient Boosting, XGBoost, LightGBM, CatBoost (when installed), Stacking

This run evaluated **15 models**. CatBoost is optional: the notebook includes it automatically when the package is installed.

All models use the same scaled training data, so the comparison is consistent.

## Results

Models sorted by **test R²**:

| Model | MAE (test) | RMSE (test) | R² (test) | R² (train) |
| --- | ---: | ---: | ---: | ---: |
| SVR | 1.6534 | 2.0764 | 0.9886 | 0.9883 |
| Linear Regression | 1.6541 | 2.0770 | 0.9886 | 0.9883 |
| Ridge | 1.6550 | 2.0780 | 0.9886 | 0.9883 |
| Lasso | 1.6573 | 2.0794 | 0.9886 | 0.9883 |
| XGBoost | 1.7075 | 2.1361 | 0.9879 | 0.9891 |
| GradientBoosting | 1.7071 | 2.1399 | 0.9879 | 0.9886 |
| HistGradientBoosting | 1.7262 | 2.1614 | 0.9876 | 0.9898 |
| Stacking | 1.7304 | 2.1644 | 0.9876 | 0.9892 |
| LightGBM | 1.7340 | 2.1708 | 0.9875 | 0.9903 |
| Bagging | 1.8280 | 2.2880 | 0.9862 | 0.9925 |
| Random Forest | 1.9435 | 2.4302 | 0.9844 | 0.9971 |
| Decision Tree | 2.0708 | 2.6029 | 0.9821 | 0.9911 |
| ExtraTrees | 2.1377 | 2.6744 | 0.9811 | 0.9989 |
| AdaBoost | 2.2472 | 2.8280 | 0.9789 | 0.9789 |
| KNN | 2.6628 | 3.3161 | 0.9709 | 0.9813 |

### Best model

**SVR** achieved the highest test R², but the margin over the linear models is negligible.

| Metric | Train | Test |
| --- | ---: | ---: |
| MAE | 1.6452 | 1.6534 |
| MSE | 4.2971 | 4.3113 |
| RMSE | 2.0729 | 2.0764 |
| R² | 0.9883 | 0.9886 |

The top three models are **SVR**, **Linear Regression**, and **Ridge**; their test R² scores are effectively identical (0.9886). The differences are small because the data is dominated by a strong, nearly linear relationship.

### Key findings

- `Previous Scores` is the dominant predictor (correlation ≈ **0.92** with the target).
- `Hours Studied` adds a smaller but consistent positive effect (correlation ≈ **0.37**).
- `Sleep Hours` and `Sample Question Papers Practiced` have weak linear correlations (below **0.05**), so they contribute little on their own.
- Train and test scores are close for the best models, which indicates good generalisation rather than heavy overfitting.

## Requirements

Python 3.11+ with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

Optional: CatBoost adds one more model when installed.

```bash
pip install catboost
```

Optional, only for the interactive app:

```bash
pip install gradio
```

## How to Run

1. Clone or download the project folder.
2. Keep `Student_Performance.csv` in the same folder as the notebook.
3. Open `Student_performance.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
4. Run all cells from top to bottom. Training takes a few minutes because 15 models are fitted in this run (up to 16 when CatBoost is installed).
5. To try the Gradio app, install `gradio`, set `RUN_GRADIO = True` in the last code cell, and run it. The app launches locally; add `share=True` only if you need a temporary public link.

## Project Files

| File | Description |
| --- | --- |
| `Student_performance.ipynb` | Data inspection, cleaning, EDA, model training, evaluation, plots, and prediction helper |
| `Student_Performance.csv` | Raw student performance dataset |
| `README.md` | Project overview, results, and run instructions |

## Notes on Reproducibility

- The train/test split and every model use `random_state=42`.
- The scaler is fitted on training data only, so test rows never influence preprocessing.
- The notebook is executed from top to bottom and contains no stale outputs.
