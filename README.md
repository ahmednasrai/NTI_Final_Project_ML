# Student Performance Prediction

This project uses machine learning to predict a student's **Performance Index** from study habits and previous exam results.

The main work is in `Student_performance.ipynb`. The file `Student_Performance.csv` contains the training data.

## Project Goal

The goal is to help estimate a student's academic performance using these inputs:

- Hours studied
- Previous exam scores
- Extracurricular activities
- Sleep hours
- Number of sample question papers practiced

The project compares several machine learning models and selects the model with the best test result.

## Dataset

The dataset contains 10,000 rows and 6 columns.

| Column | Description |
| --- | --- |
| Hours Studied | Number of hours the student studies |
| Previous Scores | The student's previous exam score |
| Extracurricular Activities | `Yes` or `No` |
| Sleep Hours | Hours of sleep per day |
| Sample Question Papers Practiced | Number of practice papers completed |
| Performance Index | Academic performance score to predict |

The notebook removes duplicate rows, fills missing values, and converts the extracurricular activity column to a numeric value.

## What the Notebook Does

1. Loads and describes the dataset.
2. Checks missing values and duplicate rows.
3. Cleans and prepares the data.
4. Creates charts to explore relationships in the data.
5. Splits the data into training and test sets.
6. Scales the features with `MinMaxScaler`.
7. Trains and compares many regression models.
8. Displays the best model and compares actual values with predicted values.
9. Creates a simple Gradio prediction interface.

The notebook compares Linear Regression, Ridge, Lasso, KNN, SVM, Decision Tree, Random Forest, XGBoost, CatBoost, LightGBM, and several other regression and ensemble models.

According to the model comparison saved in the notebook, **Linear Regression** has the highest test R2 score. The test R2 score is about **0.988**.

## Requirements

Install these Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tqdm xgboost catboost lightgbm gradio
```

## How to Run

1. Download or clone this project.
2. Open `Student_performance.ipynb` in Jupyter Notebook, VS Code, or Google Colab.
3. Put `Student_Performance.csv` in the same folder as the notebook.
4. Run all notebook cells from the beginning.
5. The notebook will train the models and create a prediction form.

## Main Files

- `Student_performance.ipynb` - Data analysis, model training, evaluation, charts, and prediction interface.
- `Student_Performance.csv` - Student performance data.
- `README.md` - Project information and instructions.