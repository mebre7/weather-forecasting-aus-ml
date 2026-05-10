# 🌧️ Australia Rainfall Prediction using Logistic Regression

## Project Overview

This project builds a binary classification model to predict whether it will rain tomorrow in Australia, using the **Rain in Australia** dataset from Kaggle. The model is trained using **Logistic Regression** and covers the full ML pipeline: data loading, EDA, preprocessing, training, and evaluation.

---

## 📊 Dataset

- **Source:** [Kaggle – Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- **Size:** ~145,460 rows × 23 columns (10 years of daily weather observations)
- **Target variable:** `RainTomorrow` (Yes / No)

### Features

| Type | Columns |
|------|---------|
| Numeric (float64) | `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustSpeed`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am`, `Temp3pm` |
| Categorical (string) | `Date`, `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`, `RainToday`, `RainTomorrow` |

---

## 🗂️ Project Structure

```
weather-forecasting-aus-ml/
├── data/
│   ├── weatherAUS.csv          # Raw dataset
│   ├── train_dataset/          # Preprocessed training split
│   ├── validate_dataset/       # Preprocessed validation split
│   ├── test_dataset/           # Preprocessed test split
│   └── model_workflow.png      # Model pipeline diagram
├── models/
│   └── aussie_rain_lr_model.joblib  # Saved trained model
├── notebook/
│   └── analyze_and_train.ipynb # Full analysis and training notebook
├── src/                        # Source scripts
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Data Cleaning
- Dropped the `Date` column (not predictive).
- Removed rows where `RainToday` or `RainTomorrow` is missing (4,673 rows removed).
- Remaining dataset: **140,787 rows**.

### 2. Exploratory Data Analysis (EDA)
- Visualized rain distribution across locations using Seaborn and Plotly.
- Analyzed correlations between numeric features and the target variable.
- Explored class imbalance in `RainTomorrow`.

### 3. Preprocessing
- **Numeric columns:** Imputed missing values (mean/median), then scaled using `MinMaxScaler`.
- **Categorical columns:** Imputed missing values (mode), then encoded using `OneHotEncoder`.
- Final feature matrix: **118 columns** (16 numeric + 102 one-hot encoded).

> ⚠️ Note: `WindGustDir`, `WindDir9am`, and `WindDir3pm` had missing values, so the encoder added an extra `nan` category column for each — resulting in 102 categorical columns instead of 99.

### 4. Train / Validation / Test Split
- Dataset split into training, validation, and test sets.
- Splits saved to `data/train_dataset/`, `data/validate_dataset/`, and `data/test_dataset/`.

### 5. Model Training
- Algorithm: **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- Trained on the preprocessed training set.
- Model saved to `models/aussie_rain_lr_model.joblib` using `joblib`.

### 6. Evaluation
- Evaluated on validation and test sets.
- Metrics reported: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

---

## ⚙️ Setup & Usage

### Prerequisites

```bash
python -m venv wf_ve
source wf_ve/bin/activate
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook notebook/analyze_and_train.ipynb
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML model and preprocessing |
| `matplotlib` / `seaborn` | Static visualizations |
| `plotly` | Interactive visualizations |
| `joblib` | Model serialization |

---

## 📈 Results

The Logistic Regression model was trained on ~10 years of Australian weather data. Performance metrics are reported in the notebook after evaluation on the held-out test set.

---

## 📝 Notes

- The `Date` column was excluded as it is not directly predictive of tomorrow's rain.
- `RainTomorrow` (target) was excluded from the feature set during training.
- Missing values in wind direction columns (`WindGustDir`, `WindDir9am`, `WindDir3pm`) were treated as a separate category by the encoder — handle these before encoding to avoid extra `nan` columns.

> The model can be further improved by exploring more complex algorithms (e.g., Random Forest, XGBoost) and performing hyperparameter tuning.