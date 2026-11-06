# 💳 Credit Card Fraud Detection – Kaggle Playground Series S6E11

## 📌 Project Overview

This project focuses on developing a Machine Learning model to detect fraudulent credit card transactions using the [Kaggle Playground Series – Season 6, Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11) dataset.

The objective is to predict the probability that a given credit card transaction is fraudulent based on 25 explanatory features, including transaction amounts, temporal patterns, categorical attributes, and customer behavior indicators.

This project covers the end-to-end Machine Learning workflow, including Exploratory Data Analysis (EDA), data preprocessing, feature transformation, model training, validation, and prediction generation.

**Kaggle Notebook:** [Predicting Credit Card Fraud](https://www.kaggle.com/code/hngvvan/predicting-credit-card-fraud)

---

## 📊 Results & Evaluation

| Model | Public Score (ROC-AUC) |
| :--- | :---: |
| HistGradientBoostingClassifier | **0.94515** |

**Evaluation Metric: ROC-AUC**

The competition uses the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) to evaluate model performance.

ROC-AUC measures how effectively a model distinguishes between fraudulent and legitimate transactions across different classification thresholds.

- **Public Score:** 0.94515
- **Model:** HistGradientBoostingClassifier
- **Prediction Output:** Fraud probabilities (`is_fraud`)

---

## 🛠️ Methodology & Implementation

### 1. Exploratory Data Analysis (EDA)

Conducted exploratory data analysis to understand the dataset's structure, feature distributions, and potential relationships before model development.

Key analysis areas:

- **Target Distribution:** Examined the distribution of fraudulent and legitimate transactions to identify class imbalance.
- **Feature Analysis:** Investigated numerical and categorical feature distributions.
- **Boolean Features:** Identified boolean variables such as `is_foreign_transaction` and `used_vpn`.
- **Data Quality:** Inspected missing values and feature data types.

### 2. Data Preprocessing

Implemented a preprocessing pipeline using Scikit-learn's `ColumnTransformer` to handle numerical and categorical features separately.

**Numerical Features:**
- Applied `SimpleImputer(strategy='median')` to handle missing values.
- Used `StandardScaler` to standardize numerical feature scales.

**Categorical Features:**
- Applied `SimpleImputer(strategy='constant', fill_value='None')` to handle missing values.
- Used `OneHotEncoder(handle_unknown='ignore')` to convert categorical variables into numerical representations.
- Configured the encoder to handle previously unseen categories in the test dataset.

**Boolean Features:**
- Converted boolean variables into integer representations to ensure consistent numerical processing.

**Pipeline Architecture:**

The preprocessing steps were integrated into a Scikit-learn pipeline to maintain a consistent transformation process during training and prediction.

This approach also helps prevent data leakage by fitting preprocessing transformations only on the training data.

### 3. Model Development & Training

**Selected Model: HistGradientBoostingClassifier**

Used `HistGradientBoostingClassifier` as the core classification algorithm.

Key configuration:

| Parameter | Value |
| :--- | :--- |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `early_stopping` | True |

**Why HistGradientBoostingClassifier?**

- Uses histogram-based gradient boosting to improve computational efficiency.
- Captures complex and non-linear relationships between features.
- Supports early stopping to help control overfitting.
- Suitable for structured tabular datasets.

### 4. Model Validation

Applied an 80/20 train-validation split using `train_test_split`.

Key configurations:

- `test_size=0.2`
- `stratify=y`

Stratified sampling ensures that the proportion of fraudulent and legitimate transactions remains approximately consistent across the training and validation sets.

The validation set was used to evaluate model performance using ROC-AUC.

### 5. Final Training & Prediction

After evaluating the model locally, the preprocessing and classification pipeline was refitted using the entire available training dataset.

The trained pipeline was then used to generate fraud probabilities for the competition's test dataset.

Finally, predictions were exported to `submission.csv` in the required Kaggle submission format.

---

## 💻 Tech Stack

| Category | Technologies |
| :--- | :--- |
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development Environment | Jupyter Notebook |
| Platform | Kaggle |

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── fraud_pipeline.ipynb    # Main notebook: EDA, preprocessing,
│                           # model training, validation and prediction
│
├── submission.csv          # Generated Kaggle submission file
│
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <https://github.com/vuvanhung2005hd-lang/Predicting-Credit-Card-Fraud.git>
cd Credit-Card-Fraud-Detection
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Download the Dataset

Download the competition dataset from Kaggle:

[Credit Card Fraud Detection – Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s6e11/data)

Place the downloaded dataset files in the appropriate directory used by the notebook.

### 4. Run the Notebook

```bash
jupyter notebook fraud_pipeline.ipynb
```

Run the notebook cells sequentially to reproduce the data analysis, preprocessing, model training, and prediction workflow.

---

## 🎯 Key Learnings

Through this project, I gained practical experience in:

- Performing Exploratory Data Analysis on structured tabular datasets.
- Identifying and understanding class imbalance in binary classification problems.
- Building preprocessing pipelines for numerical, categorical, and boolean features.
- Applying Scikit-learn's `ColumnTransformer` and `Pipeline`.
- Training and evaluating gradient boosting classification models.
- Using stratified train-validation splitting and ROC-AUC for model evaluation.
- Generating probability-based predictions for Kaggle competitions.

---



