# Machine Learning Data Preprocessing

Practical implementation of essential data preprocessing techniques for Machine Learning using Python and scikit-learn.

This repository focuses on learning preprocessing through hands-on coding, official scikit-learn examples, experimentation, and real datasets.

## Purpose

Data preprocessing is an important step in a Machine Learning workflow.

Raw datasets commonly contain:

- Missing values
- Numerical features with different scales
- Categorical features
- Outliers
- Features that require transformation
- Unnecessary or less useful features

This repository documents my practical learning of how to prepare data before training Machine Learning models.

## Technologies

- Python
- Pandas
- NumPy
- scikit-learn

## Topics Covered

### 1. Missing Values

- Mean imputation
- Median imputation
- Most frequent imputation
- Constant-value imputation
- SimpleImputer
- Train/Test data handling
- Avoiding data leakage

### 2. Feature Scaling

- StandardScaler
- MinMaxScaler
- RobustScaler
- MaxAbsScaler

### 3. Categorical Encoding

- OrdinalEncoder
- OneHotEncoder

### 4. Preprocessing Pipelines

- Pipeline
- make_pipeline
- Combining preprocessing with Machine Learning models

### 5. Feature Engineering

- Creating useful features
- PolynomialFeatures
- Feature transformations

### 6. Feature Selection

- Basic feature selection techniques
- Selecting relevant features for Machine Learning

## Repository Structure

```text
machine-learning-data-preprocessing/
│
├── 01_missing_values/
│   ├── simple_imputer_practice.py
│   └── simpleimputer_practice.csv
│
├── 02_feature_scaling/
│   ├── standard_scaler.py
│   ├── minmax_scaler.py
│   └── robust_scaler.py
│
├── 03_categorical_encoding/
│   ├── ordinal_encoder.py
│   └── onehot_encoder.py
│
├── 04_pipelines/
│   └── preprocessing_pipeline.py
│
├── 05_feature_engineering/
│   └── polynomial_features.py
│
├── 06_feature_selection/
│   └── feature_selection.py
│
├── README.md
└── requirements.txt
