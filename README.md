# Machine Learning Preprocessing

A practical implementation of essential data preprocessing techniques used in Machine Learning with Python, Pandas, NumPy, and Scikit-learn.

This repository documents hands-on practice with data cleaning, categorical encoding, feature scaling, outlier detection, feature creation, and feature transformation.

---

## About

Data preprocessing is a critical stage of the Machine Learning workflow.

Real-world datasets commonly contain:

- Missing values
- Duplicate records
- Outliers
- Categorical variables
- Different numerical scales
- Features that require transformation
- Features that can be derived from existing data

This repository contains practical implementations of these preprocessing techniques using Python and Scikit-learn.

The goal is not to memorize preprocessing APIs, but to understand how and when each technique is applied in a Machine Learning workflow.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## Topics Covered

### 1. Data Cleaning

- Missing Values
- Duplicate Values
- Outlier Detection
  - IQR Method
  - Z-Score Method

### 2. Encoding

- Label Encoding
- One-Hot Encoding
- Ordinal Encoding

### 3. Feature Scaling

- StandardScaler
- MinMaxScaler
- RobustScaler

### 4. Feature Engineering

- Creating Features
- Feature Transformation

---

## Repository Structure

```text
preprocessing/
│
├── 01_missing_values.py
├── 02_duplicate_values.py
├── 03_outlier_detection.py
│
├── 04_label_encoding.py
├── 05_one_hot_encoding.py
├── 06_ordinal_encoding.py
│
├── 07_standard_scaler.py
├── 08_minmax_scaler.py
├── 09_robust_scaler.py
│
├── 10_creating_features.py
├── 11_feature_transformation.py
│
├── simpleimputer_practice.csv
├── requirements.txt
└── README.md
