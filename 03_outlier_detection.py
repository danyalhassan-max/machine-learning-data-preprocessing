import pandas as pd


# Load dataset
df = pd.read_csv("simpleimputer_practice.csv")


# Display salary statistics
print("Salary statistics:")
print(df["salary"].describe())


# ============================================================
# IQR METHOD
# ============================================================

Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print("\nIQR Method")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# Find IQR outliers
iqr_outliers = df[
    (df["salary"] < lower_bound) |
    (df["salary"] > upper_bound)
]

print("\nIQR Outliers:")
print(iqr_outliers)


# ============================================================
# Z-SCORE METHOD
# ============================================================

mean = df["salary"].mean()
std = df["salary"].std()


# Calculate Z-score
df["salary_zscore"] = (
    df["salary"] - mean
) / std


print("\nZ-Score Values:")
print(df[["salary", "salary_zscore"]])


# Find Z-score outliers
zscore_outliers = df[
    df["salary_zscore"].abs() > 3
]


print("\nZ-Score Outliers:")
print(zscore_outliers)
