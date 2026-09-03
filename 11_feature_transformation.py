import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer


# Load dataset
df = pd.read_csv("simpleimputer_practice.csv")


# Display original data
print("Original data:")
print(df)


# Select numerical features
numeric_features = [
    "age",
    "salary",
    "experience"
]


# Create a FunctionTransformer
transformer = FunctionTransformer(
    np.log1p
)


# Apply transformation
transformed_data = transformer.fit_transform(
    df[numeric_features]
)


# Convert transformed data into DataFrame
transformed_df = pd.DataFrame(
    transformed_data,
    columns=numeric_features
)


# Display transformed data
print("\nTransformed data:")
print(transformed_df)


# Compare original and transformed salary
comparison = pd.DataFrame({
    "original_salary": df["salary"],
    "transformed_salary": transformed_df["salary"]
})


print("\nSalary transformation comparison:")
print(comparison)
