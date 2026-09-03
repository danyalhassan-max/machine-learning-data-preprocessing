import pandas as pd


# Load dataset
df = pd.read_csv("simpleimputer_practice.csv")


# Display original dataset
print("Original dataset:")
print(df)


# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())


# Display duplicate rows
print("\nDuplicate rows:")
print(df[df.duplicated()])


# Remove duplicate rows
df_cleaned = df.drop_duplicates()


# Display cleaned dataset
print("\nDataset after removing duplicates:")
print(df_cleaned)


# Verify duplicates have been removed
print("\nNumber of duplicate rows after cleaning:")
print(df_cleaned.duplicated().sum())


# Compare dataset shapes
print("\nOriginal shape:", df.shape)
print("Cleaned shape:", df_cleaned.shape)
