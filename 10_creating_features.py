import pandas as pd


# Load dataset
df = pd.read_csv("simpleimputer_practice.csv")


# Display original data
print("Original data:")
print(df)


# Create a new feature: income per year of experience
df["salary_per_experience"] = (
    df["salary"] / df["experience"]
)


# Create a new feature: age group
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 40, 100],
    labels=["Young", "Adult", "Senior"]
)


# Display new features
print("\nData after creating new features:")
print(df)


# Display only the newly created features
print("\nCreated features:")
print(
    df[
        [
            "salary_per_experience",
            "age_group"
        ]
    ]
)
