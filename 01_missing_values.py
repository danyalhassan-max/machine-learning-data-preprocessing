import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("simpleimputer_practice.csv")

# Inspect dataset
df.info()
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())


# Separate features and target
x = df.drop("purchased", axis=1)
y = df["purchased"]


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining shape:", x_train.shape)
print("Testing shape:", x_test.shape)


# Numerical features
mean_features = ["age", "experience"]
median_features = ["salary"]


# Mean imputation
mean_imputer = SimpleImputer(strategy="mean")

x_train[mean_features] = mean_imputer.fit_transform(
    x_train[mean_features]
)

x_test[mean_features] = mean_imputer.transform(
    x_test[mean_features]
)


# Median imputation
median_imputer = SimpleImputer(strategy="median")

x_train[median_features] = median_imputer.fit_transform(
    x_train[median_features]
)

x_test[median_features] = median_imputer.transform(
    x_test[median_features]
)


# Categorical features
categorical_features = ["gender", "city"]


# Most-frequent imputation
categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

x_train[categorical_features] = categorical_imputer.fit_transform(
    x_train[categorical_features]
)

x_test[categorical_features] = categorical_imputer.transform(
    x_test[categorical_features]
)


# Display processed data
print("\nProcessed training data:")
print(x_train)

print("\nProcessed testing data:")
print(x_test)


# Verify missing values are removed
print("\nMissing values in training data:")
print(x_train.isnull().sum())

print("\nMissing values in testing data:")
print(x_test.isnull().sum())
