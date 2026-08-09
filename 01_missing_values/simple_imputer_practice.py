import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# 1. Load dataset
df = pd.read_csv("simpleimputer_practice.csv")

print("\n--- Dataset Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values Before Imputation ---")
print(df.isnull().sum())


# 2. Separate features (X) and target (y)
X = df.drop("purchased", axis=1)
y = df["purchased"]


# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- Shapes ---")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# 4. Define feature groups
mean_features = ["age", "experience"]
median_features = ["salary"]
categorical_features = ["gender", "city"]


# 5. Impute numerical features
mean_imputer = SimpleImputer(strategy="mean")
median_imputer = SimpleImputer(strategy="median")

# Learn from X_train and transform X_train.
X_train.loc[:, mean_features] = mean_imputer.fit_transform(
    X_train[mean_features]
)

X_train.loc[:, median_features] = median_imputer.fit_transform(
    X_train[median_features]
)

# Use statistics learned from X_train to transform X_test.
X_test.loc[:, mean_features] = mean_imputer.transform(
    X_test[mean_features]
)

X_test.loc[:, median_features] = median_imputer.transform(
    X_test[median_features]
)


# 6. Impute categorical features
categorical_imputer = SimpleImputer(strategy="most_frequent")

X_train.loc[:, categorical_features] = categorical_imputer.fit_transform(
    X_train[categorical_features]
)

X_test.loc[:, categorical_features] = categorical_imputer.transform(
    X_test[categorical_features]
)


# 7. Inspect what the imputers learned
print("\n--- Learned Numerical Statistics ---")
print("Mean imputer:", mean_imputer.statistics_)
print("Median imputer:", median_imputer.statistics_)

print("\n--- Learned Categorical Values ---")
print("Categorical imputer:", categorical_imputer.statistics_)


# 8. Inspect final datasets
print("\n--- X_train After Imputation ---")
print(X_train)

print("\n--- X_test After Imputation ---")
print(X_test)


# 9. Verify that missing values are handled
print("\n--- Missing Values After Imputation ---")
print("X_train:")
print(X_train.isnull().sum())

print("\nX_test:")
print(X_test.isnull().sum())
