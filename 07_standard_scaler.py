import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset
df = pd.read_csv("simpleimputer_practice.csv")


# Separate features and target
x = df.drop("purchased", axis=1)
y = df["purchased"]


# Select numerical features
numeric_features = [
    "age",
    "salary",
    "experience"
]


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# Create StandardScaler
scaler = StandardScaler()


# Fit and transform training data
x_train_scaled = scaler.fit_transform(
    x_train[numeric_features]
)


# Transform testing data
x_test_scaled = scaler.transform(
    x_test[numeric_features]
)


# Display scaler parameters
print("Mean:")
print(scaler.mean_)

print("\nScale:")
print(scaler.scale_)


# Display scaled training data
print("\nScaled training data:")
print(x_train_scaled)


# Display scaled testing data
print("\nScaled testing data:")
print(x_test_scaled)
