import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# Create sample categorical data
data = pd.DataFrame({
    "gender": ["Male", "Female", "Female", "Male", "Female"],
    "city": ["Lahore", "Sargodha", "Lahore", "Islamabad", "Sargodha"]
})

print("Original data:")
print(data)


# Create One-Hot Encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# Fit and transform categorical columns
encoded_data = encoder.fit_transform(
    data[["gender", "city"]]
)


# Get encoded column names
encoded_columns = encoder.get_feature_names_out(
    ["gender", "city"]
)


# Convert encoded data into DataFrame
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoded_columns
)


# Display encoded data
print("\nEncoded data:")
print(encoded_df)


# Display feature names
print("\nEncoded feature names:")
print(encoded_columns)
