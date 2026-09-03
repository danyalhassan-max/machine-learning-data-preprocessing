import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# Create sample ordinal data
data = pd.DataFrame({
    "size": [
        "Small",
        "Large",
        "Medium",
        "Small",
        "Large"
    ]
})

print("Original data:")
print(data)


# Create Ordinal Encoder
encoder = OrdinalEncoder(
    categories=[["Small", "Medium", "Large"]]
)


# Fit and transform the ordinal column
data["size_encoded"] = encoder.fit_transform(
    data[["size"]]
)


# Display encoded data
print("\nEncoded data:")
print(data)


# Display category order
print("\nCategory order:")
print(encoder.categories_)
