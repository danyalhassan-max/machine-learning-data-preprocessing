import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Create sample categorical data
data = pd.DataFrame({
    "gender": ["Male", "Female", "Female", "Male", "Female"]
})

print("Original data:")
print(data)


# Create Label Encoder
encoder = LabelEncoder()


# Fit and transform the categorical column
data["gender_encoded"] = encoder.fit_transform(
    data["gender"]
)


# Display encoded data
print("\nEncoded data:")
print(data)


# Display the mapping
print("\nClasses:")
print(encoder.classes_)


# Display encoding values
print("\nEncoding mapping:")

for index, category in enumerate(encoder.classes_):
    print(category, "=", index)
