import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv("housing.csv")

# Select important features
important_numeric_features = ["median_income", "housing_median_age", "total_rooms", "population"]
important_categorical_features = ["ocean_proximity"]

X = df[important_numeric_features + important_categorical_features]
y = df["median_house_value"]

# Preprocessing
num_transformer = SimpleImputer(strategy="median")
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, important_numeric_features),
        ("cat", cat_transformer, important_categorical_features)
    ]
)

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model and feature lists
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/gb_model.pkl")
joblib.dump(important_numeric_features, "model/numeric_features.pkl")
joblib.dump(important_categorical_features, "model/categorical_features.pkl")

print("✅ Gradient Boosting Model trained with important features only.")
