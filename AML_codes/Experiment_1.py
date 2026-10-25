# Experiment 1: Predicting Housing Prices Using Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the Kaggle Housing Prices dataset
# Download Housing.csv from the Kaggle URL given in the dataset description
df = pd.read_csv("Housing.csv")

# 2. Basic data inspection
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe())

# 3. Separate input features and target
X = df.drop(columns="price")
y = df["price"]

# 4. Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 5. Preprocessing: one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
   ("num", "passthrough", numerical_features),
   ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features)
  ]
)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 7. Build and train the model pipeline
model = Pipeline(steps=[
   ("preprocessor", preprocessor),
   ("regressor", LinearRegression())
])
model.fit(X_train, y_train)

# 8. Generate predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# 10. Actual vs predicted visualization
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# 11. Display six sample predictions
results = pd.DataFrame({
    "Actual Value": y_test.values,
    "Predicted Value": y_pred,
    "Error": y_test.values - y_pred
})
print("\nSample Predictions:\n", results.head(6).round(2))

