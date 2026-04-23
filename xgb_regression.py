import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

file_path = "/home/hafizi/Documents/Data Driven Machine Learning/data.csv"
df = pd.read_csv(file_path)

df.drop(columns=["Sl. No."], inplace=True)

# Define input and output columns
target_columns = ["Fatigue"]
X = df.drop(columns=target_columns)  # All columns except the outputs
# y = df[target_columns]  # Target

# # Handle missing values (if any)
# X.fillna(X.mean(), inplace=True)
# y.fillna(y.mean(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# xgb_params = {
#     "objective": "reg:squarederror",  # Regression task
#     "eval_metric": "rmse",
#     "learning_rate": 0.1,
#     "max_depth": 6,
#     "n_estimators": 100,
#     "seed": 42
# }

# model = xgb.XGBRegressor(**xgb_params)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("\nModel Performance:")
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"R² Score: {r2:.4f}")

# joblib.dump(model, "xgboost_fatigue_model.pkl")
# joblib.dump(scaler, "scaler.pkl")

# print("\nModel saved successfully!")

#---------------------------------------- LOAD MODEL AND PREDICT ----------------------------------------#
loaded_model = joblib.load("xgboost_fatigue_model.pkl")
X_new = scaler.transform(pd.DataFrame([[885, 30, 0, 0, 30, 0, 30, 0, 30, 30, 0, 0, 0.26, 0.21, 0.44, 0.017, 0.022, 0.01, 0.02, 0.01, 0, 825, 0.07, 0.02, 0.01]], columns=X.columns))
y_new_pred = loaded_model.predict(X_new)
print("Predicted Fatigue Strength:", y_new_pred)
