#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Feature engineering
df['RoomsPerPerson'] = df['AveRooms'] / (df['Population'] + 1)
df['BedroomsPerRoom'] = df['AveBedrms'] / (df['AveRooms'] + 1)
df['PopulationDensity'] = df['Population'] / (df['AveOccup'] + 1)

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

print("Training features:", X.columns)
print("Training shape:", X.shape)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb_model.fit(x_train, y_train)

# Evaluate
pred = xgb_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"✅ Model trained. RMSE={rmse:.3f}, R2={r2:.3f}")

# Save both model and scaler
joblib.dump(xgb_model, "house_price_xgb.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")

