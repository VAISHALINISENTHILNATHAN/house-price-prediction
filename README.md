# House Price Prediction (California)

A machine learning project to predict **California housing prices** using regression models and a **Streamlit web app** for interactive predictions.

---

## 🏠 Project Overview

This project analyzes the **California Housing Dataset** to understand factors influencing house prices and builds machine learning models to predict the **median house value**.  

- Explored relationships between income, house age, rooms, and house value.
- Engineered features like `RoomsPerPerson`, `BedroomsPerRoom`, and `PopulationDensity`.
- Trained multiple regression models:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor (final production model)

The final model is deployed via **Streamlit** for real-time predictions.

---

## 📊 Data Analysis

- Visualized feature correlations using a **heatmap**.
- Distribution analysis of median house values.
- Scatter plots to explore relationships (e.g., `MedianIncome` vs. `MedianHouseValue`).
- Feature engineering for better predictive performance.

---

## ⚙️ Technologies & Libraries

- **Python 3.x**
- **NumPy & Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn & XGBoost** – Machine learning
- **Streamlit** – Interactive web app
- **Joblib** – Model serialization

All dependencies are listed in `requirements.txt`.

---
MODEL ANALYSIS 

| Model             | RMSE  | R² Score |
| ----------------- | ----- | -------- |
| Random Forest     | 0.529 | 0.81     |
| Gradient Boosting | 0.512 | 0.83     |
| XGBoost (Final)   | 0.498 | 0.84     |

XGBoost gave the best accuracy and is used in the Streamlit app.
