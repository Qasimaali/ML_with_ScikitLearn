# =========================================
# 🌱 Scikit-learn Fundamentals in One Code
# =========================================

# --- Step 1: Import Required Libraries ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 2: Create Sample Dataset ---
# Imagine we have data of hours studied vs marks scored
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored':  [20, 30, 40, 50, 60, 70, 76, 85, 90, 95]
}

df = pd.DataFrame(data)
print("\n--- Original Data ---")
print(df)

# --- Step 3: Split Features (X) and Target (y) ---
X = df[['Hours_Studied']]   # feature (input)
y = df['Marks_Scored']      # target (output)

# --- Step 4: Train-Test Split ---
# We keep 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n--- Training and Testing Split ---")
print("Training examples:", len(X_train))
print("Testing examples :", len(X_test))

# --- Step 5: Apply Scaling (Feature Transformation) ---
# Create scaler object
scaler = StandardScaler()

# Learn scaling parameters from training data and transform it
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled = scaler.transform(X_test)         # only transform

print("\n--- Example of Scaled Features ---")
print("Before scaling (first few values):")
print(X_train.head().to_numpy().flatten())
print("After scaling:")
print(X_train_scaled[:5].flatten())

# --- Step 6: Create and Train Model ---
model = LinearRegression()       # create model object
model.fit(X_train_scaled, y_train)   # learn from training data

# --- Step 7: Make Predictions ---
y_pred = model.predict(X_test_scaled)

# --- Step 8: Evaluate the Model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# --- Step 9: Predict for New Data ---
# Suppose a student studied 7.5 hours
new_data = np.array([[7.5]])
new_data_scaled = scaler.transform(new_data)     # scale new input
predicted_marks = model.predict(new_data_scaled)

print("\n--- Prediction Example ---")
print("If a student studies 7.5 hours, predicted marks =", round(predicted_marks[0], 2))

# --- Step 10: Summary of Functions Used ---
print("""
✅ Summary of Concepts:
.fit()           -> Trains model / scaler on data
.transform()     -> Applies learned transformation (scaler)
.fit_transform() -> Learns + applies in one go
.predict()       -> Uses trained model to predict new results
""")
