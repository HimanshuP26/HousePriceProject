# House Price Prediction with User Input

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")

print("===== Dataset =====")
print(data)

# Features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n===== Model Training Completed =====")

# Evaluate model
y_pred = model.predict(X_test)

print("\n===== Model Evaluation =====")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# TAKE INPUT FROM USER
print("\n===== Enter House Details =====")

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Predict price
new_house = [[area, bedrooms, bathrooms]]
predicted_price = model.predict(new_house)

print("\n===== Prediction Result =====")
print("Predicted House Price: ₹", round(predicted_price[0], 2))
