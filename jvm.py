import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("house.csv")
df = df.select_dtypes(include=['number']).dropna()

# Features and target
X = df.drop("price", axis=1)  # Replace with actual target column
y = df["price"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# UI elements
st.title("🏠 House Price Predictor")

st.write("Enter the following features:")

# Dynamically generate input fields for each feature
user_input = []
for col in X.columns:
    val = st.number_input(f"{col}", value=int(X[col].mean()))
    user_input.append(val)

# Make prediction
if st.button("Predict Price"):
    input_array = np.array(user_input).reshape(1, -1)
    predicted_price = model.predict(input_array)[0]
    st.success(f"Estimated House Price: {predicted_price:,.2f}")
# house-price-prediction
echo # house-price-prediction
