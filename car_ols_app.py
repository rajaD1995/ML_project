import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and selected features
model = pickle.load(open(r'ols_model.pkl', 'rb'))

# # ✨ Define selected features (from training phase)
# selected_features = ['cyl', 'disp', 'hp', 'wt', 'acc', 'yr', 'car_type', 'origin_asia', 'origin_europe']

# Title and description
st.title("🚗 Car MPG Prediction App")
st.write("This app predicts the **Miles Per Gallon (MPG)** of a car based on its features using a trained OLS model.")

# Sidebar inputs
st.sidebar.header("🔧 Input Car Features")
cyl = st.sidebar.selectbox("Number of Cylinders", [4, 6, 8])
disp = st.sidebar.number_input("Displacement (cu in)", 50, 500, 150)
hp = st.sidebar.number_input("Horsepower", 50, 300, 100)
wt = st.sidebar.number_input("Weight (lbs)", 1000, 6000, 3000)
# acc = st.sidebar.number_input("Acceleration (0–60 mph, sec)", 5.0, 25.0, 15.0)
yr = st.sidebar.slider("Model Year", 70, 82, 75)
origin = st.sidebar.selectbox("Origin", ["America", "Europe", "Asia"])
car_type = st.sidebar.selectbox("Car Type", ["Domestic", "Foreign"])

# Dummy mapping
origin_asia = 1 if origin == "Asia" else 0
origin_europe = 1 if origin == "Europe" else 0
origin_america = 1 if origin == "America" else 0
car_type = 1 if car_type == "Foreign" else 0

# Create input DataFrame
input_data = pd.DataFrame([{
    'cyl': cyl,
    'disp': disp,
    'hp': hp,
    'wt': wt,
    'yr': yr,
    'car_type': car_type,
    'origin_america': origin_america,
    'origin_asia': origin_asia,
    'origin_europe': origin_europe
}])

selected_features = ['cyl', 'disp', 'hp', 'wt', 'yr', 'car_type', 'origin_america', 'origin_asia', 'origin_europe']

if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Predict button
if st.button("🔍 Predict MPG"):
    # prediction = model.predict(input_data[selected_features])[0]
    prediction = model.predict(input_data)[0]
    st.session_state.prediction = prediction  # Save in session
    st.success(f"🔧 Predicted MPG: **{prediction:.2f}**")

# Optional: Fetch similar cars
if st.checkbox("🎯 Show cars with similar MPG"):
    prediction = model.predict(input_data)[0]
    df = pd.read_csv(r'C:\Users\USER\Documents\Python\Nareshit data analysis\stats and ML\ML\27th- l1, l2, scaling\lasso, ridge, elastic net\TASK-22_LASSO,RIDGE\car-mpg.csv')
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
    df = df.dropna(subset=['mpg'])
    tolerance = 1.0
    similar_cars = df[(df['mpg'] >= prediction - tolerance) & (df['mpg'] <= prediction + tolerance)]
    st.write(f"📋 Cars with MPG near prediction: **{prediction:.2f}**")
    st.dataframe(similar_cars[['car_name', 'mpg']])
# Display information about the model 
st.write("The model is build by Raja Debnath")
