# -----------------------------
# STREAMLIT SUPERVISED APP
# -----------------------------

import streamlit as st
import pandas as pd

# Title
st.title("🌱 Environmental Pollution Prediction")

# Load dataset
data = pd.read_csv("env.csv")

# Encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Pollution_Level'] = le.fit_transform(data['Pollution_Level'])

# Split features & target
X = data.drop("Pollution_Level", axis=1)
y = data["Pollution_Level"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Sidebar input
st.sidebar.header("Enter Environmental Details")

temp = st.sidebar.slider("Temperature (°C)", 20, 40, 30)
humidity = st.sidebar.slider("Humidity (%)", 30, 90, 60)
aqi = st.sidebar.slider("AQI", 50, 200, 100)
co2 = st.sidebar.slider("CO2 (ppm)", 350, 600, 420)
noise = st.sidebar.slider("Noise (dB)", 30, 80, 55)
green = st.sidebar.slider("Green Cover (%)", 5, 50, 20)

# Input dataframe
input_data = pd.DataFrame({
    'Temp': [temp],
    'Humidity': [humidity],
    'AQI': [aqi],
    'CO2': [co2],
    'Noise': [noise],
    'Green_Cover': [green]
})

# Prediction button
if st.button("Predict Pollution Level"):

    lr_pred = lr.predict(input_data)
    dt_pred = dt.predict(input_data)

    # Decode result
    lr_result = le.inverse_transform(lr_pred)[0]
    dt_result = le.inverse_transform(dt_pred)[0]

    st.subheader("Prediction Results")
    st.write("Logistic Regression:", lr_result)
    st.write("Decision Tree:", dt_result)

# Accuracy display
st.subheader("Model Accuracy")

st.write("Logistic Regression Accuracy:", lr.score(X_test, y_test))
st.write("Decision Tree Accuracy:", dt.score(X_test, y_test))