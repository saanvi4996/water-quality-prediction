import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("pollution_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Pollutants
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Streamlit UI
st.title("Water Quality Prediction System")

st.write("Predict pollutant levels based on station ID and year.")

# Inputs
station_id = st.selectbox(
    "Select Station ID",
    [str(i) for i in range(1, 23)]
)

year_input = st.number_input(
    "Enter Year",
    min_value=2000,
    max_value=2035,
    value=2024
)

# Predict button
if st.button("Predict Water Quality"):

    # Create input dataframe
    input_data = pd.DataFrame({
        'year': [year_input],
        'id': [station_id]
    })

    # One-hot encoding
    input_encoded = pd.get_dummies(input_data, columns=['id'])

    # Add missing columns
    missing_cols = set(model_columns) - set(input_encoded.columns)

    for col in missing_cols:
        input_encoded[col] = 0

    # Arrange columns correctly
    input_encoded = input_encoded[model_columns]

    # Prediction
    prediction = model.predict(input_encoded)[0]

    st.subheader("Predicted Pollutant Levels")

    for pollutant, value in zip(pollutants, prediction):
        st.write(f"{pollutant}: {value:.2f}")