
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('afa2e701598d20110228 (1).csv', sep=';')

# Preprocessing
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

df['year'] = df['date'].dt.year

pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

df = df.dropna(subset=pollutants)

X = df[['id', 'year']]
y = df[pollutants]

# Encode station IDs
X_encoded = pd.get_dummies(X, columns=['id'])

# Train model
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )
)

model.fit(X_encoded, y)

# Streamlit UI
st.title("Water Quality Prediction System")

st.write("Predict pollutant levels based on station ID and year.")

station_id = st.selectbox(
    "Select Station ID",
    sorted(df['id'].astype(str).unique())
)

year_input = st.number_input(
    "Enter Year",
    min_value=2000,
    max_value=2035,
    value=2024
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        'id': [station_id],
        'year': [year_input]
    })

    input_encoded = pd.get_dummies(input_data, columns=['id'])

    # Match training columns
    for col in X_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[X_encoded.columns]

    prediction = model.predict(input_encoded)[0]

    st.subheader("Predicted Pollutant Levels")

    for pollutant, value in zip(pollutants, prediction):
        st.write(f"{pollutant}: {value:.2f}")
```
