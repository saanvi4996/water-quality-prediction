from pathlib import Path

import streamlit as st
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path(__file__).parent / "data" / "water_quality.csv"

pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']


@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH, sep=';')

    # Preprocessing
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['year'] = df['date'].dt.year
    df = df.dropna(subset=pollutants)

    X = df[['id', 'year']]
    y = df[pollutants]

    # Encode station IDs
    X_encoded = pd.get_dummies(X, columns=['id'])

    # Train model (cached, so this runs once per server start, not on every click)
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=20,
            random_state=42
        )
    )
    model.fit(X_encoded, y)

    station_ids = sorted(df['id'].unique())
    year_range = (int(df['year'].min()), int(df['year'].max()))
    return model, list(X_encoded.columns), station_ids, year_range


model, model_columns, station_ids, (min_year, max_year) = train_model()

# Streamlit UI
st.title("Water Quality Prediction System")

st.write("Predict pollutant levels based on station ID and year.")
st.caption(
    f"Trained on {min_year}–{max_year} measurements. A random forest cannot "
    f"extrapolate, so years after {max_year} return the same values as {max_year}."
)

station_id = st.selectbox(
    "Select Station ID",
    [str(i) for i in station_ids]
)

year_input = st.number_input(
    "Enter Year",
    min_value=2000,
    max_value=2035,
    value=2024
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        'id': [int(station_id)],
        'year': [year_input]
    })

    input_encoded = pd.get_dummies(input_data, columns=['id'])

    # Match training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    st.subheader("Predicted Pollutant Levels")

    for pollutant, value in zip(pollutants, prediction):
        st.write(f"{pollutant}: {value:.2f}")
