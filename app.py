import streamlit as st
import pandas as pd

from pipeline import POLLUTANTS, encode, load_data, make_model


@st.cache_resource
def train_model():
    df = load_data()

    X_encoded = encode(df[['id', 'year']])
    y = df[POLLUTANTS]

    # Train model (cached, so this runs once per server start, not on every click)
    model = make_model()
    model.fit(X_encoded, y)

    station_ids = sorted(df['id'].unique())
    year_range = (int(df['year'].min()), int(df['year'].max()))
    return model, list(X_encoded.columns), station_ids, year_range


model, model_columns, station_ids, (min_year, max_year) = train_model()

# Streamlit UI
st.title("Water Quality Prediction System")

st.write(
    f"Predict levels of six water-quality parameters for any of the "
    f"{len(station_ids)} monitoring stations in a given year."
)
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

    # Match training columns
    input_encoded = encode(input_data, columns=model_columns)

    prediction = model.predict(input_encoded)[0]

    st.subheader("Predicted Pollutant Levels")

    for pollutant, value in zip(POLLUTANTS, prediction):
        st.write(f"{pollutant}: {value:.2f}")
