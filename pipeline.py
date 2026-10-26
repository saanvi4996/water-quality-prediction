"""Data loading, feature encoding and model definition shared by the app and evaluate.py."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

DATA_PATH = Path(__file__).parent / "data" / "water_quality.csv"

POLLUTANTS = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']


def load_data(path=DATA_PATH):
    """Read the station measurements, extract the year and drop rows missing a target."""
    df = pd.read_csv(path, sep=';')
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['year'] = df['date'].dt.year
    return df.dropna(subset=POLLUTANTS)


def encode(X, columns=None):
    """One-hot encode station id (first station is the baseline), keeping year as is.

    Pass the training columns to align a new input with the fitted model.
    """
    X_encoded = pd.get_dummies(X.astype({'id': int}), columns=['id'], drop_first=columns is None)
    if columns is not None:
        X_encoded = X_encoded.reindex(columns=columns, fill_value=0)
    return X_encoded


def make_model():
    return MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=20,
            random_state=42
        )
    )
