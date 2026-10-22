# Water Quality Prediction

A Streamlit app that predicts six water-quality parameters for a monitoring station in a given year, using a multi-output Random Forest trained on 2000–2021 station measurements.

Predicted parameters: dissolved oxygen (O2), nitrate (NO3), nitrite (NO2), sulphate (SO4), phosphate (PO4) and chloride (CL).

## Features

- Pick one of the 22 monitoring stations and a year to get predicted levels for all six parameters at once
- One model handles all six targets (`MultiOutputRegressor` around `RandomForestRegressor`)
- The model is trained on the bundled CSV at startup and cached, so no model file has to be downloaded or kept in sync

## Tech stack

Python · pandas · scikit-learn · Streamlit · Jupyter

## Dataset

`data/water_quality.csv`: 2,861 semicolon-separated rows of station measurements (`id`, `date`, `NH4`, `BSK5`, `Suspended`, `O2`, `NO3`, `NO2`, `SO4`, `PO4`, `CL`). After rows with missing target values are dropped, 2,780 rows remain.

## Approach

1. Parse `date` and extract `year`
2. Drop rows with missing values in the six target columns
3. Features: one-hot encoded station `id` plus `year`
4. 80/20 train/test split (`random_state=42`), Random Forest with 20 trees

## Results

These are from the notebook's held-out test split (20%, 556 rows). I re-ran them with scikit-learn 1.8 and got the same numbers.

| Parameter | MAE | R² |
|---|---|---|
| O2 | 2.57 | -0.03 |
| NO3 | 2.97 | 0.43 |
| NO2 | 0.55 | -107.36 |
| SO4 | 16.26 | 0.29 |
| PO4 | 0.28 | 0.34 |
| CL | 31.48 | 0.71 |

What these numbers show:

- Station and year alone explain chloride reasonably well, and nitrate, sulphate and phosphate only partly.
- For dissolved oxygen the model does no better than predicting the mean (R² ≈ 0).
- The NO2 R² is dominated by outliers: the training split contains a value of 109 while the test split's maximum is 6, so a few trees predict very large values. NO2 needs outlier handling or a log transform before this model is useful for it.
- Random forests cannot extrapolate, so any year after 2021 gets the 2021 prediction.

Possible next steps: add seasonality (`month`), use the other measured parameters as inputs, treat outliers, and compare against a per-station mean baseline.

## Project structure

```
water-quality-prediction/
├── app.py                          # Streamlit app (trains and caches the model at startup)
├── water_quality_prediction.ipynb  # EDA, training and evaluation
├── data/
│   └── water_quality.csv           # station measurements
├── tests/
│   └── test_app.py                 # headless smoke test of the app
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/saanvi4996/water-quality-prediction.git
cd water-quality-prediction
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

Smoke test (runs the app headless and makes one prediction):

```bash
pip install pytest
pytest
```

## Deploy (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io) choose **New app**, select this repository and branch, and set the main file to `app.py`.
3. No secrets are needed. The dataset is in the repository and the model trains in a few seconds on first load.
