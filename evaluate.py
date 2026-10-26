"""Reproduce the held-out test metrics reported in the README.

Usage: python evaluate.py
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import POLLUTANTS, encode, load_data, make_model


def main():
    df = load_data()
    X = encode(df[['id', 'year']])
    y = df[POLLUTANTS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = make_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"{len(df)} rows, {df['id'].nunique()} stations, "
          f"{len(X_train)} train / {len(X_test)} test\n")
    print("| Parameter | MSE | MAE | R2 |")
    print("|---|---|---|---|")
    for i, pollutant in enumerate(POLLUTANTS):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"| {pollutant} | {mse:.2f} | {mae:.2f} | {r2:.2f} |")


if __name__ == "__main__":
    main()
