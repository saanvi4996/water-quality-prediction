from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_predicts_all_pollutants():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception

    at.selectbox[0].set_value("22")
    at.number_input[0].set_value(2024)
    at.button[0].click().run()
    assert not at.exception

    lines = [m.value for m in at.markdown]
    for pollutant in ["O2", "NO3", "NO2", "SO4", "PO4", "CL"]:
        assert any(line.startswith(f"{pollutant}: ") for line in lines)


def test_station_ids_sorted_numerically():
    at = AppTest.from_file(APP, default_timeout=120).run()
    options = list(at.selectbox[0].options)
    assert options == sorted(options, key=int)
