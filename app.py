import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Addis Ababa Rainfall Forecast", page_icon="🌧️")


@st.cache_resource
def load_model():
    return joblib.load("rain_model.joblib")


bundle = load_model()
model = bundle["model"]
features = bundle["features"]


def build_row(date, t2m, rh, ws, ps, rain7, rain30, rh7):
    """Build the same features used in training.
    rain7 = rainfall for the last 7 days: index 0 = today, 1 = yesterday, ..."""
    doy = date.timetuple().tm_yday
    row = {
        "T2M": t2m,
        "RH2M": rh,
        "WS2M": ws,
        "PS": ps,
        "PRECTOTCORR": rain7[0],
        "rain_lag1": rain7[1],
        "rain_lag2": rain7[2],
        "rain_lag3": rain7[3],
        "rain_7d": float(np.mean(rain7)),
        "rain_30d": rain30,
        "rh_7d": rh7,
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
    }
    return pd.DataFrame([row])[features]


st.title("Addis Ababa Rainfall Forecast")
st.caption(
    " Project: predicts TOMORROW's rainfall (mm) from today's weather and "
    "recent rainfall. Trained on NASA POWER daily data for Addis Ababa, 2015 to 2023."
)

st.subheader("Today's conditions")
date = st.date_input("Today's date", dt.date(2023, 7, 15))
c1, c2 = st.columns(2)
with c1:
    t2m = st.number_input("Temperature at 2 m (°C)", 0.0, 35.0, 15.5, step=0.1)
    rh = st.number_input("Relative humidity at 2 m (%)", 0.0, 100.0, 85.0, step=1.0)
with c2:
    ws = st.number_input("Wind speed at 2 m (m/s)", 0.0, 10.0, 1.8, step=0.1)
    ps = st.number_input("Surface pressure (kPa)", 70.0, 80.0, 77.0, step=0.01)

st.subheader("Recent rainfall (mm per day)")
labels = ["Today", "1 day ago", "2 days ago", "3 days ago",
          "4 days ago", "5 days ago", "6 days ago"]
defaults = [8.0, 5.0, 12.0, 3.0, 6.0, 9.0, 4.0]
rain7 = []
cols = st.columns(4)
for i, (lab, d) in enumerate(zip(labels, defaults)):
    with cols[i % 4]:
        rain7.append(st.number_input(lab, 0.0, 150.0, d, step=0.5, key=f"r{i}"))

c3, c4 = st.columns(2)
with c3:
    rain30 = st.number_input("Average rainfall, last 30 days (mm/day)", 0.0, 50.0, 5.0, step=0.1)
with c4:
    rh7 = st.number_input("Average humidity, last 7 days (%)", 0.0, 100.0, 80.0, step=1.0)

if st.button("Predict tomorrow's rainfall", type="primary"):
    X = build_row(date, t2m, rh, ws, ps, rain7, rain30, rh7)
    pred = float(max(model.predict(X)[0], 0.0))
    st.metric("Predicted rainfall tomorrow", f"{pred:.1f} mm")

    if pred < 1:
        st.success("Little or no rain expected.")
    elif pred < 5:
        st.info("Light rain possible.")
    elif pred < 15:
        st.warning("Moderate rain likely.")
    else:
        st.error("Heavy rain possible.")

    st.caption(
        "On the 2023 test data the typical error was about 3 mm per day. "
        "tends to under-predict heavy storms: it rarely predicted more than about "
        "15 mm, even on days when real rainfall reached over 50 mm. Do not use this "
        "for real weather decisions."
    )
