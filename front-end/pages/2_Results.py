import streamlit as st
import pandas as pd
import torch
import sys
import os

torch.classes.__path__ = [] 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.model import make_predictions

st.set_page_config(page_title="Results", page_icon="📑")

## DEFAULT DATA IF NOTHING IS SHOWING
race_data = [
    {
        "driver_number": 4,
        "quali_lap_time": 230.5537333333333,
        "quali_position": 1
    },
    {
        "driver_number": 55,
        "quali_lap_time": 233.15906666666666,
        "quali_position": 3
    },
    {
        "driver_number": 16,
        "quali_lap_time": 209.16309090909093,
        "quali_position": 14
    },
    {
        "driver_number": 44,
        "quali_lap_time": 166.47116666666665,
        "quali_position": 18
    },
    {
        "driver_number": 63,
        "quali_lap_time": 208.1088235294118,
        "quali_position": 7
    },
    {
        "driver_number": 1,
        "quali_lap_time": 263.95015384615385,
        "quali_position": 5
    },
    {
        "driver_number": 10,
        "quali_lap_time": 204.6325,
        "quali_position": 6
    },
    {
        "driver_number": 27,
        "quali_lap_time": 210.465,
        "quali_position": 4
    },
    {
        "driver_number": 14,
        "quali_lap_time": 211.74764705882353,
        "quali_position": 8
    },
    {
        "driver_number": 81,
        "quali_lap_time": 227.6352666666667,
        "quali_position": 2
    },
    {
        "driver_number": 23,
        "quali_lap_time": 123.84540000000001,
        "quali_position": 16
    },
    {
        "driver_number": 22,
        "quali_lap_time": 214.64072727272725,
        "quali_position": 11
    },
    {
        "driver_number": 24,
        "quali_lap_time": 146.944,
        "quali_position": 17
    },
    {
        "driver_number": 18,
        "quali_lap_time": 194.0543076923077,
        "quali_position": 13
    },
    {
        "driver_number": 61,
        "quali_lap_time": 159.0538,
        "quali_position": 20
    },
    {
        "driver_number": 20,
        "quali_lap_time": 216.76800000000003,
        "quali_position": 15
    },
    {
        "driver_number": 30,
        "quali_lap_time": 211.7084545454546,
        "quali_position": 12
    },
    {
        "driver_number": 77,
        "quali_lap_time": 253.5219230769231,
        "quali_position": 9
    },
    {
        "driver_number": 43,
        "quali_lap_time": 121.67840000000001,
        "quali_position": 19
    }
]

track_data = {
    "year": 2024,
    "lap_length": 5.554,
    "air_temperature": 25.8,
    "humidity": 60.0,
    "pressure": 1018.0,
    "rainfall": 0,
    "track_temperature": 29.1,
    "wind_direction": 116,
    "wind_speed": 1.2,
}

sorted_results = []
train_losses = []
val_losses = []

if (
    "race_data" in st.session_state and isinstance(st.session_state["race_data"], list) and len(st.session_state["race_data"]) > 0 and
    "track_data" in st.session_state and isinstance(st.session_state["track_data"], dict) and
    "year" in st.session_state["track_data"] and st.session_state["track_data"]["year"]
):
    sorted_results, train_losses, val_losses = make_predictions(st.session_state["race_data"], st.session_state["track_data"])
else:
    sorted_results, train_losses, val_losses  = make_predictions(race_data, track_data)

st.session_state["train_losses"] = train_losses
st.session_state["val_losses"] = val_losses

df = pd.DataFrame(sorted_results)

df_reset = df.reset_index(drop=True)

st.markdown("# 🏁 Predicted Race Results")
st.caption("Data shown is the predicted average lap time for each driver based on the data inputted and the training data available.")

def highlight_alternate_rows(row):
    return ['background-color: #f24962; color: #4a4848' if row.name % 2 else '' for _ in row]

st.dataframe(df_reset.style.apply(highlight_alternate_rows, axis=1))