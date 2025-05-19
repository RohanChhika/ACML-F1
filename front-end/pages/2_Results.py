import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.model import make_predictions


st.set_page_config(page_title="Results", page_icon="📑")

driver_positions = {
    "Driver": ['Max VERSTAPPEN', 'Logan SARGEANT', 'Lando NORRIS', 'Pierre GASLY', 'Sergio PEREZ', 'Fernando ALONSO', 'Charles LECLERC','Daniel RICCIARDO', 'Gabriel BORTOLETO', 'Isack HADJAR', 'Jack DOOHAN', 'Gregoire SAUCY', 'Nikola TSOLOV', 'Kimi ANTONELLI', 'Gabriele MINI',  'Liam LAWSON', 'Arthur LECLERC', 'Ayumu IWASA', 'Frederik VESTI', 'Theo POURCHAIRE'],
    "Predicted position":[1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20],
}

df = pd.DataFrame(driver_positions)

st.markdown("# 🏁 Predicted Race Results")

def highlight_alternate_rows(row):
    return ['background-color: #f24962; color: #4a4848' if row.name % 2 else '' for _ in row]

df_reset = df.reset_index(drop=True)

st.caption("Data shown is for illustration purposes. Replace with live data from the F1 API or dataset.")

st.dataframe(df_reset.style.apply(highlight_alternate_rows, axis=1))

if "race_data" in st.session_state:
    if "track_data" in st.session_state:
        make_predictions(st.session_state["race_data"], st.session_state["track_data"])