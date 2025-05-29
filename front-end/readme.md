# F1 Race Results Predictor

This is a **Streamlit** web application that predicts the results of a Formula 1 race using qualifying data and track conditions.

---

## Features

- Predicts race finishing order based on:
  - Qualifying lap times and positions
  - Track and weather conditions
- Displays the predicted results in a styled, interactive table
- Easy to modify for new datasets

---

## How It Works

1. Driver qualifying data and environmental conditions are passed to a machine learning model.
2. The model predicts race times or positions.
3. Results are displayed in a ranked table using Streamlit.

---

## Project Structure

├── front-end/
│ ├── model/
│ │ ├── **init**.py
│ │ ├── both_years.csv
│ │ └── model.py
│ ├── pages/
│ │ ├── **init**.py
│ │ ├── 1_Driver_info.py
│ │ ├── 1_Race_conditions.py
│ │ ├── 2_Results.py
│ │ ├── 3_Model_visuals.py
│ │ └── 4_About_authors.py
│ ├── **init**.py
│ └── Home.py

Requirements
Install dependencies using:
pip install streamlit
pip install pandas
pip install torch

How to Run
streamlit run Home.py (from /front-end)
