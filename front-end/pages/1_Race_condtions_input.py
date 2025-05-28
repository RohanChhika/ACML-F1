# pages/second_page.py
import streamlit as st

st.set_page_config(page_title="Input", page_icon="🔣")

driver_names = {'Max VERSTAPPEN': 1, 'Logan SARGEANT': 2, 'Lando NORRIS': 4, 'Pierre GASLY': 10, 'Sergio PEREZ': 11, 'Fernando ALONSO': 14, 'Charles LECLERC': 16, 'Kevin MAGNUSSEN': 20, 'Maxwell ESTERSON': 21, 'Yuki TSUNODA': 22, 'ZHOU Guanyu': 24, 'Nico HULKENBERG': 27, 'Esteban OCON': 31, 'Lewis HAMILTON': 44, 'Carlos SAINZ': 55, 'George RUSSELL': 63, 'Oscar PIASTRI': 81, 'Alexander ALBON': 23, 'Felipe DRUGOVICH': 34, 'Valtteri BOTTAS': 77, 'Lance STROLL': 18, 'Daniel RICCIARDO': 3, 'Gabriel BORTOLETO': 5, 'Isack HADJAR': 41, 'Jack DOOHAN': 61, 'Gregoire SAUCY': 8, 'Nikola TSOLOV': 9, 'Kimi ANTONELLI': 12, 'Gabriele MINI': 15, 'Caio COLLET': 17, 'Tommy SMITH': 19, 'Hugh BARTER': 25, 'Nikita BEDRIN': 26, 'Ryo HIRAKAWA': 62, "Patricio O'WARD": 29, 'Liam LAWSON': 30, 'Arthur LECLERC': 39, 'Ayumu IWASA': 37, 'Frederik VESTI': 72, 'Theo POURCHAIRE': 98, 'Jake DENNIS': 36, 'Franco COLAPINTO': 43, 'Dino BEGANOVIC': 38, 'Robert SHWARTZMAN': 97, 'Luke BROWNING': 46, 'Oliver BEARMAN': 87}

st.markdown("""
    <style>
        .stButton>button {
            background-color: #d62828;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 24px;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #9b1c1c;
        }

        .main {
            background-color: #fff5f5;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0px 0px 10px rgba(214, 40, 40, 0.2);
        }

        h1 {
            color: #d62828;
        }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.title("🏁 F1 Predictor Input")

    # Input fields
    year = st.selectbox("Year", [2024,2025])
    lap_length = st.number_input("Lap Length (km)", step=0.001, format="%.3f")
    air_temperature = st.number_input("Air Temperature", step=0.1, format="%.1f")
    track_temperature = st.number_input("Track temperature", step=0.1, format="%.1f")
    wind_direction =  st.number_input("Wind direction", step=1)
    wind_speed = st.number_input("Wind speed", step=0.1)
    rainfall = st.number_input("Rainfall", step=1)
    pressure = st.number_input("Pressure", step=0.1)
    humidity = st.number_input("Humidity", step=0.1)

    # Generate button
    if st.button("Generate"):
        st.success("✅ Input captured successfully!")
        st.write("Here's what you entered:")
        st.json({
            "year": year,
            "lap_length": lap_length,
            "temperature": air_temperature,
            "wind_direction": wind_direction,
            "track_temperature": track_temperature,
            "wind_speed":wind_speed,
            "rainfall": rainfall,
            "humidity": humidity,
            "pressure": pressure
        })
        st.session_state["track_data"] = {
            "year": year,
            "lap_length": lap_length,
            "air_temperature": air_temperature,
            "track_temperature": track_temperature,
            "wind_direction": wind_direction,
            "wind_speed":wind_speed,
            "rainfall": rainfall,
            "humidity": humidity,
            "pressure": pressure
        }

    
